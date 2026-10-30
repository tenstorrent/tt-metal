# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Independent TT text-to-image generation on a caller-owned MeshDevice.

Tokenization, initial CPU random noise, positional/scheduler metadata and artifact
I/O run on the host. Every learned stage, guidance and Euler arithmetic run on TT.
No captured CUDA activation is accepted by this entry point.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import torch
from PIL import Image

from models.experimental.moss_image.tt.checkpoint import resolve_checkpoint, checkpoint_manifest, sha256


def validate_generation_options(height, width, steps, prompt, negative_prompt, guidance_scale):
    """Validate public inputs without importing TTNN or touching hardware."""
    if any(not isinstance(value, int) or isinstance(value, bool) for value in (height, width, steps)):
        raise ValueError("Dimensions and steps must be integers")
    if min(height, width) < 256 or height % 16 or width % 16 or not 1 <= steps <= 1000:
        raise ValueError("Dimensions must be multiples of 16, at least 256; steps must be 1..1000")
    if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
        raise TypeError("Prompts must be strings")
    if not math.isfinite(guidance_scale) or guidance_scale < 1:
        raise ValueError("Guidance scale must be finite and at least 1")


def generate(
    checkpoint,
    mesh_device,
    output_dir,
    *,
    variant="Base",
    prompt="the quick brown fox jumps over the lazy dog",
    negative_prompt="",
    height=1024,
    width=1024,
    steps=50,
    seed=42,
    guidance_scale=3.0,
):
    """Generate and persist an image, returning the measured status record.

    The caller owns device/fabric configuration and lifetime. Two-card runs use a
    one-chip submesh for prompt encoding, head/channel TP for DiT and replicated
    VAE decode. All temporary model state is released before returning; this
    function never closes a fixture-owned MeshDevice.
    """
    validate_generation_options(height, width, steps, prompt, negative_prompt, guidance_scale)
    import ttnn as t
    from transformers import AutoTokenizer
    from models.experimental.moss_image.tt.text_encoder import TTTextEncoder
    from models.experimental.moss_image.tt.dit import TTMossDiT
    from models.experimental.moss_image.tt.scheduler import guided_euler_step
    from models.experimental.moss_image.tt.vae import TTMossVAE
    from models.experimental.moss_image.tt.vendor.pipeline_moss_image import MOSSImageScheduler

    cards = mesh_device.get_num_devices()
    if cards not in (1, 2):
        raise ValueError("Generation requires a one-card or connected two-card mesh")
    checkpoint = resolve_checkpoint(variant, checkpoint, local_files_only=True)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(
        checkpoint=str(checkpoint),
        variant=variant,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        steps=steps,
        seed=seed,
        guidance_scale=guidance_scale,
    )
    mesh = mesh_device
    report = {
        "status": "starting",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "activation_injection": False,
        "cuda_used": False,
        "placement": {"text_encoder_cards": 1, "dit_cards": cards, "vae": "replicated"},
        "host_operations": [
            "tokenization",
            "CPU seeded initial Gaussian noise",
            "schedule and positional metadata",
            "module boundary transfers",
            "artifact I/O",
        ],
        "steps": [],
        "timings": {},
        "checkpoint": checkpoint_manifest(checkpoint, variant),
    }
    packages = {}
    for name in ("torch", "diffusers", "transformers", "ttnn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "source-tree installation (no distribution metadata)"
    report["environment"] = {
        "python": platform.python_version(),
        "packages": packages,
        "ttnn_module": t.__file__,
        "cards": cards,
    }
    source_root = Path(__file__).parent
    report["source_files"] = {
        str(path.relative_to(source_root)): sha256(path)
        for path in sorted((source_root / "tt").rglob("*"))
        if path.is_file() and path.suffix in {".py", ".json"}
    }
    report["source_files"]["demo.py"] = sha256(Path(__file__))

    def flush():
        temporary = out / "status.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(out / "status.json")

    flush()
    begin = time.perf_counter()
    try:
        report["status"] = "encoding_prompt"
        flush()
        encoder_mesh = mesh if mesh.get_num_devices() == 1 else mesh.create_submesh(t.MeshShape(1, 1))
        encoder = TTTextEncoder(checkpoint, encoder_mesh)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint / "tokenizer", local_files_only=True, trust_remote_code=False
        )
        contexts = {}
        stamp = time.perf_counter()
        for branch, prompt in [("positive", args.prompt), ("negative", args.negative_prompt)]:
            encoded, mask = encoder.encode_prompt(prompt, tokenizer)
            host = t.to_torch(t.get_device_tensors(encoded)[0])
            contexts[branch] = host.reshape(1, -1, 4096)[:, mask[0].bool()].contiguous()
            torch.save({"hidden_states": contexts[branch], "mask": mask}, out / f"conditioning_{branch}.pt")
            del encoded
        report["timings"]["text_encoder_seconds"] = time.perf_counter() - stamp
        del encoder
        gc.collect()
        if encoder_mesh is not mesh:
            # Child and parent share physical queues. The supported handoff is
            # to quiesce the parent hierarchy after child work, as in Metalium's
            # child/parent workload tests. A Python keep-alive can retain the
            # child after this local name is dropped; do not close that tracker
            # before the parent quiesces it.
            mesh.quiesce_devices()
        del encoder_mesh
        model = TTMossDiT(checkpoint, mesh)
        contexts = {key: model.upload(value) for key, value in contexts.items()}
        height, width = args.height // 16, args.width // 16
        # Native reference uses randn_tensor(fp32); CPU generator chosen explicitly
        # for reproducibility across accelerator types, without loading a capture.
        noise = torch.randn(
            (1, 128, height, width), generator=torch.Generator("cpu").manual_seed(args.seed), dtype=torch.float32
        )
        torch.save(noise, out / "initial_noise.pt")
        latent = model.upload(noise.flatten(2).transpose(1, 2), dtype=t.float32)
        scheduler = MOSSImageScheduler.from_pretrained(checkpoint / "scheduler", local_files_only=True)
        scheduler.set_timesteps(args.steps, device="cpu")
        torch.save({"timesteps": scheduler.timesteps, "sigmas": scheduler.sigmas}, out / "scheduler.pt")
        report["status"] = "denoising"
        flush()
        for index, timestep in enumerate(scheduler.timesteps):
            stamp = time.perf_counter()
            pos = model(latent, float(timestep), contexts["positive"], height, width)
            neg = model(latent, float(timestep), contexts["negative"], height, width, skip_layers=(22,))
            if index == 0:
                for branch, prediction in [("positive", pos), ("negative", neg)]:
                    torch.save(
                        model.host(prediction).transpose(1, 2).reshape(1, 128, height, width),
                        out / f"first_{branch}_prediction.pt",
                    )
            dt = float(scheduler.sigmas[index + 1] - scheduler.sigmas[index])
            latent, guided = guided_euler_step(latent, pos, neg, args.guidance_scale, dt)
            t.synchronize_device(mesh)
            elapsed = time.perf_counter() - stamp
            host = model.host(latent).transpose(1, 2).reshape(1, 128, height, width)
            torch.save(host, out / f"latent_{index+1:03d}.pt")
            report["steps"].append(
                {
                    "index": index,
                    "timestep": float(timestep),
                    "seconds": elapsed,
                    "finite": bool(torch.isfinite(host).all()),
                    "latent_rms": float(host.square().mean().sqrt()),
                }
            )
            if not report["steps"][-1]["finite"]:
                raise RuntimeError("Nonfinite denoising output")
            flush()
            print(f"Step {index+1}/{len(scheduler.timesteps)} {elapsed:.3f}s", flush=True)
            del pos, neg, guided
        report["timings"]["denoising_seconds"] = sum(s["seconds"] for s in report["steps"])
        report["status"] = "decoding"
        flush()
        # Reuse the independently generated latent. Free transformer weights first.
        packed = t.reshape(latent, (1, 1, height * width, 128))
        del model, contexts
        gc.collect()
        stamp = time.perf_counter()
        vae = TTMossVAE(checkpoint, mesh)
        image = vae.decode(packed, args.height, args.width)
        pixels = t.to_torch(t.get_device_tensors(image)[0]).float().reshape(1, args.height, args.width, 3)
        report["timings"]["vae_seconds"] = time.perf_counter() - stamp
        if not torch.isfinite(pixels).all():
            raise RuntimeError("Nonfinite decoded image")
        torch.save(pixels, out / "decoded_nhwc.pt")
        image = ((pixels[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()
        Image.fromarray(image).save(out / "image.png")
        report["image"] = "image.png"
        report["status"] = "complete"
        report["timings"]["total_seconds"] = time.perf_counter() - begin
        flush()
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        flush()
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", choices=["Base", "SFT"], default="Base")
    parser.add_argument("--device-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="the quick brown fox jumps over the lazy dog")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    args = parser.parse_args()
    if len(args.device_ids) not in (1, 2) or len(set(args.device_ids)) != len(args.device_ids):
        parser.error("Select one or two distinct reserved device IDs")
    import ttnn as t

    mesh = None
    fabric = len(args.device_ids) == 2
    try:
        if fabric:
            t.set_fabric_config(t.FabricConfig.FABRIC_1D)
        mesh = t.open_mesh_device(
            mesh_shape=t.MeshShape(1, len(args.device_ids)),
            physical_device_ids=args.device_ids,
            l1_small_size=131072,
            trace_region_size=0,
        )
        options = vars(args).copy()
        del options["device_ids"]
        generate(mesh_device=mesh, **options)
    finally:
        if mesh is not None:
            t.close_mesh_device(mesh)
        if fabric:
            t.set_fabric_config(t.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
