# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Run the complete Qwen Image 2.1 DiT denoising schedule on one or two TT cards.

Native mode computes prompt embeddings, seeded initial noise, denoising and
VAE decode on TT, with request metadata built from checkpoint configuration.
Capture mode retains component comparisons against pinned CUDA activations.
The prefix is recomputed with block-causal attention each step.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open

from models.experimental.qwen_image_2_1.checkpoint import MODEL_REVISION


def _read(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def _relative_rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.shape != expected.shape:
        raise ValueError(f"comparison shape differs: {actual.shape} vs {expected.shape}")
    actual32, expected32 = actual.float(), expected.float()
    return float((actual32 - expected32).square().mean().sqrt() / expected32.square().mean().sqrt())


def _pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Pearson correlation over the complete logical output tensor."""
    actual32 = actual.float().flatten()
    expected32 = expected.float().flatten()
    actual32 = actual32 - actual32.mean()
    expected32 = expected32 - expected32.mean()
    denominator = actual32.norm() * expected32.norm()
    if denominator == 0:
        raise ValueError("PCC is undefined for a constant output")
    return float(torch.dot(actual32, expected32) / denominator)


def _write_json(path: Path, document: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cuda-dir", type=Path, help="component-validation capture; never used with --native")
    mode.add_argument(
        "--native",
        action="store_true",
        help="integrated generation from raw text and seed, without activation or metadata captures",
    )
    parser.add_argument("--prompt", default="the quick brown fox jumps over the lazy dog")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-bdf", required=True)
    parser.add_argument("--max-steps", type=int, help="stop a fresh diagnostic run after this many steps")
    parser.add_argument("--decode-vae", action="store_true", help="decode the final device latent using the TT VAE")
    parser.add_argument(
        "--vae-checkpoint", type=Path, help="optional pinned decoder-only checkpoint for TT VAE validation"
    )
    parser.add_argument(
        "--tt-encoder-checkpoint",
        type=Path,
        help="encode the manifest's raw prompt on TT instead of using captured embeddings",
    )
    parser.add_argument(
        "--tt-noise",
        action="store_true",
        help="generate a fresh initial latent using TT RNG; disables comparisons to the independent CUDA trajectory",
    )
    parser.add_argument(
        "--timing-breakdown",
        action="store_true",
        help="synchronize module boundaries and save host-inclusive module wall times",
    )
    parser.add_argument("--tensor-parallel", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--resident-block-weights",
        action="store_true",
        help="prepare all DiT weights once and retain them in device DRAM",
    )
    args = parser.parse_args(argv)
    bdfs = args.device_bdf.split(",")
    if len(bdfs) != args.tensor_parallel or len(set(bdfs)) != len(bdfs):
        parser.error("provide one distinct comma-separated PCI BDF per tensor-parallel rank")
    if args.native:
        if not args.tt_encoder_checkpoint:
            parser.error("--native requires --tt-encoder-checkpoint")
        args.tt_noise = True
        args.decode_vae = True
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    if args.checkpoint.name != MODEL_REVISION:
        parser.error("checkpoint path must end in the pinned model revision")
    if args.tt_encoder_checkpoint and args.tt_encoder_checkpoint.name != MODEL_REVISION:
        parser.error("text encoder checkpoint path must end in the pinned model revision")
    if args.decode_vae:
        vae_checkpoint = args.vae_checkpoint or args.checkpoint
        if vae_checkpoint.name != MODEL_REVISION:
            parser.error("VAE checkpoint path must end in the pinned model revision")
        for filename in ("config.json", "diffusion_pytorch_model.safetensors"):
            if not (vae_checkpoint / "vae" / filename).is_file():
                parser.error(f"--decode-vae requires checkpoint/vae/{filename}")
    if args.native:
        manifest = {
            "model_id": "Qwen/Qwen-Image-2.1",
            "model_revision": MODEL_REVISION,
            "prompt": args.prompt,
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "backend": "TT",
            "prompt_expansion": False,
            "injected_activations": False,
            "metadata_source": "runtime checkpoint configuration and token geometry",
        }
        from models.experimental.qwen_image_2_1.tt.text_prompt import tokenize_prompt
        from models.experimental.qwen_image_2_1.tt.inference_metadata import build_metadata

        ids, drop_idx = tokenize_prompt(args.tt_encoder_checkpoint, args.prompt)
        metadata = build_metadata(args.checkpoint, ids.shape[1] - drop_idx, args.height, args.width, args.steps)
        schedule = metadata["schedule"]
    else:
        manifest = json.loads((args.cuda_dir / "manifest.json").read_text())
        schedule = json.loads((args.cuda_dir / "schedule.json").read_text())
        if manifest["model_revision"] != MODEL_REVISION or schedule["model_revision"] != MODEL_REVISION:
            parser.error("CUDA capture or schedule does not match the pinned checkpoint")
        for field in ("prompt", "seed", "height", "width"):
            if schedule.get(field) != manifest[field]:
                parser.error(f"sigma schedule {field} differs from CUDA capture")
    total = manifest["steps"]
    if schedule["steps"] != total or len(schedule["sigmas"]) != total + 1:
        parser.error("sigma schedule length differs from CUDA capture")
    if not args.native and manifest["capture_steps"] != list(range(total)):
        parser.error("per-step CUDA boundaries are required for comparison")
    count = total if args.max_steps is None else args.max_steps
    if count < 1 or count > total:
        parser.error("--max-steps must lie within the captured schedule")
    args.output_dir.mkdir(parents=True)

    if args.native:
        latents, context = None, None
        target_mask, rope = metadata["target_mask"], metadata["rope"]
        segments, key_valid = metadata["segments"], metadata["key_valid"]
        target_count = schedule["latent_tokens"]
    else:
        first = args.cuda_dir / "step_000/transformer"
        block_input = first / "transformer_blocks.0/input"
        latents = _read(first / "input/hidden_states.pt")
        context = _read(first / "input/encoder_hidden_states.pt")
        target_mask = _read(block_input / "target_token_mask.pt")
        rope = _read(block_input / "rotary_emb.pt")
        segments = json.loads((block_input / "segments.json").read_text())
        key_valid_path = block_input / "key_valid.pt"
        key_valid = _read(key_valid_path) if key_valid_path.is_file() else None
        target_count = latents.shape[1]
    sequence = target_mask.numel()
    prefix_count = sequence - target_count
    height, width = manifest["height"], manifest["width"]
    if height < 32 or width < 32 or height % 32 or width % 32:
        parser.error("captured height and width must be multiples of 32")
    expected_tokens = (height // 16) * (width // 16)
    if (latents is not None and tuple(latents.shape) != (1, expected_tokens, 64)) or prefix_count <= 0:
        parser.error("CUDA latent or token metadata does not match captured image resolution")
    if schedule.get("latent_tokens") != target_count:
        parser.error("sigma schedule token count differs from CUDA latent shape")
    if sequence != rope.shape[0] or target_mask.ndim != 1:
        parser.error("rotary embedding or target mask has incompatible sequence length")
    _write_json(args.output_dir / "manifest.json", manifest)
    _write_json(args.output_dir / "schedule.json", schedule)

    os.environ["TT_VISIBLE_DEVICES"] = args.device_bdf
    os.environ.pop("TT_METAL_VISIBLE_DEVICES", None)
    import ttnn

    from models.experimental.qwen_image_2_1.validation.export_block_weights import load_block_weights
    from models.experimental.qwen_image_2_1.tt.tt_attention import prefill_mask
    from models.experimental.qwen_image_2_1.tt.tt_block import (
        prefill_block,
        prepare_weights as prepare_block_weights,
        select_modulation_device,
    )
    from models.experimental.qwen_image_2_1.tt.tt_dit_components import rotary_caches, to_device, to_host
    from models.experimental.qwen_image_2_1.tt.tt_input import (
        image_projection,
        prepare_weights as prepare_input_weights,
        sinusoidal_timestep,
        text_projection,
        timestep_and_modulation,
    )
    from models.experimental.qwen_image_2_1.tt.tt_output import (
        output_head,
        prepare_weights as prepare_output_weights,
        select_timestep_embedding_device,
    )
    from models.experimental.qwen_image_2_1.tt.tt_scheduler import flow_euler_step

    if args.tensor_parallel == 2:
        from models.experimental.qwen_image_2_1.tt.tt_tp_block import (
            prefill_block,
            prepare_weights as prepare_block_weights,
        )

    transformer = args.checkpoint / "transformer"
    mapping = json.loads((transformer / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]

    def checkpoint_tensors(names: tuple[str, ...]) -> dict[str, torch.Tensor]:
        selected = {}
        for shard in sorted({mapping[name] for name in names}):
            with safe_open(transformer / shard, framework="pt", device="cpu") as source:
                for name in names:
                    if mapping[name] == shard:
                        selected[name] = source.get_tensor(name)
        return selected

    input_names = (
        "img_in.weight",
        "txt_in.text_norm.weight",
        "txt_in.in_layer.weight",
        "txt_in.out_layer.weight",
        "time_text_embed.timestep_embedder.linear_1.weight",
        "time_text_embed.timestep_embedder.linear_2.weight",
        "modulation.1.weight",
    )
    device = None
    started = time.monotonic()
    reports: list[dict[str, object]] = []
    timings = []

    @contextmanager
    def timed(name, step=None, layer=None):
        if args.timing_breakdown:
            ttnn.synchronize_device(device)
        begin = time.monotonic()
        yield
        if args.timing_breakdown:
            ttnn.synchronize_device(device)
            timings.append({"module": name, "step": step, "layer": layer, "seconds": time.monotonic() - begin})
            _write_json(
                args.output_dir / "runtime.json",
                {
                    "backend": "TT",
                    "timing": "synchronized host wall time, including weight loading and dispatch; not device kernel time",
                    "warmup": "no in-process warmup; existing JIT cache may be reused",
                    "samples": 1,
                    "modules": timings,
                },
            )

    def progress(status: str, error: str | None = None) -> None:
        _write_json(
            args.output_dir / "progress.json",
            {
                "status": status,
                "prompt": manifest["prompt"],
                "seed": manifest["seed"],
                "height": height,
                "width": width,
                "completed_steps": len(reports),
                "total_steps": total,
                "elapsed_seconds": time.monotonic() - started,
                "latest": reports[-1] if reports else None,
                "error": error,
            },
        )

    try:
        progress("starting")
        if args.tensor_parallel == 2:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, args.tensor_parallel),
            physical_device_ids=list(range(args.tensor_parallel)),
            l1_small_size=32768,
        )
        padded = (sequence + 31) // 32 * 32
        padded_rope = torch.cat((rope, torch.ones(padded - sequence, rope.shape[1], dtype=rope.dtype)))
        cos, sin = rotary_caches(padded_rope, device)
        mask = prefill_mask(sequence, segments, device, key_valid)
        input_weights = prepare_input_weights(checkpoint_tensors(input_names), device)
        output_state = checkpoint_tensors(("norm_out.linear.weight", "proj_out.weight"))
        output_weights = prepare_output_weights(
            output_state["norm_out.linear.weight"], output_state["proj_out.weight"], device
        )
        context_tt = to_device(context, device) if context is not None else None
        if args.tt_encoder_checkpoint:
            from models.experimental.qwen_image_2_1.tt.text_prompt import tokenize_prompt
            from models.experimental.qwen_image_2_1.tt.tt_text_encoder import QwenImage21TextEncoder
            from models.experimental.qwen_image_2_1.validation.tt_vae_decode import metrics

            progress("encoding")
            ids, drop_idx = tokenize_prompt(args.tt_encoder_checkpoint, manifest["prompt"])
            encoder = QwenImage21TextEncoder(args.tt_encoder_checkpoint, device)

            def encoder_stage(name, _value):
                _write_json(args.output_dir / "encoder_progress.json", {"status": "running", "stage": name})

            with timed("prompt_encoder"):
                context_tt = encoder.encode(ids, drop_idx, encoder_stage)
            context_shape = (1, prefix_count, 4096)
            if tuple(context_tt.shape) != context_shape:
                raise ValueError("TT encoder output differs from captured token geometry")
            encoded_host = to_host(context_tt, context_shape)
            torch.save(encoded_host, args.output_dir / "prompt_embeds.pt")
            if context is not None:
                encoder_report = metrics(encoded_host, context)
                _write_json(args.output_dir / "encoder_report.json", encoder_report)
                if not encoder_report["pcc"] >= 0.99:
                    raise AssertionError(f"TT prompt encoder PCC below 0.99: {encoder_report}")
            _write_json(args.output_dir / "encoder_progress.json", {"status": "complete", "stage": "prompt_embeds"})
        with timed("text_projection"):
            text_tt = text_projection(context_tt, input_weights)
        if args.tt_noise:
            from models.experimental.qwen_image_2_1.tt.tt_noise import initial_image_latents

            with timed("initial_noise"):
                latents_tt = initial_image_latents(device, height, width, manifest["seed"])
        else:
            latents_tt = to_device(latents, device)
        torch.save(to_host(latents_tt, (1, target_count, 64)), args.output_dir / "initial_latents.pt")
        _write_json(
            args.output_dir / "inputs.json",
            {
                "prompt_encoder": "TT" if args.tt_encoder_checkpoint else "captured CUDA",
                "initial_noise": "TT" if args.tt_noise else "captured CUDA",
                "cuda_trajectory_comparison": not args.tt_noise,
                "injected_activations": not args.native,
                "metadata_source": "runtime configuration" if args.native else "CUDA validation capture",
                "cuda_available": torch.cuda.is_available(),
            },
        )
        if args.tensor_parallel == 2:
            replicas = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(latents_tt)]
            if not all(torch.equal(replicas[0], x) for x in replicas[1:]):
                raise AssertionError("native initial-noise replicas differ; TP ranks must start from identical latents")
        resident = {}
        if args.resident_block_weights:
            progress("loading_weights")
            with timed("resident_weight_preparation"):
                for layer in range(32):
                    state = load_block_weights(args.checkpoint, layer)
                    resident[layer] = prepare_block_weights(state, device)
                    del state
        _write_json(
            args.output_dir / "execution.json",
            {
                "cards": args.tensor_parallel,
                "device_bdfs": bdfs,
                "layout": "attention-head and MLP tensor parallel" if args.tensor_parallel == 2 else "single card",
                "encoder_and_vae": "replicated across ranks" if args.tensor_parallel == 2 else "single card",
                "resident_block_weights": args.resident_block_weights,
            },
        )
        progress("running")
        for step in range(count):
            step_started = time.monotonic()
            cuda_step = args.cuda_dir / f"step_{step:03d}" if args.cuda_dir else None
            captured_timestep = (
                torch.tensor([schedule["timesteps"][step]], dtype=torch.bfloat16)
                if args.native
                else _read(cuda_step / "transformer/input/timestep.pt")
            )
            features = sinusoidal_timestep(torch.cat((captured_timestep, captured_timestep.new_zeros(1))))
            with timed("input_and_conditioning", step):
                temb, modulation = timestep_and_modulation(to_device(features, device), input_weights)
                selected = select_modulation_device(modulation, target_mask, device)
                image_tt = image_projection(latents_tt, input_weights)
                result = ttnn.concat((text_tt, image_tt), dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            block_errors = {}
            for layer in range(32):
                with timed("dit_block", step, layer):
                    if layer in resident:
                        weights = resident[layer]
                    else:
                        state = load_block_weights(args.checkpoint, layer)
                        weights = prepare_block_weights(state, device)
                        del state
                    result = prefill_block(result, weights, selected, cos, sin, mask, sequence)
                expected_path = cuda_step / f"transformer/transformer_blocks.{layer}.pt" if cuda_step else None
                if not args.tt_noise and expected_path is not None and expected_path.is_file():
                    actual_block = to_host(result, (1, sequence, 4096))[:, -target_count:]
                    expected_block = _read(expected_path)[:, -target_count:]
                    block_errors[str(layer)] = _relative_rms(actual_block, expected_block)
                del weights
            with timed("output_head", step):
                selected_temb = select_timestep_embedding_device(temb, target_mask, device)
                _, projection = output_head(result, selected_temb, output_weights)
                velocity = ttnn.slice(projection, (0, prefix_count, 0), (1, sequence, 64))
            velocity_host = to_host(velocity, (1, target_count, 64))
            cuda_velocity = (
                _read(cuda_step / "transformer/proj_out.pt")[:, -target_count:] if not args.tt_noise else None
            )
            sigma, next_sigma = schedule["sigmas"][step : step + 2]
            with timed("euler_update", step):
                latents_tt = flow_euler_step(latents_tt, velocity, sigma, next_sigma, device)
            latents_host = to_host(latents_tt, (1, target_count, 64))
            cuda_latents = _read(cuda_step / "scheduler/updated_latents.pt") if not args.tt_noise else None
            output = args.output_dir / f"step_{step:03d}"
            output.mkdir()
            torch.save(latents_host, output / "updated_latents.pt")
            torch.save(velocity_host, output / "velocity.pt")
            row = {
                "step": step,
                "sigma": sigma,
                "next_sigma": next_sigma,
                "velocity_relative_rms_error": _relative_rms(velocity_host, cuda_velocity)
                if cuda_velocity is not None
                else None,
                "latent_relative_rms_error": _relative_rms(latents_host, cuda_latents)
                if cuda_latents is not None
                else None,
                "velocity_pcc": _pcc(velocity_host, cuda_velocity) if cuda_velocity is not None else None,
                "latent_pcc": _pcc(latents_host, cuda_latents) if cuda_latents is not None else None,
                "latent_mean_abs_error": float((latents_host.float() - cuda_latents.float()).abs().mean())
                if cuda_latents is not None
                else None,
                "block_relative_rms_errors": block_errors,
                "elapsed_seconds": time.monotonic() - step_started,
            }
            reports.append(row)
            _write_json(args.output_dir / "report.json", reports)
            progress(
                "running"
                if step + 1 < count
                else ("decoding" if args.decode_vae else ("complete" if count == total else "partial"))
            )
            print(json.dumps(row), flush=True)
        if args.decode_vae:
            from models.experimental.qwen_image_2_1.tt.tt_vae import QwenImage21VAEDecoder
            from PIL import Image

            progress("decoding")
            decoder = QwenImage21VAEDecoder(vae_checkpoint, device)

            def vae_stage(name, _value, _height, _width):
                _write_json(args.output_dir / "vae_progress.json", {"status": "running", "stage": name})

            with timed("vae_decode"):
                decoded = decoder.decode(latents_tt, height, width, vae_stage)
            image = decoder.collect(decoded, height, width, decoder.config["out_channels"])
            torch.save(image, args.output_dir / "decoded_image.pt")
            pixels = ((image[:, :, 0].float() / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
            Image.fromarray(pixels[0].permute(1, 2, 0).numpy()).save(args.output_dir / "tt_vae.png")
            _write_json(args.output_dir / "vae_progress.json", {"status": "complete", "stage": "output"})
            progress("complete" if count == total else "partial")
    except Exception as error:
        progress("failed", str(error))
        raise
    finally:
        if device is not None:
            ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
