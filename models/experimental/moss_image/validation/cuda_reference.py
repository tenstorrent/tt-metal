# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Capture native integrated MOSS-Image CUDA execution without activation injection.

Run from the repository root: python -m models.experimental.moss_image.validation.cuda_reference ...
Artifacts stay outside the repository. Tensor files contain only tensors/primitives
and can be opened with torch.load(..., weights_only=True).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import torch

from models.experimental.moss_image.tt.checkpoint import PIPELINE_SHA256, checkpoint_manifest, sha256
from models.experimental.moss_image.tt.reference import load_reference
from models.experimental.moss_image.tt.vendor import pipeline_moss_image as upstream


class Capture:
    def __init__(self, path, budget_gib=20):
        self.path = Path(path).expanduser().resolve()
        if self.path.exists() and any(self.path.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing capture: {self.path}")
        self.path.mkdir(parents=True, exist_ok=True)
        self.limit = int(budget_gib * 1024**3)
        self.bytes = 0
        self.manifest = {"schema_version": 1, "status": "starting", "artifacts": {}, "steps": []}

    def save(self, key, value):
        def cpu(item):
            if isinstance(item, torch.Tensor):
                return item.detach().cpu().contiguous().clone()
            if isinstance(item, (list, tuple)):
                return type(item)(cpu(x) for x in item)
            if isinstance(item, dict):
                return {str(k): cpu(v) for k, v in item.items()}
            if isinstance(item, (torch.device, torch.dtype)):
                return str(item)
            if item is None or isinstance(item, (str, float, int, bool)):
                return item
            raise TypeError(f"Unsupported capture value {type(item)}")

        def size(item):
            if isinstance(item, torch.Tensor):
                return item.numel() * item.element_size()
            if isinstance(item, dict):
                return sum(size(v) for v in item.values())
            if isinstance(item, (list, tuple)):
                return sum(size(v) for v in item)
            return 0

        expected = size(value)
        if self.bytes + expected + 65536 > self.limit:
            raise RuntimeError(f"Capture budget {self.limit / 1024**3:.1f} GiB exceeded at {key}")
        value = cpu(value)
        path = self.path / f"{key}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, path)
        self.bytes += path.stat().st_size
        self.manifest["artifacts"][key] = {
            "path": str(path.relative_to(self.path)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        return value

    def flush(self):
        self.manifest["capture_bytes"] = self.bytes
        temporary = self.path / "manifest.json.tmp"
        temporary.write_text(json.dumps(self.manifest, indent=2) + "\n")
        temporary.replace(self.path / "manifest.json")


class OracleHooks:
    def __init__(self, pipe, capture, layers, vae_stage_captures=True):
        self.pipe, self.capture, self.layers = pipe, capture, set(layers)
        self.handles = []
        self.step = 0
        self.branch_counter = 0
        self.branch = None
        self.active_layer = None
        self.encoder_counter = 0
        self.original_attention = upstream._attention
        self.original_step = pipe.scheduler.step
        self.original_decode = pipe.vae.decode
        self.vae_stage_captures = vae_stage_captures

    def add(self, module, pre=None, post=None):
        if pre:
            self.handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        if post:
            self.handles.append(module.register_forward_hook(post, with_kwargs=True))

    def install(self):
        def text_pre(module, args, kwargs):
            self.encoder_branch = "positive" if self.encoder_counter == 0 else "negative"
            self.capture.save(f"text/{self.encoder_branch}/inputs", kwargs)

        def text_post(module, args, kwargs, output):
            self.capture.save(f"text/{self.encoder_branch}/output", output.last_hidden_state)
            self.encoder_counter += 1

        self.add(self.pipe.text_encoder, text_pre, text_post)
        # Encoder intermediate states aid full-GDN and attention component checks.
        for name, module in self.pipe.text_encoder.named_modules():
            parts = name.split(".")
            if len(parts) >= 2 and parts[-2] == "layers" and parts[-1].isdigit():

                def encoder_post(module, args, kwargs, output, name=name):
                    out = output[0] if isinstance(output, tuple) else output
                    if isinstance(out, torch.Tensor):
                        self.capture.save(
                            f"text/{self.encoder_branch}/layers/{name}",
                            {"input": args[0] if args else kwargs.get("hidden_states"), "output": out},
                        )

                self.add(module, post=encoder_post)

        def transformer_pre(module, args, kwargs):
            torch.cuda.synchronize()
            self.transformer_start = time.perf_counter()
            self.branch = "positive" if self.branch_counter % 2 == 0 else "negative"
            prefix = f"steps/{self.step:03d}"
            if self.branch == "positive":
                self.capture.save(f"{prefix}/input_latents", args[0])
                self.capture.save(f"{prefix}/timestep", args[1])
            if self.step == 0:
                self.capture.save(
                    f"conditioning/{self.branch}",
                    {
                        "hidden_states": args[2],
                        "attention_mask": args[3],
                        "skip_layers": sorted(kwargs.get("skip_layers", [])),
                    },
                )

        def transformer_post(module, args, kwargs, output):
            self.capture.save(f"steps/{self.step:03d}/{self.branch}_prediction", output.sample)
            torch.cuda.synchronize()
            self.capture.manifest.setdefault("transformer_calls", []).append(
                {
                    "step": self.step,
                    "branch": self.branch,
                    "seconds_including_capture": time.perf_counter() - self.transformer_start,
                }
            )
            self.branch_counter += 1
            self.branch = None

        self.add(self.pipe.transformer, transformer_pre, transformer_post)
        # First-step full-block outputs for both branches; fine-grained captures
        # limited to selected blocks. All hooks return None: execution is unchanged.
        for index, block in enumerate(self.pipe.transformer.layers):

            def block_pre(module, args, kwargs, index=index):
                self.active_layer = index
                if self.step == 0 and index in self.layers:
                    self.capture.save(
                        f"blocks/{self.branch}/{index:02d}/inputs",
                        {"x": args[0], "rope": args[1], "modulation": args[2]},
                    )

            def block_post(module, args, kwargs, output, index=index):
                if self.step == 0:
                    self.capture.save(f"blocks/{self.branch}/{index:02d}/output", output)
                self.active_layer = None

            self.add(block, block_pre, block_post)
            if index in self.layers:
                for name, submodule in block.named_modules():
                    if not name or any(submodule.children()):
                        continue

                    def component_post(module, args, kwargs, output, index=index, name=name):
                        if self.step == 0 and self.branch is not None:
                            self.capture.save(
                                f"blocks/{self.branch}/{index:02d}/components/{name}",
                                {"inputs": args, "output": output},
                            )

                    self.add(submodule, post=component_post)
        for name in (
            "video_embedder",
            "text_embedder",
            "rope",
            "time_embedding",
            "shared_block_modulation",
            "video_head_modulation",
            "final_norm_video",
            "final_linear_video",
        ):

            def projection_post(module, args, kwargs, output, name=name):
                if self.step == 0:
                    self.capture.save(f"projections/{self.branch}/{name}", {"inputs": args, "output": output})

            self.add(getattr(self.pipe.transformer, name), post=projection_post)

        def attention(q, k, v, rope, heads):
            output = self.original_attention(q, k, v, rope, heads)
            if self.step == 0 and self.active_layer in self.layers:
                self.capture.save(
                    f"blocks/{self.branch}/{self.active_layer:02d}/attention",
                    {"q": q, "k": k, "v": v, "rope": rope, "heads": heads, "output": output},
                )
            return output

        upstream._attention = attention

        def scheduler_step(model_output, timestep, sample, *args, **kwargs):
            self.capture.save(f"steps/{self.step:03d}/guided_prediction", model_output)
            output = self.original_step(model_output, timestep, sample, *args, **kwargs)
            updated = output[0] if isinstance(output, tuple) else output.prev_sample
            self.capture.save(f"steps/{self.step:03d}/updated_latents", updated)
            self.capture.manifest["steps"].append(
                {
                    "index": self.step,
                    "timestep": float(timestep),
                    "sigma": float(self.pipe.scheduler.sigmas[self.step]),
                    "next_sigma": float(self.pipe.scheduler.sigmas[self.step + 1]),
                }
            )
            self.step += 1
            self.capture.manifest["status"] = "denoising"
            self.capture.flush()
            return output

        self.pipe.scheduler.step = scheduler_step

        def decode(z, *args, **kwargs):
            self.capture.manifest["status"] = "decoding"
            self.capture.flush()
            self.capture.save("vae/input", z)
            begin = time.perf_counter()
            output = self.original_decode(z, *args, **kwargs)
            self.capture.save("vae/raw_output", output[0] if isinstance(output, tuple) else output.sample)
            self.capture.manifest["vae_seconds_including_capture"] = time.perf_counter() - begin
            return output

        self.pipe.vae.decode = decode
        for name, module in self.pipe.vae.named_modules() if self.vae_stage_captures else []:
            if (
                name == "post_quant_conv"
                or name in {"decoder.conv_in", "decoder.mid_block", "decoder.conv_norm_out", "decoder.conv_out"}
                or (
                    name.startswith("decoder.")
                    and (
                        (".resnets." in name and name.rsplit(".", 1)[-1].isdigit())
                        or (".attentions." in name and name.rsplit(".", 1)[-1].isdigit())
                        or name.endswith("upsamplers.0.conv")
                    )
                )
            ):

                def vae_post(module, args, kwargs, output, name=name):
                    self.capture.save(
                        f"vae/stages/{name}",
                        {"input": args[0] if args else kwargs.get("hidden_states"), "output": output},
                    )

                self.add(module, post=vae_post)

    def close(self):
        for handle in self.handles:
            handle.remove()
        upstream._attention = self.original_attention
        self.pipe.scheduler.step = self.original_step
        self.pipe.vae.decode = self.original_decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("Base", "SFT"), default="Base")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0", help="Logical device after CUDA_VISIBLE_DEVICES filtering")
    parser.add_argument("--prompt", default="the quick brown fox jumps over the lazy dog")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generator-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="CPU RNG enables independent TT noise generation with the same seed",
    )
    parser.add_argument("--capture-layers", type=int, nargs="*", default=[0, 21, 41])
    parser.add_argument("--capture-budget-gib", type=float, default=20)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--no-vae-stage-captures",
        action="store_true",
        help="Retain VAE input and raw output, omit intermediate stage duplicates",
    )
    args = parser.parse_args()
    if any(i < 0 or i >= 42 for i in args.capture_layers):
        parser.error("Capture layers must be within [0, 42)")
    capture = Capture(args.output_dir, args.capture_budget_gib)
    capture.manifest.update(
        {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": vars(args),
            "pipeline_sha256": PIPELINE_SHA256,
            "native_integrated_run": True,
            "activation_injection": False,
            "guidance_scale": 3.0,
            "shift": 5.0,
            "negative_skip_layers": [22],
            "timing_note": "Includes capture hooks, GPU synchronization, tensor transfers, hashing and disk I/O; not a performance benchmark.",
            "environment": {
                "python": platform.python_version(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors")
                },
                "torch_cuda": torch.version.cuda,
            },
        }
    )
    capture.manifest["source_files"] = {
        name: sha256(Path(__file__).resolve().parents[1] / name)
        for name in (
            "validation/cuda_reference.py",
            "tt/reference.py",
            "tt/checkpoint.py",
            "tt/vendor/pipeline_moss_image.py",
        )
    }
    capture.flush()
    hooks = None
    try:
        if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("CUDA ground-truth capture requires a CUDA device")
        torch.cuda.set_device(torch.device(args.device))
        started = time.perf_counter()
        pipe, checkpoint = load_reference(
            args.variant, args.checkpoint, device=args.device, local_files_only=args.local_files_only
        )
        capture.manifest["load_seconds"] = time.perf_counter() - started
        capture.manifest["status"] = "hashing_checkpoint"
        capture.flush()
        capture.manifest["checkpoint"] = checkpoint_manifest(checkpoint, args.variant)
        capture.manifest["environment"]["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
        capture.manifest["environment"]["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(
            torch.device(args.device)
        ).total_memory
        capture.manifest["environment"]["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
        capture.manifest["environment"]["cudnn_tf32"] = torch.backends.cudnn.allow_tf32
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        generator = torch.Generator(device="cpu" if args.generator_device == "cpu" else args.device).manual_seed(
            args.seed
        )
        capture.save(
            "rng/before",
            {
                "generator": generator.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state(torch.device(args.device)),
            },
        )
        pipe.scheduler.set_timesteps(args.steps, device=args.device)
        capture.save(
            "scheduler",
            {
                "sigmas": pipe.scheduler.sigmas,
                "timesteps": pipe.scheduler.timesteps,
                "config": dict(pipe.scheduler.config),
            },
        )
        hooks = OracleHooks(pipe, capture, args.capture_layers, vae_stage_captures=not args.no_vae_stage_captures)
        hooks.install()
        capture.manifest["status"] = "running"
        capture.flush()
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        torch.cuda.synchronize()
        begin = time.perf_counter()
        result = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=3.0,
            negative_prompt=args.negative_prompt,
            generator=generator,
            uncond_skip_layers=[22],
        )
        torch.cuda.synchronize()
        capture.manifest["generation_seconds_including_capture"] = time.perf_counter() - begin
        result.images[0].save(capture.path / "image.png")
        capture.manifest["image"] = {"path": "image.png", "sha256": sha256(capture.path / "image.png")}
        capture.save(
            "rng/after",
            {
                "generator": generator.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state(torch.device(args.device)),
            },
        )
        capture.manifest["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(torch.device(args.device))
        capture.manifest["status"] = "complete"
        capture.manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        capture.flush()
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output_dir": str(capture.path),
                    "steps": hooks.step,
                    "capture_gib": capture.bytes / 1024**3,
                    "generation_seconds_including_capture": capture.manifest["generation_seconds_including_capture"],
                }
            ),
            flush=True,
        )
    except BaseException as exc:
        capture.manifest["status"] = "failed"
        capture.manifest["error"] = f"{type(exc).__name__}: {exc}"
        capture.flush()
        raise
    finally:
        if hooks:
            hooks.close()


if __name__ == "__main__":
    main()
