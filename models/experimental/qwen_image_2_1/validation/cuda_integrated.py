# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Generate a fresh image with the pinned CUDA pipeline, without injected activations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline
from huggingface_hub import snapshot_download

from models.experimental.qwen_image_2_1.validation.cuda_reference import DEFAULT_PROMPT
from models.experimental.qwen_image_2_1.checkpoint import MODEL_ID, MODEL_REVISION


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.steps < 2 or min(args.height, args.width) < 32:
        parser.error("at least two steps and positive dimensions >=32 are required")
    if args.height % 32 or args.width % 32:
        parser.error("dimensions must be multiples of 32")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    manifest = {
        "backend": "cuda",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt": args.prompt,
        "prompt_expansion": False,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "activation_injection": False,
        "input_provenance": "raw prompt and seeded CUDA generator; upstream pipeline computes all activations",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "diffusers_version": __import__("diffusers").__version__,
        "transformers_version": __import__("transformers").__version__,
        "use_kv_cache": True,
        "true_cfg_scale": 1.0,
        "cpu_offload": True,
    }
    write_json(args.output_dir / "manifest.json", manifest)

    def progress(status: str, stage: str, steps: int = 0, **extra) -> None:
        write_json(
            args.output_dir / "progress.json",
            {
                "status": status,
                "stage": stage,
                "completed_steps": steps,
                "total_steps": args.steps,
                "elapsed_seconds": time.monotonic() - start,
                **extra,
            },
        )
        print(f"CUDA {stage}: {steps}/{args.steps}", flush=True)

    progress("loading", "checkpoint")
    handles = []
    completed = 0
    module_times = []
    module_started = {}
    try:
        checkpoint = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        pipe = QwenImage21Pipeline.from_pretrained(checkpoint, dtype=torch.bfloat16, local_files_only=True)
        pipe.enable_model_cpu_offload(gpu_id=0)

        def begin_module(name):
            def hook(_module, _args):
                torch.cuda.synchronize()
                module_started[name] = time.monotonic()

            return hook

        def end_module(name):
            def hook(_module, _args, _output):
                torch.cuda.synchronize()
                module_times.append(
                    {
                        "module": name,
                        "step": completed if name == "denoiser" else None,
                        "seconds": time.monotonic() - module_started[name],
                    }
                )
                write_json(
                    args.output_dir / "runtime.json",
                    {
                        "backend": "CUDA",
                        "timing": "synchronized host wall time; module hooks include offload/observation overhead where inside module boundary",
                        "warmup": "no in-process warmup",
                        "samples": 1,
                        "modules": module_times,
                    },
                )

            return hook

        for name, module in (
            ("prompt_encoder", pipe.text_encoder),
            ("denoiser", pipe.transformer),
            ("vae_decoder", pipe.vae.decoder),
        ):
            handles.append(module.register_forward_pre_hook(begin_module(name)))
            handles.append(module.register_forward_hook(end_module(name)))

        def transformer_input(_module, _args, kwargs):
            if completed == 0:
                # Observe native values after upstream preparation. Never return replacement inputs.
                for key in ("hidden_states", "encoder_hidden_states", "encoder_hidden_states_mask"):
                    value = kwargs.get(key)
                    if isinstance(value, torch.Tensor):
                        torch.save(value.detach().cpu(), args.output_dir / f"initial_{key}.pt")
            progress("running", "denoising", completed)

        def vae_input(_module, inputs):
            progress("decoding", "vae", completed)
            if inputs and isinstance(inputs[0], torch.Tensor):
                torch.save(inputs[0].detach().cpu(), args.output_dir / "vae_input.pt")

        handles.append(pipe.transformer.register_forward_pre_hook(transformer_input, with_kwargs=True))
        handles.append(pipe.vae.post_quant_conv.register_forward_pre_hook(vae_input))

        def after_step(_pipe, step, _timestep, values):
            nonlocal completed
            completed = step + 1
            if completed == args.steps:
                torch.save(values["latents"].detach().cpu(), args.output_dir / "final_latents.pt")
            progress("running", "denoising", completed)
            return values

        progress("encoding", "prompt_encoder")
        generator = torch.Generator(device="cuda:0").manual_seed(args.seed)
        with torch.inference_mode():
            result = pipe(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                generator=generator,
                true_cfg_scale=1.0,
                output_type="pil",
                use_kv_cache=True,
                callback_on_step_end=after_step,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        result.images[0].save(args.output_dir / "cuda.png")
        manifest["elapsed_seconds"] = time.monotonic() - start
        manifest["image_sha256"] = hashlib.sha256((args.output_dir / "cuda.png").read_bytes()).hexdigest()
        write_json(args.output_dir / "manifest.json", manifest)
        progress("complete", "saved", completed)
    except Exception as error:
        progress("failed", "error", completed, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
