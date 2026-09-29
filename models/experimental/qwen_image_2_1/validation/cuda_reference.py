# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Capture CUDA reference tensors at the boundaries of Qwen Image 2.1.

The captures deliberately come from the pinned upstream Diffusers pipeline.
The independent TT implementation consumes these tensors without importing its
own code into this process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline
from huggingface_hub import snapshot_download

from models.experimental.qwen_image_2_1.checkpoint import MODEL_ID, MODEL_REVISION

DEFAULT_PROMPT = "the quick brown fox jumps over the lazy dog"


def _first_tensor(value: object) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if hasattr(value, "sample"):
        return _first_tensor(value.sample)
    return None


class Capture:
    THIN_TENSORS = {
        "transformer/input/hidden_states",
        "transformer/input/encoder_hidden_states",
        "transformer/input/timestep",
        "transformer/input/img_mask",
        "transformer/input/encoder_hidden_states_mask",
        "transformer/output",
        "transformer/proj_out",
        "scheduler/updated_latents",
    }

    def __init__(self, root: Path, steps: set[int], verbose: bool, full_steps: set[int] | None = None):
        self.root = root
        self.steps = steps
        self.full_steps = steps if full_steps is None else full_steps
        self.verbose = verbose
        self.current_step = -1
        self.records: list[dict[str, object]] = []

    def save(self, name: str, value: object) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            return
        if self.current_step >= 0 and self.current_step not in self.steps:
            return
        if self.current_step >= 0 and self.current_step not in self.full_steps and name not in self.THIN_TENSORS:
            return
        tensor = tensor.detach().to("cpu").contiguous()
        relative = f"step_{self.current_step:03d}/{name}.pt"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.records.append(
            {
                "name": name,
                "step": self.current_step,
                "path": relative,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": digest,
            }
        )
        if self.verbose:
            print(f"captured {relative} {list(tensor.shape)} {tensor.dtype}", flush=True)

    def hook(self, name: str):
        def save_output(_module, _args, output):
            self.save(name, output)

        return save_output


def _attach_hooks(pipe: QwenImage21Pipeline, capture: Capture):
    handles = []

    def transformer_input(_module, _args, kwargs):
        capture.current_step += 1
        print(f"CUDA denoise step {capture.current_step} started", flush=True)
        for key in (
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "img_mask",
            "encoder_hidden_states_mask",
        ):
            if key in kwargs:
                capture.save(f"transformer/input/{key}", kwargs[key])
        if capture.current_step in capture.full_steps:
            path = capture.root / f"step_{capture.current_step:03d}" / "transformer" / "input" / "img_shapes.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(kwargs["img_shapes"]))

    handles.append(pipe.transformer.register_forward_pre_hook(transformer_input, with_kwargs=True))
    handles.append(pipe.transformer.register_forward_hook(capture.hook("transformer/output")))

    for name, module in pipe.transformer.named_modules():
        if name in {"img_in", "txt_in", "time_text_embed", "modulation", "norm_out", "proj_out"}:
            handles.append(module.register_forward_hook(capture.hook(f"transformer/{name}")))
        elif re.fullmatch(
            r"transformer_blocks\.\d+(?:\.(?:img_norm1|attn|img_norm2|img_mlp|attn\.(?:to_q|to_k|to_v|norm_q|norm_k|to_out\.0)|img_mlp\.(?:proj|gate_layer|out)))?",
            name,
        ):
            handles.append(module.register_forward_hook(capture.hook(f"transformer/{name}")))
            if re.fullmatch(r"transformer_blocks\.\d+\.(?:attn|img_mlp)", name):

                def submodule_input(_module, args, kwargs, module_name=name):
                    value = kwargs.get("hidden_states", args[0] if args else None)
                    capture.save(f"transformer/{module_name}/input", value)

                handles.append(module.register_forward_pre_hook(submodule_input, with_kwargs=True))
            block = re.fullmatch(r"transformer_blocks\.(\d+)", name)
            if block:
                index = int(block.group(1))

                def block_input(_module, _args, kwargs, block_index=index):
                    base = f"transformer/transformer_blocks.{block_index}/input"
                    for key in ("hidden_states", "modulation", "rotary_emb", "target_token_mask", "key_valid"):
                        if key in kwargs:
                            capture.save(f"{base}/{key}", kwargs[key])
                    if capture.current_step in capture.full_steps and kwargs.get("segments") is not None:
                        path = capture.root / f"step_{capture.current_step:03d}" / base / "segments.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(kwargs["segments"]))

                handles.append(module.register_forward_pre_hook(block_input, with_kwargs=True))

    for name, module in pipe.text_encoder.named_modules():
        if re.fullmatch(r"model\.(?:language_model\.)?layers\.\d+", name):
            handles.append(module.register_forward_hook(capture.hook(f"text_encoder/{name}")))

    for name, module in pipe.vae.decoder.named_modules():
        if name == "" or re.fullmatch(r"(?:conv_in|mid_block|conv_out|up_blocks\.\d+)", name):
            handles.append(module.register_forward_hook(capture.hook(f"vae/decoder/{name or 'output'}")))

    return handles


def _steps(value: str, count: int) -> set[int]:
    if value == "all":
        return set(range(count))
    result = {int(item) for item in value.split(",")}
    if not result or min(result) < 0 or max(result) >= count:
        raise ValueError(f"capture steps must be in [0, {count - 1}]")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--capture-steps", default="0")
    parser.add_argument(
        "--full-capture-steps",
        default=None,
        help="steps with all module boundaries; other captured steps keep only input/output/latent boundaries",
    )
    parser.add_argument("--no-cpu-offload", action="store_true")
    parser.add_argument("--weight-layers", default="0", help="comma-separated DiT layer indices; empty disables")
    parser.add_argument("--verbose-captures", action="store_true")
    args = parser.parse_args()

    if args.steps < 2:
        raise ValueError("Qwen Image 2.1's dynamic sigma shift is undefined for one step; use at least two")
    if args.height < 32 or args.width < 32 or args.height % 32 or args.width % 32:
        raise ValueError("height and width must each be positive multiples of 32 pixels")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the reference capture")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    capture_steps = _steps(args.capture_steps, args.steps)
    full_steps = capture_steps if args.full_capture_steps is None else _steps(args.full_capture_steps, args.steps)
    if not full_steps.issubset(capture_steps):
        parser.error("--full-capture-steps must be a subset of --capture-steps")
    capture = Capture(args.output_dir, capture_steps, args.verbose_captures, full_steps)
    start = time.monotonic()
    pipe = QwenImage21Pipeline.from_pretrained(checkpoint, dtype=torch.bfloat16, local_files_only=True)
    if args.no_cpu_offload:
        pipe.to("cuda:0")
    else:
        pipe.enable_model_cpu_offload(gpu_id=0)
    handles = _attach_hooks(pipe, capture)
    if args.weight_layers:
        for layer_index in [int(item) for item in args.weight_layers.split(",")]:
            layer = pipe.transformer.transformer_blocks[layer_index]
            path = args.output_dir / "weights" / f"transformer_block_{layer_index:02d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({key: value.detach().cpu() for key, value in layer.state_dict().items()}, path)
            print(f"saved {path}", flush=True)

    generator = torch.Generator(device="cuda:0").manual_seed(args.seed)

    def after_step(_pipe, step, _timestep, values):
        if step in capture.steps:
            capture.save("scheduler/updated_latents", values["latents"])
        print(f"CUDA denoise step {step} completed", flush=True)
        return values

    try:
        with torch.inference_mode():
            result = pipe(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                generator=generator,
                output_type="pil",
                use_kv_cache=True,
                callback_on_step_end=after_step,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        result.images[0].save(args.output_dir / "reference.png")
    finally:
        for handle in handles:
            handle.remove()

    manifest = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "diffusers_version": __import__("diffusers").__version__,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "prompt": args.prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "capture_steps": sorted(capture.steps),
        "full_capture_steps": sorted(capture.full_steps),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "elapsed_seconds": time.monotonic() - start,
        "captures": capture.records,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved {len(capture.records)} tensors to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
