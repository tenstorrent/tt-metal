# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Run the complete Qwen Image 2.1 DiT denoising schedule on one TT card.

CUDA supplies the pinned Qwen3-VL encoder output, initial noise and verified
sigma schedule. The image denoiser, output head and Euler updates run on TT.
The prefix is recomputed with block-causal attention each step until a TT KV
cache is implemented. Per-step CUDA targets are used only for validation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-bdf", required=True)
    parser.add_argument("--max-steps", type=int, help="stop a fresh diagnostic run after this many steps")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    if args.checkpoint.name != MODEL_REVISION:
        parser.error("checkpoint path must end in the pinned model revision")
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
    if manifest["capture_steps"] != list(range(total)):
        parser.error("per-step CUDA boundaries are required for comparison")
    count = total if args.max_steps is None else args.max_steps
    if count < 1 or count > total:
        parser.error("--max-steps must lie within the captured schedule")
    args.output_dir.mkdir(parents=True)

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
    if tuple(latents.shape) != (1, expected_tokens, 64) or prefix_count <= 0:
        parser.error("CUDA latent or token metadata does not match captured image resolution")
    if schedule.get("latent_tokens") != target_count:
        parser.error("sigma schedule token count differs from CUDA latent shape")
    if sequence != rope.shape[0] or target_mask.ndim != 1:
        parser.error("rotary embedding or target mask has incompatible sequence length")

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
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=[0])
        padded = (sequence + 31) // 32 * 32
        padded_rope = torch.cat((rope, torch.ones(padded - sequence, rope.shape[1], dtype=rope.dtype)))
        cos, sin = rotary_caches(padded_rope, device)
        mask = prefill_mask(sequence, segments, device, key_valid)
        input_weights = prepare_input_weights(checkpoint_tensors(input_names), device)
        output_state = checkpoint_tensors(("norm_out.linear.weight", "proj_out.weight"))
        output_weights = prepare_output_weights(
            output_state["norm_out.linear.weight"], output_state["proj_out.weight"], device
        )
        text_tt = text_projection(to_device(context, device), input_weights)
        latents_tt = to_device(latents, device)
        progress("running")
        for step in range(count):
            step_started = time.monotonic()
            cuda_step = args.cuda_dir / f"step_{step:03d}"
            captured_timestep = _read(cuda_step / "transformer/input/timestep.pt")
            features = sinusoidal_timestep(torch.cat((captured_timestep, captured_timestep.new_zeros(1))))
            temb, modulation = timestep_and_modulation(to_device(features, device), input_weights)
            selected = select_modulation_device(modulation, target_mask, device)
            image_tt = image_projection(latents_tt, input_weights)
            result = ttnn.concat((text_tt, image_tt), dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            block_errors = {}
            for layer in range(32):
                state = load_block_weights(args.checkpoint, layer)
                weights = prepare_block_weights(state, device)
                result = prefill_block(result, weights, selected, cos, sin, mask, sequence)
                expected_path = cuda_step / f"transformer/transformer_blocks.{layer}.pt"
                if expected_path.is_file():
                    actual_block = to_host(result, (1, sequence, 4096))[:, -target_count:]
                    expected_block = _read(expected_path)[:, -target_count:]
                    block_errors[str(layer)] = _relative_rms(actual_block, expected_block)
                del weights, state
            selected_temb = select_timestep_embedding_device(temb, target_mask, device)
            _, projection = output_head(result, selected_temb, output_weights)
            velocity = ttnn.slice(projection, (0, prefix_count, 0), (1, sequence, 64))
            velocity_host = to_host(velocity, (1, target_count, 64))
            cuda_velocity = _read(cuda_step / "transformer/proj_out.pt")[:, -target_count:]
            sigma, next_sigma = schedule["sigmas"][step : step + 2]
            latents_tt = flow_euler_step(latents_tt, velocity, sigma, next_sigma, device)
            latents_host = to_host(latents_tt, (1, target_count, 64))
            cuda_latents = _read(cuda_step / "scheduler/updated_latents.pt")
            output = args.output_dir / f"step_{step:03d}"
            output.mkdir()
            torch.save(latents_host, output / "updated_latents.pt")
            torch.save(velocity_host, output / "velocity.pt")
            row = {
                "step": step,
                "sigma": sigma,
                "next_sigma": next_sigma,
                "velocity_relative_rms_error": _relative_rms(velocity_host, cuda_velocity),
                "latent_relative_rms_error": _relative_rms(latents_host, cuda_latents),
                "velocity_pcc": _pcc(velocity_host, cuda_velocity),
                "latent_pcc": _pcc(latents_host, cuda_latents),
                "latent_mean_abs_error": float((latents_host.float() - cuda_latents.float()).abs().mean()),
                "block_relative_rms_errors": block_errors,
                "elapsed_seconds": time.monotonic() - step_started,
            }
            reports.append(row)
            _write_json(args.output_dir / "report.json", reports)
            progress("running" if step + 1 < count else ("complete" if count == total else "partial"))
            print(json.dumps(row), flush=True)
    except Exception as error:
        progress("failed", str(error))
        raise
    finally:
        if device is not None:
            ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
