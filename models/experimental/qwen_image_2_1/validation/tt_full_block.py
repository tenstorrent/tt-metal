# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare one or more chained first-step TT blocks with CUDA captures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-bdf", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument(
        "--step", type=int, choices=(0, 1), default=0, help="denoising step in the two-step CUDA capture"
    )
    parser.add_argument("--end-layer", type=int, help="last layer in a device-resident chain; defaults to --layer")
    parser.add_argument(
        "--checkpoint", type=Path, help="resolved checkpoint; used when exported layer weights are absent"
    )
    parser.add_argument(
        "--with-output-head", action="store_true", help="run final adaptive norm and projection after layer 31"
    )
    parser.add_argument(
        "--with-scheduler", action="store_true", help="apply the first Euler update to TT's projected velocity"
    )
    parser.add_argument(
        "--with-input-projections", action="store_true", help="compute image, text, and timestep conditioning on TT"
    )
    parser.add_argument("--latents-from", type=Path, help="use a saved TT latent as this step's input")
    args = parser.parse_args()
    end_layer = args.layer if args.end_layer is None else args.end_layer
    if args.layer < 0 or end_layer < args.layer or end_layer >= 32:
        parser.error("layer range must lie within 0..31 and end at or after --layer")
    if args.with_output_head and (end_layer != 31 or args.checkpoint is None):
        parser.error("--with-output-head requires --end-layer 31 and --checkpoint")
    if args.with_scheduler and not args.with_output_head:
        parser.error("--with-scheduler requires --with-output-head")
    if args.with_input_projections and (args.layer != 0 or args.checkpoint is None):
        parser.error("--with-input-projections requires --layer 0 and --checkpoint")
    if args.latents_from is not None and (args.step != 1 or not args.with_input_projections):
        parser.error("--latents-from requires --step 1 and --with-input-projections")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    step_dir = f"step_{args.step:03d}"
    base = args.cuda_dir / step_dir / "transformer"
    prefix_base = args.cuda_dir / "step_000/transformer"
    block = f"transformer_blocks.{args.layer}"

    def read(relative: str) -> torch.Tensor:
        return torch.load(base / relative, map_location="cpu", weights_only=True)

    hidden = read(f"{block}/input/hidden_states.pt")
    target_count = torch.load(
        args.cuda_dir / step_dir / "transformer/input/hidden_states.pt",
        map_location="cpu",
        weights_only=True,
    ).shape[1]
    supplied_latents = (
        torch.load(args.latents_from, map_location="cpu", weights_only=True) if args.latents_from is not None else None
    )
    if supplied_latents is not None and tuple(supplied_latents.shape) != (1, target_count, 64):
        raise ValueError(f"saved TT latent has incompatible shape: {tuple(supplied_latents.shape)}")
    if args.step > 0:
        prefix_hidden = torch.load(
            prefix_base / f"{block}/input/hidden_states.pt", map_location="cpu", weights_only=True
        )
        prefix_count = prefix_hidden.shape[1] - target_count
        hidden = torch.cat((prefix_hidden[:, :prefix_count], hidden), dim=1)
    modulation = read(f"{block}/input/modulation.pt")
    target_mask = torch.load(prefix_base / f"{block}/input/target_token_mask.pt", map_location="cpu", weights_only=True)
    rope = torch.load(prefix_base / f"{block}/input/rotary_emb.pt", map_location="cpu", weights_only=True)
    segments = json.loads((prefix_base / block / "input/segments.json").read_text())

    os.environ["TT_VISIBLE_DEVICES"] = args.device_bdf
    os.environ.pop("TT_METAL_VISIBLE_DEVICES", None)
    import ttnn

    from models.experimental.qwen_image_2_1.tt.tt_attention import prefill_mask
    from models.experimental.qwen_image_2_1.tt.tt_block import (
        prefill_block,
        prepare_weights,
        select_modulation,
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
        select_timestep_embedding,
        select_timestep_embedding_device,
    )
    from models.experimental.qwen_image_2_1.tt.tt_scheduler import flow_euler_step
    from models.experimental.qwen_image_2_1.validation.export_block_weights import load_block_weights

    def matching_host(tensor, expected: torch.Tensor) -> torch.Tensor:
        actual = to_host(tensor, (1, hidden.shape[1], expected.shape[-1]))
        return actual[:, -target_count:] if args.step > 0 else actual

    device = None
    reports = []
    try:
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=[0])
        sequence = hidden.shape[1]
        padded = (sequence + 31) // 32 * 32
        padded_rope = torch.cat((rope, torch.ones(padded - sequence, rope.shape[1], dtype=rope.dtype)))
        cos, sin = rotary_caches(padded_rope, device)
        key_valid_path = prefix_base / block / "input/key_valid.pt"
        key_valid = torch.load(key_valid_path, weights_only=True) if key_valid_path.exists() else None
        mask = prefill_mask(sequence, segments, device, key_valid)
        temb_tt = None
        if args.with_input_projections:
            transformer = args.checkpoint / "transformer"
            mapping = json.loads((transformer / "diffusion_pytorch_model.safetensors.index.json").read_text())[
                "weight_map"
            ]
            input_names = (
                "img_in.weight",
                "txt_in.text_norm.weight",
                "txt_in.in_layer.weight",
                "txt_in.out_layer.weight",
                "time_text_embed.timestep_embedder.linear_1.weight",
                "time_text_embed.timestep_embedder.linear_2.weight",
                "modulation.1.weight",
            )
            input_state = {}
            for shard in sorted({mapping[name] for name in input_names}):
                with safe_open(transformer / shard, framework="pt", device="cpu") as source:
                    for name in input_names:
                        if mapping[name] == shard:
                            input_state[name] = source.get_tensor(name)
            input_weights = prepare_input_weights(input_state, device)
            latents = supplied_latents if supplied_latents is not None else read("input/hidden_states.pt")
            context = read("input/encoder_hidden_states.pt")
            timestep = read("input/timestep.pt")
            image_tt = image_projection(to_device(latents, device), input_weights)
            text_tt = text_projection(to_device(context, device), input_weights)
            features = sinusoidal_timestep(torch.cat((timestep, timestep.new_zeros(1))))
            temb_tt, modulation_tt = timestep_and_modulation(to_device(features, device), input_weights)
            selected = select_modulation_device(modulation_tt, target_mask, device)
            result = ttnn.concat((text_tt, image_tt), dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for name, tensor in (
                ("img_in", image_tt),
                ("txt_in", text_tt),
                ("time_text_embed", temb_tt),
                ("modulation", modulation_tt),
            ):
                expected = read(f"{name}.pt")
                actual = to_host(tensor, tuple(expected.shape))
                output = args.output_dir / step_dir / "transformer" / f"{name}.pt"
                output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(actual, output)
                difference = (actual.float() - expected.float()).abs()
                report = {
                    "component": name,
                    "relative_rms_error": float(
                        difference.square().mean().sqrt() / expected.float().square().mean().sqrt()
                    ),
                    "mean_abs_error": float(difference.mean()),
                }
                reports.append(report)
                print(json.dumps(report), flush=True)
            actual_joint = matching_host(result, read(f"{block}/input/hidden_states.pt"))
            expected_joint = read(f"{block}/input/hidden_states.pt")
            difference = (actual_joint.float() - expected_joint.float()).abs()
            report = {
                "component": "block_0_input",
                "relative_rms_error": float(
                    difference.square().mean().sqrt() / expected_joint.float().square().mean().sqrt()
                ),
                "mean_abs_error": float(difference.mean()),
            }
            reports.append(report)
            print(json.dumps(report), flush=True)
        else:
            selected = select_modulation(modulation, target_mask, device)
            result = to_device(hidden, device)
        for layer in range(args.layer, end_layer + 1):
            current_block = f"transformer_blocks.{layer}"
            state_path = args.cuda_dir / f"weights/transformer_block_{layer:02d}.pt"
            if state_path.is_file():
                state = torch.load(state_path, map_location="cpu", weights_only=True)
            elif args.checkpoint is not None:
                state = load_block_weights(args.checkpoint, layer)
            else:
                raise FileNotFoundError(f"missing {state_path}; pass --checkpoint")
            expected = read(f"{current_block}.pt")
            weights = prepare_weights(state, device)
            result = prefill_block(result, weights, selected, cos, sin, mask, sequence)
            actual = matching_host(result, expected)
            relative = Path(f"{step_dir}/transformer/{current_block}.pt")
            output = args.output_dir / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(actual, output)
            difference = (actual.float() - expected.float()).abs()
            report = {
                "layer": layer,
                "shape": list(expected.shape),
                "max_abs_error": float(difference.max()),
                "mean_abs_error": float(difference.mean()),
                "relative_rms_error": float(
                    difference.square().mean().sqrt() / expected.float().square().mean().sqrt()
                ),
                "fraction_within_atol_0_01": float((difference <= 0.01).float().mean()),
            }
            reports.append(report)
            print(json.dumps(report), flush=True)
            del weights, state
        if args.with_output_head:
            transformer = args.checkpoint / "transformer"
            mapping = json.loads((transformer / "diffusion_pytorch_model.safetensors.index.json").read_text())[
                "weight_map"
            ]

            def output_weight(name: str) -> torch.Tensor:
                with safe_open(transformer / mapping[name], framework="pt", device="cpu") as source:
                    return source.get_tensor(name)

            output_weights = prepare_output_weights(
                output_weight("norm_out.linear.weight"), output_weight("proj_out.weight"), device
            )
            selected_temb = (
                select_timestep_embedding_device(temb_tt, target_mask, device)
                if temb_tt is not None
                else select_timestep_embedding(read("time_text_embed.pt"), target_mask, device)
            )
            normalized, projection = output_head(result, selected_temb, output_weights)
            for name, tensor in (("norm_out", normalized), ("proj_out", projection)):
                expected = read(f"{name}.pt")
                actual = matching_host(tensor, expected)
                output = args.output_dir / f"{step_dir}/transformer/{name}.pt"
                torch.save(actual, output)
                difference = (actual.float() - expected.float()).abs()
                report = {
                    "component": name,
                    "shape": list(expected.shape),
                    "max_abs_error": float(difference.max()),
                    "mean_abs_error": float(difference.mean()),
                    "relative_rms_error": float(
                        difference.square().mean().sqrt() / expected.float().square().mean().sqrt()
                    ),
                }
                reports.append(report)
                print(json.dumps(report), flush=True)
            if args.with_scheduler:
                latents = torch.load(
                    args.cuda_dir / step_dir / "transformer/input/hidden_states.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                if supplied_latents is not None:
                    latents = supplied_latents
                velocity = ttnn.slice(projection, (0, sequence - target_count, 0), (1, sequence, 64))
                sigmas = (1.0, 0.02, 0.0)
                updated = flow_euler_step(
                    to_device(latents, device), velocity, sigmas[args.step], sigmas[args.step + 1], device
                )
                expected = torch.load(
                    args.cuda_dir / step_dir / "scheduler/updated_latents.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                actual = to_host(updated, tuple(expected.shape))
                output = args.output_dir / step_dir / "scheduler/updated_latents.pt"
                output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(actual, output)
                difference = (actual.float() - expected.float()).abs()
                report = {
                    "component": "updated_latents",
                    "shape": list(expected.shape),
                    "max_abs_error": float(difference.max()),
                    "mean_abs_error": float(difference.mean()),
                    "relative_rms_error": float(
                        difference.square().mean().sqrt() / expected.float().square().mean().sqrt()
                    ),
                }
                reports.append(report)
                print(json.dumps(report), flush=True)
        (args.output_dir / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
    finally:
        if device is not None:
            ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
