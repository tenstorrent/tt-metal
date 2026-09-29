# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Check split-half TT RoPE against Qwen's interleaved CUDA complex RoPE."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--derived-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-bdf", required=True)
    parser.add_argument("--attention", action="store_true", help="also test masked scaled-dot-product attention")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = args.cuda_dir / "step_000/transformer"
    block = "transformer_blocks.0"

    def read(path: Path) -> torch.Tensor:
        return torch.load(path, map_location="cpu", weights_only=True)

    rope = read(base / block / "input/rotary_emb.pt")
    os.environ["TT_VISIBLE_DEVICES"] = args.device_bdf
    os.environ.pop("TT_METAL_VISIBLE_DEVICES", None)
    import ttnn

    from models.experimental.qwen_image_2_1.tt.tt_dit_components import (
        rotary_caches,
        rotary_split_half,
        split_half_indices,
        to_device,
        to_host,
    )

    device = None
    rows = []
    try:
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=[0])
        sequence = rope.shape[0]
        padded_sequence = (sequence + 31) // 32 * 32
        pad_tokens = padded_sequence - sequence
        padded_rope = torch.cat((rope, torch.ones(pad_tokens, rope.shape[1], dtype=rope.dtype)), dim=0)
        cos, sin = rotary_caches(padded_rope, device)
        indices = split_half_indices(128)
        inverse = torch.argsort(indices)
        rotated = {}
        for axis in ("q", "k"):
            source = read(base / f"{block}.attn.norm_{axis}.pt")
            expected = read(args.derived_dir / f"{axis}_after_rope.pt")
            preordered = source[..., indices].transpose(1, 2).contiguous()
            preordered = torch.nn.functional.pad(preordered, (0, 0, 0, pad_tokens))
            input_tt = to_device(preordered, device)
            output_tt = rotary_split_half(input_tt, cos, sin)
            rotated[axis] = output_tt
            actual = (
                to_host(output_tt, tuple(preordered.shape))[:, :, :sequence].transpose(1, 2)[..., inverse].contiguous()
            )
            torch.save(actual, args.output_dir / f"{axis}_after_rope.pt")
            difference = (actual.float() - expected.float()).abs()
            row = {
                "component": f"{axis}_after_rope",
                "max_abs_error": float(difference.max()),
                "mean_abs_error": float(difference.mean()),
                "fraction_within_atol_0_01": float((difference <= 0.01).float().mean()),
                "exact_fraction": float((actual == expected).float().mean()),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

        if args.attention:
            value = read(base / f"{block}.attn.to_v.pt").reshape(1, sequence, 32, 128)
            value = torch.nn.functional.pad(value.transpose(1, 2).contiguous(), (0, 0, 0, pad_tokens))
            value_tt = to_device(value, device)
            allowed = read(args.derived_dir / "attention_allowed.pt").bool()
            additive_mask = torch.full((1, 1, padded_sequence, padded_sequence), -10000.0, dtype=torch.bfloat16)
            additive_mask[0, 0, :sequence, :sequence] = torch.where(
                allowed,
                torch.zeros_like(allowed, dtype=torch.bfloat16),
                torch.full_like(allowed, -10000, dtype=torch.bfloat16),
            )
            additive_mask[0, 0, sequence:, 0] = 0
            mask_tt = to_device(additive_mask, device)
            context_tt = ttnn.transformer.scaled_dot_product_attention(
                rotated["q"],
                rotated["k"],
                value_tt,
                attn_mask=mask_tt,
                is_causal=False,
                scale=128**-0.5,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            actual_context = to_host(context_tt, (1, 32, padded_sequence, 128))[:, :, :sequence]
            actual_context = actual_context.transpose(1, 2).flatten(2, 3).contiguous()
            expected_context = read(args.derived_dir / "context.pt")
            torch.save(actual_context, args.output_dir / "context.pt")
            difference = (actual_context.float() - expected_context.float()).abs()
            row = {
                "component": "masked_attention_context",
                "max_abs_error": float(difference.max()),
                "mean_abs_error": float(difference.mean()),
                "fraction_within_atol_0_01": float((difference <= 0.01).float().mean()),
                "exact_fraction": float((actual_context == expected_context).float().mean()),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        if device is not None:
            ttnn.close_mesh_device(device)
    (args.output_dir / "report.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
