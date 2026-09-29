# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Run the complete first Qwen Image 2.1 attention module on TT."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-bdf", required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    relative = Path("step_000/transformer/transformer_blocks.0.attn.pt")
    base = args.cuda_dir / "step_000/transformer"
    block = "transformer_blocks.0"
    state = torch.load(args.cuda_dir / "weights/transformer_block_00.pt", map_location="cpu", weights_only=True)
    hidden = torch.load(base / f"{block}.attn" / "input.pt", map_location="cpu", weights_only=True)
    rope = torch.load(base / block / "input/rotary_emb.pt", map_location="cpu", weights_only=True)
    segments = json.loads((base / block / "input/segments.json").read_text())
    expected = torch.load(args.cuda_dir / relative, map_location="cpu", weights_only=True)

    os.environ["TT_VISIBLE_DEVICES"] = args.device_bdf
    os.environ.pop("TT_METAL_VISIBLE_DEVICES", None)
    import ttnn

    from models.experimental.qwen_image_2_1.tt.tt_attention import prefill_attention, prefill_mask, prepare_weights
    from models.experimental.qwen_image_2_1.tt.tt_dit_components import rotary_caches, to_device, to_host

    device = None
    try:
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=[0])
        weights = prepare_weights(state, device)
        sequence = hidden.shape[1]
        padded = (sequence + 31) // 32 * 32
        padded_rope = torch.cat((rope, torch.ones(padded - sequence, rope.shape[1], dtype=rope.dtype)))
        cos, sin = rotary_caches(padded_rope, device)
        key_valid_path = base / block / "input/key_valid.pt"
        key_valid = torch.load(key_valid_path, weights_only=True) if key_valid_path.exists() else None
        mask = prefill_mask(sequence, segments, device, key_valid)
        result = prefill_attention(to_device(hidden, device), weights, cos, sin, mask, sequence)
        actual = to_host(result, tuple(expected.shape))
        output = args.output_dir / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(actual, output)
        difference = (actual.float() - expected.float()).abs()
        report = {
            "component": "first_block_attention",
            "shape": list(expected.shape),
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
            "relative_rms_error": float(difference.square().mean().sqrt() / expected.float().square().mean().sqrt()),
            "fraction_within_atol_0_01": float((difference <= 0.01).float().mean()),
        }
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        if device is not None:
            ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
