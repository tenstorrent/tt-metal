# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Build and verify a MiniMax-H3 AdaLN table against the real checkpoint.

Not a pytest: it streams 26 GB of ``adaln_proj`` weights, so it runs on demand and
writes the table plus a manifest that ``test_adaln_precompute_minimax_h3.py``
consumes as its parity fixture.

    python models/tt_dit/tests/models/minimax_h3/build_adaln_table.py \
        --checkpoint <FL2VA/transformer> --out ~/h3_adaln_table.pt
"""

import argparse
import time
from pathlib import Path

import torch
from safetensors import safe_open

from models.tt_dit.pipelines.minimax_h3.adaln_precompute import (
    MINIMAX_H3_MODALITY_NUM,
    precompute_adaln_table,
    project_block_adaln,
    project_final_adaln,
    request_step_timesteps,
    time_embedding,
)
from models.tt_dit.pipelines.minimax_h3.packing import MINIMAX_H3_KEYFRAME_NOISE_AUG
from models.tt_dit.pipelines.minimax_h3.scheduler import MiniMaxH3Scheduler

# Blocks re-derived independently to check the table against. The projection is
# row-independent, so a sample catches layout and rounding errors; the expensive
# part is reading 520 MB per block.
VERIFY_LAYERS = (0, 1, 25, 49)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    video = MiniMaxH3Scheduler(12.0)
    audio = MiniMaxH3Scheduler(3.0)
    video.set_timesteps(args.steps)
    audio.set_timesteps(args.steps)
    step_timesteps = request_step_timesteps(video.sigmas, audio.sigmas, MINIMAX_H3_KEYFRAME_NOISE_AUG)
    total_rows = sum(int(levels.numel()) for levels in step_timesteps)
    print(
        f"steps={args.steps} evals={video.num_inference_steps} "
        f"table rows={total_rows} (per-step counts "
        f"{sorted({int(levels.numel()) for levels in step_timesteps})})",
        flush=True,
    )

    start = time.monotonic()
    table = precompute_adaln_table(args.checkpoint, step_timesteps)
    elapsed = time.monotonic() - start
    print(
        f"built in {elapsed:.1f}s: block_params {tuple(table.block_params.shape)} "
        f"{table.block_params.dtype}, {table.nbytes()/1e9:.3f} GB",
        flush=True,
    )

    checkpoint = Path(args.checkpoint)
    location, handles = {}, {}
    for shard in sorted(checkpoint.glob("model-*.safetensors")):
        handle = safe_open(shard, framework="pt", device="cpu")
        handles[shard] = handle
        for key in handle.keys():
            location[key] = shard

    def get(key):
        return handles[location[key]].get_tensor(key)

    proj_in_w, proj_in_b = get("time_embedder.proj_in.weight"), get("time_embedder.proj_in.bias")
    proj_out_w, proj_out_b = get("time_embedder.proj_out.weight"), get("time_embedder.proj_out.bias")
    weights = {
        layer: (get(f"blocks.{layer}.adaln_proj.linear.weight"), get(f"blocks.{layer}.adaln_proj.linear.bias"))
        for layer in VERIFY_LAYERS
    }
    final_weight = get("final_layer.adaln_proj.linear.weight")
    final_bias = get("final_layer.adaln_proj.linear.bias")

    block_mismatch = final_mismatch = checked = 0
    for step, levels in enumerate(step_timesteps):
        # Exactly what one denoise step of the reference would compute.
        temb = time_embedding(levels, proj_in_w, proj_in_b, proj_out_w, proj_out_b)
        rows = table.step_rows(step, torch.arange(levels.numel()))
        for layer in VERIFY_LAYERS:
            expected = project_block_adaln(temb, *weights[layer], table.hidden_size)
            got = table.block_params[
                layer, int(rows[0]) * MINIMAX_H3_MODALITY_NUM : (int(rows[-1]) + 1) * MINIMAX_H3_MODALITY_NUM
            ]
            checked += 1
            if not torch.equal(got, expected):
                block_mismatch += 1
                if block_mismatch <= 3:
                    delta = (got.float() - expected.float()).abs().max()
                    print(f"  BLOCK MISMATCH step={step} layer={layer} maxdiff={delta:.3e}", flush=True)

        shift, scale = project_final_adaln(temb, final_weight, final_bias)
        if not (
            torch.equal(table.final_shift[rows[0] : rows[-1] + 1], shift)
            and torch.equal(table.final_scale[rows[0] : rows[-1] + 1], scale)
        ):
            final_mismatch += 1

    print(
        f"parity: {checked} (step, layer) projections and {len(step_timesteps)} final-layer rows checked; "
        f"{block_mismatch} block mismatches, {final_mismatch} final mismatches",
        flush=True,
    )

    out = Path(args.out).expanduser()
    torch.save(
        {
            "timesteps": table.timesteps,
            "step_offsets": table.step_offsets,
            "block_params": table.block_params,
            "final_shift": table.final_shift,
            "final_scale": table.final_scale,
            "steps": args.steps,
            "verify_layers": VERIFY_LAYERS,
            "block_mismatch": block_mismatch,
            "final_mismatch": final_mismatch,
            "build_seconds": elapsed,
        },
        out,
    )
    print(f"wrote {out} ({out.stat().st_size/1e9:.3f} GB)", flush=True)
    raise SystemExit(1 if (block_mismatch or final_mismatch) else 0)


if __name__ == "__main__":
    main()
