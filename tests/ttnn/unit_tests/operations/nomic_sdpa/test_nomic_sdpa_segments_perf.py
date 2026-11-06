# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf harness for nomic_sdpa's segmented mode (scripts/run_safe_pytest.sh --profile): NOMIC_SDPA_REPS calls
(default 5) each of, in order,
  1. 8 x 512 unpadded (the op's tuned path),
  2. 8 x 512 with key_lengths (change 12),
  3. 8 x 512 packed with segments (texts of real lengths side by side),
  4. 4 x 1024 packed with segments,
  5. stock SDPA with the block-diagonal mask on the same 4 x 1024 layout (what packed passes ran before).
Device kernel time comes from the profiler CSV, one row a call. Outputs are only checked finite."""

import math
import os

import pytest
import torch
import ttnn

from ttnn.operations.nomic_sdpa import nomic_sdpa

HEADS, HEAD_DIM, TILE = 12, 64, 32
# Real document lengths (SciFact + XQuAD, the 4096-token budget batch): texts on tiles, side by side.
LENGTHS = [512, 400, 419, 260, 287, 398, 112, 246, 512, 352, 262]


def _pack(width):
    rows, loads = [], []
    for length in sorted(LENGTHS, reverse=True):
        span = math.ceil(length / TILE) * TILE
        for r, load in enumerate(loads):
            if load + span <= width:
                rows[r].append((load, length))
                loads[r] += span
                break
        else:
            rows.append([(0, length)])
            loads.append(span)
    return rows


def _tensors(device, rows, width):
    g = torch.Generator().manual_seed(0)
    return [
        ttnn.from_torch(
            (torch.randn(rows, HEADS, width, HEAD_DIM, generator=g) * s).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for s in (0.25, 1.0, 1.0)
    ]


def test_nomic_sdpa_segments_perf(device):
    reps = int(os.environ.get("NOMIC_SDPA_REPS", "5"))
    mem = ttnn.L1_MEMORY_CONFIG

    def run(fn):
        for _ in range(reps):
            out = fn()
            assert torch.isfinite(ttnn.to_torch(out).float()).all()
            ttnn.deallocate(out)

    t8 = _tensors(device, 8, 512)
    run(lambda: nomic_sdpa(*t8, memory_config=mem))
    run(lambda: nomic_sdpa(*t8, key_lengths=[512, 512, 419, 400, 398, 352, 287, 262], memory_config=mem))
    seg8 = _pack(512)
    seg8 = seg8 + [[] for _ in range(8 - len(seg8))] if len(seg8) <= 8 else seg8[:8]
    run(lambda: nomic_sdpa(*t8, segments=seg8, memory_config=mem))
    t4 = _tensors(device, 4, 1024)
    seg4 = _pack(1024)
    assert len(seg4) == 4, seg4
    run(lambda: nomic_sdpa(*t4, segments=seg4, memory_config=mem))

    ids = torch.full((4, 1024), -1, dtype=torch.long)
    for r, texts in enumerate(seg4):
        for i, (start, length) in enumerate(texts):
            ids[r, start : start + length] = i
    same = ids[:, None, :, None] == ids[:, None, None, :]
    mask = torch.zeros(same.shape, dtype=torch.bfloat16).masked_fill(~same, torch.finfo(torch.bfloat16).min)
    tmask = ttnn.typecast(ttnn.from_torch(mask, layout=ttnn.TILE_LAYOUT, device=device), ttnn.bfloat4_b)
    program = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=128,
        k_chunk_size=256,
        exp_approx_mode=True,
    )
    compute = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=False
    )
    run(
        lambda: ttnn.transformer.scaled_dot_product_attention(
            *t4,
            attn_mask=tmask,
            is_causal=False,
            scale=1.0,
            memory_config=mem,
            output_concat_heads=True,
            program_config=program,
            compute_kernel_config=compute,
        )
    )
