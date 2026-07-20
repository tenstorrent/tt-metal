# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the region-aware dispatch_tilize op.

Stage 1 (this file, full path): dispatch_tilize with no routing metadata must be byte-identical
to ttnn.to_layout(TILE, dtype=...) — it tilizes the whole buffer.
Stage 2 (region-aware): with expert_region_offsets + total_counts_per_expert, only the filled
prefix [0:valid_rows] is tilized; those rows must be byte-identical to to_layout, tail is undefined.
"""

import pytest
import torch
import ttnn


def _rand_rm(rows, emb, torch_dtype, device, tt_dtype):
    t = torch.randn(rows, emb, dtype=torch_dtype)
    src_dtype = torch.float32 if tt_dtype == ttnn.fp8_e4m3 else torch_dtype
    return ttnn.from_torch(
        t.to(src_dtype),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=tt_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("rows,emb", [(512, 1536), (1024, 3072), (2048, 6144)])
@pytest.mark.parametrize(
    "in_dtype", [ttnn.bfloat16, pytest.param(ttnn.fp8_e4m3, id="fp8")], ids=lambda d: str(d).split(".")[-1]
)
@pytest.mark.parametrize("out_dtype", [ttnn.bfloat8_b], ids=["bf8"])
def test_dispatch_tilize_full_vs_to_layout(device, rows, emb, in_dtype, out_dtype):
    """Stage 1: full dispatch_tilize == ttnn.to_layout(TILE, dtype) bit-for-bit."""
    if in_dtype == ttnn.fp8_e4m3 and device.arch() != ttnn.Arch.BLACKHOLE:
        pytest.skip("fp8_e4m3 is Blackhole-only")
    torch.manual_seed(0)
    x = _rand_rm(rows, emb, torch.bfloat16, device, in_dtype)

    ref = ttnn.to_layout(x, ttnn.TILE_LAYOUT, dtype=out_dtype)
    got = ttnn.experimental.deepseek_prefill.dispatch_tilize(x, output_dtype=out_dtype)

    ref_t = ttnn.to_torch(ref).float()
    got_t = ttnn.to_torch(got).float()
    assert got_t.shape == ref_t.shape, f"shape {got_t.shape} != {ref_t.shape}"
    assert torch.equal(got_t, ref_t), (
        f"not byte-identical: max|Δ|={(got_t - ref_t).abs().max().item()}, "
        f"mismatched={int((got_t != ref_t).sum())}/{got_t.numel()}"
    )
