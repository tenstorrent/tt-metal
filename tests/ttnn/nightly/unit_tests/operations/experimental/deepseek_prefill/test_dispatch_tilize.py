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


def _align32(c):
    return ((c + 31) // 32) * 32


def _build_routing(counts, experts_per_chip, device):
    """Construct [1,E] uint32 (region_offsets, counts) matching offset_cumsum's convention:
    region = exclusive prefix sum of align32(counts), restarting every experts_per_chip group.
    Returns (region_tt, counts_tt, valid_rows) where valid_rows = max_e(region[e]+align32(counts[e]))."""
    E = len(counts)
    aligned = [_align32(c) for c in counts]
    region = [0] * E
    for g in range(E // experts_per_chip):
        acc = 0
        for i in range(experts_per_chip):
            idx = g * experts_per_chip + i
            region[idx] = acc
            acc += aligned[idx]
    valid_rows = max(region[e] + aligned[e] for e in range(E))

    def _u32(vals):
        return ttnn.from_torch(
            torch.tensor(vals, dtype=torch.int64).reshape(1, E).to(torch.int32),
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return _u32(region), _u32(counts), valid_rows


# rows=2048, emb=3072, one chip (experts_per_chip == num_experts). Each case exercises a fill regime:
# dense fills the whole buffer, skew a partial prefix, one-hot a single region, empty nothing.
@pytest.mark.parametrize(
    "counts,label",
    [
        ([500, 500, 500, 500], "dense_full"),  # align32 -> 512*4 = 2048 = full buffer
        ([300, 100, 50, 20], "skew"),  # -> 320+128+64+32 = 544 valid rows
        ([600, 0, 0, 0], "one_hot"),  # -> 608 valid rows
        ([0, 0, 0, 0], "all_empty"),  # -> 0 valid rows (tail undefined; just must not hang)
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
@pytest.mark.parametrize(
    "in_dtype", [ttnn.bfloat16, pytest.param(ttnn.fp8_e4m3, id="fp8")], ids=lambda d: str(d).split(".")[-1]
)
def test_dispatch_tilize_skip_vs_to_layout(device, counts, label, in_dtype):
    """Stage 2: region-aware dispatch_tilize over [0:valid_rows] == to_layout over the same rows, bit-for-bit."""
    if in_dtype == ttnn.fp8_e4m3 and device.arch() != ttnn.Arch.BLACKHOLE:
        pytest.skip("fp8_e4m3 is Blackhole-only")
    rows, emb, out_dtype = 2048, 3072, ttnn.bfloat8_b
    experts_per_chip = len(counts)
    torch.manual_seed(0)
    x = _rand_rm(rows, emb, torch.bfloat16, device, in_dtype)
    region_tt, counts_tt, valid_rows = _build_routing(counts, experts_per_chip, device)

    ref = ttnn.to_layout(x, ttnn.TILE_LAYOUT, dtype=out_dtype)
    got = ttnn.experimental.deepseek_prefill.dispatch_tilize(
        x, region_tt, counts_tt, output_dtype=out_dtype, experts_per_chip=experts_per_chip
    )

    ref_t = ttnn.to_torch(ref).float()
    got_t = ttnn.to_torch(got).float()
    assert got_t.shape == ref_t.shape, f"shape {got_t.shape} != {ref_t.shape}"
    if valid_rows == 0:
        return  # nothing tilized; output tail is undefined by design
    ref_v = ref_t[:valid_rows]
    got_v = got_t[:valid_rows]
    assert torch.equal(got_v, ref_v), (
        f"[{label}] rows[0:{valid_rows}] not byte-identical: "
        f"max|Δ|={(got_v - ref_v).abs().max().item()}, mismatched={int((got_v != ref_v).sum())}/{got_v.numel()}"
    )
