# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the region-aware dispatch_tilize op.

Stage 1 (full path): dispatch_tilize with no routing metadata must be byte-identical to
ttnn.to_layout(TILE, dtype=...) — it tilizes the whole buffer.
Stage 2 (skip path): with total_counts_per_expert + experts_per_chip, only the filled prefix
[0:valid_rows] is tilized (valid_rows = the fullest chip's Σ align32(count)); those rows must be
byte-identical to to_layout, the padded tail is undefined.
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
@pytest.mark.parametrize("out_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bf8", "bf16"])
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
    """Construct the [1,E] uint32 total_counts_per_expert tensor and the expected valid_rows.
    valid_rows = fullest chip's fill = max over experts_per_chip groups of Σ align32(count[e])."""
    E = len(counts)
    aligned = [_align32(c) for c in counts]
    valid_rows = max(
        sum(aligned[g * experts_per_chip + i] for i in range(experts_per_chip)) for g in range(E // experts_per_chip)
    )
    counts_tt = ttnn.from_torch(
        torch.tensor(counts, dtype=torch.int64).reshape(1, E).to(torch.int32),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return counts_tt, valid_rows


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
    counts_tt, valid_rows = _build_routing(counts, experts_per_chip, device)

    ref = ttnn.to_layout(x, ttnn.TILE_LAYOUT, dtype=out_dtype)
    got = ttnn.experimental.deepseek_prefill.dispatch_tilize(
        x, counts_tt, output_dtype=out_dtype, experts_per_chip=experts_per_chip
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


# num_experts > experts_per_chip => >1 chip-group, so the reader's cross-chip max reduction is exercised
# (the production 128/4=32-group path). Fullest chip is deliberately NOT group 0 so a "sum instead of max"
# or wrong-stride regression is caught. valid_rows must == the fullest chip's aligned-count sum.
@pytest.mark.parametrize(
    "counts,experts_per_chip,label",
    [
        ([300, 100, 50, 20, 600, 10, 0, 0], 2, "mid_chip_fullest"),  # chips: 448,96,640,0 -> 640 (chip 2)
        ([10, 10, 10, 10], 2, "two_equal_chips"),  # chips: 64,64 -> 64
        ([0] * 124 + [400, 400, 400, 400], 4, "prod_shape_last_chip"),  # 32 chips of 4; last = 4*416 = 1664
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
@pytest.mark.parametrize(
    "in_dtype", [ttnn.bfloat16, pytest.param(ttnn.fp8_e4m3, id="fp8")], ids=lambda d: str(d).split(".")[-1]
)
def test_dispatch_tilize_skip_multichip(device, counts, experts_per_chip, label, in_dtype):
    """Stage 2, multi-chip: cross-chip max over experts_per_chip groups == to_layout over [0:valid_rows]."""
    if in_dtype == ttnn.fp8_e4m3 and device.arch() != ttnn.Arch.BLACKHOLE:
        pytest.skip("fp8_e4m3 is Blackhole-only")
    rows, emb, out_dtype = 2048, 3072, ttnn.bfloat8_b
    torch.manual_seed(0)
    x = _rand_rm(rows, emb, torch.bfloat16, device, in_dtype)
    counts_tt, valid_rows = _build_routing(counts, experts_per_chip, device)
    assert 0 < valid_rows <= rows, f"[{label}] valid_rows {valid_rows} out of range"

    ref = ttnn.to_layout(x, ttnn.TILE_LAYOUT, dtype=out_dtype)
    got = ttnn.experimental.deepseek_prefill.dispatch_tilize(
        x, counts_tt, output_dtype=out_dtype, experts_per_chip=experts_per_chip
    )
    ref_v = ttnn.to_torch(ref).float()[:valid_rows]
    got_v = ttnn.to_torch(got).float()[:valid_rows]
    assert torch.equal(got_v, ref_v), (
        f"[{label}] rows[0:{valid_rows}] not byte-identical: max|Δ|={(got_v - ref_v).abs().max().item()}, "
        f"mismatched={int((got_v != ref_v).sum())}/{got_v.numel()}"
    )


def test_dispatch_tilize_output_dtype_default(device):
    """Omitting output_dtype defaults to the input dtype (bf16 in -> bf16 out), matching to_layout's default."""
    rows, emb = 1024, 3072
    torch.manual_seed(0)
    x = _rand_rm(rows, emb, torch.bfloat16, device, ttnn.bfloat16)
    ref = ttnn.to_layout(x, ttnn.TILE_LAYOUT)  # default dtype = input's (bf16)
    got = ttnn.experimental.deepseek_prefill.dispatch_tilize(x)  # default output_dtype = input's (bf16)
    ref_t, got_t = ttnn.to_torch(ref).float(), ttnn.to_torch(got).float()
    assert got_t.shape == ref_t.shape and torch.equal(got_t, ref_t)


def test_dispatch_tilize_rejects_indivisible(device, expect_error):
    """num_experts not divisible by experts_per_chip is rejected at validate."""
    x = _rand_rm(512, 1536, torch.bfloat16, device, ttnn.bfloat16)
    counts_tt = ttnn.from_torch(  # width 6, not divisible by experts_per_chip=4
        torch.tensor([64] * 6, dtype=torch.int64).reshape(1, 6).to(torch.int32),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(RuntimeError, "divisible by experts_per_chip"):
        ttnn.experimental.deepseek_prefill.dispatch_tilize(
            x, counts_tt, output_dtype=ttnn.bfloat8_b, experts_per_chip=4
        )
