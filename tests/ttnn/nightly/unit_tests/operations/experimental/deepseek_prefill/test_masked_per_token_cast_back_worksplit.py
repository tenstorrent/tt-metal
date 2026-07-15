# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Work-distribution tests for ttnn.experimental.deepseek_prefill.masked_per_token_cast_back.

These tests build *fake* per-expert token counts + a synthetic FP8 dispatch buffer and run the
masked dequantization op. The reader kernel computes, per core and on-device, the valid prefix
length (total_valid_rows) and this core's balanced slice over the FLATTENED compute-block space
(rows x col-blocks), then DPRINTs it:

    [masked_decomp] core=i/N total_valid_rows=.. total_cblocks=.. cb_start=.. cb_end=.. start_row=.. start_col=.. num_blocks=..

To see that output, run with the DPRINT server watching all cores:

    export TT_METAL_DPRINT_CORES=all
    pytest tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/\
test_masked_per_token_cast_back_worksplit.py -s

The test also prints the *expected* per-core split (mirroring the kernel formula) so it can be
diffed against the DPRINT lines.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import comp_pcc

pytestmark = pytest.mark.use_module_device

BLOCK_W = 128
TILE = 32
E4M3_MAX = 448.0


# Blackhole only operation (FP8_E4M3 path).
@pytest.fixture(autouse=True)
def _require_blackhole():
    if not is_blackhole():
        pytest.skip("FP8_E4M3 path requires Blackhole")


# ---------------------------------------------------------------------------
# Fake-count test cases: (label, per-expert token counts).
# experts_per_chip == len(counts); an identity global_expert_idx_table is used so
# region_offsets[g] = cumulative ceil_tile(counts) and total_valid_rows = sum(ceil_tile).
# ---------------------------------------------------------------------------
CASES = [
    # Small prefixes: fewer valid tile-rows than cores. With the flattened compute-block split the
    # work spreads across many more cores than there are tile-rows (each tile-row = blocks_per_row
    # compute-blocks), so utilization stays high instead of collapsing to a handful of cores.
    ("uniform_4x64", [64, 64, 64, 64]),
    ("irregular_8", [130, 74, 200, 12, 96, 41, 160, 33]),
    ("tiny_single_row", [1, 0, 0, 0]),
    # Very small prefixes (1-3 tile-rows): under the OLD tile-row split these used 1-3 cores; the
    # flattened split now uses min(tile_rows * blocks_per_row, num_cores) cores.
    ("one_tile_row", [32, 0, 0, 0]),  # 1 tile-row -> 56 compute-blocks -> 56 cores
    ("three_tile_rows", [96, 0, 0, 0]),  # 3 tile-rows -> 168 compute-blocks -> all cores
    # Large prefixes: more compute-blocks than cores -> balanced +/-1 blocks per core.
    ("one_dominant_over_grid", [4000, 96, 96, 96]),
    ("near_full_capacity", [8000, 64, 64, 64]),
]

CAPACITY = 8192  # dispatch-buffer row capacity (M); valid prefix is a head of this
H = 7168  # emb_dim; H / BLOCK_W = 56 scale blocks per row (realistic DeepSeek dim)


def _ceil_tile(n):
    return ((n + TILE - 1) // TILE) * TILE


def _expected_total_valid_rows(counts):
    return sum(_ceil_tile(c) for c in counts)


def _expected_split(total_valid_rows, num_cores, width):
    """Mirror the reader kernel's balanced split over the flattened compute-block space."""
    blocks_per_row = width // BLOCK_W
    total_tile_rows = total_valid_rows // TILE
    total_compute_blocks = total_tile_rows * blocks_per_row
    out = []
    for i in range(num_cores):
        cb_start = (total_compute_blocks * i) // num_cores
        cb_end = (total_compute_blocks * (i + 1)) // num_cores
        num_blocks = cb_end - cb_start
        start_flat = cb_start * TILE
        start_row = start_flat // blocks_per_row
        start_col = start_flat % blocks_per_row
        out.append((i, start_row, start_col, num_blocks))
    return out


def _expected_active_cores(total_valid_rows, num_cores, width):
    """Number of cores that receive >=1 compute-block = min(total_compute_blocks, num_cores)."""
    total_compute_blocks = (total_valid_rows // TILE) * (width // BLOCK_W)
    return min(total_compute_blocks, num_cores)


def _make_e4m3(device, m, width):
    torch.manual_seed(0)
    x = (torch.randn(m, width) * 3.0).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    tt = ttnn.from_torch(
        x.float(),
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return x, tt


def _make_scale(device, m, width):
    torch.manual_seed(1)
    scale = (torch.rand(m, width // BLOCK_W) * 2.0).to(torch.float32)
    tt = ttnn.from_torch(
        scale,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return scale, tt


def _make_u32(device, values):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def assert_quality(result, ref, *, pcc_threshold, rtol, atol, label=""):
    """PCC + allclose gate (mirrors test_deepseek_prefill_per_token_cast.assert_quality)."""
    passed_pcc, pcc = comp_pcc(result, ref, pcc_threshold)
    close = torch.allclose(result, ref, rtol=rtol, atol=atol)
    status = "PASS" if passed_pcc and close else "FAIL"
    logger.info(f"[{status}] {label}: pcc={pcc:.6f} (min={pcc_threshold}), allclose={close} (rtol={rtol}, atol={atol})")
    assert passed_pcc, f"{label}: PCC {pcc:.6f} below {pcc_threshold}"
    assert close, f"{label}: values not allclose (rtol={rtol}, atol={atol})"


@pytest.mark.parametrize("bf16_scale", [False, True])
@pytest.mark.parametrize("label, counts", CASES, ids=[c[0] for c in CASES])
def test_masked_decompress_worksplit(device, label, counts, bf16_scale):
    experts_per_chip = len(counts)

    # region offsets = exclusive cumsum of tile-aligned counts (dispatch packs from row 0).
    region = []
    acc = 0
    for c in counts:
        region.append(acc)
        acc += _ceil_tile(c)
    total_valid_rows = acc
    assert total_valid_rows <= CAPACITY, f"{label}: prefix {total_valid_rows} > capacity {CAPACITY}"

    x_e4m3, e4m3_tt = _make_e4m3(device, CAPACITY, H)
    scale, scale_tt = _make_scale(device, CAPACITY, H)
    region_tt = _make_u32(device, region)
    counts_tt = _make_u32(device, counts)
    table_tt = _make_u32(device, list(range(experts_per_chip)))  # identity: local slot -> global id

    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y

    # Expected per-core split (what the DPRINT lines should match).
    split = _expected_split(total_valid_rows, num_cores, H)
    active = [s for s in split if s[3] > 0]
    blocks = [s[3] for s in split]
    expected_active = _expected_active_cores(total_valid_rows, num_cores, H)
    logger.info(
        f"\n===== CASE {label} =====\n"
        f"counts={counts} ceil_tile={[_ceil_tile(c) for c in counts]} region_offsets={region}\n"
        f"total_valid_rows={total_valid_rows} (tile_rows={total_valid_rows // TILE}), "
        f"total_compute_blocks={(total_valid_rows // TILE) * (H // BLOCK_W)}, "
        f"capacity={CAPACITY}, num_cores={num_cores}, blocks_per_row={H // BLOCK_W}\n"
        f"active_cores={len(active)}/{num_cores} (expected {expected_active}), "
        f"blocks per active core: min={min(blocks) if active else 0} max={max(blocks)}\n"
        f"first active cores [core: start_row, start_col, num_blocks]:\n"
        + "\n".join(f"  core {i}: start_row={sr} start_col={sc} blocks={nb}" for (i, sr, sc, nb) in active[:12])
    )

    # Utilization: the flattened split should spread work across min(total_compute_blocks, num_cores)
    # cores -- far more than the tile-row count for small prefixes -- and stay perfectly balanced (+/-1).
    assert len(active) == expected_active, f"{label}: active {len(active)} != expected {expected_active}"
    if active:
        assert max(blocks) - min(blocks) <= 1, f"{label}: load imbalance {min(blocks)}..{max(blocks)}"

    out_tt = ttnn.experimental.deepseek_prefill.masked_per_token_cast_back(
        e4m3_tt,
        scale_tt,
        region_tt,
        counts_tt,
        table_tt,
        experts_per_chip=experts_per_chip,
        output_dtype=ttnn.bfloat16,
        bf16_scale=bf16_scale,
    )
    out = ttnn.to_torch(out_tt).float()

    assert tuple(out_tt.shape) == (CAPACITY, H)

    # Functional check on the valid prefix only (garbage tail is intentionally left untouched).
    # bf16_scale narrows the scale on-device; match the golden to what the op multiplies by.
    golden_scale = scale.to(torch.bfloat16).float() if bf16_scale else scale
    golden = x_e4m3.float() * golden_scale.repeat_interleave(BLOCK_W, dim=-1)
    golden = golden.to(torch.bfloat16).float()
    prefix_out = out[:total_valid_rows]
    prefix_golden = golden[:total_valid_rows]
    normal = x_e4m3.float()[:total_valid_rows].abs() > 2.0**-6
    max_abs_err = (prefix_out[normal] - prefix_golden[normal]).abs().max().item()
    logger.info(f"[{label}] valid-prefix max_abs_err (normal e4m3 values) = {max_abs_err:.4f}")
    # bf16 output + bf16-narrowed scale round a few values one bf16 ULP past a 1e-3 band; PCC stays the
    # quality gate, so loosen the absolute tolerance for the bf16-scale path only.
    atol = 1e-2 if bf16_scale else 1e-3
    assert_quality(
        prefix_out[normal],
        prefix_golden[normal],
        pcc_threshold=0.999,
        rtol=1e-2,
        atol=atol,
        label=f"masked dequant {label} bf16_scale={bf16_scale}",
    )
