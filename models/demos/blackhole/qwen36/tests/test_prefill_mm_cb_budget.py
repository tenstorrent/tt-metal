# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only CB-budget gate for the T-scaled prefill matmuls (HP3, chunk 2048 -> 4096/8192).

The tuned 2D prefill matmul configs were swept at S=2048; their circular buffers scale with
out_block_h (defaulting to per_core_M = ceil(M/32/grid_rows)), so raising the prefill chunk
overflowed L1 at program creation (measured at chunk 4096: MLP down-proj CBs 1,653,248 B >
1,572,864 on the 11x10 grid). tp_common now caps out_block_h at the sweep-M level via
prefill_mm_blocks.capped_out_block_h; this test asserts, in pure host math against the exact
block arithmetic tp_common uses (prefill_mm_blocks is the single source — tp_common builds its
ttnn configs FROM it), that for every production shape and M in {2048, 4096, 8192}:

  * the scaled-CB footprint never exceeds its own M=2048 baseline (the validated budget),
  * the matmul validate() divisibility constraints hold,
  * M <= 2048 emits out_block_h == per_core_M — the exact pre-cap config (default path unchanged),
  * and (sanity) the UNCAPPED footprint at M=4096 does exceed the baseline, i.e. this test
    would have caught the chunk-4096 overflow.

No ttnn / torch needed:
    python -m pytest models/demos/blackhole/qwen36/tests/test_prefill_mm_cb_budget.py --noconftest -q
"""

import math

import pytest

from models.demos.blackhole.qwen36.tt.prefill_mm_blocks import (
    MAX_L1_CB_BYTES,
    PREFILL_MM_SWEEP_M,
    TILE_SIZE,
    _widest_prefill_cols,
    capped_out_block_h,
    mm2d_scaled_cb_bytes,
    prefill_mm_blocks,
)

# Qwen3.8-27B @ TP=8 (dims verified from the HF config: hidden 5120, MLP intermediate 17408,
# attn 24Q/4KV @ head_dim 256, GDN 16K/48V heads @ 128 -> value_dim 6144).
DIM = 5120
HIDDEN_TP = 17408 // 8  # 2176
ATTN_OUT_TP = 24 * 256 // 8  # 768
GDN_VALUE_TP = 6144 // 8  # 768
GRID_ROWS = 10  # prefill_grid_default()[1] on BH
MAX_COLS = 11  # PREFILL_MAX_COLS_PORTABLE == decode_grid_w on BH P150
TP8 = dict(in0_block_w_divisor=True, in0_block_w_cap=4)  # _PREFILL_TUNING[8]

# Every user of the tuned 2D builder (create_prefill_mlp_matmul_program_config /
# args.prefill_progcfg) in the per-chunk prefill path, plus the arms idle at TP=8 but sharing
# the same builder. (in0/in1/out dtypes: bf16 activations, bfp8 weights, bf16 out, fp32 interm.)
TUNED_2D_SHAPES = {
    "mlp_down (mlp.py w2_pc)": (HIDDEN_TP, DIM),
    "attn_wo (attention/tp.py _wo_proj)": (ATTN_OUT_TP, DIM),
    "mlp_gate/up (non-AGMM arm)": (DIM, HIDDEN_TP),
    "gdn_out fallback (_row_proj non-mmrs arm)": (GDN_VALUE_TP, DIM),
}

CHUNKS = (2048, 4096, 8192)


def _tuned_blocks(m, k, n):
    cols = _widest_prefill_cols(n, max(1, min(MAX_COLS, math.ceil(n / TILE_SIZE))))
    return prefill_mm_blocks(m, k, n, cols, GRID_ROWS, TP8["in0_block_w_divisor"], TP8["in0_block_w_cap"])


def _check_validate_constraints(b):
    assert b["out_block_h"] % b["out_subblock_h"] == 0
    assert b["out_block_w"] % b["out_subblock_w"] == 0
    assert b["per_core_M"] % b["out_block_h"] == 0
    assert b["per_core_N"] % b["out_block_w"] == 0


@pytest.mark.parametrize("name", sorted(TUNED_2D_SHAPES))
def test_tuned_2d_prefill_cb_budget(name):
    k, n = TUNED_2D_SHAPES[name]
    baseline = mm2d_scaled_cb_bytes(_tuned_blocks(PREFILL_MM_SWEEP_M, k, n))
    assert baseline < MAX_L1_CB_BYTES, f"{name}: sweep baseline itself over L1?"
    for m in CHUNKS:
        b = _tuned_blocks(m, k, n)
        _check_validate_constraints(b)
        got = mm2d_scaled_cb_bytes(b)
        assert got <= baseline, f"{name} @M={m}: scaled CBs {got} exceed the M=2048 baseline {baseline}"
        if m <= PREFILL_MM_SWEEP_M:
            assert b["out_block_h"] == b["per_core_M"], f"{name} @M={m}: sweep-validated config must be unchanged"


def test_uncapped_mlp_down_would_overflow_at_4096():
    """The regression this gate exists for: uncapped (out_block_h = per_core_M) mlp_down at
    M=4096 blows past its sweep budget — 1,541,632 B in this model, 1,653,248 B measured on
    silicon once the factory's constant CBs are added (job 55142, 11x10 grid, 5.1% over L1)."""
    k, n = TUNED_2D_SHAPES["mlp_down (mlp.py w2_pc)"]
    baseline = mm2d_scaled_cb_bytes(_tuned_blocks(PREFILL_MM_SWEEP_M, k, n))
    b = _tuned_blocks(4096, k, n)
    b_uncapped = dict(b, out_block_h=b["per_core_M"])
    uncapped = mm2d_scaled_cb_bytes(b_uncapped)
    assert uncapped > baseline, "expected the uncapped 4096 config to exceed the sweep budget"
    assert uncapped > 1_500_000, "model drifted: uncapped 4096 mlp_down should be ~1.54 MB of scaled CBs"
    assert mm2d_scaled_cb_bytes(b) <= baseline


def _mmrs_blocks(m, n=DIM, k_local=GDN_VALUE_TP, grid=(8, 8)):
    """Mirror of tp_common.matmul_reduce_scatter_prefill's program config arithmetic (the one
    T-scaled matmul not built through prefill_mm_blocks). Keep in lockstep with tp_common."""
    per_core_N = max(1, math.ceil(n / TILE_SIZE / grid[0]))
    per_core_M = max(1, math.ceil(m / TILE_SIZE / grid[1]))
    return dict(
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=1,
        in0_block_w=min(4, max(1, k_local // TILE_SIZE // grid[0])),
        out_block_h=capped_out_block_h(per_core_M, grid[1]),
        out_block_w=max(1, per_core_N // 2),
    )


def test_mmrs_gdn_out_cb_budget():
    """GDN out-proj (matmul_reduce_scatter_prefill, 8x8 grid, fp32 out+interm): per_core_M=32 at
    chunk 8192 means ~2.6 MB of out+interm CB uncapped — the next wall after mlp_down."""
    baseline = mm2d_scaled_cb_bytes(_mmrs_blocks(PREFILL_MM_SWEEP_M), out="fp32")
    assert baseline < MAX_L1_CB_BYTES
    for m in CHUNKS:
        b = _mmrs_blocks(m)
        _check_validate_constraints(b)
        got = mm2d_scaled_cb_bytes(b, out="fp32")
        assert got <= baseline, f"gdn_out mmrs @M={m}: scaled CBs {got} exceed the M=2048 baseline {baseline}"
        if m <= PREFILL_MM_SWEEP_M:
            assert b["out_block_h"] == b["per_core_M"]
    uncapped_8192 = mm2d_scaled_cb_bytes(dict(_mmrs_blocks(8192), out_block_h=32), out="fp32")
    assert uncapped_8192 > MAX_L1_CB_BYTES, "expected the uncapped 8192 mmrs config to overflow outright"


def test_capped_out_block_h_properties():
    """Cap invariants across the range the chunk ladder can produce."""
    for rows in (8, 10):
        baseline = max(1, math.ceil(PREFILL_MM_SWEEP_M / TILE_SIZE / rows))
        for m_tiles_rows in range(1, 65):  # per_core_M for M up to ~20k on 10 rows
            obh = capped_out_block_h(m_tiles_rows, rows)
            assert 1 <= obh <= max(baseline, m_tiles_rows)
            assert m_tiles_rows % obh == 0, "validate(): per_core_M % out_block_h must be 0"
            if m_tiles_rows <= baseline:
                assert obh == m_tiles_rows, "at or below the sweep M the config must not change"
            else:
                assert obh <= baseline, "above the sweep M the CB-scaling block must not exceed baseline"
