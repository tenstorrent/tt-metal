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
    PREFILL_L1_ACT_MAX_M,
    PREFILL_MM_SWEEP_M,
    TILE_SIZE,
    _widest_prefill_cols,
    act_bytes_per_bank,
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


def test_prefill_l1_activation_gate():
    """HP3 option b: chunk-scaled prefill activations (w2/wo partials, GDN qkvzab + qkv slices,
    attn qkv3) keep their validated L1 placement at chunk <= 4096 and move to DRAM above it —
    at chunk 8192 the L1 buffers collide with program CBs (measured: 'L1 buffer allocated at
    335616, CB region ends at 443392' on the 8x9 AGMM grid; chunk 4096 ran the 100k bench clean
    at 45.13 s). The footprint model shows why 4096 fits and 8192 cannot:

        biggest single activation = the [M, dim] w2/wo partial (bf16):
          M=2048 ~191 KB/bank, M=4096 ~381 KB/bank, M=8192 ~761 KB/bank
        vs ~1.37 MB of bank L1 shared with up to ~0.9 MB of program CBs (sweep level).
    """
    from models.demos.blackhole.qwen36.tt.prefill_mm_blocks import prefill_act_in_l1

    for m in (32, 128, 1024, 2048, 4096):
        assert prefill_act_in_l1(m), f"chunk {m} must keep the validated L1 placement"
    for m in (4096 + 32, 8192, 16384):
        assert not prefill_act_in_l1(m), f"chunk {m} must spill chunk-scaled activations to DRAM"
    assert PREFILL_L1_ACT_MAX_M == 4096

    # Evidence the 4096/8192 split is the right boundary for the dominant tensor (w2/wo partial
    # [M, 5120] bf16): with the sweep-level ~0.9 MB CB budget resident, 8192's ~761 KB/bank
    # cannot coexist (1.37 MB bank), 4096's ~381 KB/bank can (as silicon confirmed).
    bank_l1 = 1_400_000  # ~1.37 MB usable per bank
    sweep_cb = 890_368  # the tuned 2D matmul budget at S=2048 (see test_tuned_2d_prefill_cb_budget)
    per_bank = {m: act_bytes_per_bank(m, DIM) for m in (2048, 4096, 8192)}
    assert per_bank[4096] + sweep_cb < bank_l1, "chunk 4096 w2/wo partial must fit beside sweep-level CBs"
    assert per_bank[8192] + sweep_cb > bank_l1, "chunk 8192 w2/wo partial cannot fit beside sweep-level CBs"


def _agmm_cb_bytes(m_block, k_block, n_block, in0="bf16", in1="bfp8", out="bf16", interm="fp32"):
    """Scaled-CB model of all_gather_minimal_matmul_async / minimal_matmul (program factory:
    in0/in1/out double-buffered on the M/K/N block sizes, interm single-buffered) — every term
    is a fixed MinimalMatmulConfig block, so the total is chunk-length-INDEPENDENT by
    construction. Keep in lockstep with tp_common.all_gather_matmul_prefill's config."""
    from models.demos.blackhole.qwen36.tt.prefill_mm_blocks import TILE_BYTES as T

    return (
        2 * m_block * k_block * T[in0]
        + 2 * k_block * n_block * T[in1]
        + 2 * m_block * n_block * T[out]
        + m_block * n_block * T[interm]
    )


def test_agmm_l1_budget():
    """The 8192 wall the fleet hit was NOT AGMM CB growth (frames: gdn _project_qkvzab ->
    all_gather_matmul_prefill, 8x9 grid): the CBs are block-fixed (~397 KB + AG packet CBs =
    the observed 443,392 B region, chunk-independent). The colliding L1 *buffer* is the op's
    activation-gather intermediate [M, dim] — placed by out_memory_config (device op
    create_output_specs slot 0) — plus the L1 qkvzab output. This asserts the arithmetic:
    fixed CBs, 4096-with-L1 fits a bank (silicon-validated), 8192-with-L1 cannot (the measured
    clash), and the prefill_act_in_l1 gate spills exactly the 8192 case to DRAM."""
    from models.demos.blackhole.qwen36.tt.prefill_mm_blocks import prefill_act_in_l1

    # tp_common.all_gather_matmul_prefill config at TP=8: M_block=4, K_block=agmm_k_block_size
    # (K_local=640 -> 20 tiles -> 4), N_block=8.
    cb = _agmm_cb_bytes(4, 4, 8)
    assert cb == 397_312, "AGMM scaled-CB model drifted"
    assert cb < 443_392 < MAX_L1_CB_BYTES, "fixed CBs must sit within the observed CB region"

    bank_l1 = 1_400_000  # ~1.37 MB usable per bank (same budget as the activation-gate test)
    qkvzab_w = 2064  # gdn_qkvz_dim_tp + tile-padded a|b block
    for m in (2048, 4096, 8192):
        gather_interm = act_bytes_per_bank(m, DIM)  # [M, dim] gathered activation
        qkvzab_out = act_bytes_per_bank(m, qkvzab_w)
        l1_resident = gather_interm + qkvzab_out + 443_392
        if prefill_act_in_l1(m):
            assert l1_resident < bank_l1, f"chunk {m} must fit with the validated L1 placement ({l1_resident})"
        else:
            assert l1_resident > bank_l1, f"chunk {m} spills to DRAM for a reason ({l1_resident})"
