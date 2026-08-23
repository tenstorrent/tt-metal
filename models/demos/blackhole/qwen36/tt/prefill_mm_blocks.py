# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure block/CB arithmetic for the tuned 2D prefill matmuls (no ttnn import).

Split out of tp_common so the numbers are host-checkable without a device or a ttnn build:
the CB budget of every T-scaled prefill matmul is asserted on CPU in
tests/test_prefill_mm_cb_budget.py against the exact arithmetic tp_common feeds into
ttnn.MatmulMultiCoreReuseMultiCastProgramConfig.

Why this exists (HP3): the 2D mcast matmul factory sizes its scaled circular buffers on
``out_block_h``/``out_block_w`` — NOT on per_core_M — with per_core_M/out_block_h iterated
as multiple output blocks per core:

    in0 CB   = 2 * out_block_h * in0_block_w   tiles   (double-buffered)
    in1 CB   = 2 * out_block_w * in0_block_w   tiles   (double-buffered)
    out CB   =     out_block_h * out_block_w   tiles
    interm CB=     out_block_h * out_block_w   tiles   (fp32 when fp32 dest-acc / packer L1 acc)

(matmul_multicore_reuse_mcast_2d_program_factory.cpp; out_block_h defaults to per_core_M when
not set). per_core_M = ceil(M/32/grid_rows) grows with the prefill chunk, so configs tuned by
the S=2048 sweep overflow L1 at chunk 4096 (measured: MLP down-proj 1,653,248 B > 1,572,864 on
11x10). ``capped_out_block_h`` bounds the CBs at their sweep-validated level while keeping the
tuned grid; M <= PREFILL_MM_SWEEP_M yields out_block_h == per_core_M, i.e. the exact config the
sweep validated (byte-identical default path).
"""

import math

TILE_SIZE = 32
# The prefill-tuning sweep (test_mlp_matmul_sweep_prefill) measured its winners at S=2048; CB
# footprints are bounded at this M's per-core block so larger chunks reuse the validated budget.
PREFILL_MM_SWEEP_M = 2048
# Blackhole worker L1 available to statically allocated CBs (the limit in the overflow error).
MAX_L1_CB_BYTES = 1_572_864

TILE_BYTES = {"bf16": 2048, "bfp8": 1088, "fp32": 4096}


def _find_largest_divisor(n, max_div=8):
    for d in range(max_div, 0, -1):
        if n % d == 0:
            return d
    return 1


def _get_out_subblock_w(per_core_n, out_subblock_h):
    for w in range(min(per_core_n, 4 // out_subblock_h), 0, -1):
        if per_core_n % w == 0:
            return w
    return 1


def _widest_prefill_cols(n, max_cols, subblock_slack=1):
    """Widest grid whose output subblock stays within `subblock_slack` of the best achievable.

    The TP=8 counterpart to `_best_prefill_cols`. More columns is usually a win at TP=8 (the halved
    per-device N leaves cores idle), but NOT when the extra width collapses the subblock: measured
    at S=2048, mlp_gate (N=2176 -> 68 tiles) goes cols 9 -> 11, per_core_N 8 -> 7, and 7 is prime so
    out_subblock_w drops 4 -> 1 -- a 2058us -> 2118us REGRESSION, i.e. the subblock-first ranking
    was right for that shape. Guarding on the subblock keeps the wide grid exactly where it pays:

        matmul     default        this rule       measured
        attn_wo    c10_bw2_sw4    c11_bw4_sw3     803.5 -> 718.7us
        gdn_out    c10_bw2_sw4    c11_bw4_sw3     802.3 -> 719.9us
        mlp_down   c10_bw4_sw4    c11_bw4_sw3    1787.4 -> 1724.9us
        mlp_gate   c9_bw4_sw4     c9_bw4_sw4     2058.1us (unchanged -- already optimal)
    """
    n_tiles = math.ceil(n / TILE_SIZE)
    sw = {cols: _get_out_subblock_w(math.ceil(n_tiles / cols), 1) for cols in range(1, max_cols + 1)}
    floor = max(sw.values()) - subblock_slack
    return max((cols for cols, w in sw.items() if w >= floor), default=1)


def _best_prefill_cols(n, max_cols):
    """Grid width (<=max_cols) maximizing the output subblock, tie-broken to more cores — avoids the
    1x1-subblock stall (e.g. gate/up N=4352 -> 7-wide -> 1x4) the default full width can force."""
    n_tiles = math.ceil(n / TILE_SIZE)
    best_cols, best_key = 1, None
    for cols in range(1, max_cols + 1):
        sw = _get_out_subblock_w(math.ceil(n_tiles / cols), 1)
        key = (sw, cols)  # prefer wider subblock, then more columns (more compute cores)
        if best_key is None or key > best_key:
            best_key, best_cols = key, cols
    return best_cols


def capped_out_block_h(per_core_M, grid_rows, sweep_m=PREFILL_MM_SWEEP_M):
    """out_block_h bounding the scaled CBs at their M=sweep_m level (validate: per_core_M %
    out_block_h == 0, so take the largest divisor of per_core_M <= the sweep-M per-core block).

    per_core_M <= the sweep baseline returns per_core_M itself — identical to the factory
    default (out_block_h = per_core_M), so sweep-validated chunks keep their exact config.
    A prime oversized per_core_M (e.g. 13 at M=4096 on 10 rows) degenerates to out_block_h=1:
    correct, CB-minimal, at the cost of re-streaming in1 per M-block — watch the op's device
    time and pad per_core_M to a composite if it shows."""
    baseline = max(1, math.ceil(sweep_m / TILE_SIZE / grid_rows))
    if per_core_M <= baseline:
        return per_core_M
    return _find_largest_divisor(per_core_M, baseline)


def prefill_mm_blocks(m, k, n, grid_cols, grid_rows, in0_block_w_divisor, in0_block_w_cap):
    """The full block set create_prefill_matmul_program_config feeds into the 2D mcast config.

    Mirrored nowhere — tp_common builds the ttnn config FROM this dict, and the CB-budget host
    test asserts on it, so the numbers cannot drift."""
    per_core_M = max(1, math.ceil(m / TILE_SIZE / grid_rows))
    per_core_N = max(1, math.ceil(n / TILE_SIZE / grid_cols))
    out_subblock_h = 1
    out_subblock_w = _get_out_subblock_w(per_core_N, out_subblock_h)
    k_tiles = math.ceil(k / TILE_SIZE)
    if in0_block_w_divisor:
        # in0_block_w only has to divide k_tiles (no K tail in the 2D mcast kernel), so take the
        # largest legal block rather than scaling with grid width -- see _PREFILL_TUNING.
        in0_block_w = _find_largest_divisor(k_tiles, in0_block_w_cap)
    else:
        in0_block_w = min(in0_block_w_cap, max(1, k_tiles // grid_cols))
    return dict(
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        in0_block_w=in0_block_w,
        out_block_h=capped_out_block_h(per_core_M, grid_rows),
        out_block_w=per_core_N,
    )


def mm2d_scaled_cb_bytes(blocks, in0="bf16", in1="bfp8", out="bf16", interm="fp32"):
    """The T-scaling CB footprint of the 2D mcast matmul factory for a block dict (see module
    docstring for the model). Excludes the small constant CBs, so treat it as a lower bound —
    the budget test asserts it stays at the sweep-validated level, not merely under the cap."""
    obh, obw, ibw = blocks["out_block_h"], blocks["out_block_w"], blocks["in0_block_w"]
    t = TILE_BYTES
    return 2 * obh * ibw * t[in0] + 2 * obw * ibw * t[in1] + obh * obw * (t[out] + t[interm])
