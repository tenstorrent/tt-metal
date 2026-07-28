# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""FORK of rms_norm_program_descriptor.py for the `gamma_broadcast_rowsplit` perf
experiment. DO NOT graduate this file; it exists only so the gamma-delivery path can
be A/B'd on device against the op's current approach without touching the op.

Delta vs the op (everything else is byte-identical):

  * ``GAMMA_MODE`` — "dram" (the op's current per-core TensorAccessor read, the
    honest BASELINE), "ablate" (the gamma NoC reads removed, reserve/push kept —
    a TIMING UPPER BOUND on any gamma-traffic optimization; output is wrong by
    design) or "mcast" (the CANDIDATE: one injector core per virtually-contiguous
    column run reads gamma from DRAM and multicasts it to its run).
  * ``FORCE_GAMMA_RESIDENT`` — override the op's residency predicate, so the
    "make gamma resident" half of the mcast change can be separated from the
    "broadcast it" half on the geometries where the op streams gamma.
  * kernels point at this experiment's forked ``kernels/gbr_*.cpp``.

Original docstring follows.
---------------------------------------------------------------------------
rms_norm ProgramDescriptor — the Blocking Model of op_design.md §1 made concrete.

Everything below derives from ONE knob, ``L1_BLOCK_BUDGET_BYTES``:

    BLOCK_CB_UNITS    = CB pages consumed per block tile      (per layout/gamma cell)
    TILE_BLOCK_BUDGET = L1_BLOCK_BUDGET_BYTES // (BLOCK_CB_UNITS * unit_bytes)
    WT_CHUNK          = coarsest chunk of the reduced W axis that fits the budget
    NW                = Wt // WT_CHUNK                        (chunks per row)
    HT_BLOCK          = tile-rows per compute block           (1 when NW > 1)

Every CB page count, every kernel loop trip count and the core grid are
functions of those parameters — never of a whole-op dimension (Wt, W, H) and
never a magic literal.  The two ``Wt``-sized CBs (``cb_input_tiles`` when
``X_RESIDENT``, ``cb_gamma`` when ``GAMMA_RESIDENT``) are predicate-guarded
residents with a bounded streaming fallback (§1.3).

Work distribution (§5): the *independent* tile-row axis is spread over the
whole compute grid via ``ttnn.split_work_to_cores(..., row_wise=True)``; each
core owns whole rows and reduces the *dependent* W axis sequentially in-core.

When that alone under-fills the grid — or the data is WIDTH/BLOCK sharded, which
pins the split — the *dependent* W axis is ALSO split across cores (§4.2, Lamp
L1): each core reduces its slice to a raw partial, a group root combines them
and multicasts ``1/rms`` back. ``_Placement`` owns that topology; ``_Blocking``
is unchanged by it, because the per-core W extent is simply the axis it derives
against. Measured on the decode profiles: 1 busy core -> 32-56, 1.75-5.11x.

A HEIGHT_SHARDED input (Lamp L3) is the row split above made *physical*: the
shard grid pins the core assignment and the shard height pins the per-core row
count, each core still owns whole rows, and the reduce therefore stays entirely
local (``cw == 1``, no combine). The only thing that changes is CB *placement* —
``cb_input_tiles`` / ``cb_output_tiles`` are backed straight on the resident L1
shards, so the reader issues no input read and the writer no output write.

--------------------------------------------------------------------------
Chunk-uniformity invariant (implementation constraint, documented deviation)
--------------------------------------------------------------------------
op_design.md §1.2 sets ``WT_CHUNK = min(Wt, TILE_BLOCK_BUDGET)`` and allows a
short last chunk (``WT_LAST``).  A short last chunk is NOT expressible here:

  * a circular buffer's write/read pointer may never straddle ``fifo_limit``
    (dataflow_api.h:216-222) — with a mixed push sequence
    ``[C, C, ..., L]`` on a CB of capacity ``k*C`` the pointer lands
    mid-buffer and the next full push runs past the limit; and
  * ``cb_input_rm`` / ``cb_output_rm`` page sizes ARE the chunk width, so the
    tilize/untilize LLK row stride and
    ``write_sticks_after_untilize``'s ``round_up(row_bytes, tile_row_bytes)``
    pop count cannot both be satisfied by two different chunk widths.

So ``WT_CHUNK`` is picked as the **coarsest divisor of ``Wt`` that fits the
budget**, making ``WT_LAST == WT_CHUNK`` by construction.  ``WT_LAST`` stays a
separate emitted knob so the general case remains expressible.  The knob is
still "coarsest that fits L1" for every ``Wt`` with a divisor near the budget;
only a prime ``Wt`` larger than the budget degrades (correct, just more
chunks) — a follow-up could recover it with a padded tail on the TILE path.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

# ---------------------------------------------------------------------------
# EXPERIMENT knobs (this fork only)
# ---------------------------------------------------------------------------
#
# GAMMA_MODE:
#   "dram"   the op's current approach — every core reads its own gamma slice
#            through a TensorAccessor. In the row-split regime (cw == 1) that
#            slice is the WHOLE of gamma on every core, so up to 110 cores fetch
#            the same Wt pages.  BASELINE.
#   "ablate" the gamma NoC reads are deleted and the reserve/push scaffolding
#            kept, so the CB flow-control, the loop trip counts and every other
#            byte of DRAM traffic are unchanged. The output is WRONG by design;
#            the number is the UPPER BOUND on any gamma-traffic optimization.
#   "mcast"  CANDIDATE — one injector core per virtually-contiguous column run
#            reads gamma from DRAM and multicasts it over that run's rectangle.
GAMMA_MODE = "dram"

# Schedule knob for the RESIDENT gamma strip (independent of GAMMA_MODE, so the
# schedule change and the broadcast can be A/B'd separately):
#   False  the op's position — a prologue before the first input read.
#   True   after the first row-block's pass-A read, so the transfer overlaps that
#          read. Compute needs gamma only in pass B, so the slack is real.
GAMMA_LATE = False

# One injector for the WHOLE grid (it broadcasts to every run's rectangle in turn)
# vs one injector PER virtually-contiguous run (the default). Per-run injectors cost
# one extra DRAM read of gamma but overlap the two injections on two cores; a single
# injector reads gamma once and serializes both broadcasts on one core's NoC.
GAMMA_ONE_INJECTOR = False

# None -> use the op's own residency predicate. True/False force it (True only
# takes effect when the resident plan still fits the L1 wall).
FORCE_GAMMA_RESIDENT = None

# Semaphore ids for the gamma broadcast: one PAIR per family, starting here
# (family f -> data_ready = base + 2f, consumer_ready = base + 2f + 1). 0..5 are
# taken by the op's combine (SEM_GATHER / SEM_MCAST_BASE..+3 / SEM_GATHER2).
#
# The families must NOT share an id pair even though their rectangles are
# disjoint. Under GAMMA_ONE_INJECTOR one core sends to both, so both families'
# receivers would ack the SAME consumer_ready cell on that core — and SenderPipe
# resets the cell to 0 after each send, wiping the other family's acks. Per-family
# ids remove the coupling for both injector modes.
GBR_SEM_BASE = 6

# Debug side-channel: what the last create_program_descriptor() actually wired, so
# the bench can report whether the broadcast ENGAGED or silently fell back to the
# per-core DRAM read (a fallback that reads as a null result otherwise).
LAST_PLAN = {}

# ---------------------------------------------------------------------------
# The knob family (§1.2). One definition each; everything else derives.
# ---------------------------------------------------------------------------

# Block factor budget: bytes of L1 a single compute block may occupy across all
# block-scaled CBs. Halved automatically if the final CB total misses the L1
# wall below (documented fallback, same single source).
L1_BLOCK_BUDGET_BYTES = 512 * 1024

# Conservative FALLBACK for the per-core L1 wall, used only when the live device
# cannot be queried (see _l1_total_budget). Worker L1 is 1.5 MB; the remainder is
# firmware/stack/kernel-args headroom.
L1_CB_BUDGET_BYTES = 1_100_000

# The real, physical wall: everything this program commits to one core's L1 bank.
#
# Refinement 4 established that a zero-copy sharded CB is ALIASED onto the
# tensor's own buffer, which the *buffer* allocator already reserved out of the
# same per-core L1 bank, and that the bank size must be read from the live device
# (1_461_504 B on blackhole_p150b) rather than guessed — charging the shard
# against a guessed budget collapsed a HEIGHT_SHARDED (1,1,32,8192) bf16 from
# WT_CHUNK 32 to 1 with 361 KB of the bank still free. It kept the old guessed
# constant as a SECOND wall on program-allocated CBs alone.
#
# Refinement 5 retires that second wall. It was a proxy for the bank, measured
# from nothing, and strictly more conservative than the quantity it proxied — so
# it could only ever cost block size, never buy safety the real wall does not
# already give. There is one condition:
#
#     program CBs + resident shards <= bank size - L1_ALLOC_HEADROOM_BYTES
#
# It is load-bearing on the interleaved prefill column, where gamma is the SAME
# bytes for every core and is re-read once per row-block when it is not resident.
# Measured on (1,1,8192,7168) bf16: gamma resident needs 1_195_648 B, which the
# guessed 1_100_000 wall refused and the real 1_330_432 one accepts. That is
# 2 of 3 gamma passes deleted — 100 MB of the shape's 386 MB of DRAM traffic.
L1_ALLOC_HEADROOM_BYTES = 128 * 1024

# ---------------------------------------------------------------------------
# Cross-core W-split knobs (Refinement 2; op_design.md Lamp L1 / §4.2)
# ---------------------------------------------------------------------------
#
# The reduced W axis is *dependent*, so splitting it across cores costs a
# partial-sum combine + a 1/rms broadcast per row-block. Three single-source
# knobs govern when that trade is taken and how wide it goes.

# Engage the W-split only when the INDEPENDENT row axis cannot even fill one
# core per grid row -- i.e. the row split alone leaves >= (grid.x - 1)/grid.x of
# the grid idle. Structural discriminator, not a tuned threshold.
W_SPLIT_MAX_HT_FOR_SPLIT = None  # None => grid.y (derived at call time)

# ... and only when the reduced axis is wide enough that a core still gets a
# real chunk after the split. 8 tiles is the measured read-batch plateau
# (examples/double_buffer): below it the combine handshake dominates the work.
W_SPLIT_MIN_WT = 8

# L1 the gather buffers may occupy per core (HT_BLOCK * gather_tiles fp32
# tiles; see _gather_tiles). Caps how wide an *interleaved* W-split goes; a
# sharded input's CW is pinned by its shard grid and is instead absorbed by the
# halve-and-re-derive loop in _derive_blocking.
L1_GATHER_BUDGET_BYTES = 256 * 1024

# Combine topology knob (Refinement 3). The widest fan-in ONE root is allowed to
# serialize before the gather is split into two stages.
#
# A flat root gather makes a single core absorb CW * 4 KB of NoC writes and then
# run CW fp32 tile-adds back to back. Measured by ablation at the pinned perf
# config (the W-split placement with the combine legs removed vs. the real op),
# the combine is 45-49% of the whole decode kernel and grows ~73 ns per extra
# contributor:
#
#     shape         cores   full ns   no-combine ns   combine ns
#     (1,1,32,1024)    32      6_938           3_524        3_414
#     (1,1,32,2304)    36      7_555           3_827        3_728
#     (1,1,32,5120)    40      9_309           5_152        4_157
#     (1,1,32,7168)    56     10_929           5_759        5_170
#
# When the group is a dense rectangle with both extents > 1 and its area exceeds
# this cap, the gather goes TWO-STAGE (examples/tensix_all_reduce's measured
# `two_stage_grid_reduce`, 1.45-1.60x over a flat root under grid contention and
# the winner at the 1-tile latency floor): row members -> their row leader, the
# row leaders -> the root. The serial fan-in becomes cx + cy instead of cx * cy,
# and the per-core gather L1 falls the same way. Measured at matched CW (same
# core count, same slices, only the topology differs):
#
#     shape          CW  cx*cy   flat ns   two-stage ns   speedup
#     (1,1,32,1024)  32   4x8      6_938          5_987     1.16x
#     (1,1,32,2304)  36   6x6      7_555          6_513     1.16x
#     (1,1,32,5120)  40   5x8      9_309          8_320     1.12x
#     (1,1,32,7168)  56   7x8     10_929          9_200     1.19x
#
# The second stage is not free — it buys back ~73 ns per contributor removed from
# the root's serial fan-in but costs one extra gather round (~1.3 us, fitted from
# the same measurements). So it only pays above a fairly wide fan-in, and this cap
# is deliberately set at the widest flat gather still measured to be competitive
# rather than at the fitted break-even (~28): every group at or below it keeps the
# Refinement 2 flat topology byte-for-byte, and the staged one engages only in the
# range it is measured to win. Raise it above any reachable group area to force
# flat everywhere; lower it to engage staging on narrower groups.
COMBINE_MAX_FLAT_FANIN = 24

# ---------------------------------------------------------------------------
# Fused square-accumulate (Refinement 5)
# ---------------------------------------------------------------------------
#
# Phases 2 and 3 are "x^2 into cb_x_squared" then "elementwise-accumulate the
# chunk's W-tiles out of cb_x_squared". Both are FPU passes over the SAME
# block, and the second one's operation is an add — so the FPU's accumulate-
# into-DEST mode collapses them into one: `mul_tiles(x, x, acc_to_dest)` over a
# tile-row leaves Sum_w x_w^2 sitting in DEST, which is EXACTLY the raw
# elementwise accumulator both the local finalize and the cross-core combine
# already consume (op_design.md's "the combine is literally the local chunk
# accumulate"). `eltwise_chain`'s DestAccumulation walk expresses it directly:
# D0 stays acquired across an outer row's Wt inputs and is packed once per row.
#
# It removes one FPU op per input tile out of four (square, accumulate, scale,
# gamma), the whole cb_x_squared L1 round trip, and W-1 of every W packs.
# Measured by ablation at the pinned perf config, the BLOCK_SHARDED
# (1,1,8192,1024) geometry spends 63.0 of its 85.2 us purely on TRISCs (BRISC
# 0.3 us, NCRISC 2.6 us) — it is MATH-bound, so removing FPU ops is the only
# lever that reaches it.
#
# Two structural preconditions, both checked on the host so the kernel needs no
# runtime branch:
#
#   NW == 1          the accumulator must not have to survive ACROSS chunks.
#                    A DEST accumulator dies at the next tile_regs_acquire, and
#                    eltwise_chain forbids composing DEST with L1 accumulation,
#                    so a chunked reduce keeps the pairwise-add datapath.
#   !HAS_PARTIAL_W   the 0/1 mask that zeroes a short last W-tile is applied by
#                    the reduce helper's partial-scaler hook, which this path
#                    does not go through. (R1: getting this wrong is invisible
#                    to PCC — it only rescales each row.)
#
# Set False to A/B the pairwise-add datapath back (RMS_NORM_FUSE_SQ=0).
FUSE_SQUARE_ACCUM = True

# Semaphore ids. Disjoint combine groups reuse the SAME ids -- a semaphore id
# resolves to a per-core L1 cell, so group {A,B} bumping id 0 on B is a
# different cell from group {C,D} bumping id 0 on D
# (references/cross_core_reduction_design.md §5).
SEM_GATHER = 0  # stage 1: row members -> their row leader ("my partial landed")
SEM_MCAST_BASE = 1  # Mcast2D takes SEM_MCAST_BASE (data_ready) and +1 (consumer_ready)
SEM_GATHER2 = 5  # stage 2: row leaders -> the group root (two-stage combine only)

# Buffer depths (§1.2). Phase-1 minimal = 2 (double buffer).
X_DEPTH = 2
OUT_DEPTH = 2
GAMMA_DEPTH = 2

# Depth of the RESIDENT input row-strip (§1.3's fast path). At depth 1 the strip
# is single-buffered, so the reader (TILE) / tilize (RM) cannot begin row-block
# hb+1 until compute has drained hb — read and compute serialize across
# row-blocks. Depth 2 overlaps them. Predicate-guarded: the derivation walks
# down from this value to 1 and then to the streaming path, taking the deepest
# strip that fits the L1 wall.
X_RESIDENT_DEPTH = 2

# The tilize LLK always consumes a whole 32-row block from its row-page CB, so
# the row-page CBs must be physically able to hold 32 rows even when only one
# row carries data (the single-stick gamma tilize).
TILIZE_ROWS = TILE_DIM

# dtypes the reduce scaler / partial-W mask dataflow helpers can actually fill.
# reduce_helpers_dataflow.inl static_asserts on exactly these two.
_MASKABLE_DTYPES = (ttnn.float32, ttnn.bfloat16)

# Block-float dtypes: 16 datums share one exponent, so there is no per-element
# size. Two consequences are used below.
_BLOCK_FLOAT_DTYPES = tuple(
    dt for dt in (getattr(ttnn, name, None) for name in ("bfloat8_b", "bfloat4_b")) if dt is not None
)


def _row_elem_bytes(tensor):
    """Bytes per element along a ROW_MAJOR stick.

    ``Tensor.element_size()`` *raises* for a block-float dtype ("datum for
    bfp2, bfp4, bfp8 is invalid", tt_backend_api_types.hpp) because a
    block-float datum has no standalone width. That is not a gap to work
    around: a block-quantized tensor equally cannot BE row-major (no blocks in
    a stick), so every consumer of this value — the ``cb_input_rm`` /
    ``cb_gamma_rm`` page sizes and the four ``*_ROW_BYTES`` compile-time args —
    sits behind an ``is_rm`` / ``is_rm_gamma`` predicate the dtype can never
    satisfy. Return the unpacked-datum width as a structurally-unused
    placeholder instead of letting the exception escape a branch that discards
    the result anyway.
    """
    if tensor.dtype in _BLOCK_FLOAT_DTYPES:
        return 2  # unreachable placeholder — see docstring
    return tensor.element_size()


# ---------------------------------------------------------------------------
# CB indices (semantic names; the number is just the buffer slot)
# ---------------------------------------------------------------------------

CB_INPUT_TILES = 0
CB_GAMMA = 1
CB_SCALER = 2
CB_INPUT_RM = 3
CB_GAMMA_RM = 4
# --- cross-core W-split only (Refinement 2) ---
CB_ONES = 5  # Float32 scaler for the combine reduce (root cores only)
CB_GROUP_PARTIALS = 6  # stage-1 gather: raw sum(x^2), CW1 slots per tile-row (leaders)
CB_RMS_MEAN = 7  # root's combined mean(x^2)  compute -> reader (mcast source)
CB_PARTIAL_OUT = 8  # this core's raw sum(x^2)  compute -> writer (gather source)
CB_GROUP_PARTIALS2 = 9  # stage-2 gather: leaders' row sums, CW2 slots per tile-row (root)
CB_OUTPUT_TILES = 16
CB_OUTPUT_RM = 17
CB_X_SQUARED = 24
CB_PARTIALS = 25
CB_RMS_SUM = 26
CB_RMS_RECIP = 27
CB_SCALED = 28


def _env_flag(name: str, default: bool) -> bool:
    """A boolean knob with an env A/B override, same style as RMS_NORM_W_SPLIT."""
    v = os.environ.get(name)
    return default if v is None else v != "0"


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _prod(values) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


def _coarsest_divisor(n: int, cap: int) -> int:
    """Largest d with ``d | n`` and ``d <= cap`` (cap >= 1, n >= 1)."""
    cap = max(1, min(cap, n))
    for d in range(cap, 0, -1):
        if n % d == 0:
            return d
    return 1


class _Blocking:
    """The derived blocking + CB plan for one (tensor, gamma, budget) triple.

    ``wt_core`` is the W-tile extent a SINGLE core owns. Without a W-split it is
    the whole ``Wt``; under the cross-core W-split (Refinement 2) it is the
    per-core slice, and every knob below (``WT_CHUNK``/``NW``/``HT_BLOCK``, the
    residency predicates, every CB page count) derives from it exactly as
    before — the split changes the *value* of the axis extent, not the model.
    """

    def __init__(
        self,
        input_tensor,
        gamma,
        l1_block_budget_bytes,
        grid_cores,
        wt_core=None,
        rows_core_max=None,
        cw=1,
        cw1=None,
        cw2=1,
        sharded_in=False,
        sharded_out=False,
        l1_total_budget=None,
    ):
        self.sharded_in = bool(sharded_in)
        self.sharded_out = bool(sharded_out)
        # The physical L1 wall (program CBs + resident shards). Defaults to the
        # CB budget, which reproduces the pre-Refinement-4 single-budget model
        # exactly — and is what every interleaved cell sees anyway, since its
        # shard term is 0.
        self.l1_total_budget = L1_CB_BUDGET_BYTES if l1_total_budget is None else int(l1_total_budget)
        self.is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
        self.has_gamma = gamma is not None
        self.is_rm_gamma = self.has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT

        shape = list(input_tensor.shape)
        self.W = int(shape[-1])
        self.wt_global = _ceil_div(self.W, TILE_DIM)
        self.w_valid_last = self.W - (self.wt_global - 1) * TILE_DIM  # in [1, 32]
        self.has_partial_w = (self.W % TILE_DIM) != 0
        # Per-core W extent — the axis the block knobs are derived against.
        self.Wt = self.wt_global if wt_core is None else int(wt_core)
        self.cw = int(cw)
        self.w_split = self.cw > 1
        # Combine fan-in per stage. cw2 == 1 is the flat root gather (cw1 == cw);
        # cw2 > 1 is the two-stage gather, whose serial fan-in — and whose
        # per-core gather L1 — is cw1 + cw2 instead of cw1 * cw2.
        self.cw1 = self.cw if cw1 is None else int(cw1)
        self.cw2 = int(cw2)
        self.two_stage = self.w_split and self.cw2 > 1
        assert self.cw1 * self.cw2 == self.cw, f"combine stages {self.cw1}x{self.cw2} must tile CW={self.cw}"

        # §5.1 tile geometry — alignment-aware, per image, ceil everywhere.
        if self.is_rm:
            self.total_sticks = _prod(shape[:-1])
            self.ht_total = _ceil_div(self.total_sticks, TILE_DIM)
        else:
            self.total_sticks = 0
            self.ht_total = _prod(shape[:-2]) * _ceil_div(int(shape[-2]), TILE_DIM)
        self._rows_core_max = self.ht_total if rows_core_max is None else int(rows_core_max)

        self.tile_bytes = ttnn.tile_size(input_tensor.dtype)
        self.elem_bytes = _row_elem_bytes(input_tensor)
        self.gamma_tile_bytes = ttnn.tile_size(gamma.dtype) if self.has_gamma else self.tile_bytes
        self.gamma_elem_bytes = _row_elem_bytes(gamma) if self.has_gamma else self.elem_bytes
        self.fp32_tile_bytes = ttnn.tile_size(ttnn.float32)
        # --- the reduce-datapath format, and the mask that must match it -----
        #
        # cb_scaler carries the partial-W 0/1 MASK tile, which
        # reduce_accumulate_via_add's fold_partial_last reads through srcB via
        # llk_unpack_AB<ROW>(input_dfb, scaler_dfb) WITHOUT reconfiguring srcB.
        # Reduce entry already ran reconfig_data_format(input_dfb, input_dfb),
        # so srcB is configured at the format of the reduce's INPUT CB — which
        # is `cb_x_squared`, not the input tensor. Phase 0 wrote "the input
        # dtype" for both because the two were always the same value; the
        # load-bearing quantity is cb_x_squared's format, and separating them
        # is what makes a third input dtype expressible. (op_design.md R4's
        # fixed Float16_b is right for the ReduceTile datapath, wrong for this
        # one: a Float16_b mask under an fp32 reduce reads as fp32.)
        #
        # prepare_reduce_mask / calculate_and_prepare_reduce_scaler static_assert
        # on {Float16_b, Float32} (reduce_helpers_dataflow.inl), so the reduce
        # datapath has to be one of those two. bfloat8_b is not: a bf8 tile's
        # leading bytes are the shared-exponent header, so a Float16_b mask read
        # through a Bfp8_b-configured srcB decodes as all-zeros — measured, the
        # whole last reduce-dim tile then contributes 0 (probe_005: all-ones
        # W=49 summed to 32, not 49). PCC hides it completely, because dropping
        # elements only rescales each row and PCC is scale-invariant.
        #
        # So block-float inputs square into a bfloat16 cb_x_squared: the reduce
        # then programs srcA/srcB at Float16_b, the mask matches, and partial-W
        # is correct instead of quietly wrong. It is also the more accurate
        # accumulator input — x^2 spans twice the exponent range of x, which is
        # the worst case for 16-datum shared-exponent blocks.
        self.x_squared_dtype = input_tensor.dtype if input_tensor.dtype in _MASKABLE_DTYPES else ttnn.bfloat16
        self.x_squared_tile_bytes = ttnn.tile_size(self.x_squared_dtype)
        # scaler_dtype is set below, once `fuse_sq` is known: the rule is "match
        # the reduce's INPUT CB", and fusing moves that CB from cb_x_squared to
        # the Float32 accumulator.
        self.scaler_dtype = self.x_squared_dtype
        self.scaler_tile_bytes = self.x_squared_tile_bytes

        # --- BLOCK_CB_UNITS: CB pages charged per block tile (§6.3) ----------
        units = 0
        units += 1 if self.is_rm else X_DEPTH  # cb_input_tiles
        units += X_DEPTH if self.is_rm else 0  # cb_input_rm  (row pages == 1 tile/row-block)
        units += 1  # cb_x_squared
        units += 1 if self.has_gamma else 0  # cb_scaled
        units += 1 if self.is_rm else OUT_DEPTH  # cb_output_tiles
        units += OUT_DEPTH if self.is_rm else 0  # cb_output_rm
        units += GAMMA_DEPTH if self.has_gamma else 0  # cb_gamma (streaming)
        units += 1 if self.is_rm_gamma else 0  # cb_gamma_rm (conservative unit)
        self.block_cb_units = units
        # Widest block-scaled page, so TILE_BLOCK_BUDGET stays an upper bound on
        # the real per-block footprint. cb_x_squared is included because it can
        # be wider than the input tile (block-float input -> bfloat16 square).
        self.unit_bytes = max(self.tile_bytes, self.gamma_tile_bytes, self.x_squared_tile_bytes)

        # --- the block factors -----------------------------------------------
        self.tile_block_budget = max(1, l1_block_budget_bytes // (self.block_cb_units * self.unit_bytes))
        self.wt_chunk = _coarsest_divisor(self.Wt, self.tile_block_budget)
        self.nw = self.Wt // self.wt_chunk
        self.wt_last = self.Wt - (self.nw - 1) * self.wt_chunk  # == wt_chunk by construction
        if self.nw > 1:
            # R7: TileOffset::Set(wc*WT_CHUNK) into a resident row-strip is only
            # correct when a resident block is one flat Wt-tile strip.
            self.ht_block = 1
        else:
            self.ht_block = max(1, min(self.tile_block_budget // self.wt_chunk, self._rows_core_max))

        assert self.wt_last == self.wt_chunk, "chunk-uniformity invariant broken"
        assert not (self.nw > 1 and self.ht_block > 1), "R7: NW > 1 requires HT_BLOCK == 1"

        # --- fused square-accumulate (R5) ------------------------------------
        # Decided here, between the block factors it depends on and the CB plan
        # that depends on it: fusing retires cb_x_squared, and the L1 that frees
        # is real and must be visible to the residency predicates below.
        self.fuse_sq = (
            bool(_env_flag("RMS_NORM_FUSE_SQ", FUSE_SQUARE_ACCUM)) and self.nw == 1 and not self.has_partial_w
        )
        if self.fuse_sq:
            # Same single-source rule R1 established for the partial-W mask: the
            # scaler CB's format must be the format the reduce programs srcB at,
            # which is its INPUT CB's. Fusing moves that input from
            # cb_x_squared to the Float32 accumulator (cb_partials /
            # cb_partial_out), so the scaler follows it. Its *content* is unused
            # here — fuse_sq implies no partial-W mask — but the format still
            # drives the srcB reconfig, and R1 measured what a mismatch costs.
            self.scaler_dtype = ttnn.float32
            self.scaler_tile_bytes = self.fp32_tile_bytes

        # --- residency fast-path predicates (§1.3) ---------------------------
        #
        # L1 is spent in strict order of how much DRAM TRAFFIC each step removes;
        # a step that only buys pipelining comes last:
        #
        #   1. X_RESIDENT at depth 1  — removes an entire second read pass over x
        #                               (ht_per_core * Wt tile-reads). Biggest win.
        #   2. GAMMA_RESIDENT         — removes NH_core * Wt gamma re-reads.
        #   3. extra resident depth   — removes NO traffic; it only lets the
        #                               producer run a row-block ahead.
        #
        # §1.3 fixes the 1-before-2 order; 3 must come last for the same reason,
        # and getting that wrong is expensive: buying depth 2 before gamma cost
        # 1.20x on (1,1,8192,5120) (482_655 -> 577_752 ns) purely by evicting
        # gamma from L1.
        #
        # Step 3 is additionally gated on there being latency to hide at all. With
        # the grid full every core is queued on DRAM, so running the reader a
        # row-block ahead only front-loads contention: measured 1.04x SLOWER on
        # (1,1,8192,1024) (103_238 -> 107_592 ns). Same structural discriminator
        # as the read-batch knob (see _x_read_chunks).
        num_cores = min(grid_cores, self.ht_total * self.cw)
        self.grid_full = num_cores >= grid_cores
        # Row-blocks a single core loops over. Both "hold it across row-blocks"
        # levers below are worth nothing when this is 1.
        self.nh_core_max = _ceil_div(self._rows_core_max, self.ht_block)
        base = self._cb_total(x_res_depth=0, gamma_resident=False)

        # A sharded input is resident BY PLACEMENT — the shard is already in this
        # core's L1, so the residency predicate is not a choice here.
        self.x_res_depth = 1 if (self.sharded_in or self._fits(1, False)) else 0
        self.x_resident = self.x_res_depth > 0

        # Gamma residency removes (NH_core - 1) * Wt gamma re-reads per core — so
        # at NH_core == 1 it removes NOTHING: gamma is read exactly once either
        # way, and holding it resident only converts an overlappable per-chunk
        # read in pass B into a serial prologue that must complete before the
        # first input tile is even requested. Measured on (1,1,32,5120), one
        # core: resident gamma 47_571 ns vs streamed gamma 42_407 ns (1.12x).
        self.gamma_resident = False
        if self.has_gamma and self.nh_core_max > 1:
            self.gamma_resident = self._fits(self.x_res_depth, True)

        # EXPERIMENT override. A one-shot broadcast delivers the whole of gamma
        # once, so the mcast candidate structurally requires the resident spelling
        # even on the geometries (NH_core == 1) where the op deliberately streams.
        # Exposing it as its own knob keeps the two halves of that change
        # separable on device: "make gamma resident" vs "broadcast it".
        if self.has_gamma and FORCE_GAMMA_RESIDENT is not None:
            self.gamma_resident = bool(FORCE_GAMMA_RESIDENT) and self._fits(self.x_res_depth, True)

        # Extra resident depth overlaps row-block hb+1's read with hb's compute,
        # so it likewise needs NH_core > 1 — and needs spare latency to hide
        # (see the grid_full note above).
        #
        # It is also meaningless on the RM path: there cb_input_tiles is produced
        # by compute's own tilize and consumed by compute's square/mul, so its
        # producer and consumer are the SAME RISC. A second strip cannot be
        # filled ahead — it would only cost L1 (and can evict gamma).
        if self.x_resident and not self.is_rm and not self.grid_full and self.nh_core_max > 1:
            for depth in range(X_RESIDENT_DEPTH, self.x_res_depth, -1):
                if self._fits(depth, self.gamma_resident):
                    self.x_res_depth = depth
                    break

        self.program_cb_bytes, self.resident_shard_bytes = self._cb_bytes(self.x_res_depth, self.gamma_resident)
        self.cb_total_bytes = self.program_cb_bytes + self.resident_shard_bytes
        self.fits = self._fits(self.x_res_depth, self.gamma_resident)
        self.base_cb_total_bytes = base

    # -- CB plan ------------------------------------------------------------
    def cb_plan(self, x_res_depth=None, gamma_resident=None):
        """[(name, index, page_size, num_pages)] for every CB.

        `x_res_depth` is the resident-strip buffer depth (0 = streaming).

        Unused CBs are still declared (1 page) so that the compile-time CB
        descriptors the helpers read (tile size / tile dims / format) are valid
        even in `if constexpr`-discarded branches.
        """
        x_depth = self.x_res_depth if x_res_depth is None else x_res_depth
        g_res = self.gamma_resident if gamma_resident is None else gamma_resident

        H = self.ht_block
        C = self.wt_chunk
        B = H * C
        Wt = self.Wt

        if self.sharded_in:
            # Zero-copy: the CB *is* the resident shard, so its extent is the
            # whole shard, not a block multiple.
            input_tile_pages = self._rows_core_max * Wt
        elif x_depth > 0:
            input_tile_pages = x_depth * H * Wt
        elif self.is_rm:
            input_tile_pages = B  # compute->compute (tilize -> square), one block
        else:
            input_tile_pages = X_DEPTH * B

        gamma_pages = 0
        if self.has_gamma:
            gamma_pages = Wt if g_res else GAMMA_DEPTH * C

        plan = [
            ("cb_input_tiles", CB_INPUT_TILES, self.tile_bytes, input_tile_pages),
            ("cb_gamma", CB_GAMMA, self.gamma_tile_bytes, gamma_pages),
            ("cb_scaler", CB_SCALER, self.scaler_tile_bytes, 1),
            # One row-page per stick; a tile-row of the block is TILIZE_ROWS sticks.
            (
                "cb_input_rm",
                CB_INPUT_RM,
                C * TILE_DIM * self.elem_bytes,
                (X_DEPTH * H * TILIZE_ROWS) if self.is_rm else 0,
            ),
            (
                "cb_gamma_rm",
                CB_GAMMA_RM,
                C * TILE_DIM * self.gamma_elem_bytes,
                TILIZE_ROWS if self.is_rm_gamma else 0,
            ),
            (
                "cb_output_tiles",
                CB_OUTPUT_TILES,
                self.tile_bytes,
                (self._rows_core_max * Wt) if self.sharded_out else (B if self.is_rm else OUT_DEPTH * B),
            ),
            ("cb_output_rm", CB_OUTPUT_RM, self.tile_bytes, (OUT_DEPTH * B) if self.is_rm else 0),
            # Retired by the fused square-accumulate: with the sum living in
            # DEST there is no x^2 block to stage through L1 at all.
            ("cb_x_squared", CB_X_SQUARED, self.x_squared_tile_bytes, 0 if self.fuse_sq else B),
            # --- cross-core W-split (Refinement 2). Every page count is a
            # function of the SAME knobs (HT_BLOCK and the group width CW),
            # never of a whole-op dimension. Sized 1 (dummy) when CW == 1.
            ("cb_ones", CB_ONES, self.fp32_tile_bytes, 1),
            (
                "cb_group_partials",
                CB_GROUP_PARTIALS,
                self.fp32_tile_bytes,
                (H * self.cw1) if self.w_split else 0,
            ),
            (
                "cb_group_partials2",
                CB_GROUP_PARTIALS2,
                self.fp32_tile_bytes,
                (H * self.cw2) if self.two_stage else 0,
            ),
            ("cb_rms_mean", CB_RMS_MEAN, self.fp32_tile_bytes, H if self.w_split else 0),
            ("cb_partial_out", CB_PARTIAL_OUT, self.fp32_tile_bytes, H if self.w_split else 0),
            ("cb_partials", CB_PARTIALS, self.fp32_tile_bytes, 2 * H),
            ("cb_rms_sum", CB_RMS_SUM, self.fp32_tile_bytes, H),
            ("cb_rms_recip", CB_RMS_RECIP, self.fp32_tile_bytes, H),
            ("cb_scaled", CB_SCALED, self.tile_bytes, B if self.has_gamma else 0),
        ]
        return [(n, i, ps, max(1, np)) for (n, i, ps, np) in plan]

    def _is_aliased(self, name):
        """Is this CB's storage the tensor's own buffer (zero-copy sharded I/O)?

        Such a CB costs L1 but is NOT program-allocated — the buffer allocator
        reserved it when the tensor was created. See L1_ALLOC_HEADROOM_BYTES.
        """
        return (self.sharded_in and name == "cb_input_tiles") or (self.sharded_out and name == "cb_output_tiles")

    def _cb_bytes(self, x_res_depth, gamma_resident):
        """(program-allocated bytes, aliased resident-shard bytes) for one plan."""
        prog = shard = 0
        for name, _, ps, np in self.cb_plan(x_res_depth, gamma_resident):
            if self._is_aliased(name):
                shard += ps * np
            else:
                prog += ps * np
        return prog, shard

    def _cb_total(self, x_res_depth, gamma_resident):
        prog, shard = self._cb_bytes(x_res_depth, gamma_resident)
        return prog + shard

    def _fits(self, x_res_depth, gamma_resident):
        """The one physical wall: everything this program puts in the L1 bank.

        Program-allocated CBs and zero-copy resident shards are both real bytes
        in the same bank, so they are summed against the same live-device budget
        (see L1_ALLOC_HEADROOM_BYTES). The shard term is 0 on an interleaved
        tensor, so there the condition is purely "do the CBs fit".
        """
        prog, shard = self._cb_bytes(x_res_depth, gamma_resident)
        return prog + shard <= self.l1_total_budget


def _tile_geometry(input_tensor):
    """(ht_total, wt_global) — the whole-op tile grid, before any core split."""
    shape = list(input_tensor.shape)
    wt_global = _ceil_div(int(shape[-1]), TILE_DIM)
    if input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT:
        ht_total = _ceil_div(_prod(shape[:-1]), TILE_DIM)
    else:
        ht_total = _prod(shape[:-2]) * _ceil_div(int(shape[-2]), TILE_DIM)
    return ht_total, wt_global


def _l1_total_budget(device):
    """Per-core L1 the op may commit to (program CBs + any resident shard).

    Read from the live device's L1 bank size — arch- and dispatch-config
    specific — minus ``L1_ALLOC_HEADROOM_BYTES`` for allocator fragmentation.
    Falls back to ``L1_CB_BUDGET_BYTES`` if the memory view is unavailable,
    which is always safe: that constant is strictly more conservative than any
    bank this op runs on.
    """
    try:
        bank = int(ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank)
    except Exception:
        return L1_CB_BUDGET_BYTES
    # A/B hook, same style as RMS_NORM_W_SPLIT: a headroom wider than the bank
    # clamps back to L1_CB_BUDGET_BYTES, i.e. the pre-Refinement-5 wall, so the
    # "guessed budget vs live bank" comparison stays re-runnable without editing
    # the op. Combined with L1_CB_BUDGET_BYTES = 0 it forces the streaming path.
    headroom = os.environ.get("RMS_NORM_L1_HEADROOM_KB")
    headroom = int(headroom) * 1024 if headroom else L1_ALLOC_HEADROOM_BYTES
    return max(L1_CB_BUDGET_BYTES, bank - headroom)


def _derive_blocking(
    input_tensor, gamma, grid_cores, placement, sharded_in=False, sharded_out=False, l1_total_budget=None
):
    """Derive the blocking, halving the block budget until the CBs fit.

    One wall, per ``_Blocking._fits``: program-allocated CBs plus any zero-copy
    resident shard, against the live L1 bank (``l1_total_budget``).

    The halving loop also absorbs the ROOT core's gather buffer
    (``HT_BLOCK * CW`` fp32 tiles): it is charged into ``program_cb_bytes`` for
    every core, so an over-wide combine shrinks ``HT_BLOCK`` rather than
    silently overflowing L1 on the roots.
    """
    kwargs = dict(
        wt_core=placement.wt_core,
        rows_core_max=placement.rows_core_max,
        cw=placement.cw,
        cw1=placement.cw1,
        cw2=placement.cw2,
        sharded_in=sharded_in,
        sharded_out=sharded_out,
        l1_total_budget=l1_total_budget,
    )
    budget = L1_BLOCK_BUDGET_BYTES
    blk = _Blocking(input_tensor, gamma, budget, grid_cores, **kwargs)
    while not blk.fits and budget > blk.unit_bytes:
        budget //= 2
        blk = _Blocking(input_tensor, gamma, budget, grid_cores, **kwargs)
    assert blk.fits, (
        f"rms_norm: per-core L1 does not fit even at the minimum block size — "
        f"program CBs {blk.program_cb_bytes} B + resident shards "
        f"{blk.resident_shard_bytes} B against an L1 budget of {blk.l1_total_budget} B"
    )
    return blk


def _x_read_chunks(blk):
    """How many W-chunks the reader coalesces into ONE reserve/barrier/push.

    This is the read-granularity knob, and it is measured, not guessed. The two
    regimes want opposite values (blackhole_p150b, 110 cores, bf16 TILE gamma,
    device kernel ns):

      shape              cores   1 chunk/barrier   NW chunks/barrier
      (1,1,32,5120)          1            42_407              51_434
      (1,1,32,7168)          1            57_396              63_313
      (1,1,8192,5120)      110           575_867             482_655
      (1,1,8192,2304)      110           227_673             214_633

    With only a few cores busy the op is LATENCY-bound: one chunk per barrier
    lets compute start on chunk 0 while the NoC fetches chunk 1, worth up to
    1.21x. With the grid full every core is already queued on DRAM, so the op is
    THROUGHPUT-bound: there is no latency left to hide and each extra barrier
    just exposes the (now much longer) NoC tail, costing up to 1.19x.

    The discriminator is therefore structural, not a tuned threshold: has the
    independent row axis already filled the grid? Only the resident TILE path is
    affected — the streaming path's input CB is sized X_DEPTH*B and physically
    cannot hold a multi-chunk batch, and the RM path's cb_input_tiles is filled
    by compute's tilize, not by the reader.
    """
    if blk.sharded_in or not (blk.x_resident and not blk.is_rm):
        return 1
    return blk.nw if blk.grid_full else 1


# A NoC multicast addresses a rectangle in VIRTUAL coordinates, and the logical
# compute grid is not virtually contiguous: on blackhole_p150b logical x 0..6 map
# to virtual 1..7 and logical x 7..10 to virtual 10..13, so virtual columns 8-9
# are NOT worker cores. A rectangle straddling that seam multicasts into
# non-worker endpoints — measured as a device hang on the WIDTH_SHARDED cells
# whose shard grid is wider than one run. Every broadcast rectangle below is
# therefore confined to ONE run; a combine group that spans the seam broadcasts
# as one multicast family PER run instead of one over the bounding box.
MAX_MCAST_FAMILIES = 2


def _virtual_x_runs(device, grid):
    """Logical-x ranges [(lo, hi), ...] whose virtual x is contiguous."""
    vx = [int(device.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x) for x in range(grid.x)]
    runs = []
    lo = 0
    for x in range(1, grid.x):
        if vx[x] != vx[x - 1] + 1:
            runs.append((lo, x - 1))
            lo = x
    runs.append((lo, grid.x - 1))
    return runs


def _split_rect_by_runs(x0, y0, x1, y1, runs):
    """The rectangle, cut into the per-run pieces a multicast may legally address."""
    out = []
    for lo, hi in runs:
        a, b = max(x0, lo), min(x1, hi)
        if a <= b:
            out.append((a, y0, b, y1))
    return out


class _CoreWork:
    """What one core owns: a tile-row range x a W-tile slice, plus its combine role.

    The combine role is three fields, all defaulted so the flat topology (and the
    no-split path) needs no extra bookkeeping:

      ``s1_slot``   this core's tile index inside its stage-1 gather target
      ``s2_slot``   a LEADER's tile index inside the root's stage-2 gather
      ``is_leader`` does this core run stage 1? (always the root on a flat gather)

    ``leader`` is the logical core the stage-1 partial is unicast to — the row
    leader under a two-stage combine, the group root under a flat one.
    """

    __slots__ = (
        "core",
        "start_row",
        "num_rows",
        "wt_start",
        "wt_real",
        "slot",
        "group",
        "family",
        "leader",
        "s1_slot",
        "s2_slot",
        "is_leader",
    )

    def __init__(self, core, start_row, num_rows, wt_start, wt_real, slot, group, family=0):
        self.core = core
        self.start_row = start_row
        self.num_rows = num_rows
        self.wt_start = wt_start
        self.wt_real = wt_real  # W-tiles that actually exist (rest is shard padding)
        self.slot = slot  # index inside the combine group
        self.group = group  # index into _Placement.groups (-1 == no group)
        self.family = family  # which multicast family (per-run sub-rectangle) reaches it
        self.leader = None  # stage-1 gather target (filled by _assign_combine_roles)
        self.s1_slot = slot
        self.s2_slot = 0
        self.is_leader = False


def _two_stage_extents(cx, cy):
    """(cw1, cw2) for a dense cx x cy group: two-stage when it beats a flat root.

    A stage narrower than 2 buys nothing (its round-trip is pure overhead), and a
    fan-in at or below ``COMBINE_MAX_FLAT_FANIN`` is cheaper to serialize on one
    root than to split. Otherwise the rows gather to their leader (fan-in cx) and
    the leaders to the root (fan-in cy).
    """
    if cx > 1 and cy > 1 and cx * cy > COMBINE_MAX_FLAT_FANIN:
        return cx, cy
    return cx * cy, 1


def _gather_tiles(cx, cy):
    """fp32 gather tiles one core must hold for a dense cx x cy group's combine."""
    cw1, cw2 = _two_stage_extents(cx, cy)
    return cw1 + cw2 if cw2 > 1 else cw1


def _assign_combine_roles(works, group_index, root, cw1, cw2, x0, y0):
    """Fill every core's stage-1/stage-2 combine role for one group.

    Flat (``cw2 == 1``): the root is the only leader and ``s1_slot`` stays the
    core's index in the group. Two-stage: the leader of a core is the core at
    column ``x0`` of its own grid row, ``s1_slot`` is its column offset and a
    leader's ``s2_slot`` is its row offset.
    """
    for w in works:
        if w.group != group_index or w.num_rows == 0:
            continue
        if cw2 > 1:
            w.leader = ttnn.CoreCoord(x0, int(w.core.y))
            w.s1_slot = int(w.core.x) - x0
            w.s2_slot = int(w.core.y) - y0
            w.is_leader = int(w.core.x) == x0
        else:
            w.leader = root
            w.s1_slot = w.slot
            w.s2_slot = 0
            w.is_leader = int(w.core.x) == int(root.x) and int(w.core.y) == int(root.y)


class _Placement:
    """Core assignment + cross-core combine topology (op_design.md §5 and §4.2).

    Two axes are placed, not one:
      * the INDEPENDENT tile-row axis, split across *groups* (zero traffic), and
      * the DEPENDENT W axis, split across the ``cw`` cores INSIDE each group,
        which therefore need a partial-sum combine + a 1/rms broadcast.

    ``cw == 1`` degenerates to the phase-0 row-only split, and every W-split
    structure below collapses out.

    Invariant relied on by the broadcast: **every group is a rectangle**, so a
    ``Mcast2D`` per virtually-contiguous run of it serves it. Groups the data
    placement hands us that are ragged are padded up to their bounding box with
    zero-work filler cores.
    """

    def __init__(self, all_cores, works, groups, cw, wt_core, rows_core_max, cw1=None, cw2=1):
        self.all_cores = all_cores
        self.works = works
        # [{root, subrects: [(x0,y0,x1,y1), ...], acks: [n, ...]}]
        self.groups = groups
        self.cw = cw
        # Combine fan-in per stage; cw1 * cw2 == cw. cw2 == 1 is the flat root
        # gather. Uniform across every group, because one compile-time kernel
        # serves the whole grid.
        self.cw1 = cw if cw1 is None else cw1
        self.cw2 = cw2
        self.wt_core = wt_core
        self.rows_core_max = rows_core_max

    @property
    def w_split(self):
        return self.cw > 1

    @property
    def two_stage(self):
        return self.w_split and self.cw2 > 1

    @property
    def num_cores(self):
        return len(self.works)


def _rect(x0, y0, x1, y1):
    return ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))


def _rows_split(total, n):
    """Split `total` tile-rows over `n` groups, biggest-first (like split_work_to_cores)."""
    q, r = divmod(total, n)
    return [q + 1 if i < r else q for i in range(n)]


def _placement_rows(grid, ht_total):
    """Phase-0 placement: the independent tile-row axis over the whole grid."""
    (
        _num,
        all_cores,
        group_1,
        group_2,
        rows_g1,
        rows_g2,
    ) = ttnn.split_work_to_cores(grid, ht_total, row_wise=True)

    works = []
    start = 0
    for group, per_core in ((group_1, rows_g1), (group_2, rows_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            works.append(_CoreWork(core, start, per_core, 0, None, 0, -1))
            start += per_core
    assert start == ht_total, f"work split covered {start} of {ht_total} tile-rows"
    rows_max = max((w.num_rows for w in works), default=1)
    return _Placement(all_cores, works, [], 1, None, rows_max)


def _w_split_rectangle(max_run_w, band_h, wt_global, gather_tile_bytes):
    """Widest core rectangle (cw_x <= max_run_w, cw_y <= band_h) that DIVIDES `wt_global`.

    Three constraints, all structural:
      * the rectangle must fit inside ONE virtually-contiguous column run, so the
        1/rms broadcast never addresses a non-worker endpoint;
      * the group size must divide `Wt`, which keeps every core's slice the same
        width — so one compile-time `WT` / `WT_CHUNK` / `NW` serves the whole grid,
        and the last core's last tile is still the tensor's last tile (where the
        partial-W mask belongs); and
      * its combine must fit ``L1_GATHER_BUDGET_BYTES``. That cost is the
        topology's, not the area's (``_gather_tiles``): a two-stage rectangle
        holds cx + cy tiles where a flat one holds cx * cy, so the budget reaches
        a much wider group once the gather is staged.

    Ties on area are broken toward the CHEAPEST combine, then toward the squarer
    rectangle — a square minimises cx + cy at fixed area, which is exactly the
    two-stage serial fan-in.
    """
    budget_tiles = max(1, L1_GATHER_BUDGET_BYTES // gather_tile_bytes)
    best = (1, 1)
    for cy in range(1, band_h + 1):
        for cx in range(1, max_run_w + 1):
            n = cx * cy
            if wt_global % n != 0 or _gather_tiles(cx, cy) > budget_tiles:
                continue
            if (n, -_gather_tiles(cx, cy)) > (best[0] * best[1], -_gather_tiles(*best)):
                best = (cx, cy)
    return best


def _placement_wsplit(grid, ht_total, wt_global, gather_tile_bytes, runs):
    """Cross-core W-split for an INTERLEAVED input whose row axis under-fills the grid.

    Group rectangles tile the grid inside the virtually-contiguous column runs,
    y-band by y-band, so every group's broadcast is a single legal multicast.
    Returns None when no split wider than one core is available.
    """
    num_groups = min(ht_total, grid.y)
    band_h = grid.y // num_groups
    max_run_w = max(hi - lo + 1 for lo, hi in runs)
    cw_x, cw_y = _w_split_rectangle(max_run_w, band_h, wt_global, gather_tile_bytes)
    cw = cw_x * cw_y
    if cw < 2:
        return None

    # Every (run, y-band) slot that can hold a whole cw_x x cw_y rectangle.
    slots = []
    for band in range(grid.y // cw_y):
        for lo, hi in runs:
            for x0 in range(lo, hi - cw_x + 2, cw_x):
                slots.append((x0, band * cw_y))
    num_groups = min(num_groups, len(slots))
    if num_groups < 1:
        return None

    wt_core = wt_global // cw
    rows = _rows_split(ht_total, num_groups)
    cw1, cw2 = _two_stage_extents(cw_x, cw_y)

    works = []
    groups = []
    ranges = []
    start_row = 0
    for g in range(num_groups):
        x0, y0 = slots[g]
        ranges.append(_rect(x0, y0, x0 + cw_x - 1, y0 + cw_y - 1))
        root = ttnn.CoreCoord(x0, y0)
        groups.append(
            {
                "root": root,
                "subrects": [(x0, y0, x0 + cw_x - 1, y0 + cw_y - 1)],
                "acks": [cw - 1],
            }
        )
        slot = 0
        for yy in range(y0, y0 + cw_y):
            for xx in range(x0, x0 + cw_x):
                works.append(_CoreWork(ttnn.CoreCoord(xx, yy), start_row, rows[g], slot * wt_core, wt_core, slot, g, 0))
                slot += 1
        _assign_combine_roles(works, g, root, cw1, cw2, x0, y0)
        start_row += rows[g]
    assert start_row == ht_total
    return _Placement(ttnn.CoreRangeSet(ranges), works, groups, cw, wt_core, max(rows), cw1, cw2)


def _shard_geometry(tensor):
    """(shard_ht_tiles, shard_wt_tiles, cores_row_major, grid) for a sharded tensor."""
    spec = tensor.memory_config().shard_spec
    sh, sw = int(spec.shape[0]), int(spec.shape[1])
    grid = spec.grid
    cores = ttnn.corerange_to_cores(grid, None, True)
    return _ceil_div(sh, TILE_DIM), _ceil_div(sw, TILE_DIM), cores, grid


def _bbox_cores(core_range_set):
    bb = core_range_set.bounding_box()
    return bb.start.x, bb.start.y, bb.end.x, bb.end.y


def _make_group(root, gx0, gy0, gx1, gy1, runs, real_cores):
    """A combine group's broadcast plan: one multicast family per legal sub-rect.

    `real_cores` is the set of (x, y) that will actually ack (filler cores inside
    the bounding box receive the broadcast but never handshake, so they are
    excluded from every family's ack count).
    """
    subrects = _split_rect_by_runs(gx0, gy0, gx1, gy1, runs)
    assert len(subrects) <= MAX_MCAST_FAMILIES, (
        f"rms_norm: combine group spans {len(subrects)} virtual column runs; the kernels "
        f"carry {MAX_MCAST_FAMILIES} multicast families"
    )
    rx, ry = int(root.x), int(root.y)
    acks = []
    for x0, y0, x1, y1 in subrects:
        n = sum(
            1
            for yy in range(y0, y1 + 1)
            for xx in range(x0, x1 + 1)
            if (xx, yy) in real_cores and not (xx == rx and yy == ry)
        )
        acks.append(n)
    return {"root": root, "subrects": subrects, "acks": acks}


def _gamma_mcast_plan(placement, runs):
    """EXPERIMENT — the gamma broadcast's geometry: one family per virtual-x run.

    ``gamma`` does not vary along the tile-row axis, which is the axis the
    row-split (``cw == 1``) distributes across cores, so every active core reads
    the SAME ``Wt`` pages. This plans the broadcast that replaces those reads with
    one DRAM read per virtually-contiguous column run.

    Returns ``([(rect, injector, ack_count)], {(x, y): family})`` or ``None`` when
    the active core set is not expressible as at most ``MAX_MCAST_FAMILIES`` DENSE
    rectangles — a broadcast rectangle that covers a core running no kernel has
    nothing to reserve the landing CB and nothing to ack the handshake, so the
    caller falls back to the per-core DRAM read rather than guess.
    """
    active = [(int(w.core.x), int(w.core.y)) for w in placement.works if w.num_rows > 0]
    if not active:
        return None
    fams = []
    fam_of = {}
    for lo, hi in runs:
        sub = [c for c in active if lo <= c[0] <= hi]
        if not sub:
            continue
        x0, x1 = min(c[0] for c in sub), max(c[0] for c in sub)
        y0, y1 = min(c[1] for c in sub), max(c[1] for c in sub)
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        if len(sub) != area or len(fams) >= MAX_MCAST_FAMILIES:
            return None
        f = len(fams)
        fams.append(((x0, y0, x1, y1), (x0, y0), area - 1))
        for c in sub:
            fam_of[c] = f
    if GAMMA_ONE_INJECTOR and len(fams) > 1:
        inj = fams[0][1]
        for f in range(1, len(fams)):
            rect, _, _ = fams[f]
            x0, y0, x1, y1 = rect
            area = (x1 - x0 + 1) * (y1 - y0 + 1)
            inside = x0 <= inj[0] <= x1 and y0 <= inj[1] <= y1
            # The injector sits OUTSIDE this run's rectangle, so every core in it
            # is a receiver (fan-out == area, not area - 1).
            fams[f] = (rect, inj, area - 1 if inside else area)
    return fams, fam_of


def _family_of(core, subrects):
    x, y = int(core.x), int(core.y)
    for i, (x0, y0, x1, y1) in enumerate(subrects):
        if x0 <= x <= x1 and y0 <= y <= y1:
            return i
    return 0


def _placement_sharded(input_tensor, ht_total, wt_global, runs):
    """Placement pinned by the input's shard grid.

    HEIGHT: one core per shard, each owning ht_s tile-rows of the FULL W — the
            phase-0 row split made physical, so the reduce stays LOCAL (cw == 1,
            no combine, no semaphore, no multicast).
    WIDTH:  one group over every shard core (each owns a W slice of ALL rows).
    BLOCK:  one group per grid row (each row owns a tile-row band, split by W).

    A ragged shard grid (auto_shard_config emits full rows + a partial last row)
    is padded up to its bounding box with zero-work filler cores so the group
    stays the single rectangle a multicast can address. HEIGHT needs none of
    that: with no broadcast there is no rectangle to keep legal, so the core set
    IS the shard grid.
    """
    layout = input_tensor.memory_config().memory_layout
    ht_s, wt_s, cores, grid = _shard_geometry(input_tensor)
    x0, y0, x1, y1 = _bbox_cores(grid)

    works = []
    groups = []
    owned = {(int(c.x), int(c.y)) for c in cores}

    if layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        # Refinement 4 / op_design Lamp L3 — a knob-turn, not a scheme change.
        # The shard grid pins which core owns which tile-rows and the shard
        # height pins how many (rows_core_max -> HT_BLOCK); nothing else in the
        # model moves. Each core still owns WHOLE rows, so the dependent W axis
        # is never split: cw stays 1 and every W-split structure collapses out
        # exactly as it does on the interleaved row-split path.
        for i, core in enumerate(cores):
            start = i * ht_s
            rows = max(0, min(ht_s, ht_total - start))
            works.append(_CoreWork(core, start, rows, 0, min(wt_s, wt_global), 0, -1))
        return _Placement(grid, works, [], 1, wt_s, ht_s)
    # Two-stage needs a DENSE rectangle: every row must have a real core at
    # column x0 to lead it, and every row's fan-in must be the same cx. A ragged
    # shard grid (auto_shard_config's full rows + partial last row) has neither,
    # so it keeps the flat root gather.
    dense = len(cores) == (x1 - x0 + 1) * (y1 - y0 + 1)
    if layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        nx = x1 - x0 + 1
        ny = y1 - y0 + 1
        assert len(cores) == nx * ny, "BLOCK_SHARDED grid must be a full rectangle"
        for gy in range(ny):
            y = y0 + gy
            grp = _make_group(ttnn.CoreCoord(x0, y), x0, y, x1, y, runs, owned)
            groups.append(grp)
            for gx in range(nx):
                core = ttnn.CoreCoord(x0 + gx, y)
                works.append(
                    _CoreWork(
                        core,
                        gy * ht_s,
                        ht_s,
                        gx * wt_s,
                        max(0, min(wt_s, wt_global - gx * wt_s)),
                        gx,
                        gy,
                        _family_of(core, grp["subrects"]),
                    )
                )
        # Each BLOCK_SHARDED group is ONE grid row, so cw_y == 1 and the combine
        # is flat by construction (a one-row group has no second axis to stage).
        cw = nx
        cw1, cw2 = _two_stage_extents(nx, 1)
        for gy in range(ny):
            _assign_combine_roles(works, gy, ttnn.CoreCoord(x0, y0 + gy), cw1, cw2, x0, y0 + gy)
        rows_max = ht_s
    else:  # WIDTH_SHARDED — one group, every shard core owns a W slice of all rows
        cw = len(cores)  # combine width == real W slices (fillers do not contribute)
        grp = _make_group(cores[0], x0, y0, x1, y1, runs, owned)
        groups.append(grp)
        for slot, core in enumerate(cores):
            works.append(
                _CoreWork(
                    core,
                    0,
                    ht_total,
                    slot * wt_s,
                    max(0, min(wt_s, wt_global - slot * wt_s)),
                    slot,
                    0,
                    _family_of(core, grp["subrects"]),
                )
            )
        # filler cores: inside the multicast rectangle, but outside the shard grid.
        # Zero work, no gather contribution, no readiness ack — they only need
        # the broadcast to land somewhere legal on them.
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                if (xx, yy) not in owned:
                    core = ttnn.CoreCoord(xx, yy)
                    works.append(_CoreWork(core, 0, 0, 0, 0, 0, 0, _family_of(core, grp["subrects"])))
        cw1, cw2 = _two_stage_extents(x1 - x0 + 1, y1 - y0 + 1) if dense else (cw, 1)
        _assign_combine_roles(works, 0, cores[0], cw1, cw2, x0, y0)
        rows_max = ht_total

    all_cores = ttnn.CoreRangeSet([_rect(x0, y0, x1, y1)])
    return _Placement(all_cores, works, groups, cw, wt_s, rows_max, cw1, cw2)


def create_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    gamma=None,
    epsilon: float = 1e-6,
    compute_kernel_config=None,
    device=None,
) -> "ttnn.ProgramDescriptor":
    if device is None:
        device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    ht_total, wt_global = _tile_geometry(input_tensor)
    # "Sharded" here means ZERO-COPY: the shard is the per-core block, so its CB
    # is aliased straight onto the tensor buffer and no NoC transfer happens.
    # That is only meaningful for a TILE shard. eval.sharding's ROW_MAJOR granule
    # is (1 row, L1_align/elem_bytes columns), so an RM shard is a handful of
    # STICKS (e.g. [1, 128] or [3, 512]) — never the 32 sticks the in-place
    # tilize consumes. An RM shard therefore is NOT this core's block: the 32
    # sticks of one tile-row live on up to 32 DIFFERENT cores, so the read is
    # genuinely non-local and goes through the TensorAccessor, exactly as an
    # interleaved tensor's does. Only the placement of the pages changes.
    tile_shard = input_tensor.layout == ttnn.TILE_LAYOUT
    in_sharded = tile_shard and input_tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    out_sharded = tile_shard and output_tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED

    placement = _select_placement(device, grid, input_tensor, ht_total, wt_global, in_sharded)
    blk = _derive_blocking(
        input_tensor,
        gamma,
        grid.x * grid.y,
        placement,
        sharded_in=in_sharded,
        sharded_out=out_sharded,
        l1_total_budget=_l1_total_budget(device),
    )

    all_cores = placement.all_cores
    x_read_chunks = _x_read_chunks(blk)

    # ---------- circular buffers ----------
    # A sharded tensor's shard IS the per-core block and it is ALREADY in this
    # core's L1, so its CB is placed straight on the buffer (zero-copy, no NoC
    # read at all). Everything else is a normal program-allocated CB.
    #
    # Every non-sharded CB is declared over the WHOLE core set, at identical
    # sizes. That uniformity is load-bearing for the combine: a worker derives
    # the root's landing address for its partial from `get_write_ptr` on its OWN
    # copy of cb_group_partials, which is only valid because the CB sits at the
    # same L1 offset on every core.
    cbs = []
    for name, index, page_size, num_pages in blk.cb_plan():
        if in_sharded and name == "cb_input_tiles":
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(index, input_tensor))
            continue
        if out_sharded and name == "cb_output_tiles":
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(index, output_tensor))
            continue
        cbs.append(
            ttnn.CBDescriptor(
                total_size=page_size * num_pages,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=index,
                        data_format=_cb_format(name, blk, input_tensor, gamma),
                        page_size=page_size,
                    )
                ],
            )
        )

    # ---------- cross-core combine wiring (Refinement 2) ----------
    # One Mcast2D per combine group: the root broadcasts the finalized
    # mean(x^2) tile back over its rectangle. Disjoint groups reuse the same
    # semaphore ids, so the ids are created ONCE over the whole grid and every
    # group adopts them.
    semaphores = []
    mcasts = {}  # (group index, family) -> Mcast2D
    if placement.w_split:
        semaphores = [ttnn.SemaphoreDescriptor(id=SEM_GATHER, core_ranges=all_cores, initial_value=0)]
        if placement.two_stage:
            semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_GATHER2, core_ranges=all_cores, initial_value=0))
        for f in range(MAX_MCAST_FAMILIES):
            base = SEM_MCAST_BASE + 2 * f
            semaphores.append(ttnn.SemaphoreDescriptor(id=base, core_ranges=all_cores, initial_value=0))
            semaphores.append(ttnn.SemaphoreDescriptor(id=base + 1, core_ranges=all_cores, initial_value=0))
        for g, grp in enumerate(placement.groups):
            for f, (rect, ack) in enumerate(zip(grp["subrects"], grp["acks"])):
                cfg = ttnn.McastConfig(sem_ids=[SEM_MCAST_BASE + 2 * f, SEM_MCAST_BASE + 2 * f + 1])
                rect_set = ttnn.CoreRangeSet([_rect(*rect)])
                mcasts[(g, f)] = ttnn.Mcast2D(device, rect_set, grp["root"], cfg, ack)

    # ---------- EXPERIMENT: the gamma broadcast (the candidate) ----------
    # Predicate: the row-only split (cw == 1) is the regime where gamma is
    # reuse-shared by construction of the split; a W-split (cw > 1) already hands
    # every core a DISJOINT gamma slice, so there is nothing to share. Tiled gamma
    # only (the RM path's landing CB is a stick buffer consumed by compute's own
    # tilize), and only the resident spelling (a broadcast is one shot).
    gamma_fams, gamma_fam_of = [], {}
    gmcasts = {}
    if GAMMA_MODE == "mcast" and gamma is not None and not blk.is_rm_gamma and placement.cw == 1 and blk.gamma_resident:
        plan = _gamma_mcast_plan(placement, _virtual_x_runs(device, grid))
        if plan is not None:
            gamma_fams, gamma_fam_of = plan
    if gamma_fams:
        for f in range(len(gamma_fams)):
            for sid in (GBR_SEM_BASE + 2 * f, GBR_SEM_BASE + 2 * f + 1):
                semaphores.append(ttnn.SemaphoreDescriptor(id=sid, core_ranges=all_cores, initial_value=0))
        for f, (rect, inj, ack) in enumerate(gamma_fams):
            gcfg = ttnn.McastConfig(sem_ids=[GBR_SEM_BASE + 2 * f, GBR_SEM_BASE + 2 * f + 1])
            gmcasts[f] = ttnn.Mcast2D(device, ttnn.CoreRangeSet([_rect(*rect)]), ttnn.CoreCoord(*inj), gcfg, ack)

    LAST_PLAN.clear()
    LAST_PLAN.update(
        mode=GAMMA_MODE,
        late=bool(GAMMA_LATE),
        one_injector=bool(GAMMA_ONE_INJECTOR),
        engaged=bool(gamma_fams),
        gamma_resident=bool(blk.gamma_resident),
        cw=placement.cw,
        cores=len(placement.works),
        families=[f[0] for f in gamma_fams],
        injectors=[f[1] for f in gamma_fams],
        acks=[f[2] for f in gamma_fams],
    )

    # ---------- shared compile-time knobs ----------
    chunk_row_bytes = blk.wt_chunk * TILE_DIM * blk.elem_bytes
    # Real bytes in the FINAL chunk of the core that owns the tensor's last
    # W-tile (every other core's final chunk is a full one).
    last_elems = blk.W - (wt_global - blk.wt_chunk) * TILE_DIM
    last_row_bytes = last_elems * blk.elem_bytes
    g_chunk_row_bytes = blk.wt_chunk * TILE_DIM * blk.gamma_elem_bytes
    g_last_row_bytes = last_elems * blk.gamma_elem_bytes

    regime = [
        1 if blk.is_rm else 0,
        1 if blk.has_gamma else 0,
        1 if blk.is_rm_gamma else 0,
        1 if blk.x_resident else 0,
        1 if blk.gamma_resident else 0,
        1 if blk.has_partial_w else 0,
    ]
    knobs = [blk.Wt, blk.wt_chunk, blk.wt_last, blk.nw, blk.ht_block, x_read_chunks]

    # The reader and the writer share ONE compile-time-arg layout (indices 0..23,
    # the mcast block at 24..28, TensorAccessorArgs from 29). Built once here so
    # a knob added to either kernel cannot drift between the two — the CT index
    # each kernel reads is then guaranteed to name the same quantity.
    # One CT block per multicast family. Every group has the same run split (the
    # shard grid / group rectangle is uniform), so family f's block is uniform.
    mcast_ct = []
    for f in range(MAX_MCAST_FAMILIES):
        mcast_ct += list(mcasts[(0, f)].compile_time_args()) if (0, f) in mcasts else [0, 0, 0, 0, 0]
    dataflow_ct_args = (
        regime
        + knobs
        + [
            blk.w_valid_last,
            chunk_row_bytes,
            last_row_bytes,
            g_chunk_row_bytes,
            g_last_row_bytes,
            blk.total_sticks,
            1 if placement.w_split else 0,
            placement.cw,
            wt_global,
            1 if in_sharded else 0,
            1 if out_sharded else 0,
            SEM_GATHER,
            # Combine topology (Refinement 3): CW1 x CW2 == CW; CW2 == 1 is the
            # flat root gather, CW2 > 1 the two-stage one.
            placement.cw1,
            placement.cw2,
            SEM_GATHER2,
        ]
        + mcast_ct
    )
    # EXPERIMENT tail: [GAMMA_MCAST, GAMMA_ABLATE] then one 5-word mcast block per
    # gamma broadcast family (inactive families emit a zeroed block and compile
    # away in the kernel).
    dataflow_ct_args += [(1 if gamma_fams else 0) | (2 if GAMMA_LATE else 0), 1 if GAMMA_MODE == "ablate" else 0]
    for f in range(MAX_MCAST_FAMILIES):
        dataflow_ct_args += list(gmcasts[f].compile_time_args()) if f in gmcasts else [0, 0, 0, 0, 0]
    DATAFLOW_ACCESSOR_ARG_BASE = len(dataflow_ct_args)  # kernels read TensorAccessorArgs<49>
    assert DATAFLOW_ACCESSOR_ARG_BASE == 49, (
        "gbr: reader/writer read TensorAccessorArgs<49>; the shared CT block "
        f"is now {DATAFLOW_ACCESSOR_ARG_BASE} long — update both kernels together"
    )

    # ---------- reader (NCRISC / NoC0) ----------
    reader_ct_args = list(dataflow_ct_args)
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if gamma is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    src_addr = input_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0
    dst_addr = output_tensor.buffer_address()
    eps_bits = struct.unpack("<I", struct.pack("<f", float(epsilon)))[0]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for w in placement.works:
        core = w.core
        wt_real = blk.Wt if w.wt_real is None else min(w.wt_real, blk.Wt)
        is_root = 0
        root_v = (0, 0)
        leader_v = (0, 0)
        mcast_rt = [0, 0, 0, 0] * MAX_MCAST_FAMILIES
        if placement.w_split:
            root = placement.groups[w.group]["root"]
            is_root = 1 if (int(core.x) == int(root.x) and int(core.y) == int(root.y)) else 0
            rv = device.worker_core_from_logical_core(root)
            root_v = (rv.x, rv.y)
            lv = device.worker_core_from_logical_core(w.leader if w.leader is not None else root)
            leader_v = (lv.x, lv.y)
            mcast_rt = []
            for f in range(MAX_MCAST_FAMILIES):
                mc = mcasts.get((w.group, f))
                mcast_rt += list(mc.runtime_args(core)) if mc is not None else [0, 0, 0, 0]
        # The core whose slice ENDS on the tensor's last W-tile is the one that
        # must mask the tile-padded columns of that tile.
        is_last_w = 1 if (w.wt_start + wt_real == wt_global) else 0

        # EXPERIMENT: this core's gamma-broadcast role. `g_inject` marks the ONE
        # core per run that reads gamma from DRAM and broadcasts it; everybody else
        # in that run's rectangle receives it.
        g_family = gamma_fam_of.get((int(core.x), int(core.y)), 0)
        g_inject = 0
        gamma_rt = [0, 0, 0, 0] * MAX_MCAST_FAMILIES
        if gamma_fams:
            here = (int(core.x), int(core.y))
            g_inject = 0
            for f, (_, inj, _) in enumerate(gamma_fams):
                if here == inj:
                    g_inject |= 1 << f
            gamma_rt = []
            for f in range(MAX_MCAST_FAMILIES):
                mc = gmcasts.get(f)
                gamma_rt += list(mc.runtime_args(core)) if mc is not None else [0, 0, 0, 0]

        reader_rt[core.x][core.y] = (
            [
                src_addr,
                gamma_addr,
                w.start_row,
                w.num_rows,
                w.wt_start,
                w.slot,
                is_root,
                is_last_w,
                wt_real,
                w.family,
                1 if w.is_leader else 0,
            ]
            + mcast_rt
            + [g_inject, g_family]
            + gamma_rt
        )
        writer_rt[core.x][core.y] = [
            dst_addr,
            w.start_row,
            w.num_rows,
            w.wt_start,
            w.s1_slot,
            leader_v[0],
            leader_v[1],
            is_last_w,
            root_v[0],
            root_v[1],
            w.s2_slot,
            1 if w.is_leader else 0,
        ]
        compute_rt[core.x][core.y] = [w.num_rows, eps_bits, is_root, is_last_w, 1 if w.is_leader else 0]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "gbr_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---------- writer (BRISC / NoC1) ----------
    writer_ct_args = list(dataflow_ct_args)
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "gbr_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---------- compute ----------
    # Compute carries its own tail after the shared regime+knobs prefix. New
    # compute-only knobs go at the END, never inside `regime` — the reader and
    # the writer hard-code their CT indices off that same prefix.
    compute_ct_args = (
        regime
        + knobs
        + [
            blk.w_valid_last,
            blk.W,
            1 if placement.w_split else 0,
            placement.cw,
            placement.cw1,
            placement.cw2,
            1 if blk.fuse_sq else 0,
        ]
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "gbr_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=cbs,
    )


def _select_placement(device, grid, input_tensor, ht_total, wt_global, in_sharded):
    """Pick the core assignment: data-pinned when sharded, else the §5 row split
    optionally widened by the cross-core W-split (Lamp L1).

    The W-split is engaged when the INDEPENDENT row axis under-fills the grid —
    it cannot even put one core on every grid row — and the reduced axis is wide
    enough to still hand each core a real chunk after the split. Both gates are
    named constants at the top of this module, never inlined here.
    """
    runs = _virtual_x_runs(device, grid)
    if in_sharded:
        return _placement_sharded(input_tensor, ht_total, wt_global, runs)

    ht_cap = grid.y if W_SPLIT_MAX_HT_FOR_SPLIT is None else W_SPLIT_MAX_HT_FOR_SPLIT
    env = os.environ.get("RMS_NORM_W_SPLIT")
    want_split = (ht_total <= ht_cap) and (wt_global >= W_SPLIT_MIN_WT)
    if env is not None:
        want_split = env != "0"
    if want_split:
        p = _placement_wsplit(grid, ht_total, wt_global, ttnn.tile_size(ttnn.float32), runs)
        if p is not None:
            return p
    return _placement_rows(grid, ht_total)


def _cb_format(name, blk, input_tensor, gamma):
    if name in ("cb_gamma", "cb_gamma_rm"):
        return gamma.dtype if gamma is not None else input_tensor.dtype
    if name == "cb_x_squared":
        # The reduce's input CB — its format IS the reduce datapath. See
        # _Blocking.x_squared_dtype.
        return blk.x_squared_dtype
    if name == "cb_scaler":
        # Must match the format srcB is configured at inside the reduce, i.e.
        # cb_x_squared's; see _Blocking.x_squared_dtype for the full rationale.
        return blk.scaler_dtype
    if name in (
        "cb_partials",
        "cb_rms_sum",
        "cb_rms_recip",
        "cb_ones",
        "cb_group_partials",
        "cb_group_partials2",
        "cb_rms_mean",
        "cb_partial_out",
    ):
        return ttnn.float32
    return input_tensor.dtype
