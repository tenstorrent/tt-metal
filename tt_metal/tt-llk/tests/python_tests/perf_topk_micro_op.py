# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Blackhole issue-rate baseline for the SHIPPING Top-K SFPU micro-ops
(``sources/topk_micro_op_perf.cpp``).

WHY
---
A candidate threshold-selection inner loop (``perf_sfpu_count_above.py``,
``CountArm.CountD1``) measures 2.000 cycles per 32-element vector on this
Blackhole. Nothing in the shipping Top-K path had ever been measured in the same
units, so there was no baseline to compare against. This file is that baseline.

Seven arms are swept in ONE pytest invocation so every row comes from the same
session -- same silicon, same clock, same profiler build, same ELF toolchain --
and is therefore directly comparable both to its siblings and (because the two
control arms are shared verbatim with ``perf_sfpu_count_above.py``) to the
candidate:

  TopKPerfArm.CtrlLoad   control. Replay+MOP-fed plain SFPLOAD stream. Must
                         measure ~1.0 cyc/vector -- the frontend issue-rate
                         floor. If it does not, the instruction feed path is
                         still the limiter and NOTHING else here is
                         interpretable.
  TopKPerfArm.CtrlSwap   control. Replay+MOP-fed plain SFPSWAP stream. MUST be
                         ~2.0x CtrlLoad: SFPSWAP is documented at 2 backend
                         cycles with a hardware-inserted, non-fillable bubble
                         (SFPSWAP.md:110). This is the tripwire -- the one arm
                         whose answer is known independently of anything being
                         measured, and the RIGHT control for the topk arms,
                         which are all SFPSWAP lattices.
  TopKPerfArm.LocalSort  _bitonic_topk_phases_steps -- ttnn.topk's
                         ``topk_local_sort`` (its compute kernel calls it with
                         end_phase = 5; topk_test.cpp uses TOPK_LOGK - 1 = 4).
  TopKPerfArm.Merge      _bitonic_topk_merge.
  TopKPerfArm.GmgTop8    bitonic_top8_ph0_to_ph3 -- the 25-instruction
                         generalized-MoE-gate single-face micro-op. THE ONE THE
                         CANDIDATE HAS TO BEAT.
  TopKPerfArm.GmgTop8Ls  the same micro-op inside its load16/store8 envelope.
  TopKPerfArm.XlMerge    _topk_xl_merge_<512, false, true>.

READING THE NUMBER
------------------
``postprocess_tile_loop`` (helpers/perf/core.py) divides the TILE_LOOP row's
``mean(...)``/``std(...)`` columns by ``loop_factor * tile_cnt``. Both are pinned
to 1 here, so the value in ``mean(MATH_ISOLATE)`` of the .post.csv is the RAW
cycle count of the whole TILE_LOOP zone -- not a per-tile figure. These kernels
have no tile loop: their work unit is a 32-element vector and the count of those
is a compile-time ``#define``, so there is nothing the tile/loop divisor could
legitimately represent.

Cycles per 32-element vector therefore comes from a two-point slope over
``topk_perf_iter_count``, per (arm, configuration):

    cyc_per_vector = (mean@hi - mean@lo) / (hi - lo) / arm.vectors_per_body

The subtraction cancels the ~30-cycle START_PERF_MEASURE marker pair
(``test_profiler_overhead.py`` asserts 30 +/- 5 on Blackhole) AND every one-time
cost inside the zone: the SFPU init, the ADDR_MOD writes, the MOP programming,
and -- for LocalSort specifically -- the first invocation's three
``load_replay_buf<Exec>`` recordings, which ``topk_replay_init`` suppresses on
every later call. What survives the subtraction is the steady-state marginal
cost of one more invocation, which is exactly what a tile loop pays.

``vectors_per_body`` (``TopKPerfArm.vectors_per_body``) is the number of
DISTINCT 32-element input vectors one body consumes, NOT the number of loads it
issues: a bitonic sort revisits its data once per phase, so loads far exceed
data. Per-arm derivations are in helpers/llk_params.py and in the kernel header.

ITER_COUNT CHOICE
-----------------
Per-arm, because the bodies differ by three orders of magnitude in cost (one
LocalSort call is thousands of cycles; one GmgTop8 call is tens). Both points of
every pair are large enough that the ~30-cycle marker pair is a small fraction
of the zone, and the low point is large enough that a slope across the pair is
not dominated by its own noise.

For the two control arms, ``topk_perf_iter_count`` counts REPLAY PASSES, and the
kernel chunks them at 128 per ``ckernel_unpack_template::run`` call:
``TT_OP_MOP``'s loop_count field is 7 bits (count - 1 <= 127) while the ``count``
parameter is a ``std::uint8_t``, so passing 256 silently truncates to 0, the MOP runs
ZERO times, and the arm reads out as a spectacular fake result rather than as an
error. 256 and 1024 are exact multiples of 128, so the chunking overhead is the
same fraction at both points and cancels in the slope.
"""

import pytest
from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    PerfRunType,
    StableSort,
    TopKPerfArm,
    TopKSortDirection,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    TILE_COUNT,
    TOPK_MICRO_OP,
)

# Two-point slope pairs, per arm. See ITER_COUNT CHOICE above.
_ITER_COUNTS = {
    # Replay passes (2 vectors each) -> 512 and 2048 vectors, matching
    # perf_sfpu_count_above.py's points exactly so the controls line up.
    TopKPerfArm.CtrlLoad: [256, 1024],
    TopKPerfArm.CtrlSwap: [256, 1024],
    # ~3.6k SFPU instructions per call at end_phase 5.
    TopKPerfArm.LocalSort: [2, 8],
    # ~300 SFPU instructions per call.
    TopKPerfArm.Merge: [4, 16],
    # 25 instructions per call.
    TopKPerfArm.GmgTop8: [64, 256],
    # 25 + 8 loads + 6 stores = 39 instructions per call.
    TopKPerfArm.GmgTop8Ls: [64, 256],
    # 4 body iters x 16 instructions + 2 MOP issues.
    TopKPerfArm.XlMerge: [32, 128],
}

_ARMS = list(_ITER_COUNTS.keys())


def _end_phases(topk_arm):
    """i_end_phase sweep. Only LocalSort reads it.

    5 is what ttnn's topk compute kernel passes; 4 is what topk_test.cpp passes
    (TOPK_LOGK - 1 at K = 32). Both are shipping call sites, and phases 4 and 5
    take the `default:` branch of the phase switch -- a different, load/store
    heavier code path than phases 0-3 -- so the pair also prices that branch.
    """
    return [4, 5] if topk_arm is TopKPerfArm.LocalSort else [5]


def _m_iters(topk_arm):
    """_bitonic_topk_merge's merge-tree level. Only Merge reads it.

    m_iter sets `dist`, hence the operand stride, and (via 64 >> m_iter) the
    trip count -- so it is a genuine perf axis, not a label.
    """
    return [0, 1] if topk_arm is TopKPerfArm.Merge else [0]


def _k_values(topk_arm):
    """_bitonic_topk_merge's k. Only Merge reads it.

    32 is what ttnn.topk and topk_test.cpp use. 8 halves `inner_d` but doubles
    the outer trip count, so it isolates loop-structure cost from work.
    """
    return [8, 32] if topk_arm is TopKPerfArm.Merge else [32]


def _sort_dirs(topk_arm):
    """Sort direction -- a real instruction-count axis on every arm that reads it.

    * GmgTop8 / GmgTop8Ls: bitonic_top8_ph3_st4_to_1 branches on it into two
      DIFFERENT 25-instruction lattices (the ArgMax arm vs its else arm).
    * LocalSort: bitonic_topk_ph3_st4_to_1 brackets its replay with
      SFPCONFIG(0x104) + 2 SFPNOP / SFPCONFIG(0x004) + 2 SFPNOP when
      dir == ArgMin, to flip SWAP's max/min sense -- 6 extra instructions per
      phase-3 call, and phase 3 runs 4x per (face, col).
    * Merge: selects `top_min`, which only swaps the SFPSWAP operand order --
      same instruction count, measured anyway so the row is unambiguous.

    The two control arms have no direction; they are pinned to one value so the
    sweep does not emit two identical variants of each.
    """
    if topk_arm in (TopKPerfArm.CtrlLoad, TopKPerfArm.CtrlSwap, TopKPerfArm.XlMerge):
        return [TopKSortDirection.Descending]
    return [TopKSortDirection.Descending, TopKSortDirection.Ascending]


def _stables(topk_arm):
    """STABLE_SORT template argument. Only the two _bitonic_topk_* arms read it.

    It is the one axis that changes the LENGTH of the recorded compare lattices
    -- phase 0 goes 4 -> 6 slots, phase 1 6 -> 10, phase 2 9 -> 14, phase 3
    5 -> 9, and merge gains a second SFPSWAP on the same LREG pair (an
    acknowledged 1-cycle stall). So it is priced, not flagged.

    Measured for cost only. test_topk.py skips stable sort as functionally
    broken in the LLK API (tenstorrent/tt-metal#33492); a timing number for a
    fixed instruction lattice is still well defined, and the lattice is what
    would ship once the correctness bug is fixed.
    """
    if topk_arm in (TopKPerfArm.LocalSort, TopKPerfArm.Merge):
        return [StableSort.No, StableSort.Yes]
    return [StableSort.No]


# The kernel touches LRegs and DEST only: no operand is unpacked, nothing is
# packed out, and no format conversion happens in the timed region. But
# --speed-of-light (which CI uses) folds every runtime parameter into the build
# header and takes the compile_time_formats path, which dereferences
# formats_config[0] and the stimuli address block -- both None if `formats` /
# `variant_stimuli` are omitted. A minimal well-formed pair keeps that path
# valid.
#
# Float16_b matches the shipping topk configuration: test_topk.py runs
# Float16_b with dest_acc=No, which is the 16-bit DEST the _bitonic_topk_*
# load/store helpers address.
_FORMATS = input_output_formats([DataFormat.Float16_b], same=True)

# Pinned to 1 so the .post.csv carries raw zone cycles -- see READING THE NUMBER.
_LOOP_FACTOR = 1
_TILE_COUNT = 1

# std(...) columns are dropped as structurally empty when there is a single
# sample per marker (helpers/profiler.py::_stats_timings), and a slope taken
# between two points with no spread is not defensible. 5 runs populate std and
# put the run-to-run noise next to the numbers it qualifies.
_RUN_COUNT = 5


def _dest_acc(topk_arm: TopKPerfArm) -> DestAccumulation:
    """32-bit DEST only where the op requires it.

    topk_xl's fused path packs value|index into one 32-bit DEST word and loads
    it with InstrModLoadStore::INT32, so it needs dest_acc; test_topk_xl.py sets
    DestAccumulation.Yes for exactly that reason. Every other arm mirrors
    test_topk.py / the gate, which run 16-bit DEST.
    """
    return (
        DestAccumulation.Yes if topk_arm is TopKPerfArm.XlMerge else DestAccumulation.No
    )


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=_FORMATS,
    topk_arm=_ARMS,
    end_phase=_end_phases,
    m_iter=_m_iters,
    k=_k_values,
    sort_dir=_sort_dirs,
    stable=_stables,
    iter_count=lambda topk_arm: _ITER_COUNTS[topk_arm],
)
def test_perf_topk_micro_op(
    perf_report, formats, topk_arm, end_phase, m_iter, k, sort_dir, stable, iter_count
):
    configuration = PerfConfig(
        "sources/topk_micro_op_perf.cpp",
        formats,
        # MATH_ISOLATE only. Unpack and pack declare the same two zones (they
        # must, or the three-thread zone barrier deadlocks) but do no work, so
        # UNPACK_ISOLATE / PACK_ISOLATE would time an empty region,
        # L1_CONGESTION has no L1 traffic to congest, and L1_TO_L1 pairs an
        # unpack ZONE_START with a pack ZONE_END across threads that never move
        # data -- it raises in helpers/profiler.py::_stats_l1_to_l1.
        run_types=[PerfRunType.MATH_ISOLATE],
        # Everything is a compile-time template: the kernel reads no runtime
        # parameters, and the #define params must reach the preprocessor before
        # the kernel's #ifndef guards (build.h is included via params.h, above
        # the kernel's parameter block).
        templates=[
            TOPK_MICRO_OP(
                topk_perf_arm=topk_arm,
                topk_perf_iter_count=iter_count,
                topk_perf_end_phase=end_phase,
                topk_perf_m_iter=m_iter,
                topk_perf_k=k,
                topk_perf_sort_dir=sort_dir,
                topk_perf_stable=stable,
            ),
            TILE_COUNT(_TILE_COUNT),
            LOOP_FACTOR(_LOOP_FACTOR),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=_TILE_COUNT,
            tile_count_B=_TILE_COUNT,
            tile_count_res=_TILE_COUNT,
        ),
        unpack_to_dest=False,
        dest_acc=_dest_acc(topk_arm),
        compile_time_formats=True,
    )

    configuration.run(perf_report, run_count=_RUN_COUNT)
