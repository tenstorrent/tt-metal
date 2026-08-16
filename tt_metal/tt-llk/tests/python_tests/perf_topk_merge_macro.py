# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Blackhole head-to-head: ``_topk_xl_merge_`` vs an SFPLOADMACRO-scheduled
compare-exchange (``sources/topk_merge_macro_perf.cpp``).

WHY
---
``perf_topk_micro_op.py`` measures ``_topk_xl_merge_<512, false, true>`` at
2.844 cycles per 32-element vector -- the fastest Top-K micro-op in the tree by
a factor of four. Its fused K=512 body is 16 instructions covering 8 input
vectors: 8 SFPLOAD (1 cyc each) + 4 SFPSWAP (2 cyc each) + 4 SFPSTORE (1 cyc
each) = 20 cycles, i.e. 2.500 cyc/vector, plus ~11 cycles of per-call envelope
spread over 32 vectors.

Twelve of those 20 cycles are spent on work one SFPLOADMACRO can carry for
free: SFPSWAP is legal in a macro's Simple slot (SFPLOADMACRO.md:7 (‡)), a
macro-scheduled SFPSTORE writes to the address its load used
(SFPLOADMACRO.md:140) -- which for a merge is exactly where the result belongs
-- and the merge only ever keeps the max, so SFPSWAP's min half is free
collateral rather than a second result needing a route. The candidate body is
therefore 8 instructions for the same 8 vectors. Derivation, operand plumbing
and the cycle-by-cycle collision analysis are in the kernel header.

PREDICTIONS, RECORDED BEFORE THE FIRST RUN
------------------------------------------
    CtrlLoad    1.000   (SFPLOAD is IPC 1; MOP sustains 1 instr/cycle)
    CtrlSwap    2.000   (SFPSWAP.md:110 -- 2 cycles, non-fillable bubble)
    XlCall      2.844   (reproduce perf_topk_micro_op.py in this harness)
    XlBody      2.500   (8*1 + 4*2 + 4*1 = 20 cycles / 8 vectors)
    MacroBody   1.000   (8 instructions / 8 vectors -- the load-issue floor)
    MacroCall   1.438   ((4*8 + 11 + 3) / 32)

Context arms, added after the six above had already landed on their
predictions, to size the merge win against the step that actually ships:

    XlRebuild   -       unpredicted; the K != 2048 generic rebuild body was not
                        instruction-counted by hand.
    XlStep      = XlCall + 2 * XlRebuild + envelope
    MacroStep   = XlStep - (XlCall - MacroCall) = XlStep - 1.406

READING THE NUMBER
------------------
``postprocess_tile_loop`` (helpers/perf/core.py) divides the TILE_LOOP row's
``mean(...)``/``std(...)`` columns by ``loop_factor * tile_cnt``. Both are pinned
to 1 here, so the value in ``mean(MATH_ISOLATE)`` of the .post.csv is the RAW
cycle count of the whole TILE_LOOP zone -- not a per-tile figure. These kernels
have no tile loop: their work unit is a 32-element vector and the count of those
is a compile-time ``#define``, so there is nothing the tile/loop divisor could
legitimately represent.

Cycles per 32-element vector comes from a two-point slope over
``merge_iter_count``, per arm:

    cyc_per_vector = (mean@hi - mean@lo) / (hi - lo) / VECTORS_PER_BODY[arm]

The subtraction cancels the ~30-cycle START_PERF_MEASURE marker pair
(``test_profiler_overhead.py`` asserts 30 +/- 5 on Blackhole) and every one-time
cost inside the zone: the SFPU init, ``_topk_xl_init_``'s ADDR_MOD writes, the
macro SFPCONFIG programming, and the MOP template push. What survives is the
steady-state marginal cost of one more body, which is what a tile loop pays.

The parameter classes live in this file rather than in
helpers/test_variant_parameters.py deliberately: they are consumed by exactly
one kernel, and ``TemplateParameter`` is a two-line ABC whose only contract is
``convert_to_cpp``.
"""

from dataclasses import dataclass
from enum import Enum

import pytest
from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    TILE_COUNT,
    TemplateParameter,
)


class MergeArm(Enum):
    """Arm of the merge head-to-head. The value IS the integer the kernel's
    ``MERGE_ARM`` preprocessor comparison expects, so it is emitted verbatim."""

    CtrlLoad = 0
    CtrlSwap = 1
    XlCall = 2
    XlBody = 3
    MacroBody = 4
    MacroCall = 5
    # Context: the merge is only half of a K-reduction step. `_topk_xl_merge_`
    # leaves the survivors ordered but not bitonic, so every merge must be
    # followed by `_topk_xl_rebuild_` before the next one can run.
    XlRebuild = 6
    XlStep = 7
    MacroStep = 8


# Distinct 32-element input vectors one execution of an arm's body consumes.
# The divisor that turns the two-point slope (cycles per body) into the
# comparable unit, cycles per 32-element vector.
#
#   CtrlLoad / CtrlSwap : 2 -- one replay pass is two instructions.
#   XlCall / MacroCall  : 32 -- 4 body iters x 8 vectors, matching
#                         TopKPerfArm.XlMerge's divisor in helpers/llk_params.py
#                         so the rows are directly comparable to the 2.844.
#   XlBody / MacroBody  : 8 -- one body iter.
VECTORS_PER_BODY = {
    MergeArm.CtrlLoad: 2,
    MergeArm.CtrlSwap: 2,
    MergeArm.XlCall: 32,
    MergeArm.XlBody: 8,
    MergeArm.MacroBody: 8,
    MergeArm.MacroCall: 32,
    # `_topk_xl_rebuild_` rewrites K = 512 elements = 16 vectors. Normalising by
    # that makes its cyc/vector directly ADDABLE to a merge's, since a merge
    # consumes 2K = 32 vectors and emits the K the rebuild then rewrites.
    MergeArm.XlRebuild: 16,
    # Both step arms are normalised by the MERGE's 32 input vectors, so the two
    # rows differ by exactly the merge delta.
    MergeArm.XlStep: 32,
    MergeArm.MacroStep: 32,
}


@dataclass
class MERGE_ARM(TemplateParameter):
    """Select the arm.

    Emits ``#define MERGE_ARM <n>``. MUST be a ``#define`` and not a
    ``constexpr``: the kernel guards the symbol with ``#ifndef MERGE_ARM`` and
    falls back to 5, and a ``constexpr`` leaves the guard unsatisfied -- every
    swept arm would compile as arm 5 while still hashing to a distinct variant
    id, so the sweep would report six identical arms with no error anywhere.
    """

    merge_arm: MergeArm = MergeArm.MacroCall

    def convert_to_cpp(self) -> str:
        return f"#define MERGE_ARM {self.merge_arm.value}"


@dataclass
class MERGE_ITER_COUNT(TemplateParameter):
    """Number of times the arm's body runs inside the timed region.

    Emits ``#define MERGE_ITER_COUNT <n>``, a ``#define`` for the same reason as
    ``MERGE_ARM``: the kernel's ``#ifndef`` fallback is 32, so a ``constexpr``
    would collapse both slope points onto one and make the slope meaningless.
    """

    merge_iter_count: int = 32

    def convert_to_cpp(self) -> str:
        return f"#define MERGE_ITER_COUNT {self.merge_iter_count}"


# Two-point slope pairs, per arm. Both points must be large enough that the
# ~30-cycle marker pair is a small fraction of the zone, and the low point large
# enough that the slope is not dominated by its own noise.
#
# MOP limit: the *_BODY and control arms chunk their passes at 128 per
# ``ckernel_unpack_template::run`` call, because TT_OP_MOP's loop_count field is
# 7 bits (count - 1 <= 127) while the ``count`` parameter is a ``uint8_t`` --
# passing 256 silently truncates to 0, the MOP runs ZERO times, and the arm
# reads out as a spectacular fake result rather than an error. Every value below
# is either <= 128 or an exact multiple of it, so the chunking overhead is the
# same fraction at both points and cancels in the slope.
_ITER_COUNTS = {
    # Replay passes (2 vectors each) -> 512 and 2048 vectors, matching
    # perf_sfpu_count_above.py and perf_topk_micro_op.py exactly so the shared
    # controls line up across all three harnesses.
    MergeArm.CtrlLoad: [256, 1024],
    MergeArm.CtrlSwap: [256, 1024],
    # Full merge calls; same points as perf_topk_micro_op.py's XlMerge row.
    MergeArm.XlCall: [32, 128],
    MergeArm.MacroCall: [32, 128],
    # Bare bodies: 16 and 8 instructions respectively, so more iters are needed
    # for the zone to dominate the marker pair.
    MergeArm.XlBody: [128, 512],
    MergeArm.MacroBody: [128, 512],
    # Rebuild and full steps are expensive; fewer iters keep the zone sane.
    MergeArm.XlRebuild: [16, 64],
    MergeArm.XlStep: [16, 64],
    MergeArm.MacroStep: [16, 64],
}

_ARMS = list(_ITER_COUNTS.keys())

# Pinned to 1 so the .post.csv carries raw zone cycles -- see READING THE NUMBER.
_LOOP_FACTOR = 1
_TILE_COUNT = 1

# std(...) columns are dropped as structurally empty when there is a single
# sample per marker (helpers/profiler.py::_stats_timings), and a slope taken
# across arms with no spread is not defensible. 5 runs populate std and make the
# run-to-run noise visible next to the ~2.8 vs ~1.0 cycle/vector gap the
# candidate is claiming.
_RUN_COUNT = 5

# The kernel unpacks no operand and packs nothing out, but --speed-of-light
# folds every runtime parameter into the build header and takes the
# compile_time_formats path, which dereferences formats_config[0] and the
# stimuli address block. A minimal well-formed pair keeps that path valid.
#
# Float16_b in/out with dest_acc=Yes gives a 32-bit Dest, which is what the
# fused [bf16 value | u16 index] INT32 load/store walk addresses.
_FORMATS = input_output_formats([DataFormat.Float16_b], same=True)
_DEST_ACC = DestAccumulation.Yes


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=_FORMATS,
    merge_arm=_ARMS,
    merge_iter_count=lambda merge_arm: _ITER_COUNTS[merge_arm],
)
def test_perf_topk_merge_macro(perf_report, formats, merge_arm, merge_iter_count):
    configuration = PerfConfig(
        "sources/topk_merge_macro_perf.cpp",
        formats,
        # MATH_ISOLATE only. Unpack and pack declare the same two zones (they
        # must, or the three-thread zone barrier deadlocks) but do no work, so
        # UNPACK_ISOLATE / PACK_ISOLATE would time an empty region,
        # L1_CONGESTION has no L1 traffic to congest, and L1_TO_L1 pairs an
        # unpack ZONE_START with a pack ZONE_END across threads that never move
        # data -- it would raise in helpers/profiler.py::_stats_l1_to_l1.
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MERGE_ARM(merge_arm),
            MERGE_ITER_COUNT(merge_iter_count),
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
        dest_acc=_DEST_ACC,
        compile_time_formats=True,
    )

    configuration.run(perf_report, run_count=_RUN_COUNT)
