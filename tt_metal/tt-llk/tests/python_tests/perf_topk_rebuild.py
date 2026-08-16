# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Blackhole: pricing ``_topk_xl_rebuild_`` (``sources/topk_rebuild_perf.cpp``).

WHY
---
``perf_topk_merge_macro.py`` left the shipping K-reduction step at

    _topk_xl_merge_<512>     91.0 cyc/call    2.844 cyc/vector
    _topk_xl_rebuild_<512>  374.0 cyc/call   23.375 cyc/vector
    step                    464.0 cyc/call   14.500 cyc/vector

The merge was then beaten 1.978x, which moved the step only 1.107x: the rebuild
is 374 of 464 cycles (81%). This harness decomposes the rebuild into its three
phases plus its never-priced Dst-transpose content, and prices two candidates.

READING THE NUMBER
------------------
``postprocess_tile_loop`` (helpers/perf/core.py) divides the TILE_LOOP row's
``mean(...)``/``std(...)`` columns by ``loop_factor * tile_cnt``. Both are pinned
to 1, so ``mean(MATH_ISOLATE)`` in the .post.csv is the RAW cycle count of the
whole TILE_LOOP zone. Cycles per body come from a two-point slope over
``rebuild_iter_count``:

    cyc_per_body = (mean@hi - mean@lo) / (hi - lo)

which cancels the ~30-cycle START_PERF_MEASURE marker pair and every one-time
cost inside the zone (SFPU init, ``_topk_xl_init_``, the macro SFPCONFIG
programming). What survives is the steady-state marginal cost of one more body.

PREDICTED vs MEASURED (Blackhole silicon, MATH_ISOLATE, 5 runs/point)
--------------------------------------------------------------------
    arm            cyc/call   cyc/vec   PREDICTED
    CtrlLoad          1.997     0.999     1.000   control, frontend floor
    CtrlSwap          3.997     1.999     2.000   THE TRIPWIRE, 2.00x CtrlLoad
    RbCall          374.000    23.375   374       reproduces perf_topk_merge_macro
    RbXposeFace      44.000     2.750    39       one 16x16 32-bit face + bracket
    RbXposeN        143.000     8.938   132       the rebuild's WHOLE transpose
    RbBuild         102.000     6.375   111       stride-2 + sort_16_alt, 2 iters
    RbBlock         120.000     7.500   120       2 x canonical_big_block<1>
    RbXposeNFlat    197.000    12.312   122       REFUTED, +54 not -21
    RbXposeNFill    142.996     8.937   143       24 SFPNOPs cost EXACTLY ZERO
    RbBuildMacro     90.000     5.625    90       -12 (4 of 16 swaps/iter x 2)
    RbBlockMacro    108.000     6.750   108       -12 (4 of 20 swaps/col x 2)
    RbCallMacro     367.025    22.939   350       template swap NOT hidden
    RbCallSched     350.000    21.875   350       THE CANDIDATE

    rebuild: 374 -> 350 = 1.069x
    step:    (91 merge + 374) = 465 -> (46 macro merge + 350) = 396 = 1.174x

WHERE THE 374 CYCLES ARE
------------------------
RbXposeN + RbBuild + RbBlock = 143 + 102 + 120 = 365 against RbCall's 374; the
missing 9 is the second MOP template program plus the SETRWCs only the full call
pays. So the decomposition is complete:

    Dst transposes   143 cyc  38%   4 x transpose_dest_face_32b + CFG block
    lattice + ld/st  222 cyc  59%   72 SFPSWAP + 32 SFPLOAD + 32 SFPSTORE
                                    + 8 SFPTRANSP + replay/MOP pushes
    envelope           9 cyc   2%

and the SFPU half is AT ITS FLOOR for the algorithm: 9 bitonic levels x 8 vector
compare-exchanges = exactly 72 SFPSWAP at 2 cyc, and 16 vectors read and written
twice = 64 load/stores, which is the minimum for 9 levels with only 8 LRegs.

THE ~1.4x MACRO BOUND IS REFUTED; THE REAL CEILING IS 1.069x
------------------------------------------------------------
The standing estimate was that the merge's macro trick could hide "half of the
16 SFPSWAPs behind 8 loads". It hides FOUR, not eight, and the reason is
structural: a macro-scheduled Simple instruction must have VD == macroVD, the
register that macro's own load just wrote, so only a compare-exchange whose
BOTH operands are freshly loaded can ride a load. That is the first level of a
lattice and nothing else -- level 2 reads level 1's outputs. One level per
load-pass, 4 swaps, 8 cycles, minus 2 cycles of drain SFPNOP (consecutive
macros must be two issue slots apart, so the last macro's SFPSWAP is still
holding the Simple sub-unit when the next software SFPSWAP wants it) = 6 cycles
per body. Four bodies per rebuild = 24 of 374.

WHAT IS NOT RECOVERABLE, AND WHY
--------------------------------
The 143 cycles of Dst transpose are not a scheduling artefact. Of the rebuild's
9 levels, 3 fall on the 8-lane axis, and NO Vector Unit instruction moves data
along it: SFPSWAP compares lane-for-lane, SFPTRANSP swaps the register-index
axis with the row axis (both length 4), and SFPSHFT2 rotates by +/-1 within a
group of 8. Only the Matrix Unit face transpose turns Dst columns into Dst rows,
and the data must visit the transposed domain and come back, so two sweeps are
load-bearing. Within a face the sequence is also at ITS floor: MOVD2B and
MOVB2A/MOVB2D top out at MOV_4_ROWS, so 16 rows x 2 passes (Dst is 32-bit,
SrcB is not) is 20 Matrix Unit issues, and RbXposeNFlat proves that hoisting the
CFG writes out of the per-face loop makes it WORSE, not better.

The 44 - 25 = ~9 cycles per face that are neither issue slots nor CFG writes are
the MOVD2B -> TRNSPSRCB stalls, and RbXposeNFill shows they are fillable from
the Vector Unit at zero cost -- 24 free SFPU issue slots per rebuild. RbCallSched
spends 4 of them on the InstructionTemplate rewrites that the phase switch needs,
which is the difference between RbCallMacro (367, switch paid for) and
RbCallSched (350, switch free).

The parameter classes live in this file rather than in
helpers/test_variant_parameters.py deliberately: they are consumed by exactly
one kernel, and ``TemplateParameter`` is a two-line ABC whose only contract is
``convert_to_cpp``. Same rationale as perf_topk_merge_macro.py.
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


class RebuildArm(Enum):
    """Arm of the rebuild decomposition. The value IS the integer the kernel's
    ``REBUILD_ARM`` preprocessor comparison expects, so it is emitted verbatim."""

    CtrlLoad = 0
    CtrlSwap = 1
    RbCall = 2
    RbXposeFace = 3
    RbXposeN = 4
    RbBuild = 5
    RbBlock = 6
    RbXposeNFlat = 7
    RbBuildMacro = 8
    RbBlockMacro = 9
    RbXposeNFill = 10
    RbCallMacro = 11
    RbCallSched = 12


# Divisor that turns cycles-per-body into cycles-per-32-element-vector.
#
#   CtrlLoad / CtrlSwap : 2  -- one replay pass is two instructions.
#   everything else     : 16 -- `_topk_xl_rebuild_<512>` rewrites K = 512
#                         elements = 16 vectors, and every phase arm below
#                         touches that same 16 vectors exactly once. Matching
#                         MergeArm.XlRebuild's divisor in
#                         perf_topk_merge_macro.py keeps the rows directly
#                         comparable to the 23.375.
VECTORS_PER_BODY = {
    RebuildArm.CtrlLoad: 2,
    RebuildArm.CtrlSwap: 2,
    RebuildArm.RbCall: 16,
    # One face is 256 datums = 8 vectors, and it is transposed twice per
    # rebuild, so a per-face row normalised by 16 is directly addable to the
    # others.
    RebuildArm.RbXposeFace: 16,
    RebuildArm.RbXposeN: 16,
    RebuildArm.RbBuild: 16,
    RebuildArm.RbBlock: 16,
    RebuildArm.RbXposeNFlat: 16,
    RebuildArm.RbBuildMacro: 16,
    RebuildArm.RbBlockMacro: 16,
    RebuildArm.RbXposeNFill: 16,
    RebuildArm.RbCallMacro: 16,
    RebuildArm.RbCallSched: 16,
}


@dataclass
class REBUILD_ARM(TemplateParameter):
    """Select the arm.

    Emits ``#define REBUILD_ARM <n>``. MUST be a ``#define`` and not a
    ``constexpr``: the kernel guards the symbol with ``#ifndef REBUILD_ARM``
    and falls back to 2, and a ``constexpr`` leaves the guard unsatisfied --
    every swept arm would compile as arm 2 while still hashing to a distinct
    variant id, so the sweep would report ten identical arms with no error
    anywhere.
    """

    rebuild_arm: RebuildArm = RebuildArm.RbCall

    def convert_to_cpp(self) -> str:
        return f"#define REBUILD_ARM {self.rebuild_arm.value}"


@dataclass
class REBUILD_ITER_COUNT(TemplateParameter):
    """Number of times the arm's body runs inside the timed region.

    ``#define`` for the same reason as ``REBUILD_ARM``: the kernel's ``#ifndef``
    fallback is 32, so a ``constexpr`` would collapse both slope points onto one
    and make the slope meaningless.
    """

    rebuild_iter_count: int = 32

    def convert_to_cpp(self) -> str:
        return f"#define REBUILD_ITER_COUNT {self.rebuild_iter_count}"


# Two-point slope pairs, per arm.
#
# MOP limit: the control arms chunk their passes at 128 per
# ``ckernel_unpack_template::run`` call, because TT_OP_MOP's loop_count field is
# 7 bits (count - 1 <= 127) while the ``count`` parameter is a ``uint8_t`` --
# passing 256 silently truncates to 0, the MOP runs ZERO times, and the arm
# reads out as a spectacular fake result rather than an error. The control
# values below are exact multiples of 128.
_ITER_COUNTS = {
    RebuildArm.CtrlLoad: [256, 1024],
    RebuildArm.CtrlSwap: [256, 1024],
    # Rebuild-scale bodies are expensive (~40-380 cycles each); these points put
    # the low zone at >= 600 cycles and the high at ~24k, both far above the
    # ~30-cycle marker pair.
    RebuildArm.RbCall: [16, 64],
    RebuildArm.RbXposeFace: [16, 64],
    RebuildArm.RbXposeN: [16, 64],
    RebuildArm.RbBuild: [16, 64],
    RebuildArm.RbBlock: [16, 64],
    RebuildArm.RbXposeNFlat: [16, 64],
    RebuildArm.RbBuildMacro: [16, 64],
    RebuildArm.RbBlockMacro: [16, 64],
    RebuildArm.RbXposeNFill: [16, 64],
    RebuildArm.RbCallMacro: [16, 64],
    RebuildArm.RbCallSched: [16, 64],
}

_ARMS = list(_ITER_COUNTS.keys())

# Pinned to 1 so the .post.csv carries raw zone cycles -- see READING THE NUMBER.
_LOOP_FACTOR = 1
_TILE_COUNT = 1

# std(...) columns are dropped as structurally empty when there is a single
# sample per marker (helpers/profiler.py::_stats_timings). 5 runs populate std
# and make the run-to-run noise visible next to the deltas being claimed.
_RUN_COUNT = 5

# The kernel unpacks no operand and packs nothing out, but --speed-of-light
# folds every runtime parameter into the build header and takes the
# compile_time_formats path, which dereferences formats_config[0] and the
# stimuli address block. A minimal well-formed pair keeps that path valid.
#
# Float16_b in/out with dest_acc=Yes gives a 32-bit Dest, which is what the
# fused [bf16 value | u16 index] INT32 load/store walk addresses -- and what
# `transpose_dest_face_32b`'s two-pass 16-bit shuffle assumes.
_FORMATS = input_output_formats([DataFormat.Float16_b], same=True)
_DEST_ACC = DestAccumulation.Yes


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=_FORMATS,
    rebuild_arm=_ARMS,
    rebuild_iter_count=lambda rebuild_arm: _ITER_COUNTS[rebuild_arm],
)
def test_perf_topk_rebuild(perf_report, formats, rebuild_arm, rebuild_iter_count):
    configuration = PerfConfig(
        "sources/topk_rebuild_perf.cpp",
        formats,
        # MATH_ISOLATE only. Unpack and pack declare the same two zones (they
        # must, or the three-thread zone barrier deadlocks); unpack additionally
        # issues one `_llk_unpack_set_srcb_dummy_valid_()` per iteration for the
        # transposing arms, which is a handshake and not work, so
        # UNPACK_ISOLATE would time an essentially empty region.
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            REBUILD_ARM(rebuild_arm),
            REBUILD_ITER_COUNT(rebuild_iter_count),
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
