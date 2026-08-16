# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Blackhole: ``_topk_xl_rebuild_`` at K = 1024 and K = 2048
(``sources/topk_rebuild_xl_perf.cpp``).

WHY
---
``perf_topk_rebuild.py`` priced the K = 512 rebuild at 374 cyc/call and recovered
24 of them (1.069x). The ceiling there was structural: a macro-scheduled Simple
must have ``VD == macroVD``, so only a compare-exchange with BOTH operands
freshly loaded can ride a load -- the first level of a lattice and nothing else.

``canonical_big_block_with_replay<rsf>`` has two sub-blocks whose lattice is
``bitonic_sort_len_k`` -- exactly 4 SFPSWAP, a SINGLE level, storing back to the
addresses it loaded from. That is structurally the merge body, so the merge's
FULL trick applies (SFPSWAP on the macro's Simple slot AND SFPSTORE on its Store
slot). Both sub-blocks are ``if constexpr``'d away at rsf == 1, which is why the
K = 512 work never saw them:

    rsf == 4 (K=2048): sub-blocks A + B + C
    rsf == 2 (K=1024): sub-blocks B + C
    rsf == 1 (K=512) : sub-block C only

READING THE NUMBER
------------------
Identical to ``perf_topk_rebuild.py``: ``loop_factor`` and ``tile_cnt`` are
pinned to 1, so ``mean(MATH_ISOLATE)`` in the .post.csv is the RAW cycle count of
the TILE_LOOP zone, and cycles per body come from a two-point slope over
``rebuild_iter_count`` which cancels the ~30-cycle marker pair and every one-time
cost inside the zone.

PREDICTED vs MEASURED (Blackhole silicon, MATH_ISOLATE, 5 runs/point)
---------------------------------------------------------------------
Every prediction was recorded before the first run. Full table and derivation in
``sources/topk_rebuild_xl_perf.cpp``; the headline rows:

    arm            K=512          K=1024          K=2048
                   pred / meas    pred / meas     pred / meas
    CtrlLoad          - / 0.999      - / 0.999       - / 0.999   cyc/vector
    CtrlSwap          - / 1.999      - / 1.999       - / 1.999   THE TRIPWIRE
    RbCall          374 / 374      822 / 822      1810 / 1810
    RbCallSched     350 / 350      774 / 774      1714 / 1714
    RbCallFull      350 / 350      734 / 734      1554 / 1554
    XlMerge          91 / 91       171 / 171       331 / 331
    MacroMerge       46 / 46        78 / 78        142 / 142
    XlStep            - / 459        - / 987         - / 2135
    FullStep          - / 404        - / 817         - / 1701

      rebuild K=1024 :  822 ->  734 = 1.120x
      rebuild K=2048 : 1810 -> 1554 = 1.165x
      step    K=1024 :  987 ->  817 = 1.208x
      step    K=2048 : 2135 -> 1701 = 1.255x

    sub-block A + B share of the shipping rebuild:
      K=512    0 / 374  =  0.0%   (both `if constexpr`'d away -- RbSubABMacro
                                   measures 4 cycles there, an empty body)
      K=1024 100 / 822  = 12.2%
      K=2048 394 / 1810 = 21.8%

HARNESS VALIDATION. At K = 512 this file reproduces ``perf_topk_rebuild.py``'s
RbCall 374.000 / RbCallSched 350.017 and ``perf_topk_merge_macro.py``'s
XlMerge 91.000 / MacroMerge 46.000 to the cycle.

The parameter classes live in this file rather than in
helpers/test_variant_parameters.py deliberately: they are consumed by exactly one
kernel, and ``TemplateParameter`` is a two-line ABC whose only contract is
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
    RbXposeN = 3
    RbBuild = 4
    RbBlock = 5
    RbSubA = 6
    RbSubB = 7
    RbSubC = 8
    RbSubABMacro = 9
    RbBlockFull = 10
    RbCallSched = 11
    RbCallFull = 12
    # Context: the rebuild is only half of a K-reduction step.
    XlMerge = 13
    MacroMerge = 14
    XlStep = 15
    FullStep = 16


@dataclass
class REBUILD_ARM(TemplateParameter):
    """Emits ``#define REBUILD_ARM <n>``. MUST be a ``#define`` and not a
    ``constexpr``: the kernel guards the symbol with ``#ifndef REBUILD_ARM`` and
    falls back to 2, and a ``constexpr`` leaves the guard unsatisfied -- every
    swept arm would compile as arm 2 while still hashing to a distinct variant
    id, so the sweep would report identical arms with no error anywhere."""

    rebuild_arm: RebuildArm = RebuildArm.RbCall

    def convert_to_cpp(self) -> str:
        return f"#define REBUILD_ARM {self.rebuild_arm.value}"


@dataclass
class REBUILD_ITER_COUNT(TemplateParameter):
    """Number of times the arm's body runs inside the timed region. ``#define``
    for the same reason as ``REBUILD_ARM``."""

    rebuild_iter_count: int = 16

    def convert_to_cpp(self) -> str:
        return f"#define REBUILD_ITER_COUNT {self.rebuild_iter_count}"


@dataclass
class REBUILD_K(TemplateParameter):
    """K for the rebuild under test. ``#define`` for the same reason as above --
    the kernel derives ``row_scale_factor`` from it inside ``constexpr``
    initialisers that a late ``constexpr`` definition would not reach."""

    rebuild_k: int = 1024

    def convert_to_cpp(self) -> str:
        return f"#define REBUILD_K {self.rebuild_k}"


# Divisor that turns cycles-per-body into cycles-per-32-element-vector.
#
#   CtrlLoad / CtrlSwap : 2  -- one replay pass is two instructions.
#   everything else     : K / 32 -- the rebuild rewrites K elements, and every
#                         phase arm touches that same set of vectors once.
#                         Matches perf_topk_rebuild.py's divisor of 16 at K=512.
def vectors_per_body(arm: RebuildArm, k: int) -> int:
    if arm in (RebuildArm.CtrlLoad, RebuildArm.CtrlSwap):
        return 2
    return k // 32


# Two-point slope pairs.
#
# MOP limit: the control arms chunk their passes at 128 per
# ``ckernel_unpack_template::run`` call, because TT_OP_MOP's loop_count field is
# 7 bits while ``count`` is a uint8_t -- passing 256 silently truncates to 0, the
# MOP runs ZERO times, and the arm reads out as a spectacular fake result. The
# control values below are exact multiples of 128.
_CTRL_ITERS = [256, 1024]
# Call-level bodies are 300-1900 cycles; these points put the low zone well above
# the ~30-cycle marker pair and the high zone at ~60k.
_CALL_ITERS = [8, 32]
# Sub-block bodies are 50-500 cycles.
_SUB_ITERS = [16, 64]

_ITER_COUNTS = {
    RebuildArm.CtrlLoad: _CTRL_ITERS,
    RebuildArm.CtrlSwap: _CTRL_ITERS,
    RebuildArm.RbCall: _CALL_ITERS,
    RebuildArm.RbXposeN: _CALL_ITERS,
    RebuildArm.RbBuild: _CALL_ITERS,
    RebuildArm.RbBlock: _CALL_ITERS,
    RebuildArm.RbSubA: _SUB_ITERS,
    RebuildArm.RbSubB: _SUB_ITERS,
    RebuildArm.RbSubC: _SUB_ITERS,
    RebuildArm.RbSubABMacro: _SUB_ITERS,
    RebuildArm.RbBlockFull: _SUB_ITERS,
    RebuildArm.RbCallSched: _CALL_ITERS,
    RebuildArm.RbCallFull: _CALL_ITERS,
    RebuildArm.XlMerge: _SUB_ITERS,
    RebuildArm.MacroMerge: _SUB_ITERS,
    RebuildArm.XlStep: _CALL_ITERS,
    RebuildArm.FullStep: _CALL_ITERS,
}

_ARMS = list(_ITER_COUNTS.keys())
_KS = [512, 1024, 2048]

# Sub-block A only exists at rsf >= 4 (K = 2048); at K = 1024 the arm would
# measure an empty body and the row would be meaningless rather than zero.
_ARMS_BY_K = {
    # K = 512 is rsf == 1: sub-blocks A and B are `if constexpr`'d away, so their
    # arms would time an empty body. Kept as a cross-check that this harness
    # reproduces perf_topk_rebuild.py's published RbCall 374 / RbCallSched 350.
    512: [a for a in _ARMS if a not in (RebuildArm.RbSubA, RebuildArm.RbSubB)],
    1024: [a for a in _ARMS if a is not RebuildArm.RbSubA],
    2048: _ARMS,
}

# Pinned to 1 so the .post.csv carries raw zone cycles -- see READING THE NUMBER.
_LOOP_FACTOR = 1
_TILE_COUNT = 1

# std(...) columns are dropped as structurally empty when there is a single
# sample per marker (helpers/profiler.py::_stats_timings). 5 runs populate std
# and make the run-to-run noise visible next to the deltas being claimed.
_RUN_COUNT = 5

# The kernel unpacks no operand and packs nothing out, but --speed-of-light folds
# every runtime parameter into the build header and takes the
# compile_time_formats path, which dereferences formats_config[0] and the stimuli
# address block. A minimal well-formed pair keeps that path valid.
#
# Float16_b in/out with dest_acc=Yes gives a 32-bit Dest, which is what the fused
# [bf16 value | u16 index] INT32 load/store walk addresses -- and what
# `transpose_dest_face_32b`'s two-pass 16-bit shuffle assumes.
_FORMATS = input_output_formats([DataFormat.Float16_b], same=True)
_DEST_ACC = DestAccumulation.Yes


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=_FORMATS,
    rebuild_k=_KS,
    rebuild_arm=lambda rebuild_k: _ARMS_BY_K[rebuild_k],
    rebuild_iter_count=lambda rebuild_arm: _ITER_COUNTS[rebuild_arm],
)
def test_perf_topk_rebuild_xl(
    perf_report, formats, rebuild_k, rebuild_arm, rebuild_iter_count
):
    configuration = PerfConfig(
        "sources/topk_rebuild_xl_perf.cpp",
        formats,
        # MATH_ISOLATE only. Unpack and pack declare the same two zones (they
        # must, or the three-thread zone barrier deadlocks); unpack additionally
        # issues one `_llk_unpack_set_srcb_dummy_valid_()` per iteration for the
        # transposing arms, which is a handshake and not work, so UNPACK_ISOLATE
        # would time an essentially empty region.
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            REBUILD_ARM(rebuild_arm),
            REBUILD_ITER_COUNT(rebuild_iter_count),
            REBUILD_K(rebuild_k),
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
