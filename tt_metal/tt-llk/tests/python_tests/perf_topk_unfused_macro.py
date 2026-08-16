# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Blackhole: the UNFUSED ``_topk_xl_merge_`` / ``_topk_xl_rebuild_`` / step,
macro vs shipping bodies, as PAIRED ARMS of one kernel
(``sources/topk_unfused_macro_perf.cpp``).

WHY
---
``ckernel_sfpu_topk_xl.h`` now macro-schedules the unfused merge body and the
unfused rebuild's single-level stride bodies by default. This driver builds
every timing arm TWICE — once as-is (macro) and once with
``DISABLE_TOPK_XL_SFPLOADMACRO`` (the byte-identical shipping bodies) — so the
pair differs by exactly the header code under test, on the same silicon, in
the same run.

PREDICTION (recorded before the first run): the unfused body goes from 18
issues + 2 software SFPSWAPs (2 cycles each, plus the index-tracking stall)
~= 20-22 cycles to 16 single-cycle issues, i.e. 1.25-1.38x per body.
  merge   : every body macro'd            -> ~1.15-1.35x per call
  rebuild : only stride-64/32/16 bodies   -> ~1.05-1.15x per call
  step    : between the two, rebuild-weighted

READING THE NUMBER
------------------
Identical to ``perf_topk_rebuild_xl.py``: ``loop_factor`` and ``tile_cnt`` are
pinned to 1 so ``mean(MATH_ISOLATE)`` in the .post.csv is the raw cycle count
of the TILE_LOOP zone; cycles per call come from the two-point slope over
``unf_iter_count``, which cancels the marker pair and every one-time cost.

VALIDITY GATES, IN ORDER:
  1. CtrlSwap MUST measure ~2.00x CtrlLoad (both are documented
     per-instruction constants). If not, discard the whole run.
  2. Timing CANNOT distinguish a working macro from one that degenerated into
     a plain SFPLOAD — a degenerate arm measures the SAME OR BETTER. The
     number here is meaningful only while
     ``test_topk_xl_unfused_macro.py`` (chained golden + opt-out differential
     + schedule-nothing mutation control) is green on the same build.

The parameter classes live in this file rather than in
helpers/test_variant_parameters.py deliberately: they are consumed by exactly
one kernel (same rationale as perf_topk_merge_macro.py).
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


class UnfArm(Enum):
    """The value IS the integer the kernel's ``UNF_ARM`` comparison expects."""

    CtrlLoad = 0
    CtrlSwap = 1
    Merge = 2
    Rebuild = 3
    Step = 4


@dataclass
class UNF_ARM(TemplateParameter):
    """Emits ``#define UNF_ARM <n>``. MUST be a ``#define``: the kernel guards
    the symbol with ``#ifndef`` and compares it in the preprocessor."""

    unf_arm: UnfArm = UnfArm.Merge

    def convert_to_cpp(self) -> str:
        return f"#define UNF_ARM {self.unf_arm.value}"


@dataclass
class UNF_ITER_COUNT(TemplateParameter):
    unf_iter_count: int = 16

    def convert_to_cpp(self) -> str:
        return f"#define UNF_ITER_COUNT {self.unf_iter_count}"


@dataclass
class UNF_K(TemplateParameter):
    unf_k: int = 1024

    def convert_to_cpp(self) -> str:
        return f"#define UNF_K {self.unf_k}"


@dataclass
class UNF_MACRO_OFF(TemplateParameter):
    """The pairing knob: True rebuilds the shipping (pre-macro) bodies by
    defining the header's opt-out."""

    macro_off: bool = False

    def convert_to_cpp(self) -> str:
        if self.macro_off:
            return "#define DISABLE_TOPK_XL_SFPLOADMACRO 1"
        return "// unfused macro bodies: ON (default)"


# Two-point slope per arm: the difference of the two iter counts divides the
# cycle delta, cancelling the ~30-cycle marker pair and one-time costs inside
# the zone. Controls run enough MOP iterations to hit the issue-rate plateau.
_ITER_COUNTS = {
    UnfArm.CtrlLoad: [2048, 4096],
    UnfArm.CtrlSwap: [2048, 4096],
    UnfArm.Merge: [8, 16],
    UnfArm.Rebuild: [8, 16],
    UnfArm.Step: [8, 16],
}

_KS = [512, 1024, 2048]


# The controls are per-instruction constants — K- and macro-invariant — so
# sweep them once (at one K, macro on) rather than 12 times.
def _macro_off_axis(unf_arm):
    if unf_arm in (UnfArm.CtrlLoad, UnfArm.CtrlSwap):
        return [False]
    return [False, True]


def _k_axis(unf_arm):
    if unf_arm in (UnfArm.CtrlLoad, UnfArm.CtrlSwap):
        return [512]
    return _KS


# Pinned to 1 so the .post.csv carries raw zone cycles.
_LOOP_FACTOR = 1
_TILE_COUNT = 1
_RUN_COUNT = 5

# Float16_b in/out with dest_acc=Yes gives a 32-bit Dest — what the unfused
# FP32 value / INT32 index load-store walk addresses, and what
# `transpose_dest_face_32b`'s two-pass 16-bit shuffle assumes.
_FORMATS = input_output_formats([DataFormat.Float16_b], same=True)
_DEST_ACC = DestAccumulation.Yes


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=_FORMATS,
    unf_arm=list(_ITER_COUNTS.keys()),
    unf_k=lambda unf_arm: _k_axis(unf_arm),
    macro_off=lambda unf_arm: _macro_off_axis(unf_arm),
    unf_iter_count=lambda unf_arm: _ITER_COUNTS[unf_arm],
)
def test_perf_topk_unfused_macro(
    perf_report, formats, unf_arm, unf_k, macro_off, unf_iter_count
):
    configuration = PerfConfig(
        "sources/topk_unfused_macro_perf.cpp",
        formats,
        # MATH_ISOLATE only: unpack's per-iteration SrcB dummy valid is a
        # handshake, not work, and pack does nothing.
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            UNF_ARM(unf_arm),
            UNF_ITER_COUNT(unf_iter_count),
            UNF_K(unf_k),
            UNF_MACRO_OFF(macro_off),
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
