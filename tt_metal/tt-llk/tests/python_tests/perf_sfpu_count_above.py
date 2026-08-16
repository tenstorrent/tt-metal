# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Blackhole SFPU issue-rate benchmark for the replay-fed threshold-count inner loop
(``sources/sfpu_count_above_perf.cpp``), the candidate inner loop for
threshold-based Top-K selection.

Three arms, swept in ONE pytest invocation so their rows come from the same
session (same silicon, same clock, same profiler build) and are therefore
directly comparable:

  CountArm.ReplayLoad -- control. Replay-fed SFPLOADMACRO stream. Must measure
                         ~1.0 cycles per 32-element vector. If it does not, the
                         instruction feed path is still the limiter and neither
                         of the other two arms means anything.
  CountArm.ReplaySwap -- control. Replay-fed SFPSWAP stream. Must be ~2.0x
                         ReplayLoad: SFPSWAP is documented as 2 backend cycles
                         with a hardware-inserted, non-fillable bubble. This is
                         the tripwire -- the one arm whose answer is known
                         independently of anything being measured.
  CountArm.CountD1    -- the real loop: macro-scheduled SFPGT at delay 1 plus a
                         software SFPIADD accumulate, ping-ponged A/B. Read this
                         only after both controls land where they should.

The arm is a per-variant template ``#define``, so it enters the variant hash and
sweeps like any other compile-time parameter.

READING THE NUMBER
------------------
``postprocess_tile_loop`` (helpers/perf/core.py) divides the TILE_LOOP row's
``mean(...)``/``std(...)`` columns by ``loop_factor * tile_cnt``. Both are pinned
to 1 here (see ``_LOOP_FACTOR`` / ``_TILE_COUNT`` below), so the value that lands
in ``mean(MATH_ISOLATE)`` of the .post.csv is the RAW cycle count of the whole
TILE_LOOP zone for one variant -- not a per-tile figure. This kernel has no tile
loop at all: its work unit is a 32-element vector, and the count of those is
``ITER_COUNT``, a compile-time define rather than a runtime trip count, so there
is nothing for the tile/loop divisor to legitimately represent.

Cycles per vector therefore comes from a two-point slope across ITER_COUNT,
per arm:

    cyc_per_vec = (mean@2048 - mean@512) / (2048 - 512)

The subtraction cancels the fixed cost of the START_PERF_MEASURE marker pair
(~30 cycles on Blackhole, per test_profiler_overhead.py) plus the replay-buffer
load and MOP programming, none of which scale with ITER_COUNT. A single-point
``mean / ITER_COUNT`` would fold that constant into the rate and inflate the
512-vector arm by ~6%.
"""

import pytest
from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.llk_params import CountArm, DestAccumulation, PerfRunType
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    COUNT_ARM,
    COUNT_ITER_COUNT,
    COUNT_THR_BITS,
    LOOP_FACTOR,
    TILE_COUNT,
)

# 1.0f. The stimulus is whatever happens to be in Dest -- unpack does no work in
# MATH_ISOLATE -- so the threshold only has to be a legal, finite value; it does
# not steer the measured issue rate. Swept as a single value so it is recorded in
# the CSV rather than left implicit in the kernel's #ifndef fallback.
_THRESHOLD_BITS = 0x3F800000

# Two points for the slope. Both are even (the ping-pong body covers two vectors
# per replay pass, and the kernel static_asserts on that) and both are large
# enough that the ~30-cycle marker pair is a small fraction of the zone.
#
# MOP limit sanity check: the kernel runs ITER_COUNT/2 replay passes, chunked at
# MOP_MAX_ITERS=128 per ckernel_unpack_template::run call, because TT_OP_MOP's
# loop_count field is 7 bits (count-1 <= 127) while the `count` parameter is a
# uint8_t -- passing 256 truncates to 0 and the MOP runs ZERO times, which reads
# out as a spectacularly fast result rather than as an error.
#   512  -> 256 passes  -> 2 chunks of 128, remainder 0
#   2048 -> 1024 passes -> 8 chunks of 128, remainder 0
# Every chunk is <= 128, and neither value leaves a remainder, so the chunking
# adds the same 0.4%-ish RISC-V involvement per 128 passes to both points and
# cancels in the slope.
_ITER_COUNTS = [512, 2048]

# Pinned to 1 so the .post.csv carries raw zone cycles -- see READING THE NUMBER.
_LOOP_FACTOR = 1
_TILE_COUNT = 1

# std(...) columns are dropped as structurally empty when there is a single
# sample per marker (helpers/profiler.py::_stats_timings), and a slope taken
# across two arms with no spread is not defensible. 5 runs populate std and make
# the run-to-run noise visible next to the ~1.0 vs ~2.0 cycle/vector gap the
# controls are supposed to show.
_RUN_COUNT = 5

# The kernel touches only LRegs and a fixed Dst address: no operand is unpacked,
# nothing is packed out, and no format conversion happens anywhere in the timed
# region. But --speed-of-light (which CI uses) folds every runtime parameter into
# the build header and takes the compile_time_formats path, which dereferences
# formats_config[0] and the stimuli address block -- both of which are None if
# `formats`/`variant_stimuli` are omitted. So a minimal, well-formed pair is
# supplied purely to keep that path valid.
#
# Float16_b in/out with dest_acc=Yes gives a 32-bit Dest, which is what the
# kernel's INT32 SFPLOAD walk addresses.
_FORMATS = input_output_formats([DataFormat.Float16_b], same=True)
_DEST_ACC = DestAccumulation.Yes


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=_FORMATS,
    count_arm=[
        CountArm.ReplayLoad,
        CountArm.ReplaySwap,
        CountArm.CountD1,
        CountArm.MacroTriple,
        CountArm.MaskStore,
        CountArm.MacroExp,
        CountArm.HistNibble,
        CountArm.MultiPass,
        CountArm.PassSync,
    ],
    iter_count=_ITER_COUNTS,
)
def test_perf_sfpu_count_above(perf_report, formats, count_arm, iter_count):
    configuration = PerfConfig(
        "sources/sfpu_count_above_perf.cpp",
        formats,
        # MATH_ISOLATE only. Unpack and pack declare the same two zones (they must,
        # or the three-thread zone barrier deadlocks) but do no work, so
        # UNPACK_ISOLATE / PACK_ISOLATE would time an empty region, L1_CONGESTION
        # has no L1 traffic to congest, and L1_TO_L1 pairs an unpack ZONE_START
        # with a pack ZONE_END across threads that never move data -- it would
        # raise in helpers/profiler.py::_stats_l1_to_l1.
        run_types=[PerfRunType.MATH_ISOLATE],
        # Everything is a compile-time template: the kernel reads no runtime
        # parameters, and the three #define params must reach the preprocessor
        # before the kernel's #ifndef guards (build.h is included via params.h,
        # above the kernel's parameter block).
        templates=[
            COUNT_ARM(count_arm),
            COUNT_ITER_COUNT(iter_count),
            COUNT_THR_BITS(_THRESHOLD_BITS),
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
