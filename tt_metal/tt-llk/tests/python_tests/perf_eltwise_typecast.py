# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
On-silicon perf benchmark for the SFPU typecast op (issue #46751).

Measures cycles/tile for the typecast variants whose SFPLOADMACRO fast path was
re-introduced for Blackhole. The macro fast path in the calculate_typecast_*
primitives is gated `#ifndef DISABLE_SFPLOADMACRO`; compiling with
TT_METAL_DISABLE_SFPLOADMACRO=1 selects the plain-loop fallback (pre-fix
baseline). Running this module twice -- once without the env (macro ON,
optimized) and once with TT_METAL_DISABLE_SFPLOADMACRO=1 (macro OFF, baseline)
-- gives a clean A/B on the same tree.

Unlike test_perf_eltwise_unary_sfpu (which dispatches a single MathOperation),
typecast is selected by the (IN, OUT) DataFormat pair via typecast_tile<IN, OUT>
/ the shared SfpuType::typecast dispatch, so this module uses the dedicated
typecast perf kernel sources/eltwise_unary_typecast_perf.cpp with the
TYPECAST_FORMATS(in, out) template.
"""

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    MathOperation,
    PerfRunType,
    StableSort,
    Transpose,
)
from helpers.param_config import parametrize
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    FAST_MODE,
    ITERATIONS,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    STABLE_SORT,
    TILE_COUNT,
    TYPECAST_FORMATS,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

# The five (IN, OUT, dest_acc) typecast cases that exercise the SFPLOADMACRO path
# re-introduced by issue #46751.
#
# dest_acc is the production setting EXCEPT for the two rows marked below: their
# macro fast path is gated !DST_ACCUM_MODE, but ttnn.typecast forces dest_acc=Yes
# for 32-bit outputs (typecast.cpp:38-41), so production takes the plain loop. Those
# rows measure the macro in isolation and their speedup is not production-realized.
_TYPECAST_PERF_CASES = [
    # Float16_b -> UInt16: routes through calculate_typecast_fp32_to_uint16 (the
    # Compute API maps Float16_b-in-Dest -> UInt16 to it, no FP32 load from L1).
    (DataFormat.Float16_b, DataFormat.UInt16, DestAccumulation.No),
    # uint16_to_fp32 -- NOT production-reachable (Float32 out -> dest_acc=Yes).
    (DataFormat.UInt16, DataFormat.Float32, DestAccumulation.No),
    # uint32_to_fp16b, macro mode-agnostic; UInt32 needs 32-bit Dest.
    (DataFormat.UInt32, DataFormat.Float16_b, DestAccumulation.Yes),
    # uint16_to_uint32 -- NOT production-reachable (UInt32 out -> dest_acc=Yes).
    (DataFormat.UInt16, DataFormat.UInt32, DestAccumulation.No),
    # int32_to_uint16, macro mode-agnostic; Int32 needs 32-bit Dest.
    (DataFormat.Int32, DataFormat.UInt16, DestAccumulation.Yes),
]


def _is_block_float(fmt: DataFormat) -> bool:
    return fmt in (DataFormat.Bfp8_b, DataFormat.Bfp4_b, DataFormat.Bfp2_b)


@pytest.mark.perf
@parametrize(
    typecast_case=_TYPECAST_PERF_CASES,
    approx_mode=[ApproximationMode.No],
    loop_factor=[
        16,
    ],  # Number of iterations to run the test in order to minimize profiler overhead in measurement
    iterations=[
        8,
    ],  # SFPU iteration count; the typecast dispatch hardcodes 8 ITERATIONS internally
    input_dimensions=[
        [128, 64],  # tile_cnt: 8
    ],
)
def test_perf_eltwise_typecast(
    perf_report,
    typecast_case,
    approx_mode,
    loop_factor,
    iterations,
    input_dimensions,
):
    input_format, output_format, dest_acc = typecast_case
    formats = InputOutputFormat(input_format, output_format)

    # Calculate tile count from input dimensions
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    # Unpack straight into Dest when the input is 32-bit: the unpacker has no
    # SrcA/SrcB path for Int32/UInt32/Float32, so cunpack_common asserts
    # unpack_to_dest for them (matches the functional typecast test). The perf
    # kernel keeps the unpack-A acc_to_dest template arg false so this is legal
    # even when dest_acc=Yes. For 16-bit inputs the data is copied SrcA -> Dest
    # by the math datacopy A2D before the SFPU typecast runs.
    unpack_to_dest = input_format.is_32_bit()

    configuration = PerfConfig(
        "sources/eltwise_unary_typecast_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.MATH_ISOLATE,
        ],
        templates=[
            # Emits SFPU_UNARY_OPERATION = SfpuType::typecast so the kernel goes
            # through the shared unary-SFPU dispatch; TYPECAST_FORMATS supplies the
            # (input, output) pair that selects the concrete typecast kernel.
            MATH_OP(mathop=MathOperation.Typecast),
            TYPECAST_FORMATS(input_format, output_format),
            APPROX_MODE(approx_mode),
            ITERATIONS(iterations),
            FAST_MODE(FastMode.No),
            STABLE_SORT(StableSort.No),
        ],
        runtimes=[
            TILE_COUNT(tile_count_A),
            LOOP_FACTOR(loop_factor),
            NUM_FACES(num_faces=faces_to_generate),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
        ],
        variant_stimuli=StimuliConfig(
            None,
            input_format,
            None,
            input_format,
            output_format,
            tile_count_A=tile_count_A,
            tile_count_B=tile_count_B,
            tile_count_res=tile_count_A,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    # pack_src fix for the SFPU typecast (mirrors test_eltwise_unary_typecast).
    #
    # The generic format inference models pack_src from the *input* (what the
    # unpacker writes to Dest). For a typecast the SFPU overwrites Dest with the
    # *output* value, so the packer must read the output's register
    # representation, not the input's. In 16-bit Dest mode (dest_acc=No) the
    # inferred pack_src would otherwise stay equal to the input format and the
    # pack would be rejected (e.g. UInt16 -> UInt32 is not a supported packer
    # conversion). For dest_acc=Yes the 32-bit gasket converts from Dest and the
    # inference already yields the output format, so no patch is needed there.
    if dest_acc == DestAccumulation.No:
        pack_src = (
            DataFormat.Float16_b if _is_block_float(output_format) else output_format
        )
        for fmt_config in configuration.formats_config:
            fmt_config.pack_src = pack_src

    configuration.run(perf_report)
