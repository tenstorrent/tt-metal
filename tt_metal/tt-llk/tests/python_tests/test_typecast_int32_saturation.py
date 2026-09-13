# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Saturation of the fp32 -> int32 conversion.
#
# The bounds asserted here are the pair helpers/golden_generators.py
# saturate_integer specifies for a signed destination, [iinfo.min + 1,
# iinfo.max], with its comment "+1 because hardware uses sign-magnitude
# representation". Before this fix the kernel returned INT32_MIN
# for every input at or above 2^31, so the positive half of that range was
# inverted in sign rather than clamped.
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    MathOperation,
    format_dict,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TYPECAST_FORMATS,
    DestSync,
    generate_input_dim,
)

N = 1024
INT32_MAX = torch.iinfo(torch.int32).max
# saturate_integer's signed branch, not torch's cast
SIGNED_MIN = torch.iinfo(torch.int32).min + 1


def _run(in_fmt, vals):
    out_fmt = DataFormat.Int32
    formats = InputOutputFormat(in_fmt, out_fmt)
    dest_acc = DestAccumulation.Yes
    dims = [32, 32]
    v = [vals[i % len(vals)] for i in range(N)]
    src_A = torch.tensor(v, dtype=format_dict[in_fmt])
    src_B = torch.zeros(N, dtype=format_dict[in_fmt])
    nb, ntb = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, dims, [32, 32], BlocksCalculationAlgorithm.Standard
    )
    cfg = TestConfig(
        "sources/eltwise_unary_typecast_test.cpp",
        formats,
        templates=[
            generate_input_dim(dims, dims),
            APPROX_MODE(ApproximationMode.No),
            MATH_OP(mathop=MathOperation.Typecast),
            TYPECAST_FORMATS(in_fmt, out_fmt),
        ],
        runtimes=[TILE_COUNT(1), NUM_BLOCKS(nb), NUM_TILES_IN_BLOCK(ntb)],
        # int32 in L1 is two's complement; the sign-magnitude decoder is the default
        # and cannot round-trip a negative, which is what unpack_int32 documents.
        variant_stimuli=StimuliConfig(
            src_A, in_fmt, src_B, in_fmt, out_fmt,
            tile_count_A=1, tile_count_B=1, tile_count_res=1, twos_complement=True,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=in_fmt.is_32_bit(),
    )
    out = torch.tensor(cfg.run().result, dtype=torch.int32).flatten().tolist()
    return [out[i] for i in range(len(vals))]


def test_positive_overflow_clamps_to_int32_max():
    vals = [2147483648.0, 3e9, 3.4e38, float("inf")]
    got = _run(DataFormat.Float32, vals)
    assert got == [INT32_MAX] * len(vals), got


def test_negative_overflow_clamps_to_the_signed_minimum_the_golden_uses():
    vals = [-2147483648.0, -3e9, -3.4e38, float("-inf")]
    got = _run(DataFormat.Float32, vals)
    assert got == [SIGNED_MIN] * len(vals), got


def test_in_range_and_boundary_values_are_exact():
    # 2147483520.0 is the largest float below 2^31, so it must not clamp
    vals = [2147483520.0, -2147483520.0, 100.0, -100.0, 1.0, -1.0, 0.0, -0.5, 0.5,
            2000000000.0, -2000000000.0]
    got = _run(DataFormat.Float32, vals)
    want = torch.tensor(vals, dtype=torch.float32).to(torch.int32).tolist()
    assert got == want, list(zip(vals, got, want))
