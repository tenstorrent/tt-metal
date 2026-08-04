# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""|sin| <= 1 and |cos| <= 1 across the large-argument range.

The four-stage Cody-Waite reduction is exact only while j * P0 stays
representable, which gives out around |x| ~1.7e7. Past that the reduced
argument leaves [-PI/2, PI/2] and the minimax polynomial -- fitted only on
that interval -- grows without bound.

This asserts the invariant, not the accuracy. The values out there are still
imprecise, which is the separate range-extension question; what matters here
is that callers feeding the result to acos or sqrt(1 - s*s) get a number
rather than NaN. Same shape as test_exponential_clamp_negative, which asserts
a sign property for out-of-domain exp inputs rather than closeness to golden.

Every other suite stays inside a few periods -- sfpu_domains.py uses
[-pi, pi] and the ttnn sweep uses [-100, 100] -- so this range has never
been exercised.
"""

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    FastMode,
    MathOperation,
    format_dict,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    DestSync,
    generate_input_dim,
)

# Smallest arguments measured to escape [-1, 1], pinned so a regression that
# only moves the threshold still trips here.
FIRST_FAILURES = {
    MathOperation.Sin: [16845174.0, 16793388.0, 3.0e7, 1.0e8, 1.0e9],
    MathOperation.Cos: [16987826.0, 16845174.0, 3.0e7, 1.0e8, 1.0e9],
}


@pytest.mark.parametrize("mathop", [MathOperation.Sin, MathOperation.Cos])
@pytest.mark.parametrize(
    "dest_acc", [DestAccumulation.Yes, DestAccumulation.No], ids=["fp32", "bf16"]
)
def test_trig_stays_in_unit_range(mathop, dest_acc):
    torch.manual_seed(0)
    input_dimensions = [32, 32]
    fmt = (
        DataFormat.Float32
        if dest_acc == DestAccumulation.Yes
        else DataFormat.Float16_b
    )
    formats = InputOutputFormat(fmt, fmt)
    torch_fmt = format_dict[fmt]

    n = input_dimensions[0] * input_dimensions[1]
    src_A = torch.rand(n, dtype=torch.float32) * (1.0e9 - 1.0e7) + 1.0e7
    for i, v in enumerate(FIRST_FAILURES[mathop]):
        src_A[i] = v
    src_A = src_A.to(torch_fmt)
    src_B = torch.zeros(n, dtype=torch_fmt)

    tile_cnt = (input_dimensions[0] // 32) * (input_dimensions[1] // 32)
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(ApproximationMode.No),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=mathop),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=tile_cnt,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=(
            formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        ),
    )

    res = torch.tensor(configuration.run().result, dtype=torch_fmt)

    finite = torch.isfinite(res)
    assert torch.all(finite), f"{(~finite).sum()} non-finite results for a finite input"

    out_of_range = res.abs() > 1.0
    worst = res.abs().max().item()
    n_bad = int(out_of_range.sum())
    assert n_bad == 0, (
        f"{n_bad}/{n} results outside [-1, 1]; worst |out| = {worst:.6g}. "
        f"First offending inputs: "
        f"{[round(float(v), 1) for v in src_A[out_of_range][:5]]}"
    )


# Non-finite arguments, measured rather than assumed. With a bf16 destination
# they come out of the polynomial as inf, not NaN, so the clamp has to skip
# them: turning inf into 1.0 would replace an obviously broken result with a
# plausible one, which is the failure mode this change exists to prevent.
INF, NAN = float("inf"), float("nan")


@pytest.mark.parametrize("mathop", [MathOperation.Sin, MathOperation.Cos])
@pytest.mark.parametrize("dest_acc", [DestAccumulation.Yes, DestAccumulation.No],
                         ids=["fp32", "bf16"])
def test_trig_leaves_nonfinite_alone(mathop, dest_acc):
    torch.manual_seed(0)
    dims = [32, 32]
    fmt = DataFormat.Float32 if dest_acc == DestAccumulation.Yes else DataFormat.Float16_b
    formats = InputOutputFormat(fmt, fmt)
    tf = format_dict[fmt]

    n = dims[0] * dims[1]
    src_A = torch.zeros(n, dtype=tf)
    src_A[0], src_A[1], src_A[2] = INF, -INF, NAN
    src_B = torch.zeros(n, dtype=tf)
    tc = 1
    nb, ntb = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, dims, TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard)

    cfg = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp", formats,
        templates=[generate_input_dim(dims, dims), APPROX_MODE(ApproximationMode.No),
                   FAST_MODE(FastMode.No), CLAMP_NEGATIVE(True), MATH_OP(mathop=mathop)],
        runtimes=[TILE_COUNT(tc), NUM_BLOCKS(nb), NUM_TILES_IN_BLOCK(ntb)],
        variant_stimuli=StimuliConfig(src_A, formats.input_format, src_B,
                                      formats.input_format, formats.output_format,
                                      tile_count_A=tc, tile_count_B=tc, tile_count_res=tc),
        dest_acc=dest_acc,
        unpack_to_dest=(formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes),
    )
    res = torch.tensor(cfg.run().result, dtype=tf)
    got = [float(res[i]) for i in range(3)]
    print(f"\nMEDIDO {mathop} {fmt}: +inf->{got[0]}  -inf->{got[1]}  nan->{got[2]}")

    # Lo unico inaceptable seria que una entrada no finita saliera como valor finito
    # plausible: eso convierte "indefinido" en "creible" y se propaga callado.
    for etiqueta, v in zip(("+inf", "-inf", "nan"), got):
        assert not (v == v and abs(v) <= 1.0), (
            f"{mathop} {fmt}: entrada {etiqueta} -> {v}, un valor finito dentro de [-1,1]")
