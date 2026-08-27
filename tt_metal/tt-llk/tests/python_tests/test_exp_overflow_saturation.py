# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Regression: exp() must saturate to +inf for large FINITE inputs, not wrap.

The Exp sweep domain (_exp_spec in sfpu_domains.py) is deliberately range-bounded to avoid
overflow, and _APPROX_ACCURACY_MAX caps the argument at 16.0 on top of that -- so nothing
in the ordinary sweep drives xlog2 = x/ln2 + 127 above 255. The only input in the whole
suite that reaches the saturating path is the +inf special. (The kernel biases by 126.5 and
rounds to nearest rather than biasing by 127 and truncating -- same floor, same threshold;
see docs/sfpu_exp21f_optimization.md section 9.)

That leaves a gap: the bfloat16-accurate kernel (_sfpu_exp_21f_bf16_tti_) relies on its
FP32->UINT8 convert saturating at 255 for its upper clamp. If that convert ever wrapped
instead, +inf would still come out right (inf is handled distinctly) while every large
finite input silently returned a tiny wrong value -- exp(100) would read 2^-112 rather than
+inf, and no other test would notice.

The data is bfloat16 in L1, so every probe near the overflow threshold is chosen to be
exactly representable there: a literal like 88.7 is rounded to 88.5 on the way in, and a
probe written next to the threshold would be pinned to a value other than the one it names.
1e30 is the one probe without that property -- it lands on 1.000255552e+30 -- and does not
need it, being a deep-overflow stress value where every nearby representable input
saturates alike.
"""
import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    FastMode,
    MathOperation,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
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
    generate_input_dim,
)

# Every threshold probe is exactly representable in bfloat16; 1e30 is not (it lands on
# 1.000255552e+30) and does not need to be. 88.5 is the last input whose result stays under
# the bfloat16 max of 3.39e38 (exp(88.5) = 2.72e38); 89.0 and up must all be +inf.
PROBES = [
    0.0,
    1.0,
    80.0,
    88.0,
    88.5,
    89.0,
    90.0,
    100.0,
    128.0,
    200.0,
    512.0,
    1e30,
    float("inf"),
]


@pytest.mark.nightly
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    # dest_acc=No -> is_fp32_dest_acc_en=false -> the bf16 TTI kernel
    dest_acc=[DestAccumulation.No],
    input_dimensions=[[32, 32]],
)
def test_exp_overflow_saturates(
    formats: InputOutputFormat,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    n = input_dimensions[0] * input_dimensions[1]

    src_A = torch.zeros(n, dtype=torch.bfloat16)
    for i, v in enumerate(PROBES):
        src_A[i] = v
    src_B = torch.zeros(n, dtype=torch.bfloat16)
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
            MATH_OP(mathop=MathOperation.Exp),
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
        unpack_to_dest=False,
    )
    res = torch.tensor(configuration.run().result, dtype=torch.float32)
    print("\n=== exp() overflow saturation (bfloat16 dest, accurate path) ===")
    bad = []
    for i, v in enumerate(PROBES):
        got = res[i].item()
        want = float(torch.exp(torch.tensor(v, dtype=torch.float64)))
        ok = (
            "OK"
            if (
                (want > 3.4e38 and got == float("inf"))
                or (want <= 3.4e38 and abs(got - want) <= 0.01 * max(want, 1e-30))
            )
            else "BAD"
        )
        if ok == "BAD":
            bad.append((v, got, want))
        print(f"  exp({v:>10.6g}) = {got:>14.6g}   expected {want:>14.6g}   [{ok}]")
    assert not bad, (
        f"exp() is wrong on large finite inputs: {bad}. The bfloat16-accurate kernel takes "
        "its upper clamp from FP32->UINT8 saturating at 255. A wrapping convert would show "
        "up here as a large finite input returning a tiny value instead of +inf, while "
        "exp(+inf) kept working. See _sfpu_exp_21f_bf16_tti_ in ckernel_sfpu_exp.h."
    )
