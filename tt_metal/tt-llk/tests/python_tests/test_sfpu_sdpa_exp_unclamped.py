# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Upper-unclamped exp SFPU test.

Covers tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_sdpa_exp_unclamped.h
(`_sfpu_exp_21f_bf16_lower_clamp_only_` / `_ckernel_sfpu_exp_accurate_upper_unclamped_`).

The kernel is the accurate 21f exp with the *upper* input clamp removed. The
clamped reference (`_sfpu_exp_21f_bf16_`) clamps xlog2 = val/ln2 + 127 into
[0, 255], i.e. val into roughly [-88.0, 88.7]. The SDPA caller only ever feeds
val <= 0, so the upper half of that clamp is dead code and dropping it saves an
SFPLOADI + clamp per element.

The contract this test pins down is therefore "identical to exp() over the
domain the caller actually uses":
  * val in [-20, 0]  -- the SDPA domain (exponent of a negative score delta),
  * val in [-88, 0]  -- down to the surviving lower clamp,
  * val in [-4, 4]   -- straddling zero, still far below the removed upper clamp.

Above val ~= 88.7 the removed clamp would have mattered, but exp() overflows
bfloat16 there (exp(88.7) ~= 3.3e38 vs bf16 max 3.39e38), so that domain is
deliberately not swept -- there is no well-defined expected value to compare
against, and the header explicitly scopes itself to val <= 0.

`_ckernel_sfpu_exp_accurate_upper_unclamped_` static_asserts `!is_fp32_dest_acc_en`
("upper-unclamped exp variant implemented for bf16 dest only"), so the DEST is
always 16-bit: dest_acc=Yes does not compile, and a Float32 input has nowhere to go
because it would need the 32-bit DEST that assert forbids. Only the output side is
free, and fp32 output is the tighter of the two since the packer does not round the
result down on the way to L1.
"""

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    SdpaExpUnclampedGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    SFPU_SCALE_EN,
    SFPU_UNARY_SCALAR,
    TILE_COUNT,
)
from helpers.utils import passed_test

# Input is pinned to Float16_b: the kernel is bf16-dest only, so anything wider has
# nowhere to land. Only the pack-side format varies.
FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, output_format)
    for output_format in [DataFormat.Float16_b, DataFormat.Float32]
]

# bfloat16 bit patterns (upper 16 bits of the fp32 encoding) -- this is what
# sfpi::sFloat16b() consumes, matching p_sfpu::kCONST_1_FP16B == 0x3F80.
BF16_ONE = 0x3F80
BF16_HALF = 0x3F00

# Exercised as (low, high) input ranges. Every range stays inside the surviving
# lower clamp (val >= -88) and far below the removed upper clamp (val << 88.7).
INPUT_RANGES = [(-20.0, 0.0), (-88.0, 0.0), (-4.0, 4.0)]


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No],
    input_range=INPUT_RANGES,
    scale=[BF16_ONE, BF16_HALF],
    num_tiles=[1, 2],
)
def test_sfpu_sdpa_exp_unclamped(formats, dest_acc, input_range, scale, num_tiles):
    torch.manual_seed(0)

    torch_format = format_dict[formats.input_format]

    # SCALE_EN is only worth compiling when the scale is not the identity; with
    # scale == 1.0 both paths must agree, which is the cheap regression on the
    # multiply itself.
    scale_en = scale != BF16_ONE

    low, high = input_range
    stimuli_size = (num_tiles * ELEMENTS_PER_TILE,)
    src_A = torch.empty(stimuli_size, dtype=torch_format).uniform_(low, high)
    src_B = torch.zeros_like(src_A)

    golden_generator = get_golden_generator(SdpaExpUnclampedGolden)
    golden_tensor = golden_generator(src_A, scale, formats.output_format)

    configuration = TestConfig(
        "sources/sfpu_sdpa_exp_unclamped_test.cpp",
        formats,
        templates=[
            SFPU_SCALE_EN(scale_en=scale_en),
            SFPU_UNARY_SCALAR(value_bits=scale),
        ],
        runtimes=[
            TILE_COUNT(num_tiles),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=num_tiles,
            tile_count_B=1,
            tile_count_res=num_tiles,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    res_from_L1 = configuration.run().result
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "upper-unclamped exp does not match exp() over the swept domain"
