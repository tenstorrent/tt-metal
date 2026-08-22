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
  * val in [-20, 0]    -- the SDPA domain (exponent of a negative score delta),
  * val in [-88, 0]    -- down to the surviving lower clamp,
  * val in [-4, 4]     -- straddling zero, still far below the removed upper clamp,
  * val in [-400, 0]   -- deep enough to actually engage the lower clamp.

That last range is the one that tests the surviving clamp rather than assuming it.
The clamp only engages at val <= -88.03, and simply going a little past it is not
observable: at val = -100 the unclamped path still yields ~2^-110, which is under
atol. It first becomes visible around val ~= -176, where the float-to-int step in
`_float_to_int32_for_exp_21f_` recombines to exponent field 127 and returns ~1.0
against a golden of 0. -400 clears that for both scale arms -- the BF16_HALF arm
halves the effective input, so it needs ~2x the depth to reach the same window --
and the golden there is well defined: exp() underflows the bf16 DEST to 0.

Above val ~= 88.7 the *removed* clamp would have mattered, but exp() overflows
bfloat16 there (exp(88.7) ~= 3.3e38 vs bf16 max 3.39e38), so that domain is
deliberately not swept -- there is no well-defined expected value to compare
against, and the header explicitly scopes itself to val <= 0.

Both kernel selections are swept (the hand/generated selector idiom, matching
the device-profile node's impl axis): `sdpa_exp_impl=0` drives the handwritten
production accurate exp (`calculate_exponential<APPROX=false>`, the clamped
`_sfpu_exp_21f_bf16_tti_` path — the kernel SDPA uses in production today) and
`sdpa_exp_impl=1` drives the semantic upper-unclamped variant. Every swept
point sits in the domain where the two are contractually identical (the removed
clamp is dead there and the surviving lower clamp behaves the same), so both
impls verify against the same exp() golden: impl 1's cells pin "removing the
upper clamp changed nothing observable", and impl 0's cells are the hand-corr
legs that let the 2x2 sweep's hand perf selector carry verified-kernel evidence
(sweep-hardening defect 2: a perf leg without its own correctness leg is
withheld).

`_ckernel_sfpu_exp_accurate_upper_unclamped_` static_asserts `!is_fp32_dest_acc_en`
("upper-unclamped exp variant implemented for bf16 dest only"), so the DEST is
always 16-bit: dest_acc=Yes does not compile, and a Float32 input has nowhere to go
because it would need the 32-bit DEST that assert forbids. Only the output side is
free, and neither arm is the tighter one: the kernel does
`convert<vFloat16b>(y, NearestEven)` before the store unconditionally, so the device
value is already bf16-exact and the golden models that. The Float32 output arm only
skips a second, no-op conversion in the packer.
"""

from dataclasses import dataclass

import pytest
import torch
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    SdpaExpUnclampedGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, PerfRunType, format_dict
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    SFPU_SCALE_EN,
    SFPU_UNARY_SCALAR,
    TILE_COUNT,
    TemplateParameter,
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

# Exercised as (low, high) input ranges. All stay far below the removed upper clamp
# (val << 88.7); the last one reaches past the surviving lower clamp -- see docstring.
INPUT_RANGES = [(-20.0, 0.0), (-88.0, 0.0), (-4.0, 4.0), (-400.0, 0.0)]


@dataclass
class SdpaExpImplTemplate(TemplateParameter):
    # Field name = the CSV column header this param would emit; must be
    # globally unique across parameter classes (FM-F1 contract).
    sdpa_exp_impl: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t SDPA_EXP_IMPL = {self.sdpa_exp_impl}u;"


@skip_for_wormhole
@parametrize(
    formats=FORMATS,
    input_range=INPUT_RANGES,
    scale_en=[False, True],
    scale=[BF16_ONE, BF16_HALF],
    num_tiles=[1, 2],
    # 0 = handwritten production exp (clamped accurate 21f), 1 = semantic
    # upper-unclamped — same axis and meaning as the device-profile node below.
    sdpa_exp_impl=[0, 1],
)
def test_sfpu_sdpa_exp_unclamped(
    formats, input_range, scale_en, scale, num_tiles, sdpa_exp_impl
):
    if not scale_en and scale != BF16_ONE:
        # The template drops the multiply entirely, so the scale value is dead.
        pytest.skip("scale is not read when SCALE_EN is false")

    torch.manual_seed(0)

    torch_format = format_dict[formats.input_format]

    # scale_en=True with scale == 1.0 is the cheap regression on the multiply itself:
    # the identity scale must leave the result bit-identical to the scale_en=False arm.
    effective_scale = scale if scale_en else BF16_ONE

    low, high = input_range
    stimuli_size = (num_tiles * ELEMENTS_PER_TILE,)
    src_A = torch.empty(stimuli_size, dtype=torch_format).uniform_(low, high)
    src_B = torch.zeros_like(src_A)

    golden_generator = get_golden_generator(SdpaExpUnclampedGolden)
    golden_tensor = golden_generator(src_A, effective_scale, formats.output_format)

    configuration = TestConfig(
        "sources/sfpu_sdpa_exp_unclamped_test.cpp",
        formats,
        templates=[
            SdpaExpImplTemplate(sdpa_exp_impl),
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
        # Pinned, not swept: the kernel static_asserts !is_fp32_dest_acc_en.
        dest_acc=DestAccumulation.No,
        unpack_to_dest=False,
    )

    res_from_L1 = configuration.run().result
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "upper-unclamped exp does not match exp() over the swept domain"


@pytest.mark.parametrize(
    "sdpa_exp_impl,label", [(0, "handwritten_production"), (1, "semantic_unclamped")]
)
def test_sfpu_sdpa_exp_unclamped_device_profile(perf_report, sdpa_exp_impl, label):
    """Measure only the exact one-tile SFPU body from the correctness node.

    Each silicon sample is launched in a fresh pytest process with a unique
    LLK archive, so profiler state and build artefacts are never shared across
    the paired compiler configurations or repetitions.
    """
    torch.manual_seed(0)
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    src_A = torch.empty((ELEMENTS_PER_TILE,), dtype=torch.bfloat16).uniform_(-20.0, 0.0)
    src_B = torch.zeros_like(src_A)

    configuration = PerfConfig(
        "sources/sfpu_sdpa_exp_unclamped_test.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            SdpaExpImplTemplate(sdpa_exp_impl),
            SFPU_SCALE_EN(scale_en=True),
            SFPU_UNARY_SCALAR(value_bits=BF16_ONE),
        ],
        runtimes=[TILE_COUNT(1)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=DestAccumulation.No,
        unpack_to_dest=False,
        compile_time_formats=True,
    )
    configuration.run(perf_report, run_count=1)
    frame = perf_report.frame()
    rows = frame[frame["marker"] == "SDPA_EXP_UNCLAMPED_BODY"]
    assert len(rows) == 1, frame.to_string(index=False)
    cycles = float(rows.iloc[0]["mean(MATH_ISOLATE)"])
    assert cycles > 0
    print(f"SDPA_EXP_UNCLAMPED_DEVICE_PROFILE impl={label} body_cycles={cycles:.2f}")
