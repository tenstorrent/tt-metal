# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import struct
from dataclasses import dataclass

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TernarySFPUGolden,
    WhereGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    PerfRunType,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.sfpu_domains import _OP_DOMAIN_REGISTRY, exclude_undefined_pair, for_op
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DEST_SYNC,
    DISABLE_SRC_ZERO_FLAG,
    FRESH_CPP_IMPL,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    SFPU_TERNARY_OP,
    SFPU_TERNARY_SCALAR,
    TemplateParameter,
)
from helpers.utils import passed_test

_SCALAR_VALUE = 2.0
_SCALAR_VALUE_BITS = struct.unpack("<I", struct.pack("<f", _SCALAR_VALUE))[0]


@dataclass
class TTNNWhereImplTemplate(TemplateParameter):
    # Field name = the CSV column header this param would emit; must be
    # globally unique across parameter classes (FM-F1 contract).
    ttnn_where_impl: int

    def convert_to_cpp(self) -> str:
        return f"#undef TTNN_WHERE_IMPL\n#define TTNN_WHERE_IMPL {self.ttnn_where_impl}"


# Helper check function
def torch_equal_nan(a, b):
    return torch.all((a == b) | (torch.isnan(a) & torch.isnan(b)))


def _ternary_default_specs(mathop, input_format):
    """Per-operand defaults for *mathop*: its registered domain, else the built-in one.

    No ternary op has an _OP_DOMAIN_REGISTRY entry, so every op currently takes the
    built-in branch. This is the single place a registered domain would take effect, and
    callers of _run_sfpu_ternary can override any operand to reach an edge the defaults
    exclude (e.g. the c -> 0 pole that addcdiv and snake_beta pin away from).
    """
    if mathop in _OP_DOMAIN_REGISTRY:
        # OperandSpecs carries only A and B, so a registered ternary op has no third
        # operand to read; reuse B for C.
        specs = exclude_undefined_pair(mathop, for_op(mathop, input_format))
        return specs.spec_A, specs.spec_B, specs.spec_B

    # addcdiv and snake_beta divide by c, so c is held away from zero.
    divide_by_c = mathop in (MathOperation.SfpuAddcdiv, MathOperation.SfpuSnakeBeta)
    spec_ab = StimuliSpec.uniform(low=-1.0, high=1.0)
    spec_c = (
        StimuliSpec.uniform(low=1.0, high=2.0)
        if divide_by_c
        else StimuliSpec.uniform(low=-1.0, high=1.0)
    )
    return spec_ab, spec_ab, spec_c


def _run_sfpu_ternary(
    formats,
    dest_acc,
    mathop,
    input_dimensions=[64, 64],
    spec_A=None,
    spec_B=None,
    spec_C=None,
    fresh_cpp_impl=0,
):
    # The specs below carry no seed, so seed here: an unseeded redraw makes a variant
    # sitting near its tolerance pass or fail by luck. Same as the binary driver.
    torch.manual_seed(0)

    default_A, default_B, default_C = _ternary_default_specs(
        mathop, formats.input_format
    )
    spec_a = spec_A if spec_A is not None else default_A
    spec_b = spec_B if spec_B is not None else default_B
    spec_c = spec_C if spec_C is not None else default_C

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_a,
        spec_B=spec_b,
    )

    src_C, tile_cnt_C, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_c,
        spec_B=spec_c,
    )

    generate_golden = get_golden_generator(TernarySFPUGolden)
    golden = generate_golden(
        mathop,
        src_A,
        src_B,
        src_C,
        _SCALAR_VALUE_BITS,
        formats.output_format,
    )

    configuration = TestConfig(
        "sources/sfpu_ternary_test.cpp",
        formats,
        templates=[
            SFPU_TERNARY_OP(mathop),
            SFPU_TERNARY_SCALAR(_SCALAR_VALUE_BITS),
            APPROX_MODE(ApproximationMode.No),
            DISABLE_SRC_ZERO_FLAG(True),
            DEST_SYNC(),
            FRESH_CPP_IMPL(fresh_cpp_impl),
        ],
        runtimes=[NUM_BLOCKS(tile_cnt_A), NUM_TILES_IN_BLOCK(1)],
        variant_stimuli=StimuliConfig(
            src_A.flatten(),
            formats.input_format,
            src_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            buffer_C=src_C.flatten(),
            stimuli_C_format=formats.input_format,
            tile_count_C=tile_cnt_C,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result
    res_from_L1 = res_from_L1[: len(golden)]

    assert len(res_from_L1) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    golden_tensor = torch.tensor(golden, dtype=torch_format).flatten()
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).flatten()

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Bfp8_b,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=[
        MathOperation.SfpuAddcmul,
        MathOperation.SfpuAddcdiv,
        MathOperation.SfpuLerp,
        MathOperation.SfpuSnakeBeta,
    ],
)
def test_sfpu_ternary(formats, dest_acc, mathop):
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")
    if (
        formats.input_format == DataFormat.Bfp8_b
        and mathop != MathOperation.SfpuAddcmul
    ):
        pytest.skip("Bfp8_b is only supported for addcmul")

    _run_sfpu_ternary(formats, dest_acc, mathop)


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    dest_acc=[DestAccumulation.No],
    fresh_cpp_impl=[0, 1],
)
def test_fresh_cpp_addcmul(formats, dest_acc, fresh_cpp_impl):
    """A/B identical stimuli/golden; pass criterion is the suite's format-aware tolerance gate."""
    _run_sfpu_ternary(
        formats,
        dest_acc,
        MathOperation.SfpuAddcmul,
        fresh_cpp_impl=fresh_cpp_impl,
    )


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    dest_acc=[DestAccumulation.No],
    fresh_cpp_impl=[0, 1],
)
def test_fresh_cpp_snake_beta(formats, dest_acc, fresh_cpp_impl):
    """Handwritten-shaped production snake_beta (metal ckernel_sfpu_snake_beta.h:
    reciprocal Newton constant in vConstFloatPrgm0, PolynomialEvaluator Horner,
    unroll-8 pin) vs fresh typed-C++ body (fresh_cpp/snakebeta.h) A/B over
    identical stimuli/golden; pass criterion is the suite's format-aware
    tolerance gate."""
    _run_sfpu_ternary(
        formats,
        dest_acc,
        MathOperation.SfpuSnakeBeta,
        fresh_cpp_impl=fresh_cpp_impl,
    )


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    dest_acc=[DestAccumulation.No],
    fresh_cpp_impl=[0, 1],
)
def test_fresh_cpp_addcdiv(formats, dest_acc, fresh_cpp_impl):
    """Storm lane S1: A/B the fresh addcdiv (a + value * b * recip(c), the
    torch.addcdiv statement with a typed reciprocal derivation) against the
    production kernel — identical stimuli (c held away from zero), golden,
    and tolerance gate."""
    _run_sfpu_ternary(
        formats,
        dest_acc,
        MathOperation.SfpuAddcdiv,
        fresh_cpp_impl=fresh_cpp_impl,
    )


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Int32,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=MathOperation.TTNNWhere,
    test_case=["mixed", "all_ones", "all_zeros"],
    ttnn_where_impl=[0, 1],  # production handwritten macro/replay, generated SFPI
)
def test_ttnn_where(
    formats,
    dest_acc,
    mathop,
    test_case,
    ttnn_where_impl,
):

    if (
        formats.input == DataFormat.Float32 and formats.output == DataFormat.Float32
    ) and dest_acc == DestAccumulation.No:
        pytest.skip("DataFormat.Float32 not supported with DestAccumulation.No")

    if (
        formats.input == DataFormat.Float16_b and formats.output == DataFormat.Float16_b
    ) and dest_acc == DestAccumulation.Yes:
        pytest.skip("DataFormat.Float16_b not supported with DestAccumulation.Yes")

    # 64x64 = 2x2 tiles: exercises the multi-tile block loop in sfpu_ternary_test.cpp.
    input_dimensions = [64, 64]
    sfpu_false_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=sfpu_false_spec,
        spec_B=sfpu_false_spec,
    )

    src_C, tile_cnt_C, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=sfpu_false_spec,
        spec_B=sfpu_false_spec,
    )

    # Modify the condition tensor based on test case
    if test_case == "all_ones":
        src_A = torch.ones_like(src_A)
    elif test_case == "all_zeros":
        src_A = torch.zeros_like(src_A)
    # For "mixed" case, use the generated stimuli as-is

    # laneJO formal-equivalence witness-check hook (see test_sfpu_binary.py):
    # LANEJO_SRC_OVERRIDE holds {"src_A","src_B","src_C"} replayed verbatim.
    import os as _lanejo_os

    _lanejo_src = _lanejo_os.environ.get("LANEJO_SRC_OVERRIDE")
    if _lanejo_src:
        _lanejo_t = torch.load(_lanejo_src)
        src_A = _lanejo_t["src_A"].to(src_A.dtype).reshape(src_A.shape)
        src_B = _lanejo_t["src_B"].to(src_B.dtype).reshape(src_B.shape)
        src_C = _lanejo_t["src_C"].to(src_C.dtype).reshape(src_C.shape)

    golden_generator = get_golden_generator(WhereGolden)
    golden = golden_generator(src_A, src_B, src_C)

    configuration = TestConfig(
        "sources/sfpu_ternary_test.cpp",
        formats,
        templates=[
            SFPU_TERNARY_OP(mathop),
            SFPU_TERNARY_SCALAR(_SCALAR_VALUE_BITS),
            APPROX_MODE(ApproximationMode.No),
            DISABLE_SRC_ZERO_FLAG(True),
            DEST_SYNC(),
            TTNNWhereImplTemplate(ttnn_where_impl),
        ],
        runtimes=[NUM_BLOCKS(tile_cnt_A), NUM_TILES_IN_BLOCK(1)],
        variant_stimuli=StimuliConfig(
            src_A.flatten(),
            formats.input_format,
            src_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            buffer_C=src_C.flatten(),
            stimuli_C_format=formats.input_format,
            tile_count_C=tile_cnt_C,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result
    res_from_L1 = res_from_L1[: len(golden)]

    assert len(res_from_L1) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"

    golden_tensor = torch.tensor(
        golden,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format in [DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )
    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format in [DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )

    # laneJO witness-check hook (paired with LANEJO_SRC_OVERRIDE above).
    _lanejo_dump = _lanejo_os.environ.get("LANEJO_DUMP")
    if _lanejo_dump:
        torch.save(
            {"src_A": src_A, "src_B": src_B, "src_C": src_C, "result": res_tensor},
            _lanejo_dump,
        )
    if _lanejo_os.environ.get("LANEJO_SKIP_ASSERT") == "1":
        return

    assert torch_equal_nan(golden_tensor, res_tensor), "Assert against golden failed"


@pytest.mark.parametrize(
    "ttnn_where_impl,label", [(0, "handwritten_macro_replay"), (1, "generated_sfpi")]
)
def test_ttnn_where_device_profile(perf_report, ttnn_where_impl, label):
    """Measure identical Float16_b where bodies with device timestamps."""
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    input_dimensions = [64, 64]
    tile_count = 4

    # Deterministic alternating condition and distinguishable operands make
    # this useful as a correctness workload as well as a stable profile shape.
    condition = (torch.arange(64 * 64) % 2).view(64, 64).to(torch.bfloat16)
    true_value = torch.full(input_dimensions, 2.0, dtype=torch.bfloat16)
    false_value = torch.full(input_dimensions, 11.0, dtype=torch.bfloat16)

    configuration = PerfConfig(
        "sources/sfpu_ternary_test.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            SFPU_TERNARY_OP(MathOperation.TTNNWhere),
            SFPU_TERNARY_SCALAR(_SCALAR_VALUE_BITS),
            APPROX_MODE(ApproximationMode.No),
            DISABLE_SRC_ZERO_FLAG(True),
            DEST_SYNC(),
            TTNNWhereImplTemplate(ttnn_where_impl),
        ],
        runtimes=[NUM_BLOCKS(tile_count), NUM_TILES_IN_BLOCK(1)],
        variant_stimuli=StimuliConfig(
            condition.flatten(),
            formats.input_format,
            true_value.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
            buffer_C=false_value.flatten(),
            stimuli_C_format=formats.input_format,
            tile_count_C=tile_count,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
    )
    configuration.run(perf_report, run_count=1)
    rows = perf_report.frame()
    rows = rows[rows["marker"] == "TTNN_WHERE_BODY"]
    assert len(rows) >= 1, rows.to_string(index=False)
    cycles = float(rows["mean(MATH_ISOLATE)"].sum())
    assert cycles > 0
    print(f"TTNN_WHERE_DEVICE_PROFILE impl={label} body_cycles={cycles:.2f}")


@pytest.mark.parametrize(
    "ttnn_where_impl,label", [(0, "handwritten_macro_replay"), (1, "generated_sfpi")]
)
def test_ttnn_where_int32_device_profile(perf_report, ttnn_where_impl, label):
    """Measure identical Int32 where bodies with device timestamps.

    The Int32/compact measurement vehicle the 2026-08-17 where adjudication
    flagged as missing (VERDICT.md harness follow-up): the compact 3-slot
    separator-absorbed calendar (misc 0x770) is the silicon-green formed path,
    but only the fp16b (formerly 4-slot) shape had a profile node. Same idiom
    as test_ttnn_where_device_profile: deterministic alternating condition,
    distinguishable operands, TTNN_WHERE_BODY / MATH_ISOLATE.
    """
    formats = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)
    input_dimensions = [64, 64]
    tile_count = 4

    condition = (torch.arange(64 * 64) % 2).view(64, 64).to(torch.int32)
    true_value = torch.full(input_dimensions, 2, dtype=torch.int32)
    false_value = torch.full(input_dimensions, 11, dtype=torch.int32)

    configuration = PerfConfig(
        "sources/sfpu_ternary_test.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            SFPU_TERNARY_OP(MathOperation.TTNNWhere),
            SFPU_TERNARY_SCALAR(_SCALAR_VALUE_BITS),
            APPROX_MODE(ApproximationMode.No),
            DISABLE_SRC_ZERO_FLAG(True),
            DEST_SYNC(),
            TTNNWhereImplTemplate(ttnn_where_impl),
        ],
        runtimes=[NUM_BLOCKS(tile_count), NUM_TILES_IN_BLOCK(1)],
        variant_stimuli=StimuliConfig(
            condition.flatten(),
            formats.input_format,
            true_value.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
            buffer_C=false_value.flatten(),
            stimuli_C_format=formats.input_format,
            tile_count_C=tile_count,
        ),
        unpack_to_dest=True,
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
    )
    configuration.run(perf_report, run_count=1)
    rows = perf_report.frame()
    rows = rows[rows["marker"] == "TTNN_WHERE_BODY"]
    assert len(rows) >= 1, rows.to_string(index=False)
    cycles = float(rows["mean(MATH_ISOLATE)"].sum())
    assert cycles > 0
    print(f"TTNN_WHERE_INT32_DEVICE_PROFILE impl={label} body_cycles={cycles:.2f}")


# MCW test with dynamic format sweeping like main test
# Use same input/output format - no mixing
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Int32,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=MathOperation.TTNNWhere,
)
def test_ttnn_where_mcw(
    formats,
    dest_acc,
    mathop,
):
    # Multi-tile tensor dimensions (2x2 tiles of 32x32).
    height = 64
    width = 64

    # Generate dtype dynamically based on current input format

    if (
        formats.input == DataFormat.Float32 and formats.output == DataFormat.Float32
    ) and dest_acc == DestAccumulation.No:
        pytest.skip("DataFormat.Float32 not supported with DestAccumulation.No")

    if (
        formats.input == DataFormat.Float16_b and formats.output == DataFormat.Float16_b
    ) and dest_acc == DestAccumulation.Yes:
        pytest.skip("DataFormat.Float16_b not supported with DestAccumulation.Yes")

    # Create alternating pattern for condition (0, 1, 0, 1, ...)
    pattern = torch.arange(height * width) % 2
    C = pattern.view(height, width).to(format_dict[formats.input_format])

    # Set specific values for true and false tensors
    T = torch.ones(height, width, dtype=format_dict[formats.input_format]) * 2
    F = torch.ones(height, width, dtype=format_dict[formats.input_format]) * 11

    golden_generator = get_golden_generator(WhereGolden)
    golden = golden_generator(C, T, F)
    tile_count = height * width // (32 * 32)

    configuration = TestConfig(
        "sources/sfpu_ternary_test.cpp",
        formats,
        templates=[
            SFPU_TERNARY_OP(mathop),
            SFPU_TERNARY_SCALAR(_SCALAR_VALUE_BITS),
            APPROX_MODE(ApproximationMode.No),
            DISABLE_SRC_ZERO_FLAG(True),
            DEST_SYNC(),
        ],
        runtimes=[NUM_BLOCKS(tile_count), NUM_TILES_IN_BLOCK(1)],
        variant_stimuli=StimuliConfig(
            C.flatten(),
            formats.input_format,
            T.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
            buffer_C=F.flatten(),
            stimuli_C_format=formats.input_format,
            tile_count_C=tile_count,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result
    res_from_L1 = res_from_L1[: len(golden)]

    golden_tensor = torch.tensor(
        golden,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format in [DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )

    golden_tensor = golden_tensor.flatten()

    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format in [DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )

    assert len(res_tensor) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"
    assert torch_equal_nan(golden_tensor, res_tensor), "Assert against golden failed"
