# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Quasar tests for unary kernels whose Blackhole implementation uses SFPI.

Each variant includes exactly one Quasar kernel header and drives it through the
normal Quasar unpack/SFPU/pack pipeline. A case skips while that header is absent,
then activates automatically when the corresponding implementation lands.
"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    ApproximationMode,
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    input_output_formats,
    is_invalid_quasar_sfpu_format_combination,
    parametrize,
    runtime,
)
from helpers.sfpu_domains import exclude_undefined, for_op_pipeline
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    NUM_FACES,
    PERF_RUN_TYPE,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TYPECAST_FORMATS,
    UNPACKER_ENGINE_SEL,
    PerfRunType,
    TemplateParameter,
)
from helpers.tile_constants import MAX_NUM_FACES
from helpers.utils import passed_test

_CPP_SOURCE = "sources/quasar/eltwise_unary_sfpu_quasar_test.cpp"
_QSR_SFPU = "../../../../hw/ckernels/quasar/metal/llk_api/llk_sfpu"
_QSR_SFPU_DIR = (
    Path(__file__).resolve().parents[4] / "hw/ckernels/quasar/metal/llk_api/llk_sfpu"
)
_FLOAT_FORMATS = input_output_formats(
    [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32], same=True
)


@dataclass(frozen=True)
class SfpiUnaryCase:
    mathop: MathOperation
    kernel: str
    function: str
    template_args: tuple[str, ...]
    init: str = "(void)0"
    runtime_args: tuple[str, ...] = ()
    formats: tuple[InputOutputFormat, ...] = tuple(_FLOAT_FORMATS)
    integer: bool = False

    def __repr__(self) -> str:
        return f"{self.kernel}:{self.mathop.name}"


@dataclass
class SFPI_COMPAT_KERNEL(TemplateParameter):
    """Emit the test-only include and direct SFPU dispatch for one SFPI body."""

    case: SfpiUnaryCase

    def convert_to_cpp(self) -> str:
        template_args = ", ".join(self.case.template_args)
        call_args = "".join(f", {arg}" for arg in self.case.runtime_args)
        call = (
            "SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, "
            f"{self.case.function}, ({template_args}), dst_index, VectorMode::RC"
            f"{call_args})"
        )
        return "\n".join(
            [
                "#define SFPI_COMPAT_TEST",
                f'#define SFPI_COMPAT_HEADER "{_QSR_SFPU}/ckernel_sfpu_{self.case.kernel}.h"',
                f"#define SFPI_COMPAT_INIT() {self.case.init}",
                f"#define SFPI_COMPAT_CALL(dst_index) {call}",
            ]
        )


def _case(
    mathop,
    kernel,
    function,
    *template_args,
    init="(void)0",
    runtime_args=(),
    formats=None,
    integer=False,
):
    return SfpiUnaryCase(
        mathop=mathop,
        kernel=kernel,
        function=function,
        template_args=tuple(template_args),
        init=init,
        runtime_args=tuple(runtime_args),
        formats=tuple(_FLOAT_FORMATS if formats is None else formats),
        integer=integer,
    )


_INT32_FORMAT = (InputOutputFormat(DataFormat.Int32, DataFormat.Int32),)
_FP32_TO_FP16A = (InputOutputFormat(DataFormat.Float32, DataFormat.Float16),)


# One entry for every callable unary operation in the missing portable-SFPI
# kernel set.  Helper-only headers (for example ckernel_sfpu_conversions.h) are
# exercised through the kernels that consume them rather than as fake ops.
SFPI_UNARY_CASES = (
    _case(
        MathOperation.Hardsigmoid,
        "activations",
        "calculate_activation",
        "APPROX_MODE",
        "ckernel::ActivationType::Hardsigmoid",
        "SFPU_ITERATIONS",
        init="hardsigmoid_init<APPROX_MODE>()",
    ),
    _case(
        MathOperation.Add1, "add1", "calculate_add1", "APPROX_MODE", "SFPU_ITERATIONS"
    ),
    _case(
        MathOperation.BitwiseNot,
        "bitwise_not",
        "calculate_bitwise_not",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        formats=_INT32_FORMAT,
        integer=True,
    ),
    _case(
        MathOperation.CastFp32ToFp16a,
        "cast_fp32_to_fp16a",
        "cast_fp32_to_fp16a",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        formats=_FP32_TO_FP16A,
    ),
    _case(
        MathOperation.Cbrt,
        "cbrt",
        "calculate_cube_root",
        "APPROX_MODE",
        "is_fp32_dest_acc_en",
        "SFPU_ITERATIONS",
        init="cube_root_init<APPROX_MODE>()",
    ),
    _case(
        MathOperation.Celu,
        "celu",
        "calculate_celu",
        "APPROX_MODE",
        "is_fp32_dest_acc_en",
        "SFPU_ITERATIONS",
        init="celu_init()",
        runtime_args=("0x3f800000u", "0x3f800000u"),
    ),
    _case(
        MathOperation.Digamma,
        "digamma",
        "calculate_digamma",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        init="digamma_init<APPROX_MODE>()",
    ),
    _case(
        MathOperation.Elu,
        "elu",
        "calculate_elu",
        "APPROX_MODE",
        "is_fp32_dest_acc_en",
        "SFPU_ITERATIONS",
        init="elu_init()",
        runtime_args=("0x3f800000u",),
    ),
    _case(
        MathOperation.Erf,
        "erf",
        "calculate_erf",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        init="erf_init<APPROX_MODE>()",
    ),
    _case(
        MathOperation.Erfc,
        "erfc",
        "calculate_erfc",
        "SFPU_ITERATIONS",
        init="erfc_init<APPROX_MODE>()",
    ),
    _case(
        MathOperation.Erfinv,
        "erfinv",
        "calculate_erfinv",
        "APPROX_MODE",
        init="erfinv_init<APPROX_MODE>()",
    ),
    _case(
        MathOperation.Exp2,
        "exp2",
        "calculate_exp2",
        "APPROX_MODE",
        "is_fp32_dest_acc_en",
        "SFPU_ITERATIONS",
        init="exp2_init<APPROX_MODE, is_fp32_dest_acc_en>()",
    ),
    _case(
        MathOperation.Expm1,
        "expm1",
        "calculate_expm1",
        "APPROX_MODE",
        "is_fp32_dest_acc_en",
        "SFPU_ITERATIONS",
        init="expm1_init<APPROX_MODE, is_fp32_dest_acc_en>()",
    ),
    _case(
        MathOperation.Fmod,
        "fmod",
        "calculate_fmod",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        init="init_fmod<APPROX_MODE>(0x40000000u, 0x3f000000u)",
    ),
    _case(
        MathOperation.Hardmish,
        "hardmish",
        "hardmish",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        init="hardmish_init()",
    ),
    _case(
        MathOperation.Hardshrink,
        "hardshrink",
        "calculate_hardshrink",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        init="hardshrink_init()",
        runtime_args=("0x3f000000u",),
    ),
    _case(
        MathOperation.Hardtanh,
        "hardtanh",
        "calculate_hardtanh",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        init="hardtanh_init()",
        runtime_args=("0xbf800000u", "0x3f800000u"),
    ),
    _case(
        MathOperation.Heaviside,
        "heaviside",
        "calculate_heaviside",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        init="heaviside_init()",
        runtime_args=("0x3f000000u",),
    ),
    _case(
        MathOperation.I0,
        "i0",
        "calculate_i0",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        init="i0_init()",
    ),
    _case(
        MathOperation.I1,
        "i1",
        "calculate_i1",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        init="i1_init<APPROX_MODE>()",
    ),
    _case(
        MathOperation.Identity,
        "identity",
        "calculate_identity",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
    ),
    _case(
        MathOperation.Lgamma,
        "lgamma",
        "calculate_lgamma_stirling",
        "APPROX_MODE",
        "is_fp32_dest_acc_en",
        "SFPU_ITERATIONS",
        init="lgamma_stirling_init<APPROX_MODE, is_fp32_dest_acc_en>()",
    ),
    _case(
        MathOperation.LogicalNotUnary,
        "logical_not",
        "calculate_logical_not",
        "APPROX_MODE",
        "ckernel::InstrModLoadStore::DEFAULT",
        "SFPU_ITERATIONS",
        init="logical_not_unary_init()",
    ),
    _case(
        MathOperation.Polygamma,
        "polygamma",
        "calculate_polygamma",
        "APPROX_MODE",
        "is_fp32_dest_acc_en",
        "SFPU_ITERATIONS",
        init="polygamma_init<APPROX_MODE, is_fp32_dest_acc_en>()",
        runtime_args=("0x3f800000u", "0x3f800000u"),
    ),
    _case(
        MathOperation.Prelu,
        "prelu",
        "calculate_prelu",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        init="prelu_init()",
        runtime_args=("0x3e800000u",),
    ),
    _case(
        MathOperation.Rdiv,
        "rdiv",
        "calculate_rdiv",
        "APPROX_MODE",
        "is_fp32_dest_acc_en",
        "ckernel::RoundingMode::None",
        "SFPU_ITERATIONS",
        init="rdiv_init<APPROX_MODE>()",
        runtime_args=("0x40000000u",),
    ),
    _case(
        MathOperation.Remainder,
        "remainder",
        "calculate_remainder",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        init="init_remainder<APPROX_MODE>(0x40000000u, 0x3f000000u)",
    ),
    _case(
        MathOperation.Rpow,
        "rpow",
        "calculate_rpow",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        "is_fp32_dest_acc_en",
        init="sfpu_binary_pow_init<APPROX_MODE>()",
        runtime_args=("0x40000000u",),
    ),
    _case(
        MathOperation.Selu,
        "selu",
        "calculate_selu",
        "APPROX_MODE",
        "is_fp32_dest_acc_en",
        "SFPU_ITERATIONS",
        init="selu_init()",
        runtime_args=("0x3f867d5fu", "0x3fd62d7du"),
    ),
    _case(
        MathOperation.Sign,
        "sign",
        "calculate_sign",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        init="sign_init()",
        runtime_args=("0u",),
    ),
    _case(
        MathOperation.Softshrink,
        "softshrink",
        "calculate_softshrink",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        init="softshrink_init()",
        runtime_args=("0x3f000000u",),
    ),
    _case(
        MathOperation.Softsign,
        "softsign",
        "calculate_softsign",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        init="init_softsign<APPROX_MODE>()",
    ),
    _case(
        MathOperation.Tanhshrink,
        "tanhshrink",
        "calculate_tanhshrink",
        "is_fp32_dest_acc_en",
        "SFPU_ITERATIONS",
        init="tanhshrink_init<APPROX_MODE, is_fp32_dest_acc_en>()",
    ),
    *(
        _case(
            mathop,
            "unary_comp",
            function,
            "APPROX_MODE",
            "SFPU_ITERATIONS",
            init=f"{init}()",
            runtime_args=("0x3f000000u",),
        )
        for mathop, function, init in (
            (MathOperation.UnaryGt, "calculate_unary_gt", "unary_gt_init"),
            (MathOperation.UnaryLt, "calculate_unary_lt", "unary_lt_init"),
            (MathOperation.UnaryGe, "calculate_unary_ge", "unary_ge_init"),
            (MathOperation.UnaryLe, "calculate_unary_le", "unary_le_init"),
            (MathOperation.UnaryEq, "calculate_unary_eq", "unary_eq_init"),
            (MathOperation.UnaryNe, "calculate_unary_ne", "unary_ne_init"),
        )
    ),
    _case(
        MathOperation.UnaryPower,
        "unary_power",
        "calculate_unary_power",
        "APPROX_MODE",
        "is_fp32_dest_acc_en",
        "SFPU_ITERATIONS",
        init="sfpu_unary_pow_init()",
        runtime_args=("0x40000000u",),
    ),
    _case(
        MathOperation.LeftShift,
        "unary_shift",
        "calculate_left_shift",
        "APPROX_MODE",
        "DataFormat::Int32",
        "SFPU_ITERATIONS",
        init="left_shift_init()",
        runtime_args=("3u",),
        formats=_INT32_FORMAT,
        integer=True,
    ),
    _case(
        MathOperation.RightShift,
        "unary_shift",
        "calculate_right_shift",
        "APPROX_MODE",
        "DataFormat::Int32",
        "SFPU_ITERATIONS",
        init="right_shift_init()",
        runtime_args=("3u",),
        formats=_INT32_FORMAT,
        integer=True,
    ),
    _case(
        MathOperation.Xielu,
        "xielu",
        "calculate_xielu",
        "APPROX_MODE",
        "is_fp32_dest_acc_en",
        "SFPU_ITERATIONS",
        init="xielu_init<APPROX_MODE>()",
        runtime_args=("0x3f800000u", "0x3f800000u"),
    ),
)


def _input_spec(case, formats):
    if case.integer:
        if case.mathop in (MathOperation.LeftShift, MathOperation.RightShift):
            return StimuliSpec.uniform(low=0.0, high=1_000_000.0)
        return StimuliSpec.uniform(low=-1_000_000.0, high=1_000_000.0)
    if case.mathop in (
        MathOperation.LogicalNotUnary,
        MathOperation.Heaviside,
        MathOperation.UnaryEq,
        MathOperation.UnaryNe,
    ):
        return StimuliSpec.custom(values=[-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
    return exclude_undefined(
        case.mathop,
        for_op_pipeline(
            case.mathop, formats.input_format, formats.output_format
        ).spec_A,
    )


def _valid_dest_accs(case, formats):
    if case.integer or formats.input_format.is_32_bit():
        return (DestAccumulation.Yes,)
    return (DestAccumulation.No, DestAccumulation.Yes)


def _generate_combinations():
    combinations = []
    for case in SFPI_UNARY_CASES:
        for formats in case.formats:
            for dest_acc in _valid_dest_accs(case, formats):
                unpack_to_dest = (
                    formats.input_format.is_32_bit()
                    and dest_acc == DestAccumulation.Yes
                )
                if is_invalid_quasar_sfpu_format_combination(
                    formats, dest_acc, unpack_to_dest
                ):
                    continue
                for dest_sync, implied_math_format, input_dimensions in product(
                    (DestSync.Half, DestSync.Full),
                    (ImpliedMathFormat.No, ImpliedMathFormat.Yes),
                    ([32, 32], [64, 64]),
                ):
                    combinations.append(
                        (
                            case,
                            formats,
                            dest_acc,
                            dest_sync,
                            implied_math_format,
                            runtime(input_dimensions),
                        )
                    )
    return combinations


@pytest.mark.quasar
@parametrize(case_formats_dest_acc_sync_implied_dims=_generate_combinations())
def test_sfpi_compat_unary_quasar(case_formats_dest_acc_sync_implied_dims):
    """Validate portable unary SFPI bodies across Quasar's supported format matrix."""
    (
        case,
        formats,
        dest_acc,
        dest_sync,
        implied_math_format,
        input_dimensions,
    ) = case_formats_dest_acc_sync_implied_dims[0]

    header = _QSR_SFPU_DIR / f"ckernel_sfpu_{case.kernel}.h"
    if not header.is_file():
        pytest.skip(f"Quasar SFPI kernel is not implemented yet: {header.name}")

    spec = _input_spec(case, formats)
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec,
        spec_B=spec,
    )

    if case.integer:
        values = src_A.flatten().to(torch.int32)
        if case.mathop == MathOperation.BitwiseNot:
            golden_tensor = torch.bitwise_not(values)
        elif case.mathop == MathOperation.LeftShift:
            golden_tensor = torch.bitwise_left_shift(values, 3)
        elif case.mathop == MathOperation.RightShift:
            golden_tensor = torch.bitwise_right_shift(values, 3)
        else:
            raise AssertionError(f"Missing integer golden for {case.mathop.name}")
    else:
        generate_golden = get_golden_generator(UnarySFPUGolden)
        golden_tensor = generate_golden(
            case.mathop,
            src_A,
            formats.output_format,
            dest_acc,
            formats.input_format,
            input_dimensions,
        )

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )
    configuration = TestConfig(
        _CPP_SOURCE,
        formats,
        templates=[
            SFPI_COMPAT_KERNEL(case),
            APPROX_MODE(ApproximationMode.No),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(dest_sync),
            TYPECAST_FORMATS(),
            PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(MAX_NUM_FACES),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
            LOOP_FACTOR(1),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=MAX_NUM_FACES,
            twos_complement=case.integer,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_l1 = configuration.run().result
    assert len(res_from_l1) == len(golden_tensor)
    result = torch.tensor(res_from_l1, dtype=format_dict[formats.output_format])
    assert passed_test(golden_tensor, result, formats.output_format)
