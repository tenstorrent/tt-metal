# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Quasar scalar binop kernels (add/sub/mul/div/rsub_unary_tile, add/sub_unary_tile_int32),
held bit-exact against a golden that models SFPSTORE truncation and RSUB's bf16 rounding.
"""

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ScalarBinopGolden,
    get_golden_generator,
    quantize_input_to_unpack_format,
    truncate_to_bfloat16,
)
from helpers.llk_params import (
    ApproximationMode,
    DataCopyType,
    DestAccumulation,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    QuasarSfpuVariant,
    generate_quasar_sfpu_format_variants,
    generate_reduced_input_dimensions,
    input_output_formats,
    parametrize,
    quasar_mx_smoke,
    quasar_sfpu_fpu_route_variants,
    runtime,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    LOOP_FACTOR,
    NUM_FACES,
    SFPU_BINOP_MODE,
    SFPU_DEST_16BIT_FORMAT,
    SFPU_UNARY_SCALAR,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
    fp32_bits,
)
from helpers.tile_constants import MAX_NUM_FACES
from helpers.utils import passed_test, result_tensor

_CPP_SOURCE = "sources/quasar/sfpu_binop_scalar_quasar_test.cpp"

# The kernels ignore the approximation mode; it is pinned so every variant compiles once.
_APPROX_MODE = ApproximationMode.No

_FLOAT_OPS = [
    MathOperation.ScalarAdd,
    MathOperation.ScalarSub,
    MathOperation.ScalarMul,
    MathOperation.ScalarDiv,
    MathOperation.ScalarRsub,
]
_INT32_OPS = [MathOperation.AddInt32, MathOperation.SubInt32]

_FLOAT_FORMATS = input_output_formats(
    [DataFormat.Float16_b, DataFormat.Float32]
) + quasar_mx_smoke(DataFormat.MxFp8P, DataFormat.Float16_b)


def _runs_mx_through_unpack_to_dest(variant: QuasarSfpuVariant) -> bool:
    # MX through Unpack-to-Dest returns garbage on emu-quasar for every op; MX runs via the FPU route.
    return variant.formats.input_format.is_mx_format() and variant.unpack_to_dest


def _float_routes(formats_list):
    """Unpack-to-Dest and FPU-datacopy routes. The resolver alone never yields an FPU
    route where Unpack-to-Dest is also valid, so quasar_sfpu_fpu_route_variants adds them.
    """
    routes = []
    for variant in generate_quasar_sfpu_format_variants(
        None, formats_list, full_format_route_sweep=True
    ) + quasar_sfpu_fpu_route_variants(formats_list):
        if variant not in routes and not _runs_mx_through_unpack_to_dest(variant):
            routes.append(variant)
    return routes


# Float16 (A-family) is the one 16-bit Dest format where RSUB must not round to bf16.
_FLOAT16 = InputOutputFormat(DataFormat.Float16, DataFormat.Float16)

# Every float producer route, including one MX pair and Float16; the op sweep runs all of them.
_FLOAT_ROUTES = _float_routes(_FLOAT_FORMATS + [_FLOAT16])

# bf16 into a 16-bit Dest has both producers; the FPU one is the narrow FPU-to-SFPU handshake.
assert {
    v.unpack_to_dest
    for v in _FLOAT_ROUTES
    if v.formats.input_format == DataFormat.Float16_b
    and v.dest_acc == DestAccumulation.No
} == {
    True,
    False,
}, "the op sweep must run bf16 into a 16-bit Dest through both producers"

# One route per Dest width, for the scalar and Dest-layout sweeps.
_FLOAT_VARIANTS = generate_quasar_sfpu_format_variants(None, _FLOAT_FORMATS)
_INT32_VARIANTS = generate_quasar_sfpu_format_variants(
    None, input_output_formats([DataFormat.Int32], same=True)
)

# |scalar| <= 8 on uniform(-1, 1) inputs keeps every result well inside the bf16 range.
_SCALAR = 2.0
_MORE_SCALARS = [0.0, 1.0, -2.0, 8.0, 0.25]

# A negative scalar makes SUB add; inputs within +/-2^30 keep every result from overflowing.
_INT32_SCALARS = [1000, -5]
_INT32_INPUT_SPAN = 2**30


def _scalar_bits(mathop: MathOperation, scalar: float) -> int:
    """The 32-bit scalar the kernel is given. For DIV the host passes 1 / divisor."""
    if mathop == MathOperation.ScalarDiv:
        return fp32_bits(1.0 / scalar)
    return fp32_bits(scalar)


def _narrows_fp32_input(variant: QuasarSfpuVariant) -> bool:
    """fp32 into a 16-bit Dest is narrowed non-deterministically on emu-quasar (truncated in
    one run, rounded up in the next), so these routes allow the one ULP that moves the result.
    """
    return variant.formats.input_format.is_32_bit() and not variant.sfpu_src.is_32_bit()


def _truncate_to_float16(values: torch.Tensor) -> torch.Tensor:
    """fp32 -> Float16 toward zero, as SFPSTORE narrows into a Float16 Dest."""
    narrowed = values.to(torch.float16)
    overshoot = narrowed.to(torch.float32).abs() > values.abs()
    toward_zero = torch.nextafter(narrowed, torch.zeros_like(narrowed))
    return torch.where(overshoot, toward_zero, narrowed)


def _dest_16bit_format(variant: QuasarSfpuVariant) -> DataFormat:
    """The format a 16-bit Dest holds; Float16_b stands in when the Dest is 32-bit."""
    return DataFormat.Float16_b if variant.sfpu_dst.is_32_bit() else variant.sfpu_dst


def _float_golden(
    mathop: MathOperation,
    src_A: torch.Tensor,
    value_bits: int,
    variant: QuasarSfpuVariant,
) -> torch.Tensor:
    """SFPSTORE truncates into a 16-bit Dest, except that RSUB rounds to bf16 nearest-even on
    Float16_b; a 32-bit Dest holds fp32, which the packer rounds to nearest-even."""
    golden = get_golden_generator(ScalarBinopGolden)
    output_format = variant.formats.output_format
    src_A = quantize_input_to_unpack_format(src_A, variant.formats.input_format).to(
        torch.float32
    )
    dest_format = _dest_16bit_format(variant)
    rounds_to_bf16 = (
        mathop == MathOperation.ScalarRsub and dest_format == DataFormat.Float16_b
    )
    if variant.sfpu_dst.is_32_bit() or rounds_to_bf16:
        return golden(
            mathop, src_A, value_bits, output_format, variant.dest_acc
        ).flatten()
    exact = golden(mathop, src_A, value_bits, DataFormat.Float32, DestAccumulation.Yes)
    stored = (
        _truncate_to_float16(exact)
        if dest_format == DataFormat.Float16
        else truncate_to_bfloat16(exact)
    )
    return stored.to(format_dict[output_format]).flatten()


def _int32_golden(
    mathop: MathOperation, src_A: torch.Tensor, scalar: int
) -> torch.Tensor:
    """x + scalar or x - scalar, wrapped to int32 the way SFPIADD wraps."""
    x = src_A.flatten().to(torch.int64)
    result = x + scalar if mathop == MathOperation.AddInt32 else x - scalar
    wrapped = (result + 2**31) % 2**32 - 2**31
    return wrapped.to(torch.int32)


def _run_sfpu_binop_scalar_quasar(
    variant: QuasarSfpuVariant,
    mathop: MathOperation,
    scalar,
    input_dimensions=(32, 32),
):
    torch.manual_seed(0)
    formats = variant.formats
    is_int32 = mathop in _INT32_OPS

    if is_int32:
        spec = StimuliSpec.uniform(low=-_INT32_INPUT_SPAN, high=_INT32_INPUT_SPAN)
        value_bits = scalar & 0xFFFFFFFF
    else:
        spec = StimuliSpec.uniform(low=-1.0, high=1.0)
        value_bits = _scalar_bits(mathop, scalar)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=list(input_dimensions),
        stimuli_format_B=formats.input_format,
        input_dimensions_B=list(input_dimensions),
        spec_A=spec,
        spec_B=spec,
    )

    golden_tensor = (
        _int32_golden(mathop, src_A, scalar)
        if is_int32
        else _float_golden(mathop, src_A, value_bits, variant)
    )

    unpack_to_dest = variant.unpack_to_dest
    test_config_kwargs = {
        "test_name": _CPP_SOURCE,
        "formats": formats,
        "templates": [
            SFPU_BINOP_MODE(mathop),
            SFPU_UNARY_SCALAR(value_bits),
            SFPU_DEST_16BIT_FORMAT(_dest_16bit_format(variant)),
            APPROX_MODE(_APPROX_MODE),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(MAX_NUM_FACES),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
            LOOP_FACTOR(1),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=MAX_NUM_FACES,
            # calculate_add_int32 / calculate_sub_int32 add in two's complement.
            twos_complement=is_int32,
        ),
        "unpack_to_dest": unpack_to_dest,
        "dest_acc": variant.dest_acc,
    }
    configuration = create_test_or_perf_config(
        is_perf=False, run_types=None, test_config_kwargs=test_config_kwargs
    )
    variant.apply_formats(configuration.formats_config)

    res_from_L1 = configuration.run().result
    res_tensor = result_tensor(res_from_L1, formats.output_format, golden_tensor)

    if is_int32:
        assert torch.equal(res_tensor, golden_tensor), (
            f"{mathop.name} {scalar}: "
            f"{int((res_tensor != golden_tensor).sum())} lanes differ"
        )
        return

    # Bit-exact everywhere the input reaches the SFPU unchanged.
    max_ulp = 1 if _narrows_fp32_input(variant) else 0
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format, max_ulp=max_ulp
    )


@pytest.mark.quasar
@parametrize(format_variant=_FLOAT_ROUTES, mathop=_FLOAT_OPS)
def test_sfpu_binop_scalar_quasar(format_variant, mathop):
    """Float add/sub/mul/div/rsub with a scalar, on every Quasar SFPU route."""
    _run_sfpu_binop_scalar_quasar(format_variant, mathop, _SCALAR)


@pytest.mark.quasar
@parametrize(
    format_variant=_FLOAT_VARIANTS,
    # ADD and SUB are the same SFPMAD and DIV is a MUL, so these three cover the code paths.
    mathop=[MathOperation.ScalarSub, MathOperation.ScalarMul, MathOperation.ScalarRsub],
    scalar=_MORE_SCALARS,
)
def test_sfpu_binop_scalar_values_quasar(format_variant, mathop, scalar):
    """The rest of the scalar axis: zero, unity, a sign flip, a large multiplier, a fraction."""
    _run_sfpu_binop_scalar_quasar(format_variant, mathop, scalar)


@pytest.mark.quasar
@parametrize(
    format_variant=_FLOAT_VARIANTS,
    # The ops share one row loop; ADD and RSUB (the one with extra registers live
    # across it) cover the Dest walk.
    mathop=[MathOperation.ScalarAdd, MathOperation.ScalarRsub],
    input_dimensions=runtime(
        lambda format_variant: generate_reduced_input_dimensions(
            format_variant.dest_acc
        )
    ),
)
def test_sfpu_binop_scalar_dest_layout_quasar(format_variant, mathop, input_dimensions):
    """Every distinct Dest layout, including a matrix that walks bank 0 -> 1 -> 0."""
    _run_sfpu_binop_scalar_quasar(
        format_variant, mathop, _SCALAR, input_dimensions=input_dimensions
    )


@pytest.mark.quasar
@parametrize(
    format_variant=_INT32_VARIANTS,
    mathop=_INT32_OPS,
    scalar=_INT32_SCALARS,
)
def test_sfpu_binop_scalar_int32_quasar(format_variant, mathop, scalar):
    """Int32 add/sub with a scalar, two's complement."""
    _run_sfpu_binop_scalar_quasar(format_variant, mathop, scalar)
