# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import struct

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import ScalarBinopGolden, get_golden_generator
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    FRESH_CPP_IMPL,
    SFPU_BINOP_MODE,
    SFPU_UNARY_SCALAR,
)
from helpers.utils import passed_test


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


# The scalar is a swept axis: zero, unity, a sign flip, a large multiplier, and a value
# small enough to matter against the tolerance. Kept deliberately small -- inputs are
# uniform(-1, 1), so |scalar| <= 8 keeps every op's result inside the range where the
# default bf16 tolerance is meaningful.
#
# Split across two tests rather than swept in one: the full axis is 6 scalars x 5 ops x
# 2 formats x 2 dest modes, which is more hardware variants than presubmit should spend on
# one kernel parameter. Presubmit drives the ops at a single representative scalar and the
# remaining values run nightly.
_PRESUBMIT_SCALAR = 2.0
_SCALARS = (0.0, 1.0, 2.0, -2.0, 8.0, 0.25)
_NIGHTLY_SCALARS = tuple(s for s in _SCALARS if s != _PRESUBMIT_SCALAR)

# ScalarDiv is the one op whose scalar is not the value the kernel sees: the host inverts the
# divisor at compile time and the kernel only multiplies, so `d` never reaches the device.
# That also means a divide-by-zero cannot be reached through this op at all -- 1/0 would be
# computed on the host -- so 0.0 is not a legal divisor here rather than an untested edge.
_ZERO_DIVISOR_UNREACHABLE = (
    "ScalarDiv inverts the divisor on the host; d=0 is not a device path"
)


def _scalar_bits_for(mathop, scalar):
    """The 32-bit pattern the kernel is given for *mathop* at *scalar*."""
    if mathop == MathOperation.ScalarDiv:
        return _bits(1.0 / scalar)
    return _bits(scalar)


# Keep inputs small and bounded so the bf16 result stays accurate across all five scalar
# ops (add/sub/mul/div/rsub) and both dest-accumulation modes.
_DEFAULT_TENSOR_SPEC = StimuliSpec.uniform(low=-1.0, high=1.0)


def _run_sfpu_binop_scalar(
    formats,
    dest_acc,
    mathop,
    scalar=_PRESUBMIT_SCALAR,
    input_dimensions=[32, 32],
    spec_A=None,
    fresh_cpp_impl=None,
):
    """Drive one scalar binop variant.

    *spec_A* overrides the tensor operand. The scalar axis has been swept since the
    presubmit/nightly split, but the tensor operand had no knob at all and was pinned to
    the default above, so the only way to reach an edge on it was to edit this function.
    """
    torch.manual_seed(0)
    scalar_bits = _scalar_bits_for(mathop, scalar)

    spec_a = _DEFAULT_TENSOR_SPEC if spec_A is None else spec_A

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_a,
    )

    generate_golden = get_golden_generator(ScalarBinopGolden)
    golden = generate_golden(mathop, src_A, scalar_bits, formats.output_format)

    configuration = TestConfig(
        "sources/sfpu_binop_scalar_test.cpp",
        formats,
        templates=([] if fresh_cpp_impl is None else [FRESH_CPP_IMPL(fresh_cpp_impl)])
        + [
            SFPU_BINOP_MODE(mathop),
            SFPU_UNARY_SCALAR(scalar_bits),
            APPROX_MODE(ApproximationMode.No),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A.flatten(),
            formats.input_format,
            src_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    # laneJN bit-exact sweep hook (corpus/tools/bitexact_sweep.py) — same
    # env-gated contract as test_sfpu_unary.py's eltwise_unary_sfpu.
    import os as _os

    _lanejn_raw_a = _os.environ.get("LANEJN_RAW_A")
    if _lanejn_raw_a:
        from pathlib import Path as _Path

        configuration.variant_stimuli.lanejn_raw_a = _Path(_lanejn_raw_a).read_bytes()

    res_from_L1 = configuration.run().result

    _lanejn_dump = _os.environ.get("LANEJN_DUMP")
    if _lanejn_dump:
        import numpy as _np

        _stim = configuration.variant_stimuli
        _np.savez(
            _lanejn_dump,
            src_raw=_np.frombuffer(
                getattr(_stim, "lanejn_src_a_raw", b""), dtype=_np.uint8
            ),
            res_raw=_np.frombuffer(
                getattr(_stim, "lanejn_raw_reads", {}).get("Res", b""),
                dtype=_np.uint8,
            ),
            meta=_np.array(
                [
                    f"{mathop}(scalar_bits=0x{scalar_bits:08x})",
                    str(fresh_cpp_impl),
                    str(formats.input_format.name),
                    str(formats.output_format.name),
                    str(input_dimensions),
                    str(_stim.tile_count_A),
                    str(_stim.tile_count_res),
                    str(dest_acc),
                    "-",
                ]
            ),
        )

    res_from_L1 = res_from_L1[:1024]

    assert len(res_from_L1) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"

    if _os.environ.get("LANEJN_SKIP_ASSERT") == "1":
        return

    torch_format = format_dict[formats.output_format]
    golden_tensor = torch.tensor(golden, dtype=torch_format).flatten()
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).flatten()

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


_SCALAR_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float32,
    ],
    same=True,
)

_SCALAR_OPS = [
    MathOperation.ScalarAdd,
    MathOperation.ScalarSub,
    MathOperation.ScalarMul,
    MathOperation.ScalarDiv,
    MathOperation.ScalarRsub,
]


def _skip_unsupported(formats, dest_acc, mathop, scalar):
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")
    if (
        formats.input_format == DataFormat.Float16_b
        and dest_acc == DestAccumulation.Yes
    ):
        pytest.skip("Float16_b not supported with DestAccumulation.Yes")
    if mathop == MathOperation.ScalarDiv and scalar == 0.0:
        pytest.skip(_ZERO_DIVISOR_UNREACHABLE)


@parametrize(
    formats=_SCALAR_FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=_SCALAR_OPS,
)
def test_sfpu_binop_scalar(formats, dest_acc, mathop):
    _skip_unsupported(formats, dest_acc, mathop, _PRESUBMIT_SCALAR)
    _run_sfpu_binop_scalar(formats, dest_acc, mathop, scalar=_PRESUBMIT_SCALAR)


@pytest.mark.nightly
@parametrize(
    formats=_SCALAR_FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=_SCALAR_OPS,
    scalar=list(_NIGHTLY_SCALARS),
)
def test_sfpu_binop_scalar_values(formats, dest_acc, mathop, scalar):
    """The rest of the scalar axis: zero, unity, a sign flip and a fractional multiplier."""
    _skip_unsupported(formats, dest_acc, mathop, scalar)
    _run_sfpu_binop_scalar(formats, dest_acc, mathop, scalar=scalar)


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    dest_acc=[DestAccumulation.No],
    mathop=[MathOperation.ScalarAdd],
    fresh_cpp_impl=[0, 1],
)
def test_fresh_cpp_binop_scalar(formats, dest_acc, mathop, fresh_cpp_impl):
    """Storm lane S1: A/B the fresh ScalarAdd (x + s, scalar decoded from the
    same raw fp32 bits the production dispatch sends) against the production
    calculate_binop_with_scalar — identical stimuli, scalar, golden, and
    format-aware tolerance gate."""
    _run_sfpu_binop_scalar(
        formats,
        dest_acc,
        mathop,
        scalar=_PRESUBMIT_SCALAR,
        fresh_cpp_impl=fresh_cpp_impl,
    )


# Not swept here yet: edge values on the *tensor* operand. All five ops are
# x (+|-|*|/) c for a compile-time c, which is smooth in x -- no pole, no knee -- so cat A
# and cat D contribute nothing and edge_spec() returns None for every one of them. What is
# left is cat B, gated per op on SPECIALS_READY_OPS, which is empty until the goldens
# define a result for non-finite inputs.
#
# A wrapper written now would therefore skip every variant it collected: nightly runtime
# and a test name that reads like protection, with no executable assertion behind it. What
# it needed to exist was the spec_A hook on _run_sfpu_binop_scalar, and that is in place —
# so add the wrapper in the commit that makes the first scalar golden specials-ready,
# where its skips turn into runs:
#
#     @pytest.mark.nightly
#     @parametrize(formats=_SCALAR_FORMATS, dest_acc=[...], mathop=_SCALAR_OPS)
#     def test_sfpu_binop_scalar_edges(formats, dest_acc, mathop):
#         _skip_unsupported(formats, dest_acc, mathop, _PRESUBMIT_SCALAR)
#         specials = mathop in SPECIALS_READY_OPS and specials_safe(
#             formats.input_format, formats.output_format, dest_acc
#         )
#         spec_A = edge_spec(mathop, formats.input_format, formats.output_format,
#                            specials=specials)
#         ...
#         _run_sfpu_binop_scalar(formats, dest_acc, mathop,
#                                scalar=_PRESUBMIT_SCALAR, spec_A=spec_A)
#
# Also deliberately out of scope there: |scalar| > 8 and the +/-tiny, +/-large tensor
# values. Both need a per-op tolerance first -- the default bf16 tolerance is only
# meaningful while the result stays in range -- which is its own piece of work.
