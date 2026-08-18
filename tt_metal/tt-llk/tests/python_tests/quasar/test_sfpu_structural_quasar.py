# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tile-structural half of the SFPU parity set on Quasar.

Three kernels -- ``tiled_prod``, ``int_sum`` (row and column) and
``alt_complex_rotate90`` -- take one tile in and one tile out like a unary op, but are
*not* element-wise: they address Dest by slot and combine slots with each other. That is
why they sit here rather than in the consolidated unary harness, whose golden contract is
per element.

They are also the only parity ops with no Blackhole tt-llk test to mirror, so their
goldens (``StructuralSFPUGolden``) were written from the Blackhole kernel sources rather
than validated against a passing run. Until a Quasar kernel exists, a mismatch here is
evidence about the golden as much as about the kernel -- see the class docstring.

Every op is gated on its kernel header through ``helpers/sfpu_port_quasar.py``, so this
module collects nothing today.
"""

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import StructuralSFPUGolden, get_golden_generator
from helpers.llk_params import (
    ApproximationMode,
    DataCopyType,
    DestAccumulation,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    generate_sfpu_format_dest_acc_combinations,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.sfpu_port_quasar import Arity, entries, is_ported
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TYPECAST_FORMATS,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_constants import MAX_NUM_FACES
from helpers.utils import passed_test

_CPP_SOURCE = "sources/quasar/sfpu_structural_quasar_test.cpp"

# Integer-domain structural ops. int_sum accumulates in int32 Dest; the other two are
# float. Kept as a set rather than read from the port entry because int_sum's two modes
# share one entry and both are integer.
_INT_STRUCTURAL_OPS = (MathOperation.IntSumRow, MathOperation.IntSumCol)

# Per-op stimulus band. Bounds are picked so the *accumulated* result stays representable,
# which for these ops is a much tighter constraint than for an element-wise op:
#   tiled_prod  -- a running product over 8 slots, so |x| must stay near 1 or the eighth
#                  partial product overflows even fp32; [0.5, 1.5] keeps 1.5**8 ~= 25;
#   int_sum     -- sums 8 slots, so cap at 2**24 to stay well inside int32;
#   rotate90    -- a permutation with a sign flip, exact for anything representable.
_STRUCTURAL_DOMAINS = {
    MathOperation.TiledProd: (0.5, 1.5),
    MathOperation.IntSumRow: (-(2**24), 2**24),
    MathOperation.IntSumCol: (-(2**24), 2**24),
    MathOperation.AltComplexRotate90: (-8.0, 8.0),
}


def _float_formats_dest_acc():
    formats = input_output_formats(
        [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
    )
    return [
        (fmt, dest_acc)
        for fmt, dest_acc in generate_sfpu_format_dest_acc_combinations(formats)
        if not (
            fmt.input_format == DataFormat.Float16 and dest_acc == DestAccumulation.Yes
        )
    ]


def _int_formats_dest_acc():
    """int_sum runs on the I32 Dest layout, which forces a 32-bit dest."""
    return [
        (fmt, DestAccumulation.Yes)
        for fmt in input_output_formats([DataFormat.Int32], same=True)
    ]


def _get_valid_implied_math_formats(fmt: FormatConfig):
    if fmt.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]


def _is_unpack_to_dest(fmt: FormatConfig, dest_acc: DestAccumulation) -> bool:
    return fmt.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes


def _structural_cases():
    """(mathop, formats, dest_acc) for every ported structural parity op.

    Empty while the parity kernels are unported, which is the normal state today.
    """
    cases = []
    for entry in entries(Arity.STRUCTURAL):
        for mathop in entry.ops:
            if not is_ported(mathop):
                continue
            matrix = (
                _int_formats_dest_acc()
                if mathop in _INT_STRUCTURAL_OPS
                else _float_formats_dest_acc()
            )
            for fmt, dest_acc in matrix:
                cases.append((mathop, fmt, dest_acc))
    return cases


_STRUCTURAL_CASES = _structural_cases()


@pytest.mark.quasar
@pytest.mark.skipif(
    not _STRUCTURAL_CASES,
    reason="no tile-structural SFPU parity kernel is ported to Quasar yet",
)
@parametrize(
    structural_case=_STRUCTURAL_CASES,
    implied_math_format=lambda structural_case: _get_valid_implied_math_formats(
        structural_case[1]
    ),
    input_dimensions=runtime([[32, 32]]),
)
def test_sfpu_structural_quasar(structural_case, implied_math_format, input_dimensions):
    """Tile-structural SFPU parity ops on Quasar, against StructuralSFPUGolden.

    One tile in, one tile out. The sweep is pinned to a single 32x32 tile because these
    kernels' slot arithmetic is defined against exactly one tile's worth of Dest: int_sum
    reads slots in faces 1 and 2, and tiled_prod's final unrolled step reaches the slot
    after the face. Sweeping a larger input would not exercise more of the kernel, it
    would change what the kernel means.
    """
    mathop, formats, dest_acc = structural_case
    low, high = _STRUCTURAL_DOMAINS[mathop]
    is_int = mathop in _INT_STRUCTURAL_OPS

    torch.manual_seed(42)
    src_raw, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=StimuliSpec.uniform(low=0.0, high=1.0),
        spec_B=StimuliSpec.uniform(low=0.0, high=1.0),
    )
    scaled = low + src_raw.to(torch.float32) * (high - low)
    if is_int:
        scaled = scaled.round()
    src_A = scaled.to(format_dict[formats.input_format])

    num_faces = MAX_NUM_FACES

    generate_golden = get_golden_generator(StructuralSFPUGolden)
    golden_tensor = generate_golden(
        mathop, src_A, formats.output_format, num_faces=num_faces
    )

    unpack_to_dest = _is_unpack_to_dest(formats, dest_acc)

    configuration = TestConfig(
        _CPP_SOURCE,
        formats,
        templates=[
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(),
            # The shared unary dispatch in sfpu_operations_quasar.h has a typecast branch
            # referencing these non-dependent globals, so every build defining that header
            # must define them too.
            TYPECAST_FORMATS(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
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
            num_faces=num_faces,
            twos_complement=formats.input_format.is_integer(),
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format_out = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format_out)
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
