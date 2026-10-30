# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
``dbg_get_array_row(dbg_array_id::SRCA, ...)`` borrows dest row 0 to stage the SrcA row it
cannot read directly, and must put that row back before returning. The row is 32 datums wide
and the SFPU saves it in two halves, so it needs two registers; sharing one loses the first
half and the restore writes the surviving half over both.

Every in-tree caller passes ``dbg_array_id::DEST``, which never enters that path, so without
this test the save/restore can regress undetected. The kernel writes a known tile into DEST,
calls the helper, and reads DEST back -- the tile must come back bit-exact with either DEST width.
Row 0 carries planted patterns a converting save would change: integer values that read as float
denormals, and float denormals and negative zero, which a flush or a half-width save alters.
"""

import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig

# Int32 bit patterns a converting save would change: an FP32-mode save flushes the ones that read as
# denormals (exponent 0) and a BF16-mode save drops their low half. Planted in dest row 0 so the
# check does not depend on the random stimuli happening to include one.
INT32_AT_RISK = [
    1,
    0x7FFFFF,
    0x00373F0B,
    -1,
    -0x7FFFFFFF,
    0x12345678,
    0x00008000,
    0x7FFFFFFF,
]
# Float bit patterns with the same property: denormals (a converting save flushes them), negative zero
# (a float compare cannot tell it from zero, so the check below is bitwise), the smallest normal and
# the largest finite value as controls.
FP32_AT_RISK_BITS = [
    0x00000001,
    0x007FFFFF,
    0x80000001,
    0x80000000,
    0x00800000,
    0x7F7FFFFF,
]
BF16_AT_RISK_BITS = [0x0001, 0x007F, 0x8001, 0x8000, 0x0080, 0x7F7F]


def as_float_bits(bits, width):
    """A float tensor whose raw bits are the given patterns (sign-bit patterns overflow a signed literal)."""
    signed = [b - (1 << width) if b >= 1 << (width - 1) else b for b in bits]
    if width == 32:
        return torch.tensor(signed, dtype=torch.int32).view(torch.float32)
    return torch.tensor(signed, dtype=torch.int16).view(torch.bfloat16)


def bit_view(tensor):
    """The tensor's raw bits as a same-width integer tensor, so negative zero and NaN compare exactly."""
    return tensor.view(torch.int16 if tensor.dtype == torch.bfloat16 else torch.int32)


# The RISC-DEST debug window is only available on Blackhole.
@blackhole_only
@parametrize(
    formats=input_output_formats(
        [DataFormat.Float32, DataFormat.Float16_b, DataFormat.Int32],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_srca_row_dump_restores_dest_row_0(formats, dest_acc):
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    if formats.input_format == DataFormat.Int32:
        src_A[: len(INT32_AT_RISK)] = torch.tensor(INT32_AT_RISK, dtype=src_A.dtype)
    elif formats.input_format == DataFormat.Float32:
        src_A[: len(FP32_AT_RISK_BITS)] = as_float_bits(FP32_AT_RISK_BITS, 32)
    elif formats.input_format == DataFormat.Float16_b:
        src_A[: len(BF16_AT_RISK_BITS)] = as_float_bits(BF16_AT_RISK_BITS, 16)

    configuration = TestConfig(
        "sources/dbg_get_array_row_srca_test.cpp",
        formats,
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert (
        len(res_from_L1) == src_A.numel()
    ), f"Result tensor length {len(res_from_L1)} does not match source length {src_A.numel()}"

    torch_format = format_dict[formats.output_format]
    expected = src_A.to(torch_format)
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # Only row 0 is at risk -- the helper borrows that row and nothing else -- so report on it
    # directly rather than letting a whole-tile mismatch hide which half came back wrong.
    differing = (
        (bit_view(expected) != bit_view(res_tensor)).nonzero().flatten().tolist()
    )
    assert not differing, (
        "dbg_get_array_row(SRCA) did not restore the dest row it borrowed. "
        f"{len(differing)} datum(s) differ, first at index {differing[0]}: "
        f"expected {expected[differing[0]]}, got {res_tensor[differing[0]]}. "
        "The save must hold both halves of the row, without converting them."
    )
