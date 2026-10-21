# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    DataCopyGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, DestSync, format_dict
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TemplateParameter,
    generate_input_dim,
)
from helpers.utils import passed_test


@dataclass
class DATACOPY_CUSTOM_PACK_RAW(TemplateParameter):
    """Pack reads DEST raw, so the bf16 output carries -0.0 and denormals exactly as the copy left them."""

    pack_raw: bool = False

    def convert_to_cpp(self) -> str:
        return f"#define DATACOPY_CUSTOM_PACK_RAW {str(self.pack_raw).lower()}"


# One [128, 256] block = 32 tiles, enough to span several DEST sections.
INPUT_DIMENSIONS = [128, 256]


def _run_unary_datacopy_custom(formats, dest_acc, src_A=None, pack_raw=False):
    """Run copy_tile_custom over INPUT_DIMENSIONS; return (src_A, golden, result) as flat tensors.

    ``src_A`` optionally replaces the seeded random stimulus (same flat shape).
    """
    input_dimensions = INPUT_DIMENSIONS
    num_faces = 4

    seeded_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    if src_A is None:
        src_A = seeded_A
    else:
        assert src_A.shape == seeded_A.shape

    generate_golden = get_golden_generator(DataCopyGolden)
    golden_tensor = generate_golden(
        src_A, formats.output_format, num_faces, input_dimensions
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, input_dimensions, TILE_DIMENSIONS
    )

    configuration = TestConfig(
        "sources/eltwise_unary_datacopy_custom_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            DATACOPY_CUSTOM_PACK_RAW(pack_raw),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    outcome = configuration.run()
    res_from_L1 = outcome.result

    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)
    return src_A, golden_tensor, res_tensor


# The op is bf16 -> bf16 only, no fp32 DEST: bind the formats inline rather than sweeping one option.
FORMATS = input_output_formats([DataFormat.Float16_b])[0]
DEST_ACC = DestAccumulation.No


@blackhole_only
def test_unary_datacopy_custom():
    _, golden_tensor, res_tensor = _run_unary_datacopy_custom(FORMATS, DEST_ACC)
    assert passed_test(golden_tensor, res_tensor, FORMATS.output_format)


# Smallest positive bf16 denormal, 2^-133; bf16 denormals are its multiples 1..127 (7 mantissa bits).
BF16_DENORMAL_MIN = 2.0**-133
BF16_DENORMAL_MANTISSAS = 1 << 7


@blackhole_only
def test_unary_datacopy_custom_preserves_signed_zero_and_denormals():
    """A copy must pass bf16 -0.0 and denormals through bit-for-bit, as copy_tile does.

    MOVA2D flushes any datum with a zero exponent byte while the Src zero flag holds the operand default
    (the hw_configure state for bf16), turning -0.0 into +0.0 and denormals into 0 in DEST, where a following
    SFPU op reads them. The custom copy must set the flag to keep. Two things would hide that: the packer's
    default bf16 rounding conversion flushes the same datums on the way to L1 (so the pack reads DEST raw
    here), and the tolerance compare treats -0.0 == +0.0 and 1e-40 ~= 0 (so this checks the raw bf16 words).
    """
    torch.manual_seed(0)
    n = INPUT_DIMENSIONS[0] * INPUT_DIMENSIONS[1]
    src_A = torch.randn(n, dtype=torch.float32)
    lanes = torch.arange(n)
    # Lane pattern per group of four: -0.0, +0.0, a denormal, a normal. Every other denormal lane
    # (lane % 8 == 2) is negative so both signs of the denormal band are covered.
    src_A[lanes % 4 == 0] = -0.0
    src_A[lanes % 4 == 1] = 0.0
    denormal_lanes = lanes % 4 == 2
    src_A[denormal_lanes] = (
        BF16_DENORMAL_MIN
        * torch.randint(
            1, BF16_DENORMAL_MANTISSAS, (int(denormal_lanes.sum()),)
        ).float()
        * torch.where(lanes[denormal_lanes] % 8 == 2, -1.0, 1.0)
    )

    src_A, _, res_tensor = _run_unary_datacopy_custom(
        FORMATS, DEST_ACC, src_A=src_A, pack_raw=True
    )

    expected_bits = src_A.to(torch.bfloat16).view(torch.int16)
    result_bits = res_tensor.to(torch.bfloat16).view(torch.int16)
    mismatch = expected_bits != result_bits
    assert not mismatch.any(), (
        f"{int(mismatch.sum())} of {n} bf16 words changed in the copy; first lane {int(mismatch.nonzero()[0])}: "
        f"expected 0x{int(expected_bits[mismatch][0]) & 0xFFFF:04x}, got 0x{int(result_bits[mismatch][0]) & 0xFFFF:04x}"
    )
