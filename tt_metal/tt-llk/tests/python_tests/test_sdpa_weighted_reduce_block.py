# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
The batched SDPA weighted reduce (api/compute/experimental/sdpa_weighted_reduce.h, weighted_reduce_block): num_chunks qk
tiles through one unpack context transaction (sources/sdpa_weighted_reduce_block_test.cpp).

Each chunk computes out[1, 32] = weights[1, 8] x qk[8, 32] with the header's two MVMULs into DEST slot c, 16 rows apart,
so a standard pack of DEST tile 0 holds chunk c's first MVMUL row in face c's first row. The weights are a constant W and
qk tile c a constant Q_c, so that row is 8 * W * Q_c in every lane, whatever the exact lane mapping; a wrong tile walk or a
SrcA face left at face 1 shows as another chunk's value or as zero.
"""

from dataclasses import dataclass

import pytest
import torch
from conftest import blackhole_only, skip_for_coverage
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, format_dict
from helpers.pack import pack_bfp16
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import NUM_FACES, TemplateParameter
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

pytestmark = blackhole_only

NUM_HEADS = 8
FACE = 16
TILE = 32
BF16 = DataFormat.Float16_b
WEIGHT = 1.0


@dataclass
class WEIGHTED_REDUCE_BLOCK(TemplateParameter):
    weighted_reduce_chunks: int = 1
    weighted_reduce_row_pack: bool = False

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t NUM_CHUNKS = {self.weighted_reduce_chunks};\n"
            f"constexpr bool ROW_PACK = {str(self.weighted_reduce_row_pack).lower()};"
        )


class WeightedReduceBlockStimuli(StimuliConfig):
    """weights (one 32x32 tile) in buffer_A, num_chunks two-face qk tiles back to back in buffer_B, raw bytes."""

    def __init__(self, packed_weights, packed_qk, num_chunks):
        super().__init__(
            buffer_A=torch.zeros(1, dtype=torch.float32),
            stimuli_A_format=BF16,
            tile_count_A=1,
            buffer_B=torch.zeros(1, dtype=torch.float32),
            stimuli_B_format=BF16,
            tile_count_B=num_chunks,
            stimuli_res_format=BF16,
            tile_count_res=1,
        )
        self.packed_weights = packed_weights
        self.packed_qk = packed_qk

    def write(self, location: str = "0,0"):
        from ttexalens.tt_exalens_lib import write_to_device

        write_to_device(location, self.buf_a_addr, self.packed_weights)
        write_to_device(location, self.buf_b_addr, self.packed_qk)


def _run_block(num_chunks, row_pack):
    torch_format = format_dict[BF16]
    weights_tile = torch.zeros((TILE, TILE), dtype=torch_format)
    weights_tile[0, :NUM_HEADS] = WEIGHT
    packed_weights = bytes(
        pack_bfp16(
            tilize_block(weights_tile, [TILE, TILE], stimuli_format=BF16).flatten()
        )
    )
    q_values = [0.5 * (c + 1) for c in range(num_chunks)]
    packed_qk = b"".join(
        bytes(pack_bfp16(torch.full((2 * FACE * FACE,), q, dtype=torch_format)))
        for q in q_values
    )

    configuration = TestConfig(
        "sources/sdpa_weighted_reduce_block_test.cpp",
        InputOutputFormat(BF16, BF16),
        templates=[
            WEIGHTED_REDUCE_BLOCK(
                weighted_reduce_chunks=num_chunks, weighted_reduce_row_pack=row_pack
            )
        ],
        runtimes=[NUM_FACES(num_faces=4, num_faces_A=4, num_faces_B=2)],
        variant_stimuli=WeightedReduceBlockStimuli(
            packed_weights, packed_qk, num_chunks
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
    )
    res = torch.tensor(configuration.run().result, dtype=torch_format)
    return q_values, res


@skip_for_coverage
@pytest.mark.parametrize("num_chunks", [1, 2, 4])
def test_sdpa_weighted_reduce_block(num_chunks):
    torch_format = format_dict[BF16]
    q_values, res = _run_block(num_chunks, row_pack=False)
    out = untilize_block(res, BF16, [TILE, TILE])

    # Chunk c's first MVMUL row lands in face c's first row: rows 0 and 16, columns 0 to 15 and 16 to 31.
    for c, q in enumerate(q_values):
        row, col = (c // 2) * FACE, (c % 2) * FACE
        golden = torch.full((FACE,), NUM_HEADS * WEIGHT * q, dtype=torch_format)
        assert passed_test(
            golden, out[row, col : col + FACE], BF16
        ), f"chunk {c} of {num_chunks}"


@skip_for_coverage
@pytest.mark.parametrize("num_chunks", [1, 2, 4, 8])
def test_sdpa_weighted_reduce_block_row_pack(num_chunks):
    """weighted_reduce_pack_block: chunk c's 32 outputs (both MVMULs) land in row c of the result, 32 values per row."""
    torch_format = format_dict[BF16]
    q_values, res = _run_block(num_chunks, row_pack=True)
    for c, q in enumerate(q_values):
        golden = torch.full((TILE,), NUM_HEADS * WEIGHT * q, dtype=torch_format)
        assert passed_test(
            golden, res[c * TILE : (c + 1) * TILE], BF16
        ), f"chunk {c} of {num_chunks}"
