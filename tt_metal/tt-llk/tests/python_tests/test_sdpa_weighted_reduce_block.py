# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""weighted_reduce_block and weighted_reduce_pack_block (api/compute/experimental/sdpa_weighted_reduce.h) through
sdpa_weighted_reduce_block_test.cpp. Every head weight, qk face and qk row is distinct, so a face, row or chunk read from
the wrong place changes the sums; the sums are small integers, exact in bf16.
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
from helpers.test_variant_parameters import TemplateParameter
from helpers.tilize_untilize import tilize_block, untilize_block

pytestmark = blackhole_only

NUM_HEADS = 8
FACE = 16
TILE = 32
BF16 = DataFormat.Float16_b
SENTINEL = 0xA5


@dataclass
class WEIGHTED_REDUCE_BLOCK(TemplateParameter):
    weighted_reduce_chunks: int = 1
    weighted_reduce_row_pack: bool = False
    weighted_reduce_qk_faces: int = 2
    weighted_reduce_first_chunk: int = 0
    weighted_reduce_first_slot: int = 0

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"constexpr std::uint32_t NUM_CHUNKS = {self.weighted_reduce_chunks};",
                f"constexpr bool ROW_PACK = {str(self.weighted_reduce_row_pack).lower()};",
                f"constexpr std::uint32_t QK_NUM_FACES = {self.weighted_reduce_qk_faces};",
                f"constexpr std::uint32_t FIRST_CHUNK = {self.weighted_reduce_first_chunk};",
                f"constexpr std::uint32_t FIRST_SLOT = {self.weighted_reduce_first_slot};",
            ]
        )


class WeightedReduceBlockStimuli(StimuliConfig):
    """weights (one 32x32 tile) in buffer_A, the qk tiles back to back in buffer_B, raw bytes; the result region holds a
    sentinel so rows the kernel does not write can be told apart."""

    def __init__(self, packed_weights, packed_qk, num_chunks, res_tiles):
        super().__init__(
            buffer_A=torch.zeros(1, dtype=torch.float32),
            stimuli_A_format=BF16,
            tile_count_A=1,
            buffer_B=torch.zeros(1, dtype=torch.float32),
            stimuli_B_format=BF16,
            tile_count_B=num_chunks,
            stimuli_res_format=BF16,
            tile_count_res=res_tiles,
        )
        self.packed_weights = packed_weights
        self.packed_qk = packed_qk

    def write(self, location: str = "0,0"):
        from ttexalens.tt_exalens_lib import write_to_device

        write_to_device(location, self.buf_a_addr, self.packed_weights)
        write_to_device(location, self.buf_b_addr, self.packed_qk)
        self.clear_result_buffer(location, fill_byte=SENTINEL)


def _qk_faces(c, qk_faces):
    """Chunk c's qk tile as qk_faces 16x16 faces: face 0 holds 0 to 3, face 1 holds 4 to 7, faces 2 and 3 (never read)
    hold 64 and up."""
    k = torch.arange(FACE).view(FACE, 1)
    j = torch.arange(FACE).view(1, FACE)
    faces = [(c + k + 3 * j) % 4, 4 + (2 * c + k + j) % 4]
    faces += [64 + 8 * f + (c + k + j) % 4 for f in range(2, qk_faces)]
    return faces


def _run(num_chunks, row_pack, qk_faces=2, first_chunk=0, first_slot=0):
    torch_format = format_dict[BF16]
    weights = torch.arange(1, NUM_HEADS + 1, dtype=torch.float32)
    weights_tile = torch.zeros((TILE, TILE), dtype=torch_format)
    weights_tile[0, :NUM_HEADS] = weights.to(torch_format)
    packed_weights = pack_bfp16(
        tilize_block(weights_tile, [TILE, TILE], stimuli_format=BF16).flatten()
    )
    faces = [_qk_faces(c, qk_faces) for c in range(num_chunks)]
    packed_qk = b"".join(
        pack_bfp16(torch.cat([f.flatten() for f in chunk]).to(torch_format))
        for chunk in faces
    )
    # out[p] = sum over the eight heads of weight[h] * qk[h, p]; qk rows 8 to 15 meet zero weights.
    golden = [
        torch.cat([weights @ f[:NUM_HEADS].float() for f in chunk[:2]]).to(torch_format)
        for chunk in faces
    ]
    res_tiles = -(-(first_chunk + num_chunks) // TILE) if row_pack else 1

    configuration = TestConfig(
        "sources/sdpa_weighted_reduce_block_test.cpp",
        InputOutputFormat(BF16, BF16),
        templates=[
            WEIGHTED_REDUCE_BLOCK(
                weighted_reduce_chunks=num_chunks,
                weighted_reduce_row_pack=row_pack,
                weighted_reduce_qk_faces=qk_faces,
                weighted_reduce_first_chunk=first_chunk,
                weighted_reduce_first_slot=first_slot,
            )
        ],
        variant_stimuli=WeightedReduceBlockStimuli(
            packed_weights, packed_qk, num_chunks, res_tiles
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
    )
    res = torch.tensor(configuration.run().result, dtype=torch_format)
    return golden, res


@skip_for_coverage
@pytest.mark.parametrize("num_chunks, qk_faces", [(1, 2), (2, 2), (4, 2), (4, 4)])
def test_sdpa_weighted_reduce_block(num_chunks, qk_faces):
    """Chunk c's DEST slot is face c of DEST tile 0: the first MVMUL's 16 outputs in its row 0, the second's in its row 8."""
    golden, res = _run(num_chunks, row_pack=False, qk_faces=qk_faces)
    out = untilize_block(res, BF16, [TILE, TILE])
    for c, want in enumerate(golden):
        row, col = (c // 2) * FACE, (c % 2) * FACE
        got = torch.cat([out[row, col : col + FACE], out[row + 8, col : col + FACE]])
        assert torch.equal(got, want), f"chunk {c} of {num_chunks}: {got} != {want}"


@skip_for_coverage
@pytest.mark.parametrize(
    "num_chunks, qk_faces, first_chunk, first_slot",
    [
        (1, 2, 0, 0),
        (2, 2, 0, 0),
        (4, 2, 0, 0),
        (8, 2, 0, 0),
        (16, 2, 0, 0),
        (32, 2, 0, 0),
        (11, 2, 27, 3),
        (16, 2, 20, 5),
        (8, 4, 0, 0),
        (11, 4, 27, 3),
    ],
)
def test_sdpa_weighted_reduce_block_row_pack(
    num_chunks, qk_faces, first_chunk, first_slot
):
    """weighted_reduce_pack_block: chunk c's 32 outputs land in row first_chunk + c, counted across tiles, and no other
    row of the result is written."""
    golden, res = _run(num_chunks, True, qk_faces, first_chunk, first_slot)
    rows = res.view(-1, TILE)
    sentinel = torch.tensor([SENTINEL * 0x101], dtype=torch.int32).to(torch.int16)
    sentinel = sentinel.view(format_dict[BF16])
    for r in range(rows.shape[0]):
        c = r - first_chunk
        if 0 <= c < num_chunks:
            assert torch.equal(
                rows[r], golden[c]
            ), f"chunk {c} in row {r}: {rows[r]} != {golden[c]}"
        else:
            assert torch.equal(rows[r], sentinel.expand(TILE)), f"row {r} was written"
