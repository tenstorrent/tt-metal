# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact packed words, including sentinel IDs, half tiles, and in-place output."""

from dataclasses import dataclass

import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE
from helpers.llk_params import DestAccumulation, DestSync
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, TemplateParameter

pytestmark = [skip_for_quasar, skip_for_wormhole]

PACKED_PADDING = 0xFF80FFFF
SENTINEL_ID = 0xFFFFFFFF
SCORE_MASK = 0xFFFF0000
INDEX_MASK = 0xFFFF
HALF_TILE_ELEMENTS = ELEMENTS_PER_TILE // 2


@dataclass
class Packing(TemplateParameter):
    pack_k: int
    half: int
    alias: int

    def convert_to_cpp(self):
        return (
            f"constexpr std::uint32_t PACK_K = {self.pack_k};\n"
            f"constexpr std::uint32_t PACK_HALF = {self.half};\n"
            f"constexpr std::uint32_t PACK_ALIAS = {self.alias};"
        )


@parametrize(
    case=[(512, 0), (512, 1), (1024, 0), (2048, 0)],
    alias_sync=[
        (0, DestSync.Full),
        (1, DestSync.Full),
        (2, DestSync.Full),
        (1, DestSync.Half),
        (2, DestSync.Half),
    ],
)
def test_topk_pack_supplied_indices(case, alias_sync):
    alias, dest_sync = alias_sync
    k, half = case
    tiles = (k + ELEMENTS_PER_TILE - 1) // ELEMENTS_PER_TILE
    generator = torch.Generator().manual_seed(5554)
    # Arbitrary score words also prove low score bits cannot contaminate IDs.
    scores = torch.randint(
        0, 1 << 32, (tiles * ELEMENTS_PER_TILE,), generator=generator, dtype=torch.int64
    )
    ids = torch.randint(
        0, 1 << 32, (tiles * ELEMENTS_PER_TILE,), generator=generator, dtype=torch.int64
    )
    ids[::11] = SENTINEL_ID
    ids[1:5] = torch.tensor([0, INDEX_MASK, INDEX_MASK + 1, SCORE_MASK])
    formats = InputOutputFormat(DataFormat.UInt32, DataFormat.UInt32)
    config = TestConfig(
        "sources/topk_pack_supplied_indices_test.cpp",
        formats,
        templates=[Packing(k, half, alias), DEST_SYNC(dest_sync)],
        variant_stimuli=StimuliConfig(
            scores,
            DataFormat.UInt32,
            ids,
            DataFormat.UInt32,
            DataFormat.UInt32,
            tile_count_A=tiles,
            tile_count_B=tiles,
            tile_count_res=tiles,
        ),
        unpack_to_dest=True,
        dest_acc=DestAccumulation.Yes,
    )
    result = torch.tensor(config.run().result, dtype=torch.int64)[
        : tiles * ELEMENTS_PER_TILE
    ]
    expected = torch.full_like(scores, PACKED_PADDING)
    offset = half * HALF_TILE_ELEMENTS
    expected[:k] = (scores[offset : offset + k] & SCORE_MASK) | (ids[:k] & INDEX_MASK)
    expected[:k][ids[:k] == SENTINEL_ID] = PACKED_PADDING
    assert torch.equal(result, expected), torch.where(result != expected)[0][
        :16
    ].tolist()
