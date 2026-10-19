# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact packed words, including sentinel IDs, half tiles, and in-place output."""

from dataclasses import dataclass

import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TemplateParameter

pytestmark = [skip_for_quasar, skip_for_wormhole]


@dataclass
class Packing(TemplateParameter):
    k: int
    half: int
    alias: int

    def convert_to_cpp(self):
        return (
            f"constexpr std::uint32_t PACK_K = {self.k};\n"
            f"constexpr std::uint32_t PACK_HALF = {self.half};\n"
            f"constexpr std::uint32_t PACK_ALIAS = {self.alias};"
        )


@parametrize(case=[(512, 0), (512, 1), (1024, 0), (2048, 0)], alias=[0, 1, 2])
def test_topk_pack_supplied_indices(case, alias):
    k, half = case
    tiles = (k + 1023) // 1024
    generator = torch.Generator().manual_seed(5554)
    # Arbitrary score words also prove low score bits cannot contaminate IDs.
    scores = torch.randint(
        0, 1 << 32, (tiles * 1024,), generator=generator, dtype=torch.int64
    )
    ids = torch.randint(
        0, 1 << 32, (tiles * 1024,), generator=generator, dtype=torch.int64
    )
    ids[::11] = 0xFFFFFFFF
    ids[1:5] = torch.tensor([0, 0xFFFF, 0x10000, 0xFFFF0000])
    formats = InputOutputFormat(DataFormat.UInt32, DataFormat.UInt32)
    config = TestConfig(
        "sources/topk_pack_supplied_indices_test.cpp",
        formats,
        templates=[Packing(k, half, alias)],
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
    result = torch.tensor(config.run().result, dtype=torch.int64)[: tiles * 1024]
    expected = torch.full_like(scores, 0xFF80FFFF)
    offset = half * 512
    expected[:k] = (scores[offset : offset + k] & 0xFFFF0000) | (ids[:k] & 0xFFFF)
    expected[:k][ids[:k] == 0xFFFFFFFF] = 0xFF80FFFF
    assert torch.equal(result, expected), torch.where(result != expected)[0][
        :16
    ].tolist()
