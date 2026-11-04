# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CSA distributed indices -> window-relative bank/row indices, with exact integer goldens."""

from dataclasses import dataclass
from itertools import product

import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_INDEX, DEST_SYNC, TemplateParameter

pytestmark = [skip_for_wormhole, skip_for_quasar]


@dataclass
class CSA_REMAP(TemplateParameter):
    csa_row_offset: int

    def convert_to_cpp(self):
        return f"constexpr std::uint32_t CSA_ROW_OFFSET = {self.csa_row_offset}u;"


@parametrize(
    row_offset=[0, 1, 7, 8, 255, 256, 32768], dst_index=[0, 1], dest_sync=list(DestSync)
)
def test_csa_index_remap(row_offset, dst_index, dest_sync):
    # Enumerate every device/bank pair and both sides of local-row boundaries.
    # The golden uses arithmetic chunk coordinates, independently of the SFPU's masks.
    cases = list(
        product(
            range(8),
            range(8),
            [0, 1, 7, 8, 30, 31, 32, 33, 255, 256, 511, 512, 8191, 8192, 16382, 16383],
        )
    )
    packed, expected = [], []
    for device, bank, local_row in cases:
        packed.append(device * (8 * 16384) + bank * 16384 + local_row)
        chunk, within_chunk = divmod(local_row, 32)
        position = ((chunk * 8 + device) * 8 + bank) * 32 + within_chunk + row_offset
        output_row, output_bank = divmod(position, 8)
        expected.append(output_bank * 16384 | output_row)

    # Three complete DEST tiles: remap one, and verify both surrounding tiles
    # retain their input. Repeat in three sections to cover both half-sync banks.
    inputs, goldens = [], []
    for section in range(3):
        tiles = (
            torch.arange(3 * 1024, dtype=torch.int64) + 0x123400 + section * 4096
        ).reshape(3, 1024)
        tiles[dst_index] = torch.tensor(packed).roll(section * 37)
        golden = tiles.clone()
        golden[dst_index] = torch.tensor(expected).roll(section * 37)
        inputs.append(tiles.flatten())
        goldens.append(golden.flatten())

    src = torch.cat(inputs).to(torch.uint32)
    config = TestConfig(
        "sources/csa_index_remap_test.cpp",
        InputOutputFormat(DataFormat.UInt32, DataFormat.UInt32),
        templates=[CSA_REMAP(row_offset), DEST_SYNC(dest_sync)],
        runtimes=[DEST_INDEX(dst_index)],
        variant_stimuli=StimuliConfig(
            src,
            DataFormat.UInt32,
            torch.zeros(1024, dtype=torch.uint32),
            DataFormat.UInt32,
            DataFormat.UInt32,
            tile_count_A=9,
            tile_count_B=1,
            tile_count_res=9,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
    )
    result = torch.as_tensor(config.run().result).to(torch.int64)
    assert torch.equal(
        result.flatten(), torch.cat(goldens)
    ), "CSA remap or untouched DEST lanes differ"
