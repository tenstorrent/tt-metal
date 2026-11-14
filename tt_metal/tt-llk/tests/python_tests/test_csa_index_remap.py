# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CSA distributed indices -> window-relative bank/row indices, with exact integer goldens."""

from dataclasses import dataclass
from itertools import product

import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_INDEX, DEST_SYNC, TemplateParameter

pytestmark = blackhole_only

DEVICES = 8
BANKS = 8
ROWS_PER_CHUNK = 32
# Output word [bank:3 at bit 14 | row-in-bank:14]: 8 banks x 16384 rows.
ROWS_PER_BANK = 16384
CAPACITY = BANKS * ROWS_PER_BANK
# Compressed positions covered by one 32-row chunk of every device/bank pair.
POSITIONS_PER_CHUNK = DEVICES * BANKS * ROWS_PER_CHUNK


@dataclass
class CSA_REMAP(TemplateParameter):
    csa_row_offset: int

    def convert_to_cpp(self):
        return f"constexpr std::uint32_t CSA_ROW_OFFSET = {self.csa_row_offset}u;"


def _window_position(device, bank, local_row, row_offset):
    # Arithmetic chunk coordinates, independently of the SFPU's masks.
    chunk, within_chunk = divmod(local_row, ROWS_PER_CHUNK)
    return (
        ((chunk * DEVICES + device) * BANKS + bank) * ROWS_PER_CHUNK
        + within_chunk
        + row_offset
    )


def _last_chunk_in_window(row_offset):
    # Last chunk whose positions fit the window for every device/bank pair.
    return (CAPACITY - row_offset) // POSITIONS_PER_CHUNK - 1


def _run_remap(row_offset, dst_index, dest_sync, packed, expected):
    # Three complete DEST tiles: remap one, and verify both surrounding tiles
    # retain their input. Repeat in three sections to cover both half-sync banks.
    assert len(packed) == 1024 and len(expected) == 1024
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
    return torch.as_tensor(config.run().result).to(torch.int64), torch.cat(goldens)


@parametrize(
    row_offset=[0, 1, 7, 8, 255, 256, 32768], dst_index=[0, 1], dest_sync=list(DestSync)
)
def test_csa_index_remap(row_offset, dst_index, dest_sync):
    # Enumerate every device/bank pair and both sides of local-row boundaries,
    # up to the last chunk that fits the 2^17-position window at this offset.
    top = _last_chunk_in_window(row_offset) * ROWS_PER_CHUNK + ROWS_PER_CHUNK - 1
    local_rows = [0, 1, 7, 8, 30, 31, 32, 33, 255, 256, 511, 512]
    local_rows += [top - 32, top - 31, top - 1, top]
    packed, expected = [], []
    for device, bank, local_row in product(range(DEVICES), range(BANKS), local_rows):
        packed.append(
            device * (BANKS * ROWS_PER_BANK) + bank * ROWS_PER_BANK + local_row
        )
        output_row, output_bank = divmod(
            _window_position(device, bank, local_row, row_offset), BANKS
        )
        assert output_row < ROWS_PER_BANK, "enumerated input lies outside the window"
        expected.append(output_bank * ROWS_PER_BANK + output_row)

    result, golden = _run_remap(row_offset, dst_index, dest_sync, packed, expected)
    assert torch.equal(
        result.flatten(), golden
    ), "CSA remap or untouched DEST lanes differ"


@parametrize(row_offset=[0, 32768], dst_index=[0], dest_sync=list(DestSync))
def test_csa_index_remap_overflow_keeps_bank(row_offset, dst_index, dest_sync):
    # Positions beyond the window are outside the contract, but the row field
    # must stay within its 14 bits so they can never alias onto another bank.
    first = (_last_chunk_in_window(row_offset) + 1) * ROWS_PER_CHUNK
    local_rows = [first, first + 1, first + 31, first + 32, first + 33, first + 64]
    local_rows += [4095, 4096, 8191, 8192, 12287, 12288, 14000, 16000, 16382, 16383]
    packed, expected = [], []
    for device, bank, local_row in product(range(DEVICES), range(BANKS), local_rows):
        packed.append(
            device * (BANKS * ROWS_PER_BANK) + bank * ROWS_PER_BANK + local_row
        )
        output_row, output_bank = divmod(
            _window_position(device, bank, local_row, row_offset), BANKS
        )
        assert output_row >= ROWS_PER_BANK, "probe input must lie beyond the window"
        expected.append(output_bank * ROWS_PER_BANK + output_row % ROWS_PER_BANK)

    result, golden = _run_remap(row_offset, dst_index, dest_sync, packed, expected)
    assert torch.equal(
        result.flatten(), golden
    ), "out-of-window row spilled into the bank field, or untouched DEST lanes differ"
