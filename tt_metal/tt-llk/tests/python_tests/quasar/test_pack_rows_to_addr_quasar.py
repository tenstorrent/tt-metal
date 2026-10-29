# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Standalone tt-llk test for Quasar pack rows to an L1 address (llk_pack_rows.h). Every input tile is
# copied to DEST, and the first num_rows_to_pack 16-datum rows of it are packed into its own two-tile
# result slot, tile % 4 16B units from the slot start. Input and output formats match, so the packed
# rows must equal the first rows of the input tile byte for byte, and every other byte of the result
# buffer must keep the guard value written before the run. The check is on raw bytes rather than a
# golden so that it also proves nothing outside the packed rows is written.

import pytest
from helpers.device_io import read_from_device
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, DestSync
from helpers.param_config import (
    DEST_SYNC_TILE_LIMITS,
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    NUM_BLOCKS,
    NUM_ROWS_TO_PACK,
    NUM_TILES_IN_BLOCK,
)

GUARD_BYTE = 0xA5
ROW_NUM_DATUMS = 16
# Three DEST sections, so the packs walk bank 0 -> 1 -> 0 and cover the wrap back to the first bank.
NUM_DEST_SECTIONS = 3
# _llk_pack_rows_ asserts is_valid_L1_address, which rejects anything below MEM_MAP_END (0x38640 in
# tt-2xx/quasar/dev_mem_map.h). The stimuli start at 0x21000, so a small input would put the result there.
MIN_RESULT_L1_ADDRESS = 0x40000


class GuardedResultStimuliConfig(StimuliConfig):
    """Places the result buffer at MIN_RESULT_L1_ADDRESS or above, and fills it with GUARD_BYTE each time the
    stimuli are written, before the kernel runs."""

    def _calculate_tile_sizes(self):
        super()._calculate_tile_sizes()
        self.buf_res_addr = max(self.buf_res_addr, MIN_RESULT_L1_ADDRESS)

    def write(self, location: str = "0,0"):
        super().write(location)
        self.clear_result_buffer(location, fill_byte=GUARD_BYTE)


@pytest.mark.quasar
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Int8,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync_mode=[DestSync.Half, DestSync.Full],
    num_rows_to_pack=runtime([1, 8, 15, 16, 17, 33, 50, 64]),
)
def test_pack_rows_to_addr_quasar(formats, dest_acc, dest_sync_mode, num_rows_to_pack):
    if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No:
        pytest.skip("32-bit formats require 32-bit DEST")
    max_tiles_in_dest = DEST_SYNC_TILE_LIMITS[dest_sync_mode] // (
        2 if dest_acc == DestAccumulation.Yes else 1
    )
    input_dimensions = [32 * NUM_DEST_SECTIONS * max_tiles_in_dest, 32]

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    num_blocks, tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync_mode, dest_acc, formats, input_dimensions
    )

    stimuli = GuardedResultStimuliConfig(
        src_A,
        formats.input_format,
        src_B,
        formats.input_format,
        formats.output_format,
        tile_count_A=tile_cnt_A,
        tile_count_B=tile_cnt_A,
        tile_count_res=2 * tile_cnt_A,
    )

    configuration = TestConfig(
        "sources/quasar/pack_rows_to_addr_quasar_test.cpp",
        formats,
        templates=[DEST_SYNC(dest_sync_mode)],
        runtimes=[
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(tiles_in_block),
            NUM_ROWS_TO_PACK(num_rows_to_pack),
        ],
        variant_stimuli=stimuli,
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
    )
    configuration.run()

    input_tile_bytes = stimuli.tile_size_A_bytes
    slot_bytes = 2 * stimuli.buf_res_tile_size
    input_raw = read_from_device(
        TestConfig.TENSIX_LOCATION,
        stimuli.buf_a_addr,
        num_bytes=tile_cnt_A * input_tile_bytes,
    )
    result_raw = read_from_device(
        TestConfig.TENSIX_LOCATION,
        stimuli.buf_res_addr,
        num_bytes=tile_cnt_A * slot_bytes,
    )

    rows_bytes = num_rows_to_pack * ROW_NUM_DATUMS * formats.output_format.size
    assert rows_bytes.denominator == 1, "packed rows must be a whole number of bytes"
    rows_bytes = int(rows_bytes)
    expected = bytearray([GUARD_BYTE]) * len(result_raw)
    for tile in range(tile_cnt_A):
        start = tile * slot_bytes + 16 * (tile % 4)
        tile_start = tile * input_tile_bytes
        expected[start : start + rows_bytes] = input_raw[
            tile_start : tile_start + rows_bytes
        ]

    mismatches = [i for i in range(len(expected)) if result_raw[i] != expected[i]]
    assert not mismatches, (
        f"{len(mismatches)} result bytes differ, first at byte {mismatches[0]} "
        f"(slot {mismatches[0] // slot_bytes})"
    )
