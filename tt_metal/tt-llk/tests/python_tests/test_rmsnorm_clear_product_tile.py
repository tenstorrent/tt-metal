# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Clear reused RMSNorm product tiles without changing the reserved accumulator."""

from dataclasses import dataclass

import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, TemplateParameter

pytestmark = [skip_for_wormhole, skip_for_quasar]

ELEMENTS_PER_TILE = 1024
CYCLES = 2


@dataclass
class RMSNORM_CLEAR_PRODUCT(TemplateParameter):
    capacity: int
    dest_tiles: int

    def convert_to_cpp(self):
        return (
            f"constexpr unsigned RMSNORM_CAPACITY = {self.capacity};\n"
            f"constexpr unsigned RMSNORM_DEST_TILES = {self.dest_tiles};\n"
            f"constexpr unsigned RMSNORM_CLEAR_CYCLES = {CYCLES};"
        )


@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=[DestSync.Half, DestSync.Full],
)
def test_rmsnorm_clear_product_tile(dest_acc, dest_sync):
    # The chunked API reserves the last slot and supports at most eight slots.
    # FP32 half sync fits four; FP32 full sync also exercises bank-relative
    # addressing for product slots 4..6 and the accumulator in slot 7.
    capacity = (
        4 if dest_acc == DestAccumulation.Yes and dest_sync == DestSync.Half else 8
    )
    # BF16 full sync has eight additional physical slots outside the helper's
    # logical capacity. Keep them as guards against clearing the wrong bank.
    dest_tiles = (
        16
        if dest_acc == DestAccumulation.No and dest_sync == DestSync.Full
        else capacity
    )
    data_format = (
        DataFormat.Float32 if dest_acc == DestAccumulation.Yes else DataFormat.Float16_b
    )
    dtype = format_dict[data_format]
    lane = torch.arange(ELEMENTS_PER_TILE).reshape(1, -1)
    tile = torch.arange(dest_tiles).reshape(-1, 1)
    # Nonzero signed integers are exact in both formats. Distinct row, face,
    # and tile patterns reveal partial clears and damage to neighboring slots.
    magnitude = 1 + (17 * lane + 7 * (lane // 16) + 19 * tile) % 127
    seed = (magnitude * (1 - 2 * ((lane + tile) % 2))).to(dtype)
    output_tiles = CYCLES * (capacity - 1) * dest_tiles

    config = TestConfig(
        "sources/rmsnorm_clear_product_tile_test.cpp",
        InputOutputFormat(data_format, data_format),
        templates=[DEST_SYNC(dest_sync), RMSNORM_CLEAR_PRODUCT(capacity, dest_tiles)],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            seed.flatten(),
            data_format,
            seed[0],
            data_format,
            data_format,
            tile_count_A=dest_tiles,
            tile_count_B=1,
            tile_count_res=output_tiles,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )
    result = torch.tensor(config.run().result, dtype=dtype).reshape(
        CYCLES, capacity - 1, dest_tiles, ELEMENTS_PER_TILE
    )
    expected = seed.reshape(1, 1, dest_tiles, ELEMENTS_PER_TILE).repeat(
        CYCLES, capacity - 1, 1, 1
    )
    for target in range(capacity - 1):
        expected[:, target, target, :] = 0
    # Check every lane of the cleared tile, accumulator, and all neighbors.
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
