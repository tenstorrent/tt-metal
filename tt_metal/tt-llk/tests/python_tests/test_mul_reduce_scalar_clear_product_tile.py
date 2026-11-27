# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Clear reused mul_reduce_scalar product tiles without changing the reserved accumulator."""

from dataclasses import dataclass

import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, format_dict
from helpers.param_config import parametrize
from helpers.perf.blaze import execute_llk
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import DEST_SYNC, TILE_COUNT, TemplateParameter

pytestmark = [skip_for_wormhole, skip_for_quasar]

ELEMENTS_PER_TILE = 1024
CYCLES = 2


@dataclass
class MUL_REDUCE_SCALAR_CLEAR_PRODUCT(TemplateParameter):
    capacity: int
    dest_tiles: int

    def convert_to_cpp(self):
        return (
            f"constexpr unsigned CLEAR_PRODUCT_CAPACITY = {self.capacity};\n"
            f"constexpr unsigned CLEAR_PRODUCT_DEST_TILES = {self.dest_tiles};\n"
            f"constexpr unsigned CLEAR_PRODUCT_CYCLES = {CYCLES};"
        )


@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=[DestSync.Half, DestSync.Full],
)
def test_mul_reduce_scalar_clear_product_tile(
    dest_acc,
    dest_sync,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    capacity = (
        4 if dest_acc == DestAccumulation.Yes and dest_sync == DestSync.Half else 8
    )
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
    magnitude = 1 + (17 * lane + 7 * (lane // 16) + 19 * tile) % 127
    seed = (magnitude * (1 - 2 * ((lane + tile) % 2))).to(dtype)
    output_tiles = CYCLES * (capacity - 1) * dest_tiles

    outcome = execute_llk(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        test_config_kwargs=dict(
            test_name="sources/mul_reduce_scalar_clear_product_tile_test.cpp",
            formats=InputOutputFormat(data_format, data_format),
            templates=[
                DEST_SYNC(dest_sync),
                MUL_REDUCE_SCALAR_CLEAR_PRODUCT(capacity, dest_tiles),
            ],
            runtimes=[TILE_COUNT(output_tiles)],
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
        ),
    )
    if is_perf:
        return
    result = torch.tensor(outcome.result, dtype=dtype).reshape(
        CYCLES, capacity - 1, dest_tiles, ELEMENTS_PER_TILE
    )
    expected = seed.reshape(1, 1, dest_tiles, ELEMENTS_PER_TILE).repeat(
        CYCLES, capacity - 1, 1, 1
    )
    for target in range(capacity - 1):
        expected[:, target, target, :] = 0
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
