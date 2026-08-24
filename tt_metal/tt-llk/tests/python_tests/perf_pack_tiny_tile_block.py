# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import (
    input_output_formats,
    parametrize,
    select_perf_tile_sizes,
)
from helpers.perf.core import PerfConfig
from test_pack_tiny_tile_block import _make_config

_FUNCTIONAL_TILE_DIMS = [
    (1, 32),
    (2, 32),
    (4, 32),
    (8, 32),
    (16, 32),
    (16, 16),
    (32, 32),
]


@skip_for_wormhole
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16, DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    tile_dims=select_perf_tile_sizes(_FUNCTIONAL_TILE_DIMS),
    num_tiles=[8],
)
def test_perf_pack_tiny_tile_block(
    perf_report, formats, dest_acc, tile_dims, num_tiles
):
    test_cfg, _, _ = _make_config(tile_dims, num_tiles, formats, dest_acc)
    configuration = PerfConfig(
        test_cfg.test_name,
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=list(test_cfg.templates),
        runtimes=list(test_cfg.runtimes),
        variant_stimuli=test_cfg.variant_stimuli,
        dest_acc=test_cfg.dest_acc,
        unpack_to_dest=test_cfg.unpack_to_dest,
    )
    configuration.run(perf_report)
