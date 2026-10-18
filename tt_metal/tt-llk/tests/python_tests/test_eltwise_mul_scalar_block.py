# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""One SrcB scalar reused across contiguous SrcA tiles, then replaced on re-entry."""

import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]


@parametrize(
    block_layout=[(1, 0), (3, 1), (4, 0)],
    dest_acc=list(DestAccumulation),
    dest_sync=list(DestSync),
)
def test_eltwise_mul_scalar_block(block_layout, dest_acc, dest_sync):
    block_size, dst_index = block_layout
    scalars = torch.tensor([1.5, -0.5, 0.0, 2.0], dtype=torch.bfloat16)
    count = len(scalars) * block_size * 1024
    # Exactly representable products allow an elementwise, zero-tolerance check
    # in both DEST widths. Other SrcB lanes are poison: only B[0] is a scalar.
    src = ((torch.arange(count) * 17 % 63 - 31).to(torch.float32) / 8).to(
        torch.bfloat16
    )
    scalar_tiles = torch.full((len(scalars), 1024), -17.0, dtype=torch.bfloat16)
    scalar_tiles[:, 0] = scalars
    golden = (
        src.reshape(len(scalars), -1).float() * scalars.float()[:, None]
    ).flatten()
    config = TestConfig(
        "sources/eltwise_mul_scalar_block_test.cpp",
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
        templates=[DEST_SYNC(dest_sync)],
        runtimes=[
            NUM_BLOCKS(len(scalars)),
            NUM_TILES_IN_BLOCK(block_size),
            DEST_INDEX(dst_index),
        ],
        variant_stimuli=StimuliConfig(
            src,
            DataFormat.Float16_b,
            scalar_tiles.flatten(),
            DataFormat.Float16_b,
            DataFormat.Float32,
            tile_count_A=len(scalars) * block_size,
            tile_count_B=len(scalars),
            tile_count_res=len(scalars) * block_size,
        ),
        dest_acc=dest_acc,
    )
    result = torch.as_tensor(config.run().result).float().flatten()
    assert torch.equal(
        result, golden
    ), "Scalar block multiply lost a tile, reused an old scalar, or wrote the wrong DEST offset"
