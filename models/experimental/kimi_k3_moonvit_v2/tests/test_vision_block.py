# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Vision_Block` against each reference block (`common.check_ring_layer`). The block's
input is the fp32 residual stream."""

from __future__ import annotations

import ttnn

from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    ATTENTION_GRIDS,
    REAL,
    REAL_ROWS_FROM_EACH_CHIP_ASSERTED_GRIDS,
    RING,
    SAMPLED_LAYERS,
    SYNTHETIC,
    check_ring_layer,
    padding_layout_grid,
)
from models.experimental.kimi_k3_moonvit_v2.tt.vision_block import TT_Vision_Block

RELATIVE_RMSE = 0.012


def check_vision_block(mesh_device, sp_axis, num_links, config, reference, layer, grid, **kwargs) -> None:
    check_ring_layer(
        mesh_device,
        sp_axis,
        num_links,
        config,
        reference,
        layer,
        grid,
        tt_class=TT_Vision_Block,
        reference_call=lambda block, x, cu_seqlens, max_seqlen, rope: block(
            x, cu_seqlens, max_seqlen, rope_freqs_cis=rope
        ),
        input_dtype=ttnn.float32,
        relative_rmse=RELATIVE_RMSE,
        **kwargs,
    )


@RING
@SYNTHETIC
@REAL
@SAMPLED_LAYERS
@ATTENTION_GRIDS
def test_vision_block(mesh_device, sp_axis, num_links, config, reference, grid, layer):
    check_vision_block(mesh_device, sp_axis, num_links, config, reference, layer, grid, real_rows_from_each_chip=False)


@RING
@SYNTHETIC
@REAL
@SAMPLED_LAYERS
@REAL_ROWS_FROM_EACH_CHIP_ASSERTED_GRIDS
def test_vision_block_real_rows_from_each_chip(mesh_device, sp_axis, num_links, config, reference, size, layer):
    """The ring at the production padded length (`max_patches`), on an image with real rows on every chip."""
    check_vision_block(
        mesh_device,
        sp_axis,
        num_links,
        config,
        reference,
        layer,
        padding_layout_grid(size),
        real_rows_from_each_chip=True,
    )
