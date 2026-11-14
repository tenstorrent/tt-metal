# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Vision_Attention` (ring attention over the 4-chip ring) against the reference
`blocks[layer].attention_qkvpacked` (`common.check_ring_layer`)."""

from __future__ import annotations

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
from models.experimental.kimi_k3_moonvit_v2.tt.vision_attention import TT_Vision_Attention

RELATIVE_RMSE = 0.025


def check_vision_attention(mesh_device, sp_axis, num_links, config, reference, layer, grid, **kwargs) -> None:
    check_ring_layer(
        mesh_device,
        sp_axis,
        num_links,
        config,
        reference,
        layer,
        grid,
        tt_class=TT_Vision_Attention,
        reference_call=lambda block, *args: block.attention_qkvpacked(*args),
        state_prefixes=("wqkv.", "wo."),
        relative_rmse=RELATIVE_RMSE,
        **kwargs,
    )


# ATTENTION_GRIDS fit in chip 0's 16,384 rows, so chips 1-3 hold only padding there;
# `test_vision_attention_real_rows_from_each_chip` puts real rows on every chip.
@RING
@SYNTHETIC
@REAL
@SAMPLED_LAYERS
@ATTENTION_GRIDS
def test_vision_attention(mesh_device, sp_axis, num_links, config, reference, grid, layer):
    check_vision_attention(
        mesh_device, sp_axis, num_links, config, reference, layer, grid, real_rows_from_each_chip=False
    )


@RING
@SYNTHETIC
@REAL
@SAMPLED_LAYERS
@REAL_ROWS_FROM_EACH_CHIP_ASSERTED_GRIDS
def test_vision_attention_real_rows_from_each_chip(mesh_device, sp_axis, num_links, config, reference, size, layer):
    """The ring at the production padded length (`max_patches`), on an image with real rows on every chip."""
    check_vision_attention(
        mesh_device,
        sp_axis,
        num_links,
        config,
        reference,
        layer,
        padding_layout_grid(size),
        real_rows_from_each_chip=True,
    )
