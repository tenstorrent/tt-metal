# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Vision_Block` against each reference block. The device runs window-ordered rows; the
output is put back in raster order to compare."""

from __future__ import annotations

import torch

from models.experimental.kimi_k3_moonvit_v2.tt import host_prep
from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    assert_pcc,
    DEVICE_GRIDS,
    NOT_WRITTEN,
    REFERENCE_LAYERS,
    RING,
    reference_rope,
    ring_parallel,
    sp_gather,
    sp_shard,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import check_grid, compute_kernel_config, resolve_parallel
from models.experimental.kimi_k3_moonvit_v2.tt.rope import TT_Rope
from models.experimental.kimi_k3_moonvit_v2.tt.vision_attention import logical_length_tensor
from models.experimental.kimi_k3_moonvit_v2.tt.vision_block import TT_Vision_Block


@RING
@REFERENCE_LAYERS
@DEVICE_GRIDS
@NOT_WRITTEN
def test_vision_block(mesh_device, sp_axis, num_links, config, reference, grid, layer):
    parallel = resolve_parallel(mesh_device, *ring_parallel(mesh_device, sp_axis, num_links), config)
    tt = TT_Vision_Block(
        config, mesh_device=mesh_device, parallel=parallel, compute_kernel_config=compute_kernel_config(mesh_device)
    )
    block = reference.vision_tower.encoder.blocks[layer]
    tt.load_torch_state_dict(block.state_dict())

    grid_thws = torch.tensor(grid)
    h, w = check_grid(grid_thws, config)
    x = torch.randn(h * w, config.hidden_size, dtype=torch.bfloat16)
    cu_seqlens, max_seqlen, rope_freqs_cis = reference_rope(reference, grid_thws)
    with torch.no_grad():
        golden = block(x, cu_seqlens, max_seqlen, rope_freqs_cis=rope_freqs_cis)

    rope = TT_Rope(config, mesh_device=mesh_device, parallel=parallel).write(*host_prep.rope_tables(config, h, w))
    x_window = host_prep.to_window_order(x, h, w, config.merge_kernel_size)
    logical_n = logical_length_tensor(mesh_device, h * w)
    out = tt(sp_shard(x_window, mesh_device, sp_axis, config.max_patches), rope, logical_n)
    out = sp_gather(out, mesh_device, sp_axis, h * w)
    assert_pcc(golden, host_prep.from_window_order(out, h, w, config.merge_kernel_size))
