# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Vision_Attention` (ring attention over the 4-chip ring) against the reference
`blocks[layer].attention_qkvpacked`. The device runs window-ordered rows; the output is put back in
raster order to compare."""

from __future__ import annotations

import copy

import torch

from models.experimental.kimi_k3_moonvit_v2.tt import host_prep
from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    SYNTHETIC,
    REAL,
    SAMPLED_LAYERS,
    random_tensor,
    assert_quality,
    ATTENTION_GRIDS,
    RING,
    reference_rope,
    ring_parallel,
    sp_gather,
    sp_shard,
    assert_quality_per_chip,
    REAL_ROWS_FROM_EACH_CHIP_ASSERTED_GRIDS,
    padding_layout_grid,
    sampled_rows,
)
from models.experimental.kimi_k3_moonvit_v2.tests.goldens import to_fp32, use_sampled_attention
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import check_grid, compute_kernel_config, resolve_parallel
from models.experimental.kimi_k3_moonvit_v2.tt.rope import TT_Rope
from models.experimental.kimi_k3_moonvit_v2.tt.vision_attention import TT_Vision_Attention, logical_length_tensor

# ATTENTION_GRIDS fit in chip 0's 16,384 rows, so chips 1-3 hold only padding there;
# `test_vision_attention_full_length` puts real rows on every chip.


@RING
@SYNTHETIC
@REAL
@SAMPLED_LAYERS
@ATTENTION_GRIDS
def test_vision_attention(mesh_device, sp_axis, num_links, config, reference, grid, layer):
    parallel = resolve_parallel(mesh_device, *ring_parallel(mesh_device, sp_axis, num_links), config)
    tt = TT_Vision_Attention(
        config, mesh_device=mesh_device, parallel=parallel, compute_kernel_config=compute_kernel_config(mesh_device)
    )
    block = reference.vision_tower.encoder.blocks[layer]
    tt.load_torch_state_dict({k: v for k, v in block.state_dict().items() if k.startswith(("wqkv.", "wo."))})

    grid_thws = torch.tensor(grid)
    h, w = check_grid(grid_thws, config)
    x = random_tensor(h * w, config.hidden_size)
    cu_seqlens, max_seqlen, rope_freqs_cis = reference_rope(reference, grid_thws)
    with torch.no_grad():
        golden = block.attention_qkvpacked(x, cu_seqlens, max_seqlen, rope_freqs_cis)

    rope = TT_Rope(config, mesh_device=mesh_device, parallel=parallel).write(*host_prep.rope_tables(config, h, w))
    x_window = host_prep.to_window_order(x, h, w, config.merge_kernel_size)
    logical_n = logical_length_tensor(mesh_device, h * w)
    out = tt(sp_shard(x_window, mesh_device, sp_axis, config.max_patches), rope, logical_n)
    out = sp_gather(out, mesh_device, sp_axis, h * w)
    assert_quality(golden, host_prep.from_window_order(out, h, w, config.merge_kernel_size), relative_rmse=0.025)


@RING
@SYNTHETIC
@REAL
@SAMPLED_LAYERS
@REAL_ROWS_FROM_EACH_CHIP_ASSERTED_GRIDS
def test_vision_attention_full_length(mesh_device, sp_axis, num_links, config, reference, size, layer):
    """The ring at the production padded length (`max_patches`), on an image with real rows on every
    chip. The fp32 golden covers `sampled_rows` of each chip only, each against every key
    (`use_sampled_attention`), and each chip's rows are checked on their own."""
    parallel = resolve_parallel(mesh_device, *ring_parallel(mesh_device, sp_axis, num_links), config)
    tt = TT_Vision_Attention(
        config, mesh_device=mesh_device, parallel=parallel, compute_kernel_config=compute_kernel_config(mesh_device)
    )
    block = reference.vision_tower.encoder.blocks[layer]
    tt.load_torch_state_dict({k: v for k, v in block.state_dict().items() if k.startswith(("wqkv.", "wo."))})
    grid_thws = torch.tensor(padding_layout_grid(size))
    h, w = check_grid(grid_thws, config)
    x = random_tensor(h * w, config.hidden_size)
    chips = sampled_rows(h * w, config.max_patches)
    window_to_raster = host_prep.to_window_order(torch.arange(h * w)[:, None], h, w, config.merge_kernel_size)[:, 0]
    query_rows = window_to_raster[torch.cat(chips)].sort().values
    cu_seqlens, max_seqlen, rope_freqs_cis = reference_rope(reference, grid_thws)
    with torch.no_grad(), use_sampled_attention(query_rows):
        golden = to_fp32(copy.deepcopy(block)).attention_qkvpacked(x.float(), cu_seqlens, max_seqlen, rope_freqs_cis)
    golden = host_prep.to_window_order(golden, h, w, config.merge_kernel_size)

    rope = TT_Rope(config, mesh_device=mesh_device, parallel=parallel).write(*host_prep.rope_tables(config, h, w))
    x_window = host_prep.to_window_order(x, h, w, config.merge_kernel_size)
    logical_n = logical_length_tensor(mesh_device, h * w)
    out = tt(sp_shard(x_window, mesh_device, sp_axis, config.max_patches), rope, logical_n)
    out = sp_gather(out, mesh_device, sp_axis, h * w)
    assert_quality_per_chip(golden, out, chips, relative_rmse=0.025)
