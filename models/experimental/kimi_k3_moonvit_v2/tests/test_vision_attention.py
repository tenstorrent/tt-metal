# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Vision_Attention` (ring attention over the 1x4 mesh) against the reference
`blocks[layer].attention_qkvpacked`. The device runs window-ordered rows; the output is put back in
raster order to compare."""

from __future__ import annotations

import torch

from models.experimental.kimi_k3_moonvit_v2.reference import functional
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
from models.experimental.kimi_k3_moonvit_v2.tt.vision_attention import TT_Vision_Attention, logical_length_tensor

# TODO: Both DEVICE_GRIDS fit in chip 0's 16,384 rows, so chips 1-3 hold only padding and the ring
# transfer and padding mask are barely exercised. Add (1) a case with `max_patches` lowered to 4,096
# so a 40x100 or 64x64 image spans all 4 chips, and (2) a full-size 512x128 image (no padding) whose
# golden is computed for ~1,024 sampled query rows against all keys, checked against upstream
# `eager_attention` on a small image. Same for test_vision_block.py.


@RING
@REFERENCE_LAYERS
@DEVICE_GRIDS
@NOT_WRITTEN
def test_vision_attention(mesh_device, sp_axis, num_links, config, reference, grid, layer):
    parallel = resolve_parallel(mesh_device, *ring_parallel(mesh_device, sp_axis, num_links), config)
    tt = TT_Vision_Attention(
        config, mesh_device=mesh_device, parallel=parallel, compute_kernel_config=compute_kernel_config(mesh_device)
    )
    block = reference.vision_tower.encoder.blocks[layer]
    tt.load_torch_state_dict({k: v for k, v in block.state_dict().items() if k.startswith(("wqkv.", "wo."))})

    grid_thws = torch.tensor(grid)
    h, w = check_grid(grid_thws, config)
    x = torch.randn(h * w, config.hidden_size, dtype=torch.bfloat16)
    cu_seqlens, max_seqlen, rope_freqs_cis = reference_rope(reference, grid_thws)
    with torch.no_grad():
        golden = block.attention_qkvpacked(x, cu_seqlens, max_seqlen, rope_freqs_cis)

    rope = TT_Rope(config, mesh_device=mesh_device, parallel=parallel).get_rot_mats(h, w)
    x_window = functional.to_window_order(x, h, w, config.merge_kernel_size)
    logical_n = logical_length_tensor(mesh_device, h * w)
    out = tt(sp_shard(x_window, mesh_device, sp_axis, config.max_patches), rope, logical_n)
    out = sp_gather(out, mesh_device, sp_axis, h * w)
    assert_pcc(golden, functional.from_window_order(out, h, w, config.merge_kernel_size))
