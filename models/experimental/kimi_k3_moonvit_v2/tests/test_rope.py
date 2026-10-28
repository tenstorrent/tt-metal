# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Rope` tables, and `rotary_embedding_llama` with them against the reference `apply_rope`.

The tables are written in place, so each test first fills them for another image."""

from __future__ import annotations

import torch

import ttnn
from models.experimental.kimi_k3_moonvit_v2.tt import host_prep
from models.experimental.kimi_k3_moonvit_v2.reference.modeling_kimi_k3 import apply_rope
from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    assert_pcc,
    device_config,
    row_grid,
    ROW_GRIDS,
    P150_AND_RING,
    REFERENCE,
    reference_rope,
    ring_parallel,
    sp_gather,
    sp_shard,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import check_grid, resolve_parallel
from models.experimental.kimi_k3_moonvit_v2.tt.rope import TT_Rope


def build(mesh_device, sp_axis, num_links, config) -> TT_Rope:
    parallel = resolve_parallel(mesh_device, *ring_parallel(mesh_device, sp_axis, num_links), config)
    return TT_Rope(config, mesh_device=mesh_device, parallel=parallel)


@P150_AND_RING
@ROW_GRIDS
def test_rope_tables(mesh_device, sp_axis, num_links, config, size):
    config = device_config(config, sp_axis)
    grid = row_grid(size, sp_axis)
    h, w = check_grid(torch.tensor(grid), config)
    tt_rope = build(mesh_device, sp_axis, num_links, config)
    previous = tt_rope.write(*host_prep.rope_tables(config, 8, 8))
    rope = tt_rope.write(*host_prep.rope_tables(config, h, w))
    assert all(a is b for a, b in zip(rope, previous))
    cos, sin = host_prep.rope_cos_sin(config, h, w)
    cos, sin = (host_prep.to_window_order(t, h, w, config.merge_kernel_size) for t in (cos, sin))
    assert_pcc(cos, sp_gather(rope.cos, mesh_device, sp_axis, h * w), pcc=0.9999)
    assert_pcc(sin, sp_gather(rope.sin, mesh_device, sp_axis, h * w), pcc=0.9999)


@P150_AND_RING
@REFERENCE
@ROW_GRIDS
def test_rope_apply(mesh_device, sp_axis, num_links, config, reference, size):
    config = device_config(config, sp_axis)
    grid = row_grid(size, sp_axis)
    grid_thws = torch.tensor(grid)
    h, w = check_grid(grid_thws, config)
    q = torch.randn(h * w, config.num_attention_heads, config.head_dim, dtype=torch.bfloat16)
    _, _, rope_freqs_cis = reference_rope(reference, grid_thws)
    golden, _ = apply_rope(q, q, rope_freqs_cis)

    tt_rope = build(mesh_device, sp_axis, num_links, config)
    tt_rope.write(*host_prep.rope_tables(config, 8, 8))
    rope = tt_rope.write(*host_prep.rope_tables(config, h, w))
    q_window = host_prep.to_window_order(q, h, w, config.merge_kernel_size)
    q_tt = sp_shard(q_window.transpose(0, 1), mesh_device, sp_axis, config.max_patches)  # [1, heads, S/sp, head_dim]
    out = ttnn.experimental.rotary_embedding_llama(q_tt, rope.cos, rope.sin, rope.trans_mat, is_decode_mode=False)
    out = sp_gather(out, mesh_device, sp_axis, h * w).transpose(0, 1)
    assert_pcc(golden, host_prep.from_window_order(out, h, w, config.merge_kernel_size))
