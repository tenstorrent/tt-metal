# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Rope` tables, and `rotary_embedding_llama` with them against the reference `apply_rope`."""

from __future__ import annotations

import torch

import ttnn
from models.experimental.kimi_k3_moonvit_v2.reference import functional
from models.experimental.kimi_k3_moonvit_v2.reference.modeling_kimi_k3 import apply_rope
from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    assert_pcc,
    DEVICE_GRIDS,
    device_config,
    NOT_WRITTEN,
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
@DEVICE_GRIDS
@NOT_WRITTEN
def test_rope_tables(mesh_device, sp_axis, num_links, config, grid):
    config = device_config(config, sp_axis)
    h, w = check_grid(torch.tensor(grid), config)
    rope = build(mesh_device, sp_axis, num_links, config).get_rot_mats(h, w)
    cos, sin = functional.rope_cos_sin(config, h, w)
    assert_pcc(cos, sp_gather(rope.cos, mesh_device, sp_axis, h * w), pcc=0.9999)
    assert_pcc(sin, sp_gather(rope.sin, mesh_device, sp_axis, h * w), pcc=0.9999)


@P150_AND_RING
@REFERENCE
@DEVICE_GRIDS
@NOT_WRITTEN
def test_rope_apply(mesh_device, sp_axis, num_links, config, reference, grid):
    config = device_config(config, sp_axis)
    grid_thws = torch.tensor(grid)
    h, w = check_grid(grid_thws, config)
    q = torch.randn(h * w, config.num_attention_heads, config.head_dim, dtype=torch.bfloat16)
    _, _, rope_freqs_cis = reference_rope(reference, grid_thws)
    golden, _ = apply_rope(q, q, rope_freqs_cis)

    rope = build(mesh_device, sp_axis, num_links, config).get_rot_mats(h, w)
    q_tt = sp_shard(q.transpose(0, 1), mesh_device, sp_axis, config.max_patches)  # [1, heads, S/sp, head_dim]
    out = ttnn.experimental.rotary_embedding_llama(q_tt, rope.cos, rope.sin, rope.trans_mat, is_decode_mode=False)
    assert_pcc(golden, sp_gather(out, mesh_device, sp_axis, h * w).transpose(0, 1))
