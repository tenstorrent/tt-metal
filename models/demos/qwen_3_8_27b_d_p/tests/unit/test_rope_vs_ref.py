# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Partial RoPE on device vs the torch reference.

Three properties, each its own failure mode:
 1. the composed rotation matches ``apply_rotary_pos_emb`` (the half-split convention, not
    pairwise interleave);
 2. the dims past ``rotary_dim`` pass through **bit-identically** — a full rotation of a
    256-wide head is a silent PCC loss, not an error;
 3. the SP-sharded cos/sin give row ``r`` the angles for ITS token block, which is the property
    that lets the query side skip the indexed whole-cache RoPE table entirely.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.qwen_3_8_27b_d_p.reference.modeling import Qwen35RotaryEmbedding, apply_rotary_pos_emb
from models.demos.qwen_3_8_27b_d_p.tt.rope import RotarySetup, apply_partial_rope

from ..test_factory import mesh_setup, parametrize_mesh, unit_test_config
from .helpers import check_pcc, randn


@parametrize_mesh()
def test_partial_rope_vs_ref(mesh, submesh_shape, device_params):
    cfg = unit_test_config()
    mesh_config, _ccl = mesh_setup(mesh)
    n_heads = cfg.num_attention_heads // mesh_config.tp
    s_local = 128
    total = s_local * mesh_config.sp
    start_pos = 0

    ref_rope = Qwen35RotaryEmbedding(cfg)
    cos, sin = ref_rope(torch.arange(start_pos, start_pos + total)[None, :])
    x = randn(1, n_heads, total, cfg.head_dim, seed=21)
    with torch.no_grad():
        expected, _ = apply_rotary_pos_emb(x, x, cos, sin)

    setup = RotarySetup(mesh, cfg, mesh_config)
    tt_cos, tt_sin = setup.chunk_mats(start_pos, total)
    # The activation is SP-sharded on the sequence, like the real query tensor.
    tt_x = ttnn.from_torch(
        x.float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.sequence_parallel(mesh, seq_dim=2),
    )
    out = apply_partial_rope(tt_x, tt_cos, tt_sin)
    shards = ttnn.get_device_tensors(out)
    got = torch.cat([ttnn.to_torch(shards[r * mesh_config.tp]) for r in range(mesh_config.sp)], dim=2)
    check_pcc("partial_rope", expected, got, shape=(1, n_heads, total, cfg.head_dim))


@parametrize_mesh(graded_only=True)
def test_partial_rope_leaves_the_tail_untouched(mesh, submesh_shape, device_params):
    """Dims ``[rotary_dim:]`` must come back unchanged — checked on device, not just in the
    reference, because the slice/concat is where a device-side off-by-one would live."""
    cfg = unit_test_config()
    mesh_config, _ccl = mesh_setup(mesh)
    s_local, n_heads = 64, 2
    total = s_local * mesh_config.sp
    x = randn(1, n_heads, total, cfg.head_dim, seed=22)

    setup = RotarySetup(mesh, cfg, mesh_config)
    tt_cos, tt_sin = setup.chunk_mats(0, total)
    tt_x = ttnn.from_torch(
        x.float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.sequence_parallel(mesh, seq_dim=2),
    )
    out = apply_partial_rope(tt_x, tt_cos, tt_sin)
    before = ttnn.to_torch(ttnn.get_device_tensors(tt_x)[0])
    after = ttnn.to_torch(ttnn.get_device_tensors(out)[0])
    assert torch.equal(
        after[..., cfg.rotary_dim :], before[..., cfg.rotary_dim :]
    ), "the non-rotary tail changed: the rotation is covering more than rotary_dim"
    assert not torch.equal(after[..., : cfg.rotary_dim], before[..., : cfg.rotary_dim])


@parametrize_mesh()
def test_sp_rows_get_their_own_positions(mesh, submesh_shape, device_params):
    """Row ``r``'s cos/sin must be the global table's rows for ITS token block, including a
    non-zero chunk start. Getting this wrong rotates every SP row from position 0."""
    cfg = unit_test_config()
    mesh_config, _ccl = mesh_setup(mesh)
    s_local = 64
    total = s_local * mesh_config.sp
    start_pos = 5120

    ref_rope = Qwen35RotaryEmbedding(cfg)
    cos, _sin = ref_rope(torch.arange(start_pos, start_pos + total)[None, :])
    setup = RotarySetup(mesh, cfg, mesh_config)
    tt_cos, _tt_sin = setup.chunk_mats(start_pos, total)

    shards = ttnn.get_device_tensors(tt_cos)
    for r in range(mesh_config.sp):
        got = ttnn.to_torch(shards[r * mesh_config.tp]).reshape(1, 1, s_local, cfg.rotary_dim)
        expected = cos[:, r * s_local : (r + 1) * s_local].reshape(1, 1, s_local, cfg.rotary_dim)
        check_pcc(f"rope_cos[row{r}]", expected, got)
