# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS indexed-RoPE device test. Validates that gpt_oss_d_p's ``build_indexed_rope`` (whole-cache,
block-cyclic, SP-sharded YaRN cos/sin, truncate=False) + the indexed ``apply_rope`` path apply the
SAME rotation at a nonzero tile-aligned offset as the plain non-indexed op (``rotary_embedding_llama``,
which test_attention_vs_ref validates against the HF reference at offset 0) does at the corresponding
absolute positions.

The raw ``rotary_embedding_indexed`` op is exercised thoroughly in
``deepseek_v3_d_p/tests/op_unit_tests/test_deepseek_prefill_rotary_embedding_indexed.py``; this test
covers the gpt-oss-specific glue (YaRN table build + block-cyclic assembly + the offset dispatch)
that the attention reference test (offset 0, non-indexed) does not.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss_d_p.tt.attention.operations import apply_rope
from models.demos.gpt_oss_d_p.tt.rope import build_indexed_rope, build_transformation_mat, build_yarn_cos_sin

HEAD_DIM = 64


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("n_heads", [8], ids=["h8"])
@pytest.mark.parametrize("chunk, offset", [(128, 128)], ids=["c128-off128"])
def test_indexed_rope_matches_nonindexed_at_offset(mesh_device, n_heads, chunk, offset, reset_seeds):
    """Indexed RoPE at ``kv_actual_global=offset`` == non-indexed RoPE with cos/sin sliced to
    ``[offset, offset+chunk)``. Single card (sp=1): block-cyclic degenerates to identity, so this
    isolates the YaRN-table + indexed-offset-dispatch correctness (block-cyclic layout is covered
    separately by the pure-torch inverse test)."""
    rows, cols = tuple(mesh_device.shape)
    sp_axis, tp_axis = 0, 1
    sp = rows
    assert offset % (ttnn.TILE_SIZE * sp) == 0 and chunk % (ttnn.TILE_SIZE * sp) == 0
    max_seq_len = offset + chunk  # cache big enough to hold the prefix + this chunk

    torch.manual_seed(0)
    q = torch.randn(1, n_heads, chunk, HEAD_DIM)  # this chunk's Q (tokens at global [offset, offset+chunk))

    xform = build_transformation_mat(mesh_device)

    q_dims = [None, None]
    q_dims[sp_axis] = 2  # seq on SP rows
    q_dims[tp_axis] = 1  # heads on TP cols

    def to_dev(t, shard_seq):
        dims = [None, None]
        if shard_seq:
            dims[sp_axis] = 2
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=tuple(dims))
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def q_to_dev():
        return ttnn.from_torch(
            q,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=tuple(q_dims)),
        )

    # --- indexed path: whole-cache block-cyclic cos/sin + rotary_embedding_indexed at the offset ---
    idx_cos_sin = build_indexed_rope(
        mesh_device, head_dim=HEAD_DIM, max_seq_len=max_seq_len, chunk_size=chunk, sp_axis=sp_axis
    )
    out_idx = apply_rope(q_to_dev(), idx_cos_sin, xform, kv_actual_global=offset, cluster_axis=sp_axis)
    idx_host = ttnn.to_torch(ttnn.get_device_tensors(out_idx)[0]).float()[:, :n_heads]

    # --- non-indexed reference: same YaRN cos/sin, sliced to [offset, offset+chunk) ---
    cos, sin = build_yarn_cos_sin(max_seq_len, HEAD_DIM)  # [1, 1, max_seq_len, head_dim]
    cos_sl = to_dev(cos[:, :, offset : offset + chunk, :], shard_seq=True)
    sin_sl = to_dev(sin[:, :, offset : offset + chunk, :], shard_seq=True)
    out_ref = apply_rope(q_to_dev(), [cos_sl, sin_sl], xform)  # kv_actual_global=None -> non-indexed
    ref_host = ttnn.to_torch(ttnn.get_device_tensors(out_ref)[0]).float()[:, :n_heads]

    ok, pcc = comp_pcc(ref_host, idx_host, 0.99)
    logger.info(f"indexed vs non-indexed RoPE @ offset={offset}: pcc={pcc}")
    assert ok, f"indexed RoPE at offset {offset} disagrees with the non-indexed reference: {pcc}"
