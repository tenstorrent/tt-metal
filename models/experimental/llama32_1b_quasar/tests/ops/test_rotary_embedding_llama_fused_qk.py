# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.experimental.rotary_embedding_llama_fused_qk`` (decode fused QK RoPE).

Model call site (modules/attention/attention_1d.py):
  * L988  _rotary_embed_decode_fused — applies RoPE to Q and K in a single kernel
          during the decode path (use_qk_fused=True):

            return ttnn.experimental.rotary_embedding_llama_fused_qk(
                q_heads_pre_rot, k_heads_pre_rot,
                rot_mats[0], rot_mats[1], cfg.transformation_mat_decode,
            )

On a single (1,1) device n_heads = 32, n_kv_heads = 8, head_dim = 64. Inputs
(decode layout):
  * q:        [1, batch, n_heads, head_dim]
  * k:        [1, batch, n_kv_heads, head_dim]
  * cos/sin:  [1, batch, TILE, head_dim]
  * trans_mat:[1, 1, TILE, TILE]

The fused op returns the rotated (q, k) pair. No simple torch reference — assert
each output's shape / dtype / finiteness.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tensor_utils import get_rot_transformation_mat
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize(
    "batch",
    [pytest.param(batch, id=f"decode-batch{batch}") for batch in U.DECODE_BATCHES],
)
def test_rotary_embedding_llama_fused_qk(ttnn_mesh_device, reset_seeds, batch):
    mesh = ttnn_mesh_device
    hd = U.HEAD_DIM

    # Q/K decode heads: [1, batch, n(_kv)_heads, head_dim]. Head axis is tile-padded
    # to 32 in the shard (nearest_32(32)=nearest_32(8)=32), matching the model's
    # decode_create_qkv_head_memcfg output.
    q_torch = U.torch_rand((1, batch, U.N_HEADS, hd))
    k_torch = U.torch_rand((1, batch, U.N_KV_HEADS, hd))
    # cos/sin are repeated across Q AND K, so cos_sin batch == 2*batch (device
    # validate: cos_sin_batch_size == q_batch + k_batch; _fused_qk_device_op.cpp:85-90).
    cos_torch = U.torch_rand((1, 2 * batch, U.TILE, hd))
    sin_torch = U.torch_rand((1, 2 * batch, U.TILE, hd))

    # The fused kernel requires Q and K on NON-OVERLAPPING HEIGHT_SHARDED grids, and
    # cos/sin/trans_mat sharded over the union (2*batch cores). Grids mirror
    # RotarySetup / _reshard_k_for_fused (attention_1d.py:1242-1250): Q on the first
    # `batch` cores, K on the next `batch` cores (rows of 8).
    q_memcfg = U.height_sharded_batch_memcfg(mesh, batch, (U.TILE, hd))
    k_start = ttnn.CoreCoord(batch % 8, batch // 8)
    k_memcfg = U.height_sharded_batch_memcfg(mesh, batch, (U.TILE, hd), start_core=k_start)
    cos_sin_memcfg = U.height_sharded_batch_memcfg(mesh, 2 * batch, (U.TILE, hd))

    q = U.to_tt(q_torch, mesh, memory_config=q_memcfg)
    k = U.to_tt(k_torch, mesh, memory_config=k_memcfg)
    cos = U.to_tt(cos_torch, mesh, memory_config=cos_sin_memcfg)
    sin = U.to_tt(sin_torch, mesh, memory_config=cos_sin_memcfg)

    # Decode trans-mat: the base 32x32 rotation repeated over the 2*batch Q+K cores,
    # sharded to a single (32,32) tile per core (attention_1d.py:2046-2062).
    trans_mat_torch = get_rot_transformation_mat().repeat(1, 1, 2 * batch, 1)  # [1,1,2*batch*32,32]
    trans_mat_memcfg = U.height_sharded_batch_memcfg(mesh, 2 * batch, (U.TILE, U.TILE))
    trans_mat = U.to_tt(trans_mat_torch, mesh, memory_config=trans_mat_memcfg)

    q_out, k_out = ttnn.experimental.rotary_embedding_llama_fused_qk(q, k, cos, sin, trans_mat)

    U.assert_shape_dtype(q_out, shape=(1, batch, U.N_HEADS, hd), dtype=ttnn.bfloat16, mesh_device=mesh)
    # k_out head axis is tile-padded to 32 (nearest_32(n_kv_heads=8)=32), matching the sharded layout.
    U.assert_shape_dtype(k_out, shape=(1, batch, U.TILE, hd), dtype=ttnn.bfloat16, mesh_device=mesh)
