# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.experimental.nlp_create_qkv_heads_decode`` (decode head split).

Model call site (modules/attention/attention_1d.py):
  * L685  decode_forward STAGE 3 — splits the fused QKV projection into Q/K/V
          heads for the single-token-per-user decode path:

            q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
                xqkv_fused,
                num_heads=n_local_heads,
                num_kv_heads=n_local_kv_heads,
                overlap_qk_coregrid=self._decode_overlap_qk_coregrid,
                memory_config=cfg.decode_create_qkv_head_memcfg,
            )

Just before the call the model reshapes the fused QKV to
``(1, 1, max_batch_size, QKV_DIM)`` (attention_1d.py:682). On a single (1,1)
device n_local_heads = 32 and n_local_kv_heads = 8. Output heads are
``[1, batch, n_heads, head_dim]`` (Q) and ``[1, batch, n_kv_heads, head_dim]``
(K/V). No simple torch reference — assert head shapes / dtype / finiteness.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize(
    "batch",
    [pytest.param(batch, id=f"decode-batch{batch}") for batch in U.DECODE_BATCHES],
)
def test_nlp_create_qkv_heads_decode(ttnn_mesh_device, reset_seeds, batch):
    mesh = ttnn_mesh_device

    # Fused QKV projection reshaped for decode: [1, 1, batch, QKV_DIM].
    xqkv_torch = U.torch_rand((1, 1, batch, U.QKV_DIM))
    xqkv_fused = U.to_tt(xqkv_torch, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv_fused,
        num_heads=U.N_HEADS,
        num_kv_heads=U.N_KV_HEADS,
        overlap_qk_coregrid=True,  # matches _decode_overlap_qk_coregrid when use_qk_fused=False
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Decode layout: [1, batch, n_heads, head_dim] for Q; n_kv_heads for K/V.
    U.assert_shape_dtype(q_heads, shape=(1, batch, U.N_HEADS, U.HEAD_DIM), dtype=ttnn.bfloat16, mesh_device=mesh)
    U.assert_shape_dtype(k_heads, shape=(1, batch, U.N_KV_HEADS, U.HEAD_DIM), dtype=ttnn.bfloat16, mesh_device=mesh)
    U.assert_shape_dtype(v_heads, shape=(1, batch, U.N_KV_HEADS, U.HEAD_DIM), dtype=ttnn.bfloat16, mesh_device=mesh)
