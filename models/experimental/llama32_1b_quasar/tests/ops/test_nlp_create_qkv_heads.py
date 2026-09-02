# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.experimental.nlp_create_qkv_heads`` (prefill head split).

Model call site (modules/attention/attention_1d.py):
  * L470  prefill_forward STAGE 4 — splits the fused QKV projection output into
          separate Q / K / V head tensors:

            q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
                xqkv_fused,
                num_heads=n_local_heads,
                num_kv_heads=n_local_kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

Input is the fused QKV projection ``[1, 1, seq, QKV_DIM]`` (QKV_DIM = 3072 =
Q_DIM 2048 + 2*KV_DIM 512). On a single (1,1) device n_local_heads = 32 and
n_local_kv_heads = 8. The op has no simple torch reference, so we assert the
three output tensors have the expected head shapes / dtype and are finite.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize(
    "seq",
    [pytest.param(seq, id=f"prefill-seq{seq}") for seq in U.PREFILL_SEQ_LENS],
)
def test_nlp_create_qkv_heads(ttnn_mesh_device, reset_seeds, seq):
    mesh = ttnn_mesh_device

    # Fused QKV projection output: [1, 1, seq, QKV_DIM] (Q|K|V concatenated on -1).
    xqkv_torch = U.torch_rand((1, 1, seq, U.QKV_DIM))
    xqkv_fused = U.to_tt(xqkv_torch, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=U.N_HEADS,
        num_kv_heads=U.N_KV_HEADS,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Q: [1, n_heads, seq, head_dim]; K/V: [1, n_kv_heads, seq, head_dim].
    U.assert_shape_dtype(q_heads, shape=(1, U.N_HEADS, seq, U.HEAD_DIM), dtype=ttnn.bfloat16, mesh_device=mesh)
    U.assert_shape_dtype(k_heads, shape=(1, U.N_KV_HEADS, seq, U.HEAD_DIM), dtype=ttnn.bfloat16, mesh_device=mesh)
    U.assert_shape_dtype(v_heads, shape=(1, U.N_KV_HEADS, seq, U.HEAD_DIM), dtype=ttnn.bfloat16, mesh_device=mesh)

    # Values: split the fused QKV on -1 and reshape each into [1, heads, seq, head_dim]
    # (transpose_k_heads=False, so K keeps the same [1, n_kv, seq, hd] layout as Q/V).
    q_ref = xqkv_torch[..., : U.Q_DIM].reshape(1, seq, U.N_HEADS, U.HEAD_DIM).permute(0, 2, 1, 3)
    k_ref = xqkv_torch[..., U.Q_DIM : U.Q_DIM + U.KV_DIM].reshape(1, seq, U.N_KV_HEADS, U.HEAD_DIM).permute(0, 2, 1, 3)
    v_ref = xqkv_torch[..., U.Q_DIM + U.KV_DIM :].reshape(1, seq, U.N_KV_HEADS, U.HEAD_DIM).permute(0, 2, 1, 3)
    U.assert_pcc(q_ref, q_heads, pcc=0.999, mesh_device=mesh)
    U.assert_pcc(k_ref, k_heads, pcc=0.999, mesh_device=mesh)
    U.assert_pcc(v_ref, v_heads, pcc=0.999, mesh_device=mesh)
