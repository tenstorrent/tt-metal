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
(K/V).

Two tests:
  * ``test_nlp_create_qkv_heads_decode`` — interleaved input. Routes to the
    ``InterleavedProgramFactory`` (reader_interleaved_* kernel). The model never
    feeds interleaved input, but keeping this exercises that program factory.
  * ``test_nlp_create_qkv_heads_decode_sharded`` — WIDTH_SHARDED input, the path
    the model actually uses. select_program_factory() branches on
    input.is_sharded() to a *different* program factory + reader kernel
    (nlp_create_qkv_heads_decode_device_operation.cpp:12-24), so this is the real
    coverage. Verified with a torch reference (the split has a closed form).
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


@U.with_default_mesh()
@pytest.mark.parametrize(
    "batch",
    [pytest.param(batch, id=f"decode-batch{batch}") for batch in U.DECODE_BATCHES],
)
def test_nlp_create_qkv_heads_decode_sharded(ttnn_mesh_device, reset_seeds, batch):
    """WIDTH_SHARDED input — the program-factory / reader kernel the model actually uses.

    Input contract (device_operation.cpp:55-66): WIDTH_SHARDED, ROW_MAJOR, shard[0] ==
    padded_batch. We place the full QKV width on a single core (canonical
    max-width-shard, test_nlp_create_qkv_heads_decode.py:106-137) so batch=1 fits the
    emulator; the model's default output config is L1_HEIGHT_SHARDED (attention_1d.py:1723).
    """
    mesh = ttnn_mesh_device
    total_heads = U.N_HEADS + 2 * U.N_KV_HEADS  # 48; head_dim * total_heads == QKV_DIM

    xqkv_torch = U.torch_rand((1, 1, batch, U.QKV_DIM))
    # Host tile tensor first so we can read the tile-padded batch (canonical pattern).
    xqkv_host = ttnn.from_torch(
        xqkv_torch,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
    )
    padded_batch = xqkv_host.padded_shape[2]

    one_core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    in_memcfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(one_core, [padded_batch, total_heads * U.HEAD_DIM], ttnn.ShardOrientation.ROW_MAJOR),
    )
    xqkv_fused = ttnn.to_device(xqkv_host, mesh, memory_config=in_memcfg)

    q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv_fused,
        num_heads=U.N_HEADS,
        num_kv_heads=U.N_KV_HEADS,
        overlap_qk_coregrid=True,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,  # model default (attention_1d.py:1723)
    )

    # Value check: split the fused QKV on -1 into [1, batch, heads, head_dim].
    q_ref = xqkv_torch[:, :, :batch, : U.Q_DIM].reshape(1, batch, U.N_HEADS, U.HEAD_DIM)
    k_ref = xqkv_torch[:, :, :batch, U.Q_DIM : U.Q_DIM + U.KV_DIM].reshape(1, batch, U.N_KV_HEADS, U.HEAD_DIM)
    v_ref = xqkv_torch[:, :, :batch, U.Q_DIM + U.KV_DIM :].reshape(1, batch, U.N_KV_HEADS, U.HEAD_DIM)
    U.assert_pcc(q_ref, q_heads, pcc=0.999, mesh_device=mesh)
    U.assert_pcc(k_ref, k_heads, pcc=0.999, mesh_device=mesh)
    U.assert_pcc(v_ref, v_heads, pcc=0.999, mesh_device=mesh)
