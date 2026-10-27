# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.experimental.nlp_concat_heads_decode`` (decode head concat).

Model call site (modules/attention/attention_1d.py):
  * L732  decode_forward STAGE 9 — concatenates the per-head decode SDPA output
          into a single hidden dimension before the WO matmul:

            attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(
                attn_output_sharded, num_heads=n_local_heads
            )

Input is the decode SDPA output ``[1, batch, n_heads, head_dim]`` moved to the
scores memcfg at attention_1d.py:727-729 (``decode_scores_memcfg(batch)``). That
memcfg is HEIGHT_SHARDED, shard ``(ceil(n_local_heads/32)*32, head_dim)`` = (32, 64),
one shard per user over ``batch`` cores (attention_1d.py:1727-1734). The op's device
validate requires exactly this (nlp_concat_heads_decode_device_operation.cpp:23-47:
HEIGHT_SHARDED, shard[0]==padded_heads, shard[1]==head_dim, num_cores==num_users).
On a single (1,1) device n_heads = 32, head_dim = 64. Output is
``[1, 1, batch, n_heads*head_dim]`` (= Q_DIM 2048). No simple torch reference —
assert output shape / dtype / finiteness.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize(
    "batch",
    [pytest.param(batch, id=f"decode-batch{batch}") for batch in U.DECODE_BATCHES],
)
def test_nlp_concat_heads_decode(ttnn_mesh_device, reset_seeds, batch):
    mesh = ttnn_mesh_device

    # Per-head decode attention output: [1, batch, n_heads(=32, tile-padded), head_dim].
    attn_torch = U.torch_rand((1, batch, U.N_HEADS, U.HEAD_DIM))
    # HEIGHT_SHARDED over `batch` cores, shard (padded_heads=32, head_dim=64) —
    # matches decode_scores_memcfg (attention_1d.py:1727-1734).
    scores_memcfg = U.height_sharded_batch_memcfg(mesh, batch, (U.N_HEADS, U.HEAD_DIM))
    attn_output_sharded = U.to_tt(attn_torch, mesh, memory_config=scores_memcfg)

    attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(attn_output_sharded, num_heads=U.N_HEADS)

    # Concatenated hidden dim: [1, 1, padded_batch, n_heads*head_dim] == [1, 1, padded_batch, Q_DIM].
    # The op tile-pads the user/batch dim to a full tile (nearest_32(batch)): batch=1 -> 32, 32 -> 32.
    padded_batch = ((batch + U.TILE - 1) // U.TILE) * U.TILE
    U.assert_shape_dtype(
        attn_output_cat,
        shape=(1, 1, padded_batch, U.N_HEADS * U.HEAD_DIM),
        dtype=ttnn.bfloat16,
        mesh_device=mesh,
    )
