# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.experimental.rotary_embedding_llama`` (separate Q/K RoPE).

Model call sites (modules/attention/attention_1d.py):
  * L491  prefill_forward STAGE 6 — Q rotary (is_decode_mode=False)
  * L499  prefill_forward STAGE 6 — K rotary (is_decode_mode=False)
  * L997  _rotary_embed_decode_nonfused — Q rotary (is_decode_mode=True)
  * L1000 _rotary_embed_decode_nonfused — K rotary (is_decode_mode=True)

        q_heads = ttnn.experimental.rotary_embedding_llama(
            q_heads_pre_rot, rot_mats[0], rot_mats[1],
            cfg.transformation_mat_prefill, is_decode_mode=False,
        )

This file covers the PREFILL (interleaved, emulator-friendly) path for both the
Q head-count (n_heads = 32) and the K head-count (n_kv_heads = 8). Inputs:
  * heads:    [1, n(_kv)_heads, seq, head_dim]
  * cos/sin:  [1, 1, seq, head_dim]
  * trans_mat:[1, 1, TILE, TILE] (the base 32x32 rotation matrix)

Transformation-matrix note: the rotary op applies the 32x32 rotation tile-wise, so
the trans-mat is always [1,1,32,32] regardless of head_dim — this is what the model's
own config resolver builds for prefill (get_rot_transformation_mat(),
attention_1d.py:2064-2075) and what the canonical op test uses even for head_dim=64
(tests/ttnn/nightly/.../test_rotary_embedding_llama.py:41-42,
get_rot_transformation_mat(dhead=ttnn.TILE_SIZE)). The earlier [1,1,64,64] trans-mat
(from the RotarySetupHelper in test_attention_1d.py:214-218) is what the op rejects
("Transformation matrix must have 4th dim equal to TILE_WIDTH (32)").

RoPE has a torch reference but it is fiddly (Meta-format cos/sin); shape / dtype /
finiteness is an accepted assertion for this op per the suite guidance.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tensor_utils import get_rot_transformation_mat
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize(
    "n_heads",
    [pytest.param(U.N_HEADS, id="q-heads"), pytest.param(U.N_KV_HEADS, id="k-heads")],
)
@pytest.mark.parametrize(
    "seq",
    [pytest.param(seq, id=f"prefill-seq{seq}") for seq in U.PREFILL_SEQ_LENS],
)
def test_rotary_embedding_llama(ttnn_mesh_device, reset_seeds, seq, n_heads):
    mesh = ttnn_mesh_device
    hd = U.HEAD_DIM

    heads_torch = U.torch_rand((1, n_heads, seq, hd))
    cos_torch = U.torch_rand((1, 1, seq, hd))
    sin_torch = U.torch_rand((1, 1, seq, hd))

    heads = U.to_tt(heads_torch, mesh)
    cos = U.to_tt(cos_torch, mesh)
    sin = U.to_tt(sin_torch, mesh)
    # Prefill trans-mat is the base 32x32 rotation ([1,1,32,32]) — applied tile-wise
    # for head_dim=64. Matches attention_1d.py:2066 and the canonical op test.
    trans_mat = U.to_tt(get_rot_transformation_mat(), mesh)

    out = ttnn.experimental.rotary_embedding_llama(heads, cos, sin, trans_mat, is_decode_mode=False)

    U.assert_shape_dtype(out, shape=(1, n_heads, seq, hd), dtype=ttnn.bfloat16, mesh_device=mesh)
