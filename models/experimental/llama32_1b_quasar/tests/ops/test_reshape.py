# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.reshape``.

``ttnn.reshape`` is used purely to fold / unfold the batch and sequence axes so
long-sequence matmuls fit on device; it is value-preserving (row-major), so the
reference is ``torch.reshape``.

Model call sites (distinct patterns covered here):
  * mlp_1d.py:314        fold long prefill seq -> [1, seq//cutoff, cutoff, -1]
                         (prefill_len_cutoff = 512 on BH / 1024 otherwise; mlp_1d.py:858)
  * attention_1d.py:425  fold qkv seq        -> [1, seq//MAX_QKV_MM_SEQ_LEN, 2048, -1]
  * attention_1d.py:459  unfold qkv back      -> [1, 1, seq, -1]
  * attention_1d.py:599  fold for WO matmul   -> [1, seq//MAX_MM_SEQ_LEN, 1024, -1]
  * mlp_1d.py:289        collapse leading dims -> [1, 1, a*b*c, N]

MAX_QKV_MM_SEQ_LEN = 2048, MAX_MM_SEQ_LEN = 1024 (attention_1d.py:66,72).
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U

# (id, in_shape, out_shape). -1 is resolved by both torch.reshape and ttnn.reshape.
_RESHAPE_SITES = [
    # mlp_1d.py:314 — fold long prefill seq (cutoff = 512)
    ("mlp_prefill_fold", (1, 1, 1024, U.DIM), (1, 1024 // 512, 512, -1)),
    # attention_1d.py:459 — unfold folded qkv back to a single long sequence
    ("attn_qkv_unfold", (1, 2, 512, U.DIM), (1, 1, 1024, -1)),
    # mlp_1d.py:289 — collapse leading dims to [1, 1, M, N]
    ("mlp_collapse", (1, 4, 256, U.DIM), (1, 1, 4 * 256, U.DIM)),
    # attention_1d.py:599 — fold for WO matmul (MAX_MM_SEQ_LEN = 1024)
    ("attn_wo_fold", (1, 1, 2048, U.DIM), (1, 2048 // 1024, 1024, -1)),
]


@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, in_shape, out_shape",
    [pytest.param(*s, id=s[0]) for s in _RESHAPE_SITES],
)
def test_reshape(ttnn_mesh_device, reset_seeds, name, in_shape, out_shape):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(in_shape)
    x = U.to_tt(x_torch, mesh)

    out = ttnn.reshape(x, list(out_shape))

    ref = torch.reshape(x_torch.float(), out_shape)
    U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)
