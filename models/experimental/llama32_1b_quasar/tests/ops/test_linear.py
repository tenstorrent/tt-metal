# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.linear``.

Covers every distinct ``ttnn.linear`` call site in the model, each with its real
matmul dims (K = in_dim, N = out_dim). One parametrization per weight matrix:

  * MLP W1 gate:   [*, DIM]          x [DIM, INTERMEDIATE]  (mlp_1d.py:227 decode / :321 prefill)
  * MLP W3 up:     [*, DIM]          x [DIM, INTERMEDIATE]  (mlp_1d.py:241 / :330)
  * MLP W2 down:   [*, INTERMEDIATE] x [INTERMEDIATE, DIM]  (mlp_1d.py:273 / :371)
  * Attention QKV: [*, DIM]          x [DIM, QKV_DIM]       (attention_1d.py:439 / :660)
  * Attention WO:  [*, Q_DIM]        x [Q_DIM, DIM]         (attention_1d.py:606 / :1222)
  * LM head:       [*, DIM]          x [DIM, VOCAB]         (lm_head_1d.py:133 / :144)

The model drives these matmuls with DRAM-sharded / width-sharded program configs;
here we exercise the *plain* interleaved bf16 ``ttnn.linear`` (no program_config) so
each op runs standalone on the emulator. Reference is ``x @ w``.

Rows M come from the prefill sequence lengths (U.PREFILL_SEQ_LENS) and the decode
batch sizes (U.DECODE_BATCHES) the demo drives through the model.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U

# (name, in_dim/K, out_dim/N, pcc). Large-K matmuls (W2 down, K=8192) accumulate more
# bf16 rounding error over the reduction, so allow a slightly looser 0.98.
_LINEAR_SITES = [
    ("mlp_w1_gate", U.DIM, U.INTERMEDIATE, 0.99),
    ("mlp_w3_up", U.DIM, U.INTERMEDIATE, 0.99),
    ("mlp_w2_down", U.INTERMEDIATE, U.DIM, 0.98),  # K=8192 → looser pcc
    ("attn_qkv", U.DIM, U.QKV_DIM, 0.99),
    ("attn_wo", U.Q_DIM, U.DIM, 0.99),
    ("lm_head", U.DIM, U.VOCAB, 0.99),
]

_M_SIZES = [pytest.param(seq, id=f"prefill-seq{seq}") for seq in U.PREFILL_SEQ_LENS] + [
    pytest.param(batch, id=f"decode-batch{batch}") for batch in U.DECODE_BATCHES
]


@U.with_default_mesh()
@pytest.mark.parametrize("m", _M_SIZES)
@pytest.mark.parametrize(
    "name, in_dim, out_dim, pcc",
    [pytest.param(*s, id=s[0]) for s in _LINEAR_SITES],
)
def test_linear(ttnn_mesh_device, reset_seeds, name, in_dim, out_dim, pcc, m):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand((1, 1, m, in_dim))
    w_torch = U.torch_rand((in_dim, out_dim))

    x = U.to_tt(x_torch, mesh)
    w = U.to_tt(w_torch, mesh)

    out = ttnn.linear(x, w, dtype=ttnn.bfloat16)

    ref = x_torch.float() @ w_torch.float()
    U.assert_pcc(ref, out, pcc=pcc, mesh_device=mesh)
