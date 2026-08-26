# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.experimental.nlp_concat_heads`` (prefill head concat).

Model call site (modules/attention/attention_1d.py):
  * L588  prefill_forward STAGE 11 — concatenates the per-head SDPA output back
          into a single hidden dimension before the WO matmul:

            attn_output_concat = ttnn.experimental.nlp_concat_heads(
                attn_output, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

Input is the per-head attention output ``[1, n_heads, seq, head_dim]`` (the
single-user reshape at attention_1d.py:586). On a single (1,1) device
n_heads = 32, head_dim = 64, so the concatenated output is
``[1, 1, seq, n_heads*head_dim]`` = ``[1, 1, seq, Q_DIM]`` (Q_DIM = 2048).
No simple torch reference — assert output shape / dtype / finiteness.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize(
    "seq",
    [pytest.param(seq, id=f"prefill-seq{seq}") for seq in U.PREFILL_SEQ_LENS],
)
def test_nlp_concat_heads(ttnn_mesh_device, reset_seeds, seq):
    mesh = ttnn_mesh_device

    # Per-head attention output: [1, n_heads, seq, head_dim].
    attn_torch = U.torch_rand((1, U.N_HEADS, seq, U.HEAD_DIM))
    attn_output = U.to_tt(attn_torch, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    attn_output_concat = ttnn.experimental.nlp_concat_heads(attn_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Concatenated hidden dim: [1, 1, seq, n_heads*head_dim] == [1, 1, seq, Q_DIM].
    U.assert_shape_dtype(
        attn_output_concat,
        shape=(1, 1, seq, U.N_HEADS * U.HEAD_DIM),
        dtype=ttnn.bfloat16,
        mesh_device=mesh,
    )
    # Values: heads concatenated along the last dim -> [1, 1, seq, n_heads*head_dim].
    ref = attn_torch.permute(0, 2, 1, 3).reshape(1, 1, seq, U.N_HEADS * U.HEAD_DIM)
    U.assert_pcc(ref, attn_output_concat, pcc=0.999, mesh_device=mesh)
