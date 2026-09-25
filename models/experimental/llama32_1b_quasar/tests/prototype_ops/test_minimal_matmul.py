# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.experimental.minimal_matmul``.

The model opts into minimal_matmul (instead of ttnn.linear) for two large prefill
matmuls when the folded seq_len > 128 (TTTv1 parity):

  * MLP W2 down:   [*, INTERMEDIATE] x [INTERMEDIATE, DIM]   (mlp_1d.py:364)
  * Attention QKV: [*, DIM]          x [DIM, QKV_DIM]        (attention_1d.py:432)

Kwarg structure copied from the call sites: positional (x, weight), plus
``compute_kernel_config`` and a ``ttnn.MinimalMatmulConfig`` (8/8/8 blocks over the
compute grid — mlp_1d.py:977 / attention_1d.py:1769). The block sizes mirror the
model; the compute grid is taken from the device rather than hardcoding 8x8 so the
op fits the emulator.

Reference is ``x @ w``; large-K (INTERMEDIATE=8192) uses a looser pcc.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U

# Only the prefill lengths above the minimal_matmul threshold (seq_len > 128).
_MM_SEQ_LENS = [s for s in U.PREFILL_SEQ_LENS if s > 128]

# (name, in_dim/K, out_dim/N, pcc)
_MM_SITES = [
    ("mlp_w2_down", U.INTERMEDIATE, U.DIM, 0.98),  # K=8192 → looser pcc
    ("attn_qkv", U.DIM, U.QKV_DIM, 0.99),
]


def _compute_kernel_config() -> ttnn.WormholeComputeKernelConfig:
    # HiFi2 / fp16 acc — matches the mlp/attention prefill kernel configs.
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


@U.with_default_mesh()
@pytest.mark.parametrize("seq", _MM_SEQ_LENS)
@pytest.mark.parametrize(
    "name, in_dim, out_dim, pcc",
    [pytest.param(*s, id=s[0]) for s in _MM_SITES],
)
def test_minimal_matmul(ttnn_mesh_device, reset_seeds, name, in_dim, out_dim, pcc, seq):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand((1, 1, seq, in_dim))
    w_torch = U.torch_rand((in_dim, out_dim))

    x = U.to_tt(x_torch, mesh)
    w = U.to_tt(w_torch, mesh)

    grid = mesh.compute_with_storage_grid_size()  # emulator-sized (source uses 8x8)
    config = ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        compute_with_storage_grid_size=grid,
    )

    out = ttnn.experimental.minimal_matmul(
        x,
        w,
        compute_kernel_config=_compute_kernel_config(),
        config=config,
    )

    ref = x_torch.float() @ w_torch.float()
    U.assert_pcc(ref, out, pcc=pcc, mesh_device=mesh)
