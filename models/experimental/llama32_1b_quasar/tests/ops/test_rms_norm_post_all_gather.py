# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.rms_norm_post_all_gather`` (distributed RMSNorm, part 2).

Model call site:
  modules/rmsnorm/rmsnorm_1d.py:306  (_prefill_1d_distributed)
    tt_out = ttnn.rms_norm_post_all_gather(
        x, tt_stats, epsilon=cfg.eps, weight=self.weight_distributed,
        compute_kernel_config=cfg.compute_kernel_config)

This op combines the (gathered) partial statistics from
``ttnn.rms_norm_pre_all_gather`` with the input and the weight to produce the
final RMSNorm output. On the single (1, 1) emulator mesh the all-gather is a
no-op (one shard already holds the full hidden dim), so chaining
``pre_all_gather -> post_all_gather`` reproduces a full RMSNorm and we can
compare against torch.nn.RMSNorm via PCC.

Weight layout matches the module (rmsnorm_1d.py weights are ROW_MAJOR,
shape (1, 1, dim // 32, 32)); the post op requires the stats' first three padded
dims to match the input and its last padded dim to be a TILE multiple.
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U


def _compute_kernel_config():
    # Mirrors _resolve_1d_config (rmsnorm_1d.py:482-487).
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    ref = torch.nn.RMSNorm(x.shape[-1], eps=eps, dtype=torch.float32)
    ref.weight.data.copy_(weight.float())
    return ref(x.float())


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [pytest.param((1, 1, seq, U.DIM), id=f"prefill-seq{seq}") for seq in U.PREFILL_SEQ_LENS],
)
def test_rms_norm_post_all_gather(ttnn_mesh_device, reset_seeds, shape):
    mesh = ttnn_mesh_device
    dim = shape[-1]
    ckc = _compute_kernel_config()

    x_torch = U.torch_rand(shape)
    w_torch = U.torch_rand((dim,))

    x = U.to_tt(x_torch, mesh)
    # weight: (dim,) -> (1, 1, dim // TILE, TILE), ROW_MAJOR (rmsnorm_1d.py:531 / 546-551).
    w = U.to_tt(w_torch.reshape(1, 1, dim // U.TILE, U.TILE), mesh, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Part 1: partial stats (single device -> full stats, gather is a no-op).
    stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=ckc, dtype=ttnn.bfloat16)

    # Part 2: final normalized output.
    out = ttnn.rms_norm_post_all_gather(
        x,
        stats,
        epsilon=U.NORM_EPS,
        weight=w,
        compute_kernel_config=ckc,
    )

    ref = _torch_rms_norm(x_torch, w_torch, U.NORM_EPS)
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)
