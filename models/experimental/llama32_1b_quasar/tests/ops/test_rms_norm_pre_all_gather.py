# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.rms_norm_pre_all_gather`` (distributed RMSNorm, part 1).

Model call site:
  modules/rmsnorm/rmsnorm_1d.py:288  (_prefill_1d_distributed)
    tt_stats = ttnn.rms_norm_pre_all_gather(
        x, compute_kernel_config=cfg.compute_kernel_config, dtype=ttnn.bfloat16)

What it computes
----------------
This op produces the *partial* per-shard statistics for distributed RMSNorm
(the local sum-of-squares E[x^2], packed into a TILE-wide stats tensor of dtype
BFLOAT16). Those stats are then all-gathered across devices and consumed by
``ttnn.rms_norm_post_all_gather``. On the single (1, 1) emulator mesh there is no
gather, so the stats already cover the full hidden dim.

Because the stats tensor is an internal partial-reduction (not a full RMSNorm
output), building a clean torch reference is awkward — we assert the output
shape family / dtype / finiteness instead (per op_utils guidance). The full
end-to-end RMSNorm result is validated in test_rms_norm_post_all_gather.py, which
chains pre -> post.

Model input shapes: prefill activations [1, 1, seq, DIM]. Weight is applied only
in the post stage, so no weight is passed here.
"""

import pytest

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


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [pytest.param((1, 1, seq, U.DIM), id=f"prefill-seq{seq}") for seq in U.PREFILL_SEQ_LENS],
)
def test_rms_norm_pre_all_gather(ttnn_mesh_device, reset_seeds, shape):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh)

    stats = ttnn.rms_norm_pre_all_gather(
        x,
        compute_kernel_config=_compute_kernel_config(),
        dtype=ttnn.bfloat16,
    )

    # Stats: TILE layout, BFLOAT16, leading dims match input, last dim tile-multiple.
    assert tuple(stats.shape)[:3] == tuple(shape)[:3], f"leading dims mismatch: {tuple(stats.shape)} vs {shape}"
    assert stats.shape[-1] % U.TILE == 0, f"stats last dim {stats.shape[-1]} not a multiple of TILE"
    U.assert_shape_dtype(stats, dtype=ttnn.bfloat16, finite=True, mesh_device=mesh)
