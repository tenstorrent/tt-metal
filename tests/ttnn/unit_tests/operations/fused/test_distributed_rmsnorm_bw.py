# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Simulated-device distributed RMSNorm backward (concat stands in for all-gather).

Mirrors test_distributed_rmsnorm_allgather.py in the nightly suite: the sharded hidden dim is
chunked on one device and the all-gather of the one-tile stats columns is modelled by ttnn.concat,
so the two backward stages can be checked against a full-width reference without a mesh.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics

pytestmark = pytest.mark.use_module_device

EPS = 1e-5


def _torch_rmsnorm_bw(x, dy, eps, weight=None):
    x = x.detach().clone().float().requires_grad_(True)
    dy = dy.detach().clone().float()
    w = None
    if weight is not None:
        w = weight.detach().clone().float().requires_grad_(True)
    y = torch.nn.functional.rms_norm(x, (x.shape[-1],), w, eps)
    y.backward(dy)
    return x.grad, None if w is None else w.grad


def _assert_close(ref, actual, torch_dtype):
    # Accuracy here is set by the row reduction, not by the tensor dtype. ttnn.sum over the last dim
    # lowers to a matmul against ones whose sources are bf16, so it holds a ~3e-4 relative floor no
    # matter the input dtype or compute kernel config (an explicit matmul-with-ones reproduces it
    # bit for bit, and moreh_sum rejects fp32 outright). sum(x * gained) cancels heavily, which
    # amplifies that floor into the gradients. fp32 therefore buys no accuracy over bf16 and is the
    # looser case, because it is compared against an exact reference rather than a quantized one.
    tol = 0.1 if torch_dtype == torch.bfloat16 else 0.3
    assert_numeric_metrics(
        ref.to(torch_dtype),
        actual.to(torch_dtype),
        pcc_threshold=0.999,
        check_ulp=False,
        rtol=tol,
        atol=tol,
        frobenius_threshold=0.08,
    )


def _compute_kernel_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _shard(device, torch_tensor, num_devices, ttnn_dtype):
    return [
        ttnn.from_torch(
            chunk,
            dtype=ttnn_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for chunk in torch.chunk(torch_tensor, num_devices, dim=-1)
    ]


def _run_distributed_rms_norm_bw(device, batch, seq_len, hidden_dim_total, num_devices, with_weight, ttnn_dtype):
    assert hidden_dim_total % num_devices == 0
    hidden_per_dev = hidden_dim_total // num_devices
    torch_dtype = torch.bfloat16 if ttnn_dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(1234)

    torch_x = torch.randn((batch, 1, seq_len, hidden_dim_total), dtype=torch_dtype)
    torch_w = torch.randn((hidden_dim_total,), dtype=torch_dtype) if with_weight else None
    torch_dy = torch.randn((batch, 1, seq_len, hidden_dim_total), dtype=torch_dtype)
    torch_dx, torch_dw = _torch_rmsnorm_bw(torch_x, torch_dy, EPS, torch_w)

    compute_kernel_config = _compute_kernel_config()

    tt_x = _shard(device, torch_x, num_devices, ttnn_dtype)
    tt_dy = _shard(device, torch_dy, num_devices, ttnn_dtype)
    tt_w = (
        _shard(device, torch_w.reshape(1, 1, 1, hidden_dim_total), num_devices, ttnn_dtype)
        if with_weight
        else [None] * num_devices
    )

    tt_stats_gathered = ttnn.concat(
        [ttnn.rms_norm_pre_all_gather(t, compute_kernel_config=compute_kernel_config, dtype=ttnn_dtype) for t in tt_x],
        dim=3,
    )
    tt_bw_stats_gathered = ttnn.concat(
        [
            ttnn.rms_norm_pre_all_gather_bw(
                tt_x[i],
                tt_dy[i],
                tt_stats_gathered,
                epsilon=EPS,
                weight=tt_w[i],
                compute_kernel_config=compute_kernel_config,
            )
            for i in range(num_devices)
        ],
        dim=3,
    )

    tt_dx = []
    tt_dw = []
    for i in range(num_devices):
        grads = ttnn.rms_norm_post_all_gather_bw(
            tt_x[i],
            tt_dy[i],
            tt_stats_gathered,
            tt_bw_stats_gathered,
            epsilon=EPS,
            weight=tt_w[i],
            compute_kernel_config=compute_kernel_config,
        )
        assert len(grads) == 2, f"expected [input_grad, weight_grad], got {len(grads)} entries"
        tt_dx.append(grads[0])
        if with_weight:
            assert grads[1] is not None, "weight_grad must be returned when weight is provided"
            assert tuple(grads[1].shape) == (
                1,
                1,
                1,
                hidden_per_dev,
            ), f"weight_grad must match the weight shard shape, got {tuple(grads[1].shape)}"
            tt_dw.append(grads[1])
        else:
            assert grads[1] is None, "weight_grad must be None when weight is not provided"

    _assert_close(torch_dx, ttnn.to_torch(ttnn.concat(tt_dx, dim=-1)), torch_dtype)
    if with_weight:
        _assert_close(
            torch_dw.reshape(1, 1, 1, hidden_dim_total),
            ttnn.to_torch(ttnn.concat(tt_dw, dim=-1)),
            torch_dtype,
        )


@pytest.mark.parametrize(
    "batch, seq_len, hidden_dim_total, num_simulated_devices",
    [
        (1, 32, 128, 2),
        (1, 32, 256, 4),
        # Batch > 1 exercises the reduction of dL/dgamma over the leading dims.
        (2, 64, 128, 2),
    ],
)
@pytest.mark.parametrize("with_weight", [True, False])
def test_distributed_rms_norm_bw_single_device(
    device, batch, seq_len, hidden_dim_total, num_simulated_devices, with_weight
):
    _run_distributed_rms_norm_bw(
        device, batch, seq_len, hidden_dim_total, num_simulated_devices, with_weight, ttnn.bfloat16
    )


def test_distributed_rms_norm_bw_float32(device):
    _run_distributed_rms_norm_bw(
        device, batch=1, seq_len=32, hidden_dim_total=128, num_devices=2, with_weight=True, ttnn_dtype=ttnn.float32
    )


def test_distributed_rms_norm_bw_rejects_mismatched_device_counts(device, expect_error):
    """stats and bw_stats scale by the device count read off their width, so a pair gathered over
    different device sets would rescale every gradient rather than fail."""
    num_devices = 4
    hidden_dim_total = 256
    compute_kernel_config = _compute_kernel_config()
    torch.manual_seed(0)

    torch_x = torch.randn((1, 1, 32, hidden_dim_total), dtype=torch.bfloat16)
    tt_x = _shard(device, torch_x, num_devices, ttnn.bfloat16)
    tt_dy = _shard(device, torch.randn_like(torch_x), num_devices, ttnn.bfloat16)

    per_shard_stats = [
        ttnn.rms_norm_pre_all_gather(t, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16) for t in tt_x
    ]
    tt_stats_gathered = ttnn.concat(per_shard_stats, dim=3)
    bw_stats_short = ttnn.concat(
        [
            ttnn.rms_norm_pre_all_gather_bw(
                tt_x[i], tt_dy[i], tt_stats_gathered, epsilon=EPS, compute_kernel_config=compute_kernel_config
            )
            for i in range(num_devices - 1)
        ],
        dim=3,
    )

    with expect_error(RuntimeError, "all-gathered over the same devices"):
        ttnn.rms_norm_post_all_gather_bw(
            tt_x[0],
            tt_dy[0],
            tt_stats_gathered,
            bw_stats_short,
            epsilon=EPS,
            compute_kernel_config=compute_kernel_config,
        )


def test_distributed_rms_norm_bw_rejects_unsharded_weight(device, expect_error):
    """weight must be sharded alongside the input; the full row would silently broadcast."""
    num_devices = 2
    hidden_dim_total = 128
    compute_kernel_config = _compute_kernel_config()
    torch.manual_seed(0)

    torch_x = torch.randn((1, 1, 32, hidden_dim_total), dtype=torch.bfloat16)
    tt_x = _shard(device, torch_x, num_devices, ttnn.bfloat16)
    tt_dy = _shard(device, torch.randn_like(torch_x), num_devices, ttnn.bfloat16)
    full_weight = ttnn.from_torch(
        torch.randn((1, 1, 1, hidden_dim_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_stats_gathered = ttnn.concat(
        [
            ttnn.rms_norm_pre_all_gather(t, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16)
            for t in tt_x
        ],
        dim=3,
    )

    with expect_error(RuntimeError, "weight last dim"):
        ttnn.rms_norm_pre_all_gather_bw(
            tt_x[0],
            tt_dy[0],
            tt_stats_gathered,
            epsilon=EPS,
            weight=full_weight,
            compute_kernel_config=compute_kernel_config,
        )
