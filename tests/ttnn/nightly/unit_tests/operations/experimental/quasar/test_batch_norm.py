# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 32, 64),  # 4 planes of 2 tiles: one round, batch 2
        (2, 3, 64, 128),  # 6 planes of 8 tiles: two rounds, duplicate planes
        (1, 2, 32, 32),  # 2 planes of 1 tile: batch 1
        (1, 1, 96, 96),  # 1 plane of 9 tiles: batch 3
    ],
)
@pytest.mark.parametrize("affine", ["none", "weight", "weight_bias"])
def test_quasar_batch_norm(device, shape, affine):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip("Quasar-native kernels (DFB implicit sync, multi-hart DM / NEO threading)")
    torch.manual_seed(0)
    eps = 1e-5
    N, C, H, W = shape
    x = torch.randn(shape, dtype=torch.bfloat16)
    mean = torch.randn((1, C, 1, 1), dtype=torch.bfloat16)
    var = torch.rand((1, C, 1, 1), dtype=torch.bfloat16) + 0.5
    weight = torch.randn((1, C, 1, 1), dtype=torch.bfloat16) if affine != "none" else None
    bias = torch.randn((1, C, 1, 1), dtype=torch.bfloat16) if affine == "weight_bias" else None

    ref = torch.nn.functional.batch_norm(
        x.float(),
        mean.float().reshape(C),
        var.float().reshape(C),
        weight=weight.float().reshape(C) if weight is not None else None,
        bias=bias.float().reshape(C) if bias is not None else None,
        training=False,
        eps=eps,
    )

    to_dev = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device) if t is not None else None
    out = ttnn.experimental.quasar.batch_norm(
        to_dev(x), running_mean=to_dev(mean), running_var=to_dev(var), eps=eps, weight=to_dev(weight), bias=to_dev(bias)
    )
    out = ttnn.to_torch(out).float()
    assert_with_pcc(ref, out, 0.999)


# ---- contract rejections (host-side validation, run on every arch) ----


def _bn_operands(device, C=2, param_hw=(1, 1), dtype=torch.bfloat16):
    to_dev = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device)
    x = to_dev(torch.randn((1, C, 32, 32), dtype=dtype))
    mean = to_dev(torch.randn((1, C, *param_hw), dtype=dtype))
    var = to_dev(torch.rand((1, C, *param_hw), dtype=dtype) + 0.5)
    return x, mean, var


def test_quasar_batch_norm_rejects_training(device, expect_error):
    x, mean, var = _bn_operands(device)
    with expect_error(RuntimeError, "only implements inference"):
        ttnn.experimental.quasar.batch_norm(x, running_mean=mean, running_var=var, training=True)


def test_quasar_batch_norm_requires_running_stats(device, expect_error):
    x, mean, _ = _bn_operands(device)
    with expect_error(RuntimeError, "requires running_mean and running_var"):
        ttnn.experimental.quasar.batch_norm(x, running_mean=mean)


def test_quasar_batch_norm_rejects_non_scalar_params(device, expect_error):
    x, mean, var = _bn_operands(device, param_hw=(2, 1))
    with expect_error(RuntimeError, r"must be \[1, C, 1, 1\]"):
        ttnn.experimental.quasar.batch_norm(x, running_mean=mean, running_var=var)


def test_quasar_batch_norm_rejects_channel_mismatch(device, expect_error):
    x, _, _ = _bn_operands(device, C=2)
    _, mean, var = _bn_operands(device, C=3)
    with expect_error(RuntimeError, "must match the input's channel count"):
        ttnn.experimental.quasar.batch_norm(x, running_mean=mean, running_var=var)


def test_quasar_batch_norm_rejects_fp32_input(device, expect_error):
    x, mean, var = _bn_operands(device, dtype=torch.float32)
    with expect_error(RuntimeError, "input must be BFLOAT16"):
        ttnn.experimental.quasar.batch_norm(x, running_mean=mean, running_var=var)


def test_quasar_batch_norm_rejects_bad_thread_plan(device, expect_error, monkeypatch):
    monkeypatch.setenv("TTNN_QSR_BATCH_NORM_THREADS", "3,4,2")
    x, mean, var = _bn_operands(device)
    with expect_error(RuntimeError, "must be a multiple of reader threads"):
        ttnn.experimental.quasar.batch_norm(x, running_mean=mean, running_var=var)


@pytest.mark.parametrize("plan", ["1,4,1", "2,4,4", "1,1,1"])
def test_quasar_batch_norm_thread_plan_override(device, monkeypatch, plan):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip("Quasar-native kernels (DFB implicit sync, multi-hart DM / NEO threading)")
    monkeypatch.setenv("TTNN_QSR_BATCH_NORM_THREADS", plan)
    torch.manual_seed(0)
    C = 3
    x = torch.randn((2, C, 64, 128), dtype=torch.bfloat16)
    mean = torch.randn((1, C, 1, 1), dtype=torch.bfloat16)
    var = torch.rand((1, C, 1, 1), dtype=torch.bfloat16) + 0.5
    ref = torch.nn.functional.batch_norm(
        x.float(), mean.float().reshape(C), var.float().reshape(C), training=False, eps=1e-5
    )
    to_dev = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.experimental.quasar.batch_norm(to_dev(x), running_mean=to_dev(mean), running_var=to_dev(var), eps=1e-5)
    assert_with_pcc(ref, ttnn.to_torch(out).float(), 0.999)
