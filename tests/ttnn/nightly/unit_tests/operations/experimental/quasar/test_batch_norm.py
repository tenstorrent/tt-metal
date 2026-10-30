# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

QUASAR_ONLY = "Quasar-native kernels (DFB implicit sync, multi-hart DM / NEO threading)"


def _run_batch_norm(device, shape, affine, eps=1e-5):
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
    # PCC is blind to a uniform scale (e.g. a wrong eps fold), so also bound the error.
    torch.testing.assert_close(out, ref, rtol=0.02, atol=0.02)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 32, 64),  # 4 planes of 2 tiles: one round, batch 2
        (2, 3, 64, 128),  # 6 planes of 8 tiles: two rounds on a small grid, duplicate planes
        (1, 2, 32, 32),  # 2 planes of 1 tile: batch 1
        (1, 1, 96, 96),  # 1 plane of 9 tiles: batch 3
    ],
)
@pytest.mark.parametrize("affine", ["none", "weight", "weight_bias"])
def test_quasar_batch_norm(device, shape, affine):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip(QUASAR_ONLY)
    torch.manual_seed(0)
    _run_batch_norm(device, shape, affine)


# Second round per core on any grid: five planes per core leave the second round partially occupied with the
# default 4 NEOs, so the per-plane parameter fetch, the scale / shift ring reuse and the eps tile retention
# across rounds are all exercised (each plane has its own channel statistics).
@pytest.mark.parametrize("affine", ["none", "weight_bias"])
def test_quasar_batch_norm_multiple_rounds(device, affine):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip(QUASAR_ONLY)
    torch.manual_seed(0)
    grid = device.compute_with_storage_grid_size()
    _run_batch_norm(device, (1, 5 * grid.x * grid.y, 32, 64), affine)


# ---- contract rejections (host-side validation, run on every arch) ----


def test_quasar_batch_norm_rejects_empty_input(device, expect_error):
    to_dev = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device)
    x = to_dev(torch.zeros((0, 2, 32, 32), dtype=torch.bfloat16))
    mean = to_dev(torch.zeros((1, 2, 1, 1), dtype=torch.bfloat16))
    var = to_dev(torch.ones((1, 2, 1, 1), dtype=torch.bfloat16))
    with expect_error(RuntimeError, "does not support empty tensors"):
        ttnn.experimental.quasar.batch_norm(x, running_mean=mean, running_var=var)


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
        pytest.skip(QUASAR_ONLY)
    monkeypatch.setenv("TTNN_QSR_BATCH_NORM_THREADS", plan)
    torch.manual_seed(0)
    _run_batch_norm(device, (2, 3, 64, 128), "none")
