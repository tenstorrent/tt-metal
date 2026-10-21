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
        (1, 1, 32, 64),  # Wt=2 < window: no reload
        (1, 1, 32, 256),  # Wt=8: tail=4, reload 4
        (1, 1, 96, 1024),  # 3 rows: uneven across 4 NEOs
        (2, 3, 64, 2048),  # many rows, Wt=64
        (1, 1, 32, 100),  # ragged W
        (1, 1, 32, 96),  # Wt=3: window 6, block 1 (odd tile count)
    ],
)
@pytest.mark.parametrize("affine", ["none", "gamma", "gamma_beta"])
def test_quasar_layer_norm(device, shape, affine):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip("Quasar-native kernels (DFB implicit sync, multi-hart DM / NEO threading)")
    torch.manual_seed(0)
    eps = 1e-5
    x = torch.randn(shape, dtype=torch.bfloat16)
    W = shape[-1]
    gamma = torch.randn((1, 1, 1, W), dtype=torch.bfloat16) if affine != "none" else None
    beta = torch.randn((1, 1, 1, W), dtype=torch.bfloat16) if affine == "gamma_beta" else None

    ref = torch.nn.functional.layer_norm(
        x.float(),
        (W,),
        weight=gamma.float().reshape(W) if gamma is not None else None,
        bias=beta.float().reshape(W) if beta is not None else None,
        eps=eps,
    )

    tx = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    tg = ttnn.from_torch(gamma, layout=ttnn.TILE_LAYOUT, device=device) if gamma is not None else None
    tb = ttnn.from_torch(beta, layout=ttnn.TILE_LAYOUT, device=device) if beta is not None else None

    out = ttnn.experimental.quasar.layer_norm(tx, epsilon=eps, weight=tg, bias=tb)
    out = ttnn.to_torch(out).float()
    assert_with_pcc(ref, out, 0.999)


# ---- contract rejections (host-side validation, run on every arch) ----


def test_quasar_layer_norm_rejects_residual(device, expect_error):
    x = ttnn.from_torch(torch.randn((1, 1, 32, 64), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "does not support residual_input_tensor"):
        ttnn.experimental.quasar.layer_norm(x, residual_input_tensor=x)


def test_quasar_layer_norm_rejects_fp32_input(device, expect_error):
    x = ttnn.from_torch(torch.randn((1, 1, 32, 64), dtype=torch.float32), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "input must be BFLOAT16"):
        ttnn.experimental.quasar.layer_norm(x)


def test_quasar_layer_norm_rejects_sharded_program_config(device, expect_error):
    x = ttnn.from_torch(torch.randn((1, 1, 32, 64), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "only implements the Welford path"):
        ttnn.experimental.quasar.layer_norm(x, program_config=ttnn.LayerNormDefaultProgramConfig(use_welford=False))


def test_quasar_layer_norm_rejects_narrow_affine(device, expect_error):
    # Both pad to 128 columns, but gamma covers only 97 of the 100 valid input columns.
    x = ttnn.from_torch(torch.randn((1, 1, 32, 100), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(torch.randn((1, 1, 1, 97), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "must cover the input logical width"):
        ttnn.experimental.quasar.layer_norm(x, weight=g)


def test_quasar_layer_norm_rejects_bad_thread_plan(device, expect_error, monkeypatch):
    monkeypatch.setenv("TTNN_QSR_LAYERNORM_THREADS", "2,4,3")
    x = ttnn.from_torch(torch.randn((1, 1, 32, 64), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "must be a multiple of writer threads"):
        ttnn.experimental.quasar.layer_norm(x)


@pytest.mark.parametrize("plan", ["1,4,1", "2,4,4", "1,1,1"])
def test_quasar_layer_norm_thread_plan_override(device, monkeypatch, plan):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip("Quasar-native kernels (DFB implicit sync, multi-hart DM / NEO threading)")
    monkeypatch.setenv("TTNN_QSR_LAYERNORM_THREADS", plan)
    torch.manual_seed(0)
    x = torch.randn((1, 1, 96, 256), dtype=torch.bfloat16)
    ref = torch.nn.functional.layer_norm(x.float(), (256,), eps=1e-5)
    tx = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(ttnn.experimental.quasar.layer_norm(tx, epsilon=1e-5)).float()
    assert_with_pcc(ref, out, 0.999)
