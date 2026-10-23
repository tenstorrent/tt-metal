# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

QUASAR_ONLY = "Quasar-native kernels (DFB implicit sync, multi-hart DM / NEO threading)"


# bf16 output against an fp32 reference: one bf16 ulp of the largest outputs (|out| < 8) is 0.03. With 16-bit DST
# the x - mean intermediate is also bf16 and gets scaled by gamma, so those cases get a wider absolute band.
BF16_ATOL = 0.03
BF16_DEST_ATOL = 0.05


def _check(ref, out, atol=BF16_ATOL):
    assert_with_pcc(ref, out, 0.999)
    # PCC is blind to a uniform scale (e.g. a wrong variance divisor), so also bound the error.
    torch.testing.assert_close(out, ref, rtol=0.02, atol=atol)


def _run_layer_norm(device, x, affine, compute_kernel_config=None, eps=1e-5):
    W = x.shape[-1]
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

    out = ttnn.experimental.quasar.layer_norm(
        tx, epsilon=eps, weight=tg, bias=tb, compute_kernel_config=compute_kernel_config
    )
    return ref, ttnn.to_torch(out).float()


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 64),  # Wt=2 < window: no reload
        (1, 1, 32, 256),  # Wt=8: batch 4, reload 4
        (1, 1, 96, 1024),  # 3 rows: uneven across 4 NEOs
        (2, 3, 64, 2048),  # many rows, Wt=64
        (1, 1, 32, 100),  # ragged W
        (1, 1, 32, 96),  # Wt=3: window 6, block 3 (odd tile count)
        (1, 1, 32, 192),  # Wt=6: batch 3 (not a multiple of 4), block 3
        (1, 1, 32, 160),  # Wt=5 (prime): batch 1, window 2, block 1 -- the tightest ring
        (1, 1, 40, 64),  # ragged H: the padded rows are normalised and discarded
        (100,),  # rank 1: implicit tile-height dim ([100] pads to [32, 128])
    ],
)
@pytest.mark.parametrize("affine", ["none", "gamma", "gamma_beta"])
def test_quasar_layer_norm(device, shape, affine):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip(QUASAR_ONLY)
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    _check(*_run_layer_norm(device, x, affine))


# Second round per core: with the default 4 NEOs, five tile rows per core leave the second round partially
# occupied (3 of 4 NEOs recompute NEO 0's row), which exercises the lock-step duplicate path and the reuse
# of the per-NEO stat rings / eps tile across rounds.
@pytest.mark.parametrize("affine", ["none", "gamma_beta"])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False])
def test_quasar_layer_norm_multiple_rounds(device, affine, fp32_dest_acc_en):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip(QUASAR_ONLY)
    if fp32_dest_acc_en and os.environ.get("TT_METAL_SIMULATOR"):
        # craq-sim derives the MOVD2B source format from the SrcA bank's last unpack instead of the explicit ALU
        # config, so the second round's 32-bit transpose_dest (SrcA bank tagged Tf32 by pass 2) aborts with
        # "UnimplementedFunctionality: tensix_execute_movd2b: tf32 use_dst32b=1 dest_32b_lo=1". Passes on the
        # emulator (emu-quasar-2x3, 2026-10-23: PCC 1.0 / 0.999996).
        pytest.skip("craq-sim does not model the 32-bit transpose_dest of a second NEO round")
    torch.manual_seed(0)
    grid = device.compute_with_storage_grid_size()
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32_dest_acc_en
    )
    x = torch.randn((1, 1, 32 * 5 * grid.x * grid.y, 96), dtype=torch.bfloat16)
    _check(
        *_run_layer_norm(device, x, affine, compute_kernel_config=cfg),
        atol=BF16_ATOL if fp32_dest_acc_en else BF16_DEST_ATOL,
    )


# Inputs away from zero: the Welford mean must cancel the offset before the variance is formed, which a
# zero-mean sweep cannot tell from a plain sum of squares. The offset is moderate on purpose: the FPU reads
# the fp32 mean as TF32 (10-bit mantissa) in the x - mean broadcast subtraction, so the absolute error grows
# as |mean| * 2^-11 * rstd (measured on craq-sim: offset 50 / std 0.5 -> 0.07, matching the prediction).
@pytest.mark.parametrize("shape", [(1, 1, 32, 100), (1, 1, 40, 64)])
@pytest.mark.parametrize("affine", ["none", "gamma_beta"])
def test_quasar_layer_norm_offset_input(device, shape, affine):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip(QUASAR_ONLY)
    torch.manual_seed(0)
    x = (torch.randn(shape) * 0.5 + 5).to(torch.bfloat16)
    _check(*_run_layer_norm(device, x, affine))


# fp32_dest_acc_en=False: bf16 stat rings, 16-bit DST (block of up to 4 tiles), 16-bit transpose_dest.
@pytest.mark.parametrize("shape", [(1, 1, 32, 256), (1, 1, 32, 96), (1, 1, 32, 100)])
@pytest.mark.parametrize("affine", ["none", "gamma_beta"])
def test_quasar_layer_norm_bf16_dest(device, shape, affine):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip(QUASAR_ONLY)
    torch.manual_seed(0)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False
    )
    x = torch.randn(shape, dtype=torch.bfloat16)
    _check(*_run_layer_norm(device, x, affine, compute_kernel_config=cfg), atol=BF16_DEST_ATOL)


def test_quasar_layer_norm_program_cache_reuse(device):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip(QUASAR_ONLY)
    torch.manual_seed(0)
    for _ in range(2):
        x = torch.randn((1, 1, 64, 256), dtype=torch.bfloat16)
        _check(*_run_layer_norm(device, x, "gamma_beta"))


# ---- contract rejections (host-side validation, run on every arch) ----


def test_quasar_layer_norm_rejects_empty_input(device, expect_error):
    x = ttnn.from_torch(torch.zeros((1, 1, 32, 0), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "does not support empty tensors"):
        ttnn.experimental.quasar.layer_norm(x)


def test_quasar_layer_norm_rejects_residual(device, expect_error):
    x = ttnn.from_torch(torch.randn((1, 1, 32, 64), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "does not support residual_input_tensor"):
        ttnn.experimental.quasar.layer_norm(x, residual_input_tensor=x)


def test_quasar_layer_norm_rejects_recip_tensor(device, expect_error):
    x = ttnn.from_torch(torch.randn((1, 1, 32, 64), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "does not support recip_tensor"):
        ttnn.experimental.quasar.layer_norm(x, recip_tensor=x)


def test_quasar_layer_norm_rejects_fp32_input(device, expect_error):
    x = ttnn.from_torch(torch.randn((1, 1, 32, 64), dtype=torch.float32), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "input must be BFLOAT16"):
        ttnn.experimental.quasar.layer_norm(x)


def test_quasar_layer_norm_rejects_non_bf16_affine(device, expect_error):
    x = ttnn.from_torch(torch.randn((1, 1, 32, 64), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(torch.ones((1, 1, 1, 64), dtype=torch.int32), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "gamma/beta must be BFLOAT16"):
        ttnn.experimental.quasar.layer_norm(x, weight=g)


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
        pytest.skip(QUASAR_ONLY)
    monkeypatch.setenv("TTNN_QSR_LAYERNORM_THREADS", plan)
    torch.manual_seed(0)
    x = torch.randn((1, 1, 96, 256), dtype=torch.bfloat16)
    _check(*_run_layer_norm(device, x, "none"))
