# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
ULP-based accuracy characterization for ttnn.softmax vs PyTorch (issue #33741).

BF16 golden: torch.ops.aten._softmax.default(x, dim, half_to_float=False)
  with BF16 input x.  Matches the reference used in test_softmax_accuracy and
  keeps the reduction in the same dtype family as the fused device path.

FP32 golden: the same ATen op with FP32 input x.
  FP32 tests use fp32_dest_acc_en=True only (required on BH for FP32 softmax).
  wide_uniform stress is BF16-only and restricted to softmax over H (dim=-2);
  wide_uniform over W is not covered here due to known mismatches on BH.

ULP is measured in the output dtype (BF16 or FP32).  Elements where |golden|
is very small relative to the tensor's dynamic range are excluded from ULP
(where the metric breaks down) and validated with a scaled absolute-tolerance
check instead.

Metrics are logged at INFO for every parametrized case (pass or fail).  To
print them in the terminal, run pytest with e.g.:

  pytest .../test_softmax_ulp.py --log-cli-level=INFO

or capture to a file:

  pytest .../test_softmax_ulp.py --log-file=softmax_ulp.log --log-file-level=INFO
"""

import logging

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn
from tests.ttnn.unit_tests.operations.test_utils import get_compute_kernel_options
from tests.ttnn.utils_for_testing import measure_ulp_with_near_zero_atol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_softmax_compute_kernel_config(device, fp32_dest_acc_en: bool):
    """
    Prefer arch-aware kernel config (matches test_softmax_accuracy).  Some
    bindings or architectures may not accept all kwargs; fall back to
    Wormhole-style config like test_utils.get_compute_kernel_options.
    """
    try:
        return ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=True,
        )
    except TypeError:
        return get_compute_kernel_options(fp32_dest_acc_en)


def _golden_softmax_bf16(x_bf16: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.ops.aten._softmax.default(x_bf16, dim, half_to_float=False)


def _golden_softmax_fp32(x_fp32: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.ops.aten._softmax.default(x_fp32, dim, half_to_float=False)


def _run_ttnn_softmax(
    input_torch: torch.Tensor,
    ttnn_dtype,
    device,
    dim: int,
    compute_kernel_config,
    numeric_stable: bool,
) -> torch.Tensor:
    tt_input = ttnn.from_torch(input_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(tt_input, 42)
    tt_out = ttnn.softmax(
        tt_input,
        dim=dim,
        compute_kernel_config=compute_kernel_config,
        numeric_stable=numeric_stable,
    )
    return ttnn.to_torch(tt_out)


# ---------------------------------------------------------------------------
# Test parameters (sweeps; adjust ranges/steps for coverage vs runtime)
# ---------------------------------------------------------------------------

_SOFTMAX_W_SIZES = sorted(set(range(256, 8193, 1024)) | {128, 256, 512, 1024, 2048, 4096, 8192, 32768})
_SOFTMAX_H_SIZES = sorted(set(range(128, 4097, 1024)) | {128, 512, 1024, 2048, 4096})
_SOFTMAX_HW_SQUARES = list(range(64, 513, 64))
_SOFTMAX_HW_MIXED = [(96, 160), (128, 192), (160, 96)]
_SOFTMAX_W_FIXED = 64  # fixed non-softmax dim for H-reduction cases
_SOFTMAX_H_FIXED = 32  # fixed non-softmax dim for W-reduction cases
_SOFTMAX_BATCH_W = [2, 4, 8]


def _build_softmax_shapes_and_dims():
    out = []
    for w in _SOFTMAX_W_SIZES:
        out.append(((1, 1, _SOFTMAX_H_FIXED, w), -1, f"W-{w}"))
    for h in _SOFTMAX_H_SIZES:
        out.append(((1, 1, h, _SOFTMAX_W_FIXED), -2, f"H-{h}"))
    for side in _SOFTMAX_HW_SQUARES:
        out.append(((1, 1, side, side), -1, f"HWsq-W-{side}"))
        out.append(((1, 1, side, side), -2, f"HWsq-H-{side}"))
    for hh, ww in _SOFTMAX_HW_MIXED:
        out.append(((1, 1, hh, ww), -1, f"W-{hh}x{ww}"))
        out.append(((1, 1, hh, ww), -2, f"H-{hh}x{ww}"))
    out.extend(
        [
            ((1, 1, 37, 41), -1, "W-odd-41"),
            ((1, 1, 37, 41), -2, "H-odd-37"),
            ((1, 1, 2048, 64), -2, "H-2048-W64"),
        ]
    )
    for b in _SOFTMAX_BATCH_W:
        out.append(((b, 1, 64, 128), -1, f"W-batch{b}"))
    return out


_SHAPES_AND_DIMS = _build_softmax_shapes_and_dims()

# wide_uniform [-1e3,1e3] over softmax-W currently mis-matches fused softmax on BH
# (zeros / bogus mass on long W); keep stress coverage only for softmax over H.
_SHAPES_AND_DIMS_H_SOFTMAX_ONLY = [s for s in _SHAPES_AND_DIMS if s[1] == -2]


# ---------------------------------------------------------------------------
# BF16 tests
# ---------------------------------------------------------------------------

# BF16: BH runs stayed at or below 10 ULP for normal inputs; keep small margin over
# test_softmax_accuracy-style worst case (~13).
_BF16_ULP_THRESHOLD = 12
_BF16_NEAR_ZERO_ATOL_FRACTION = 0.02


@pytest.mark.parametrize(
    "shape, dim, desc",
    _SHAPES_AND_DIMS,
    ids=[c[2] for c in _SHAPES_AND_DIMS],
)
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32_acc_off", "fp32_acc_on"])
@pytest.mark.parametrize("numeric_stable", [False, True], ids=["numstab_off", "numstab_on"])
def test_softmax_ulp_bf16_normal(device, shape, dim, desc, fp32_dest_acc_en, numeric_stable):
    """BF16 softmax ULP vs torch golden; standard normal input."""
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)

    golden = _golden_softmax_bf16(x, dim)
    compute_kernel_config = _make_softmax_compute_kernel_config(device, fp32_dest_acc_en)
    actual = _run_ttnn_softmax(x, ttnn.bfloat16, device, dim, compute_kernel_config, numeric_stable)

    passed, max_ulp, max_atol_err, msg = measure_ulp_with_near_zero_atol(
        golden, actual, _BF16_ULP_THRESHOLD, _BF16_NEAR_ZERO_ATOL_FRACTION
    )
    logger.info(
        "ttnn.softmax ULP dtype=BF16 desc=%r distribution=normal shape=%s dim=%s "
        "fp32_dest_acc_en=%s numeric_stable=%s max_ulp=%s ulp_threshold=%s "
        "max_atol_err=%s passed=%s | %s",
        desc,
        shape,
        dim,
        fp32_dest_acc_en,
        numeric_stable,
        max_ulp,
        _BF16_ULP_THRESHOLD,
        max_atol_err,
        passed,
        msg,
    )
    assert passed, f"[BF16 {desc} normal fp32_acc={fp32_dest_acc_en} numstab={numeric_stable}] {msg}"


@pytest.mark.parametrize(
    "shape, dim, desc",
    _SHAPES_AND_DIMS_H_SOFTMAX_ONLY,
    ids=[c[2] for c in _SHAPES_AND_DIMS_H_SOFTMAX_ONLY],
)
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32_acc_off", "fp32_acc_on"])
@pytest.mark.parametrize("numeric_stable", [False, True], ids=["numstab_off", "numstab_on"])
def test_softmax_ulp_bf16_wide_uniform_h(device, shape, dim, desc, fp32_dest_acc_en, numeric_stable):
    """BF16 softmax over H with wide uniform logits (supported path on BH)."""
    torch.manual_seed(42)
    x = torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3).to(torch.bfloat16)

    golden = _golden_softmax_bf16(x, dim)
    compute_kernel_config = _make_softmax_compute_kernel_config(device, fp32_dest_acc_en)
    actual = _run_ttnn_softmax(x, ttnn.bfloat16, device, dim, compute_kernel_config, numeric_stable)

    passed, max_ulp, max_atol_err, msg = measure_ulp_with_near_zero_atol(
        golden, actual, _BF16_ULP_THRESHOLD, _BF16_NEAR_ZERO_ATOL_FRACTION
    )
    logger.info(
        "ttnn.softmax ULP dtype=BF16 desc=%r distribution=wide_uniform shape=%s dim=%s "
        "fp32_dest_acc_en=%s numeric_stable=%s max_ulp=%s ulp_threshold=%s "
        "max_atol_err=%s passed=%s | %s",
        desc,
        shape,
        dim,
        fp32_dest_acc_en,
        numeric_stable,
        max_ulp,
        _BF16_ULP_THRESHOLD,
        max_atol_err,
        passed,
        msg,
    )
    assert passed, f"[BF16 {desc} wide_uniform fp32_acc={fp32_dest_acc_en} numstab={numeric_stable}] {msg}"


# ---------------------------------------------------------------------------
# FP32 tests
# ---------------------------------------------------------------------------

# FP32: fp32_dest_acc_en=True only; normal inputs. BH max ~9.3e4 (large W, numstab on).
_FP32_ULP_THRESHOLD = 200_000
_FP32_NEAR_ZERO_ATOL_FRACTION = 0.005


@pytest.mark.parametrize(
    "shape, dim, desc",
    _SHAPES_AND_DIMS,
    ids=[c[2] for c in _SHAPES_AND_DIMS],
)
@pytest.mark.parametrize("numeric_stable", [False, True], ids=["numstab_off", "numstab_on"])
def test_softmax_ulp_fp32_normal_fp32_acc_on(device, shape, dim, desc, numeric_stable):
    """FP32 softmax ULP vs torch golden; normal input; fp32_dest_acc_en=True only."""
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32)

    golden = _golden_softmax_fp32(x, dim)
    compute_kernel_config = _make_softmax_compute_kernel_config(device, fp32_dest_acc_en=True)
    actual = _run_ttnn_softmax(x, ttnn.float32, device, dim, compute_kernel_config, numeric_stable)

    passed, max_ulp, max_atol_err, msg = measure_ulp_with_near_zero_atol(
        golden, actual, _FP32_ULP_THRESHOLD, _FP32_NEAR_ZERO_ATOL_FRACTION
    )
    logger.info(
        "ttnn.softmax ULP dtype=FP32 desc=%r distribution=normal shape=%s dim=%s "
        "fp32_dest_acc_en=True numeric_stable=%s max_ulp=%s ulp_threshold=%s "
        "max_atol_err=%s passed=%s | %s",
        desc,
        shape,
        dim,
        numeric_stable,
        max_ulp,
        _FP32_ULP_THRESHOLD,
        max_atol_err,
        passed,
        msg,
    )
    assert passed, f"[FP32 {desc} normal fp32_acc=True numstab={numeric_stable}] {msg}"
