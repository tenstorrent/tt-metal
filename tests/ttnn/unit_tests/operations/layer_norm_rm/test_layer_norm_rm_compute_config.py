# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Compute config tests for layer_norm_rm.

Tests the effect of fp32_dest_acc_en on correctness and precision.
When fp32_dest_acc_en is active (DST_ACCUM_MODE == true), fast_tilize can
corrupt data by writing bfloat16 bits into fp32-configured dest slots.
These tests help verify whether the guard in can_use_fast_tilize() is working.
"""

import pytest
import torch
import torch.nn.functional as F
import ttnn

from ttnn.operations.layer_norm_rm import layer_norm_rm


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


def torch_layer_norm_rm(x, gamma=None, beta=None, epsilon=1e-5):
    """PyTorch reference computed in float32."""
    W = x.shape[-1]
    x_f32 = x.float()
    g = gamma.float() if gamma is not None else None
    b = beta.float() if beta is not None else None
    return F.layer_norm(x_f32, [W], weight=g, bias=b, eps=epsilon).to(x.dtype)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def to_device(tensor_torch, device):
    return ttnn.from_torch(
        tensor_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )


def compute_pcc(actual, expected):
    """Compute Pearson correlation coefficient."""
    a = actual.float().flatten()
    e = expected.float().flatten()
    a_c = a - a.mean()
    e_c = e - e.mean()
    num = (a_c * e_c).sum()
    den = a_c.norm() * e_c.norm()
    return (num / den).item() if den > 1e-30 else 1.0


# ---------------------------------------------------------------------------
# Test shapes
# ---------------------------------------------------------------------------

SHAPES = [
    pytest.param((1, 1, 32, 32), id="32x32"),
    pytest.param((1, 1, 64, 128), id="64x128"),
    pytest.param((1, 1, 32, 256), id="32x256"),
    pytest.param((2, 1, 64, 64), id="batch_64x64"),
]


# ---------------------------------------------------------------------------
# Test: fp32_dest_acc_en=False (default behavior, should match baseline)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_fp32_dest_off_pure(shape, device):
    """fp32_dest_acc_en=False with pure normalization — should match baseline."""
    torch.manual_seed(42)
    x_torch = torch.randn(shape, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x_torch)

    x_tt = to_device(x_torch, device)
    result_tt = layer_norm_rm(
        x_tt,
        compute_kernel_config={"fp32_dest_acc_en": False},
    )
    result_torch = ttnn.to_torch(result_tt)

    pcc = compute_pcc(result_torch, expected)
    max_abs_err = (result_torch.float() - expected.float()).abs().max().item()

    print(f"\n[fp32_dest=OFF] shape={shape} PCC={pcc:.6f} max_abs_err={max_abs_err:.6f}")
    assert pcc >= 0.999, f"PCC too low: {pcc:.6f}"


@pytest.mark.parametrize("shape", SHAPES)
def test_fp32_dest_off_gamma_beta(shape, device):
    """fp32_dest_acc_en=False with gamma+beta — should match baseline."""
    torch.manual_seed(42)
    W = shape[-1]
    x_torch = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.randn(W, dtype=torch.bfloat16)
    beta_torch = torch.randn(W, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x_torch, gamma=gamma_torch, beta=beta_torch)

    x_tt = to_device(x_torch, device)
    gamma_tt = to_device(gamma_torch.reshape(1, 1, 1, W), device)
    beta_tt = to_device(beta_torch.reshape(1, 1, 1, W), device)
    result_tt = layer_norm_rm(
        x_tt,
        gamma_tt,
        beta_tt,
        compute_kernel_config={"fp32_dest_acc_en": False},
    )
    result_torch = ttnn.to_torch(result_tt)

    pcc = compute_pcc(result_torch, expected)
    max_abs_err = (result_torch.float() - expected.float()).abs().max().item()

    print(f"\n[fp32_dest=OFF gamma+beta] shape={shape} PCC={pcc:.6f} max_abs_err={max_abs_err:.6f}")
    assert pcc >= 0.999, f"PCC too low: {pcc:.6f}"


# ---------------------------------------------------------------------------
# Test: fp32_dest_acc_en=True (triggers DST_ACCUM_MODE, may hit fast_tilize bug)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_fp32_dest_on_pure(shape, device):
    """fp32_dest_acc_en=True with pure normalization.

    When can_use_fast_tilize() doesn't guard against DST_ACCUM_MODE,
    fast_tilize writes bfloat16 bits into fp32-configured dest slots,
    causing data corruption and PCC collapse.
    """
    torch.manual_seed(42)
    x_torch = torch.randn(shape, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x_torch)

    x_tt = to_device(x_torch, device)
    result_tt = layer_norm_rm(
        x_tt,
        compute_kernel_config={"fp32_dest_acc_en": True},
    )
    result_torch = ttnn.to_torch(result_tt)

    pcc = compute_pcc(result_torch, expected)
    max_abs_err = (result_torch.float() - expected.float()).abs().max().item()

    print(f"\n[fp32_dest=ON] shape={shape} PCC={pcc:.6f} max_abs_err={max_abs_err:.6f}")
    assert pcc >= 0.999, f"PCC too low: {pcc:.6f}"


@pytest.mark.parametrize("shape", SHAPES)
def test_fp32_dest_on_gamma_beta(shape, device):
    """fp32_dest_acc_en=True with gamma+beta.

    Exercises the full affine path under fp32 dest accumulation.
    fast_tilize corruption affects both gamma/beta tilization and
    the main input tilization.
    """
    torch.manual_seed(42)
    W = shape[-1]
    x_torch = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.randn(W, dtype=torch.bfloat16)
    beta_torch = torch.randn(W, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x_torch, gamma=gamma_torch, beta=beta_torch)

    x_tt = to_device(x_torch, device)
    gamma_tt = to_device(gamma_torch.reshape(1, 1, 1, W), device)
    beta_tt = to_device(beta_torch.reshape(1, 1, 1, W), device)
    result_tt = layer_norm_rm(
        x_tt,
        gamma_tt,
        beta_tt,
        compute_kernel_config={"fp32_dest_acc_en": True},
    )
    result_torch = ttnn.to_torch(result_tt)

    pcc = compute_pcc(result_torch, expected)
    max_abs_err = (result_torch.float() - expected.float()).abs().max().item()

    print(f"\n[fp32_dest=ON gamma+beta] shape={shape} PCC={pcc:.6f} max_abs_err={max_abs_err:.6f}")
    assert pcc >= 0.999, f"PCC too low: {pcc:.6f}"


# ---------------------------------------------------------------------------
# Test: Direct comparison — fp32 ON vs OFF on same input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_fp32_dest_on_vs_off(shape, device):
    """Compare fp32_dest ON vs OFF on the same input.

    If fast_tilize corruption is present, the ON result will diverge
    significantly from the OFF result. This test prints both PCCs
    side-by-side for easy comparison.
    """
    torch.manual_seed(42)
    W = shape[-1]
    x_torch = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.randn(W, dtype=torch.bfloat16)
    beta_torch = torch.randn(W, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x_torch, gamma=gamma_torch, beta=beta_torch)

    x_tt = to_device(x_torch, device)
    gamma_tt = to_device(gamma_torch.reshape(1, 1, 1, W), device)
    beta_tt = to_device(beta_torch.reshape(1, 1, 1, W), device)

    # Run with fp32_dest OFF
    result_off = ttnn.to_torch(
        layer_norm_rm(
            x_tt,
            gamma_tt,
            beta_tt,
            compute_kernel_config={"fp32_dest_acc_en": False},
        )
    )

    # Re-create device tensors (consumed by generic_op)
    x_tt = to_device(x_torch, device)
    gamma_tt = to_device(gamma_torch.reshape(1, 1, 1, W), device)
    beta_tt = to_device(beta_torch.reshape(1, 1, 1, W), device)

    # Run with fp32_dest ON
    result_on = ttnn.to_torch(
        layer_norm_rm(
            x_tt,
            gamma_tt,
            beta_tt,
            compute_kernel_config={"fp32_dest_acc_en": True},
        )
    )

    pcc_off = compute_pcc(result_off, expected)
    pcc_on = compute_pcc(result_on, expected)
    pcc_on_vs_off = compute_pcc(result_on, result_off)

    max_diff = (result_on.float() - result_off.float()).abs().max().item()

    print(f"\n[ON vs OFF] shape={shape}")
    print(f"  PCC(OFF vs ref)  = {pcc_off:.6f}")
    print(f"  PCC(ON  vs ref)  = {pcc_on:.6f}")
    print(f"  PCC(ON  vs OFF)  = {pcc_on_vs_off:.6f}")
    print(f"  max_diff(ON-OFF) = {max_diff:.6f}")

    # Both should be good
    assert pcc_off >= 0.999, f"fp32_dest=OFF PCC too low: {pcc_off:.6f}"
    assert pcc_on >= 0.999, f"fp32_dest=ON PCC too low: {pcc_on:.6f}"


# ---------------------------------------------------------------------------
# Test: math_fidelity options with fp32_dest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param(ttnn.MathFidelity.HiFi4, id="HiFi4"),
        pytest.param(ttnn.MathFidelity.HiFi2, id="HiFi2"),
        pytest.param(ttnn.MathFidelity.LoFi, id="LoFi"),
    ],
)
def test_math_fidelity_with_fp32_dest(math_fidelity, device):
    """Test math fidelity combined with fp32_dest_acc_en."""
    torch.manual_seed(42)
    shape = (1, 1, 64, 128)
    W = shape[-1]
    x_torch = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.randn(W, dtype=torch.bfloat16)
    beta_torch = torch.randn(W, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x_torch, gamma=gamma_torch, beta=beta_torch)

    x_tt = to_device(x_torch, device)
    gamma_tt = to_device(gamma_torch.reshape(1, 1, 1, W), device)
    beta_tt = to_device(beta_torch.reshape(1, 1, 1, W), device)
    result_tt = layer_norm_rm(
        x_tt,
        gamma_tt,
        beta_tt,
        compute_kernel_config={
            "fp32_dest_acc_en": True,
            "math_fidelity": math_fidelity,
        },
    )
    result_torch = ttnn.to_torch(result_tt)

    pcc = compute_pcc(result_torch, expected)
    max_abs_err = (result_torch.float() - expected.float()).abs().max().item()

    print(f"\n[fp32+{math_fidelity}] PCC={pcc:.6f} max_abs_err={max_abs_err:.6f}")

    # LoFi is expected to be less precise but still reasonable
    min_pcc = 0.99 if math_fidelity == ttnn.MathFidelity.LoFi else 0.999
    assert pcc >= min_pcc, f"PCC too low: {pcc:.6f} for {math_fidelity}"


# ---------------------------------------------------------------------------
# Test: Wide shapes — detailed absolute difference comparison ON vs OFF
# ---------------------------------------------------------------------------

WIDE_SHAPES = [
    pytest.param((1, 1, 32, 512), id="32x512"),
    pytest.param((1, 1, 32, 1024), id="32x1024"),
    pytest.param((1, 1, 64, 1024), id="64x1024"),
    pytest.param((1, 1, 32, 2048), id="32x2048"),
    pytest.param((2, 1, 64, 1024), id="batch_64x1024"),
]


@pytest.mark.parametrize("shape", WIDE_SHAPES)
def test_fp32_dest_wide_abs_diff(shape, device):
    """Wide shapes: element-wise absolute difference between fp32_dest ON vs OFF.

    Prints per-element stats so you can see exactly how much the fast_tilize
    corruption distorts values across a wide row.
    """
    torch.manual_seed(42)
    W = shape[-1]
    x_torch = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.randn(W, dtype=torch.bfloat16)
    beta_torch = torch.randn(W, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x_torch, gamma=gamma_torch, beta=beta_torch)

    # --- fp32_dest OFF ---
    x_tt = to_device(x_torch, device)
    gamma_tt = to_device(gamma_torch.reshape(1, 1, 1, W), device)
    beta_tt = to_device(beta_torch.reshape(1, 1, 1, W), device)
    result_off = ttnn.to_torch(
        layer_norm_rm(
            x_tt,
            gamma_tt,
            beta_tt,
            compute_kernel_config={"fp32_dest_acc_en": False},
        )
    )

    # --- fp32_dest ON ---
    x_tt = to_device(x_torch, device)
    gamma_tt = to_device(gamma_torch.reshape(1, 1, 1, W), device)
    beta_tt = to_device(beta_torch.reshape(1, 1, 1, W), device)
    result_on = ttnn.to_torch(
        layer_norm_rm(
            x_tt,
            gamma_tt,
            beta_tt,
            compute_kernel_config={"fp32_dest_acc_en": True},
        )
    )

    # --- Absolute differences ---
    diff_off_ref = (result_off.float() - expected.float()).abs()
    diff_on_ref = (result_on.float() - expected.float()).abs()
    diff_on_off = (result_on.float() - result_off.float()).abs()

    pcc_off = compute_pcc(result_off, expected)
    pcc_on = compute_pcc(result_on, expected)

    # Count how many elements differ between ON and OFF
    nonzero_diffs = (diff_on_off > 0).sum().item()
    total_elems = diff_on_off.numel()

    print(f"\n{'='*60}")
    print(f"[WIDE ABS DIFF] shape={shape}  (W={W})")
    print(f"{'='*60}")
    print(f"  PCC(OFF vs ref) = {pcc_off:.6f}")
    print(f"  PCC(ON  vs ref) = {pcc_on:.6f}")
    print(f"  ---")
    print(
        f"  OFF vs ref:  max={diff_off_ref.max().item():.6f}  mean={diff_off_ref.mean().item():.6f}  median={diff_off_ref.median().item():.6f}"
    )
    print(
        f"  ON  vs ref:  max={diff_on_ref.max().item():.6f}  mean={diff_on_ref.mean().item():.6f}  median={diff_on_ref.median().item():.6f}"
    )
    print(
        f"  ON  vs OFF:  max={diff_on_off.max().item():.6f}  mean={diff_on_off.mean().item():.6f}  median={diff_on_off.median().item():.6f}"
    )
    print(f"  Elements that differ (ON vs OFF): {nonzero_diffs}/{total_elems} ({100*nonzero_diffs/total_elems:.1f}%)")

    # Histogram of ON-vs-OFF absolute differences
    buckets = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, float("inf")]
    print(f"  Diff histogram (ON vs OFF):")
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        count = ((diff_on_off >= lo) & (diff_on_off < hi)).sum().item()
        pct = 100 * count / total_elems
        label = f"    [{lo:.2f}, {hi:.2f})" if hi != float("inf") else f"    [{lo:.2f}, inf)"
        print(f"{label}: {count:>8} ({pct:>5.1f}%)")

    assert pcc_on >= 0.999, f"fp32_dest=ON PCC too low: {pcc_on:.6f}"
