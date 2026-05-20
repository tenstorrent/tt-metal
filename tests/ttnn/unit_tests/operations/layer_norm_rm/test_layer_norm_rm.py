# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance test for layer_norm (layer_norm_rm operation).

This is the immutable acceptance test handed to the kernel implementer.
DO NOT MODIFY THIS FILE — it is the spec.

The implementer's `layer_norm` (imported from `ttnn.operations.layer_norm`)
must pass every parametrized cell that is reachable under their declared
SUPPORTED / EXCLUSIONS axes. Cells outside the implementer's SUPPORTED axes
remain in this file as a record of the eventual target universe; they will
appear as xfails/skips when run through the harness — that's fine.

PCC tolerances by dtype:
  float32   → 0.9999
  bfloat16  → 0.995
  bfloat8_b → 0.99

These match `eval/golden_tests/layer_norm_rm/helpers.py:87-91`.

Run from repo root with:
  scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/layer_norm_rm/test_layer_norm_rm.py
"""

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm import layer_norm  # type: ignore


# --- PyTorch reference (mirrors eval/golden_tests/layer_norm_rm/helpers.py:33-50) ----


def pytorch_layer_norm(
    input_tensor: torch.Tensor,
    gamma: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    *,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """Reference LayerNorm — computed in fp32, returned in the input dtype.

    Normalization is per-last-dim. gamma scales, beta shifts; either may be None.
    """
    original_dtype = input_tensor.dtype
    x = input_tensor.to(torch.float32)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (x - mean) / torch.sqrt(var + epsilon)
    if gamma is not None:
        normalized = normalized * gamma.to(torch.float32).reshape(-1)
    if beta is not None:
        normalized = normalized + beta.to(torch.float32).reshape(-1)
    return normalized.to(original_dtype)


# --- PCC tolerance (same as golden suite — do NOT tighten by "op complexity") -------

_PCC_BY_DTYPE = {
    ttnn.float32: 0.9999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}


def _pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Pearson correlation coefficient between two tensors, computed in fp64."""
    a = actual.detach().to(torch.float64).flatten()
    e = expected.detach().to(torch.float64).flatten()
    # Drop non-finite (shouldn't happen with finite inputs + finite gamma/beta).
    mask = torch.isfinite(a) & torch.isfinite(e)
    a, e = a[mask], e[mask]
    a_mean, e_mean = a.mean(), e.mean()
    a_c, e_c = a - a_mean, e - e_mean
    denom = torch.sqrt((a_c * a_c).sum() * (e_c * e_c).sum())
    if denom.item() == 0.0:
        # Both are constants (degenerate) — PCC undefined; fall back to allclose.
        return 1.0 if torch.allclose(a, e, atol=1e-3) else 0.0
    return (a_c * e_c).sum().item() / denom.item()


# --- ttnn dtype → torch dtype (bfloat8_b has no native torch dtype) -----------------

_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,  # no torch bf8b; reference in bf16
}


# --- Parametrizations ----------------------------------------------------------------

# Shapes are chosen to cover, for each rank in {2, 3, 4}:
#   - tile-aligned H and W
#   - W not aligned (last partial W-tile)
#   - H not aligned (last partial H-tile)
#   - large W (LLM hidden) and small W (sanity)
SHAPES = [
    # rank 4
    pytest.param((1, 1, 32, 64), id="4d_aligned_small"),
    pytest.param((1, 1, 64, 128), id="4d_aligned_mid"),
    pytest.param((1, 1, 32, 4096), id="4d_aligned_wide"),
    pytest.param((2, 1, 64, 256), id="4d_aligned_batched"),
    pytest.param((1, 1, 32, 47), id="4d_W_partial_47"),
    pytest.param((1, 1, 32, 100), id="4d_W_partial_100"),
    pytest.param((1, 1, 17, 64), id="4d_H_partial_17"),
    pytest.param((1, 1, 50, 47), id="4d_HW_partial"),
    # rank 3
    pytest.param((1, 32, 128), id="3d_aligned"),
    pytest.param((4, 128, 512), id="3d_aligned_batched"),
    pytest.param((1, 32, 50), id="3d_W_partial"),
    # rank 2
    pytest.param((32, 64), id="2d_aligned"),
    pytest.param((128, 512), id="2d_aligned_wider"),
    pytest.param((32, 17), id="2d_W_partial"),
    pytest.param((17, 64), id="2d_H_partial"),
]


# Dtype + layout combinations. bfloat8_b is TILE-only (bf8b is a block-quantized
# format with no row-major encoding — see feature_spec.py:54).
DTYPE_LAYOUT_COMBOS = [
    pytest.param(ttnn.bfloat16, ttnn.TILE_LAYOUT, id="bf16_tile"),
    pytest.param(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, id="bf16_rm"),
    pytest.param(ttnn.float32, ttnn.TILE_LAYOUT, id="fp32_tile"),
    pytest.param(ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, id="fp32_rm"),
    pytest.param(ttnn.bfloat8_b, ttnn.TILE_LAYOUT, id="bf8b_tile"),
]


# Affine modes. Gamma/beta are always RM per the op spec.
AFFINE_MODES = [
    pytest.param("no_affine", id="no_affine"),
    pytest.param("gamma_only", id="gamma_only"),
    pytest.param("gamma_beta", id="gamma_beta"),
]


def _build_affine(
    affine_mode: str,
    W: int,
    dtype,
    device,
    torch_dtype,
) -> tuple[ttnn.Tensor | None, torch.Tensor | None, ttnn.Tensor | None, torch.Tensor | None]:
    """Build gamma / beta in (ttnn, torch) pairs based on affine_mode.

    Returns (ttnn_gamma, torch_gamma, ttnn_beta, torch_beta).
    Gamma/beta dtype matches the input dtype (per the op spec).
    Gamma/beta layout is always ROW_MAJOR_LAYOUT.
    """
    ttnn_gamma = torch_gamma = ttnn_beta = torch_beta = None

    if affine_mode in ("gamma_only", "gamma_beta"):
        torch_gamma = torch.randn(W, dtype=torch_dtype)
        ttnn_gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, W),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    if affine_mode == "gamma_beta":
        torch_beta = torch.randn(W, dtype=torch_dtype)
        ttnn_beta = ttnn.from_torch(
            torch_beta.reshape(1, 1, 1, W),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return ttnn_gamma, torch_gamma, ttnn_beta, torch_beta


# --- Acceptance tests ---------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype, layout", DTYPE_LAYOUT_COMBOS)
@pytest.mark.parametrize("affine_mode", AFFINE_MODES)
def test_layer_norm(device, shape, dtype, layout, affine_mode):
    """
    Acceptance: layer_norm with default epsilon and default compute_kernel_config
    matches the PyTorch reference (computed in fp32) within the per-dtype PCC tolerance.

    Covers:
      - rank 2 / 3 / 4 inputs
      - tile-aligned, W-partial, H-partial, and H+W-partial shapes
      - bfloat16 / float32 (both TILE and ROW_MAJOR) and bfloat8_b (TILE only)
      - no_affine / gamma_only / gamma_beta
    """
    torch.manual_seed(42)

    # Skip structurally invalid combos (mirrors feature_spec.py INVALID).
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b + ROW_MAJOR is structurally invalid (bf8b has no RM encoding)")

    torch_dtype = _TORCH_DTYPE[dtype]
    torch_input = torch.randn(shape, dtype=torch_dtype)

    W = shape[-1]
    ttnn_gamma, torch_gamma, ttnn_beta, torch_beta = _build_affine(
        affine_mode,
        W,
        dtype,
        device,
        torch_dtype,
    )

    torch_expected = pytorch_layer_norm(torch_input, torch_gamma, torch_beta, epsilon=1e-5)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm(ttnn_input, ttnn_gamma, ttnn_beta)

    # Shape and layout preserved.
    assert list(ttnn_output.shape) == list(
        shape
    ), f"Shape mismatch: got {list(ttnn_output.shape)}, expected {list(shape)}"
    assert ttnn_output.layout == layout, f"Layout mismatch: got {ttnn_output.layout}, expected {layout}"
    assert ttnn_output.dtype == dtype, f"Dtype mismatch: got {ttnn_output.dtype}, expected {dtype}"

    torch_output = ttnn.to_torch(ttnn_output)
    pcc = _pcc(torch_output.to(torch.float32), torch_expected.to(torch.float32))
    threshold = _PCC_BY_DTYPE[dtype]

    assert pcc >= threshold, (
        f"PCC {pcc:.6f} below threshold {threshold:.6f} for shape={shape}, "
        f"dtype={dtype}, layout={layout}, affine={affine_mode}\n"
        f"  max abs diff = {(torch_output - torch_expected).abs().max().item():.6f}\n"
        f"  actual[:4]   = {torch_output.to(torch.float32).flatten()[:4].tolist()}\n"
        f"  expected[:4] = {torch_expected.to(torch.float32).flatten()[:4].tolist()}"
    )


# --- Custom epsilon ---------------------------------------------------------


@pytest.mark.parametrize("epsilon", [1e-3, 1e-5, 1e-8])
def test_layer_norm_custom_epsilon(device, epsilon):
    """Epsilon is plumbed through to the kernel and changes the result deterministically."""
    torch.manual_seed(42)
    shape = (1, 1, 32, 256)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    torch_expected = pytorch_layer_norm(torch_input, epsilon=epsilon)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm(ttnn_input, epsilon=epsilon)
    torch_output = ttnn.to_torch(ttnn_output)

    pcc = _pcc(torch_output.to(torch.float32), torch_expected.to(torch.float32))
    assert pcc >= _PCC_BY_DTYPE[ttnn.bfloat16], f"PCC {pcc:.6f} below threshold for epsilon={epsilon}"


# --- Custom compute_kernel_config ------------------------------------------


@pytest.mark.parametrize(
    "compute_kernel_config",
    [
        pytest.param(
            ttnn.ComputeConfigDescriptor(),
            id="default_HiFi4",
        ),
        pytest.param(
            ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi),
            id="LoFi",
        ),
        pytest.param(
            ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True),
            id="fp32_dest_acc_en",
        ),
        pytest.param(
            ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
            ),
            id="HiFi4_fp32_dest",
        ),
    ],
)
def test_layer_norm_compute_kernel_config(device, compute_kernel_config):
    """compute_kernel_config is passed through to the compute KernelDescriptor.

    LoFi may degrade accuracy somewhat; we still expect PCC >= bf16 threshold
    on this well-conditioned random input (no extreme scale / outliers).
    """
    torch.manual_seed(42)
    shape = (1, 1, 64, 256)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    torch_gamma = torch.randn(256, dtype=torch.bfloat16)
    torch_beta = torch.randn(256, dtype=torch.bfloat16)
    torch_expected = pytorch_layer_norm(torch_input, torch_gamma, torch_beta)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, 256),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_beta = ttnn.from_torch(
        torch_beta.reshape(1, 1, 1, 256),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm(
        ttnn_input,
        ttnn_gamma,
        ttnn_beta,
        compute_kernel_config=compute_kernel_config,
    )
    torch_output = ttnn.to_torch(ttnn_output)

    pcc = _pcc(torch_output.to(torch.float32), torch_expected.to(torch.float32))
    # LoFi may degrade — allow one PCC notch lower than the standard bf16 threshold.
    # All others use the strict bf16 threshold.
    threshold = (
        0.99
        if str(compute_kernel_config) and "LoFi" in str(getattr(compute_kernel_config, "math_fidelity", ""))
        else _PCC_BY_DTYPE[ttnn.bfloat16]
    )
    assert pcc >= threshold, f"PCC {pcc:.6f} below threshold {threshold:.6f} for compute_kernel_config"


# --- Validation: bad inputs must raise -------------------------------------


def test_layer_norm_validate_rank_lt_2(device):
    """Input rank < 2 must raise ValueError."""
    torch_input = torch.randn(32, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises((ValueError, RuntimeError)):
        layer_norm(ttnn_input)


def test_layer_norm_validate_gamma_width_mismatch(device):
    """gamma with width != input width must raise."""
    torch_input = torch.randn(32, 64, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bad_gamma = ttnn.from_torch(
        torch.randn(1, 1, 1, 32, dtype=torch.bfloat16),  # width=32, input width=64
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises((ValueError, RuntimeError)):
        layer_norm(ttnn_input, bad_gamma)


def test_layer_norm_validate_gamma_layout_not_rm(device):
    """gamma in TILE layout must raise (op spec mandates RM gamma/beta)."""
    torch_input = torch.randn(32, 64, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bad_gamma = ttnn.from_torch(
        torch.randn(1, 1, 1, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises((ValueError, RuntimeError)):
        layer_norm(ttnn_input, bad_gamma)


def test_layer_norm_validate_gamma_dtype_mismatch(device):
    """gamma dtype != input dtype must raise."""
    torch_input = torch.randn(32, 64, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bad_gamma = ttnn.from_torch(
        torch.randn(1, 1, 1, 64, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises((ValueError, RuntimeError)):
        layer_norm(ttnn_input, bad_gamma)
