# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision matrix for rms_norm — the authoritative precision characterization.

Added by Refinement 1 (numerical configurability expansion). Locks in the full
R1 precision surface:
  * dtype           ∈ {bfloat16, float32, bfloat8_b}
  * fp32_dest_acc_en ∈ {True, False}   (skipping the {float32, False} EXCLUSION)
  * math_fidelity    ∈ {HiFi4, HiFi2}  (HiFi4 = maxed corner; HiFi2 = perf config)
  * both input distributions (uniform / normal), gamma present / absent.

Assert only on PCC (per /numeric-formats-metal §11) at the golden-suite
TOLERANCES gate; print rel-RMS + allclose for observability. bfloat8_b is a
block-float format — TILE + tile-aligned only (bf8b+RM / bf8b+non-aligned are
INVALID), and its gamma is carried at bf16 (bf8b RM gamma is impossible).

The `device` fixture comes from the directory conftest (module-scoped) — do NOT
open a device manually.
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import comp_allclose


# torch has no native bf8b; reference it in bf16 (matches the golden helpers).
_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,
}

# PCC gate per dtype — identical to the golden suite TOLERANCES.
_PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}


def _reference(x, gamma, eps):
    x = x.to(torch.float32)
    out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out


# All tile-aligned so bf8b (TILE + tile-aligned only) is exercised on every
# shape; small→wide to reach the multi-tile reduce path.
SHAPES = [
    pytest.param((32, 64), id="32x64_small"),
    pytest.param((2, 64, 128), id="2x64x128_3d"),
    pytest.param((2, 4, 128, 512), id="2x4x128x512_4d"),
    pytest.param((1, 1, 128, 4096), id="1x1x128x4096_wide"),
]


@pytest.mark.parametrize("distribution", ["rand", "randn"], ids=["uniform", "normal"])
@pytest.mark.parametrize("with_gamma", [True, False], ids=["gamma", "no_gamma"])
@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param(ttnn.MathFidelity.HiFi4, id="HiFi4"),
        pytest.param(ttnn.MathFidelity.HiFi2, id="HiFi2"),
    ],
)
@pytest.mark.parametrize("fp32_acc", [True, False], ids=["fp32_acc", "bf16_acc"])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="f32"),
        pytest.param(ttnn.bfloat8_b, id="bf8b"),
    ],
)
@pytest.mark.parametrize("shape", SHAPES)
def test_rms_norm_precision_matrix(device, shape, dtype, fp32_acc, math_fidelity, with_gamma, distribution):
    # {float32, fp32_dest_acc_en=False} is the op EXCLUSION (lossy) — skip it.
    if dtype == ttnn.float32 and not fp32_acc:
        pytest.skip("EXCLUSION: {float32, fp32_dest_acc_en=False} is refused (lossy).")

    torch.manual_seed(42)
    eps = 1e-6
    W = shape[-1]

    if distribution == "rand":
        torch_input = torch.rand(shape, dtype=torch.float32)
    else:
        torch_input = torch.randn(shape, dtype=torch.float32)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_gamma = None
    ttnn_gamma = None
    if with_gamma:
        torch_gamma = torch.randn(W, dtype=torch.float32)
        # bf8b has no ROW_MAJOR representation; carry the (RM) gamma at bf16 for a
        # bf8b input — the realistic mixed-precision LLM pattern.
        gamma_dtype = ttnn.bfloat16 if dtype == ttnn.bfloat8_b else dtype
        ttnn_gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, W),
            dtype=gamma_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    config = ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity, fp32_dest_acc_en=fp32_acc)
    ttnn_out = rms_norm(ttnn_input, gamma=ttnn_gamma, epsilon=eps, compute_kernel_config=config)
    got = ttnn.to_torch(ttnn_out).to(torch.float32)
    exp = _reference(torch_input, torch_gamma, eps)

    rel_rms = ((got - exp).pow(2).mean().sqrt() / exp.pow(2).mean().sqrt().clamp(min=1e-10)).item()
    _, allclose_str = comp_allclose(exp, got)
    print(
        f"\n[precision-matrix] shape={tuple(shape)} dtype={dtype} fp32_acc={fp32_acc} "
        f"fidelity={math_fidelity} gamma={with_gamma} dist={distribution}\n"
        f"    rel_rms={rel_rms:.6f}  {allclose_str}"
    )

    assert_with_pcc(exp, got, pcc=_PCC[dtype])
