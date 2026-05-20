# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — streaming reduce + multi-core distribution.

Exercises the cells that were OOM under Phase 0:
  - W ∈ {4096, 8192} (NUM_BLOCKS > 1 streaming path)
  - multi-core distribution (total_tile_rows > 1)

Each case is hand-picked from the `feature_spec.py` OOM list and asserts
the per-dtype golden PCC tolerances. Adds:
  - test_streaming_no_affine: wide-W without gamma/beta
  - test_streaming_with_gamma_beta: wide-W with affine (per-block gamma/beta tilize)
  - test_multicore_distribution: many tile-rows across cores

DO NOT DELETE - documents the Refinement 2 contract.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm import layer_norm


# --- Reference (matches eval/golden_tests/layer_norm_rm/helpers.py) -----

_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
}


def _pytorch_layer_norm(
    x: torch.Tensor,
    gamma: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    x32 = x.to(torch.float32)
    mean = x32.mean(dim=-1, keepdim=True)
    var = x32.var(dim=-1, keepdim=True, unbiased=False)
    out = (x32 - mean) / torch.sqrt(var + eps)
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    if beta is not None:
        out = out + beta.to(torch.float32).reshape(-1)
    return out.to(x.dtype)


def _pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.detach().to(torch.float64).flatten()
    e = expected.detach().to(torch.float64).flatten()
    finite = torch.isfinite(a) & torch.isfinite(e)
    a, e = a[finite], e[finite]
    a_c = a - a.mean()
    e_c = e - e.mean()
    denom = torch.sqrt((a_c * a_c).sum() * (e_c * e_c).sum())
    if denom.item() == 0.0:
        return 1.0 if torch.allclose(a, e, atol=1e-3) else 0.0
    return (a_c * e_c).sum().item() / denom.item()


_PCC_BY_DTYPE = {
    ttnn.float32: 0.9999,
    ttnn.bfloat16: 0.995,
}


# --- Streaming (wide-W) ------------------------------------------------------

# Wide-W shapes that were OOM under Phase 0. (1, 1, 32, *) keeps Ht_local=1
# per core so the test isolates the streaming path from multi-core effects.
STREAMING_SHAPES = [
    pytest.param((1, 1, 32, 4096), id="w4096"),
    pytest.param((1, 1, 32, 8192), id="w8192"),
]

STREAMING_LAYOUTS = [
    pytest.param(ttnn.TILE_LAYOUT, id="tile"),
    pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="rm"),
]


@pytest.mark.parametrize("shape", STREAMING_SHAPES)
@pytest.mark.parametrize("dtype", [pytest.param(ttnn.bfloat16, id="bf16")])
@pytest.mark.parametrize("layout", STREAMING_LAYOUTS)
def test_streaming_no_affine(device, shape, dtype, layout):
    """Wide-W (NUM_BLOCKS > 1) without gamma/beta. Tests the core streaming path."""
    torch.manual_seed(42)
    torch_dtype = _TORCH_DTYPE[dtype]
    x = torch.randn(shape, dtype=torch_dtype)

    expected = _pytorch_layer_norm(x)

    ttnn_x = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn_y = layer_norm(ttnn_x)
    y = ttnn.to_torch(ttnn_y)

    assert list(ttnn_y.shape) == list(shape)
    assert ttnn_y.dtype == dtype
    assert ttnn_y.layout == layout

    pcc = _pcc(y.to(torch.float32), expected.to(torch.float32))
    threshold = _PCC_BY_DTYPE[dtype]
    assert (
        pcc >= threshold
    ), f"PCC {pcc:.6f} below threshold {threshold:.6f} for shape={shape}, dtype={dtype}, layout={layout}"


@pytest.mark.parametrize("shape", STREAMING_SHAPES)
@pytest.mark.parametrize("dtype", [pytest.param(ttnn.bfloat16, id="bf16")])
@pytest.mark.parametrize("layout", STREAMING_LAYOUTS)
def test_streaming_with_gamma_beta(device, shape, dtype, layout):
    """Wide-W with gamma+beta. Tests the per-block gamma/beta tilize path."""
    torch.manual_seed(7)
    torch_dtype = _TORCH_DTYPE[dtype]
    W = shape[-1]

    x = torch.randn(shape, dtype=torch_dtype)
    gamma = torch.randn(W, dtype=torch_dtype)
    beta = torch.randn(W, dtype=torch_dtype)

    expected = _pytorch_layer_norm(x, gamma, beta)

    ttnn_x = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn_gamma = ttnn.from_torch(
        gamma.reshape(1, 1, 1, W),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_beta = ttnn.from_torch(
        beta.reshape(1, 1, 1, W),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_y = layer_norm(ttnn_x, ttnn_gamma, ttnn_beta)
    y = ttnn.to_torch(ttnn_y)

    pcc = _pcc(y.to(torch.float32), expected.to(torch.float32))
    threshold = _PCC_BY_DTYPE[dtype]
    assert pcc >= threshold, f"PCC {pcc:.6f} below threshold {threshold:.6f} for shape={shape}, gamma+beta"


# --- Multi-core distribution -------------------------------------------------

# Shapes that produce total_tile_rows > 1 so the work is actually distributed
# across cores. (2,1,64,4096) was on the Phase-0 OOM list and is a real
# multi-core × wide-W stress.
MULTICORE_SHAPES = [
    pytest.param((4, 1, 32, 256), id="ht4_w256"),  # 4 tile-rows, single block
    pytest.param((2, 1, 64, 4096), id="ht4_w4096"),  # 4 tile-rows, streaming
    pytest.param((1, 1, 128, 4096), id="ht4_w4096_alt"),  # 4 tile-rows, streaming
    pytest.param((1, 1, 1024, 256), id="ht32_w256"),  # 32 tile-rows, single block
]


@pytest.mark.parametrize("shape", MULTICORE_SHAPES)
@pytest.mark.parametrize("dtype", [pytest.param(ttnn.bfloat16, id="bf16")])
def test_multicore_distribution(device, shape, dtype):
    """total_tile_rows distributed across the compute grid via split_work_to_cores."""
    torch.manual_seed(11)
    torch_dtype = _TORCH_DTYPE[dtype]

    x = torch.randn(shape, dtype=torch_dtype)
    expected = _pytorch_layer_norm(x)

    ttnn_x = ttnn.from_torch(
        x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_y = layer_norm(ttnn_x)
    y = ttnn.to_torch(ttnn_y)

    assert list(ttnn_y.shape) == list(shape)
    pcc = _pcc(y.to(torch.float32), expected.to(torch.float32))
    threshold = _PCC_BY_DTYPE[dtype]
    assert pcc >= threshold, f"PCC {pcc:.6f} below threshold {threshold:.6f} for shape={shape}"
