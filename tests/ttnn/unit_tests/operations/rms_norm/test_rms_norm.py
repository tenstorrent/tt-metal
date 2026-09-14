# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for ttnn.operations.rms_norm — the immutable specification.

    RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma

Covers the Phase 0 rectangle: bfloat16 / float32, TILE and ROW_MAJOR input layouts (output
layout == input layout), rank 2/3/4, optional gamma (ROW_MAJOR (1,1,1,W) by contract, TILE also
accepted), custom epsilon, an explicit maxed-out compute config, and one WIDTH_SHARDED case.

The shape set pins every blocking regime of op_design.md:
  - (2, 4, 128, 512)   row-split, many tile-rows (R1)
  - (1, 1, 32, 4096)   one tile-row, wide: W-split forced by occupancy (R2)
  - (1, 1, 64, 12288)  W too wide for one core's L1: W-split forced by residency (R2)
  - (1, 1, 32, 2048)   WIDTH_SHARDED over 8 cores: cross-core combine on resident shards (R3)

PCC thresholds are keyed by dtype and match the golden suite: float32 0.999, bfloat16 0.995.
The implementer must not modify this file.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.rms_norm import rms_norm

PCC_BY_DTYPE = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
}

TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}


def torch_rms_norm(x, gamma=None, epsilon=1e-6):
    """Reference in fp32, returned in the input dtype (same convention as the golden suite)."""
    orig = x.dtype
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + epsilon)
    y = xf / rms
    if gamma is not None:
        y = y * gamma.to(torch.float32).reshape(-1)
    return y.to(orig)


def _make_inputs(shape, dtype, with_gamma):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32).to(TORCH_DTYPE[dtype])
    g = torch.randn(shape[-1], dtype=torch.float32).to(TORCH_DTYPE[dtype]) if with_gamma else None
    return x, g


def _to_device(device, x, dtype, layout, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _gamma_to_device(device, g, dtype, gamma_layout):
    # Contract: gamma is a (1, 1, 1, W) tensor; ROW_MAJOR by default, TILE also accepted.
    return _to_device(device, g.reshape(1, 1, 1, -1), dtype, gamma_layout)


def _check(ttnn_output, expected, shape, layout, dtype):
    assert list(ttnn_output.shape) == list(shape), f"shape mismatch: {ttnn_output.shape} vs {shape}"
    assert ttnn_output.layout == layout, f"output layout {ttnn_output.layout} != input layout {layout}"
    assert ttnn_output.dtype == dtype, f"output dtype {ttnn_output.dtype} != input dtype {dtype}"
    actual = ttnn.to_torch(ttnn_output).to(torch.float32)
    assert_with_pcc(expected.to(torch.float32), actual, PCC_BY_DTYPE[dtype])


SHAPES = [
    pytest.param((32, 32), id="2d_single_tile"),
    pytest.param((1, 1, 64, 128), id="4d_2x4_tiles"),
    pytest.param((4, 128, 512), id="3d_multi_batch"),
    pytest.param((2, 4, 128, 512), id="4d_non_square_multi_batch_R1"),
    pytest.param((1, 1, 32, 4096), id="one_tile_row_wide_R2_occupancy"),
    pytest.param((1, 1, 64, 12288), id="two_tile_rows_very_wide_R2_residency"),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("with_gamma", [True, False], ids=["gamma", "no_gamma"])
def test_rms_norm(device, shape, layout, dtype, with_gamma):
    x, g = _make_inputs(shape, dtype, with_gamma)
    expected = torch_rms_norm(x, g)

    ttnn_x = _to_device(device, x, dtype, layout)
    ttnn_g = _gamma_to_device(device, g, dtype, ttnn.ROW_MAJOR_LAYOUT) if with_gamma else None

    ttnn_out = rms_norm(ttnn_x, gamma=ttnn_g)
    _check(ttnn_out, expected, shape, layout, dtype)


@pytest.mark.parametrize("shape", [(1, 1, 64, 128), (1, 1, 32, 4096)], ids=["small", "wide"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_rms_norm_tile_layout_gamma(device, shape, dtype):
    """gamma supplied as a TILE_LAYOUT (1,1,1,W) tensor (padded to one tile-row on device)."""
    x, g = _make_inputs(shape, dtype, True)
    expected = torch_rms_norm(x, g)

    ttnn_x = _to_device(device, x, dtype, ttnn.TILE_LAYOUT)
    ttnn_g = _gamma_to_device(device, g, dtype, ttnn.TILE_LAYOUT)

    ttnn_out = rms_norm(ttnn_x, gamma=ttnn_g)
    _check(ttnn_out, expected, shape, ttnn.TILE_LAYOUT, dtype)


@pytest.mark.parametrize("epsilon", [1e-5, 1e-2], ids=["eps_1e-5", "eps_1e-2"])
def test_rms_norm_epsilon(device, epsilon):
    shape, dtype = (2, 64, 256), ttnn.bfloat16
    x, g = _make_inputs(shape, dtype, True)
    # Scale the input down so epsilon is a visible fraction of mean(x^2).
    x = (x.to(torch.float32) * 0.05).to(TORCH_DTYPE[dtype])
    expected = torch_rms_norm(x, g, epsilon=epsilon)

    ttnn_x = _to_device(device, x, dtype, ttnn.TILE_LAYOUT)
    ttnn_g = _gamma_to_device(device, g, dtype, ttnn.ROW_MAJOR_LAYOUT)

    ttnn_out = rms_norm(ttnn_x, gamma=ttnn_g, epsilon=epsilon)
    _check(ttnn_out, expected, shape, ttnn.TILE_LAYOUT, dtype)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_rms_norm_explicit_compute_config(device, dtype):
    """The maxed-out precision corner passed explicitly (HiFi4 + fp32 DEST accumulation)."""
    shape = (1, 2, 64, 1024)
    x, g = _make_inputs(shape, dtype, True)
    expected = torch_rms_norm(x, g, epsilon=1e-5)

    ttnn_x = _to_device(device, x, dtype, ttnn.TILE_LAYOUT)
    ttnn_g = _gamma_to_device(device, g, dtype, ttnn.ROW_MAJOR_LAYOUT)

    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    ttnn_out = rms_norm(ttnn_x, gamma=ttnn_g, epsilon=1e-5, compute_kernel_config=config)
    _check(ttnn_out, expected, shape, ttnn.TILE_LAYOUT, dtype)


def test_rms_norm_width_sharded(device):
    """WIDTH_SHARDED input: W=2048 split over 8 cores ([32, 256] shards) -> cross-core combine (R3)."""
    grid = device.compute_with_storage_grid_size()
    if grid.x < 8:
        pytest.skip("needs a compute grid at least 8 cores wide")

    shape, dtype = (1, 1, 32, 2048), ttnn.bfloat16
    x, g = _make_inputs(shape, dtype, True)
    expected = torch_rms_norm(x, g)

    sharded_config = ttnn.create_sharded_memory_config(
        shape=(32, 256),
        core_grid=ttnn.CoreGrid(x=8, y=1),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    ttnn_x = _to_device(device, x, dtype, ttnn.TILE_LAYOUT, memory_config=sharded_config)
    ttnn_g = _gamma_to_device(device, g, dtype, ttnn.TILE_LAYOUT)

    ttnn_out = rms_norm(ttnn_x, gamma=ttnn_g, memory_config=ttnn_x.memory_config())
    assert ttnn_out.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    _check(ttnn_out, expected, shape, ttnn.TILE_LAYOUT, dtype)


def test_rms_norm_rejects_rank_below_two(device, expect_error):
    x = torch.randn(64, dtype=torch.bfloat16)
    ttnn_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    # The error text must mention the offending axis ("rank") — pinned by op_design.md.
    with expect_error((ValueError, RuntimeError), "rank"):
        rms_norm(ttnn_x)


def test_rms_norm_rejects_gamma_width_mismatch(device, expect_error):
    shape, dtype = (1, 1, 64, 128), ttnn.bfloat16
    x, _ = _make_inputs(shape, dtype, False)
    ttnn_x = _to_device(device, x, dtype, ttnn.TILE_LAYOUT)
    bad_gamma = torch.randn(1, 1, 1, 256, dtype=torch.bfloat16)
    ttnn_bad_gamma = _to_device(device, bad_gamma, dtype, ttnn.ROW_MAJOR_LAYOUT)
    # The error text must mention "gamma" — pinned by op_design.md.
    with expect_error((ValueError, RuntimeError), "gamma"):
        rms_norm(ttnn_x, gamma=ttnn_bad_gamma)
