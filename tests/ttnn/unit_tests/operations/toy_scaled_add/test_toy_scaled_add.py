# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and program-cache tests for toy_scaled_add: out = a + alpha * (b * gamma).

Every test runs against each implementation in IMPLEMENTATIONS, so all of them answer to one
contract. The cache tests pin what a cache hit must get right: a repeat call with new buffers and a
new alpha reuses the cached program and still computes the new result.

    scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/toy_scaled_add/test_toy_scaled_add.py
"""

import pytest
import torch

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.toy_scaled_add import toy_scaled_add as toy_scaled_add_generic

from tests.ttnn.utils_for_testing import assert_with_pcc

IMPLEMENTATIONS = {
    "generic": toy_scaled_add_generic,  # Python program descriptor through ttnn.generic_op
}

TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
PCC = {ttnn.bfloat16: 0.9998, ttnn.float32: 0.99999}


def reference(a, b, alpha, gamma):
    a, b = a.float(), b.float()
    scaled = b * gamma.float() if gamma is not None else b
    return a + alpha * scaled


def to_device(t, device, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def height_sharded(shape, num_cores, grid_x):
    """Height-shard [..., H, W] over num_cores (row-major), shard height rounded up to whole tiles."""
    rows = 1
    for d in shape[:-1]:
        rows *= d
    tile_rows = -(-rows // 32)
    shard_h = -(-tile_rows // num_cores) * 32
    grid = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(grid_x, 8), row_wise=True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_h, shape[-1]], ttnn.ShardOrientation.ROW_MAJOR),
    )


@pytest.fixture(params=sorted(IMPLEMENTATIONS))
def op(request):
    return IMPLEMENTATIONS[request.param]


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],  # one tile, one core
        [2, 3, 64, 128],  # leading dims fold into rows
        [1, 1, 32 * 130, 64],  # more tile-rows than cores: two core groups
        [1, 1, 96, 2048],  # wide rows: long gamma row held in L1
    ],
)
@pytest.mark.parametrize("with_gamma", [False, True])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
def test_interleaved(device, op, shape, with_gamma, dtype, memory_config):
    torch.manual_seed(0)
    a = torch.randn(shape, dtype=TORCH_DTYPE[dtype])
    b = torch.randn(shape, dtype=TORCH_DTYPE[dtype])
    gamma = torch.randn([1, 1, 1, shape[-1]], dtype=TORCH_DTYPE[dtype]) if with_gamma else None
    alpha = -0.75

    out = op(
        to_device(a, device, dtype, memory_config),
        to_device(b, device, dtype, memory_config),
        alpha=alpha,
        gamma=to_device(gamma, device, dtype) if with_gamma else None,
    )
    assert out.dtype == dtype
    assert out.memory_config() == memory_config
    assert_with_pcc(reference(a, b, alpha, gamma), ttnn.to_torch(out).float(), PCC[dtype])


@pytest.mark.parametrize(
    "shape, num_cores",
    [
        ([1, 1, 256, 64], 8),  # one tile-row per core
        ([1, 2, 320, 128], 8),  # 20 tile-rows over 8 cores: last shard partial
        ([1, 1, 64, 256], 4),  # one shard holds two tile-rows
    ],
)
@pytest.mark.parametrize("with_gamma", [False, True])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_height_sharded(device, op, shape, num_cores, with_gamma, dtype):
    torch.manual_seed(1)
    grid_x = device.compute_with_storage_grid_size().x
    sharded = height_sharded(shape, num_cores, grid_x)
    a = torch.randn(shape, dtype=TORCH_DTYPE[dtype])
    b = torch.randn(shape, dtype=TORCH_DTYPE[dtype])
    gamma = torch.randn([1, 1, 1, shape[-1]], dtype=TORCH_DTYPE[dtype]) if with_gamma else None
    alpha = 1.5

    out = op(
        to_device(a, device, dtype, sharded),
        to_device(b, device, dtype, sharded),
        alpha=alpha,
        gamma=to_device(gamma, device, dtype) if with_gamma else None,
    )
    assert out.memory_config() == sharded
    assert_with_pcc(reference(a, b, alpha, gamma), ttnn.to_torch(out).float(), PCC[dtype])


@pytest.mark.parametrize("in_dtype, out_dtype", [(ttnn.bfloat16, ttnn.float32), (ttnn.float32, ttnn.bfloat16)])
@pytest.mark.parametrize(
    "compute_kernel_config",
    [
        None,
        ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
        ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False),
    ],
    ids=["default", "hifi4_fp32_dest", "hifi2"],
)
def test_dtypes_and_compute_config(device, op, in_dtype, out_dtype, compute_kernel_config):
    torch.manual_seed(2)
    shape = [1, 1, 128, 256]
    a = torch.randn(shape, dtype=TORCH_DTYPE[in_dtype])
    b = torch.randn(shape, dtype=TORCH_DTYPE[in_dtype])
    gamma = torch.randn([1, 1, 1, shape[-1]], dtype=TORCH_DTYPE[in_dtype])

    out = op(
        to_device(a, device, in_dtype),
        to_device(b, device, in_dtype),
        alpha=0.5,
        gamma=to_device(gamma, device, in_dtype),
        dtype=out_dtype,
        compute_kernel_config=compute_kernel_config,
    )
    assert out.dtype == out_dtype
    assert_with_pcc(reference(a, b, 0.5, gamma), ttnn.to_torch(out).float(), 0.9995)


@pytest.mark.parametrize("sharded", [False, True], ids=["interleaved", "height_sharded"])
def test_preallocated_and_in_place(device, op, sharded):
    torch.manual_seed(3)
    shape = [1, 1, 256, 128]
    memory_config = height_sharded(shape, 8, device.compute_with_storage_grid_size().x) if sharded else None
    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)
    expected = reference(a, b, 2.0, None)

    preallocated = to_device(torch.zeros(shape, dtype=torch.bfloat16), device, ttnn.bfloat16, memory_config)
    out = op(
        to_device(a, device, ttnn.bfloat16, memory_config),
        to_device(b, device, ttnn.bfloat16, memory_config),
        alpha=2.0,
        output_tensor=preallocated,
    )
    assert out.buffer_address() == preallocated.buffer_address()
    assert_with_pcc(expected, ttnn.to_torch(preallocated).float(), PCC[ttnn.bfloat16])

    a_dev = to_device(a, device, ttnn.bfloat16, memory_config)
    op(a_dev, to_device(b, device, ttnn.bfloat16, memory_config), alpha=2.0, output_tensor=a_dev)
    assert_with_pcc(expected, ttnn.to_torch(a_dev).float(), PCC[ttnn.bfloat16])


@pytest.mark.parametrize("sharded", [False, True], ids=["interleaved", "height_sharded"])
@pytest.mark.parametrize("with_gamma", [False, True])
def test_cache_hit_applies_new_buffers_and_alpha(device, op, sharded, with_gamma):
    """A repeat call differing only in buffers and alpha must hit the cached program and still
    compute its own result."""
    torch.manual_seed(4)
    shape = [1, 2, 320, 128]
    memory_config = (
        height_sharded(shape, 8, device.compute_with_storage_grid_size().x) if sharded else ttnn.DRAM_MEMORY_CONFIG
    )
    device.clear_program_cache()
    device.enable_program_cache()

    results = []
    for alpha in (1.0, -3.0, 0.25):
        a = torch.randn(shape, dtype=torch.bfloat16)
        b = torch.randn(shape, dtype=torch.bfloat16)
        gamma = torch.randn([1, 1, 1, shape[-1]], dtype=torch.bfloat16) if with_gamma else None
        # Keep every call's tensors alive so each call gets fresh buffer addresses.
        inputs = (
            to_device(a, device, ttnn.bfloat16, memory_config),
            to_device(b, device, ttnn.bfloat16, memory_config),
            to_device(gamma, device, ttnn.bfloat16) if with_gamma else None,
        )
        out = op(inputs[0], inputs[1], alpha=alpha, gamma=inputs[2])
        results.append((inputs, out, reference(a, b, alpha, gamma)))

    assert device.num_program_cache_entries() == 1
    for _, out, expected in results:
        assert_with_pcc(expected, ttnn.to_torch(out).float(), PCC[ttnn.bfloat16])


def test_cache_key_separates_programs(device, op):
    """Changes that alter the compiled program are separate cache entries."""
    device.clear_program_cache()
    device.enable_program_cache()
    shape = [1, 1, 64, 64]
    a = to_device(torch.randn(shape, dtype=torch.bfloat16), device, ttnn.bfloat16)
    b = to_device(torch.randn(shape, dtype=torch.bfloat16), device, ttnn.bfloat16)
    gamma = to_device(torch.randn([1, 1, 1, 64], dtype=torch.bfloat16), device, ttnn.bfloat16)

    op(a, b)
    op(a, b, gamma=gamma)
    op(a, b, dtype=ttnn.float32)
    op(a, b, compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi))
    assert device.num_program_cache_entries() == 4
    op(a, b, alpha=7.0)
    assert device.num_program_cache_entries() == 4


def test_rejects_inputs_that_do_not_fit_together(device, op, expect_error):
    shape = [1, 1, 64, 64]
    a = to_device(torch.randn(shape, dtype=torch.bfloat16), device, ttnn.bfloat16)
    b = to_device(torch.randn([1, 1, 64, 128], dtype=torch.bfloat16), device, ttnn.bfloat16)
    with expect_error((ValueError, RuntimeError), "padded shape"):
        op(a, b)
    wide_gamma = to_device(torch.randn([1, 1, 1, 128], dtype=torch.bfloat16), device, ttnn.bfloat16)
    with expect_error((ValueError, RuntimeError), "gamma"):
        op(a, a, gamma=wide_gamma)


def test_refuses_inputs_outside_the_support_contract(device, op, expect_error):
    """Refusals of the support contract raise the typed exceptions of ttnn.operations._op_contract, the
    same on both routes."""
    shape = [1, 1, 64, 64]
    a = to_device(torch.randn(shape, dtype=torch.bfloat16), device, ttnn.bfloat16)

    bfp8 = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(UnsupportedAxisValue, "dtype"):
        op(bfp8, bfp8)

    half_tiles = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        tile=ttnn.Tile([16, 32]),
    )
    with expect_error(UnsupportedAxisValue, "tile"):
        op(half_tiles, half_tiles)

    row_major = ttnn.from_torch(torch.randn(shape, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(UnsupportedAxisValue, "layout"):
        op(a, row_major)

    width_sharded = ttnn.create_sharded_memory_config(
        [64, 32], ttnn.CoreGrid(y=1, x=2), ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
    )
    with expect_error(UnsupportedAxisValue, "memory_layout"):
        op(a, a, memory_config=width_sharded)

    sharded_dram = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.num_cores_to_corerangeset(2, ttnn.CoreCoord(8, 8), row_wise=True),
            [32, 64],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    with expect_error(ExcludedCell, "HEIGHT_SHARDED"):
        op(a, a, memory_config=sharded_dram)
