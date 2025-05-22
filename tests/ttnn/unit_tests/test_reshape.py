# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull


@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        ((10, 20), (20, 10)),
    ],
)
@pytest.mark.parametrize("enable_cache", [True])
def test_ttnn_reshape_with_cache(device, enable_cache, input_shape, output_shape):
    if enable_cache:
        device.enable_program_cache()

    a = torch.randn(input_shape, dtype=torch.bfloat16)
    b = torch.randn(input_shape, dtype=torch.bfloat16)

    tt_a = ttnn.from_torch(a, device=device)
    tt_b = ttnn.from_torch(b, device=device)

    a = a.reshape(output_shape)
    b = b.reshape(output_shape)

    tt_a = ttnn.reshape(tt_a, output_shape)
    tt_b = ttnn.reshape(tt_b, output_shape)

    assert torch.allclose(a, ttnn.to_torch(tt_a))
    assert torch.allclose(b, ttnn.to_torch(tt_b))


@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        ((10, 20), (20, 10)),
    ],
)
@pytest.mark.parametrize("enable_cache", [True])
def test_tensor_reshape_with_cache(device, enable_cache, input_shape, output_shape):
    if enable_cache:
        device.enable_program_cache()

    a = torch.randn(input_shape, dtype=torch.bfloat16)
    b = torch.randn(output_shape, dtype=torch.bfloat16)

    tt_a = ttnn.from_torch(a, device=device)
    tt_b = ttnn.from_torch(b, device=device)

    a = a.reshape(output_shape)
    b = b.reshape(output_shape)

    tt_a = tt_a.reshape(output_shape)
    tt_b = tt_b.reshape(output_shape)

    assert torch.allclose(a, ttnn.to_torch(tt_a))
    assert torch.allclose(b, ttnn.to_torch(tt_b))


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [4])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [64])
def test_reshape_sharded_rm(device, n, c, h, w):
    pytest.skip("skipped to unblock P0 issue 16975 but needs to be fixed and removed for issue 17030")

    if device.core_grid.y < 8:
        pytest.skip("n300 does not have 8x8 grid")

    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.reshape(n, c, h * 2, w // 2)
    torch_output_tensor = torch_output_tensor.transpose(1, 2)

    core_grid = ttnn.CoreGrid(x=8, y=8)
    sharded_mem_config = ttnn.create_sharded_memory_config(
        torch_input_tensor.shape,
        core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=sharded_mem_config
    )

    tt_output_tensor = ttnn.experimental.view(tt_input_tensor, n, c, h * 2, w // 2)

    sharded_mem_config = ttnn.create_sharded_memory_config(
        tt_output_tensor.shape,
        core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_output_tensor = ttnn.transpose(tt_output_tensor, 1, 2, memory_config=sharded_mem_config)

    tt_output_tensor = ttnn.to_memory_config(tt_output_tensor, ttnn.L1_MEMORY_CONFIG)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.9999)


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [128])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [16])
def test_reshape_cw_div2_rm(device, n, c, h, w):
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.reshape(n, c * 2, h, w // 2)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_tensor = ttnn.reshape_on_device(input_tensor, n, c * 2, h, w // 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [128])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [16])
def test_reshape_cw_mul2_rm(device, n, c, h, w):
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.reshape(n, c // 2, h, w * 2)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_tensor = ttnn.reshape_on_device(input_tensor, n, c // 2, h, w * 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [128])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [16])
def test_reshape_hw_div2_rm(device, n, c, h, w):
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.reshape(n, c, h * 2, w // 2)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_tensor = ttnn.reshape_on_device(input_tensor, n, c, h * 2, w // 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [128])
@pytest.mark.parametrize("h", [256])
@pytest.mark.parametrize("w", [8])
def test_reshape_hw_mul2_rm(device, n, c, h, w):
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.reshape(n, c, h // 2, w * 2)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_tensor = ttnn.reshape_on_device(input_tensor, n, c, h // 2, w * 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


def run_reshape_hw_rm_with_program_cache(device, n, c, h, w, use_program_cache):
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.reshape(n, c, h // 2, w * 2)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_tensor = ttnn.reshape_on_device(input_tensor, n, c, h // 2, w * 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [128])
@pytest.mark.parametrize("h", [256])
@pytest.mark.parametrize("w", [8])
def test_reshape_hw_rm_with_program_cache(device, n, c, h, w, use_program_cache):
    for _ in range(2):
        run_reshape_hw_rm_with_program_cache(device, n, c, h, w, use_program_cache)
        # dummy tensor to change tensor alloc
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.from_torch(
            py_dummy_tensor,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    assert device.num_program_cache_entries() == 1


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_reshape(h, w):
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.reshape(w, h)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    output_tensor = ttnn.reshape(input_tensor, (w, h))
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_reshape_negative_1(h, w):
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.reshape(-1)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    output_tensor = ttnn.reshape(input_tensor, (-1,))
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("n", [32, 32])
@pytest.mark.parametrize("c", [2 * 32, 2 * 32])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("w", [1, 4])
def test_reshape_in_4D(n, c, h, w):
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.reshape(h, w, n, c)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    output_tensor = ttnn.reshape(input_tensor, (h, w, n, c))
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("n", [32, 64])
@pytest.mark.parametrize("c", [32, 64])
@pytest.mark.parametrize("h", [32, 64])
@pytest.mark.parametrize("w", [32, 64])
def test_reshape_in_4D_on_device(device, n, c, h, w):
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.reshape(h, w, n, c)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.reshape(input_tensor, (h, w, n, c))
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


def test_permute_reshape(device):
    input_shape = (1, 4, 64, 32)
    output_shape = (1, 64, 128)

    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.permute(torch_input_tensor, (0, 2, 1, 3))
    torch_output_tensor = torch.reshape(torch_output_tensor, output_shape)

    output_tensor = ttnn.from_torch(torch_input_tensor)
    output_tensor = ttnn.to_device(output_tensor, device)
    output_tensor = ttnn.permute(output_tensor, (0, 2, 1, 3))
    output_tensor = ttnn.reshape(output_tensor, output_shape)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


def test_reshape_with_negative_dim(device):
    input_shape = (1, 4, 64, 32)
    output_shape = (1, -1, 2, 32)
    expected_output_shape = (1, 128, 2, 32)

    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_output = torch.reshape(torch_input, output_shape)

    tt_input = ttnn.from_torch(torch_input)
    tt_input = ttnn.to_device(tt_input, device)
    tt_output = ttnn.reshape(tt_input, output_shape)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert list(expected_output_shape) == list(torch_output.shape)
    assert list(expected_output_shape) == list(tt_output.shape)
    assert_with_pcc(torch_output, tt_output, 0.9999)


def test_reshape_tile_layout_mamba(device):
    torch_input_tensor = torch.randn((1, 1, 2048, 64), dtype=torch.bfloat16)
    reshape_shape = (1, 2, 1024, 64)
    torch_result = torch_input_tensor.reshape(reshape_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_output = ttnn.reshape(input_tensor, reshape_shape)

    output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_result, output, 0.9999)


def test_reshape_tile_layout_only_change_shape(device):
    torch_input_tensor = torch.randn((1, 64, 32, 4 * 32), dtype=torch.bfloat16)
    reshape_shape = (1, 32, 64, 4 * 32)
    torch_result = torch_input_tensor.reshape(reshape_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_output = ttnn.reshape(input_tensor, reshape_shape)

    output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_result, output, 0.9999)


@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        ((1, 8, 8), (1, 16, 4)),
        ((1, 17, 1), (1, 1, 17)),
        ((1, 32, 17), (1, 17, 32)),
        ((2, 32, 17), (2, 17, 32)),
        ((2, 2, 1), (1, 4, 1)),
        ((16, 1, 5), (4, 2, 10)),
        ((1, 256, 1), (1, 256)),
        ((1, 256, 1), (1, 256, 1, 1)),
        ((1, 180, 1), (1, 180, 1, 1)),
        ((1, 256, 1024), (1, 256, 16, 64)),
        ((1, 1445, 192), (1445, 192)),
        ((1, 256), (1, 1, 256)),
        ((16, 1, 32), (16, 1, 32)),
        ((1, 32, 4608), (1, 32, 16, 3, 96)),  # issue 13889
        ((128, 1, 1, 128), (128, 128)),  # issue 14676
        ((16, 33, 1), (176, 3, 1)),
        ((2888, 49, 96), (8, 19, 19, 7, 7, 96)),  # issue 12153
        ((5, 4, 208, 156), (3, 13, 8, 2080)),  # issue 14513
        ((22, 23, 1), (1, 22, 23)),
        ((1, 1500, 1, 512), (1, 1500, 8, 64)),
        ((32, 1, 96, 64), (1, 32, 96, 64)),  # issue 20238
    ],
)
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "dtype", [(torch.bfloat16, ttnn.bfloat16), (torch.int32, ttnn.uint32), (torch.float32, ttnn.float32)]
)
def test_reshape_tile(device, input_shape, output_shape, layout, memory_config, dtype):
    if memory_config == ttnn.L1_MEMORY_CONFIG and input_shape in [(2888, 49, 96), (1, 1500, 1, 512)]:
        pytest.xfail("Test case is too big for L1")

    torch_dtype, ttnn_dtype = dtype

    size = math.prod(input_shape)
    torch_input_tensor = torch.linspace(1, size, size, dtype=torch_dtype).reshape(input_shape)

    if torch_dtype == torch.int32:
        torch_input_tensor = torch_input_tensor.abs()

    torch_result = torch_input_tensor.reshape(output_shape)
    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=layout, dtype=ttnn_dtype, device=device, memory_config=memory_config
    )
    ttnn_output = ttnn.reshape(input_tensor, output_shape)
    output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_result, output, 0.9999)


def test_reshape_tile_program_cache(device, use_program_cache):
    for input_shape, output_shape in ((1, 8, 8), (1, 16, 4)), ((16, 1, 5), (4, 2, 10)):
        for _ in range(3):
            torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
            torch_result = torch_input_tensor.reshape(output_shape)

            input_tensor = ttnn.from_torch(
                torch_input_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
            )
            ttnn_output = ttnn.reshape(input_tensor, output_shape)

            output = ttnn.to_torch(ttnn_output)
            assert_with_pcc(torch_result, output, 0.9999)


# issue 15048
def test_previously_failing_test(device):
    src_shape = (1, 56, 56, 64)
    target_shape = (1, 1, 56 * 56, 64)
    torch_input_tensor = torch.randn(src_shape, dtype=torch.bfloat16)
    torch_result = torch_input_tensor.reshape(target_shape)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.reshape(input_tensor, target_shape)
    output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_result, output, 0.9999)


# Since Inner dim is 1 of bfloat16, can't do on device, testing fallback on host
@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        ((1, 256, 1), (1, 256)),
        ((1, 1024, 1), (1, 4, 256)),
        ((1, 128, 1), (1, 128)),
    ],
)
def test_reshape_host(input_shape, output_shape, device):
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_result = torch_input_tensor.reshape(output_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_output = ttnn.reshape(input_tensor, output_shape)

    output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_result, output, 0.9999)


# required for Embedding
@skip_for_grayskull("avoid this test while issue 15702 is resolved")
@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        ((1, 12), (12, 1)),
        ((1, 32), (32, 1)),
        ((64, 32), (1, 1, 64, 32)),
    ],
)
def test_reshape_int(input_shape, output_shape, device):
    torch_input_tensor = torch.randint(0, 100, input_shape)
    torch_result = torch_input_tensor.reshape(output_shape)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = ttnn.reshape(input_tensor, output_shape)

    output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_result, output, 0.9999)


@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        ((1, 1, 756, 128), (1, 27, 28, 128)),
        ((1, 256, 16), (16, 256)),
        ((1, 256, 1024), (1, 256, 16, 64)),
        ((16, 16), (32, 8)),
        ((1, 1445, 192), (1445, 192)),
        ((1, 256), (1, 1, 256)),
        ((16, 1, 32), (16, 1, 32)),
    ],
)
def test_fp32_support(input_shape, output_shape, device):
    torch_input_tensor = torch.randint(0, 100, input_shape)
    torch_result = torch_input_tensor.reshape(output_shape)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.reshape(input_tensor, output_shape)

    output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_result, output, 0.9999)


@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        ((1, 1, 864, 128), (1, 27, 32, 128)),
        ((1, 256, 32), (32, 256)),
        ((1, 256, 1024), (1, 128, 32, 64)),
        ((64, 32), (32, 64)),
        ((1, 1445, 192), (1445, 192)),
        ((1, 256), (1, 1, 256)),
        ((16, 1, 32), (16, 1, 32)),
    ],
)
def test_bf8_support(input_shape, output_shape, device):
    torch_input_tensor = torch.randint(0, 100, input_shape)
    torch_result = torch_input_tensor.reshape(output_shape)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.reshape(input_tensor, output_shape)

    output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_result, output, 0.9999)


@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        ([0], [0, 1]),
        ([0], [1, 0]),
        ([0, 5], [0, 0, 5]),
        ([5, 0], [0, 5, 0]),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
)
@pytest.mark.parametrize(
    "ttnn_reshape",
    [True, False],
)
@pytest.mark.parametrize(
    "use_device, memory_config",
    [(True, None), (True, ttnn.L1_MEMORY_CONFIG), (False, None)],
)
def test_reshape_zero_element(input_shape, output_shape, layout, ttnn_reshape, use_device, memory_config, device):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    if use_device:
        tt_input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, memory_config=memory_config)
    else:
        tt_input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout)
    if ttnn_reshape:
        tt_output_tensor = ttnn.reshape(tt_input_tensor, output_shape)
    else:
        tt_output_tensor = ttnn.experimental.view(tt_input_tensor, output_shape)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert tt_output_tensor.shape == torch.Size(output_shape)


@pytest.mark.xfail(
    reason="Test that the previously supported reshape accounting for the physical shape is no longer possible"
)
@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        ([32, 256], [1, 256]),
    ],
)
def test_reshape_replicated_tensor(mesh_device, input_shape, output_shape):
    torch_input_tensor = torch.randn(input_shape)
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper, device=mesh_device
    )
    tt_output_tensor = ttnn.reshape(tt_input_tensor, ttnn.Shape(output_shape))

    for tensor_shard in ttnn.get_device_tensors(tt_output_tensor):
        tt_output_tensor = ttnn.to_torch(tensor_shard)
        assert tt_output_tensor.shape == torch.Size(output_shape)
