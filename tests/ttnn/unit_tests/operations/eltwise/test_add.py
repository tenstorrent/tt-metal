# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("size", [64])
def test_add_1D_tensor_and_scalar(device, scalar, size):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((size,), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor + scalar

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = input_tensor + scalar
    output_tensor = ttnn.to_torch(output_tensor, torch_rank=1)

    print("torch_output_tensor", torch_output_tensor)
    print("output_tensor", output_tensor)
    print(ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor))

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99964
    assert output_tensor.shape == (size,)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("size", [64])
def test_add_1D_tensor_and_scalar_float(device, scalar, size):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((size,), dtype=torch.float)
    torch_output_tensor = torch_input_tensor + scalar

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = input_tensor + scalar
    output_tensor = ttnn.to_torch(output_tensor, torch_rank=1)

    print("torch_output_tensor", torch_output_tensor)
    print("output_tensor", output_tensor)
    print(ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor))

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
    assert output_tensor.shape == (size,)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_add_2D_tensors(device, h, w):
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.add(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_add_2D_tensors_with_program_cache(device, h, w, use_program_cache):
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.add(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("scalar", [0.42])
def test_add_scalar(device, h, w, scalar):
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = scalar + torch_input_tensor_a

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    output = input_tensor_a + scalar
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("scalar", [0.42])
def test_reverse_add_scalar(device, h, w, scalar):
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = scalar + torch_input_tensor_a

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    output = scalar + input_tensor_a
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_add_4D_tensors(device, h, w):
    torch_input_tensor_a = torch.rand((5, 64, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((5, 64, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.add(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_add_with_broadcast(device, h, w):
    # See #4005, we basically are using ttnn.repeat to get this to pass.
    torch_input_tensor_a = torch.rand((2, 16, 1, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((2, 16, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("h", [500])
@pytest.mark.parametrize("w", [512])
def test_expand_and_broadcast(device, h, w):
    torch_input_tensor_a = torch.rand((1, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_add_with_broadcast_on_batch(device, h, w):
    torch_input_tensor_a = torch.rand((1, 16, 1, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((2, 16, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("shape", [(8, 16, 384, 384)])
@pytest.mark.parametrize("scalar", [0.125])
def test_add_attention_scores_to_scalar(device, shape, scalar):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor + scalar

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_tensor = ttnn.add(input_tensor, scalar, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
    assert output_tensor.shape == shape


@pytest.mark.parametrize("shape_a", [(8, 16, 128, 128)])
@pytest.mark.parametrize("shape_b", [(1, 16, 128, 128)])
def test_add_with_batch_broadcast(device, shape_a, shape_b):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
    assert output_tensor.shape == shape_a


@pytest.mark.parametrize("shape_a", [(4096, 4096)])
@pytest.mark.parametrize("shape_b", [(1, 4096)])
def test_add_dram_and_l1_tensor(device, shape_a, shape_b):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
    assert output_tensor.shape == shape_a


@pytest.mark.parametrize("shape", [(1, 1, 32, 32)])
@pytest.mark.parametrize("activations", [None, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]])
def test_add_and_apply_activations(device, shape, activations):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b
    if activations is not None:
        for activation in activations:
            if activation == "relu":
                torch_output_tensor = torch.relu(torch_output_tensor)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, activations=activations)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.99988)
    assert output_tensor.shape == shape


@pytest.mark.parametrize("shape", [(1, 1, 32, 32)])
@pytest.mark.parametrize("activations", [None, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]])
def test_in_place_add_and_apply_activations(device, shape, activations):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b
    if activations is not None:
        for activation in activations:
            if activation == "relu":
                torch_output_tensor = torch.relu(torch_output_tensor)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.add_(input_tensor_a, input_tensor_b, activations=activations)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.99988)
    assert output_tensor.shape == shape


@pytest.mark.parametrize("shape", [(32, 32)])
def test_prim_add(device, shape):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.prim.binary(input_tensor_a, input_tensor_b, ttnn.BinaryOpType.ADD, output_tensor=input_tensor_a)
    output_tensor = ttnn.to_torch(input_tensor_a)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.99988)
    assert output_tensor.shape == shape


@pytest.mark.skip(reason="#11002/#4005: Bcast does not appear to be doing what we expect.  Leaving test for reference.")
@pytest.mark.parametrize("shape_a", [(1, 1, 8192, 320)])
@pytest.mark.parametrize("shape_b", [(2, 1, 1, 320)])
def test_add_with_different_batch(device, shape_a, shape_b):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    device_grid_size = ttnn.CoreGrid(y=compute_with_storage_grid_size.y, x=compute_with_storage_grid_size.x)

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(1024, 64),
        core_grid=device_grid_size,  # ttnn.CoreGrid(y=8, x=5),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_tensor_a = ttnn.to_memory_config(input_tensor_a, block_sharded_mem_config)

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    # Intended to swap code below with: output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # print("here!!!!!!!!!!!!!!!!!!!!!!!")
    # output_tensor = ttnn.bcast(
    #     input_tensor_a,
    #     input_tensor_b,
    #     ttnn.BcastOpMath.ADD,
    #     ttnn.BcastOpDim.H,
    #     memory_config=input_tensor_a.memory_config(),
    # )
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    output_tensor = ttnn.to_torch(output_tensor)

    # We do not support broadcasting as one would expect,
    # our bcast will return a tensor without the batch 2
    # we also get incorrect pcc as well
    torch_output_tensor = torch_output_tensor[:1]

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
    assert output_tensor.shape == shape_a
