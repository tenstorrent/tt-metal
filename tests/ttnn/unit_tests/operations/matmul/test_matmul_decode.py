# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "m, k, n",
    [
        (32, 1024, 4096),
        (32, 4096, 1024),
    ],
)
@pytest.mark.parametrize(
    "num_inputA_cores",
    [
        (2),
    ],
)
def test_matmul_decode(device, m, k, n, num_inputA_cores):
    torch.manual_seed(0)

    num_inputB_cores = n // 32
    print(f"num_inputA_cores: {num_inputA_cores}, num_inputB_cores: {num_inputB_cores}")
    torch_input_tensor_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k, n), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a.to(torch.float32) @ torch_input_tensor_b.to(torch.float32)

    input_a_core_range_set = ttnn.num_cores_to_corerangeset(
        num_inputA_cores, device.compute_with_storage_grid_size(), True
    )
    input_b_core_range_set = ttnn.num_cores_to_corerangeset(
        num_inputB_cores, device.compute_with_storage_grid_size(), True
    )
    in0_memory_config = ttnn.create_sharded_memory_config(
        (m, k // num_inputA_cores),
        core_grid=input_a_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    in1_memory_config = ttnn.create_sharded_memory_config(
        (k, n // num_inputB_cores),
        core_grid=input_b_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_memory_config
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in1_memory_config
    )
    # for x in range(10):
    output_tensor = ttnn.matmul_decode(input_tensor_a, input_tensor_b)

    assert output_tensor.shape == (m, n)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.99)


@pytest.mark.parametrize(
    "m, k, n",
    [
        (32, 4096, 1024),
    ],
)
@pytest.mark.parametrize(
    "k_blocks, n_blocks",
    [
        (4, 32),
    ],
)
@pytest.mark.parametrize(
    "num_inputA_cores",
    [
        (2),
    ],
)
def test_matmul_decode_partial_width_sharded(device, m, k, n, k_blocks, n_blocks, num_inputA_cores):
    torch.manual_seed(0)

    kc = k // k_blocks
    nc = n // n_blocks
    num_inputB_cores = k_blocks * n_blocks
    print(
        f"num_inputA_cores: {num_inputA_cores}, num_inputB_cores: {num_inputB_cores}, "
        f"kc: {kc}, nc: {nc}, k_blocks: {k_blocks}, n_blocks: {n_blocks}"
    )

    torch_input_tensor_a = torch.ones((m, k), dtype=torch.bfloat16) / 4
    torch_input_tensor_b = torch.ones((k, n), dtype=torch.bfloat16) / 32
    ref = torch_input_tensor_a.to(torch.float32) @ torch_input_tensor_b.to(torch.float32)

    # Reshape + permute B so that a width-sharded tensor distributes a 2D (K x N)
    # block grid across cores: core c (row-major) holds B[kb*kc:(kb+1)*kc, nb*nc:(nb+1)*nc]
    # with c = kb * n_blocks + nb.
    torch_input_tensor_b_reshaped = torch_input_tensor_b.reshape(kc, k_blocks * n_blocks * nc)

    input_a_core_range_set = ttnn.num_cores_to_corerangeset(
        num_inputA_cores, device.compute_with_storage_grid_size(), True
    )
    input_b_core_range_set = ttnn.num_cores_to_corerangeset(
        num_inputB_cores, device.compute_with_storage_grid_size(), True
    )
    in0_memory_config = ttnn.create_sharded_memory_config(
        (m, k // num_inputA_cores),
        core_grid=input_a_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    in1_memory_config = ttnn.create_sharded_memory_config(
        (kc, nc),
        core_grid=input_b_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
        dtype=ttnn.bfloat8_b,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b_reshaped, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in1_memory_config
    )
    print("input_tensor_a.shape:", input_tensor_a.shape)
    print("input_tensor_b.shape:", input_tensor_b.shape)
    for x in range(10):
        output_tensor = ttnn.matmul_decode(input_tensor_a, input_tensor_b, partial_width_sharded=True)

    assert output_tensor.shape == (m, n)

    out = ttnn.to_torch(output_tensor)
    assert_with_pcc(ref, out, 0.99)
