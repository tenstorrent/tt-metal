# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import time

torch.set_printoptions(sci_mode=False)

import ttnn
import tracy
from tests.ttnn.utils_for_testing import assert_with_pcc

valid_tile_heights = [1, 2, 4, 8, 16, 32]


def get_tile_height(m):
    for tile_height in valid_tile_heights:
        if m <= tile_height:
            return tile_height
    return 32


@pytest.mark.parametrize(
    "m, k, n",
    [
        (1, 1024, 4096),
        (4, 1024, 4096),
        (8, 1024, 4096),
        (16, 1024, 4096),
        (32, 1024, 4096),
        # DENOISE
        (64, 1024, 4096),
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

    tile_height = get_tile_height(m)
    inputA_tile_size = ttnn.Tile((tile_height, 32))
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
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        tile=inputA_tile_size,
        device=device,
        memory_config=in0_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in1_memory_config
    )
    tracy.signpost(f"MatmulDecode: m: {m} k: {k} n: {n}")
    for x in range(2):
        output_tensor = ttnn.matmul_decode(input_tensor_a, input_tensor_b)

    assert output_tensor.shape == (m, n)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.99)


@pytest.mark.parametrize(
    "m, k, n, k_blocks, n_blocks",
    [
        (32, 4096, 1024, 4, 32),
        (64, 4096, 1024, 4, 32),
        # (32, 64, 256, 2, 8),
    ],
)
@pytest.mark.parametrize(
    "num_inputA_cores",
    [
        (2),
    ],
)
def test_matmul_decode_partial_width_sharded(device, m, k, n, k_blocks, n_blocks, num_inputA_cores):
    torch.manual_seed(time.time())
    kc = k // k_blocks
    nc = n // n_blocks
    num_inputB_cores = k_blocks * n_blocks
    print(
        f"num_inputA_cores: {num_inputA_cores}, num_inputB_cores: {num_inputB_cores}, "
        f"kc: {kc}, nc: {nc}, k_blocks: {k_blocks}, n_blocks: {n_blocks}"
    )

    # torch_input_tensor_a = torch.tensor([(x ) for x in range(m)], dtype=torch.bfloat16).reshape(m, 1).expand(m, k).contiguous()
    # torch_input_tensor_a += torch.tensor([(x ) for x in range(k)], dtype=torch.bfloat16).reshape(1, k).expand(m, k).contiguous()

    torch_input_tensor_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k, n), dtype=torch.bfloat16)
    # torch_input_tensor_b = torch.tensor([x%32 for x in range(n)], dtype=torch.bfloat16).reshape(1, n).expand(k, n).contiguous() /32
    # torch_input_tensor_b = torch.tensor([x%32 for x in range(k)], dtype=torch.bfloat16).reshape(k, 1).expand(k, n).contiguous() /32

    ref = torch_input_tensor_a.to(torch.float32) @ torch_input_tensor_b.to(torch.float32)

    # Reshape + permute B so that a width-sharded tensor distributes a 2D (K x N)
    # block grid across cores: core c (row-major) holds B[kb*kc:(kb+1)*kc, nb*nc:(nb+1)*nc]
    # with c = kb * n_blocks + nb.
    torch_input_tensor_b_reshaped = torch_input_tensor_b.reshape(k_blocks, kc, n)
    torch_input_tensor_b_reshaped = torch.permute(torch_input_tensor_b_reshaped, (1, 0, 2))
    print("torch_input_tensor_b_reshaped.shape:", torch_input_tensor_b_reshaped.shape)
    torch_input_tensor_b_reshaped = torch_input_tensor_b_reshaped.reshape(kc, n * k_blocks)

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
        dtype=ttnn.bfloat16,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b_reshaped,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_memory_config,
        dtype=ttnn.bfloat16,
    )
    print("input_tensor_a.shape:", input_tensor_a.shape)
    print("input_tensor_b.shape:", input_tensor_b.shape)
    for x in range(10):
        output_tensor = ttnn.matmul_decode(input_tensor_a, input_tensor_b, partial_width_sharded=True)

    assert output_tensor.shape == (m, n)

    out = ttnn.to_torch(output_tensor).float()
    print("ref:", ref)
    print("out:", out)
    # assert torch.allclose(ref, out, atol=0.2, rtol=0.15)
    assert_with_pcc(ref, out, 0.99)
