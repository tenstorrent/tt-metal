# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
import transformers


#######
# Multi-Device Tensor tests running in async mode
#######


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_ttnn_to_and_from_multi_device_shard(pcie_mesh_device, layout, memory_config, dtype):
    """Shard a tensor across devices, compose it back and verify loopback tensor is same as the original tensor"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor

    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Unsupported test permutation: bfloat8_b with ROW_MAJOR_LAYOUT")

    pcie_mesh_device.enable_async(True)

    for i in range(100):
        torch_tensor = torch.rand((1, 1, 256, 512), dtype=torch.bfloat16)
        ttnn_tensor = ttnn.from_torch(
            torch_tensor, dtype=dtype, layout=layout, mesh_mapper=ShardTensorToMesh(pcie_mesh_device, dim=3)
        )
        ttnn_tensor = ttnn.to_device(ttnn_tensor, pcie_mesh_device, memory_config=memory_config)
        ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
        torch_loop_back_tensor = ttnn.to_torch(
            ttnn_loop_back_tensor, mesh_composer=ConcatMeshToTensor(pcie_mesh_device, dim=3)
        )
        assert_with_pcc(torch_tensor, torch_loop_back_tensor, pcc=0.9999)

    pcie_mesh_device.enable_async(False)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_multi_device_check_per_device_shard(pcie_mesh_device, layout, memory_config, dtype):
    """This test checks if the tensor is correctly sharded across devices"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor

    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Unsupported test permutation: bfloat8_b with ROW_MAJOR_LAYOUT")

    pcie_mesh_device.enable_async(True)

    num_loops = 50
    if dtype == ttnn.bfloat8_b:
        # On host bfloat8_b conversion is slow. Decrease num loops.
        num_loops = 10
    for i in range(num_loops):
        torch_tensor = torch.rand((8, 1, 1024, 1024), dtype=torch.bfloat16)

        ttnn_tensor = ttnn.from_torch(
            torch_tensor, dtype=dtype, layout=layout, mesh_mapper=ShardTensorToMesh(pcie_mesh_device, dim=3)
        )
        ttnn_tensor = ttnn.to_device(ttnn_tensor, pcie_mesh_device, memory_config=memory_config)
        ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)

        shard_offset, shard_size = 0, int(1024 / len(pcie_mesh_device.get_device_ids()))
        for device_tensor in ttnn.get_device_tensors(ttnn_loop_back_tensor):
            device_tensor_torch = ttnn.to_torch(device_tensor)
            assert_with_pcc(
                device_tensor_torch, torch_tensor[..., shard_offset : shard_offset + shard_size], pcc=0.9999
            )
            shard_offset += shard_size

    pcie_mesh_device.enable_async(False)


@pytest.mark.parametrize("shape", [(1, 1, 512, 512), (1, 1, 1040, 1040)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_multi_device_replicate(pcie_mesh_device, shape, layout, memory_config):
    """Test ReplicateTensorToMesh to broadcast a tensor across multiple devices"""
    from ttnn import ReplicateTensorToMesh, ListMeshToTensor

    pcie_mesh_device.enable_async(True)

    for i in range(100):
        full_tensor = torch.rand(shape, dtype=torch.bfloat16)

        ttnn_tensor = ttnn.from_torch(
            full_tensor,
            mesh_mapper=ReplicateTensorToMesh(pcie_mesh_device),
            layout=layout,
            memory_config=memory_config,
            device=pcie_mesh_device,
        )
        ttnn_tensor = ttnn.to_device(ttnn_tensor, pcie_mesh_device)
        ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
        loopback_replicated_tensors = ttnn.to_torch(
            ttnn_loop_back_tensor, mesh_composer=ListMeshToTensor(pcie_mesh_device)
        )
        for loopback_replicated_tensor in loopback_replicated_tensors:
            assert torch.all(full_tensor == loopback_replicated_tensor)

    pcie_mesh_device.enable_async(False)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b])
def test_ttnn_to_multi_device_tilized_parallel(pcie_mesh_device, layout, memory_config, dtype):
    """Test multi chip layout conversions on worker threads"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor, ListMeshToTensor

    shard_dim = 3
    pcie_mesh_device.enable_async(True)
    for loop in range(20):
        torch_tensor = torch.rand((8, 1, 1024, 1024), dtype=torch.bfloat16)
        ttnn_tensor = ttnn.from_torch(
            torch_tensor,
            mesh_mapper=ShardTensorToMesh(pcie_mesh_device, dim=shard_dim),
            layout=layout,
            memory_config=memory_config,
            device=pcie_mesh_device,
        )
        if loop < 10:
            # Test Concat Composer
            readback_tensor = ttnn.to_torch(
                ttnn_tensor, mesh_composer=ConcatMeshToTensor(pcie_mesh_device, dim=shard_dim), device=pcie_mesh_device
            )
        else:
            # Test Mesh Composer
            readback_tensors = ttnn.to_torch(
                ttnn_tensor, mesh_composer=ListMeshToTensor(pcie_mesh_device), device=pcie_mesh_device
            )
            readback_tensor = torch.cat(readback_tensors, dim=shard_dim)
        assert torch.all(readback_tensor == torch_tensor)
    pcie_mesh_device.enable_async(False)


@pytest.mark.parametrize("program_cache", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 512, 512), (1, 3, 1024, 1024)])
def test_multi_device_unary_binary_op_chain(pcie_mesh_device, program_cache, shape):
    """Multidevice API test: Running tensor-parallel multi-device chain of eltwise ops"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor

    pcie_mesh_device.enable_async(True)
    for device in pcie_mesh_device.get_device_ids():
        if program_cache:
            pcie_mesh_device.get_device(device).enable_program_cache()

    torch_silu = torch.nn.SiLU()
    for i in range(50):
        torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
        torch_output_golden = torch.add(
            torch.subtract(
                torch.exp(torch.nn.functional.relu(torch.nn.functional.gelu(torch_input_tensor))),
                torch.exp(torch_input_tensor),
            ),
            torch_silu(torch_input_tensor),
        )

        ttnn_input_tensor = ttnn.from_torch(
            torch_input_tensor,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ShardTensorToMesh(pcie_mesh_device, dim=3),
            device=pcie_mesh_device,
        )
        ttnn_output_tensor = ttnn.add(
            ttnn.sub(ttnn.exp(ttnn.relu(ttnn.gelu(ttnn_input_tensor))), ttnn.exp(ttnn_input_tensor)),
            ttnn.silu(ttnn_input_tensor),
        )
        ttnn_output_tensor = ttnn.from_device(ttnn_output_tensor)
        ttnn_torch_output_tensor = ttnn.to_torch(
            ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(pcie_mesh_device, dim=3)
        )
        assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.98)

    pcie_mesh_device.enable_async(False)


@pytest.mark.parametrize("program_cache", [False, True])
@pytest.mark.parametrize("input_a_shape", [(4, 1, 512, 512), (16, 1, 512, 512)])
def test_multi_device_data_parallel_op_chain(pcie_mesh_device, program_cache, input_a_shape):
    """Multidevice API: Running data-parallel chain of ops with matmul"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor, ReplicateTensorToMesh

    pcie_mesh_device.enable_async(True)
    for device in pcie_mesh_device.get_device_ids():
        if program_cache:
            pcie_mesh_device.get_device(device).enable_program_cache()

    torch_silu = torch.nn.SiLU()
    torch_mish = torch.nn.Mish()
    for i in range(5):
        torch_input_a_tensor = torch.rand(input_a_shape, dtype=torch.bfloat16)
        torch_input_b_tensor = torch.rand((1, 1, 512, 512), dtype=torch.bfloat16)
        torch_output_golden = torch_mish(
            torch_silu(
                torch.nn.functional.relu(torch.nn.functional.gelu(torch_input_a_tensor @ torch_input_b_tensor))
                @ torch.exp(torch_input_a_tensor)
            )
        )

        ttnn_input_a_tensor = ttnn.from_torch(
            torch_input_a_tensor,
            layout=ttnn.TILE_LAYOUT,
            device=pcie_mesh_device,
            mesh_mapper=ShardTensorToMesh(pcie_mesh_device, dim=0),
        )
        ttnn_input_b_tensor = ttnn.from_torch(
            torch_input_b_tensor,
            layout=ttnn.TILE_LAYOUT,
            device=pcie_mesh_device,
            mesh_mapper=ReplicateTensorToMesh(pcie_mesh_device),
        )
        ttnn_output_tensor = ttnn.from_device(
            ttnn.mish(
                ttnn.silu(
                    ttnn.relu(ttnn.gelu(ttnn_input_a_tensor @ ttnn_input_b_tensor)) @ ttnn.exp(ttnn_input_a_tensor)
                )
            )
        )
        ttnn_torch_output_tensor = ttnn.to_torch(
            ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(pcie_mesh_device, dim=0)
        )
        assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.97)

    pcie_mesh_device.enable_async(False)


@pytest.mark.parametrize(
    "layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_multi_device_argmax(pcie_mesh_device, layout, mem_config):
    pcie_mesh_device.enable_async(True)

    torch.manual_seed(0)
    torch_input = torch.randn(1, 1, 32, 4096)
    reference_output = torch_input.squeeze(1).view(32, 1, -1).float().squeeze().argmax(axis=-1)

    tt_out_11BH = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=pcie_mesh_device,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(pcie_mesh_device),
    )

    tt_out_11BH = ttnn.argmax(tt_out_11BH, dim=-1)
    tt_out_1B = ttnn.reshape(tt_out_11BH[:1, :, :, :], ttnn.Shape([1, 32]))
    tt_out_1B = ttnn.to_torch(tt_out_1B, mesh_composer=ttnn.ConcatMeshToTensor(pcie_mesh_device, dim=0))[0]

    assert_with_pcc(tt_out_1B, reference_output, pcc=0.97)

    pcie_mesh_device.enable_async(False)


@pytest.mark.parametrize("pcie_mesh_device", [2], indirect=True)
def test_multi_device_explicit_dealloc(pcie_mesh_device):
    """Multidevice API: Ensure that deallocating multi-device tensors works as expected"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor, ReplicateTensorToMesh

    if pcie_mesh_device.get_num_devices() <= 1:
        pytest.skip("Requires multiple devices to run")

    # Create input tensors that cause OOM during op execution
    # Explictly deallocate buffers after each op to ensure we don't run OOM.
    torch_input_a_tensor = torch.rand((512, 1, 2048, 2048), dtype=torch.bfloat16)
    torch_input_b_tensor = torch.rand((1, 1, 2048, 2048), dtype=torch.bfloat16)

    ttnn_input_a_tensor = ttnn.from_torch(
        torch_input_a_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=pcie_mesh_device,
        mesh_mapper=ShardTensorToMesh(pcie_mesh_device, dim=0),
    )
    ttnn_input_b_tensor = ttnn.from_torch(
        torch_input_b_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=pcie_mesh_device,
        mesh_mapper=ReplicateTensorToMesh(pcie_mesh_device),
    )
    ttnn_output_tensor_1 = ttnn_input_a_tensor @ ttnn_input_b_tensor
    ttnn_output_tensor_2 = ttnn.gelu(ttnn_output_tensor_1)
    ttnn_output_tensor_1.deallocate()
    ttnn_input_b_tensor.deallocate()
    ttnn_output_tensor_3 = ttnn.relu(ttnn_output_tensor_2)
    ttnn_output_tensor_2.deallocate()
    ttnn_output_tensor_4 = ttnn_output_tensor_3 @ ttnn_input_a_tensor
    ttnn_output_tensor_3.deallocate()
    ttnn_output_tensor = ttnn.from_device(ttnn_output_tensor_4)
    ttnn_torch_output_tensor = ttnn.to_torch(
        ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(pcie_mesh_device, dim=0)
    )


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("size", [64])
@pytest.mark.parametrize("pcie_mesh_device", [2], indirect=True)
def test_add_1D_tensor_and_scalar(pcie_mesh_device, scalar, size):
    torch.manual_seed(0)

    pcie_mesh_device.enable_async(True)

    torch_input_tensor = torch.rand((size,), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor + scalar

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=pcie_mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(pcie_mesh_device),
    )
    output_tensor = input_tensor + scalar
    output_tensors = ttnn.to_torch(output_tensor, mesh_composer=ttnn.ListMeshToTensor(pcie_mesh_device))
    for output_tensor in output_tensors:
        assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
        assert output_tensor.shape == (1, size)
