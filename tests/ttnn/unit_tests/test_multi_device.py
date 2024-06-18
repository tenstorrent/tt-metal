# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import ttnn
import tempfile
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc


from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor, ListMeshToTensor


#######
# Test MultiDevice Initialization, Open/Close
#######
def test_device_mesh_open_close_explicit(silicon_arch_name, silicon_arch_wormhole_b0):
    """Manually open and close multi-device"""
    num_pcie_devices = ttnn.get_num_pcie_devices()
    if num_pcie_devices <= 1:
        pytest.skip("Requires multiple devices to run")

    device_grid, device_ids = ttnn.DeviceGrid(2, 2), ttnn.get_pcie_device_ids()
    multi_device = ttnn.open_device_mesh(device_grid, device_ids)
    ttnn.close_device_mesh(multi_device)


def test_multi_device_subset_mesh(silicon_arch_name, silicon_arch_wormhole_b0):
    """Manually open and close multi-device"""
    num_pcie_devices = ttnn.get_num_pcie_devices()
    if num_pcie_devices <= 1:
        pytest.skip("Requires multiple devices to run")

    device_grid, device_ids = ttnn.DeviceGrid(1, 2), ttnn.get_pcie_device_ids()
    multi_device = ttnn.open_device_mesh(device_grid, device_ids)
    assert multi_device.get_num_devices() == 2
    ttnn.close_device_mesh(multi_device)

    multi_device = ttnn.open_device_mesh(device_grid, device_ids)
    assert multi_device.get_num_devices() == 2
    ttnn.close_device_mesh(multi_device)


def test_multi_device_open_close_full_device_mesh_fixture(device_mesh):
    """Using `device_mesh` pytest fixture defined in conftest.py"""
    pass


def test_multi_device_open_close_full_device_mesh_fixture(device_mesh):
    """Using `device_mesh` pytest fixture defined in conftest.py"""
    pass


def test_multi_device_open_close_using_context_manager(silicon_arch_name, silicon_arch_wormhole_b0):
    """Using context manager to open and close multi-device"""
    pytest.skip("Issue #6983")
    device_grid, device_ids = ttnn.DeviceGrid(2, 2), ttnn.get_device_ids()
    if len(device_ids) <= 1:
        pytest.skip()
    with ttnn.create_device_mesh(device_grid, device_ids) as device_mesh:
        # Do something with multi_device
        pass


def test_multi_device_open_close_galaxy_mesh(silicon_arch_name, silicon_arch_wormhole_b0):
    if ttnn.get_num_devices() < 32:
        pytest.skip("Test is only valid on Galaxy")

    """Manually open and close multi-device"""
    device_grid, device_ids = ttnn.DeviceGrid(1, 4), ttnn.get_device_ids()
    multi_device = ttnn.open_device_mesh(device_grid, device_ids)
    assert multi_device.get_num_devices() == 4
    ttnn.close_device_mesh(multi_device)

    device_grid, device_ids = ttnn.DeviceGrid(8, 1), ttnn.get_device_ids()
    multi_device = ttnn.open_device_mesh(device_grid, device_ids)
    assert multi_device.get_num_devices() == 8
    ttnn.close_device_mesh(multi_device)

    device_grid, device_ids = ttnn.DeviceGrid(8, 4), ttnn.get_device_ids()
    multi_device = ttnn.open_device_mesh(device_grid, device_ids)
    assert multi_device.get_num_devices() == 32
    ttnn.close_device_mesh(multi_device)

    device_grid = ttnn.DeviceGrid(3, 2)
    multi_device = ttnn.open_device_mesh(device_grid, device_ids)
    assert multi_device.get_num_devices() == 6
    ttnn.close_device_mesh(multi_device)


#######
# Simple Multi-Device Tensor tests
#######


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b])
def test_ttnn_to_multi_device_multiple_times(device_mesh, layout, memory_config, dtype):
    """Test ttnn.to_device(..) works when the tensor is already on device"""
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Unsupported test permutation: bfloat8_b with ROW_MAJOR_LAYOUT")

    torch_tensor = torch.rand((1, 1, 32, 32 * device_mesh.get_num_devices()), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(
        torch_tensor, dtype=dtype, layout=layout, mesh_mapper=ShardTensorToMesh(device_mesh, dim=3)
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, device_mesh, memory_config=memory_config)
    ttnn_tensor = ttnn.to_device(ttnn_tensor, device_mesh, memory_config=memory_config)
    ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
    torch_loop_back_tensor = ttnn.to_torch(ttnn_loop_back_tensor, mesh_composer=ConcatMeshToTensor(device_mesh, dim=3))

    assert_with_pcc(torch_tensor, torch_loop_back_tensor, pcc=0.9999)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_ttnn_to_and_from_multi_device_shard(device_mesh, layout, memory_config, dtype):
    """Shard a tensor across devices, compose it back and verify loopback tensor is same as the original tensor"""
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Unsupported test permutation: bfloat8_b with ROW_MAJOR_LAYOUT")

    torch_tensor = torch.rand((1, 1, 32, 32 * device_mesh.get_num_devices()), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(
        torch_tensor, dtype=dtype, layout=layout, mesh_mapper=ShardTensorToMesh(device_mesh, dim=3)
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, device_mesh, memory_config=memory_config)
    ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
    torch_loop_back_tensor = ttnn.to_torch(ttnn_loop_back_tensor, mesh_composer=ConcatMeshToTensor(device_mesh, dim=3))

    assert_with_pcc(torch_tensor, torch_loop_back_tensor, pcc=0.9999)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_multi_device_check_per_device_shard(device_mesh, layout, memory_config, dtype):
    """This test checks if the tensor is correctly sharded across devices"""
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Unsupported test permutation: bfloat8_b with ROW_MAJOR_LAYOUT")

    torch_tensor = torch.rand((1, 1, 32, 64 * device_mesh.get_num_devices()), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(
        torch_tensor, dtype=dtype, mesh_mapper=ShardTensorToMesh(device_mesh, dim=3), layout=layout
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, device_mesh, memory_config=memory_config)
    ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)

    shard_offset, shard_size = 0, 64
    for device_tensor in ttnn.get_device_tensors(ttnn_loop_back_tensor):
        device_tensor_torch = ttnn.to_torch(device_tensor)
        assert_with_pcc(device_tensor_torch, torch_tensor[..., shard_offset : shard_offset + shard_size], pcc=0.9999)
        shard_offset += shard_size


@pytest.mark.parametrize("shape", [(1, 1, 32, 128), (1, 1, 16, 32)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_multi_device_replicate(device_mesh, shape, layout, memory_config):
    """Test ReplicateTensorToMesh to broadcast a tensor across multiple devices"""
    from ttnn import ReplicateTensorToMesh, ListMeshToTensor

    full_tensor = torch.rand(shape, dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(
        full_tensor,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
        layout=layout,
        memory_config=memory_config,
        device=device_mesh,
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, device_mesh)
    ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
    loopback_replicated_tensors = ttnn.to_torch(ttnn_loop_back_tensor, mesh_composer=ListMeshToTensor(device_mesh))
    for loopback_replicated_tensor in loopback_replicated_tensors:
        assert torch.all(full_tensor == loopback_replicated_tensor)


def test_ttnn_multi_device_all_gather(pcie_device_mesh):
    """Multidevice API test for ttnn.all_gather CCL operation"""
    if pcie_device_mesh.get_num_devices() <= 1:
        pytest.skip("Requires multiple devices to run")
    full_tensor = torch.rand((1, 1, 32, 32 * pcie_device_mesh.get_num_devices()), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(full_tensor, mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=3))
    ttnn_tensor = ttnn.to_device(ttnn_tensor, pcie_device_mesh)
    ttnn_tensor = ttnn.all_gather(ttnn_tensor, dim=3, num_links=1)

    device_tensors: typing.List[ttnn.Tensor] = ttnn.get_device_tensors(ttnn_tensor)
    for device_tensor in device_tensors:
        device_tensor_torch = ttnn.to_torch(device_tensor)
        assert torch.all(device_tensor_torch == full_tensor)


def test_multi_device_single_op_unary(device_mesh):
    """Multidevice API test: Running tensor-parallel multi-device single-op unary"""
    torch_input_tensor = torch.rand((1, 1, 32, 32 * device_mesh.get_num_devices()), dtype=torch.bfloat16)
    torch_output_golden = torch.nn.functional.gelu(torch_input_tensor)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ShardTensorToMesh(device_mesh, dim=3),
        device=device_mesh,
    )
    ttnn_output_tensor = ttnn.gelu(ttnn_input_tensor)

    ttnn_torch_output_tensor = ttnn.to_torch(ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(device_mesh, dim=3))
    assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.999)


def test_multi_device_single_op_binary(device_mesh):
    """Multidevice API test: Running tensor-parallel multi-device single-op binary"""
    torch_input_a_tensor = torch.rand((1, 1, 32, 32 * device_mesh.get_num_devices()), dtype=torch.bfloat16)
    torch_input_b_tensor = torch.rand((1, 1, 32, 32 * device_mesh.get_num_devices()), dtype=torch.bfloat16)
    torch_output_golden = torch_input_a_tensor + torch_input_b_tensor

    ttnn_input_a_tensor = ttnn.from_torch(
        torch_input_a_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ShardTensorToMesh(device_mesh, dim=3),
    )
    ttnn_input_b_tensor = ttnn.from_torch(
        torch_input_b_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ShardTensorToMesh(device_mesh, dim=3),
    )
    ttnn_output_tensor = ttnn.add(ttnn_input_a_tensor, ttnn_input_b_tensor)

    ttnn_torch_output_tensor = ttnn.to_torch(ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(device_mesh, dim=3))
    assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.999)


def test_multi_device_multi_op(device_mesh):
    """Multidevice API test: Running tensor-parallel multi-device multi-op"""
    torch_input_tensor = torch.rand((1, 1, 32, 32 * device_mesh.get_num_devices()), dtype=torch.bfloat16)
    torch_output_golden = torch.nn.functional.gelu(torch_input_tensor)
    torch_output_golden = torch.exp(torch_output_golden)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ShardTensorToMesh(device_mesh, dim=3),
        device=device_mesh,
    )
    ttnn_gelu_output = ttnn.gelu(ttnn_input_tensor)
    ttnn_output_tensor = ttnn.exp(ttnn_gelu_output)

    ttnn_torch_output_tensor = ttnn.to_torch(ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(device_mesh, dim=3))
    assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.999)


def test_multi_device_data_parallel_matmul_op(device_mesh):
    """Multidevice API: Data Parallel on matmul"""
    torch_input_a_tensor = torch.rand((device_mesh.get_num_devices(), 1, 32, 32), dtype=torch.bfloat16)
    torch_input_b_tensor = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
    torch_output_golden = torch_input_a_tensor @ torch_input_b_tensor

    ttnn_input_a_tensor = ttnn.from_torch(
        torch_input_a_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ShardTensorToMesh(device_mesh, dim=0),
    )
    ttnn_input_b_tensor = ttnn.from_torch(
        torch_input_b_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    ttnn_output_tensor = ttnn_input_a_tensor @ ttnn_input_b_tensor

    ttnn_torch_output_tensor = ttnn.to_torch(ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(device_mesh, dim=0))
    assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.993)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_multi_device_as_tensor_api(device_mesh, layout, memory_config, dtype):
    """Multidevice API: Data Parallel on matmul using cached tensor"""
    torch.manual_seed(0)
    torch_input_a_tensor = torch.rand((device_mesh.get_num_devices(), 1, 32, 32), dtype=torch.bfloat16)
    torch_input_b_tensor = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
    torch_output_golden = torch_input_a_tensor @ torch_input_b_tensor

    ttnn_input_a_tensor = ttnn.as_tensor(
        torch_input_a_tensor,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
        device=device_mesh,
        mesh_mapper=ShardTensorToMesh(device_mesh, dim=0),
    )

    with tempfile.NamedTemporaryFile() as temp_file:
        save_tensor = ttnn.as_tensor(
            torch_input_b_tensor,
            dtype=dtype,
            layout=layout,
            device=device_mesh,
            memory_config=memory_config,
            cache_file_name=f"{temp_file.name}.weight",
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )

        ttnn_input_b_tensor = ttnn.as_tensor(
            torch_input_b_tensor,
            dtype=dtype,
            layout=layout,
            device=device_mesh,
            memory_config=memory_config,
            cache_file_name=f"{temp_file.name}.weight",
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )

        ttnn_output_tensor = ttnn_input_a_tensor @ ttnn_input_b_tensor

        ttnn_torch_output_tensor = ttnn.to_torch(
            ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(device_mesh, dim=0)
        )
        if dtype == ttnn.bfloat4_b:
            assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.87)
        else:
            assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.991)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_multi_device_as_tensor_api_sharded_tensor(device_mesh, layout, memory_config, dtype):
    """Multidevice API: Data Parallel on matmul using cached tensor"""
    input_tensor = torch.rand((device_mesh.get_num_devices(), 1, 32, 32), dtype=torch.bfloat16)

    with tempfile.NamedTemporaryFile() as temp_file:
        save_tensor = ttnn.as_tensor(
            input_tensor,
            dtype=dtype,
            layout=layout,
            device=device_mesh,
            memory_config=memory_config,
            cache_file_name=f"{temp_file.name}.weight",
            mesh_mapper=ShardTensorToMesh(device_mesh, dim=0),
        )
        load_tensor = ttnn.as_tensor(
            input_tensor,
            dtype=dtype,
            layout=layout,
            device=device_mesh,
            memory_config=memory_config,
            cache_file_name=f"{temp_file.name}.weight",
            mesh_mapper=ShardTensorToMesh(device_mesh, dim=0),
        )
        torch_loaded_tensor = ttnn.to_torch(load_tensor, mesh_composer=ConcatMeshToTensor(device_mesh, dim=0))
        expected_pcc = 0.98 if dtype == ttnn.bfloat4_b else 0.99
        assert_with_pcc(input_tensor, torch_loaded_tensor, pcc=expected_pcc)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b])
def test_multi_device_permute(device_mesh, layout, memory_config, dtype):
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Unsupported test permutation: bfloat8_b with ROW_MAJOR_LAYOUT")

    torch_tensor = torch.rand((32, 1, 160, 64 * device_mesh.get_num_devices()), dtype=torch.bfloat16)
    torch_golden = torch.permute(torch_tensor, (0, 1, 3, 2))

    ttnn_tensor = ttnn.from_torch(
        torch_tensor, dtype=dtype, layout=layout, mesh_mapper=ShardTensorToMesh(device_mesh, dim=3)
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, device_mesh, memory_config=memory_config)
    ttnn_permute = ttnn.permute(ttnn_tensor, (0, 1, 3, 2))
    ttnn_loop_back_tensor = ttnn.from_device(ttnn_permute)
    torch_loop_back_tensor = ttnn.to_torch(ttnn_loop_back_tensor, mesh_composer=ConcatMeshToTensor(device_mesh, dim=2))

    assert_with_pcc(torch_golden, torch_loop_back_tensor, pcc=0.9999)


def test_max(device_mesh):
    gate_logits_1SB8 = ttnn.from_torch(
        torch.randn(1, 1, 32, 8),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    gate_logits_1SB8 = ttnn.to_device(gate_logits_1SB8, device_mesh)
    weights_ex0_1SB1 = ttnn.max(gate_logits_1SB8, dim=3)
    print(weights_ex0_1SB1)


def test_ttnn_multi_device_all_gather_all_devices(t3k_device_mesh):
    """Multidevice API test for ttnn.all_gather CCL operation for full 8-device T3K"""
    if t3k_device_mesh.get_num_devices() < 8:
        pytest.skip()

    full_tensor = torch.ones((1, 1, 32, 32 * t3k_device_mesh.get_num_devices()), dtype=torch.bfloat16)
    for i in range(t3k_device_mesh.get_num_devices()):
        full_tensor[..., i * 32 : (i + 1) * 32] = i

    ttnn_tensor = ttnn.from_torch(full_tensor, mesh_mapper=ShardTensorToMesh(t3k_device_mesh, dim=3))
    ttnn_tensor = ttnn.to_device(ttnn_tensor, t3k_device_mesh)
    ttnn_tensor = ttnn.all_gather(ttnn_tensor, dim=3, num_links=1)

    device_tensors: typing.List[ttnn.Tensor] = ttnn.get_device_tensors(ttnn_tensor)
    for device_tensor in device_tensors:
        device_tensor_torch = ttnn.to_torch(device_tensor)
        assert torch.all(device_tensor_torch == full_tensor)


def test_sharded_matmul(t3k_device_mesh):
    q_heads_1B4D = ttnn.from_torch(
        torch.randn(1, 32, 32, 128),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=t3k_device_mesh,
        mesh_mapper=ReplicateTensorToMesh(t3k_device_mesh),
    )
    keys_1BDP = ttnn.from_torch(
        torch.randn(1, 32, 128, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=t3k_device_mesh,
        mesh_mapper=ReplicateTensorToMesh(t3k_device_mesh),
    )

    q_heads_1B4D = ttnn.to_device(q_heads_1B4D, t3k_device_mesh)
    keys_1BDP = ttnn.to_device(keys_1BDP, t3k_device_mesh)

    q_heads_1B4D = ttnn.to_memory_config(
        q_heads_1B4D,
        ttnn.create_sharded_memory_config(
            shape=(32, 128),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    keys_1BDP = ttnn.to_memory_config(
        keys_1BDP,
        ttnn.create_sharded_memory_config(
            shape=(128, 32),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    compute_kernel_attn = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    attn_1B4P = ttnn.matmul(
        q_heads_1B4D,
        keys_1BDP,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=4, x=8),
        compute_kernel_config=compute_kernel_attn,
    )

    print(attn_1B4P)


def test_4b_tensor(device_mesh):
    tensor = ttnn.from_torch(
        torch.randn(1, 1, 32, 32),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    tensor = ttnn.to_device(tensor, device_mesh)
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    x = ttnn.to_device(x, device_mesh)
    tensor = ttnn.matmul(
        x,
        tensor,
        core_grid=ttnn.CoreGrid(y=4, x=8),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
        ),
        use_1d_systolic_array=True,
    )


def test_slicing(device_mesh):
    tensor = ttnn.from_torch(
        torch.randn(1, 32, 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    tensor = ttnn.to_device(tensor, device_mesh)
    tensor = tensor[:, :, :, :1]
    assert all([device_tensor.shape == tensor.shape for device_tensor in ttnn.get_device_tensors(tensor)])


def test_clone(device_mesh):
    results_11BH = ttnn.from_torch(
        torch.randn(1, 1, 32, 128),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    results_11BH = ttnn.to_device(results_11BH, device_mesh)
    results_11BH = ttnn.clone(results_11BH, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
    print(results_11BH)


def test_device_shard_to_torch(device_mesh):
    """Test `ttnn.get_device_tensor(..) API"""
    torch_input_tensor = torch.rand((1, 1, 32, 32 * device_mesh.get_num_devices()), dtype=torch.bfloat16)
    torch_output_golden = torch.nn.functional.gelu(torch_input_tensor)
    torch_output_golden = torch.exp(torch_output_golden)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ShardTensorToMesh(device_mesh, dim=3),
        device=device_mesh,
    )

    ttnn_gelu_output = ttnn.gelu(ttnn_input_tensor)
    ttnn_output_tensor = ttnn.exp(ttnn_gelu_output)

    # Skip the compose/torch.cat call entirely
    for i, device in enumerate(device_mesh.get_devices()):
        device_tensor = ttnn.get_device_tensor(ttnn_output_tensor, device)
        torch_device_tensor = ttnn.to_torch(device_tensor)
        assert_with_pcc(torch_device_tensor, torch_output_golden[..., i * 32 : (i + 1) * 32], pcc=0.999)


@pytest.mark.parametrize("height", [7])
@pytest.mark.parametrize("width", [3])
def test_validate_as_tensor(tmp_path, device_mesh, height, width):
    torch_input_tensor = torch.rand((height, width), dtype=torch.float32)

    memory_config = ttnn.L1_MEMORY_CONFIG
    tensor = ttnn.as_tensor(
        torch_input_tensor,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device_mesh),
        cache_file_name=tmp_path / "cache_file",
    )
    assert tensor.dtype == ttnn.float32
    assert tensor.devices() == device_mesh.get_devices()
    assert tensor.layout == ttnn.TILE_LAYOUT
    assert ttnn.get_memory_config(tensor) == memory_config

    tensor = ttnn.as_tensor(
        torch_input_tensor,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device_mesh),
        cache_file_name=tmp_path / "cache_file",
    )
    assert tensor.dtype == ttnn.float32
    assert tensor.devices() == device_mesh.get_devices()
    assert tensor.layout == ttnn.TILE_LAYOUT
    assert ttnn.get_memory_config(tensor) == memory_config

    for device in device_mesh.get_devices():
        device_tensor = ttnn.get_device_tensor(tensor, device)
        assert torch.allclose(ttnn.to_torch(device_tensor), torch_input_tensor)
