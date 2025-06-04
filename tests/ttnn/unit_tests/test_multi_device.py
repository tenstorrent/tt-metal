# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import ttnn
import tempfile
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc


from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor


#######
# Test MultiDevice Initialization, Open/Close
#######
def test_mesh_device_open_close_explicit(silicon_arch_name, silicon_arch_wormhole_b0):
    """Manually open and close multi-device"""
    num_pcie_devices = ttnn.get_num_pcie_devices()
    if num_pcie_devices <= 1:
        pytest.skip("Requires multiple devices to run")

    multi_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
    ttnn.close_mesh_device(multi_device)


def test_multi_device_subset_mesh(silicon_arch_name, silicon_arch_wormhole_b0):
    """Manually open and close multi-device"""
    num_pcie_devices = ttnn.get_num_pcie_devices()
    if num_pcie_devices <= 1:
        pytest.skip("Requires multiple devices to run")

    mesh_shape = ttnn.MeshShape(1, 2)
    multi_device = ttnn.open_mesh_device(mesh_shape)
    assert multi_device.get_num_devices() == 2
    ttnn.close_mesh_device(multi_device)

    multi_device = ttnn.open_mesh_device(mesh_shape)
    assert multi_device.get_num_devices() == 2
    ttnn.close_mesh_device(multi_device)


def test_multi_device_open_close_full_mesh_device_fixture(mesh_device):
    """Using `mesh_device` pytest fixture defined in conftest.py"""
    pass


def test_multi_device_open_close_full_mesh_device_fixture(mesh_device):
    """Using `mesh_device` pytest fixture defined in conftest.py"""
    pass


def test_multi_device_open_close_using_context_manager(silicon_arch_name, silicon_arch_wormhole_b0):
    """Using context manager to open and close multi-device"""
    if ttnn.get_num_devices() < 4:
        pytest.skip()
    mesh_shape = ttnn.MeshShape(2, 2)
    with ttnn.create_mesh_device(mesh_shape) as mesh_device:
        # Do something with multi_device
        pass


def test_multi_device_open_close_galaxy_mesh(silicon_arch_name, silicon_arch_wormhole_b0):
    if ttnn.get_num_devices() < 32:
        pytest.skip("Test is only valid on Galaxy")

    """Manually open and close multi-device"""
    mesh_shape, device_ids = ttnn.MeshShape(1, 4), ttnn.get_device_ids()
    multi_device = ttnn.open_mesh_device(mesh_shape, device_ids)
    assert multi_device.get_num_devices() == 4
    ttnn.close_mesh_device(multi_device)

    mesh_shape, device_ids = ttnn.MeshShape(8, 1), ttnn.get_device_ids()
    multi_device = ttnn.open_mesh_device(mesh_shape, device_ids)
    assert multi_device.get_num_devices() == 8
    ttnn.close_mesh_device(multi_device)

    mesh_shape, device_ids = ttnn.MeshShape(8, 4), ttnn.get_device_ids()
    multi_device = ttnn.open_mesh_device(mesh_shape, device_ids)
    assert multi_device.get_num_devices() == 32
    ttnn.close_mesh_device(multi_device)

    mesh_shape = ttnn.MeshShape(3, 2)
    multi_device = ttnn.open_mesh_device(mesh_shape, device_ids)
    assert multi_device.get_num_devices() == 6
    ttnn.close_mesh_device(multi_device)


#######
# Simple Multi-Device Tensor tests
#######


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}, {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b])
def test_ttnn_to_multi_device_multiple_times(mesh_device, layout, memory_config, dtype):
    """Test ttnn.to_device(..) works when the tensor is already on device"""
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Unsupported test permutation: bfloat8_b with ROW_MAJOR_LAYOUT")

    torch_tensor = torch.rand((1, 1, 32, 32 * mesh_device.get_num_devices()), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(
        torch_tensor, dtype=dtype, layout=layout, mesh_mapper=ShardTensorToMesh(mesh_device, dim=3)
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device, memory_config=memory_config)
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device, memory_config=memory_config)
    ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
    torch_loop_back_tensor = ttnn.to_torch(ttnn_loop_back_tensor, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))

    assert_with_pcc(torch_tensor, torch_loop_back_tensor, pcc=0.9999)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}, {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_ttnn_to_and_from_multi_device_shard(mesh_device, layout, memory_config, dtype):
    """Shard a tensor across devices, compose it back and verify loopback tensor is same as the original tensor"""
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Unsupported test permutation: bfloat8_b with ROW_MAJOR_LAYOUT")

    torch_tensor = torch.rand((1, 1, 32, 32 * mesh_device.get_num_devices()), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(
        torch_tensor, dtype=dtype, layout=layout, mesh_mapper=ShardTensorToMesh(mesh_device, dim=3)
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device, memory_config=memory_config)
    ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
    torch_loop_back_tensor = ttnn.to_torch(ttnn_loop_back_tensor, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))

    assert_with_pcc(torch_tensor, torch_loop_back_tensor, pcc=0.9999)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}, {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_multi_device_check_per_device_shard(mesh_device, layout, memory_config, dtype):
    """This test checks if the tensor is correctly sharded across devices"""
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Unsupported test permutation: bfloat8_b with ROW_MAJOR_LAYOUT")

    torch_tensor = torch.rand((1, 1, 32, 64 * mesh_device.get_num_devices()), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(
        torch_tensor, dtype=dtype, mesh_mapper=ShardTensorToMesh(mesh_device, dim=3), layout=layout
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device, memory_config=memory_config)
    ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)

    shard_offset, shard_size = 0, 64
    for device_tensor in ttnn.get_device_tensors(ttnn_loop_back_tensor):
        device_tensor_torch = ttnn.to_torch(device_tensor)
        assert_with_pcc(device_tensor_torch, torch_tensor[..., shard_offset : shard_offset + shard_size], pcc=0.9999)
        shard_offset += shard_size


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}, {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize("shape", [(1, 1, 32, 128), (1, 1, 16, 32)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_multi_device_replicate(mesh_device, shape, layout, memory_config):
    """Test ReplicateTensorToMesh to broadcast a tensor across multiple devices"""
    from ttnn import ReplicateTensorToMesh

    full_tensor = torch.rand(shape, dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(
        full_tensor,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
        layout=layout,
        memory_config=memory_config,
        device=mesh_device,
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)
    ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
    loopback_replicated_tensors = [
        ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(ttnn_loop_back_tensor.cpu())
    ]
    for loopback_replicated_tensor in loopback_replicated_tensors:
        assert torch.all(full_tensor == loopback_replicated_tensor)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}, {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
def test_ttnn_multi_device_all_gather(pcie_mesh_device):
    """Multidevice API test for ttnn.all_gather CCL operation"""
    if pcie_mesh_device.get_num_devices() <= 1:
        pytest.skip("Requires multiple devices to run")
    full_tensor = torch.rand((1, 1, 32, 32 * pcie_mesh_device.get_num_devices()), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(full_tensor, mesh_mapper=ShardTensorToMesh(pcie_mesh_device, dim=3))
    ttnn_tensor = ttnn.to_device(ttnn_tensor, pcie_mesh_device)
    ttnn_tensor = ttnn.all_gather(ttnn_tensor, dim=3, num_links=1)

    device_tensors: typing.List[ttnn.Tensor] = ttnn.get_device_tensors(ttnn_tensor)
    for device_tensor in device_tensors:
        device_tensor_torch = ttnn.to_torch(device_tensor)
        assert torch.all(device_tensor_torch == full_tensor)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}, {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
def test_multi_device_single_op_unary(mesh_device):
    """Multidevice API test: Running tensor-parallel multi-device single-op unary"""
    torch_input_tensor = torch.rand((1, 1, 32, 32 * mesh_device.get_num_devices()), dtype=torch.bfloat16)
    torch_output_golden = torch.nn.functional.gelu(torch_input_tensor)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
        device=mesh_device,
    )
    ttnn_output_tensor = ttnn.gelu(ttnn_input_tensor)

    ttnn_torch_output_tensor = ttnn.to_torch(ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))
    assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.999)


def test_multi_device_single_op_unary_with_cache(mesh_device):
    """Multidevice API test: Running tensor-parallel multi-device single-op unary with cache"""
    mesh_device.enable_program_cache()

    torch_input_tensor = torch.rand((1, 1, 32, 32 * mesh_device.get_num_devices()), dtype=torch.bfloat16)
    torch_output_golden = torch.nn.functional.gelu(torch_input_tensor)
    torch_golden = torch.nn.functional.gelu(torch_output_golden)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
        device=mesh_device,
    )
    ttnn_output_tensor = ttnn.gelu(ttnn_input_tensor)
    final_output_tensor = ttnn.gelu(ttnn_output_tensor)

    ttnn_torch_output_tensor = ttnn.to_torch(final_output_tensor, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))
    assert_with_pcc(ttnn_torch_output_tensor, torch_golden, pcc=0.999)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}, {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
def test_multi_device_single_op_binary(mesh_device):
    """Multidevice API test: Running tensor-parallel multi-device single-op binary"""
    torch_input_a_tensor = torch.rand((1, 1, 32, 32 * mesh_device.get_num_devices()), dtype=torch.bfloat16)
    torch_input_b_tensor = torch.rand((1, 1, 32, 32 * mesh_device.get_num_devices()), dtype=torch.bfloat16)
    torch_output_golden = torch_input_a_tensor + torch_input_b_tensor

    ttnn_input_a_tensor = ttnn.from_torch(
        torch_input_a_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
    )
    ttnn_input_b_tensor = ttnn.from_torch(
        torch_input_b_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
    )
    ttnn_output_tensor = ttnn.add(ttnn_input_a_tensor, ttnn_input_b_tensor)

    ttnn_torch_output_tensor = ttnn.to_torch(ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))
    assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.999)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}, {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
def test_multi_device_multi_op(mesh_device):
    """Multidevice API test: Running tensor-parallel multi-device multi-op"""
    torch_input_tensor = torch.rand((1, 1, 32, 32 * mesh_device.get_num_devices()), dtype=torch.bfloat16)
    torch_output_golden = torch.nn.functional.gelu(torch_input_tensor)
    torch_output_golden = torch.exp(torch_output_golden)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
        device=mesh_device,
    )
    ttnn_gelu_output = ttnn.gelu(ttnn_input_tensor)
    ttnn_output_tensor = ttnn.exp(ttnn_gelu_output)

    ttnn_torch_output_tensor = ttnn.to_torch(ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))
    assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.999)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}, {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
def test_multi_device_data_parallel_matmul_op(mesh_device):
    """Multidevice API: Data Parallel on matmul"""
    torch_input_a_tensor = torch.rand((mesh_device.get_num_devices(), 1, 32, 32), dtype=torch.bfloat16)
    torch_input_b_tensor = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
    torch_output_golden = torch_input_a_tensor @ torch_input_b_tensor

    ttnn_input_a_tensor = ttnn.from_torch(
        torch_input_a_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn_input_b_tensor = ttnn.from_torch(
        torch_input_b_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    ttnn_output_tensor = ttnn_input_a_tensor @ ttnn_input_b_tensor

    ttnn_torch_output_tensor = ttnn.to_torch(ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(mesh_device, dim=0))
    assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.993)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}, {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_multi_device_as_tensor_api(mesh_device, layout, memory_config, dtype):
    """Multidevice API: Data Parallel on matmul using cached tensor"""
    torch.manual_seed(0)
    torch_input_a_tensor = torch.rand((mesh_device.get_num_devices(), 1, 32, 32), dtype=torch.bfloat16)
    torch_input_b_tensor = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
    torch_output_golden = torch_input_a_tensor @ torch_input_b_tensor

    ttnn_input_a_tensor = ttnn.as_tensor(
        torch_input_a_tensor,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
        device=mesh_device,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=0),
    )

    with tempfile.NamedTemporaryFile() as temp_file:
        save_tensor = ttnn.as_tensor(
            torch_input_b_tensor,
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=memory_config,
            cache_file_name=f"{temp_file.name}.weight",
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )

        ttnn_input_b_tensor = ttnn.as_tensor(
            torch_input_b_tensor,
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=memory_config,
            cache_file_name=f"{temp_file.name}.weight",
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )

        ttnn_output_tensor = ttnn_input_a_tensor @ ttnn_input_b_tensor

        ttnn_torch_output_tensor = ttnn.to_torch(
            ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(mesh_device, dim=0)
        )
        if dtype == ttnn.bfloat4_b:
            assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.87)
        else:
            assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.991)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}, {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_multi_device_as_tensor_api_sharded_tensor(mesh_device, layout, memory_config, dtype):
    """Multidevice API: Data Parallel on matmul using cached tensor"""
    input_tensor = torch.rand((mesh_device.get_num_devices(), 1, 32, 32), dtype=torch.bfloat16)

    with tempfile.NamedTemporaryFile() as temp_file:
        save_tensor = ttnn.as_tensor(
            input_tensor,
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=memory_config,
            cache_file_name=f"{temp_file.name}.weight",
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=0),
        )
        load_tensor = ttnn.as_tensor(
            input_tensor,
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=memory_config,
            cache_file_name=f"{temp_file.name}.weight",
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=0),
        )
        torch_loaded_tensor = ttnn.to_torch(load_tensor, mesh_composer=ConcatMeshToTensor(mesh_device, dim=0))
        expected_pcc = 0.98 if dtype == ttnn.bfloat4_b else 0.99
        assert_with_pcc(input_tensor, torch_loaded_tensor, pcc=expected_pcc)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}, {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b])
def test_multi_device_permute(mesh_device, layout, memory_config, dtype):
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Unsupported test permutation: bfloat8_b with ROW_MAJOR_LAYOUT")

    torch_tensor = torch.rand((32, 1, 160, 64 * mesh_device.get_num_devices()), dtype=torch.bfloat16)
    torch_golden = torch.permute(torch_tensor, (0, 1, 3, 2))

    ttnn_tensor = ttnn.from_torch(
        torch_tensor, dtype=dtype, layout=layout, mesh_mapper=ShardTensorToMesh(mesh_device, dim=3)
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device, memory_config=memory_config)
    ttnn_permute = ttnn.permute(ttnn_tensor, (0, 1, 3, 2))
    ttnn_loop_back_tensor = ttnn.from_device(ttnn_permute)
    torch_loop_back_tensor = ttnn.to_torch(ttnn_loop_back_tensor, mesh_composer=ConcatMeshToTensor(mesh_device, dim=2))

    assert_with_pcc(torch_golden, torch_loop_back_tensor, pcc=0.9999)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}, {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
def test_max(mesh_device):
    gate_logits_1SB8 = ttnn.from_torch(
        torch.randn(1, 1, 32, 8),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    gate_logits_1SB8 = ttnn.to_device(gate_logits_1SB8, mesh_device)
    weights_ex0_1SB1 = ttnn.max(gate_logits_1SB8, dim=3)
    print(weights_ex0_1SB1)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}, {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
def test_ttnn_multi_device_all_gather_all_devices(t3k_mesh_device):
    """Multidevice API test for ttnn.all_gather CCL operation for full 8-device T3K"""

    full_tensor = torch.ones((1, 1, 32, 32 * t3k_mesh_device.get_num_devices()), dtype=torch.bfloat16)
    for i in range(t3k_mesh_device.get_num_devices()):
        full_tensor[..., i * 32 : (i + 1) * 32] = i

    ttnn_tensor = ttnn.from_torch(full_tensor, mesh_mapper=ShardTensorToMesh(t3k_mesh_device, dim=3))
    ttnn_tensor = ttnn.to_device(ttnn_tensor, t3k_mesh_device)
    ttnn_tensor = ttnn.all_gather(ttnn_tensor, dim=3, num_links=1)

    device_tensors: typing.List[ttnn.Tensor] = ttnn.get_device_tensors(ttnn_tensor)
    for device_tensor in device_tensors:
        device_tensor_torch = ttnn.to_torch(device_tensor)
        assert torch.all(device_tensor_torch == full_tensor)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}, {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
def test_sharded_matmul(t3k_mesh_device):
    q_heads_1B4D = ttnn.from_torch(
        torch.randn(1, 32, 32, 128),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=t3k_mesh_device,
        mesh_mapper=ReplicateTensorToMesh(t3k_mesh_device),
    )
    keys_1BDP = ttnn.from_torch(
        torch.randn(1, 32, 128, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=t3k_mesh_device,
        mesh_mapper=ReplicateTensorToMesh(t3k_mesh_device),
    )

    q_heads_1B4D = ttnn.to_device(q_heads_1B4D, t3k_mesh_device)
    keys_1BDP = ttnn.to_device(keys_1BDP, t3k_mesh_device)

    core_grid = ttnn.CoreGrid(y=4, x=8)
    compute_grid_size = t3k_mesh_device.compute_with_storage_grid_size()
    if (compute_grid_size.x < core_grid.x) or (compute_grid_size.y < core_grid.y):
        pytest.skip("Test requires larger grid size")

    q_heads_1B4D = ttnn.to_memory_config(
        q_heads_1B4D,
        ttnn.create_sharded_memory_config(
            shape=(32, 128),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    keys_1BDP = ttnn.to_memory_config(
        keys_1BDP,
        ttnn.create_sharded_memory_config(
            shape=(128, 32),
            core_grid=core_grid,
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
        core_grid=core_grid,
        compute_kernel_config=compute_kernel_attn,
    )

    print(attn_1B4P)


def test_4b_tensor(mesh_device):
    tensor = ttnn.from_torch(
        torch.randn(1, 1, 32, 32),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    tensor = ttnn.to_device(tensor, mesh_device)
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    x = ttnn.to_device(x, mesh_device)
    tensor = ttnn.matmul(
        x,
        tensor,
        core_grid=ttnn.CoreGrid(y=4, x=8),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
        ),
    )


def test_slicing(mesh_device):
    tensor = ttnn.from_torch(
        torch.randn(1, 32, 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    tensor = ttnn.to_device(tensor, mesh_device)
    tensor = tensor[:, :, :, :1]
    assert all([device_tensor.shape == tensor.shape for device_tensor in ttnn.get_device_tensors(tensor)])


def test_clone(mesh_device):
    results_11BH = ttnn.from_torch(
        torch.randn(1, 1, 32, 128),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    results_11BH = ttnn.to_device(results_11BH, mesh_device)
    results_11BH = ttnn.clone(results_11BH, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
    print(results_11BH)


@pytest.mark.parametrize("height", [7])
@pytest.mark.parametrize("width", [3])
def test_validate_as_tensor(tmp_path, mesh_device, height, width):
    torch_input_tensor = torch.rand((height, width), dtype=torch.float32)

    memory_config = ttnn.L1_MEMORY_CONFIG
    tensor = ttnn.as_tensor(
        torch_input_tensor,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        cache_file_name=tmp_path / "cache_file",
    )
    assert tensor.dtype == ttnn.float32
    assert tensor.layout == ttnn.TILE_LAYOUT
    assert ttnn.get_memory_config(tensor) == memory_config

    tensor = ttnn.as_tensor(
        torch_input_tensor,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        cache_file_name=tmp_path / "cache_file",
    )
    assert tensor.dtype == ttnn.float32
    assert tensor.layout == ttnn.TILE_LAYOUT
    assert ttnn.get_memory_config(tensor) == memory_config


def test_visualize_mesh_device(t3k_mesh_device):
    ttnn.visualize_mesh_device(t3k_mesh_device)


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x2_grid")], indirect=True)
def test_all_gather_multiple_submeshes(mesh_device):
    """Test all_gather with multiple submeshes"""
    if mesh_device.get_num_devices() < 8:
        pytest.skip()

    def model(submesh):
        # Reshape to a 1x4 mesh to enforce ring connected topological order.
        submesh.reshape(ttnn.MeshShape(1, 4))
        full_tensor = torch.ones((1, 1, 32, 32 * submesh.get_num_devices()), dtype=torch.bfloat16)
        for i in range(submesh.get_num_devices()):
            full_tensor[..., i * 32 : (i + 1) * 32] = i

        ttnn_tensor = ttnn.from_torch(full_tensor, mesh_mapper=ShardTensorToMesh(submesh, dim=3))
        ttnn_tensor = ttnn.to_device(ttnn_tensor, submesh)
        ttnn_tensor = ttnn.all_gather(ttnn_tensor, dim=3, num_links=1)

        for device_tensor in ttnn.get_device_tensors(ttnn_tensor):
            device_tensor_torch = ttnn.to_torch(device_tensor)
            assert torch.all(device_tensor_torch == full_tensor)

    submesh_devices = mesh_device.create_submeshes(ttnn.MeshShape(2, 2))
    for submesh in submesh_devices:
        model(submesh)


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 8), id="1x8_line")], indirect=True)
def test_line_all_gather_after_reshape(mesh_device):
    if mesh_device.get_num_devices() < 8:
        pytest.skip()
    mesh_device.reshape(ttnn.MeshShape(2, 4))

    torch_input_tensor = torch.rand((1, 1, 64, 128), dtype=torch.bfloat16)

    mesh_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=list(mesh_device.shape), dims=(2, 3)),
    )
    output_tensor = ttnn.all_gather(
        mesh_tensor,
        dim=2,
        cluster_axis=0,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
    )


def test_distribute_api(mesh_device):
    torch_hidden_states = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        hidden_states = ttnn.from_torch(
            torch_hidden_states,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )


def test_heterogenous_operation_dispatch():
    if ttnn.get_num_devices() < 8:
        pytest.skip()

    a = torch.rand((1, 1, 32, 128), dtype=torch.bfloat16)
    torch_gelu = torch.nn.functional.gelu(a)
    torch_silu = torch.nn.functional.silu(a)

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    submesh_0 = mesh_device.create_submesh(ttnn.MeshShape(1, 4))
    submesh_1 = mesh_device.create_submesh(ttnn.MeshShape(1, 1), offset=ttnn.MeshCoordinate(1, 1))

    ttnn_input_0 = ttnn.from_torch(
        a, device=submesh_0, mesh_mapper=ttnn.ShardTensorToMesh(submesh_0, dim=-1), layout=ttnn.TILE_LAYOUT
    )
    ttnn_input_1 = ttnn.from_torch(
        a, device=submesh_1, mesh_mapper=ttnn.ShardTensorToMesh(submesh_1, dim=-1), layout=ttnn.TILE_LAYOUT
    )

    ttnn_gelu = ttnn.gelu(ttnn_input_0)
    ttnn_silu = ttnn.silu(ttnn_input_1)

    assert_with_pcc(
        ttnn.to_torch(ttnn_gelu, mesh_composer=ttnn.ConcatMeshToTensor(submesh_0, dim=-1)), torch_gelu, pcc=0.9999
    )
    assert_with_pcc(
        ttnn.to_torch(ttnn_silu, mesh_composer=ttnn.ConcatMeshToTensor(submesh_1, dim=-1)), torch_silu, pcc=0.9999
    )
