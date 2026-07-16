# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
import torch


def test_open_device():
    """Simple unit test to test device open/close APIs"""
    device = ttnn.open_device(device_id=0, num_command_queues=1)
    ttnn.close_device(device)


def test_createdevice_matches_open_device_grid_defaults():
    open_device_handle = None
    create_device_handle = None
    try:
        open_device_handle = ttnn.open_device(device_id=0)
        open_device_grid = open_device_handle.compute_with_storage_grid_size()
        ttnn.close_device(open_device_handle)
        open_device_handle = None

        create_device_handle = ttnn.CreateDevice(device_id=0)
        create_device_grid = create_device_handle.compute_with_storage_grid_size()

        assert open_device_grid == create_device_grid
    finally:
        if open_device_handle is not None:
            ttnn.close_device(open_device_handle)
        if create_device_handle is not None:
            ttnn.close_device(create_device_handle)


def test_dispatch_core_config_defaults_follow_cluster_policy():
    cluster_type = ttnn._ttnn.cluster.get_cluster_type()
    config = ttnn.DispatchCoreConfig()
    eth_default_dispatch_clusters = {
        ttnn._ttnn.cluster.ClusterType.N300,
        ttnn._ttnn.cluster.ClusterType.T3K,
        ttnn._ttnn.cluster.ClusterType.N300_2x2,
    }
    expected_type = (
        ttnn.DispatchCoreType.ETH if cluster_type in eth_default_dispatch_clusters else ttnn.DispatchCoreType.WORKER
    )
    assert config.type == expected_type

    is_blackhole = "blackhole" in ttnn._ttnn.device.get_arch_name()
    expected_axis = ttnn.DispatchCoreAxis.COL if is_blackhole else ttnn.DispatchCoreAxis.ROW
    assert config.axis == expected_axis


def test_dispatch_core_config_constructor_rules(expect_error):
    with expect_error(Exception, "COL axis is not supported for ETH dispatch core type"):
        ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH, ttnn.DispatchCoreAxis.COL)

    config = ttnn.DispatchCoreConfig(axis=ttnn.DispatchCoreAxis.COL)
    assert config.type == ttnn.DispatchCoreType.WORKER
    assert config.axis == ttnn.DispatchCoreAxis.COL

    is_blackhole = "blackhole" in ttnn._ttnn.device.get_arch_name()
    if is_blackhole:
        with expect_error(
            Exception, "ROW dispatch core axis is not supported for blackhole arch unless fabric tensix MUX"
        ):
            ttnn.DispatchCoreConfig(
                axis=ttnn.DispatchCoreAxis.ROW, fabric_tensix_config=ttnn.FabricTensixConfig.DISABLED
            )
        assert (
            ttnn.DispatchCoreConfig(fabric_tensix_config=ttnn.FabricTensixConfig.MUX).axis == ttnn.DispatchCoreAxis.ROW
        )


def test_manage_device():
    """Simple unit test to test device context manager APIs"""
    with ttnn.manage_device(0) as device:
        pass


def test_l1_size():
    assert ttnn.get_max_worker_l1_unreserved_size() > 1024 * 1024


@pytest.mark.parametrize(
    "device_params",
    [{"worker_l1_size": 16385}],
    indirect=True,
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b])
def test_worker_l1_size(device, layout, dtype):
    torch_tensor = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)

    core_grid = ttnn.CoreGrid(y=1, x=1)
    memory_config = ttnn.create_sharded_memory_config(torch_tensor.shape, core_grid, ttnn.ShardStrategy.BLOCK)
    ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout)
    ttnn_tensor = ttnn.to_device(ttnn_tensor, device, memory_config=memory_config)
    ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
    torch_loop_back_tensor = ttnn.to_torch(ttnn_loop_back_tensor)


@pytest.mark.parametrize(
    "device_params",
    [{"worker_l1_size": 16385}],
    indirect=True,
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_worker_l1_fail(device, layout, dtype, expect_error):
    torch_tensor = torch.rand((1, 1, 32, 1024), dtype=torch.bfloat16)

    core_grid = ttnn.CoreGrid(y=1, x=1)
    memory_config = ttnn.create_sharded_memory_config(torch_tensor.shape, core_grid, ttnn.ShardStrategy.BLOCK)
    ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout)
    with expect_error(RuntimeError, ".*Out of Memory:.*"):
        ttnn_tensor = ttnn.to_device(
            ttnn_tensor,
            device,
            memory_config=memory_config,
        )


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_dispatch_context_init_and_terminate(mesh_device, dtype, layout):
    tensor = torch.rand((1, 1, 32, 1024), dtype=torch.bfloat16)
    core_grid = ttnn.CoreGrid(y=1, x=1)
    memory_config = ttnn.create_sharded_memory_config(tensor.shape, core_grid, ttnn.ShardStrategy.BLOCK)
    ttnn_tensor = ttnn.from_torch(tensor, dtype=dtype, layout=layout)

    with ttnn.device.setup_fast_dispatch(mesh_device):
        ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device, memory_config=memory_config).cpu()

    assert ttnn_tensor.shape == tensor.shape


@pytest.mark.requires_fast_runtime_mode_off
def test_async_sd_state_preserved_across_fd(mesh_device):
    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    assert ttnn.device.is_asynchronous_slow_dispatch_enabled(mesh_device)

    with ttnn.device.setup_fast_dispatch(mesh_device):
        pass

    assert ttnn.device.is_asynchronous_slow_dispatch_enabled(mesh_device)


def test_pad_to_tile_shape_removed():
    """ttnn.pad_to_tile_shape has been removed; verify it is no longer accessible."""
    assert not hasattr(ttnn, "pad_to_tile_shape"), (
        "ttnn.pad_to_tile_shape should be removed. "
        "Use ttnn.to_layout(tensor, ttnn.TILE_LAYOUT) or "
        "models.common.tensor_utils.align_shape_to_tile instead."
    )
