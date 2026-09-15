# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn


def test_fabric_mux_exports_are_experimental_only():
    assert ttnn.experimental.fabric_mux.Config is not None
    assert ttnn.experimental.fabric_mux.ChannelType.FULL_SIZE is not None
    assert ttnn.experimental.fabric_mux.ChannelType.HEADER_ONLY is not None
    assert ttnn.experimental.fabric_mux.KernelBuildOptLevel.O3 is not None
    top_level_names = (
        "FabricMuxChannelType",
        "FabricMuxConfig",
        "KernelBuildOptLevel",
        "fabric_mux_connection_ct_args",
        "fabric_mux_connection_rt_args",
        "get_tt_fabric_channel_buffer_size_bytes",
    )
    assert all(not hasattr(ttnn, name) for name in top_level_names)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_fabric_mux_argument_builders_preserve_runtime_abi(mesh_device):
    fabric_mux = ttnn.experimental.fabric_mux
    logical_core = ttnn.CoreCoord(0, 0)
    virtual_core = mesh_device.worker_core_from_logical_core(logical_core)
    channel_type = fabric_mux.ChannelType.FULL_SIZE
    base_l1_address = ttnn.get_allocator_base_address(mesh_device, ttnn.BufferType.L1)
    mux_config = fabric_mux.Config(
        num_full_size_channels=1,
        num_header_only_channels=0,
        num_buffers_per_full_size_channel=2,
        num_buffers_per_header_only_channel=0,
        full_size_channel_buffer_size_bytes=fabric_mux.channel_buffer_size_bytes(),
        base_l1_address=base_l1_address,
    )

    compile_time_args = fabric_mux.client_compile_time_args(
        num_clients=1,
        channel_type=channel_type,
        config=mux_config,
    )
    assert len(compile_time_args) == 5
    assert compile_time_args[:2] == [
        mux_config.num_buffers(channel_type),
        mux_config.buffer_size_bytes(channel_type),
    ]
    assert compile_time_args[4] == 1
    assert base_l1_address <= compile_time_args[2] < compile_time_args[3] < mux_config.memory_map_end_address()

    kernel_compile_time_args = mux_config.kernel_compile_time_args()
    assert len(kernel_compile_time_args) == 19
    assert kernel_compile_time_args[:5] == [
        1,
        2,
        fabric_mux.channel_buffer_size_bytes(),
        0,
        8,
    ]
    assert kernel_compile_time_args[5:7] == compile_time_args[2:4]
    assert kernel_compile_time_args[-1] == 0

    mux_program_descriptor = ttnn.ProgramDescriptor()
    kernel_runtime_args = mux_config.kernel_runtime_args(
        source_node_id=mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(0, 0)),
        destination_node_id=mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(0, 1)),
        link_index=0,
        program_descriptor=mux_program_descriptor,
        mux_logical_core=logical_core,
    )
    assert kernel_runtime_args[0] == 4
    assert len(kernel_runtime_args) == 12
    assert kernel_runtime_args[-2:] == [0, 1]
    assert [semaphore.id for semaphore in mux_program_descriptor.semaphores] == [0, 1]

    program_descriptor = ttnn.ProgramDescriptor()
    active_args = fabric_mux.client_runtime_args(
        connection_valid=True,
        is_termination_master=True,
        channel_type=channel_type,
        mux_virtual_core=virtual_core,
        client_index=0,
        client_logical_core=logical_core,
        config=mux_config,
        program_descriptor=program_descriptor,
        termination_master_virtual_core=virtual_core,
    )
    assert len(active_args) == 17
    assert active_args[:4] == [1, 1, virtual_core.x, virtual_core.y]
    assert active_args[15:] == [virtual_core.x, virtual_core.y]
    assert [semaphore.id for semaphore in program_descriptor.semaphores] == list(range(5))

    inactive_args = fabric_mux.client_runtime_args(
        connection_valid=False,
        is_termination_master=False,
        channel_type=channel_type,
        mux_virtual_core=virtual_core,
        client_index=0,
        client_logical_core=logical_core,
        config=mux_config,
        program_descriptor=program_descriptor,
        termination_master_virtual_core=virtual_core,
        termination_master_semaphore_id=active_args[10],
    )
    assert len(inactive_args) == 17
    assert inactive_args[:2] == [0, 0]
    assert inactive_args[10] == active_args[10]
    assert [semaphore.id for semaphore in program_descriptor.semaphores] == list(range(9))
