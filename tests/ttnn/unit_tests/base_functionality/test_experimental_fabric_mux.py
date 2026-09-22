# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn


def test_fabric_mux_exports_are_experimental_only():
    assert ttnn.experimental.fabric_mux.Config is not None, "fabric_mux.Config is not exposed"
    assert ttnn.experimental.fabric_mux.ChannelType.FULL_SIZE is not None, "ChannelType.FULL_SIZE is not exposed"
    assert ttnn.experimental.fabric_mux.ChannelType.HEADER_ONLY is not None, "ChannelType.HEADER_ONLY is not exposed"
    assert ttnn.experimental.fabric_mux.KernelBuildOptLevel.O3 is not None, "KernelBuildOptLevel.O3 is not exposed"
    top_level_names = (
        "FabricMuxChannelType",
        "FabricMuxConfig",
        "KernelBuildOptLevel",
        "fabric_mux_connection_ct_args",
        "fabric_mux_connection_rt_args",
        "get_tt_fabric_channel_buffer_size_bytes",
    )
    leaked = [name for name in top_level_names if hasattr(ttnn, name)]
    assert not leaked, f"fabric mux names must stay under ttnn.experimental, but ttnn exports {leaked}"


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
    assert len(compile_time_args) == 5, f"client CT-arg count: expected 5, got {len(compile_time_args)}"
    expected_ct_prefix = [
        mux_config.num_buffers(channel_type),
        mux_config.buffer_size_bytes(channel_type),
    ]
    assert (
        compile_time_args[:2] == expected_ct_prefix
    ), f"client CT args must start with (num_buffers, buffer_size): expected {expected_ct_prefix}, got {compile_time_args[:2]}"
    assert compile_time_args[4] == 1, f"client CT arg 4 (num_clients): expected 1, got {compile_time_args[4]}"
    assert base_l1_address <= compile_time_args[2] < compile_time_args[3] < mux_config.memory_map_end_address(), (
        f"channel addresses must be ordered within the mux L1 map [{base_l1_address}, "
        f"{mux_config.memory_map_end_address()}), got {compile_time_args[2]} and {compile_time_args[3]}"
    )

    kernel_compile_time_args = mux_config.kernel_compile_time_args()
    assert (
        len(kernel_compile_time_args) == 19
    ), f"mux kernel CT-arg count: expected 19, got {len(kernel_compile_time_args)}"
    expected_kernel_ct_prefix = [
        1,
        2,
        fabric_mux.channel_buffer_size_bytes(),
        0,
        8,
    ]
    assert (
        kernel_compile_time_args[:5] == expected_kernel_ct_prefix
    ), f"mux kernel CT-arg prefix: expected {expected_kernel_ct_prefix}, got {kernel_compile_time_args[:5]}"
    assert (
        kernel_compile_time_args[5:7] == compile_time_args[2:4]
    ), f"mux and client must agree on channel addresses: {kernel_compile_time_args[5:7]} vs {compile_time_args[2:4]}"
    assert kernel_compile_time_args[-1] == 0, f"last mux kernel CT arg: expected 0, got {kernel_compile_time_args[-1]}"

    mux_program_descriptor = ttnn.ProgramDescriptor()
    kernel_runtime_args = mux_config.kernel_runtime_args(
        source_node_id=mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(0, 0)),
        destination_node_id=mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(0, 1)),
        link_index=0,
        program_descriptor=mux_program_descriptor,
        mux_logical_core=logical_core,
    )
    assert kernel_runtime_args[0] == 4, f"mux RT arg 0: expected 4, got {kernel_runtime_args[0]}"
    assert len(kernel_runtime_args) == 12, f"mux RT-arg count: expected 12, got {len(kernel_runtime_args)}"
    assert kernel_runtime_args[-2:] == [0, 1], f"mux RT-arg tail: expected [0, 1], got {kernel_runtime_args[-2:]}"
    mux_semaphore_ids = [semaphore.id for semaphore in mux_program_descriptor.semaphores]
    assert mux_semaphore_ids == [0, 1], f"mux descriptor semaphore ids: expected [0, 1], got {mux_semaphore_ids}"

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
    assert len(active_args) == 17, f"connected client RT-arg count: expected 17, got {len(active_args)}"
    expected_active_prefix = [1, 1, virtual_core.x, virtual_core.y]
    assert (
        active_args[:4] == expected_active_prefix
    ), f"connected client RT-arg prefix: expected {expected_active_prefix}, got {active_args[:4]}"
    assert active_args[15:] == [
        virtual_core.x,
        virtual_core.y,
    ], f"termination master NoC coords: expected {[virtual_core.x, virtual_core.y]}, got {active_args[15:]}"
    active_semaphore_ids = [semaphore.id for semaphore in program_descriptor.semaphores]
    assert active_semaphore_ids == list(
        range(5)
    ), f"a connected client must allocate 5 semaphores, got ids {active_semaphore_ids}"

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
    assert len(inactive_args) == 17, f"disconnected client RT-arg count: expected 17, got {len(inactive_args)}"
    assert inactive_args[:2] == [
        0,
        0,
    ], f"disconnected client must report connection_valid=0 and is_termination_master=0, got {inactive_args[:2]}"
    assert (
        inactive_args[10] == active_args[10]
    ), f"supplied termination semaphore must be reused: expected {active_args[10]}, got {inactive_args[10]}"
    reused_semaphore_ids = [semaphore.id for semaphore in program_descriptor.semaphores]
    assert reused_semaphore_ids == list(range(9)), (
        "reusing the termination semaphore must add 4 semaphores, not 5; " f"got ids {reused_semaphore_ids}"
    )
