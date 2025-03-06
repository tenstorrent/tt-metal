# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger


def create_and_load_sub_device_manager_with_fabric_interface(
    mesh_device,
    worker_sub_devices,
    ccl_worker_sub_device_id,
    local_allocator_size,
    enable_persistent_fabric=True,
    wrap_fabric_around_mesh=False,
    context_switch_interval_override=None,
):
    assert ccl_worker_sub_device_id < len(worker_sub_devices)
    mesh_sub_device_manager_id, fabric_subdevice_id = mesh_device.create_sub_device_manager_with_fabric(
        worker_sub_devices, local_allocator_size
    )
    # fabric sub-device id can also be queried from device, no need to explicitly pass it in
    mesh_device.load_sub_device_manager(mesh_sub_device_manager_id)
    if enable_persistent_fabric:
        ttnn.initialize_edm_fabric(
            mesh_device,
            wrap_fabric_around_mesh=wrap_fabric_around_mesh,
            context_switch_interval_override=context_switch_interval_override,
        )
    return mesh_sub_device_manager_id


def teardown_fabric_interface(mesh_device):
    logger.debug(f"Tearing down fabric (this may take a while if context switch interval is large)")
    ttnn.teardown_edm_fabric(mesh_device)
    ttnn.synchronize_device(mesh_device)


def create_global_semaphore_with_same_address(mesh_device, cores, initial_value):
    semaphore_handles = ttnn.create_global_semaphore_with_same_address(mesh_device, cores, initial_value)
    addrs = ttnn.get_global_semaphore_address(semaphore_handles)
    logger.debug(f"from remote semaphore handle addresses: {addrs}")
    # assert all addresses are the same
    assert len(set(addrs)) == 1
    return semaphore_handles
