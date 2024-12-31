# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


def create_and_load_sub_device_manager_with_fabric_interface(
    mesh_device, worker_sub_devices, ccl_worker_sub_device_id, local_allocator_size, enable_persistent_fabric=True
):
    assert ccl_worker_sub_device_id < len(worker_sub_devices)
    mesh_sub_device_manager_id, fabric_subdevice_id = mesh_device.create_sub_device_manager_with_fabric(
        worker_sub_devices, local_allocator_size
    )
    # fabric sub-device id can also be queried from device, no need to explicitly pass it in
    mesh_device.load_sub_device_manager(mesh_sub_device_manager_id)
    if enable_persistent_fabric:
        ttnn.initialize_edm_fabric(mesh_device)
    return mesh_sub_device_manager_id


def teardown_fabric_interface(mesh_device):
    ttnn.teardown_edm_fabric(mesh_device)
    for device_id in mesh_device.get_device_ids():
        ttnn.synchronize_device(mesh_device.get_device(device_id))
