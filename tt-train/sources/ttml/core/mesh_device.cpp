// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_device.hpp"

namespace ttml::core {

MeshDevice::MeshDevice(const tt::tt_metal::distributed::MeshShape& shape, const std::vector<int>& device_ids) :
    m_mesh_device(ttnn::distributed::open_mesh_device(
        shape,
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /* num_command_queues=*/1,
        tt::tt_metal::DispatchCoreConfig{},
        /*offset=*/std::nullopt,
        /*physical_device_ids=*/device_ids)) {
    assert(m_mesh_device);
}

[[nodiscard]] ttnn::distributed::MeshDevice& MeshDevice::get_device() {
    assert(m_mesh_device);
    return *m_mesh_device;
}

MeshDevice::~MeshDevice() {
    assert(m_mesh_device);
    ttnn::distributed::close_mesh_device(m_mesh_device);
}

}  // namespace ttml::core
