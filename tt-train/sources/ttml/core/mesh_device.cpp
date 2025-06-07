// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_device.hpp"

namespace ttml::core {

MeshDevice::MeshDevice(tt::tt_metal::MeshShape shape) :
    m_mesh_device(ttnn::open_mesh_device(
        shape,
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /* num_command_queues*/ 1,
        tt::tt_metal::DispatchCoreConfig{})) {
    assert(m_mesh_device);
}

[[nodiscard]] ttnn::MeshDevice& MeshDevice::get_device() {
    assert(m_mesh_device);
    return *m_mesh_device;
}

MeshDevice::~MeshDevice() {
    assert(m_mesh_device);
    ttnn::close_mesh_device(m_mesh_device);
}

}  // namespace ttml::core
