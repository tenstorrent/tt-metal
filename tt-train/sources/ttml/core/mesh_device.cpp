// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_device.hpp"

#include <core/ttnn_all_includes.hpp>

namespace ttml::core {

MeshDevice::MeshDevice([[maybe_unused]] int device_index) :
    m_mesh_device(ttnn::distributed::api::open_mesh_device(
        ttnn::distributed::MeshShape(1, 1),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /* num_command_queues*/ 1,
        DispatchCoreConfig{},
        ttnn::distributed::MeshType::RowMajor)) {
    assert(m_mesh_device);
}

[[nodiscard]] ttnn::distributed::MeshDevice& MeshDevice::get_device() {
    assert(m_mesh_device);
    return *m_mesh_device;
}

MeshDevice::~MeshDevice() {
    ttnn::distributed::api::close_mesh_device(m_mesh_device);
}

}  // namespace ttml::core
