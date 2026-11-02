// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_device.hpp"

#include <cstdlib>

#include "hostdevcommon/common_values.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttml::core {

namespace {
// TTML_TRACE_REGION_SIZE (bytes): a metal trace region for capturing and replaying whole training steps
// (ttnn.begin_trace_capture / execute_trace on the mesh device); the default is no trace region.
size_t trace_region_size_from_env() {
    const char* value = std::getenv("TTML_TRACE_REGION_SIZE");
    return value != nullptr ? static_cast<size_t>(std::strtoull(value, nullptr, 10)) : DEFAULT_TRACE_REGION_SIZE;
}
}  // namespace

MeshDevice::MeshDevice(const tt::tt_metal::distributed::MeshShape& shape, const std::vector<int>& device_ids) :
    m_mesh_device(ttnn::distributed::open_mesh_device(
        shape,
        DEFAULT_L1_SMALL_SIZE,
        trace_region_size_from_env(),
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

[[nodiscard]] std::shared_ptr<ttnn::distributed::MeshDevice> MeshDevice::get_device_ptr() const {
    return m_mesh_device;
}

MeshDevice::~MeshDevice() {
    assert(m_mesh_device);
    ttnn::distributed::close_mesh_device(m_mesh_device);
}

}  // namespace ttml::core
