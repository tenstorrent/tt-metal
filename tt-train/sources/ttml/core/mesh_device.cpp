// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_device.hpp"

#include <cstdlib>
#include <string>

#include "hostdevcommon/common_values.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttml::core {

namespace {

// Experiment: TTML_NUM_COMMAND_QUEUES=2 opens a second command queue (for collectives beside compute).
size_t num_command_queues() {
    const char* value = std::getenv("TTML_NUM_COMMAND_QUEUES");
    return value == nullptr ? 1 : static_cast<size_t>(std::stoul(value));
}

}  // namespace

MeshDevice::MeshDevice(const tt::tt_metal::distributed::MeshShape& shape, const std::vector<int>& device_ids) :
    m_mesh_device(ttnn::distributed::open_mesh_device(
        shape,
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        num_command_queues(),
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
