// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_device.hpp"

#include <cstdlib>

#include "hostdevcommon/common_values.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttml::core {

MeshDevice::MeshDevice(const tt::tt_metal::distributed::MeshShape& shape, const std::vector<int>& device_ids) :
    m_mesh_device(ttnn::distributed::open_mesh_device(
        shape,
        // TTML_L1_SMALL_SIZE opts into a nonzero L1_SMALL region, required by
        // ttnn ops using the sliding-window halo path (conv2d, pool, upsample).
        // Default (0) is unchanged for existing transformer workloads.
        [] {
            const char* env = std::getenv("TTML_L1_SMALL_SIZE");
            return env != nullptr ? static_cast<size_t>(std::stoul(env)) : static_cast<size_t>(DEFAULT_L1_SMALL_SIZE);
        }(),
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

[[nodiscard]] std::shared_ptr<ttnn::distributed::MeshDevice> MeshDevice::get_device_ptr() const {
    return m_mesh_device;
}

MeshDevice::~MeshDevice() {
    assert(m_mesh_device);
    ttnn::distributed::close_mesh_device(m_mesh_device);
}

}  // namespace ttml::core
