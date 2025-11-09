// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_device.hpp"

#include <core/ttnn_all_includes.hpp>
#include <tt-metalium/cluster.hpp>
#include <tt-metalium/dispatch_core_common.hpp>

namespace ttml::core {

namespace {
inline tt::tt_metal::DispatchCoreType decide_dispatch_core_type() {
    using tt::tt_metal::ClusterType;
    auto cluster_type = tt::tt_metal::GetClusterType();
    switch (cluster_type) {
        case ClusterType::N300:
        case ClusterType::T3K:
        case ClusterType::N300_2x2: return tt::tt_metal::DispatchCoreType::ETH;
        default: return tt::tt_metal::DispatchCoreType::WORKER;
    }
}
}  // namespace

MeshDevice::MeshDevice(const tt::tt_metal::distributed::MeshShape& shape, const std::vector<int>& device_ids) :
    m_mesh_device(ttnn::distributed::open_mesh_device(
        shape,
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /* num_command_queues=*/1,
        tt::tt_metal::DispatchCoreConfig{decide_dispatch_core_type()},
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
