// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_device.hpp"

#include "hostdevcommon/common_values.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace ttml::core {

namespace {

// ttml::core::MeshDevice builds its own MetalEnv when it opens a device, and only one
// environment for the cluster may exist at a time.
tt::tt_metal::MetalEnvDescriptor make_env_descriptor() {
    tt::tt_metal::FabricConfigDescriptor fabric_config_descriptor;
    if (auto fabric_config = ttnn_fixed::distributed::selected_fabric_config()) {
        fabric_config_descriptor.fabric_config = *fabric_config;
    }
    return tt::tt_metal::MetalEnvDescriptor(/*mock_cluster_desc_path=*/std::nullopt, fabric_config_descriptor);
}

}  // namespace

MeshDevice::MeshDevice(const tt::tt_metal::distributed::MeshShape& shape, const std::vector<int>& device_ids) :
    m_env(std::make_unique<tt::tt_metal::MetalEnv>(make_env_descriptor())),
    m_mesh_device(m_env->create_mesh_device(
        tt::tt_metal::distributed::MeshDeviceConfig(shape, /*offset=*/std::nullopt, device_ids),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /* num_command_queues=*/1,
        tt::tt_metal::DispatchCoreConfig{})) {
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
