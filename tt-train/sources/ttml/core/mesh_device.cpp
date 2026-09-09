// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_device.hpp"

#include <limits>

#include "hostdevcommon/common_values.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace ttml::core {

namespace {

// Describes the MetalEnv that ttml::core::MeshDevice constructs for itself, enabling fabric only
// when a topology was selected for this run.
tt::tt_metal::MetalEnvDescriptor make_env_descriptor() {
    tt::tt_metal::FabricConfigDescriptor fabric_config_descriptor;
    if (auto fabric_config = ttnn_fixed::distributed::selected_fabric_config()) {
        fabric_config_descriptor.fabric_config = *fabric_config;
        // Workaround by explicitly reserving all available routing planes
        fabric_config_descriptor.num_routing_planes = std::numeric_limits<uint8_t>::max();
    }
    return tt::tt_metal::MetalEnvDescriptor(/*mock_cluster_desc_path=*/std::nullopt, fabric_config_descriptor);
}

// Opens a mesh device whose MetalContext is owned by `env` rather than by the MeshDevice. Touching
// the system mesh first registers the context on the env, so create_mesh_device reuses it instead of
// creating its own. A MeshDevice-owned context is destroyed inside close(), taking the DeviceManager
// and every Device with it while the program cache is still populated; freeing that cache in
// ~MeshDeviceImpl then dereferences the dangling IDevice* of every cached program.
std::shared_ptr<ttnn::distributed::MeshDevice> open_mesh_device(
    tt::tt_metal::MetalEnv& env,
    const tt::tt_metal::distributed::MeshShape& shape,
    const std::vector<int>& device_ids) {
    env.get_system_mesh();
    return env.create_mesh_device(
        tt::tt_metal::distributed::MeshDeviceConfig(shape, /*offset=*/std::nullopt, device_ids),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /* num_command_queues=*/1,
        tt::tt_metal::DispatchCoreConfig{});
}

}  // namespace

// ttml::core::MeshDevice builds its own MetalEnv when it opens a device, and only one
// environment for the cluster may exist at a time.
MeshDevice::MeshDevice(const tt::tt_metal::distributed::MeshShape& shape, const std::vector<int>& device_ids) :
    m_env(std::make_unique<tt::tt_metal::MetalEnv>(make_env_descriptor())),
    m_mesh_device(open_mesh_device(*m_env, shape, device_ids)) {
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
