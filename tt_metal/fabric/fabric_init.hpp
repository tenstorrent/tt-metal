// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.hpp"
#include "tt_metal.hpp"

namespace tt::tt_metal {
class DispatchCoreConfig;
class MetalEnvImpl;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {

class FabricBuildSession;

// Start the build session for one fabric launch (from the registered session factory, or the default builder's).
// Call once per launch, before compiling any device, and keep it alive until fabric is torn down.
// Returns nullptr when the fabric config does not build tt-fabric routers.
std::unique_ptr<FabricBuildSession> begin_fabric_build_session(tt::tt_metal::MetalEnvImpl& env_impl);

// Compile fabric kernels needed to support scaleout systems, using this launch's build session.
std::unique_ptr<tt::tt_metal::Program> create_and_compile_fabric_program(
    tt::tt_metal::MetalEnvImpl& env_impl,
    const tt::tt_metal::DispatchCoreConfig& dispatch_core_config,
    tt::tt_metal::IDevice* device,
    const FabricBuildSession& session);

// Perform additional configuration (writing to specific L1 addresses, etc.) for fabric kernels on this device.
void configure_fabric_cores(tt::tt_metal::MetalEnvImpl& env_impl, tt::tt_metal::IDevice* device);

}  // namespace tt::tt_fabric
