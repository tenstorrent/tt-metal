// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>

#include "device.hpp"
#include "tt_metal.hpp"

namespace tt::tt_metal {
class DispatchCoreConfig;
class MetalEnvImpl;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {

class FabricBuildSession;
class FabricBuilder;

// Start the build session for one fabric launch (from the registered session factory, or the default builder's).
// Call once per launch, before building any device, and keep it alive until fabric is torn down.
// Returns nullptr when the fabric config does not build tt-fabric routers.
std::unique_ptr<FabricBuildSession> begin_fabric_build_session(tt::tt_metal::MetalEnvImpl& env_impl);

// A device's fabric program between the two build phases: routers created and wired, router kernels not yet created.
struct FabricDeviceBuild {
    std::unique_ptr<tt::tt_metal::Program> program;
    std::shared_ptr<FabricBuilder> builder;
};

// Phase 1: create and wire this device's routers, publishing their channel layouts. nullopt when the device has no
// routers or the fabric config does not build tt-fabric routers.
std::optional<FabricDeviceBuild> build_fabric_routers(
    tt::tt_metal::MetalEnvImpl& env_impl,
    const tt::tt_metal::DispatchCoreConfig& dispatch_core_config,
    tt::tt_metal::IDevice* device,
    const FabricBuildSession& session);

// Between the phases, once per launch on every rank (a collective across hosts): gives each router the channel layout
// of the router across its link.
void exchange_fabric_router_channel_layouts(tt::tt_metal::MetalEnvImpl& env_impl);

// Phase 2: create the router kernels and compile the device's fabric program.
std::unique_ptr<tt::tt_metal::Program> compile_fabric_program(
    tt::tt_metal::MetalEnvImpl& env_impl, tt::tt_metal::IDevice* device, FabricDeviceBuild build);

// Perform additional configuration (writing to specific L1 addresses, etc.) for fabric kernels on this device.
void configure_fabric_cores(tt::tt_metal::MetalEnvImpl& env_impl, tt::tt_metal::IDevice* device);

}  // namespace tt::tt_fabric
