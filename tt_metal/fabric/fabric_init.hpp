// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "device.hpp"
#include "tt_metal.hpp"
#include "tt_metal/fabric/builder/router_channel_layouts.hpp"

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
// of the router across its link. Also the launch's gate: pass phase 1's exception here rather than throwing it, and
// every rank fails together. A single rank just rethrows it.
// Follow-ups: gate where every rank runs (fabric-disabled or managed-mode ranks skip this), a phase-2 gate, error text.
void exchange_fabric_router_channel_layouts(tt::tt_metal::MetalEnvImpl& env_impl, const std::exception_ptr& phase1_error);

// The gate's rule, run on every rank's headers: any failed rank, or ranks on different launches, is fatal on every
// rank. Fabric configs may differ by mesh; the control plane validates them.
void check_fabric_launch_headers(const std::vector<RouterChannelLayouts::LaunchHeader>& headers);

// Phase 2: create the router kernels and compile the device's fabric program.
std::unique_ptr<tt::tt_metal::Program> compile_fabric_program(
    tt::tt_metal::MetalEnvImpl& env_impl, tt::tt_metal::IDevice* device, FabricDeviceBuild build);

// Called once per device while its fabric program is built, after the router kernels are created and before the
// program is compiled. The callee may add kernels, semaphores and runtime args to `fabric_program`; it must not write
// to the device. Devices compile in parallel, so it is called from several threads, one per device, and must touch
// only its own device's program. An empty function (the default) leaves fabric as it is.
using FabricProgramExtension =
    std::function<void(tt::tt_metal::IDevice* device, tt::tt_metal::Program& fabric_program)>;

// Test-only. Process-wide: set before the devices open and cleared after they close. Not thread-safe against a
// concurrent open.
void set_fabric_program_extension(FabricProgramExtension extension);

// Perform additional configuration (writing to specific L1 addresses, etc.) for fabric kernels on this device.
void configure_fabric_cores(tt::tt_metal::MetalEnvImpl& env_impl, tt::tt_metal::IDevice* device);

}  // namespace tt::tt_fabric
