// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include "device.hpp"
#include "tt_metal.hpp"

namespace tt::tt_metal {
class DispatchCoreConfig;
class MetalEnvImpl;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {

// Compile fabric kernels needed to support scaleout systems.
std::unique_ptr<tt::tt_metal::Program> create_and_compile_fabric_program(
    tt::tt_metal::MetalEnvImpl& env_impl,
    const tt::tt_metal::DispatchCoreConfig& dispatch_core_config,
    tt::tt_metal::IDevice* device);

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
