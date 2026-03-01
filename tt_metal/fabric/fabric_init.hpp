// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.hpp"
#include "tt_metal.hpp"
#include <memory>
#include <vector>

namespace tt::tt_fabric {

class FabricBuilder;

/**
 * Result of fabric build phase 1.
 * Contains the builder (with servicing-aware allocators) and the program
 * for use in phase 2 after peer state exchange.
 */
struct FabricBuildPhase1Result {
    std::unique_ptr<FabricBuilder> builder;
    std::unique_ptr<tt::tt_metal::Program> program;
    bool has_routers = false;
};

// Build and compile the fabric program for a device.
// When all_devices is non-empty, uses a two-phase build with an internal barrier
// to coordinate allocator state exchange across all devices before compilation.
// When all_devices is empty, uses the single-phase full build.
std::unique_ptr<tt::tt_metal::Program> create_and_compile_fabric_program(
    tt::tt_metal::IDevice* device, const std::vector<tt::tt_metal::IDevice*>& all_devices = {});

// Perform additional configuration (writing to specific L1 addresses, etc.) for fabric kernels on this device.
void configure_fabric_cores(tt::tt_metal::IDevice* device);

}  // namespace tt::tt_fabric
