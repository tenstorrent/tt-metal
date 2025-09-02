// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.hpp"
#include "tt_metal.hpp"

namespace tt::tt_fabric {

// Compile fabric kernels needed to support scaleout systems.
std::unique_ptr<tt::tt_metal::Program> create_and_compile_fabric_program(tt::tt_metal::IDevice* device);

// Perform additional configuration (writing to specific L1 addresses, etc.) for fabric kernels on this device.
void configure_fabric_cores(tt::tt_metal::IDevice* device);

}  // namespace tt::tt_fabric
