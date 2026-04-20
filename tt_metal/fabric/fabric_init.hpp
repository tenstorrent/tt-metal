// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.hpp"
#include "tt_metal.hpp"

namespace tt::tt_fabric {

// Compile fabric kernels needed to support scaleout systems.
std::unique_ptr<tt::tt_metal::Program> create_and_compile_fabric_program(tt::tt_metal::IDevice* device);

// Perform additional configuration (writing to specific L1 addresses, etc.) for fabric kernels on this device.
// Returns true if all active ETH channels completed their soft reset successfully; false if any channel's
// assert/deassert_risc_reset_at_core threw (e.g. timeout on a dead remote chip).  The caller should skip
// operations that require reads from the device (e.g. l1_barrier) when this returns false, because those
// reads will also hang/timeout on the same dead channels.
bool configure_fabric_cores(tt::tt_metal::IDevice* device);

}  // namespace tt::tt_fabric
