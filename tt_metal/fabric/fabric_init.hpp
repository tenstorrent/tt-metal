// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_set>

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
//
// pre_known_dead_channels: ETH channel IDs confirmed problematic by terminate_stale_erisc_routers()
// — either the probe read timed out (physically dead link) or the L1 status word was corrupt (not a
// valid EDMStatus value).  assert_risc_reset_at_core() is skipped entirely for these channels.
// Rationale: assert_risc_reset_at_core() calls read_non_mmio first; on a dead/corrupt channel that
// read times out and leaves a stuck command in the relay ETH core's 4-slot queue (CMD_BUF_SIZE=4).
// With 4 dead channels the queue fills and the last channel's read_non_mmio enters a no-timeout
// while(full) loop → indefinite hang.  See #42429.
bool configure_fabric_cores(
    tt::tt_metal::IDevice* device,
    const std::unordered_set<uint32_t>& pre_known_dead_channels = {});

}  // namespace tt::tt_fabric
