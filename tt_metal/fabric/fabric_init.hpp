// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <unordered_set>

#include "device.hpp"
#include "tt_metal.hpp"

namespace tt::tt_fabric {

// Compile fabric kernels needed to support scaleout systems.
std::unique_ptr<tt::tt_metal::Program> create_and_compile_fabric_program(tt::tt_metal::IDevice* device);

// Result returned by configure_fabric_cores().
// all_channels_healthy: true iff every active ETH channel completed its soft reset.
// newly_dead_channels:  channels that NEWLY failed soft reset in this call — i.e. not in
//                       pre_known_dead_channels.  Empty when the only dead channels were
//                       already known before configure_fabric_cores() was called.
struct FabricCoresHealth {
    bool all_channels_healthy;
    std::unordered_set<uint32_t> newly_dead_channels;
};

// Perform additional configuration (writing to specific L1 addresses, etc.) for fabric kernels on this device.
// Returns FabricCoresHealth describing per-channel health.  When all_channels_healthy is false:
//   - newly_dead_channels is non-empty → unexpected new failure; caller should TT_THROW.
//   - newly_dead_channels is empty     → all failures were pre-confirmed in pre_known_dead_channels;
//                                        caller may continue in degraded mode with a warning.
//
// pre_known_dead_channels: ETH channel IDs confirmed problematic by terminate_stale_erisc_routers()
// — either the probe read timed out (physically dead link) or the L1 status word was corrupt (not a
// valid EDMStatus value).  assert_risc_reset_at_core() is skipped entirely for these channels.
// Rationale: assert_risc_reset_at_core() calls read_non_mmio first; on a dead/corrupt channel that
// read times out and leaves a stuck command in the relay ETH core's 4-slot queue (CMD_BUF_SIZE=4).
// With 4 dead channels the queue fills and the last channel's read_non_mmio enters a no-timeout
// while(full) loop → indefinite hang.  See #42429.
FabricCoresHealth configure_fabric_cores(
    tt::tt_metal::IDevice* device,
    const std::unordered_set<uint32_t>& pre_known_dead_channels = {});

// ---------------------------------------------------------------------------
// Test seam: fault-injection into configure_fabric_cores() for Scenario W
//
// When set (non-null), configure_fabric_cores() calls this function instead of
// (or immediately before) cluster.assert_risc_reset_at_core() for each active
// ETH channel.  If the function throws, the catch block is exercised:
//   - dead_channels.insert(router_chan)
//   - newly_dead_channels.insert(router_chan)
//   - all_channels_healthy = false
//
// Signature: void(tt::tt_metal::IDevice* device, uint32_t eth_chan_id)
// The function may throw any std::exception to trigger the catch path.
//
// Thread-local so parallel test workers on different threads do not interfere.
// Set before MeshDevice::create(), cleared immediately after via
// clear_configure_cores_inject_fn().
//
// PRODUCTION: s_configure_cores_inject_fn_ is default-constructed (empty
// std::function) on every thread — the if-check is a single bool on an
// inline-initialized thread_local, zero overhead in non-test builds.
// ---------------------------------------------------------------------------
using ConfigureFabricCoresInjectFn = std::function<void(tt::tt_metal::IDevice*, uint32_t /*eth_chan_id*/)>;

// Set/clear the per-thread configure-cores inject seam.
// NOT thread-safe with concurrent callers on the same thread (tests call this
// single-threaded before MeshDevice::create()).
void set_configure_cores_inject_fn(ConfigureFabricCoresInjectFn fn);
void clear_configure_cores_inject_fn();

}  // namespace tt::tt_fabric
