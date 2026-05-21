// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <unordered_set>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal {
class IDevice;
using DeviceAddr = uint64_t;
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental::service {

// These functions are NOT thread-safe. Expected usage is sequential calls from
// the main thread during application setup/teardown.
//
// Arch/mode gate: claim() and allocate_l1() TT_FATAL unless the cluster is BH or
// UBB Galaxy AND a manual FD session is currently active (i.e. initialize_fast_dispatch
// has been called and terminate_fast_dispatch has not yet been called).

// --- Reservation ---

// Reserve one or more free FD-column cores for service use.
// TT_FATALs if any core in the list is already claimed. Use is_claimed() to check first.
// After claim(), the pool filter registered with dispatch_core_manager ensures
// subsequent initialize_fast_dispatch() calls never allocate these cores to FD.
void claim(IDevice* device, const std::vector<CoreCoord>& cores);

// Release one or more claimed cores. Silent no-op for any core that is not currently
// claimed (safe to call in teardown/destructor paths regardless of prior state).
void release(IDevice* device, const std::vector<CoreCoord>& cores);

bool is_claimed(IDevice* device, CoreCoord core);

std::unordered_set<CoreCoord> claimed_cores(ChipId device_id);

// Called from Device::close() to drop all claims for a device.
void on_device_close(ChipId device_id);

// --- Per-core L1 allocator ---
// Valid only for currently-claimed cores. TT_FATALs otherwise.
// Alignment is fixed at claim() time to HalMemType::DRAM alignment so NoC rd/wr
// to allocations are valid (mirrors BankManager's lockstep L1 path).
// TT_FATALs on OOM.

DeviceAddr allocate_l1(IDevice* device, CoreCoord core, size_t size);
void deallocate_l1(IDevice* device, CoreCoord core, DeviceAddr addr);
size_t bytes_available(IDevice* device, CoreCoord core);

}  // namespace tt::tt_metal::experimental::service
