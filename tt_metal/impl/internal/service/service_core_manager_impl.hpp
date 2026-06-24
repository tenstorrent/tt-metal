// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <stddef.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

#include "impl/allocator/algorithms/free_list_opt.hpp"

namespace tt::tt_metal {
class IDevice;
class MetalEnvImpl;
using DeviceAddr = uint64_t;
}  // namespace tt::tt_metal

namespace tt::tt_metal::internal {

// Implementation behind the thin ServiceCoreManager facade. Lives under impl/ so only internal
// translation units (dispatch routing, placement validation, device teardown) can reach the
// methods that are not part of the user-facing surface.
//
// Per-Device state holds:
// 1. Per-core allocator for all cores per device
// 2. Snapshot of FD worker grid per device
// 3. If a core already has a service launched on it
class ServiceCoreManagerImpl {
public:
    // Stores a reference to the MetalEnvImpl that owns the cluster, rtoptions and hal we query
    // (mirrors dispatch_core_manager). The env outlives the MetalContext that owns us.
    explicit ServiceCoreManagerImpl(MetalEnvImpl& env);

    // ── User-facing surface (forwarded by ServiceCoreManager) ──────────────────────────────────
    std::vector<CoreCoord> get_claimable_cores(IDevice* device) const;
    void claim(IDevice* device, const std::vector<CoreCoord>& cores);
    void release(IDevice* device, const std::vector<CoreCoord>& cores);
    void wait_done(IDevice* device, CoreCoord core) const;
    std::unordered_set<CoreCoord> claimed_cores(ChipId device_id) const;
    DeviceAddr allocate_l1(IDevice* device, CoreCoord core, size_t size);
    // Reserve [addr, L1_top) so a later allocate_l1() won't overlap externally-owned L1 at the top
    // of this core. Must be called before any allocate_l1() on the core. See the facade declaration.
    void reserve_l1_to_top(IDevice* device, CoreCoord core, DeviceAddr addr);
    void deallocate_l1(IDevice* device, CoreCoord core, DeviceAddr addr);
    size_t bytes_available(IDevice* device, CoreCoord core) const;

    // ── Internal-only surface (not exposed on the facade) ──────────────────────────────────────

    // Called from Device::close() to drop all claims for a device.
    void on_device_close(ChipId device_id);

    // Returns the lowest start address currently handed out by the per-core L1 allocator,
    // or nullopt if nothing has been allocated on that core yet.
    // Because allocate_l1() allocates top-down (bottom_up=false), this is the frontier —
    // the lowest point service buffers have reached. Used by validate_circular_buffer_region
    // to detect collisions with CBs that grow up from DEFAULT_UNRESERVED.
    std::optional<DeviceAddr> lowest_allocated_address(ChipId device_id, CoreCoord core) const;

    // Returns the FD-mode compute grid snapshotted at first claim() for a device, or
    // nullopt if no service cores are currently claimed. Used to cap
    // compute_with_storage_grid_size() in SD mode so SD workloads don't accidentally
    // target dispatch column cores running persistent service kernels - preserving the
    // disjoint worker-set invariant between the regular worker grid and service cores.
    std::optional<CoreCoord> get_safe_compute_grid(ChipId device_id) const;

    // Called on every EnqueueMeshWorkload - common case is no claims, in which case routing is skipped.
    bool has_any_claims() const;
    // True if `core` is claimed as a service core on `device_id`. Used at enqueue time to route
    // programs to the SD path and to device-scope placement/CB validation.
    bool is_service_core(ChipId device_id, CoreCoord core) const;
    // Enforces launch-once: marks a claimed service core as launched, TT_FATALing if it
    // already was. A claimed core accepts a single service-workload enqueue until release()
    // clears the claim. No-op for cores not claimed on device_id (e.g. worker cores).
    void mark_launched(ChipId device_id, CoreCoord core);

    ServiceCoreManagerImpl(const ServiceCoreManagerImpl&) = delete;
    ServiceCoreManagerImpl& operator=(const ServiceCoreManagerImpl&) = delete;

private:
    struct CoreState {
        std::unique_ptr<allocator::FreeListOpt> alloc;
        bool launched = false;  // a service workload has been enqueued on this core (launch-once)
    };
    struct DeviceServiceState {
        std::unordered_map<CoreCoord, CoreState> cores;
        CoreCoord fd_compute_grid;
    };

    MetalEnvImpl& env_;
    std::unordered_map<ChipId, DeviceServiceState> devices_;
};

}  // namespace tt::tt_metal::internal
