// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <stddef.h>
#include <unordered_set>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal {
class IDevice;
class MetalEnvImpl;
using DeviceAddr = uint64_t;
}  // namespace tt::tt_metal

namespace tt::tt_metal::internal {

class ServiceCoreManagerImpl;

// Internal, unstable API - read the stability/usage conditions in tt_metal/api/internal/README.md
// before depending on anything here.
//
// Manages reservation of free FD dispatch-column cores for long-running service kernels,
// and per-core L1 allocation for those cores.
//
// A "service core" is an FD dispatch-column core that FD has not allocated to its own
// pipeline (prefetcher, dispatcher, fabric-mux, etc.). Services run persistent SD-launched
// kernels on these cores concurrently with FD workloads on the regular worker grid — two
// disjoint worker spaces, no overlap.
//
// Lives on MetalContext (one instance per context, keyed internally by ChipId). Obtain it via
// MetalContext::instance().get_service_core_manager(); do not construct it directly.
//
// This is the thin, user-facing surface. Internal dispatch routing / placement-validation
// methods live on ServiceCoreManagerImpl, reachable only through impl() by translation units
// that include impl/internal/service/service_core_manager_impl.hpp.
//
// Supported configurations: Blackhole or UBB Galaxy, with an active manual FD session.
// claim() and get_claimable_cores() TT_FATAL outside these conditions.
//
// NOTE: not thread-safe api. Expected usage is sequential calls from the main thread during
// application setup/teardown.
//
// No-mixing contract: a program runs entirely on claimed service cores or entirely on the worker
// grid - never both. And a MeshWorkload is entirely a service workload (all programs on service
// cores) or entirely a normal one - the two kinds cannot be combined in one workload.
// EnqueueMeshWorkload TT_FATALs on either violation.
//
// Launch once contract: For now, a claimed service core accepts exactly ONE service workload
// enqueue. A second EnqueueMeshWorkload targeting an already launched core TT_FATALs, until the
// core is released - regardless of whether the service kernel on it has actually finished.
// Re-enqueue therefore requires release() then claim() again. This will most likely change in the future.
//
// Runtime flow:
//
//   // 1. Launch App in FD
//
//   // 2. Claim service cores (must be done while FD is active)
//   auto& svc = MetalContext::instance().get_service_core_manager();
//   auto claimable = svc.get_claimable_cores(device);
//   svc.claim(device, claimable);
//
//   // 3. Allocate per-core L1 and build the service workload
//   for (auto& core : claimable)
//       counter_addrs[core] = svc.allocate_l1(device, core, sizeof(uint32_t));
//
//   // add_program just records the program; it has no device handle so it does no routing.
//   // EnqueueMeshWorkload knows the device, so that is where programs targeting claimed service
//   // cores are split off to the SD path while regular programs go via FD.
//   // User is responsible for service kernel lifetime (fire and forget).
//   service_workload.add_program(device_range, std::move(service_program));
//   EnqueueMeshWorkload(mesh_cq, service_workload, false);
//
//   // 4. Regular FD workloads run transparently alongside service kernels
//   for (int i = 0; i < N; i++)
//       EnqueueMeshWorkload(mesh_cq, standard_fd_workload, false);
//
//   // 5. Stop service kernel and release cores
//   WriteToDeviceL1(device, stop_core, stop_addr, 1);  // or via GlobalSemaphore
//   svc.release(device, claimable);
//
class ServiceCoreManager {
public:
    // Constructed by MetalContext. Stores a reference to the MetalEnvImpl that owns the cluster,
    // rtoptions and hal it queries (mirrors dispatch_core_manager).
    explicit ServiceCoreManager(MetalEnvImpl& env);
    ~ServiceCoreManager();

    // Returns dispatch-column cores not yet allocated to FD infra or claimed as service cores.
    // TT_FATALs if:
    //   - no manual FD session is active (the pool is only fully partitioned after
    //     initialize_fast_dispatch() completes; calling before that returns a misleading set)
    //   - no cores are available (all dispatch-column cores are in use by FD or already claimed)
    // Use this immediately before claim() — the query and then claim is the intended idiom.
    std::vector<CoreCoord> get_claimable_cores(IDevice* device) const;

    // Reserve one or more free FD-column cores for service use. Constructs a per-core L1
    // allocator for each claimed core. Claimed cores are explicitly removed from
    // dispatch_core_manager's pool on every initialize_fast_dispatch() so FD never
    // allocates them, regardless of how many FD teardown/re-init cycles occur.
    // TT_FATALs if:
    //   - arch is not Blackhole or UBB Galaxy
    //   - no manual FD session is active (initialize_fast_dispatch not yet called)
    //   - any core in the list is already claimed
    void claim(IDevice* device, const std::vector<CoreCoord>& cores);

    // Release one or more claimed cores and destroy their L1 allocators. All addresses
    // handed out by allocate_l1() for these cores become invalid after this call.
    // Silent no-op for unclaimed cores — safe to call in teardown/destructor paths.
    // Caller contract: the service kernel must already be stopped before release() - the
    // runtime cannot detect completion of a persistent (looping) kernel. Pair with wait_done()
    // when the caller needs to block until the kernel has actually exited.
    void release(IDevice* device, const std::vector<CoreCoord>& cores);

    // Block until the persistent service kernel on `core` of `device` has returned (left the GO
    // run-state). Intended for use just before release(), so the per-core L1 isn't torn down
    // while the kernel is still running.
    void wait_done(IDevice* device, CoreCoord core) const;

    // Returns the set of currently claimed cores for a device.
    std::unordered_set<CoreCoord> claimed_cores(ChipId device_id) const;

    // Per-core L1 allocator. Valid only for currently claimed cores - TT_FATALs otherwise.
    // Alignment is fixed at claim() time to HalMemType::DRAM so NoC read/write to allocations
    // are always valid (mirrors BankManager's lockstep L1 path). TT_FATALs on OOM.
    // Each service core owns a completely independent L1 range - no interaction with
    // the worker-grid BankManager.
    // Allocates top-down (from L1_END downward) so service buffers and CBs (which grow up
    // from DEFAULT_UNRESERVED) stay in disjoint zones — same convention as worker-core L1 buffers.
    DeviceAddr allocate_l1(IDevice* device, CoreCoord core, size_t size);
    void deallocate_l1(IDevice* device, CoreCoord core, DeviceAddr addr);
    size_t bytes_available(IDevice* device, CoreCoord core) const;

    // Access to the internal implementation. Include
    // impl/internal/service/service_core_manager_impl.hpp to use the returned reference
    ServiceCoreManagerImpl& impl();
    const ServiceCoreManagerImpl& impl() const;

    ServiceCoreManager(const ServiceCoreManager&) = delete;
    ServiceCoreManager& operator=(const ServiceCoreManager&) = delete;

private:
    std::unique_ptr<ServiceCoreManagerImpl> pimpl_;
};

ServiceCoreManager& service_core_manager();

}  // namespace tt::tt_metal::internal
