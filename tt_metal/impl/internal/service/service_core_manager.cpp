// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/service/service_core_manager.hpp>
#include "impl/internal/service/service_core_manager_impl.hpp"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/tt_align.hpp>
#include "impl/allocator/algorithms/free_list_opt.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/device.hpp>

#include <llrt/hal.hpp>
#include "llrt/llrt.hpp"
#include "llrt/hal/generated/dev_msgs.hpp"

namespace tt::tt_metal::internal {

namespace {

// Get DRAM aligned L1 range for per-core allocator
std::pair<DeviceAddr, DeviceAddr> l1_service_range(const Hal& hal) {
    const DeviceAddr dram_align = hal.get_alignment(HalMemType::DRAM);
    const DeviceAddr base =
        tt::align(hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED), dram_align);
    const DeviceAddr l1_end = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
                              hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
    return {base, l1_end - base};
}

}  // namespace

// ServiceCoreManagerImpl

ServiceCoreManagerImpl::ServiceCoreManagerImpl(MetalEnvImpl& env) : env_(env) {}

void ServiceCoreManagerImpl::claim(IDevice* device, const std::vector<CoreCoord>& cores) {
    const auto& cluster = env_.get_cluster();
    TT_FATAL(
        cluster.is_ubb_galaxy() || cluster.arch() == tt::ARCH::BLACKHOLE,
        "Service core claims are only supported on Blackhole and UBB Galaxy clusters.");
    TT_FATAL(
        env_.get_rtoptions().get_fast_dispatch(),
        "Service cores can only be claimed while Fast Dispatch is active. "
        "Call initialize_fast_dispatch() before claim().");

    auto& state = devices_[device->id()];
    // Capture FD Grid snapshot. Useful to create truly disjoint-worker-sets if we switch to SD
    // later.
    if (state.cores.empty()) {
        state.fd_compute_grid = device->compute_with_storage_grid_size();
    }

    // Init per-core allocator
    auto [base, size] = l1_service_range(env_.get_hal());
    const DeviceAddr dram_align = env_.get_hal().get_alignment(HalMemType::DRAM);
    for (const auto& core : cores) {
        auto& slot = state.cores[core];
        TT_FATAL(!slot.alloc, "internal::ServiceCoreManager::claim: core {} is already claimed", core);
        slot.alloc = std::make_unique<allocator::FreeListOpt>(
            size, base, dram_align, dram_align, allocator::FreeListOpt::SearchPolicy::FIRST);
        slot.alloc->init();
    }
}

void ServiceCoreManagerImpl::release(IDevice* device, const std::vector<CoreCoord>& cores) {
    auto it = devices_.find(device->id());
    if (it == devices_.end()) {
        log_warning(
            tt::LogMetal,
            "internal::ServiceCoreManager::release: device {} has no claimed service cores; nothing released "
            "(releasing the wrong device?).",
            device->id());
        return;
    }
    for (const auto& core : cores) {
        if (it->second.cores.erase(core) == 0) {
            log_warning(
                tt::LogMetal,
                "internal::ServiceCoreManager::release: core {} was not claimed on device {}; skipped.",
                core,
                device->id());
        }
    }
}

void ServiceCoreManagerImpl::wait_done(IDevice* device, CoreCoord core) const {
    const auto physical_core = device->virtual_core_from_logical_core(core, CoreType::WORKER);
    std::unordered_set<CoreCoord> not_done{physical_core};
    tt::llrt::internal_::wait_until_cores_done(device->id(), dev_msgs::RUN_MSG_GO, not_done);
}

std::unordered_set<CoreCoord> ServiceCoreManagerImpl::claimed_cores(ChipId device_id) const {
    std::unordered_set<CoreCoord> out;
    auto it = devices_.find(device_id);
    if (it != devices_.end()) {
        for (const auto& [c, _] : it->second.cores) {
            out.insert(c);
        }
    }
    return out;
}

std::optional<CoreCoord> ServiceCoreManagerImpl::get_safe_compute_grid(ChipId device_id) const {
    auto it = devices_.find(device_id);
    if (it == devices_.end() || it->second.cores.empty()) {
        return std::nullopt;
    }
    return it->second.fd_compute_grid;
}

std::vector<CoreCoord> ServiceCoreManagerImpl::get_claimable_cores(IDevice* device) const {
    TT_FATAL(
        env_.get_rtoptions().get_fast_dispatch(),
        "get_claimable_cores() requires Fast Dispatch to be active. "
        "Call initialize_fast_dispatch() first.");
    auto available = MetalContext::instance().get_dispatch_core_manager().get_available_dispatch_cores(device->id());
    // Filter out cores already claimed in this session so consecutive calls reflect current state.
    const auto claimed = claimed_cores(device->id());
    std::erase_if(available, [&claimed](const CoreCoord& c) { return claimed.contains(c); });
    TT_FATAL(
        !available.empty(),
        "No claimable dispatch-column cores on device {}. "
        "All dispatch-column cores are in use by FD infra or already claimed as service cores.",
        device->id());
    return available;
}

void ServiceCoreManagerImpl::on_device_close(ChipId device_id) { devices_.erase(device_id); }

bool ServiceCoreManagerImpl::has_any_claims() const {
    for (const auto& [id, state] : devices_) {
        if (!state.cores.empty()) {
            return true;
        }
    }
    return false;
}

void ServiceCoreManagerImpl::mark_launched(ChipId device_id, CoreCoord core) {
    auto dit = devices_.find(device_id);
    if (dit == devices_.end()) {
        return;
    }
    auto cit = dit->second.cores.find(core);
    if (cit == dit->second.cores.end()) {
        return;  // not a claimed service core on this device
    }
    TT_FATAL(
        !cit->second.launched,
        "A service workload is already running on core {} (device {}). A claimed service core accepts a single "
        "enqueue; release() it before enqueueing again.",
        core,
        device_id);
    cit->second.launched = true;
}

bool ServiceCoreManagerImpl::is_service_core(ChipId device_id, CoreCoord core) const {
    auto it = devices_.find(device_id);
    return it != devices_.end() && it->second.cores.contains(core);
}

DeviceAddr ServiceCoreManagerImpl::allocate_l1(IDevice* device, CoreCoord core, size_t size) {
    auto dit = devices_.find(device->id());
    TT_FATAL(
        dit != devices_.end() && dit->second.cores.contains(core),
        "internal::ServiceCoreManager::allocate_l1 called on unclaimed core {}",
        core);
    auto addr = dit->second.cores.at(core).alloc->allocate(size, /*bottom_up=*/false);
    TT_FATAL(addr.has_value(), "internal::ServiceCoreManager: L1 OOM on core {} ({} bytes requested)", core, size);
    return *addr;
}

void ServiceCoreManagerImpl::reserve_l1_to_top(IDevice* device, CoreCoord core, DeviceAddr addr) {
    auto dit = devices_.find(device->id());
    TT_FATAL(
        dit != devices_.end() && dit->second.cores.contains(core),
        "internal::ServiceCoreManager::reserve_l1_to_top called on unclaimed core {}",
        core);
    // Carve out [addr, L1_top) so subsequent allocate_l1() calls don't hand out an address that
    // overlaps externally-owned L1 sitting at the top of this core (e.g. MeshSocket config buffer
    // / data FIFO that the device allocator placed there). The device allocator and this per-core
    // allocator both grow top-down from L1_END with no mutual awareness, so without this
    // reservation they collide. Reserving to the top (rather than each buffer's exact size) covers
    // the whole socket region in one allocation regardless of individual buffer footprints, and
    // must be called before any allocate_l1() on this core so the top range is still free.
    auto& alloc = dit->second.cores.at(core).alloc;
    DeviceAddr top = addr;
    for (const auto& [start, end] : alloc->available_addresses(0)) {
        top = std::max(top, end);
    }
    TT_FATAL(
        top > addr,
        "internal::ServiceCoreManager::reserve_l1_to_top: addr {:#x} is at/above L1 range top {:#x} on core {}",
        static_cast<uint64_t>(addr),
        static_cast<uint64_t>(top),
        core);
    auto reserved = alloc->allocate_at_address(addr, top - addr);
    TT_FATAL(
        reserved.has_value(),
        "internal::ServiceCoreManager::reserve_l1_to_top: could not reserve [{:#x}, {:#x}) on core {} "
        "(out of range or already allocated)",
        static_cast<uint64_t>(addr),
        static_cast<uint64_t>(top),
        core);
}

void ServiceCoreManagerImpl::deallocate_l1(IDevice* device, CoreCoord core, DeviceAddr addr) {
    auto dit = devices_.find(device->id());
    TT_FATAL(
        dit != devices_.end() && dit->second.cores.contains(core),
        "internal::ServiceCoreManager::deallocate_l1 called on unclaimed core {}",
        core);
    dit->second.cores.at(core).alloc->deallocate(addr);
}

size_t ServiceCoreManagerImpl::bytes_available(IDevice* device, CoreCoord core) const {
    auto dit = devices_.find(device->id());
    TT_FATAL(
        dit != devices_.end() && dit->second.cores.contains(core),
        "internal::ServiceCoreManager::bytes_available called on unclaimed core {}",
        core);
    const auto& alloc = dit->second.cores.at(core).alloc;
    size_t total = 0;
    for (const auto& [start, end] : alloc->available_addresses(0)) {
        total += end - start;
    }
    return total;
}

std::optional<DeviceAddr> ServiceCoreManagerImpl::lowest_allocated_address(ChipId device_id, CoreCoord core) const {
    auto dit = devices_.find(device_id);
    if (dit == devices_.end()) {
        return std::nullopt;
    }
    auto cit = dit->second.cores.find(core);
    if (cit == dit->second.cores.end()) {
        return std::nullopt;
    }
    // FreeListOpt tracks this directly (updated on allocate, recomputed on deallocate)
    return cit->second.alloc->lowest_occupied_address();
}

// ServiceCoreManager Public interface

ServiceCoreManager::ServiceCoreManager(MetalEnvImpl& env) : pimpl_(std::make_unique<ServiceCoreManagerImpl>(env)) {}
ServiceCoreManager::~ServiceCoreManager() = default;

std::vector<CoreCoord> ServiceCoreManager::get_claimable_cores(IDevice* device) const {
    return pimpl_->get_claimable_cores(device);
}
void ServiceCoreManager::claim(IDevice* device, const std::vector<CoreCoord>& cores) { pimpl_->claim(device, cores); }
void ServiceCoreManager::release(IDevice* device, const std::vector<CoreCoord>& cores) {
    pimpl_->release(device, cores);
}
std::unordered_set<CoreCoord> ServiceCoreManager::claimed_cores(ChipId device_id) const {
    return pimpl_->claimed_cores(device_id);
}
DeviceAddr ServiceCoreManager::allocate_l1(IDevice* device, CoreCoord core, size_t size) {
    return pimpl_->allocate_l1(device, core, size);
}
void ServiceCoreManager::deallocate_l1(IDevice* device, CoreCoord core, DeviceAddr addr) {
    pimpl_->deallocate_l1(device, core, addr);
}
size_t ServiceCoreManager::bytes_available(IDevice* device, CoreCoord core) const {
    return pimpl_->bytes_available(device, core);
}
void ServiceCoreManager::reserve_l1_to_top(IDevice* device, CoreCoord core, DeviceAddr addr) {
    pimpl_->reserve_l1_to_top(device, core, addr);
}

void ServiceCoreManager::wait_done(IDevice* device, CoreCoord core) const { pimpl_->wait_done(device, core); }

ServiceCoreManagerImpl& ServiceCoreManager::impl() { return *pimpl_; }
const ServiceCoreManagerImpl& ServiceCoreManager::impl() const { return *pimpl_; }

ServiceCoreManager& service_core_manager() { return MetalContext::instance().get_service_core_manager(); }

}  // namespace tt::tt_metal::internal
