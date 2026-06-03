// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/internal/service/service_core_manager.hpp>

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include <tt_stl/assert.hpp>
#include <tt-metalium/tt_align.hpp>
#include "impl/allocator/algorithms/free_list_opt.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include <tt-metalium/device.hpp>

#include <llrt/hal.hpp>
#include "llrt/llrt.hpp"
#include "llrt/hal/generated/dev_msgs.hpp"

namespace tt::tt_metal::internal {

// Per-Device state holds:
// 1. Per-core allocator for all cores per device
// 2. Snapshot of FD worker grid per device
struct ServiceCoreManager::Impl {
    struct CoreState {
        std::unique_ptr<allocator::FreeListOpt> alloc;
    };
    struct DeviceServiceState {
        std::unordered_map<CoreCoord, CoreState> cores;
        CoreCoord fd_compute_grid;
    };
    std::unordered_map<ChipId, DeviceServiceState> devices;
};

ServiceCoreManager::ServiceCoreManager() : impl_(std::make_unique<Impl>()) {}
ServiceCoreManager::~ServiceCoreManager() = default;

// ServiceCoreManager singleton
ServiceCoreManager& ServiceCoreManager::get() {
    static ServiceCoreManager instance;
    return instance;
}

namespace {

// Get DRAM aligned L1 range for per-core allocator
std::pair<DeviceAddr, DeviceAddr> l1_service_range() {
    const auto& hal = MetalContext::instance().hal();
    const DeviceAddr dram_align = hal.get_alignment(HalMemType::DRAM);
    const DeviceAddr base =
        tt::align(hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED), dram_align);
    const DeviceAddr l1_end = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
                              hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
    return {base, l1_end - base};
}

}  // namespace

void ServiceCoreManager::claim(IDevice* device, const std::vector<CoreCoord>& cores) {
    const auto& cluster = MetalContext::instance().get_cluster();
    TT_FATAL(
        cluster.is_ubb_galaxy() || cluster.arch() == tt::ARCH::BLACKHOLE,
        "Service core claims are only supported on Blackhole and UBB Galaxy clusters.");
    TT_FATAL(
        MetalContext::instance().rtoptions().get_fast_dispatch(),
        "Service cores can only be claimed while Fast Dispatch is active. "
        "Call initialize_fast_dispatch() before claim().");

    auto& state = impl_->devices[device->id()];
    // Capture FD Grid snapshot. Useful to create truly disjoint-worker-sets if we switch to SD
    // later.
    if (state.cores.empty()) {
        state.fd_compute_grid = device->compute_with_storage_grid_size();
    }

    // Init per-core allocator
    auto [base, size] = l1_service_range();
    const DeviceAddr dram_align = MetalContext::instance().hal().get_alignment(HalMemType::DRAM);
    for (const auto& core : cores) {
        auto& slot = state.cores[core];
        TT_FATAL(!slot.alloc, "internal::ServiceCoreManager::claim: core {} is already claimed", core);
        slot.alloc = std::make_unique<allocator::FreeListOpt>(
            size, base, dram_align, dram_align, allocator::FreeListOpt::SearchPolicy::FIRST);
        slot.alloc->init();
    }
}

void ServiceCoreManager::release(IDevice* device, const std::vector<CoreCoord>& cores) {
    auto it = impl_->devices.find(device->id());
    if (it == impl_->devices.end()) {
        return;
    }
    for (const auto& core : cores) {
        it->second.cores.erase(core);
    }
}

std::unordered_set<CoreCoord> ServiceCoreManager::claimed_cores(ChipId device_id) const {
    std::unordered_set<CoreCoord> out;
    auto it = impl_->devices.find(device_id);
    if (it != impl_->devices.end()) {
        for (const auto& [c, _] : it->second.cores) {
            out.insert(c);
        }
    }
    return out;
}

std::optional<CoreCoord> ServiceCoreManager::get_safe_compute_grid(ChipId device_id) const {
    auto it = impl_->devices.find(device_id);
    if (it == impl_->devices.end() || it->second.cores.empty()) {
        return std::nullopt;
    }
    return it->second.fd_compute_grid;
}

std::vector<CoreCoord> ServiceCoreManager::get_claimable_cores(IDevice* device) const {
    TT_FATAL(
        MetalContext::instance().rtoptions().get_fast_dispatch(),
        "get_claimable_cores() requires Fast Dispatch to be active. "
        "Call initialize_fast_dispatch() first.");
    auto available = MetalContext::instance().get_dispatch_core_manager().get_available_dispatch_cores(device->id());
    // Filter out cores already claimed in this session so consecutive calls reflect current state.
    const auto claimed = claimed_cores(device->id());
    if (!claimed.empty()) {
        std::vector<CoreCoord> out;
        out.reserve(available.size());
        for (const auto& c : available) {
            if (!claimed.count(c)) {
                out.push_back(c);
            }
        }
        available = std::move(out);
    }
    TT_FATAL(
        !available.empty(),
        "No claimable dispatch-column cores on device {}. "
        "All dispatch-column cores are in use by FD infra or already claimed as service cores.",
        device->id());
    return available;
}

void ServiceCoreManager::on_device_close(ChipId device_id) { impl_->devices.erase(device_id); }

bool ServiceCoreManager::has_any_claims() const {
    for (const auto& [id, state] : impl_->devices) {
        if (!state.cores.empty()) {
            return true;
        }
    }
    return false;
}

bool ServiceCoreManager::is_service_core(CoreCoord core) const {
    for (const auto& [id, state] : impl_->devices) {
        if (state.cores.contains(core)) {
            return true;
        }
    }
    return false;
}

void ServiceCoreManager::wait_done(IDevice* device, CoreCoord core) const {
    auto physical_core = device->virtual_core_from_logical_core(core, CoreType::WORKER);
    std::unordered_set<CoreCoord> cores{physical_core};
    tt::llrt::internal_::wait_until_cores_done(device->id(), dev_msgs::RUN_MSG_GO, cores);
}

DeviceAddr ServiceCoreManager::allocate_l1(IDevice* device, CoreCoord core, size_t size) {
    auto dit = impl_->devices.find(device->id());
    TT_FATAL(
        dit != impl_->devices.end() && dit->second.cores.contains(core),
        "internal::ServiceCoreManager::allocate_l1 called on unclaimed core {}",
        core);
    auto addr = dit->second.cores.at(core).alloc->allocate(size, /*bottom_up=*/false);
    TT_FATAL(addr.has_value(), "internal::ServiceCoreManager: L1 OOM on core {} ({} bytes requested)", core, size);
    return *addr;
}

void ServiceCoreManager::deallocate_l1(IDevice* device, CoreCoord core, DeviceAddr addr) {
    auto dit = impl_->devices.find(device->id());
    TT_FATAL(
        dit != impl_->devices.end() && dit->second.cores.contains(core),
        "internal::ServiceCoreManager::deallocate_l1 called on unclaimed core {}",
        core);
    dit->second.cores.at(core).alloc->deallocate(addr);
}

size_t ServiceCoreManager::bytes_available(IDevice* device, CoreCoord core) const {
    auto dit = impl_->devices.find(device->id());
    TT_FATAL(
        dit != impl_->devices.end() && dit->second.cores.contains(core),
        "internal::ServiceCoreManager::bytes_available called on unclaimed core {}",
        core);
    const auto& alloc = dit->second.cores.at(core).alloc;
    size_t total = 0;
    for (const auto& [start, end] : alloc->available_addresses(0)) {
        total += end - start;
    }
    return total;
}

std::optional<DeviceAddr> ServiceCoreManager::lowest_allocated_address(ChipId device_id, CoreCoord core) const {
    auto dit = impl_->devices.find(device_id);
    if (dit == impl_->devices.end()) {
        return std::nullopt;
    }
    auto cit = dit->second.cores.find(core);
    if (cit == dit->second.cores.end()) {
        return std::nullopt;
    }
    const auto ranges = cit->second.alloc->allocated_addresses();
    if (ranges.empty()) {
        return std::nullopt;
    }
    DeviceAddr lo = ranges.front().first;
    for (const auto& [start, end] : ranges) {
        lo = std::min(lo, start);
    }
    return lo;
}

}  // namespace tt::tt_metal::internal
