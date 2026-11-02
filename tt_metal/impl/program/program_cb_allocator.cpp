// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "program_cb_allocator.hpp"

#include <algorithm>
#include <optional>
#include <unistd.h>
#include <vector>

#include <graph_tracking.hpp>
#include <tt_stl/assert.hpp>
#include "allocator.hpp"
#include "buffer_types.hpp"
#include "core_coord.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/buffers/circular_buffer.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/device/device_impl.hpp"
#include "impl/internal/service/service_core_manager_impl.hpp"
#include "impl/memory_tracking/memory_stats_shm.hpp"
#include "program_impl.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"
#include <internal/service/service_core_manager.hpp>

namespace tt::tt_metal::detail {

void Metal1CircularBufferAllocator::register_cores(const CoreRangeSet& cores) {
    register_ephemeral_allocation(cb_allocation_metadata_, cores);
}

void Metal1CircularBufferAllocator::release_seals() { cb_allocation_metadata_.seals.clear(); }

void Metal1CircularBufferAllocator::invalidate() {
    // Set unconditionally, before the early return below, so compile_and_allocate re-runs.
    owner_.compile_and_allocate_needed_ = true;
    if (local_circular_buffer_allocation_needed_) {
        return;
    }
    invalidate_ephemeral_allocation_metadata(cb_allocation_metadata_);
    local_circular_buffer_allocation_needed_ = true;
}


void Metal1CircularBufferAllocator::allocate(const IDevice* device) {
    TTZoneScopedD(PROGRAM);

    // If device is a MeshDevice, we need to track all its sub-devices
    std::vector<const IDevice*> devices_to_track;
    const tt::tt_metal::distributed::MeshDevice* mesh_device =
        dynamic_cast<const tt::tt_metal::distributed::MeshDevice*>(device);
    if (mesh_device != nullptr) {
        // Mesh device: track all sub-devices
        const auto sub_devices = mesh_device->impl().get_devices();
        devices_to_track.reserve(sub_devices.size());
        for (IDevice* sub_device : sub_devices) {
            devices_to_track.push_back(sub_device);
        }
    } else {
        // Single device
        devices_to_track.push_back(device);
    }

    // Track which devices are NEW (not already tracked)
    std::vector<const IDevice*> new_devices;
    new_devices.reserve(devices_to_track.size());
    for (const IDevice* dev : devices_to_track) {
        auto [iter, inserted] = cb_devices_.insert(dev);
        if (inserted) {
            new_devices.push_back(dev);
        }
    }

    // If CB layout already calculated, skip allocation but report for new devices
    if (not local_circular_buffer_allocation_needed_) {
        // Report CB allocations for any NEW devices (using cached addresses)
        if (!new_devices.empty() && !owner_.builder().circular_buffers().empty()) {
            for (const IDevice* dev : new_devices) {
                for (const auto& circular_buffer : owner_.builder().circular_buffers()) {
                    if (!circular_buffer->globally_allocated()) {
                        tt::tt_metal::GraphTracker::instance().track_allocate_cb(
                            circular_buffer->core_ranges(),
                            circular_buffer->address(),
                            circular_buffer->size(),
                            circular_buffer->globally_allocated(),
                            dev);
                    }
                }

                // Also register program with the NEW device
                auto* device_obj = dynamic_cast<Device*>(const_cast<IDevice*>(dev));
                if (device_obj) {
                    device_obj->register_program(&owner_);
                    if (device_obj->get_shm_stats_provider()) {
                        device_obj->get_shm_stats_provider()->update_from_allocator(device_obj, getpid());
                    }
                }
            }
        }
        return;
    }

    for (const auto& circular_buffer : owner_.builder().circular_buffers()) {
        if (circular_buffer->globally_allocated()) {
            // Track globally allocated CBs too (they use L1 memory allocated via the allocator)
            for (const IDevice* dev : devices_to_track) {
                tt::tt_metal::GraphTracker::instance().track_allocate_cb(
                    circular_buffer->core_ranges(),
                    circular_buffer->address(),
                    circular_buffer->size(),
                    circular_buffer->globally_allocated(),
                    dev);
            }
            continue;
        }

        const uint64_t computed_addr = allocate_dfb(
            cb_allocation_metadata_,
            circular_buffer->core_ranges(),
            circular_buffer->size(),
            device->allocator()->get_alignment(BufferType::DRAM));
        // Report CB allocation for ALL devices being tracked
        for (const IDevice* dev : devices_to_track) {
            tt::tt_metal::GraphTracker::instance().track_allocate_cb(
                circular_buffer->core_ranges(),
                computed_addr,
                circular_buffer->size(),
                circular_buffer->globally_allocated(),
                dev);
        }
        circular_buffer->set_locally_allocated_address(computed_addr);
    }

    // Register program ONLY with NEW devices (prevents duplicate registration)
    for (const IDevice* dev : new_devices) {
        auto* device_obj = dynamic_cast<Device*>(const_cast<IDevice*>(dev));
        if (device_obj) {
            device_obj->register_program(&owner_);
            // Update locally-allocated CB stats via query (accurate even for cached programs)
            if (device_obj->get_shm_stats_provider()) {
                device_obj->get_shm_stats_provider()->update_from_allocator(device_obj, getpid());
            }
        }
    }
    local_circular_buffer_allocation_needed_ = false;
}


namespace {
void merge_cb_stats_intervals(
    std::vector<std::pair<uint64_t, uint64_t>>& merged, const std::vector<std::pair<uint64_t, uint64_t>>& incoming) {
    for (const auto& region : incoming) {
        // Ends increase with starts because the accumulator contains disjoint intervals.
        auto first =
            std::lower_bound(merged.begin(), merged.end(), region.first, [](const auto& existing, uint64_t start) {
                return existing.second < start;
            });
        if (first != merged.end() && first->first <= region.first && first->second >= region.second) {
            continue;
        }
        auto combined = region;
        auto last = first;
        while (last != merged.end() && last->first <= combined.second) {
            combined.first = std::min(combined.first, last->first);
            combined.second = std::max(combined.second, last->second);
            ++last;
        }
        if (first == last) {
            merged.insert(first, combined);
        } else {
            *first = combined;
            merged.erase(first + 1, last);
        }
    }
}
}  // namespace

void Metal1CircularBufferAllocator::merge_l1_regions_by_core_range(
    std::map<CoreRange, std::vector<std::pair<uint64_t, uint64_t>>>& regions_per_range) const {
    for (const auto& cb_allocator : cb_allocation_metadata_.regions) {
        merge_cb_stats_intervals(regions_per_range[cb_allocator.core_range], cb_allocator.l1_regions);
    }
}

std::map<CoreCoord, std::vector<std::pair<uint64_t, uint64_t>>> Metal1CircularBufferAllocator::expand_l1_regions_per_core(
    const std::map<CoreRange, std::vector<std::pair<uint64_t, uint64_t>>>& regions_per_range) {
    std::map<CoreCoord, std::vector<std::pair<uint64_t, uint64_t>>> regions_per_core;
    for (const auto& [core_range, intervals] : regions_per_range) {
        for (uint32_t x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (uint32_t y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                merge_cb_stats_intervals(regions_per_core[CoreCoord(x, y)], intervals);
            }
        }
    }
    return regions_per_core;
}


void Metal1CircularBufferAllocator::deallocate() {
    // Deallocate all circular buffers for this program on ALL devices
    // This notifies the GraphTracker to report deallocations
    if (!cb_devices_.empty() && !owner_.builder().circular_buffers().empty()) {
        for (const IDevice* idevice : cb_devices_) {
            tt::tt_metal::GraphTracker::instance().track_deallocate_cb(idevice);
        }

        // Unregister program from ALL devices (matches registration)
        for (const IDevice* idevice : cb_devices_) {
            auto* device = dynamic_cast<Device*>(const_cast<IDevice*>(idevice));
            if (device) {
                device->unregister_program(&owner_);
                // Update locally-allocated CB stats via query (accurate after deallocation)
                if (device->get_shm_stats_provider()) {
                    device->get_shm_stats_provider()->update_from_allocator(device, getpid());
                }
            }
        }

        cb_devices_.clear();  // Clear device set after deallocation
    }
}


void Metal1CircularBufferAllocator::validate_region(const IDevice* device) {
    TTZoneScopedD(PROGRAM);

    // TODO: Circular buffer allocation and validation could be better optimized by determining usage per sub-device
    std::optional<DeviceAddr> lowest_address =
        device->lowest_occupied_compute_l1_address(owner_.determine_sub_device_ids(device));
    uint32_t max_l1_size = device->l1_size_per_core();
    const auto& allocator = device->allocator_impl();
    const bool hybrid_mode = allocator->get_config().allocator_mode == AllocatorMode::HYBRID;

    // In HYBRID mode, per-core allocations live on each per-device allocator (not the mesh
    // allocator). Collect the physical allocators we must consult so we can see per-core state.
    std::vector<AllocatorImpl*> physical_allocators;
    if (hybrid_mode) {
        if (const auto* mesh = dynamic_cast<const tt::tt_metal::distributed::MeshDevice*>(device)) {
            const auto mesh_devices = mesh->impl().get_devices();
            physical_allocators.reserve(mesh_devices.size());
            for (IDevice* dev : mesh_devices) {
                physical_allocators.push_back(dev->allocator_impl().get());
            }
        } else {
            physical_allocators.push_back(allocator.get());
        }
    }

    // Flatten MeshDevice into constituent physical devices so ServiceCoreManager (keyed by ChipId) can be queried per
    // core
    const auto& svc = tt::tt_metal::MetalContext::instance(owner_.get_context_id()).get_service_core_manager().impl();
    std::vector<const IDevice*> devices_for_svc_check;
    if (svc.has_any_claims()) {
        if (const auto* mesh = dynamic_cast<const tt::tt_metal::distributed::MeshDevice*>(device)) {
            const auto mesh_devices = mesh->impl().get_devices();
            devices_for_svc_check.reserve(mesh_devices.size());
            for (IDevice* dev : mesh_devices) {
                devices_for_svc_check.push_back(dev);
            }
        } else {
            devices_for_svc_check.push_back(device);
        }
    }

    for (const auto& cb_allocator : cb_allocation_metadata_.regions) {
        if (cb_allocator.l1_regions.empty()) {
            continue;
        }
        uint64_t cb_region_end = cb_allocator.l1_regions.back().second;  // cb_allocator.get_cb_region_end();
        if (cb_region_end > max_l1_size) {
            TT_THROW(
                "Statically allocated circular buffers on core range {} grow to {} B which is beyond max L1 size of {} "
                "B",
                cb_allocator.core_range.str(),
                cb_region_end,
                max_l1_size);
        }

        // Service cores allocate L1 independently per core (not lock-step like workers), growing down
        // from L1_END. CBs grow up from DEFAULT_UNRESERVED. Min frontier across the CB range catches
        // the most constrained core - collision if any frontier sits below the CB region end
        const bool on_service_core =
            std::any_of(devices_for_svc_check.begin(), devices_for_svc_check.end(), [&](const IDevice* dev) {
                return svc.is_service_core(dev->id(), cb_allocator.core_range.start_coord);
            });

        if (on_service_core) {
            std::optional<DeviceAddr> svc_lowest;
            for (const IDevice* dev : devices_for_svc_check) {
                for (const auto& core : cb_allocator.core_range) {
                    auto a = svc.lowest_allocated_address(dev->id(), core);
                    if (a.has_value()) {
                        svc_lowest = svc_lowest.has_value() ? std::make_optional(std::min(*svc_lowest, *a)) : a;
                    }
                }
            }
            if (svc_lowest.has_value() && svc_lowest.value() < cb_region_end) {
                TT_THROW(
                    "Circular buffers on service-core range {} in program {} clash with ServiceCoreManager-allocated "
                    "L1 (lowest service allocation at {}, CB region ends at {})",
                    cb_allocator.core_range.str(),
                    owner_.id,
                    svc_lowest.value(),
                    cb_region_end);
            }
            continue;  // Worker-grid checks below are irrelevant for service cores.
        }

        if (hybrid_mode) {
            // Per-core allocations (experimental_set_per_core_allocation) can land at different
            // addresses per core, so query only the banks this CB covers on each physical allocator.
            // Prevents per-core tensors on unrelated cores from spuriously tightening this CB's
            // budget, and lets us see per-core state that lives on per-device (not mesh) allocators.
            lowest_address = std::nullopt;
            for (const auto& core : cb_allocator.core_range) {
                for (auto* phys_alloc : physical_allocators) {
                    auto bank_id = phys_alloc->get_bank_ids_from_logical_core(BufferType::L1, core).front();
                    auto addr = phys_alloc->get_lowest_occupied_l1_address(bank_id);
                    if (addr.has_value()) {
                        lowest_address =
                            lowest_address.has_value() ? std::make_optional(std::min(*lowest_address, *addr)) : addr;
                    }
                }
            }
        }
        if (lowest_address.has_value() and lowest_address.value() < cb_region_end) {
            TT_THROW(
                "Statically allocated circular buffers in program {} clash with L1 buffers on core range {}. L1 buffer "
                "allocated at {} and static circular buffer region ends at {}",
                owner_.id,
                cb_allocator.core_range.str(),
                lowest_address.value(),
                cb_region_end);
        }
    }
}


void Metal1CircularBufferAllocator::validate_core_ranges(const IDevice* device) {
    auto grid_size = device->compute_with_storage_grid_size();
    // Flatten MeshDevice into constituent physical devices so ServiceCoreManager (keyed by ChipId) can be queried per
    // core. Mirrors validate_circular_buffer_region.
    const auto& svc = tt::tt_metal::MetalContext::instance(owner_.get_context_id()).get_service_core_manager().impl();
    std::unordered_set<CoreCoord> claimed;
    if (svc.has_any_claims()) {
        if (const auto* mesh = dynamic_cast<const tt::tt_metal::distributed::MeshDevice*>(device)) {
            for (auto device_id : mesh->get_device_ids()) {
                auto chip_claimed = svc.claimed_cores(device_id);
                claimed.insert(chip_claimed.begin(), chip_claimed.end());
            }
        } else {
            claimed = svc.claimed_cores(device->id());
        }
    }
    auto entirely_on_service_cores = [&](const CoreRange& cr) {
        if (claimed.empty()) {
            return false;
        }
        for (uint32_t x = cr.start_coord.x; x <= cr.end_coord.x; ++x) {
            for (uint32_t y = cr.start_coord.y; y <= cr.end_coord.y; ++y) {
                if (!claimed.contains(CoreCoord{x, y})) {
                    return false;
                }
            }
        }
        return true;
    };
    for (const auto& cb : owner_.builder().circular_buffers()) {
        for (const auto& cr : cb->core_ranges().ranges()) {
            const bool in_worker_grid = cr.end_coord.x < grid_size.x && cr.end_coord.y < grid_size.y;
            TT_FATAL(
                in_worker_grid || entirely_on_service_cores(cr),
                "Circular buffer core range {} in program {} exceeds device compute grid ({}x{}) and is "
                "not entirely on cores claimed via ServiceCoreManager",
                cr.str(),
                owner_.id,
                grid_size.x,
                grid_size.y);
        }
    }
}

}  // namespace tt::tt_metal::detail
