// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/dispatch_context.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/distributed_context.hpp>

#include <algorithm>
#include <functional>
#include <optional>

#include "impl/context/context_types.hpp"
#include "mesh_device_impl.hpp"
#include "mesh_command_queue.hpp"
#include "fd_mesh_command_queue.hpp"
#include "sd_mesh_command_queue.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/debug/dprint_server.hpp"
#include "impl/device/device_manager.hpp"
#include "impl/device/device_impl.hpp"
#include "impl/dispatch/cq_shared_state.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "impl/dispatch/dispatch_query_manager.hpp"
#include "llrt/hal/generated/dev_msgs.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/llrt.hpp"
#include "tt_cluster.hpp"

namespace tt::tt_metal::experimental {

struct DispatchContext::StashedQueues {
    std::vector<std::unique_ptr<distributed::MeshCommandQueueBase>> queues;
};

struct DispatchContext::FdL1Conflict {
    ChipId chip;
    CoreCoord core;
    const char* role;
    const char* ledger;
    DeviceAddr lowest;
    DeviceAddr window_end;
};

namespace {

std::optional<DeviceAddr> lowest_arena_address(const AllocatorImpl& allocator, const CoreCoord& core) {
    std::optional<DeviceAddr> lowest;
    for (const auto& range : allocator.persistent_l1().occupied_ranges(core)) {
        lowest = lowest.has_value() ? std::min(*lowest, range.first) : std::make_optional(range.first);
    }
    return lowest;
}

struct LedgerReading {
    std::optional<DeviceAddr> lowest;
    const char* source = "";
};

LedgerReading read_ledger(
    const AllocatorImpl& allocator, const CoreCoord& core, const char* free_list_source, const char* arena_source) {
    LedgerReading reading;
    if (!allocator.has_bank(BufferType::L1, core)) {
        return reading;
    }

    const uint32_t bank = allocator.get_bank_ids_from_logical_core(BufferType::L1, core).at(0);
    if (auto free_list_lowest = allocator.get_lowest_occupied_l1_address(bank); free_list_lowest.has_value()) {
        reading.lowest = free_list_lowest;
        reading.source = free_list_source;
    }

    if (auto arena_lowest = lowest_arena_address(allocator, core);
        arena_lowest.has_value() && (!reading.lowest.has_value() || *arena_lowest < *reading.lowest)) {
        reading.lowest = arena_lowest;
        reading.source = arena_source;
    }
    return reading;
}

}  // namespace

// Define the static member with custom deleter
std::unique_ptr<DispatchContext, DispatchContext::Deleter> DispatchContext::dispatch_context_ptr_ = nullptr;

DispatchContext::~DispatchContext() = default;

void DispatchContext::reset() {
    num_fd_inits_ = 0;
    fast_dispatch_enabled_ = false;
    stashed_sd_queues_.reset();
}

DispatchContext& DispatchContext::get() {
    if (!dispatch_context_ptr_) {
        dispatch_context_ptr_ = std::unique_ptr<DispatchContext, Deleter>(new DispatchContext());
    }
    return *dispatch_context_ptr_;
}

DispatchCoreAxis DispatchContext::get_dispatch_core_axis(distributed::MeshDevice* mesh_device) const {
    return MetalContext::instance(extract_context_id(mesh_device)).get_dispatch_core_config().get_dispatch_core_axis();
}

void DispatchContext::initialize_fast_dispatch(distributed::MeshDevice* mesh_device) {
    initialize_fast_dispatch(mesh_device, FastDispatchSetupOptions{});
}

std::vector<DispatchContext::FdL1Conflict> DispatchContext::find_fd_l1_conflicts(
    MetalContext& context,
    distributed::MeshDevice* mesh_device,
    const std::vector<::tt::tt_metal::Device*>& devices,
    bool write_only) const {
    const DispatchMemMap& mem_map = context.dispatch_mem_map();
    auto& dispatch_core_manager = context.get_dispatch_core_manager();

    // Lockstep allocations live in the allocator of the MeshDevice view that
    // created them. Walk the live tree rooted at the system mesh so this check
    // does not depend on HYBRID mirroring them into each physical allocator.
    std::vector<distributed::MeshDevice*> mesh_views;
    distributed::MeshDevice* root = mesh_device;
    while (root->get_parent_mesh()) {
        root = root->get_parent_mesh().get();
    }
    std::function<void(distributed::MeshDevice*)> collect_views = [&](distributed::MeshDevice* mesh) {
        // A remote-only view (including one not yet fully initialized) has no
        // local devices, SubDeviceManagerTracker, or allocator on this host.
        if (!mesh->get_view().get_devices().empty()) {
            mesh_views.push_back(mesh);
        }
        for (const auto& submesh : mesh->get_submeshes()) {
            collect_views(submesh.get());
        }
    };
    collect_views(root);

    std::vector<FdL1Conflict> conflicts;
    for (::tt::tt_metal::Device* device : devices) {
        const uint16_t channel = context.get_cluster().get_assigned_channel_for_device(device->id());

        std::vector<distributed::MeshDevice*> views_over_device;
        for (distributed::MeshDevice* view : mesh_views) {
            for (IDevice* view_device : view->get_view().get_devices()) {
                if (view_device->id() == device->id()) {
                    views_over_device.push_back(view);
                    break;
                }
            }
        }
        for (distributed::MeshDevice* view : views_over_device) {
            if (view->get_active_sub_device_manager_id() != view->get_default_sub_device_manager_id()) {
                TT_THROW(
                    "Fast-dispatch L1 preflight does not support a live non-default sub-device manager on mesh {}. "
                    "Unload it before entering a manual Fast Dispatch session.",
                    view->id());
            }
        }

        auto check_core = [&](const tt_cxy_pair& core_with_chip, const char* role, DeviceAddr window_end) {
            if (core_with_chip.chip != device->id()) {
                TT_THROW(
                    "Fast-dispatch L1 preflight does not support a {} interface core on chip {} while checking device "
                    "{}. Refusing instead of silently skipping a potentially destructive remote-chip topology.",
                    role,
                    core_with_chip.chip,
                    device->id());
            }

            const CoreCoord core(core_with_chip.x, core_with_chip.y);
            LedgerReading best = read_ledger(*device->allocator_impl(), core, "chip", "chip arena");
            for (distributed::MeshDevice* view : views_over_device) {
                LedgerReading reading = read_ledger(*view->allocator_impl(), core, "mesh", "mesh arena");
                if (reading.lowest.has_value() && (!best.lowest.has_value() || *reading.lowest < *best.lowest)) {
                    best = reading;
                }
            }

            if (best.lowest.has_value() && *best.lowest < window_end) {
                conflicts.push_back({device->id(), core, role, best.source, *best.lowest, window_end});
            }
        };

        for (uint8_t cq_id = 0; cq_id < device->num_hw_cqs(); cq_id++) {
            const DeviceAddr cmddat_end = mem_map.cmddat_q_base(cq_id) + mem_map.cmddat_q_size();
            const DeviceAddr scratch_end = mem_map.scratch_db_base(cq_id) + mem_map.scratch_db_size();
            const DeviceAddr ringbuffer_end = mem_map.scratch_db_base(cq_id) + mem_map.ringbuffer_size();
            const DeviceAddr prefetch_end =
                write_only ? std::max(cmddat_end, scratch_end) : std::max(scratch_end, ringbuffer_end);

            // dispatch_s performs DEVICE_PRINT aggregation only on CQ0 and only
            // when this device has at least one configured print core.
            const auto& dprint_server = context.dprint_server();
            const bool dprint_on = cq_id == 0 && context.get_dispatch_query_manager().dispatch_s_enabled() &&
                                   dprint_server && !dprint_server->get_print_cores(device->id()).empty();
            const DeviceAddr dispatch_end = mem_map.dispatch_s_buffer_end(cq_id) +
                                            (dprint_on ? mem_map.dispatch_s_device_print_l1_cache_size() : 0);

            check_core(dispatch_core_manager.prefetcher_core(device->id(), channel, cq_id), "prefetch", prefetch_end);
            check_core(dispatch_core_manager.dispatcher_core(device->id(), channel, cq_id), "dispatch", dispatch_end);
        }
    }
    return conflicts;
}

std::string DispatchContext::format_fd_l1_conflicts(const std::vector<FdL1Conflict>& conflicts) const {
    std::string report;
    for (const auto& conflict : conflicts) {
        report += fmt::format(
            "  chip {} core ({},{}) [{}]: {} ledger has L1 handed out down to 0x{:X}, "
            "fast-dispatch firmware writes up to 0x{:X}\n",
            conflict.chip,
            conflict.core.x,
            conflict.core.y,
            conflict.role,
            conflict.ledger,
            conflict.lowest,
            conflict.window_end);
    }
    return report;
}

void DispatchContext::unwind_failed_fd_setup(
    MetalContext& context, const std::vector<::tt::tt_metal::Device*>& devices) {
    for (::tt::tt_metal::Device* device : devices) {
        device->command_queue_programs_.clear();
        device->command_queues_.clear();
    }
    context.set_fast_dispatch_mode(false);
}

void DispatchContext::initialize_fast_dispatch(
    distributed::MeshDevice* mesh_device, const FastDispatchSetupOptions& options) {
    // If the mesh device is inactive, do not attempt to initialize fast dispatch.
    if (mesh_device->impl().view_->get_devices().empty()) {
        return;
    }

    auto& context = MetalContext::instance(extract_context_id(mesh_device));
    const auto& cluster = context.get_cluster();

    // Mock/emulated devices skip firmware/dispatch entirely, so there is no real hardware to
    // toggle between Slow and Fast Dispatch. Treat the transition as a no-op to avoid touching
    // dispatch cores/command queues that are never created for these targets (init_command_queue_host
    // leaves command_queues_ empty on mock/emulated), which otherwise segfaults on teardown.
    // See https://github.com/tenstorrent/tt-metal/issues/50634.
    if (cluster.is_mock_or_emulated()) {
        return;
    }

    fast_dispatch_enabled_ = context.rtoptions().get_fast_dispatch();
    TT_FATAL(
        !fast_dispatch_enabled_,
        "Fast Dispatch can only be manually enabled when running the workload with Slow Dispatch mode.");
    TT_FATAL(
        num_fd_inits_ == 0,
        "Fast Dispatch is already manually initialized. Terminate the current manual Fast Dispatch session before "
        "initializing another one.");
    TT_FATAL(
        cluster.is_ubb_galaxy() || cluster.arch() == tt::ARCH::BLACKHOLE,
        "Manually setting up and tearing down Fast Dispatch is only supported on Galaxy and Blackhole clusters.");

    const auto& device_manager = context.device_manager();
    const auto& active_devices = device_manager->get_all_active_devices_impl();

    uint8_t num_hw_cqs = active_devices[0]->num_hw_cqs();

    // Enable Fast Dispatch and reinitialize dispatch managers to pick up FD core descriptor before allocating cores
    context.set_fast_dispatch_mode(true);

    try {
        for (const auto& dev : active_devices) {
            TT_FATAL(dev->num_hw_cqs() == num_hw_cqs, "All devices must have the same number of command queues.");
            dev->init_command_queue_host();
        }

        // Dispatch cores are assigned, but no fast-dispatch firmware has been
        // written yet. Refuse before bring-up can overwrite resident L1.
        const auto conflicts = find_fd_l1_conflicts(context, mesh_device, active_devices, options.write_only);
        if (!conflicts.empty()) {
            const std::string report = format_fd_l1_conflicts(conflicts);
            if (!options.allow_destructive) {
                TT_THROW(
                    "Fast-dispatch bring-up would overwrite resident L1 on dispatch cores (tt-blaze #2019):\n{}"
                    "Free or relocate those allocations before entering fast dispatch, or pass "
                    "allow_destructive=true to proceed and accept that the listed L1 will be corrupted.",
                    report);
            }
            log_warning(
                tt::LogAlways,
                "allow_destructive=true: fast-dispatch bring-up will overwrite resident L1 on dispatch cores:\n{}",
                report);
        }
    } catch (...) {
        unwind_failed_fd_setup(context, active_devices);
        throw;
    }

    // Query the number of command queues requested
    device_manager->initialize_dispatch_firmware(/*force_recreate_topology=*/true);

    auto& mesh_device_impl = mesh_device->impl();

    // Drain pending SD work and stash the SD queues for restoration on terminate
    for (auto& cq : mesh_device_impl.mesh_command_queues_) {
        cq->finish();
    }
    for (const auto& dev : active_devices) {
        dev->set_smc_dispatch_telemetry_slow_dispatch_enabled(false);
    }
    stashed_sd_queues_ = std::make_unique<StashedQueues>();
    for (auto& cq : mesh_device_impl.mesh_command_queues_) {
        stashed_sd_queues_->queues.push_back(std::move(cq));
    }
    mesh_device_impl.mesh_command_queues_.clear();
    mesh_device_impl.mesh_command_queues_.reserve(num_hw_cqs);

    auto cq_shared_state = std::make_shared<CQSharedState>();
    cq_shared_state->sub_device_cq_owner.resize(1);

    for (std::size_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
        mesh_device_impl.mesh_command_queues_.push_back(std::make_unique<distributed::FDMeshCommandQueue>(
            mesh_device,
            cq_id,
            mesh_device_impl.dispatch_thread_pool_,
            mesh_device_impl.reader_thread_pool_,
            cq_shared_state,
            std::bind(&distributed::MeshDeviceImpl::lock_api, &mesh_device_impl),
            mesh_device_impl.active_distributed_context_));
    }
    fast_dispatch_enabled_ = true;
    num_fd_inits_++;
}

void DispatchContext::terminate_fast_dispatch(distributed::MeshDevice* mesh_device) {
    // If the mesh device is inactive, do not attempt to terminate fast dispatch.
    if (mesh_device->impl().view_->get_devices().empty()) {
        return;
    }

    auto& context = MetalContext::instance(extract_context_id(mesh_device));
    const auto& cluster = context.get_cluster();

    // Mirror initialize_fast_dispatch: the FD/SD toggle is a no-op on mock/emulated targets, so
    // there is nothing to tear down. See https://github.com/tenstorrent/tt-metal/issues/50634.
    if (cluster.is_mock_or_emulated()) {
        return;
    }

    TT_FATAL(fast_dispatch_enabled_, "Can only manually terminate fast dispatch after initializing it.");
    TT_FATAL(num_fd_inits_ == 1, "Fast Dispatch termination requires exactly one active manual Fast Dispatch session.");

    const auto& device_manager = context.device_manager();
    const auto& active_devices = device_manager->get_all_active_devices_impl();

    auto& mesh_device_impl = mesh_device->impl();
    mesh_device_impl.mesh_command_queues_.clear();

    // Restore stashed SD queues to preserve pre-FD state (e.g. asynchronous_slow_dispatch_enabled_)
    TT_FATAL(
        stashed_sd_queues_ && !stashed_sd_queues_->queues.empty(),
        "No stashed SD queues to restore; was initialize_fast_dispatch called?");
    for (auto& cq : stashed_sd_queues_->queues) {
        mesh_device_impl.mesh_command_queues_.push_back(std::move(cq));
    }
    stashed_sd_queues_.reset();
    for (const auto& dev : active_devices) {
        dev->set_smc_dispatch_telemetry_slow_dispatch_enabled(true);
    }

    for (const auto& dev : active_devices) {
        for (int cq_id = 0; cq_id < dev->num_hw_cqs(); cq_id++) {
            dev->command_queues_[cq_id].get()->terminate();
        }
    }

    for (const auto& dev : active_devices) {
        auto dispatch_cores = device_manager->get_virtual_dispatch_cores(dev->id());
        tt::llrt::internal_::wait_until_cores_done(context, dev->id(), dev_msgs::RUN_MSG_GO, dispatch_cores, 0);
    }

    // HWCommandQueue holds a reference to sysmem_manager_. Clear now so any future
    // init_command_queue_host call can safely replace sysmem_manager_ without dangling references
    for (const auto& device : active_devices) {
        device->command_queue_programs_.clear();
        device->command_queues_.clear();
    }

    fast_dispatch_enabled_ = false;

    // Disable Fast Dispatch and reinitialize dispatch managers to pick up SD core descriptor
    context.set_fast_dispatch_mode(false);
    num_fd_inits_--;
}

void DispatchContext::enable_asynchronous_slow_dispatch(distributed::MeshDevice* mesh_device) {
    TT_FATAL(
        !MetalContext::instance().rtoptions().get_fast_dispatch(),
        "{} can only be called when Fast Dispatch is disabled.",
        __func__);
    auto& sd_mesh_cq = dynamic_cast<distributed::SDMeshCommandQueue&>(mesh_device->mesh_command_queue());
    sd_mesh_cq.enable_asynchronous_slow_dispatch();
}

void DispatchContext::disable_asynchronous_slow_dispatch(distributed::MeshDevice* mesh_device) {
    TT_FATAL(
        !MetalContext::instance().rtoptions().get_fast_dispatch(),
        "{} can only be called when Fast Dispatch is disabled.",
        __func__);
    auto& sd_mesh_cq = dynamic_cast<distributed::SDMeshCommandQueue&>(mesh_device->mesh_command_queue());
    sd_mesh_cq.disable_asynchronous_slow_dispatch();
}

bool DispatchContext::is_asynchronous_slow_dispatch_enabled(distributed::MeshDevice* mesh_device) const {
    auto& sd_mesh_cq = dynamic_cast<distributed::SDMeshCommandQueue&>(mesh_device->mesh_command_queue());
    return sd_mesh_cq.is_asynchronous_slow_dispatch_enabled();
}

}  // namespace tt::tt_metal::experimental
