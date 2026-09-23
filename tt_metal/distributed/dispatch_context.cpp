// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/dispatch_context.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/distributed_context.hpp>

#include <algorithm>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "impl/context/context_types.hpp"
#include "mesh_device_impl.hpp"
#include "mesh_command_queue.hpp"
#include "fd_mesh_command_queue.hpp"
#include "sd_mesh_command_queue.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/device/device_manager.hpp"
#include "impl/device/device_impl.hpp"
#include "impl/dispatch/cq_shared_state.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "llrt/hal/generated/dev_msgs.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/llrt.hpp"
#include "tt_cluster.hpp"

namespace tt::tt_metal::experimental {

struct DispatchContext::StashedQueues {
    std::vector<std::unique_ptr<distributed::MeshCommandQueueBase>> queues;
};

namespace {

// One resident L1 allocation whose data sits on a core that fast dispatch is about to claim.
struct FdL1Conflict {
    ChipId chip;
    CoreCoord core;
    const char* ledger;  // "chip" / "mesh" / "chip arena" / "mesh arena": which bookkeeping recorded it
    const char* kind;    // "per-core" / "lockstep-sharded" / "lockstep-nd-sharded" / "interleaved" / "arena"
    DeviceAddr address;  // the allocation's address on this core; reported, never used to decide
    DeviceAddr bytes_per_core;
    uint32_t num_cores;  // how many cores the allocation spans (0 = every L1 bank, i.e. interleaved)
};

// Does this buffer hold data in L1 on `core`? Addresses are irrelevant here. A lockstep buffer
// reserves its address range on every bank, but its bytes live only on its shard grid, so the
// shared free list cannot answer this question and the buffer object has to.
bool l1_buffer_touches_core(const Buffer& buffer, const CoreCoord& core) {
    if (buffer.buffer_type() != BufferType::L1) {
        return false;  // DRAM / TRACE are not L1; L1_SMALL is its own region at the top of the bank.
    }
    if (buffer.has_shard_spec()) {
        return buffer.shard_spec().grid().contains(core);
    }
    if (const auto& distribution = buffer.buffer_distribution_spec(); distribution.has_value()) {
        // cores() is every core the spec was configured with; only cores_with_data() received a shard.
        const auto& cores = distribution->cores_with_data();
        return std::find(cores.begin(), cores.end(), core) != cores.end();
    }
    // Interleaved: pages are spread round-robin over every L1 bank, so the buffer is attributed to every
    // core. Conservative for a buffer with fewer pages than banks.
    return true;
}

const char* l1_buffer_kind(const Buffer& buffer) {
    if (per_core_allocation::is_per_core_allocation(buffer)) {
        return "per-core";
    }
    if (buffer.has_shard_spec()) {
        return "lockstep-sharded";
    }
    if (buffer.buffer_distribution_spec().has_value()) {
        return "lockstep-nd-sharded";
    }
    return "interleaved";
}

// Append one conflict per L1 buffer, and per persistent-arena region, that `allocator` has placed on
// `core`, at any address. Buffers are attributed to cores by their shard grid, not by the free list, so
// a lockstep tensor sharded elsewhere does not count even though it reserves the same address range on
// this bank (the "lockstep false positive").
void collect_conflicts(
    const AllocatorImpl& allocator,
    const char* ledger,
    const char* arena_ledger,
    ChipId chip,
    const CoreCoord& core,
    std::vector<FdL1Conflict>& conflicts) {
    if (!allocator.has_bank(BufferType::L1, core)) {
        return;
    }

    // Fast path. Every allocated buffer's address is recorded in a free list (lockstep, or this bank's
    // per-core list under HYBRID). An empty free list for this bank means no buffer can be on this core,
    // and the walk below is skipped.
    const uint32_t bank = allocator.get_bank_ids_from_logical_core(BufferType::L1, core).at(0);
    if (allocator.get_lowest_occupied_l1_address(bank).has_value()) {
        // get_allocated_buffers() copies the set under the allocator mutex; the Buffer pointers are
        // dereferenced without it. A manual fast-dispatch session is entered from one host thread with
        // no concurrent allocation, which is the contract this preflight relies on.
        for (Buffer* buffer : allocator.get_allocated_buffers()) {
            if (buffer == nullptr || !l1_buffer_touches_core(*buffer, core)) {
                continue;
            }
            // get_per_core_address TT_FATALs for a core the buffer does not span; the grid test above
            // guarantees it does.
            const DeviceAddr address = per_core_allocation::is_per_core_allocation(*buffer)
                                           ? per_core_allocation::get_per_core_address(*buffer, core)
                                           : static_cast<DeviceAddr>(buffer->address());
            conflicts.push_back(
                {chip,
                 core,
                 ledger,
                 l1_buffer_kind(*buffer),
                 address,
                 buffer->aligned_size_per_bank(),
                 buffer->num_cores().value_or(0)});
        }
    }

    // Arena regions are not Buffer objects and are tracked per core, so they are read directly.
    for (const auto& range : allocator.persistent_l1().occupied_ranges(core)) {
        conflicts.push_back({chip, core, arena_ledger, "arena", range.first, range.second - range.first, 1});
    }
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

namespace {

// Preflight helpers. Free functions rather than members: they read no DispatchContext state, and
// keeping them file-local keeps FdL1Conflict out of the public header.
std::vector<FdL1Conflict> find_fd_l1_conflicts(
    MetalContext& metal_context,
    distributed::MeshDevice* mesh_device,
    const std::vector<::tt::tt_metal::Device*>& devices) {
    auto& dispatch_core_manager = metal_context.get_dispatch_core_manager();

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
        if (mesh->is_initialized() && !mesh->get_view().get_devices().empty()) {
            mesh_views.push_back(mesh);
        }
        for (const auto& submesh : mesh->get_submeshes()) {
            collect_views(submesh.get());
        }
    };
    collect_views(root);

    std::vector<FdL1Conflict> conflicts;
    // Walk through the IDevice interface: everything this preflight needs (id, allocator_impl) is
    // public there, while Device keeps allocator_impl private.
    for (IDevice* device : devices) {
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

        // The cores fast dispatch has claimed on this chip. They were assigned during
        // init_command_queue_host(), before any firmware is written, and the dispatch core manager is
        // asked for the whole set rather than for named roles, so every role, queue and split-dispatch
        // placement the configuration uses is covered.
        for (const CoreCoord& core : dispatch_core_manager.get_assigned_dispatch_cores(device->id())) {
            // Per-core buffers are recorded in the chip's allocator; lockstep buffers in the allocator
            // of the mesh view that created them (the HYBRID mirror marks ranges but registers no
            // Buffer). Each ledger is walked for buffers whose data is on this core.
            collect_conflicts(*device->allocator_impl(), "chip", "chip arena", device->id(), core, conflicts);
            for (distributed::MeshDevice* view : views_over_device) {
                collect_conflicts(*view->allocator_impl(), "mesh", "mesh arena", device->id(), core, conflicts);
            }
        }
    }
    return conflicts;
}

std::string format_fd_l1_conflicts(const std::vector<FdL1Conflict>& conflicts) {
    std::string report;
    for (const auto& conflict : conflicts) {
        const std::string span =
            conflict.num_cores == 0 ? std::string("every L1 bank") : fmt::format("{} core(s)", conflict.num_cores);
        report += fmt::format(
            "  chip {} dispatch core ({},{}): {} ledger: {} allocation at 0x{:X}, {} B/core, spans {}\n",
            conflict.chip,
            conflict.core.x,
            conflict.core.y,
            conflict.ledger,
            conflict.kind,
            conflict.address,
            conflict.bytes_per_core,
            span);
    }
    return report;
}

}  // namespace

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

    auto& mesh_device_impl = mesh_device->impl();
    auto& metal_context = mesh_device_impl.metal_context();
    auto& metal_env = mesh_device_impl.metal_env();
    const auto& cluster = metal_env.get_cluster();

    // Mock/emulated devices skip firmware/dispatch entirely, so there is no real hardware to
    // toggle between Slow and Fast Dispatch. Treat the transition as a no-op to avoid touching
    // dispatch cores/command queues that are never created for these targets (init_command_queue_host
    // leaves command_queues_ empty on mock/emulated), which otherwise segfaults on teardown.
    // See https://github.com/tenstorrent/tt-metal/issues/50634.
    if (cluster.is_mock_or_emulated()) {
        return;
    }

    fast_dispatch_enabled_ = metal_env.get_rtoptions().get_fast_dispatch();
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

    const auto& device_manager = metal_context.device_manager();
    const auto& active_devices = device_manager->get_all_active_devices_impl();

    uint8_t num_hw_cqs = active_devices[0]->num_hw_cqs();

    // Enable Fast Dispatch and reinitialize dispatch managers to pick up FD core descriptor before allocating cores
    metal_context.set_fast_dispatch_mode(true);

    try {
        for (const auto& dev : active_devices) {
            TT_FATAL(dev->num_hw_cqs() == num_hw_cqs, "All devices must have the same number of command queues.");
            dev->init_command_queue_host();
        }

        // Dispatch cores are assigned, but no fast-dispatch firmware has been written yet. Dispatch
        // cores may not hold L1 allocations while fast dispatch is active, at any address, so refuse
        // now if any do.
        const auto conflicts = find_fd_l1_conflicts(metal_context, mesh_device, active_devices);
        if (!conflicts.empty()) {
            const std::string report = format_fd_l1_conflicts(conflicts);
            if (!options.allow_destructive) {
                TT_THROW(
                    "Fast dispatch would claim dispatch cores that hold resident L1 (tt-blaze #2019):\n{}"
                    "L1 allocations may not live on a dispatch core while a fast-dispatch session is active. "
                    "Free or relocate them before entering fast dispatch, or pass allow_destructive=true to "
                    "proceed and accept that the listed L1 may be corrupted.",
                    report);
            }
            log_warning(
                tt::LogAlways,
                "allow_destructive=true: fast-dispatch bring-up may overwrite resident L1 on dispatch cores:\n{}",
                report);
        }
    } catch (...) {
        unwind_failed_fd_setup(metal_context, active_devices);
        throw;
    }

    // Query the number of command queues requested
    device_manager->initialize_dispatch_firmware(/*force_recreate_topology=*/true);

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

    auto& mesh_device_impl = mesh_device->impl();
    auto& metal_context = mesh_device_impl.metal_context();
    const auto& cluster = mesh_device_impl.metal_env().get_cluster();

    // Mirror initialize_fast_dispatch: the FD/SD toggle is a no-op on mock/emulated targets, so
    // there is nothing to tear down. See https://github.com/tenstorrent/tt-metal/issues/50634.
    if (cluster.is_mock_or_emulated()) {
        return;
    }

    TT_FATAL(fast_dispatch_enabled_, "Can only manually terminate fast dispatch after initializing it.");
    TT_FATAL(num_fd_inits_ == 1, "Fast Dispatch termination requires exactly one active manual Fast Dispatch session.");

    const auto& device_manager = metal_context.device_manager();
    const auto& active_devices = device_manager->get_all_active_devices_impl();

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
        tt::llrt::internal_::wait_until_cores_done(metal_context, dev->id(), dev_msgs::RUN_MSG_GO, dispatch_cores, 0);
    }

    // HWCommandQueue holds a reference to sysmem_manager_. Clear now so any future
    // init_command_queue_host call can safely replace sysmem_manager_ without dangling references
    for (const auto& device : active_devices) {
        device->command_queue_programs_.clear();
        device->command_queues_.clear();
    }

    fast_dispatch_enabled_ = false;

    // Disable Fast Dispatch and reinitialize dispatch managers to pick up SD core descriptor
    metal_context.set_fast_dispatch_mode(false);
    num_fd_inits_--;
}

void DispatchContext::set_configure_only(distributed::MeshDevice* mesh_device, bool enable) {
    TT_FATAL(
        !mesh_device->impl().metal_env().get_rtoptions().get_fast_dispatch(),
        "{} can only be called when Fast Dispatch is disabled.",
        __func__);
    auto& sd_mesh_cq = dynamic_cast<distributed::SDMeshCommandQueue&>(mesh_device->mesh_command_queue());
    sd_mesh_cq.set_configure_only(enable);
}

void DispatchContext::enable_asynchronous_slow_dispatch(distributed::MeshDevice* mesh_device) {
    TT_FATAL(
        !mesh_device->impl().metal_env().get_rtoptions().get_fast_dispatch(),
        "{} can only be called when Fast Dispatch is disabled.",
        __func__);
    auto& sd_mesh_cq = dynamic_cast<distributed::SDMeshCommandQueue&>(mesh_device->mesh_command_queue());
    sd_mesh_cq.enable_asynchronous_slow_dispatch();
}

void DispatchContext::disable_asynchronous_slow_dispatch(distributed::MeshDevice* mesh_device) {
    TT_FATAL(
        !mesh_device->impl().metal_env().get_rtoptions().get_fast_dispatch(),
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
