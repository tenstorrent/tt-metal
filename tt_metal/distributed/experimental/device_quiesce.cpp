// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/device_quiesce.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-umd/driver_atomics.hpp>

#include "api/internal/service/service_core_manager.hpp"
#include "context/metal_context.hpp"
#include "device/device_impl.hpp"
#include "device/device_manager.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "fabric/fabric_host_utils.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "impl/dispatch/dispatch_engine_cores.hpp"
#include "impl/dispatch/dispatch_query_manager.hpp"
#include "llrt/hal.hpp"
#include "llrt/hal/generated/dev_msgs.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::experimental {
namespace {

struct QuiesceCoreTarget {
    ChipId chip_id = 0;
    CoreCoord virtual_core{0, 0};
    HalProgrammableCoreType prog_type = HalProgrammableCoreType::TENSIX;
};

dev_msgs::quiesce_msg_t read_quiesce_slot(MetalContext& ctx, const QuiesceCoreTarget& target) {
    const uint64_t addr = ctx.hal().get_dev_addr(target.prog_type, HalL1MemAddrType::QUIESCE);
    auto msg = ctx.hal().get_dev_msgs_factory(target.prog_type).create<dev_msgs::quiesce_msg_t>();
    ctx.get_cluster().read_core(msg.data(), msg.size(), tt_cxy_pair(target.chip_id, target.virtual_core), addr);
    return msg;
}

void write_quiesce_target_op_count(
    MetalContext& ctx, const QuiesceCoreTarget& target, uint32_t slot, uint32_t target_op_count) {
    TT_ASSERT(slot < dev_msgs::QUIESCE_MAX_CQS);
    const auto factory = ctx.hal().get_dev_msgs_factory(target.prog_type);
    const uint64_t base_addr = ctx.hal().get_dev_addr(target.prog_type, HalL1MemAddrType::QUIESCE);
    const uint64_t target_addr =
        base_addr + factory.offset_of<dev_msgs::quiesce_msg_t>(dev_msgs::quiesce_msg_t::Field::target_op_count) +
        (slot * sizeof(uint32_t));
    const tt_cxy_pair core_cxy(target.chip_id, target.virtual_core);
    ctx.get_cluster().write_core(&target_op_count, sizeof(target_op_count), core_cxy, target_addr);
    tt_driver_atomics::sfence();
    // Non-posted MMIO readback from the same core drains posted NOC2AXI writes into L1 without
    // touching UMD's MemBarFlag (which throws on CoreType::DISPATCH and mutates Tensix L1).
    uint32_t readback = 0;
    ctx.get_cluster().read_core(&readback, sizeof(readback), core_cxy, target_addr);
}

void write_quiesce_control(MetalContext& ctx, const QuiesceCoreTarget& target, uint32_t control_val) {
    const auto factory = ctx.hal().get_dev_msgs_factory(target.prog_type);
    const uint64_t base_addr = ctx.hal().get_dev_addr(target.prog_type, HalL1MemAddrType::QUIESCE);
    const uint64_t ctrl_addr =
        base_addr + factory.offset_of<dev_msgs::quiesce_msg_t>(dev_msgs::quiesce_msg_t::Field::control);
    ctx.get_cluster().write_core(
        &control_val, sizeof(control_val), tt_cxy_pair(target.chip_id, target.virtual_core), ctrl_addr);
    tt_driver_atomics::sfence();
}

bool poll_until_all_quiesced(
    MetalContext& ctx,
    const std::vector<QuiesceCoreTarget>& targets,
    std::optional<std::chrono::steady_clock::time_point> deadline,
    std::vector<QuiesceCoreDiagnostic>& out_unquiesced) {
    out_unquiesced.clear();
    if (targets.empty()) {
        return true;
    }

    std::vector<uint8_t> done(targets.size(), 0);
    size_t remaining = targets.size();

    std::vector<ChipId> distinct_chips;
    distinct_chips.reserve(targets.size());
    for (const auto& t : targets) {
        if (std::find(distinct_chips.begin(), distinct_chips.end(), t.chip_id) == distinct_chips.end()) {
            distinct_chips.push_back(t.chip_id);
        }
    }

    while (remaining > 0) {
        for (ChipId chip_id : distinct_chips) {
            ctx.get_cluster().advance_device_execution(chip_id);
        }

        for (size_t i = 0; i < targets.size(); ++i) {
            if (done[i]) {
                continue;
            }
            auto msg = read_quiesce_slot(ctx, targets[i]);
            if (msg.view().status() == dev_msgs::QUIESCE_STATUS_QUIESCED) {
                done[i] = 1;
                --remaining;
            }
        }

        if (remaining == 0) {
            return true;
        }

        if (deadline.has_value() && std::chrono::steady_clock::now() >= *deadline) {
            for (ChipId chip_id : distinct_chips) {
                ctx.get_cluster().advance_device_execution(chip_id);
            }
            for (size_t i = 0; i < targets.size(); ++i) {
                if (done[i]) {
                    continue;
                }
                auto msg = read_quiesce_slot(ctx, targets[i]);
                if (msg.view().status() == dev_msgs::QUIESCE_STATUS_QUIESCED) {
                    done[i] = 1;
                    --remaining;
                } else {
                    out_unquiesced.push_back(QuiesceCoreDiagnostic{
                        .chip_id = targets[i].chip_id,
                        .virtual_core = targets[i].virtual_core,
                        .core_type = targets[i].prog_type,
                        .raw_status = msg.view().status(),
                        .control = msg.view().control(),
                        .op_started_count_cq0 = msg.view().op_started_count()[0],
                        .op_started_count_cq1 = msg.view().op_started_count()[1],
                    });
                }
            }
            return out_unquiesced.empty();
        }
    }
    return true;
}

std::chrono::milliseconds elapsed_since(std::chrono::steady_clock::time_point start_time) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
}

}  // namespace

QuiesceResult quiesce_for_soft_reset(distributed::MeshDevice& mesh_device, const QuiesceOptions& options) {
    const auto start_time = std::chrono::steady_clock::now();

    TT_FATAL(mesh_device.is_initialized(), "quiesce_for_soft_reset() requires an initialized MeshDevice.");
    const auto& parent_mesh = mesh_device.get_parent_mesh();
    const bool owns_all_open_devices =
        mesh_device.is_parent_mesh() ||
        (!parent_mesh->is_initialized() && parent_mesh->num_devices() == mesh_device.num_devices());
    TT_FATAL(
        owns_all_open_devices,
        "quiesce_for_soft_reset() must be called on the root MeshDevice owning all open devices in its context "
        "(e.g., from open_mesh_device / open_unit_mesh, not a carved submesh or partial unit-mesh group).");
    TT_FATAL(
        mesh_device.arch() == tt::ARCH::QUASAR,
        "quiesce_for_soft_reset() is only supported on Quasar AI-IP targets (got arch {}).",
        enchantum::to_string(mesh_device.arch()));

    auto& mesh_impl = mesh_device.impl();
    auto& ctx = mesh_impl.metal_context();
    auto& cluster = ctx.get_cluster();
    const auto target_type = cluster.get_target_device_type();
    TT_FATAL(
        target_type != tt::TargetDevice::Mock && target_type != tt::TargetDevice::Emule,
        "quiesce_for_soft_reset() requires Silicon or Simulator target (Mock and Emule do not run RISC-V firmware "
        "quiesce loops).");
    TT_FATAL(
        !options.timeout.has_value() || options.timeout->count() > 0,
        "quiesce_for_soft_reset() requires a positive timeout when specified (got {} ms).",
        options.timeout->count());
    TT_FATAL(
        !mesh_impl.has_active_tensor_prefetcher(),
        "quiesce_for_soft_reset() cannot run while an experimental DRISC TensorPrefetcher is active; call "
        "StopTensorPrefetcher() first.");

    const auto local_devices = mesh_impl.get_devices();
    for (auto* dev : local_devices) {
        TT_FATAL(
            ctx.get_service_core_manager().claimed_cores(dev->id()).empty(),
            "quiesce_for_soft_reset() cannot run while persistent ServiceCoreManager cores are claimed on device {}; "
            "stop and release service cores first.",
            dev->id());
    }

    const std::optional<std::chrono::steady_clock::time_point> deadline =
        options.timeout.has_value()
            ? std::optional<std::chrono::steady_clock::time_point>(start_time + *options.timeout)
            : std::nullopt;

    // Drain any in-flight host dispatch tasks and prevent new submissions during quiesce.
    mesh_device.wait_for_thread_pool();
    auto api_lock = mesh_impl.acquire_api_unique_lock();
    mesh_device.wait_for_thread_pool();

    // Stop FDMeshCommandQueue completion-queue reader threads before Phase 1 so host never issues
    // late send_completion_queue_read_ptr() MMIO writes after dispatch parks. Safe against full CQ
    // deadlock because cq_dispatch sinks D2H writes once quiesce_msg_t.control is non-zero.
    mesh_impl.stop_command_queue_readers_for_quiesce();

    std::unordered_set<tt_cxy_pair> parked_cores;

    // -------------------------------------------------------------------------
    // Phase 1: Drain & park Fast Dispatch cores at a clean op boundary
    // -------------------------------------------------------------------------
    if (ctx.device_manager()->is_dispatch_firmware_active()) {
        std::vector<QuiesceCoreTarget> phase1_targets;
        for (auto* dev : local_devices) {
            if (!dev->is_mmio_capable()) {
                continue;
            }
            for (const auto& virtual_core : ctx.device_manager()->get_virtual_dispatch_cores(dev->id())) {
                const tt_cxy_pair cxy(dev->id(), virtual_core);
                if (parked_cores.insert(cxy).second) {
                    phase1_targets.push_back(QuiesceCoreTarget{
                        .chip_id = dev->id(),
                        .virtual_core = virtual_core,
                        .prog_type = dev->get_programmable_core_type(virtual_core),
                    });
                }
            }
        }

        const bool use_fabric_at_op_count =
            (local_devices.size() > 1) && tt::tt_fabric::is_tt_fabric_config(ctx.get_fabric_config());

        if (use_fabric_at_op_count) {
            // Synchronize multi-chip cutoff at a common remaining-op offset so cross-chip Fabric collectives
            // never straddle a boundary where one chip executes an op and its peer skips it.
            constexpr uint32_t kFabricOpSafetyMargin = 2U;
            const uint8_t num_cqs = mesh_device.num_hw_cqs();
            const uint32_t num_cqs_per_core = ctx.get_dispatch_query_manager().cq_dispatch_layout().num_cqs_per_core;
            const CoreType dispatch_core_type = ctx.get_dispatch_core_manager().get_dispatch_core_type();

            std::unordered_set<tt_cxy_pair> prefetcher_cores;
            for (uint8_t cq_id = 0; cq_id < num_cqs; ++cq_id) {
                const uint32_t slot = (num_cqs_per_core > 1) ? cq_id : 0U;
                struct PerChipCounts {
                    QuiesceCoreTarget target;
                    uint32_t enqueued = 0;
                    uint32_t remaining = 0;
                };
                std::vector<PerChipCounts> chip_counts;
                chip_counts.reserve(local_devices.size());
                uint32_t min_remaining = std::numeric_limits<uint32_t>::max();

                for (auto* dev : local_devices) {
                    const uint16_t channel = cluster.get_assigned_channel_for_device(dev->id());
                    const tt_cxy_pair logical_prefetch =
                        ctx.get_dispatch_core_manager().prefetcher_core(dev->id(), channel, cq_id);
                    const ChipId servicing_chip_id = static_cast<ChipId>(logical_prefetch.chip);
                    const CoreCoord virtual_prefetch = cluster.get_virtual_coordinate_from_logical_coordinates(
                        servicing_chip_id, CoreCoord(logical_prefetch.x, logical_prefetch.y), dispatch_core_type);
                    auto* servicing_dev = ctx.device_manager()->get_active_device(servicing_chip_id);
                    QuiesceCoreTarget target{
                        .chip_id = servicing_chip_id,
                        .virtual_core = virtual_prefetch,
                        .prog_type = servicing_dev->get_programmable_core_type(virtual_prefetch),
                    };
                    prefetcher_cores.insert(tt_cxy_pair(target.chip_id, target.virtual_core));

                    auto msg = read_quiesce_slot(ctx, target);
                    const uint32_t started = msg.view().op_started_count()[slot];
                    const uint32_t enqueued = mesh_impl.enqueued_program_seq_count_for_device(cq_id, dev->id());
                    const uint32_t rem = (enqueued >= started) ? (enqueued - started) : 0U;
                    min_remaining = std::min(min_remaining, rem);
                    chip_counts.push_back(PerChipCounts{.target = target, .enqueued = enqueued, .remaining = rem});
                }

                const uint32_t target_remaining =
                    (min_remaining > kFabricOpSafetyMargin) ? (min_remaining - kFabricOpSafetyMargin) : 0U;
                for (const auto& entry : chip_counts) {
                    const uint32_t target_op = entry.enqueued - target_remaining;
                    write_quiesce_target_op_count(ctx, entry.target, slot, target_op);
                }
            }

            for (const auto& target : phase1_targets) {
                const bool is_prefetcher = prefetcher_cores.contains(tt_cxy_pair(target.chip_id, target.virtual_core));
                write_quiesce_control(
                    ctx, target, is_prefetcher ? dev_msgs::QUIESCE_CTRL_AT_OP_COUNT : dev_msgs::QUIESCE_CTRL_NEXT_OP);
            }
        } else {
            for (const auto& target : phase1_targets) {
                write_quiesce_control(ctx, target, dev_msgs::QUIESCE_CTRL_NEXT_OP);
            }
        }

        std::vector<QuiesceCoreDiagnostic> unquiesced;
        if (!poll_until_all_quiesced(ctx, phase1_targets, deadline, unquiesced)) {
            ctx.set_quiesce_context_state(MetalContext::QuiesceContextState::QuiesceTimedOut);
            ctx.detach_debug_servers_for_quiesce();
            log_warning(
                tt::LogMetal,
                "quiesce_for_soft_reset() timed out in Phase 1 (dispatch drain) with {} unquiesced core(s).",
                unquiesced.size());
            return QuiesceResult{
                .status = QuiesceStatus::TimedOutInDispatch,
                .elapsed = elapsed_since(start_time),
                .unquiesced_cores = std::move(unquiesced),
            };
        }
    }

    // -------------------------------------------------------------------------
    // Phase 1.5: Shut down RT/streaming profilers and terminate TT-Fabric routers
    // -------------------------------------------------------------------------
    mesh_impl.shutdown_profilers_for_quiesce();
    if (!ctx.device_manager()->quiesce_fabric(deadline)) {
        ctx.set_quiesce_context_state(MetalContext::QuiesceContextState::QuiesceTimedOut);
        ctx.detach_debug_servers_for_quiesce();
        log_warning(tt::LogMetal, "quiesce_for_soft_reset() timed out in Phase 1.5 (TT-Fabric router termination).");
        return QuiesceResult{
            .status = QuiesceStatus::TimedOutInFabric,
            .elapsed = elapsed_since(start_time),
            .unquiesced_cores = {},
        };
    }

    // -------------------------------------------------------------------------
    // Phase 2: Park all TENSIX workers + any remaining booted DISPATCH tiles
    // -------------------------------------------------------------------------
    std::vector<QuiesceCoreTarget> phase2_targets;
    for (auto* dev : local_devices) {
        const auto& soc_desc = cluster.get_soc_desc(dev->id());
        const CoreCoord worker_grid = soc_desc.get_grid_size(CoreType::TENSIX);
        for (uint32_t y = 0; y < worker_grid.y; ++y) {
            for (uint32_t x = 0; x < worker_grid.x; ++x) {
                const CoreCoord virtual_worker = cluster.get_virtual_coordinate_from_logical_coordinates(
                    dev->id(), CoreCoord(x, y), CoreType::WORKER);
                if (parked_cores.insert(tt_cxy_pair(dev->id(), virtual_worker)).second) {
                    phase2_targets.push_back(QuiesceCoreTarget{
                        .chip_id = dev->id(),
                        .virtual_core = virtual_worker,
                        .prog_type = HalProgrammableCoreType::TENSIX,
                    });
                }
            }
        }

        if (ctx.hal().has_programmable_core_type(HalProgrammableCoreType::DISPATCH) &&
            !ctx.rtoptions().get_use_quasar_tensix_dispatch_cores() && soc_desc.get_num_dispatch_engine_cores() > 0) {
            for (const CoreCoord& logical_dispatch_core :
                 detail::get_quasar_soc_dispatch_engine_logical_cores(soc_desc)) {
                const CoreCoord virtual_dispatch = cluster.get_virtual_coordinate_from_logical_coordinates(
                    dev->id(), logical_dispatch_core, CoreType::DISPATCH);
                if (parked_cores.insert(tt_cxy_pair(dev->id(), virtual_dispatch)).second) {
                    phase2_targets.push_back(QuiesceCoreTarget{
                        .chip_id = dev->id(),
                        .virtual_core = virtual_dispatch,
                        .prog_type = HalProgrammableCoreType::DISPATCH,
                    });
                }
            }
        }
    }

    for (const auto& target : phase2_targets) {
        write_quiesce_control(ctx, target, dev_msgs::QUIESCE_CTRL_NEXT_OP);
    }

    // Detach DPrintServer, WatcherServer, and profiler dump thread *after* worker quiesce_msg_t.control
    // is set (so device_print_quiesce_requested() is already true on workers) and *before* Phase 2
    // polling finishes so zero host background threads touch NOC2AXI once Quiesced is returned.
    ctx.detach_debug_servers_for_quiesce();

    std::vector<QuiesceCoreDiagnostic> unquiesced_workers;
    if (!poll_until_all_quiesced(ctx, phase2_targets, deadline, unquiesced_workers)) {
        ctx.set_quiesce_context_state(MetalContext::QuiesceContextState::QuiesceTimedOut);
        log_warning(
            tt::LogMetal,
            "quiesce_for_soft_reset() timed out in Phase 2 (worker drain/park) with {} unquiesced core(s).",
            unquiesced_workers.size());
        return QuiesceResult{
            .status = QuiesceStatus::TimedOutInWorkers,
            .elapsed = elapsed_since(start_time),
            .unquiesced_cores = std::move(unquiesced_workers),
        };
    }

    // Final non-posted read_core() calls above already observed QUIESCE_STATUS_QUIESCED on every core.
    // Issue a host memory fence without calling cluster.l1_barrier() (which would issue new NOC2AXI writes).
    tt_driver_atomics::mfence();

    // Mark MetalContext quiesced so MeshDevice::close() and process-exit teardowns emit zero MMIO/NOC2AXI
    // transactions, and close MeshDevice host state inline.
    ctx.set_quiesce_context_state(MetalContext::QuiesceContextState::QuiescedForSoftReset);
    api_lock.unlock();
    mesh_device.close();

    return QuiesceResult{
        .status = QuiesceStatus::Quiesced,
        .elapsed = elapsed_since(start_time),
        .unquiesced_cores = {},
    };
}

}  // namespace tt::tt_metal::experimental
