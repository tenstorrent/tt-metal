// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>

#include <tt_stl/assert.hpp>
#include "device.hpp"
#include "mesh_device.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "dispatch/kernels/cq_commands.hpp"
#include "llrt/hal.hpp"
#include "hal_types.hpp"
#include "dispatch/launch_message_ring_buffer_state.hpp"
#include <tt_stl/strong_type.hpp>
#include "dispatch/system_memory_manager.hpp"
#include "trace_buffer.hpp"
#include "tt_align.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/trace/dispatch.hpp"
#include "dispatch/worker_config_buffer.hpp"
#include <impl/dispatch/dispatch_query_manager.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>
#include "tt_metal/impl/dispatch/device_command_calculator.hpp"

namespace tt::tt_metal::trace_dispatch {

void reset_host_dispatch_state_for_trace(
    uint32_t num_sub_devices,
    DispatchArray<LaunchMessageRingBufferState>& worker_launch_message_buffer_state,
    DispatchArray<uint32_t>& expected_num_workers_completed,
    DispatchArray<WorkerConfigBufferMgr>& config_buffer_mgr,
    DispatchArray<LaunchMessageRingBufferState>& worker_launch_message_buffer_state_reset,
    DispatchArray<uint32_t>& expected_num_workers_completed_reset,
    DispatchArray<WorkerConfigBufferMgr>& config_buffer_mgr_reset) {
    // Record the original value of expected_num_workers_completed, and reset it to 0.
    std::copy(
        expected_num_workers_completed.begin(),
        expected_num_workers_completed.begin() + num_sub_devices,
        expected_num_workers_completed_reset.begin());
    std::fill(expected_num_workers_completed.begin(), expected_num_workers_completed.begin() + num_sub_devices, 0);

    // Record original value of launch msg buffer
    std::copy(
        worker_launch_message_buffer_state.begin(),
        worker_launch_message_buffer_state.begin() + num_sub_devices,
        worker_launch_message_buffer_state_reset.begin());
    for (uint32_t i = 0; i < num_sub_devices; ++i) {
        // Set launch msg wptr to 0. Every time trace runs on device, it will ensure that the workers
        // reset their rptr to be in sync with device.
        worker_launch_message_buffer_state[i].reset();
    }
    // Record original value of config buffer manager
    std::copy(config_buffer_mgr.begin(), config_buffer_mgr.begin() + num_sub_devices, config_buffer_mgr_reset.begin());
    for (uint32_t i = 0; i < num_sub_devices; ++i) {
        // Sync values in the trace need to match up with the counter starting at 0 again.
        config_buffer_mgr[i].mark_completely_full(expected_num_workers_completed[i]);
    }
}

void load_host_dispatch_state(
    uint32_t num_sub_devices,
    DispatchArray<LaunchMessageRingBufferState>& worker_launch_message_buffer_state,
    DispatchArray<uint32_t>& expected_num_workers_completed,
    DispatchArray<WorkerConfigBufferMgr>& config_buffer_mgr,
    DispatchArray<LaunchMessageRingBufferState>& worker_launch_message_buffer_state_reset,
    DispatchArray<uint32_t>& expected_num_workers_completed_reset,
    DispatchArray<WorkerConfigBufferMgr>& config_buffer_mgr_reset) {
    std::copy(
        expected_num_workers_completed_reset.begin(),
        expected_num_workers_completed_reset.begin() + num_sub_devices,
        expected_num_workers_completed.begin());
    std::copy(
        worker_launch_message_buffer_state_reset.begin(),
        worker_launch_message_buffer_state_reset.begin() + num_sub_devices,
        worker_launch_message_buffer_state.begin());
    std::copy(
        config_buffer_mgr_reset.begin(), config_buffer_mgr_reset.begin() + num_sub_devices, config_buffer_mgr.begin());
}

void issue_trace_commands(
    distributed::MeshDevice* mesh_device,
    SystemMemoryManager& sysmem_manager,
    const TraceDispatchMetadata& dispatch_md,
    uint8_t cq_id,
    const DispatchArray<uint32_t>& expected_num_workers_completed,
    const DispatchArray<LaunchMessageRingBufferState>& worker_launch_message_buffer_state,
    const TraceBoundaryPrograms* boundary,
    const TraceBoundaryPrograms* previous) {
    MetalContext& metal_ctx = MetalContext::instance(sysmem_manager.get_context_id());

    const bool has_first_program_config = boundary != nullptr && boundary->first_program_config_num_pages > 0;
    // The first program's kernel config can be written before the barrier when only the previous trace's last program
    // may still be running (the wait below covers the others) and that program does not use the ranges it goes to.
    const bool early_first_program_config =
        has_first_program_config && previous != nullptr && dispatch_md.trace_worker_descriptors.size() == 1 &&
        !previous->last_program_config_regions.empty() &&
        expected_num_workers_completed[*dispatch_md.trace_worker_descriptors.begin()->first] >=
            previous->last_program_num_workers &&
        std::none_of(
            boundary->first_program_config_regions.begin(),
            boundary->first_program_config_regions.end(),
            [&](const KernelConfigRegion& a) {
                return std::any_of(
                    previous->last_program_config_regions.begin(),
                    previous->last_program_config_regions.end(),
                    [&](const KernelConfigRegion& b) {
                        return a.core_type_index == b.core_type_index && a.begin < b.end && b.begin < a.end;
                    });
            });

    const uint32_t exec_buf_sizeB = metal_ctx.hal().get_alignment(HalMemType::HOST);  // CQ_PREFETCH_CMD_EXEC_BUF
    const uint32_t barrier_sizeB = dispatch_md.cmd_sequence_sizeB - exec_buf_sizeB;
    uint32_t wait_sizeB = 0;
    if (early_first_program_config) {
        DeviceCommandCalculator calculator(metal_ctx);
        calculator.add_dispatch_wait();
        wait_sizeB = calculator.write_offset_bytes();
    }

    // Wait until every program of the previous trace but the last has completed. Programs complete in order, so that
    // is the worker count before the last program.
    auto write_wait_for_previous = [&](void* region) {
        const auto index = *dispatch_md.trace_worker_descriptors.begin()->first;
        TT_ASSERT(expected_num_workers_completed[index] >= previous->last_program_num_workers);
        HugepageDeviceCommand command(metal_ctx, region, wait_sizeB);
        command.add_dispatch_wait(
            CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM,
            0,
            metal_ctx.dispatch_mem_map().get_dispatch_stream_index(index),
            expected_num_workers_completed[index] - previous->last_program_num_workers);
    };

    // The trace-boundary barrier: wait for all workers and reset their launch message read pointers.
    auto write_barrier = [&](void* region) {
        HugepageDeviceCommand command_sequence(metal_ctx, region, barrier_sizeB);

        DispatcherSelect dispatcher_for_go_signal = DispatcherSelect::DISPATCH_MASTER;
        if (metal_ctx.get_dispatch_query_manager().dispatch_s_enabled()) {
            uint16_t index_bitmask = 0;
            for (const auto& id : dispatch_md.sub_device_ids) {
                index_bitmask |= 1 << *id;
            }
            command_sequence.add_notify_dispatch_s_go_signal_cmd(false, index_bitmask);
            dispatcher_for_go_signal = DispatcherSelect::DISPATCH_SUBORDINATE;
        }

        for (const auto& [id, desc] : dispatch_md.trace_worker_descriptors) {
            auto index = *id;

            // Wait to ensure that all kernels have completed. Then send the reset_rd_ptr go_signal.
            command_sequence.add_dispatch_go_signal_mcast(
                expected_num_workers_completed[index],
                metal_ctx.hal().make_go_msg_u32(
                    dev_msgs::RUN_MSG_REPLAY_TRACE,
                    metal_ctx.dispatch_mem_map().get_dispatch_message_update_offset(index)),
                metal_ctx.dispatch_mem_map().get_dispatch_stream_index(index),
                desc.num_traced_programs_needing_go_signal_multicast && mesh_device->impl().has_noc_mcast_txns(id)
                    ? index
                    : CQ_DISPATCH_CMD_GO_NO_MULTICAST_OFFSET,
                worker_launch_message_buffer_state[index].get_mcast_wptr(),
                dispatcher_for_go_signal);
        }

        // Wait to ensure that all workers have reset their read_ptr. dispatch_d will stall until all workers have
        // completed this step, before sending kernel config data to workers or notifying dispatch_s that its safe to
        // send the go_signal. Clear the dispatch <--> worker semaphore, since trace starts at 0.
        for (const auto& [id, desc] : dispatch_md.trace_worker_descriptors) {
            auto index = *id;
            uint32_t expected_num_workers = expected_num_workers_completed[index];
            if (desc.num_traced_programs_needing_go_signal_multicast) {
                expected_num_workers += mesh_device->num_worker_cores(HalProgrammableCoreType::TENSIX, id);
            }

            if (metal_ctx.get_dispatch_query_manager().distributed_dispatcher()) {
                command_sequence.add_dispatch_wait(
                    CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM | CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM,
                    0,
                    metal_ctx.dispatch_mem_map().get_dispatch_stream_index(index),
                    expected_num_workers,
                    1);
            }
            command_sequence.add_dispatch_wait(
                CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM | CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM,
                0,
                metal_ctx.dispatch_mem_map().get_dispatch_stream_index(index),
                expected_num_workers);
        }
    };

    uint32_t page_size_log2 = __builtin_ctz(dispatch_md.trace_buffer_page_size);
    TT_ASSERT(
        (dispatch_md.trace_buffer_page_size & (dispatch_md.trace_buffer_page_size - 1)) == 0,
        "Page size must be a power of 2");
    auto write_exec_buf = [&](void* region, uint32_t address, uint32_t num_pages, uint32_t start_page) {
        HugepageDeviceCommand command(metal_ctx, region, exec_buf_sizeB);
        command.add_prefetch_exec_buf(address, page_size_log2, num_pages, start_page);
    };

    // Each exec buffer ends a fetch queue entry that stalls the prefetcher, since it reuses the command data queue.
    auto issue_entry = [&](uint32_t head_sizeB,
                           const std::function<void(void*)>& write_head,
                           uint32_t address,
                           uint32_t num_pages,
                           uint32_t start_page) {
        const uint32_t sizeB = head_sizeB + exec_buf_sizeB;
        auto* region = static_cast<uint8_t*>(sysmem_manager.issue_queue_reserve(sizeB, cq_id));
        if (head_sizeB != 0) {
            write_head(region);
        }
        write_exec_buf(region + head_sizeB, address, num_pages, start_page);
        sysmem_manager.issue_queue_push_back(sizeB, cq_id);
        sysmem_manager.fetch_queue_reserve_back(cq_id);
        const bool stall_prefetcher = true;
        sysmem_manager.fetch_queue_write(sizeB, cq_id, stall_prefetcher);
    };

    const uint32_t trace_address = dispatch_md.trace_buffer_address;
    const uint32_t trace_pages = dispatch_md.trace_buffer_num_pages;
    if (has_first_program_config) {
        const uint32_t config_address = trace_address + boundary->first_program_config_address_offset;
        const uint32_t config_start = boundary->first_program_config_start_page;
        const uint32_t config_pages = boundary->first_program_config_num_pages;
        if (early_first_program_config) {
            issue_entry(wait_sizeB, write_wait_for_previous, config_address, config_pages, config_start);
            issue_entry(barrier_sizeB, write_barrier, trace_address, trace_pages, 0);
        } else {
            issue_entry(barrier_sizeB, write_barrier, config_address, config_pages, config_start);
            issue_entry(0, {}, trace_address, trace_pages, 0);
        }
        return;
    }
    issue_entry(barrier_sizeB, write_barrier, trace_address, trace_pages, 0);
}

uint32_t compute_trace_cmd_size(ContextId context_id, uint32_t num_sub_devices) {
    MetalContext& metal_ctx = MetalContext::instance(context_id);
    const auto& hal = metal_ctx.hal();
    uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t go_signals_cmd_size =
        align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), pcie_alignment) * num_sub_devices;

    uint32_t cmd_sequence_sizeB =
        (metal_ctx.get_dispatch_query_manager().dispatch_s_enabled() *
         hal.get_alignment(
             HalMemType::HOST)) +  // dispatch_d -> dispatch_s sem update (send only if dispatch_s is running)
        go_signals_cmd_size +      // go signal cmd
        ((hal.get_alignment(
              HalMemType::HOST) +  // wait to ensure that reset go signal was processed (dispatch_d)
                                   // when dispatch_s and dispatch_d are running on 2 cores, workers update dispatch_s.
                                   // dispatch_s is responsible for resetting worker count and giving dispatch_d the
                                   // latest worker state. This is encapsulated in the dispatch_s wait command (only to
                                   // be sent when dispatch is distributed on 2 cores)
          (metal_ctx.get_dispatch_query_manager().distributed_dispatcher()) * hal.get_alignment(HalMemType::HOST)) *
         num_sub_devices) +
        hal.get_alignment(HalMemType::HOST);  // CQ_PREFETCH_CMD_EXEC_BUF

    return cmd_sequence_sizeB;
}

void update_worker_state_post_trace_execution(
    const std::unordered_map<SubDeviceId, TraceWorkerDescriptor>& trace_worker_descriptors,
    DispatchArray<LaunchMessageRingBufferState>& worker_launch_message_buffer_state,
    DispatchArray<WorkerConfigBufferMgr>& config_buffer_mgr,
    DispatchArray<uint32_t>& expected_num_workers_completed) {
    for (const auto& [id, desc] : trace_worker_descriptors) {
        auto index = *id;
        // Update the expected worker cores counter due to trace programs completion
        expected_num_workers_completed[index] = desc.num_completion_worker_cores;
        // After trace runs, the rdptr on each worker will be incremented by the number of programs in the trace
        // Update the wptr on host to match state. If the trace doesn't execute on a
        // class of worker (unicast or multicast), it doesn't reset or modify the
        // state for those workers.
        if (desc.num_traced_programs_needing_go_signal_multicast) {
            worker_launch_message_buffer_state[index].set_mcast_wptr(
                desc.num_traced_programs_needing_go_signal_multicast);
        }
        // The config buffer manager is unaware of what memory is used inside the trace, so mark all memory as used so
        // that it will force a stall and avoid stomping on in-use state.
        // TODO(jbauman): Reuse old state from the trace.
        config_buffer_mgr[index].mark_completely_full(expected_num_workers_completed[index]);
    }
}

// Assumes pages are interleaved across all banks starting at 0
std::size_t compute_interleaved_trace_buf_page_size(uint32_t buf_size, const uint32_t num_banks) {
    // Tuneable parameters for the trace buffer - heavily affect prefetcher
    // read performance. TODO: Explore ideal page size for the trace buffer
    // to maximize read bandwidth.
    // Min size is bounded by NOC transfer efficiency
    // Max size is bounded by Prefetcher CmdDatQ size
    constexpr uint32_t kExecBufPageMin = kMinTraceBufPageSize;
    constexpr uint32_t kExecBufPageMax = kMaxTraceBufPageSize;
    // If the trace buffer uses at least 2 pages per bank (for a specific page size), require using that page size or
    // larger to improve prefetcher read performance. This limits wasted space to at most 33% or the page size * the
    // number of banks, whichever is smaller.
    constexpr uint32_t kMinPagesPerBankForceUse = 2;

    // The algorithm below currently minimizes the amount of wasted space due to
    // padding
    std::vector<uint32_t> candidates;
    candidates.reserve(__builtin_clz(kExecBufPageMin) - __builtin_clz(kExecBufPageMax) + 1);
    for (uint32_t size = 1; size <= kExecBufPageMax; size <<= 1) {
        if (size >= kExecBufPageMin) {
            candidates.push_back(size);
        }
    }
    uint32_t min_waste = -1;
    uint32_t pick = 0;
    // Pick the largest size that minimizes waste
    for (const uint32_t size : candidates) {
        // Pad data to the next fully banked size
        uint32_t fully_banked = num_banks * size;
        uint32_t padded_size = (buf_size + fully_banked - 1) / fully_banked * fully_banked;
        uint32_t waste = padded_size - buf_size;
        if (waste <= min_waste || (buf_size / (num_banks * size) >= kMinPagesPerBankForceUse)) {
            min_waste = waste;
            pick = size;
        }
    }
    TT_FATAL(
        pick >= kExecBufPageMin and pick <= kExecBufPageMax,
        "pick {} not between min_size {} and max_size {}",
        pick,
        kExecBufPageMin,
        kExecBufPageMax);
    return pick;
}

}  // namespace tt::tt_metal::trace_dispatch
