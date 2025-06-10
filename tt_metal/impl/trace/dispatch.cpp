// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dev_msgs.h"
#include <algorithm>
#include <functional>

#include "assert.hpp"
#include "device.hpp"
#include "impl/context/metal_context.hpp"
#include "dispatch/kernels/cq_commands.hpp"
#include "dispatch_core_common.hpp"
#include "hal.hpp"
#include "hal_types.hpp"
#include "dispatch/launch_message_ring_buffer_state.hpp"
#include <tt_stl/strong_type.hpp>
#include "dispatch/system_memory_manager.hpp"
#include "trace_buffer.hpp"
#include "tt_align.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/trace/dispatch.hpp"
#include "dispatch/worker_config_buffer.hpp"

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
    IDevice* device,
    SystemMemoryManager& sysmem_manager,
    const TraceDispatchMetadata& dispatch_md,
    uint8_t cq_id,
    const DispatchArray<uint32_t>& expected_num_workers_completed,
    CoreCoord dispatch_core) {
    void* cmd_region = sysmem_manager.issue_queue_reserve(dispatch_md.cmd_sequence_sizeB, cq_id);

    HugepageDeviceCommand command_sequence(cmd_region, dispatch_md.cmd_sequence_sizeB);

    DispatcherSelect dispatcher_for_go_signal = DispatcherSelect::DISPATCH_MASTER;
    if (MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
        uint16_t index_bitmask = 0;
        for (const auto& id : dispatch_md.sub_device_ids) {
            index_bitmask |= 1 << *id;
        }
        command_sequence.add_notify_dispatch_s_go_signal_cmd(false, index_bitmask);
        dispatcher_for_go_signal = DispatcherSelect::DISPATCH_SUBORDINATE;
    }

    go_msg_t reset_launch_message_read_ptr_go_signal;
    reset_launch_message_read_ptr_go_signal.signal = RUN_MSG_RESET_READ_PTR;
    reset_launch_message_read_ptr_go_signal.master_x = (uint8_t)dispatch_core.x;
    reset_launch_message_read_ptr_go_signal.master_y = (uint8_t)dispatch_core.y;

    for (const auto& [id, desc] : dispatch_md.trace_worker_descriptors) {
        const auto& noc_data_start_idx = device->noc_data_start_index(
            id,
            desc.num_traced_programs_needing_go_signal_multicast,
            desc.num_traced_programs_needing_go_signal_unicast);

        const auto& num_noc_mcast_txns =
            desc.num_traced_programs_needing_go_signal_multicast ? device->num_noc_mcast_txns(id) : 0;
        const auto& num_noc_unicast_txns =
            desc.num_traced_programs_needing_go_signal_unicast ? device->num_virtual_eth_cores(id) : 0;
        auto index = *id;
        reset_launch_message_read_ptr_go_signal.dispatch_message_offset =
            MetalContext::instance().dispatch_mem_map().get_dispatch_message_update_offset(index);

        // Wait to ensure that all kernels have completed. Then send the reset_rd_ptr go_signal.
        command_sequence.add_dispatch_go_signal_mcast(
            expected_num_workers_completed[index],
            *reinterpret_cast<uint32_t*>(&reset_launch_message_read_ptr_go_signal),
            MetalContext::instance().dispatch_mem_map().get_dispatch_stream_index(index),
            num_noc_mcast_txns,
            num_noc_unicast_txns,
            noc_data_start_idx,
            dispatcher_for_go_signal);
    }

    // Wait to ensure that all workers have reset their read_ptr. dispatch_d will stall until all workers have completed
    // this step, before sending kernel config data to workers or notifying dispatch_s that its safe to send the
    // go_signal. Clear the dispatch <--> worker semaphore, since trace starts at 0.
    for (const auto& [id, desc] : dispatch_md.trace_worker_descriptors) {
        auto index = *id;
        uint32_t expected_num_workers = expected_num_workers_completed[index];
        if (desc.num_traced_programs_needing_go_signal_multicast) {
            expected_num_workers += device->num_worker_cores(HalProgrammableCoreType::TENSIX, id);
        }
        if (desc.num_traced_programs_needing_go_signal_unicast) {
            expected_num_workers += device->num_virtual_eth_cores(id);
        }

        if (MetalContext::instance().get_dispatch_query_manager().distributed_dispatcher()) {
            command_sequence.add_dispatch_wait(
                CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM | CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM,
                0,
                MetalContext::instance().dispatch_mem_map().get_dispatch_stream_index(index),
                expected_num_workers,
                1);
        }
        command_sequence.add_dispatch_wait(
            CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM | CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM,
            0,
            MetalContext::instance().dispatch_mem_map().get_dispatch_stream_index(index),
            expected_num_workers);
    }

    uint32_t page_size_log2 = __builtin_ctz(dispatch_md.trace_buffer_page_size);
    TT_ASSERT(
        (dispatch_md.trace_buffer_page_size & (dispatch_md.trace_buffer_page_size - 1)) == 0,
        "Page size must be a power of 2");

    command_sequence.add_prefetch_exec_buf(
        dispatch_md.trace_buffer_address, page_size_log2, dispatch_md.trace_buffer_num_pages);

    sysmem_manager.issue_queue_push_back(dispatch_md.cmd_sequence_sizeB, cq_id);

    sysmem_manager.fetch_queue_reserve_back(cq_id);

    const bool stall_prefetcher = true;
    sysmem_manager.fetch_queue_write(dispatch_md.cmd_sequence_sizeB, cq_id, stall_prefetcher);
}

uint32_t compute_trace_cmd_size(uint32_t num_sub_devices) {
    const auto& hal = MetalContext::instance().hal();
    uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    uint32_t go_signals_cmd_size =
        align(sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd), pcie_alignment) * num_sub_devices;

    uint32_t cmd_sequence_sizeB =
        MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled() *
            hal.get_alignment(
                HalMemType::HOST) +  // dispatch_d -> dispatch_s sem update (send only if dispatch_s is running)
        go_signals_cmd_size +        // go signal cmd
        (hal.get_alignment(
             HalMemType::HOST) +  // wait to ensure that reset go signal was processed (dispatch_d)
                                  // when dispatch_s and dispatch_d are running on 2 cores, workers update dispatch_s.
                                  // dispatch_s is responsible for resetting worker count and giving dispatch_d the
                                  // latest worker state. This is encapsulated in the dispatch_s wait command (only to
                                  // be sent when dispatch is distributed on 2 cores)
         (MetalContext::instance().get_dispatch_query_manager().distributed_dispatcher()) *
             hal.get_alignment(HalMemType::HOST)) *
            num_sub_devices +
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
        if (desc.num_traced_programs_needing_go_signal_unicast) {
            worker_launch_message_buffer_state[index].set_unicast_wptr(
                desc.num_traced_programs_needing_go_signal_unicast);
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
    constexpr uint32_t kExecBufPageMin = 1024;
    constexpr uint32_t kExecBufPageMax = 4096;
    // The algorithm below currently minimizes the amount of wasted space due to
    // padding. TODO: Tune for performance.
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
        if (waste <= min_waste) {
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
