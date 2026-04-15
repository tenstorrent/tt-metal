// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "legacy_trace_allocator.hpp"

#include <set>

#include "hal/generated/dev_msgs.hpp"
#include "llrt/hal.hpp"
#include "worker_config_buffer.hpp"
#include "tt_metal/impl/program/dispatch.hpp"

namespace tt::tt_metal {

void LegacyTraceAllocator::allocate_trace_programs(const Hal& hal, std::vector<TraceNode*>& trace_nodes) {
    std::set<SubDeviceId> sub_device_ids;
    for (auto* node : trace_nodes) {
        sub_device_ids.insert(node->sub_device_id);
    }
    for (const auto& sub_device_id : sub_device_ids) {
        allocate_trace_programs_on_subdevice(hal, trace_nodes, sub_device_id);
    }
}

void LegacyTraceAllocator::allocate_trace_programs_on_subdevice(
    const Hal& hal, std::vector<TraceNode*>& trace_nodes, SubDeviceId sub_device_id) {
    uint32_t programmable_core_count = hal.get_programmable_core_type_count();

    // Initialize a fresh WorkerConfigBufferMgr with the same layout as the
    // normal dispatch path: one ring buffer per programmable core type, plus
    // two entries for multicast / unicast launch-message slot tracking.
    WorkerConfigBufferMgr config_buffer_mgr;
    for (uint32_t idx = 0; idx < programmable_core_count; idx++) {
        config_buffer_mgr.init_add_buffer(ringbuffer_configs_[idx].start, ringbuffer_configs_[idx].size);
    }
    config_buffer_mgr.init_add_buffer(0, dev_msgs::launch_msg_buffer_num_entries - 1);
    config_buffer_mgr.init_add_buffer(0, 1);

    uint32_t expected_num_workers_completed = 0;
    bool first_program_dispatched = false;

    for (auto* node_ptr : trace_nodes) {
        auto& node = *node_ptr;
        if (node.sub_device_id != sub_device_id) {
            continue;
        }

        // Reinitialize dispatch_metadata (same contract as SimpleTraceAllocator).
        node.dispatch_metadata = TraceDispatchMetadata{};
        node.dispatch_metadata.binary_kernel_config_addrs.resize(programmable_core_count);
        node.dispatch_metadata.nonbinary_kernel_config_addrs.resize(programmable_core_count);

        uint32_t num_workers = node.num_workers;

        auto updated_worker_counts =
            program_dispatch::get_expected_num_workers_completed_updates(expected_num_workers_completed, num_workers);

        if (updated_worker_counts.wrapped) [[unlikely]] {
            config_buffer_mgr.mark_completely_full(0);
            node.dispatch_metadata.reset_worker_counts_before_program = true;
        }

        expected_num_workers_completed = updated_worker_counts.current;
        uint32_t previous_expected = updated_worker_counts.previous;

        // Use the standard dispatch helper to reserve space in the ring buffer.
        program_dispatch::ProgramDispatchMetadata dispatch_md;
        program_dispatch::reserve_space_in_kernel_config_buffer(
            config_buffer_mgr,
            node.program->get_program_config_sizes(),
            ProgramBinaryStatus::Committed,
            num_workers,
            previous_expected,
            dispatch_md);

        // Map ProgramDispatchMetadata → TraceDispatchMetadata.
        // In the legacy contiguous allocation, binary and nonbinary share the
        // same base address.  update_traced_program_dispatch_commands computes
        //   binary_write_offset = binary_addr - kernel_text_offset
        // so we must set binary_addr to base + kernel_text_offset for Tensix
        // (where the dispatcher has a separate binary write offset) so that the
        // firmware sees the same value as the old update_program_dispatch_commands
        // path, which wrote the base address into both offset slots.
        for (uint32_t i = 0; i < programmable_core_count; i++) {
            uint32_t addr = dispatch_md.kernel_config_addrs[i].addr;
            node.dispatch_metadata.nonbinary_kernel_config_addrs[i] = {.addr = addr};

            auto core_type = hal.get_programmable_core_type(i);
            bool has_separate_binary_offset = (core_type == HalProgrammableCoreType::TENSIX);
            bool binary_in_config = hal.get_core_kernel_stored_in_config_buffer(core_type);
            if (has_separate_binary_offset && binary_in_config) {
                ProgramConfig& program_config = node.program->get_program_config(i);
                node.dispatch_metadata.binary_kernel_config_addrs[i] = {
                    .addr = addr + program_config.kernel_text_offset};
            } else {
                node.dispatch_metadata.binary_kernel_config_addrs[i] = {.addr = addr};
            }
        }
        node.dispatch_metadata.send_binary = true;
        node.dispatch_metadata.sync_count = dispatch_md.sync_count;
        node.dispatch_metadata.stall_first = dispatch_md.stall_first;
        node.dispatch_metadata.stall_before_program = dispatch_md.stall_before_program;

        if (!first_program_dispatched) {
            // Force a stall on the first program. Dummy GO signals may be
            // prepended before the trace stream and must complete before we
            // overwrite config data.
            node.dispatch_metadata.sync_count = 0;
            node.dispatch_metadata.stall_first = true;
            first_program_dispatched = true;
        }
    }
}

}  // namespace tt::tt_metal
