// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ringbuffer_trace_allocator.hpp"

#include <optional>
#include <set>

#include "hal/generated/dev_msgs.hpp"
#include "llrt/hal.hpp"
#include "worker_config_buffer.hpp"
#include "tt_metal/impl/program/dispatch.hpp"

namespace tt::tt_metal {

void RingbufferTraceAllocator::allocate_trace_programs(const Hal& hal, std::vector<TraceNode*>& trace_nodes) {
    std::set<SubDeviceId> sub_device_ids;
    for (auto* node : trace_nodes) {
        sub_device_ids.insert(node->sub_device_id);
    }
    for (const auto& sub_device_id : sub_device_ids) {
        allocate_trace_programs_on_subdevice(hal, trace_nodes, sub_device_id);
    }
}

void RingbufferTraceAllocator::allocate_trace_programs_on_subdevice(
    const Hal& hal, std::vector<TraceNode*>& trace_nodes, SubDeviceId sub_device_id) {
    uint32_t programmable_core_count = hal.get_programmable_core_type_count();

    WorkerConfigBufferMgr config_buffer_mgr;
    for (uint32_t idx = 0; idx < programmable_core_count; idx++) {
        config_buffer_mgr.init_add_buffer(ringbuffer_configs_[idx].start, ringbuffer_configs_[idx].size);
    }
    config_buffer_mgr.init_add_buffer(0, dev_msgs::launch_msg_buffer_num_entries - 1);
    config_buffer_mgr.init_add_buffer(0, 1);

    uint32_t expected_num_workers_completed = 0;
    bool first_program_dispatched = false;

    std::optional<uint64_t> previous_program_id;
    TraceDispatchMetadata previous_dispatch_metadata{};
    bool previous_dispatch_can_be_reused = false;

    for (auto* node_ptr : trace_nodes) {
        auto& node = *node_ptr;
        if (node.sub_device_id != sub_device_id) {
            continue;
        }

        node.dispatch_metadata = TraceDispatchMetadata{};
        node.dispatch_metadata.binary_kernel_config_addrs.resize(programmable_core_count);
        node.dispatch_metadata.nonbinary_kernel_config_addrs.resize(programmable_core_count);

        uint32_t num_workers = node.num_workers;

        auto updated_worker_counts =
            program_dispatch::get_expected_num_workers_completed_updates(expected_num_workers_completed, num_workers);

        if (updated_worker_counts.wrapped) [[unlikely]] {
            config_buffer_mgr.mark_completely_full(0);
            node.dispatch_metadata.reset_worker_counts_before_program = true;
            previous_program_id.reset();
            previous_dispatch_can_be_reused = false;
        }

        expected_num_workers_completed = updated_worker_counts.current;
        uint32_t previous_expected = updated_worker_counts.previous;

        uint64_t current_program_id = node.program->get_id();
        bool reuse_previous =
            previous_dispatch_can_be_reused && previous_program_id.has_value() &&
            current_program_id == *previous_program_id;

        if (reuse_previous) {
            // Extend the lifetime of the previous program's config buffer entries
            // so they remain valid for this dispatch as well.
            config_buffer_mgr.extend_last_alloc_sync_count(expected_num_workers_completed, programmable_core_count);

            // Only reserve launch message slots; config data is reused in-place.
            auto program_config_sizes = node.program->get_program_config_sizes();
            std::vector<uint32_t> launch_only_sizes(program_config_sizes.size(), 0);
            launch_only_sizes[launch_only_sizes.size() - 2] = program_config_sizes[program_config_sizes.size() - 2];
            launch_only_sizes[launch_only_sizes.size() - 1] = program_config_sizes[program_config_sizes.size() - 1];

            auto [sync_info, reservation] = config_buffer_mgr.reserve(launch_only_sizes);

            if (sync_info.need_sync) {
                node.dispatch_metadata.stall_before_program = true;
                node.dispatch_metadata.sync_count = sync_info.sync_count;
                config_buffer_mgr.free(sync_info.sync_count);
            }
            config_buffer_mgr.alloc(expected_num_workers_completed);

            // Copy addresses from the previous dispatch.
            node.dispatch_metadata.nonbinary_kernel_config_addrs = previous_dispatch_metadata.nonbinary_kernel_config_addrs;
            node.dispatch_metadata.binary_kernel_config_addrs = previous_dispatch_metadata.binary_kernel_config_addrs;
            node.dispatch_metadata.send_binary = false;
        } else {
            // Normal path: full allocation via the standard dispatch helper.
            program_dispatch::ProgramDispatchMetadata dispatch_md;
            program_dispatch::reserve_space_in_kernel_config_buffer(
                config_buffer_mgr,
                node.program->get_program_config_sizes(),
                ProgramBinaryStatus::Committed,
                num_workers,
                previous_expected,
                dispatch_md);

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
        }

        if (!first_program_dispatched) {
            node.dispatch_metadata.sync_count = 0;
            node.dispatch_metadata.stall_first = true;
            node.dispatch_metadata.stall_before_program = false;
            first_program_dispatched = true;
        }

        previous_program_id = current_program_id;
        previous_dispatch_metadata = node.dispatch_metadata;
        previous_dispatch_can_be_reused = !reuse_previous;
    }
}

}  // namespace tt::tt_metal
