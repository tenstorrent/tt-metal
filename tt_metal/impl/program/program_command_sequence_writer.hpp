// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tt_metal/impl/program/program_command_sequence.hpp"

namespace tt::tt_metal::program_dispatch::queue_write_detail {

struct ProgramCommandWritePlan {
    uint32_t fetch_size;
    bool one_shot;
};

inline ProgramCommandWritePlan make_program_command_write_plan(
    const ProgramCommandSequence& program_command_sequence,
    bool stall_first,
    bool stall_before_program,
    bool send_binary,
    uint32_t max_prefetch_command_size) {
    const uint32_t fetch_size =
        program_command_sequence.get_one_shot_fetch_size(stall_first, stall_before_program, send_binary);
    return {.fetch_size = fetch_size, .one_shot = fetch_size <= max_prefetch_command_size};
}

// Keep the queue policy testable without constructing a device-backed SystemMemoryManager.
template <typename CommandQueueManager>
void write_program_command_sequence_to_queue(
    const ProgramCommandSequence& program_command_sequence,
    CommandQueueManager& manager,
    uint32_t command_queue_id,
    bool stall_first,
    bool stall_before_program,
    bool send_binary,
    ProgramCommandWritePlan write_plan) {
    if (write_plan.one_shot) {
        manager.issue_queue_reserve(write_plan.fetch_size, command_queue_id);
    }
    uint32_t one_shot_write_ptr = manager.get_issue_queue_write_ptr(command_queue_id);

    auto write_data_to_queue = [&](const void* data, uint32_t size_bytes) {
        if (size_bytes == 0) {
            return;
        }

        if (write_plan.one_shot) {
            manager.cq_write(data, size_bytes, one_shot_write_ptr);
            one_shot_write_ptr += size_bytes;
        } else {
            manager.issue_queue_reserve(size_bytes, command_queue_id);
            manager.cq_write(data, size_bytes, manager.get_issue_queue_write_ptr(command_queue_id));
            manager.issue_queue_push_back(size_bytes, command_queue_id);
            manager.fetch_queue_reserve_back(command_queue_id);
            manager.fetch_queue_write(size_bytes, command_queue_id);
        }
    };

    write_data_to_queue(
        program_command_sequence.preamble_command_sequence.data(),
        program_command_sequence.preamble_command_sequence.size_bytes());

    const uint32_t stall_sequence_index = program_command_sequence.current_stall_seq_idx;
    if (stall_first) {
        write_data_to_queue(
            program_command_sequence.stall_command_sequences[stall_sequence_index].data(),
            program_command_sequence.stall_command_sequences[stall_sequence_index].size_bytes());
    }

    for (const HostMemDeviceCommand& commands : program_command_sequence.runtime_args_command_sequences) {
        write_data_to_queue(commands.data(), commands.size_bytes());
    }

    for (const HostMemDeviceCommand& commands : program_command_sequence.program_config_buffer_command_sequences) {
        write_data_to_queue(commands.data(), commands.size_bytes());
    }

    if (stall_before_program) {
        write_data_to_queue(
            program_command_sequence.stall_command_sequences[stall_sequence_index].data(),
            program_command_sequence.stall_command_sequences[stall_sequence_index].size_bytes());
    }

    if (send_binary) {
        if (program_command_sequence.prefetcher_cache_used) {
            write_data_to_queue(
                program_command_sequence.program_binary_setup_prefetcher_cache_command.data(),
                program_command_sequence.program_binary_setup_prefetcher_cache_command.size_bytes());
        }
        write_data_to_queue(
            program_command_sequence.program_binary_command_sequence.data(),
            program_command_sequence.program_binary_command_sequence.size_bytes());
    } else {
        write_data_to_queue(
            program_command_sequence.wait_barrier_command_sequence.data(),
            program_command_sequence.wait_barrier_command_sequence.size_bytes());
    }

    write_data_to_queue(
        program_command_sequence.launch_msg_command_sequence.data(),
        program_command_sequence.launch_msg_command_sequence.size_bytes());
    write_data_to_queue(
        program_command_sequence.go_msg_command_sequence.data(),
        program_command_sequence.go_msg_command_sequence.size_bytes());

    if (write_plan.one_shot) {
        manager.issue_queue_push_back(write_plan.fetch_size, command_queue_id);
        manager.fetch_queue_reserve_back(command_queue_id);
        manager.fetch_queue_write(write_plan.fetch_size, command_queue_id);
    }
}

}  // namespace tt::tt_metal::program_dispatch::queue_write_detail
