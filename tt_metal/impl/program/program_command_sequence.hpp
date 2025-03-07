// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include "device_command.hpp"

struct CQDispatchWritePackedCmd;
struct launch_msg_t;

namespace tt::tt_metal {

inline namespace v0 {

class CircularBuffer;

}  // namespace v0

constexpr uint32_t UncachedStallSequenceIdx = 0;
constexpr uint32_t CachedStallSequenceIdx = 1;
struct ProgramCommandSequence {
    HostMemDeviceCommand preamble_command_sequence;
    uint32_t current_stall_seq_idx = 0;
    HostMemDeviceCommand stall_command_sequences[2];
    std::vector<HostMemDeviceCommand> runtime_args_command_sequences;
    uint32_t runtime_args_fetch_size_bytes;
    HostMemDeviceCommand device_command_sequence;
    std::vector<uint32_t*> cb_configs_payloads;
    std::vector<std::vector<std::shared_ptr<CircularBuffer>>> circular_buffers_on_core_ranges;
    std::vector<launch_msg_t*> go_signals;
    uint32_t program_config_buffer_data_size_bytes;
    std::vector<CQDispatchWritePackedCmd*> launch_msg_write_packed_cmd_ptrs;
    std::vector<CQDispatchWritePackedCmd*> unicast_launch_msg_write_packed_cmd_ptrs;
    CQDispatchGoSignalMcastCmd* mcast_go_signal_cmd_ptr;
};

}  // namespace tt::tt_metal
