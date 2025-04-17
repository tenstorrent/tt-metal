// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include "tt_metal/impl/dispatch/device_command.hpp"

struct CQDispatchWritePackedCmd;
struct launch_msg_t;

namespace tt::tt_metal {

class CircularBuffer;

constexpr uint32_t UncachedStallSequenceIdx = 0;
constexpr uint32_t CachedStallSequenceIdx = 1;

struct ProgramCommandSequence {
    struct RtaUpdate {
        const void* src;
        void* dst;
        uint32_t size;
    };
    HostMemDeviceCommand preamble_command_sequence;
    uint32_t current_stall_seq_idx = 0;
    HostMemDeviceCommand stall_command_sequences[2];
    std::vector<HostMemDeviceCommand> runtime_args_command_sequences;
    uint32_t runtime_args_fetch_size_bytes;
    HostMemDeviceCommand device_command_sequence;
    std::vector<uint32_t*> cb_configs_payloads;
    std::vector<std::vector<std::shared_ptr<CircularBuffer>>> circular_buffers_on_core_ranges;
    // Note: some RTAs may be have their RuntimeArgsData modified so the source-of-truth of their data is the command
    // sequence. They won't be listed in rta_updates.
    std::vector<RtaUpdate> rta_updates;
    std::vector<launch_msg_t*> go_signals;
    uint32_t program_config_buffer_data_size_bytes;
    std::vector<CQDispatchWritePackedCmd*> launch_msg_write_packed_cmd_ptrs;
    std::vector<CQDispatchWritePackedCmd*> unicast_launch_msg_write_packed_cmd_ptrs;
    CQDispatchGoSignalMcastCmd* mcast_go_signal_cmd_ptr;
};

}  // namespace tt::tt_metal
