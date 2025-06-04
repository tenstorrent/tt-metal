// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "program/program_impl.hpp"

namespace tt::tt_metal {

struct TraceDispatchMetadata {
    bool send_binary = true;
    // Addresses to write kernel binaries to.
    std::vector<ConfigBufferEntry> binary_kernel_config_addrs;
    // Addresses to write a contiguous chunk containing RTAs, CBs, and semaphores.
    std::vector<ConfigBufferEntry> nonbinary_kernel_config_addrs;
    uint32_t sync_count = 0;
    uint32_t stall_first = false;
    uint32_t stall_before_program = false;
};

// This struct contains all the information needed to execute a program on a device.
struct TraceNode {
    std::shared_ptr<detail::ProgramImpl> program;
    uint32_t program_runtime_id;
    SubDeviceId sub_device_id;

    // Matches rta_updates in the ProgramCommandSequence
    std::vector<std::vector<uint8_t>> rta_data;
    // Matches cb_configs_payloads in the ProgramCommandSequence
    std::vector<std::vector<uint32_t>> cb_configs_payloads;

    TraceDispatchMetadata dispatch_metadata;
};

}  // namespace tt::tt_metal
