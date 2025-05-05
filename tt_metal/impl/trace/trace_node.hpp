// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "program/program_impl.hpp"

namespace tt::tt_metal {

// This struct contains all the information needed to execute a program on a device.
struct TraceNode {
    std::shared_ptr<detail::ProgramImpl> program;
    uint32_t program_runtime_id;
    SubDeviceId sub_device_id;

    // Matches rta_updates in the ProgramCommandSequence
    std::vector<std::vector<uint8_t>> rta_data;
    // Matches cb_configs_payloads in the ProgramCommandSequence
    std::vector<std::vector<uint32_t>> cb_configs_payloads;
};

}  // namespace tt::tt_metal
