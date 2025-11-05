// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>

#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/code_profiling_types.hpp"
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric {

// Code Profiling Timer Type Enum is declared in code_profiling_types.hpp
const std::unordered_map<std::string, CodeProfilingTimerType> CodeProfilingTimerTypeMap = {
    {"RECEIVER_CHANNEL_FORWARD", CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD},
    {"SENDER_CHANNEL_FORWARD", CodeProfilingTimerType::SENDER_CHANNEL_FORWARD},
};

CodeProfilingTimerType convert_to_code_profiling_timer_type(const std::string& timer_str);

std::string convert_code_profiling_timer_type_to_str(const CodeProfilingTimerType& timer_type);

}  // namespace tt::tt_fabric
