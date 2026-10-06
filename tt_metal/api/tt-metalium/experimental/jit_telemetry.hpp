// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tt::tt_metal::experimental::jit_telemetry {

// Aggregates of one JIT telemetry token, e.g. "program_config_size.total.TENSIX" (bytes). Token
// names may change; treat a missing token as unknown.
struct TokenStats {
    std::string name;
    std::string unit;
    uint32_t count = 0;
    double total = 0;
    double min = 0;
    double max = 0;
};

/**
 * Start a capture: until end_capture(), every token also aggregates what it records, from any
 * thread, into the capture. Captures may nest or overlap and do not affect the process-wide values.
 *
 * Return value: the capture id.
 */
uint64_t begin_capture();

/**
 * Close a capture and return the tokens that recorded into it. Throws if `capture_id` is not open.
 */
std::vector<TokenStats> end_capture(uint64_t capture_id);

/**
 * Process-wide values of every token.
 */
std::vector<TokenStats> snapshot();

}  // namespace tt::tt_metal::experimental::jit_telemetry
