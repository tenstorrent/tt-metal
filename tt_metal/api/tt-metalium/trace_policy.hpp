// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

// Bit-flag set of policies applied to a trace-capture region.
//
// NONE (0) is the default: no policies are enforced. Each named bit opts into one specific
// behavior for the duration of a capture region. The type is named generically (not "safety")
// so future trace policies of any kind can be added as additional bits without renaming.
enum class TracePolicy : uint32_t {
    NONE = 0,
    // Opt into a fatal on ops whose program-cache key depends on live device state (e.g. matmul
    // auto-config, which queries free L1 to pick blocking parameters). Such ops are non-deterministic
    // across captures; set this bit when the capture must be reproducible and you want it enforced.
    REQUIRE_STABLE_CACHE = 1u << 0,
    // FUTURE_POLICY     = 1u << 1,
};

inline TracePolicy operator|(TracePolicy a, TracePolicy b) {
    return static_cast<TracePolicy>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

// Returns true if `bit` is set within `value`.
inline bool is_trace_policy_set(TracePolicy value, TracePolicy bit) {
    return (static_cast<uint32_t>(value) & static_cast<uint32_t>(bit)) != 0;
}

}  // namespace tt::tt_metal
