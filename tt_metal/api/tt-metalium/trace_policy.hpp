// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

// Bit-flag set of policies applied to a trace-capture region.
//
// STRICT (0) is the default and applies no relaxations: trace-safety checks fire as normal.
// Each named bit opts into one specific behavior for the duration of a capture region.
// The type is named generically (not "safety") so future non-safety trace policies can be
// added as additional bits without renaming.
enum class TracePolicy : uint32_t {
    STRICT = 0,
    // Permit ops whose program-cache key depends on live device state (e.g. matmul auto-config,
    // which queries free L1 to pick blocking parameters). Such ops are non-deterministic across
    // captures; setting this bit asserts the caller has verified the capture is stable regardless.
    ALLOW_UNSTABLE_CACHE = 1u << 0,
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
