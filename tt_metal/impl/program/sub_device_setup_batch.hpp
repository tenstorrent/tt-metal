// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>
#include "impl/dispatch/vector_aligned.hpp"

namespace tt::tt_metal::program_dispatch::setup_batch {

// Keep each command group intact and preserve emission order across fetches.
inline void append_setup_commands(
    std::vector<vector_aligned<uint32_t>>& batches, ttsl::Span<const uint32_t> commands, size_t max_bytes) {
    const auto bytes = commands.size() * sizeof(uint32_t);
    TT_FATAL(bytes <= max_bytes, "Sub-device setup command exceeds the maximum fetch size");
    if (batches.empty() || batches.back().size() * sizeof(uint32_t) > max_bytes - bytes) {
        batches.emplace_back();
    }
    batches.back().insert(batches.back().end(), commands.begin(), commands.end());
}

inline bool can_combine_setup(size_t reset_bytes, size_t setup_bytes, size_t max_bytes) {
    return reset_bytes <= max_bytes && setup_bytes <= max_bytes - reset_bytes;
}

}  // namespace tt::tt_metal::program_dispatch::setup_batch
