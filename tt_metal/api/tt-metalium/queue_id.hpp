// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt_stl/strong_type.hpp>
#include <cstdint>

namespace tt::tt_metal {
using QueueId = tt::stl::StrongType<uint8_t, struct QueueIdTag>;

inline std::optional<uint8_t> raw_optional(const std::optional<QueueId>& cq_id) {
    return cq_id.has_value() ? std::make_optional(cq_id.value().get()) : std::nullopt;
}

}  // namespace tt::tt_metal
