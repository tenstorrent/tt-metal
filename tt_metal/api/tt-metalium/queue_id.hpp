// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// #warning "tt-metalium/queue_id.hpp header is deprecated. Replaced with #include <ttnn/api/ttnn/common/queue_id.hpp>"

#include <optional>
#include <tt_stl/strong_type.hpp>
#include <cstdint>

namespace tt::tt_metal {

using QueueId
    [[deprecated("This is deprecated and will be removed. Replaced with ttnn::QueueId in "
                 "#include <ttnn/api/ttnn/common/queue_id.hpp>")]] = ttsl::StrongType<uint8_t, struct QueueIdTag>;

[[deprecated(
    "This is deprecated and will be removed. Replaced with ttnn::raw_optional in "
    "#include <ttnn/api/ttnn/common/queue_id.hpp>")]]
inline std::optional<uint8_t> raw_optional(const std::optional<QueueId>& cq_id) {
    return cq_id.has_value() ? std::make_optional(cq_id.value().get()) : std::nullopt;
}

}  // namespace tt::tt_metal
