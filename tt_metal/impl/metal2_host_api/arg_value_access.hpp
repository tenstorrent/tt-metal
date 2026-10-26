// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/experimental/metal2_host_api/arg_value.hpp>

namespace tt::tt_metal::experimental {

// Implementation-only access to an ArgValue's stored value.
// (ArgValue's public interface is write-only by design; see arg_value.hpp.)
struct ArgValueAccess {
    static std::uint32_t u32(const ArgValue& value) noexcept { return value.u32_; }
};

}  // namespace tt::tt_metal::experimental
