// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include <tt-metalium/runtime_args_data.hpp>

namespace tt::tt_metal::detail {

struct RuntimeArgsDataAccess {
    static std::uint32_t*& ptr(RuntimeArgsData& data) { return data.rt_args_data; }
    static const std::uint32_t* ptr(const RuntimeArgsData& data) { return data.rt_args_data; }
    static std::size_t& count(RuntimeArgsData& data) { return data.rt_args_count; }
    static std::size_t count(const RuntimeArgsData& data) { return data.rt_args_count; }
};

}  // namespace tt::tt_metal::detail
