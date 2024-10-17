// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace tt::tt_metal {
// RuntimeArgsData provides an indirection to the runtime args
// Prior to generating the cq cmds for the device, this points into a vector within the kernel
// After generation, this points into the cq cmds so that runtime args API calls
// update the data directly in the command
struct RuntimeArgsData {
    std::uint32_t * rt_args_data;
    std::size_t rt_args_count;

    inline std::uint32_t & operator[](std::size_t index);

    inline const std::uint32_t& operator[](std::size_t index) const;

    inline std::uint32_t & at(std::size_t index);

    inline const std::uint32_t& at(std::size_t index) const;

    inline std::uint32_t * data() noexcept;

    inline const std::uint32_t * data() const noexcept;

    inline std::size_t size() const noexcept;
};

}
