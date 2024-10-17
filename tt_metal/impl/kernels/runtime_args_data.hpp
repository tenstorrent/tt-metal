// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace tt::tt_metal {
// RuntimeArgsData provides an indirection to the runtime args
// Prior to generating the cq cmds for the device, this points into a vector within the kernel
// After generation, this points into the cq cmds so that runtime args API calls
// update the data directly in the command
struct RuntimeArgsData {
    std::uint32_t * rt_args_data;
    std::size_t rt_args_count;

    inline bool in_bounds(std::size_t index) const {
        if(index >= rt_args_count) [[unlikely]] {
            std::cerr << "TT_FATAL: Index " << index << " is larger than runtime args size " 
                      << rt_args_count << " at " << __FILE__ << ":" << __LINE__ << std::endl;
            throw std::out_of_range(
                "Index " + std::to_string(index) + " is larger than runtime args size " + std::to_string(rt_args_count)
            );
            __builtin_unreachable();
        }
        return true;
    }

    inline std::uint32_t & operator[](std::size_t index) noexcept {
        return this->rt_args_data[index];
    }
    inline const std::uint32_t& operator[](std::size_t index) const noexcept {
        return this->rt_args_data[index];
    }
    inline std::uint32_t & at(std::size_t index) {
        if (in_bounds(index)) [[likely]] {
            return this->rt_args_data[index];
        }
        __builtin_unreachable();
    }
    inline const std::uint32_t& at(std::size_t index) const {
        if (in_bounds(index)) [[likely]] {
            return this->rt_args_data[index];
        }
        __builtin_unreachable();
    }
    inline std::uint32_t * data() noexcept {
        return rt_args_data;
    }
    inline const std::uint32_t * data() const noexcept {
        return rt_args_data;
    }
    inline std::size_t size() const noexcept{
        return rt_args_count;
    }
};

};
