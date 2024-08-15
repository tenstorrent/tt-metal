// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/common/assert.hpp"

namespace tt::tt_metal {
// RuntimeArgsData provides an indirection to the runtime args
// Prior to generating the cq cmds for the device, this points into a vector within the kernel
// After generation, this points into the cq cmds so that runtime args API calls
// update the data directly in the command
struct RuntimeArgsData {
    uint32_t * rt_args_data;
    size_t rt_args_count;

    inline uint32_t & operator[](size_t index) {
        TT_ASSERT(index < rt_args_count, "Index specified is larger than runtime args size");
        return this->rt_args_data[index];
    }
    inline const uint32_t& operator[](size_t index) const {
        TT_ASSERT(index < rt_args_count, "Index specified is larger than runtime args size");
        return this->rt_args_data[index];
    }
    inline uint32_t & at(size_t index) {
        TT_FATAL(index < rt_args_count, "Index specified is larger than runtime args size");
        return this->rt_args_data[index];
    }
    inline const uint32_t& at(size_t index) const {
        TT_FATAL(index < rt_args_count, "Index specified is larger than runtime args size");
        return this->rt_args_data[index];
    }
    inline uint32_t * data() noexcept {
        return rt_args_data;
    }
    inline const uint32_t * data() const noexcept {
        return rt_args_data;
    }
    inline size_t size() const noexcept{
        return rt_args_count;
    }
};

};
