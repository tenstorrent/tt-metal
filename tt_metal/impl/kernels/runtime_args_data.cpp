// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/kernels/runtime_args_data.hpp" // Public API Header

#include "tt_metal/common/assert.hpp"

namespace tt::tt_metal {

std::uint32_t & RuntimeArgsData::operator[](std::size_t index) {
    TT_ASSERT(index < rt_args_count, "Index specified is larger than runtime args size");
    return this->rt_args_data[index];
}

const std::uint32_t & RuntimeArgsData::operator[](std::size_t index) const {
    TT_ASSERT(index < rt_args_count, "Index specified is larger than runtime args size");
    return this->rt_args_data[index];
}

std::uint32_t & RuntimeArgsData::at(std::size_t index) {
    TT_FATAL(index < rt_args_count, "Index specified is larger than runtime args size");
    return this->rt_args_data[index];
}

const std::uint32_t & RuntimeArgsData::at(std::size_t index) const {
    TT_FATAL(index < rt_args_count, "Index specified is larger than runtime args size");
    return this->rt_args_data[index];
}

std::uint32_t * RuntimeArgsData::data() noexcept {
    return rt_args_data;
}

const std::uint32_t * RuntimeArgsData::data() const noexcept {
    return rt_args_data;
}

std::size_t RuntimeArgsData::size() const noexcept {
    return rt_args_count;
}

}

