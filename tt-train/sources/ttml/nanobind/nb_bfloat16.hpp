// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "tt-metalium/bfloat16.hpp"

namespace nanobind::detail {
template <>
struct dtype_traits<bfloat16> {
    static constexpr dlpack::dtype value{
        static_cast<uint8_t>(dlpack::dtype_code::Float),  // type code
        16,                                               // size in bits
        1                                                 // lanes (simd), usually set to 1
    };
    static constexpr auto name = const_name("bfloat16");
};
}  // namespace nanobind::detail
