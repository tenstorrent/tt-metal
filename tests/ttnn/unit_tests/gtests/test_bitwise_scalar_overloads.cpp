// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::binary::test {

// bitwise_and/or/xor have both an int32_t and a uint32_t scalar overload; every integer scalar type must still
// resolve to exactly one of them.
template <typename T>
concept BitwiseScalarCallable = requires(const Tensor& t, T s) {
    ttnn::bitwise_and(t, s);
    ttnn::bitwise_or(t, s);
    ttnn::bitwise_xor(t, s);
};

static_assert(BitwiseScalarCallable<int8_t>);
static_assert(BitwiseScalarCallable<int16_t>);
static_assert(BitwiseScalarCallable<int32_t>);
static_assert(BitwiseScalarCallable<int64_t>);
static_assert(BitwiseScalarCallable<long>);
static_assert(BitwiseScalarCallable<long long>);
static_assert(BitwiseScalarCallable<uint8_t>);
static_assert(BitwiseScalarCallable<uint16_t>);
static_assert(BitwiseScalarCallable<uint32_t>);
static_assert(BitwiseScalarCallable<uint64_t>);
static_assert(BitwiseScalarCallable<std::size_t>);

template <typename T>
concept SelectsUInt32Overload = requires(const Tensor& t, T s) { ttnn::bitwise_and<T>(t, s); };

static_assert(SelectsUInt32Overload<uint32_t>);
static_assert(!SelectsUInt32Overload<int32_t>);
static_assert(!SelectsUInt32Overload<int64_t>);
static_assert(!SelectsUInt32Overload<uint64_t>);

}  // namespace ttnn::operations::binary::test
