// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace ckernel {

// One input and one output CB of Float16_b, integer (Int32 / UInt32) CBs aside
template <std::size_t n, typename T>
constexpr bool llk_copy16_cbs(const T (&formats)[n]) {
    std::uint32_t count = 0;
    for (std::size_t i = 0; i < n; i++) {
        if (formats[i] == 255 || formats[i] == 8 || formats[i] == 24) {
            continue;
        }
        if (formats[i] != 5) {
            return false;
        }
        count++;
    }
    return count == 2;
}

}  // namespace ckernel

#define LLK_COPY_KERNEL_STR_(...) #__VA_ARGS__
#define LLK_COPY_KERNEL_STR(x) LLK_COPY_KERNEL_STR_(x)
// A unary op kernel whose SFPU chain is empty (identity) copies tiles through with no math between unpack and pack
#if defined(SFPU_OP_CHAIN_0) && defined(SFPU_OP_CHAIN_0_FUNC_0)
#define LLK_SFPU_CHAIN_EMPTY (sizeof(LLK_COPY_KERNEL_STR(SFPU_OP_CHAIN_0_FUNC_0)) == 1)
#else
#define LLK_SFPU_CHAIN_EMPTY false
#endif

// Use after the JIT descriptors: a half-sync 16-bit DEST identity copy of one Float16_b input to one output, whose pack runs as
// its unpacker reads the next tile (sharded identity +0.25 % with the early pack start and copy_tile's early poll)
#define LLK_COPY_KERNEL \
    (LLK_SFPU_CHAIN_EMPTY && !DST_ACCUM_MODE && DST_SYNC_MODE == DstSync::SyncHalf && ::ckernel::llk_copy16_cbs(unpack_src_format))
