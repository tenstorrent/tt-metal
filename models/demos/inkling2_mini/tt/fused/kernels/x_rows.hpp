// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Decode activation tiles with only their first x_rows rows in use (the rest zero in the source): read those rows of
// faces 0 and 1 and fill the rest of the bf16 tile from this core's L1 zeros. The destination then holds the same
// bytes as a whole-tile read, with a fraction of the NOC traffic on the activation's shard cores.
#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

template <uint32_t x_rows>
FORCE_INLINE void read_x_tile(uint64_t src, uint32_t dst, uint64_t zeros) {
    static_assert(x_rows >= 1 && x_rows <= 16, "rows of faces 0 and 1 (bf16 tile: 32 B per face row)");
    static_assert(MEM_ZEROS_SIZE >= 512, "one face of zeros");
    constexpr uint32_t used = x_rows * 32, face = 512;
    noc_async_read(src, dst, used);
    noc_async_read(src + face, dst + face, used);
    if constexpr (used < face) {
        noc_async_read(zeros, dst + used, face - used);
        noc_async_read(zeros, dst + face + used, face - used);
    }
    noc_async_read(zeros, dst + 2 * face, face);
    noc_async_read(zeros, dst + 3 * face, face);
}

// The same into a 16 x 32 tile (faces 0 and 1 only).
template <uint32_t x_rows>
FORCE_INLINE void read_x_half_tile(uint64_t src, uint32_t dst, uint64_t zeros) {
    static_assert(x_rows >= 1 && x_rows <= 16, "rows of faces 0 and 1 (bf16 tile: 32 B per face row)");
    constexpr uint32_t used = x_rows * 32, face = 512;
    noc_async_read(src, dst, used);
    noc_async_read(src + face, dst + face, used);
    if constexpr (used < face) {
        noc_async_read(zeros, dst + used, face - used);
        noc_async_read(zeros, dst + face + used, face - used);
    }
}
