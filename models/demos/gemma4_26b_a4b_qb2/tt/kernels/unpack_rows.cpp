// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// UnpackRows (WS2, batch-1 decode): the inverse of PackRows into two width-sharded [1,1,1,W] bf16 TILE
// tensors, run on their shard cores: shard core k owns tiles k*per .. k*per+per-1 of each output; output
// tile j row 0 <- packed tile (j' / 32) row (j' % 32) with j' = j (a) or na + j (b); rows 1-31 are zero.
// compile-time args: [per, na] + TensorAccessorArgs(packed)
// runtime args: [packed_addr, a_addr, b_addr, k] (a/b: the outputs' shard base address, same on every core)
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t packed_addr = get_arg_val<uint32_t>(0);
    const uint32_t a_addr = get_arg_val<uint32_t>(1);
    const uint32_t b_addr = get_arg_val<uint32_t>(2);
    const uint32_t k = get_arg_val<uint32_t>(3);
    constexpr uint32_t per = get_compile_time_arg_val(0);
    constexpr uint32_t na = get_compile_time_arg_val(1);
    constexpr auto p_args = TensorAccessorArgs<2>();
    constexpr uint32_t tile_bytes = 2048, face_bytes = 512, row_bytes = 32;
    const auto s_p = TensorAccessor(p_args, packed_addr, tile_bytes);
    for (uint32_t which = 0; which < 2; ++which) {
        const uint32_t base = which == 0 ? a_addr : b_addr;
        for (uint32_t t = 0; t < per; ++t) {
            const uint32_t local = base + t * tile_bytes;
            volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local);
            // rows 1-31 are zero; row 0 (faces 0 and 1, first 32 bytes each) comes from the read below
            for (uint32_t w = 0; w < tile_bytes / 4; ++w) {
                const uint32_t in_face = w & (face_bytes / 4 - 1);
                if (w < 2 * (face_bytes / 4) && in_face < row_bytes / 4) {
                    continue;
                }
                words[w] = 0;
            }
            const uint32_t i = (which == 0 ? 0 : na) + k * per + t;
            const uint32_t r = i & 31;
            const uint64_t src = s_p.get_noc_addr(i >> 5) + (r >> 4) * 2 * face_bytes + (r & 15) * row_bytes;
            noc_async_read(src, local, row_bytes);
            noc_async_read(src + face_bytes, local + face_bytes, row_bytes);
        }
    }
    noc_async_read_barrier();
}
