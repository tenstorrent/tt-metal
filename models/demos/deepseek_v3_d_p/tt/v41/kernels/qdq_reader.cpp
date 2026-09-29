// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the fused V4.1 QDQ (qdq_compute.cpp): fills the reduce scaler tiles once, then streams this core's
// contiguous run of input tiles in blocks.
//
// compile_time_args = [cb_in, cb_scaler, halves, block, TensorAccessorArgs(input)...]
// runtime args      = [src_addr, num_tiles, start_tile]
//
// Scaler tiles (bf16, 1.0 in row 0 of the faces that take part, zero elsewhere; the MAX row reduce multiplies
// each column by its scaler entry, so a zero column is masked out of the |x| max):
//   halves == 1: one tile, all four faces (group 32 = the whole tile row).
//   halves == 2: tile 0 faces 0 and 2 (columns 0-15), tile 1 faces 1 and 3 (columns 16-31).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t halves = get_compile_time_arg_val(2);
    constexpr uint32_t block = get_compile_time_arg_val(3);
    constexpr auto src_args = TensorAccessorArgs<4>();

    constexpr uint32_t FACE_U32 = 16 * 16 / 2;  // bf16 face, two values per word
    constexpr uint32_t ROW_U32 = 16 / 2;
    constexpr uint32_t ONE_PAIR = 0x3F803F80u;  // two bf16 1.0

    Noc noc;
    DataflowBuffer scaler(cb_scaler);
    scaler.reserve_back(halves);
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scaler.get_write_ptr());
    for (uint32_t h = 0; h < halves; ++h) {
        for (uint32_t face = 0; face < 4; ++face) {
            const bool active = halves == 1 || (face % 2) == h;
            for (uint32_t w = 0; w < FACE_U32; ++w) {
                sc[(h * 4 + face) * FACE_U32 + w] = (active && w < ROW_U32) ? ONE_PAIR : 0u;
            }
        }
    }
    scaler.push_back(halves);

    const auto src = TensorAccessor(src_args, src_addr);
    DataflowBuffer in(cb_in);
    const uint32_t page_bytes = get_local_cb_interface(cb_in).fifo_page_size;
    const uint32_t end_tile = start_tile + num_tiles;
    for (uint32_t t = start_tile; t < end_tile;) {
        const uint32_t n = end_tile - t < block ? end_tile - t : block;
        in.reserve_back(n);
        for (uint32_t k = 0; k < n; ++k) {
            noc.async_read(src, in, page_bytes, {.page_id = t + k}, {.offset_bytes = k * page_bytes});
        }
        noc.async_read_barrier();
        in.push_back(n);
        t += n;
    }
}
