// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Phase 2 writer of sdpa_decode_splitfree: per (row, group) pair of this core, the bf16 output tiles of row b
// (pages b*DHt + c): whole tiles when the call attends one KV group, else the group's HPG head rows as
// 32-byte face lines (the stock write_partial_tiles_to_memory), so the groups of one row write disjoint rows.
//
// compile-time args: the sf:: prefix (24) + TensorAccessorArgs(cur_pos, out)
// runtime args: [core, pos_addr, out_addr]
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "splitfree_common.hpp"
#include "splitfree_dataflow.hpp"

void kernel_main() {
    using namespace sf;
    Noc noc;
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t pos_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    constexpr auto pos_args = TensorAccessorArgs<CT_ACCESSORS>();
    constexpr auto out_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr uint32_t out_tile = get_tile_size(CB_OUT_FINAL);
    constexpr uint32_t ELEMENT_SIZE = 2;  // bf16
    constexpr uint32_t FACE_LINE_BYTES = 16 * ELEMENT_SIZE;
    constexpr uint32_t FACE_BYTES = 16 * 16 * ELEMENT_SIZE;
    const auto out = TensorAccessor(out_args, out_addr, out_tile);

    volatile tt_l1_ptr uint32_t* pos_l1 = read_cur_pos<CB_POS_D, POS_STICK>(noc, pos_args, pos_addr);
    uint32_t pos[B];
    for (uint32_t b = 0; b < B; ++b) {
        pos[b] = pos_l1[b];
    }

    CircularBuffer cb_out(CB_OUT_FINAL);
    for (uint32_t bg = core; bg < BG; bg += NCORES) {
        const uint32_t b = bg / G;
        const uint32_t g = bg - b * G;
        const Row r = row_of(pos[b]);
        if (!r.live) {
            continue;
        }
        cb_out.wait_front(DHT);
        const uint32_t l1 = cb_out.get_read_ptr();
        for (uint32_t c = 0; c < DHT; ++c) {
            const uint32_t page = b * DHT + c;
            if constexpr (G == 1) {
                noc.async_write(CoreLocalMem<uint32_t>(l1 + c * out_tile), out, out_tile, {}, {.page_id = page});
            } else {
                for (uint32_t h = 0; h < HPG; ++h) {
                    const uint32_t row = g * HPG + h;
                    const uint32_t off = (row < 16) ? row * FACE_LINE_BYTES : (row + 16) * FACE_LINE_BYTES;
                    const uint32_t src = l1 + c * out_tile + off;
                    noc.async_write(
                        CoreLocalMem<uint32_t>(src), out, FACE_LINE_BYTES, {}, {.page_id = page, .offset_bytes = off});
                    noc.async_write(
                        CoreLocalMem<uint32_t>(src + FACE_BYTES),
                        out,
                        FACE_LINE_BYTES,
                        {},
                        {.page_id = page, .offset_bytes = off + FACE_BYTES});
                }
            }
        }
        noc.async_write_barrier();
        cb_out.pop_front(DHT);
    }
}
