// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Phase 2 writer of sdpa_decode_splitfree: per combine job (bg, blk) of this core, the job's CT bf16 output tiles
// of row b (pages b*DHt + blk*CT + c), the group's HPG head rows only (for_each_row_segment: the face-row
// segments the stock write_partial_tiles_to_memory writes line by line), whatever the group count: the merge read
// the group's rows of the partials alone, so the other rows of a tile are not results. The groups of one row
// write disjoint rows, the jobs of one row disjoint tiles; the padding head rows (>= G * HPG) are never written.
//
// compile-time args: the sf:: prefix (CT_ACCESSORS args) + TensorAccessorArgs(cur_pos, out)
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
    const auto out = TensorAccessor(out_args, out_addr, out_tile);

    volatile tt_l1_ptr uint32_t* pos_l1 = read_cur_pos<CB_POS_D, POS_STICK>(noc, pos_args, pos_addr);
    uint32_t pos[B];
    for (uint32_t b = 0; b < B; ++b) {
        pos[b] = pos_l1[b];
    }

    CircularBuffer cb_out(CB_OUT_FINAL);
    for_each_combine_job(core, pos, [&](uint32_t b, uint32_t g, uint32_t /*bg*/, uint32_t blk, const Row& /*r*/) {
        cb_out.wait_front(CT);
        const uint32_t l1 = cb_out.get_read_ptr();
        for (uint32_t c = 0; c < CT; ++c) {
            const uint32_t page = b * DHT + blk * CT + c;
            const uint32_t src = l1 + c * out_tile;
            for_each_row_segment<2>(g * HPG, HPG, [&](uint32_t off, uint32_t bytes) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(src + off), out, bytes, {}, {.page_id = page, .offset_bytes = off});
            });
        }
        noc.async_write_barrier();
        cb_out.pop_front(CT);
    });
}
