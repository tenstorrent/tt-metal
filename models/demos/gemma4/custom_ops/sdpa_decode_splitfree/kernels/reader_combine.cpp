// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Phase 2 reader of sdpa_decode_splitfree: for every (row, group) pair of this core (pair bg on core bg mod
// NCORES) the live units' partials in ascending unit order, each (m, l, O) from its scratch slot into the
// double-buffered input CBs. cur_pos is read once into the compute's CB.
//
// compile-time args: the sf:: prefix (24) + TensorAccessorArgs(cur_pos, scratch)
// runtime args: [core, pos_addr, scratch_addr]
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
    const uint32_t scratch_addr = get_arg_val<uint32_t>(2);
    constexpr auto pos_args = TensorAccessorArgs<CT_ACCESSORS>();
    constexpr auto scratch_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr uint32_t f32_tile = get_tile_size(CB_IN_O);
    const auto scratch = TensorAccessor(scratch_args, scratch_addr, f32_tile);

    volatile tt_l1_ptr uint32_t* pos_l1 = read_cur_pos<CB_POS_C, POS_STICK>(noc, pos_args, pos_addr);
    uint32_t pos[B];
    for (uint32_t b = 0; b < B; ++b) {
        pos[b] = pos_l1[b];
    }
    CircularBuffer(CB_POS_C).push_back(1);

    CircularBuffer cb_m(CB_IN_M);
    CircularBuffer cb_l(CB_IN_L);
    CircularBuffer cb_o(CB_IN_O);
    for (uint32_t bg = core; bg < BG; bg += NCORES) {
        const Row r = row_of(pos[bg / G]);
        if (!r.live) {
            continue;
        }
        const uint32_t lo = unit_lo(r);
        const uint32_t hi = unit_hi(r);
        for (uint32_t u = lo; u <= hi; ++u) {
            const uint32_t page0 = slot_of(bg, u) * TILES_PER_SLOT;
            cb_m.reserve_back(1);
            noc.async_read(scratch, CoreLocalMem<uint32_t>(cb_m.get_write_ptr()), f32_tile, {.page_id = page0}, {});
            cb_l.reserve_back(1);
            noc.async_read(scratch, CoreLocalMem<uint32_t>(cb_l.get_write_ptr()), f32_tile, {.page_id = page0 + 1}, {});
            cb_o.reserve_back(DHT);
            uint32_t o_ptr = cb_o.get_write_ptr();
            for (uint32_t c = 0; c < DHT; ++c) {
                noc.async_read(scratch, CoreLocalMem<uint32_t>(o_ptr), f32_tile, {.page_id = page0 + 2 + c}, {});
                o_ptr += f32_tile;
            }
            noc.async_read_barrier();
            cb_m.push_back(1);
            cb_l.push_back(1);
            cb_o.push_back(DHT);
        }
    }
}
