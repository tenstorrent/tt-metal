// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Phase 2 reader of sdpa_decode_splitfree: for every combine job (bg, blk) of this core (splitfree_common.hpp:
// for_each_combine_job) the live units' partials from their scratch slots, in ascending unit order (the job's O
// tiles are [blk * CT, (blk + 1) * CT) of a slot's O). Stats merge (STATS): the row's m and l in one batch
// (m_u at page u - lo, l_u at NSLOTS + u - lo of a 2 * NSLOTS reservation), then per dest window of NB O tiles
// the units' tiles in batches of UB units (UB * NB pages, the last batch padded): every reservation is a fixed
// size dividing its CB, so the compute indexes the pages without a wrap. Pairwise merge: per unit its m, l and CT
// O tiles into the double-buffered input CBs. cur_pos is read once into the compute's CB.
//
// compile-time args: the sf:: prefix (CT_ACCESSORS args) + TensorAccessorArgs(cur_pos, scratch)
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

    CircularBuffer cb_o(CB_IN_O);
    if constexpr (STATS) {
        CircularBuffer cb_s(CB_IN_S);
        for_each_combine_job(core, pos, [&](uint32_t /*b*/, uint32_t /*g*/, uint32_t bg, uint32_t blk, const Row& r) {
            const uint32_t lo = unit_lo(r);
            const uint32_t n = unit_hi(r) - lo + 1;
            cb_s.reserve_back(2 * NSLOTS);
            const uint32_t s_ptr = cb_s.get_write_ptr();
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t page0 = slot_of(bg, lo + i) * TILES_PER_SLOT;
                noc.async_read(scratch, CoreLocalMem<uint32_t>(s_ptr + i * f32_tile), f32_tile, {.page_id = page0}, {});
                noc.async_read(
                    scratch,
                    CoreLocalMem<uint32_t>(s_ptr + (NSLOTS + i) * f32_tile),
                    f32_tile,
                    {.page_id = page0 + 1},
                    {});
            }
            noc.async_read_barrier();
            cb_s.push_back(2 * NSLOTS);
            for (uint32_t t0 = 0; t0 < CT; t0 += NB) {
                for (uint32_t i0 = 0; i0 < n; i0 += UB) {
                    const uint32_t i1 = (i0 + UB < n) ? i0 + UB : n;
                    cb_o.reserve_back(UB * NB);
                    uint32_t o_ptr = cb_o.get_write_ptr();
                    for (uint32_t i = i0; i < i1; ++i) {
                        const uint32_t page = slot_of(bg, lo + i) * TILES_PER_SLOT + 2 + blk * CT + t0;
                        for (uint32_t t = 0; t < NB; ++t) {
                            noc.async_read(scratch, CoreLocalMem<uint32_t>(o_ptr), f32_tile, {.page_id = page + t}, {});
                            o_ptr += f32_tile;
                        }
                    }
                    noc.async_read_barrier();
                    cb_o.push_back(UB * NB);
                }
            }
        });
    } else {
        CircularBuffer cb_m(CB_IN_M);
        CircularBuffer cb_l(CB_IN_L);
        for_each_combine_job(core, pos, [&](uint32_t /*b*/, uint32_t /*g*/, uint32_t bg, uint32_t blk, const Row& r) {
            const uint32_t lo = unit_lo(r);
            const uint32_t hi = unit_hi(r);
            for (uint32_t u = lo; u <= hi; ++u) {
                const uint32_t page0 = slot_of(bg, u) * TILES_PER_SLOT;
                cb_m.reserve_back(1);
                noc.async_read(scratch, CoreLocalMem<uint32_t>(cb_m.get_write_ptr()), f32_tile, {.page_id = page0}, {});
                cb_l.reserve_back(1);
                noc.async_read(
                    scratch, CoreLocalMem<uint32_t>(cb_l.get_write_ptr()), f32_tile, {.page_id = page0 + 1}, {});
                cb_o.reserve_back(CT);
                uint32_t o_ptr = cb_o.get_write_ptr();
                for (uint32_t c = 0; c < CT; ++c) {
                    noc.async_read(
                        scratch, CoreLocalMem<uint32_t>(o_ptr), f32_tile, {.page_id = page0 + 2 + blk * CT + c}, {});
                    o_ptr += f32_tile;
                }
                noc.async_read_barrier();
                cb_m.push_back(1);
                cb_l.push_back(1);
                cb_o.push_back(CT);
            }
        });
    }
}
