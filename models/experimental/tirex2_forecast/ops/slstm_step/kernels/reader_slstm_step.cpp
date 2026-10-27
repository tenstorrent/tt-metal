// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for the fused sLSTM loop. Work item i = one d-tile of unit i / 4 (tile row unit / 4, head unit % 4,
// d-tile j = i % 4). Once: each item's R block (k-tile x gate) = 4 x 4 bf16 tiles, item-local order k * 4 + g.
// Every step and item: its 4 Wx tiles (gates i f z o). Before the first item of each touched unit at step
// t > 0 it waits for that unit's semaphore to reach 4 t (all 4 y tiles of step t - 1 are in the local y
// buffer, own or from partner cores) and hands compute one token on the ready CB.
//
// Wx [T, 1, B, 2048] page (t * Bt + row) * 64 + g * 16 + h * 4 + j; R [512, 2048] page (h * 4 + k) * 64 +
// g * 16 + h * 4 + j.
//
// Compile-time args: TensorAccessorArgs(wx), TensorAccessorArgs(r)
// Runtime args: wx_addr, r_addr, num_items, first_item, T, record, Bt, ablate, num_touched_units

#include "api/debug/kernel_profiler.hpp"
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t wx_addr = get_arg_val<uint32_t>(0);
    const uint32_t r_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_items = get_arg_val<uint32_t>(2);
    const uint32_t first_item = get_arg_val<uint32_t>(3);
    const uint32_t T = get_arg_val<uint32_t>(4);
    const uint32_t Bt = get_arg_val<uint32_t>(6);
    const bool skip_wx = (get_arg_val<uint32_t>(7) & 4) != 0;  // timing ablation: no Wx DRAM reads
    const uint32_t ablate = get_arg_val<uint32_t>(7);
    const uint32_t L = get_arg_val<uint32_t>(8);

    constexpr auto wx_args = TensorAccessorArgs<0>();
    constexpr auto r_args = TensorAccessorArgs<wx_args.next_compile_time_args_offset()>();
    constexpr uint32_t wx_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t r_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t y_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t ready_cb_id = tt::CBIndex::c_7;

    Noc noc;
    CircularBuffer cb_wx(wx_cb_id);
    CircularBuffer cb_r(r_cb_id);
    CircularBuffer cb_y(y_cb_id);
    CircularBuffer cb_ready(ready_cb_id);
    const uint32_t wx_bytes = get_tile_size(wx_cb_id);
    const uint32_t r_bytes = get_tile_size(r_cb_id);
    const auto wx = TensorAccessor(wx_args, wx_addr);
    const auto r = TensorAccessor(r_args, r_addr);
    const uint32_t u0 = first_item / 4;

    // The y buffer (L units x 2 parities x 4 tiles) is fixed memory: hand all of it to compute once.
    cb_y.reserve_back(8 * L);
    cb_y.push_back(8 * L);

    cb_r.reserve_back(16 * num_items);
    for (uint32_t e = 0; e < num_items; ++e) {
        const uint32_t i = first_item + e;
        const uint32_t h = (i / 4) % 4;
        const uint32_t j = i % 4;
        for (uint32_t k = 0; k < 4; ++k) {
            for (uint32_t g = 0; g < 4; ++g) {
                const uint32_t page = (h * 4 + k) * 64 + g * 16 + h * 4 + j;
                noc.async_read(r, cb_r, r_bytes, {.page_id = page}, {.offset_bytes = (e * 16 + k * 4 + g) * r_bytes});
            }
        }
    }
    noc.async_read_barrier();
    cb_r.push_back(16 * num_items);

    // Wx of a whole step is read and pushed before that step's semaphore waits (the Wx CB holds 2 steps), so the
    // DRAM reads overlap the previous step's compute and the y exchange instead of following it.
    for (uint32_t t = 0; t < T; ++t) {
        // Per item reserve/push: cores with fewer items would otherwise wrap a 4 * num_items block in the CB.
        for (uint32_t e = 0; e < num_items; ++e) {
            const uint32_t i = first_item + e;
            const uint32_t unit = i / 4;
            const uint32_t base = (t * Bt + unit / 4) * 64 + (unit % 4) * 4 + i % 4;
            cb_wx.reserve_back(4);
            for (uint32_t g = 0; g < 4 && !skip_wx; ++g) {
                noc.async_read(wx, cb_wx, wx_bytes, {.page_id = base + g * 16}, {.offset_bytes = g * wx_bytes});
            }
            noc.async_read_barrier();
            cb_wx.push_back(4);
        }
        for (uint32_t e = 0; e < num_items && t > 0; ++e) {
            const uint32_t i = first_item + e;
            if (e == 0 || i % 4 == 0) {
                auto sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(i / 4 - u0));
                if (!(ablate & 128)) {
                    if (t >= 100 && t < 104) {
                        DeviceZoneScopedN("R-PEER");
                        noc_semaphore_wait_min(sem, 4 * t);
                    } else {
                        noc_semaphore_wait_min(sem, 4 * t);
                    }
                }
                cb_ready.reserve_back(1);
                cb_ready.push_back(1);
            }
        }
    }
}
