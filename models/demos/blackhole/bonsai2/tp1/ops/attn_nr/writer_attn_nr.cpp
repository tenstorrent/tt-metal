// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer for attn_nr: row 0 of the head's Dt output tiles becomes row hl of tiles 0..Dt-1 of the [1, 1, H, D] output
// (q: SDPA decode's layout; k: the cache update's), two 16-element bf16 runs per tile. A NoC transfer keeps source and
// destination at the same offset within a 64 B line, and row hl sits at a 32 B multiple, so each run is staged in
// scratch at the destination's offset mod 64 first. Persistent zeroed outputs keep their padding rows zero.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 6, cb_stage = 7;

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr auto qo_a = TensorAccessorArgs<1>();
    constexpr auto ko_a = TensorAccessorArgs<qo_a.next_compile_time_args_offset()>();
    // runtime: 0 q_out, 1 k_out, 2 role (0 q, 1 k), 3 head index within its role
    const uint32_t role = get_arg_val<uint32_t>(2);
    const uint32_t hl = get_arg_val<uint32_t>(3);
    const uint32_t tb = get_tile_size(cb_out);  // bf16 2048

    Noc noc;
    CircularBuffer co(cb_out), st(cb_stage);
    co.wait_front(Dt);
    const CoreLocalMem<volatile uint32_t> src(co.get_read_ptr());
    const CoreLocalMem<volatile uint32_t> stg(st.get_write_ptr());
    const uint32_t f = (hl < 16) ? 0u : 2u;
    const uint32_t dst_off = (f * 256 + (hl % 16) * 16) * 2;  // row hl, face f (bytes); face f+1 is +512
    const uint32_t mis = dst_off % 64;
    // Stage every run: slot (t, k) is 128 B, the run sits at +mis so source and destination agree mod 64.
    for (uint32_t t = 0; t < Dt; t++) {
        for (uint32_t k = 0; k < 2; k++) {
            const uint32_t s_w = (t * tb + k * 512) / 4;          // row 0 of face k of tile t
            const uint32_t d_w = ((t * 2 + k) * 128 + mis) / 4;
            for (uint32_t x = 0; x < 8; x++) {
                stg[d_w + x] = src[s_w + x];
            }
        }
    }
    auto scatter = [&](const auto& dst) {
        for (uint32_t t = 0; t < Dt; t++) {
            for (uint32_t k = 0; k < 2; k++) {
                noc.async_write(st, dst, 32, {.offset_bytes = (t * 2 + k) * 128 + mis},
                                {.page_id = t, .offset_bytes = dst_off + k * 512});
            }
        }
        noc.async_write_barrier();
    };
    if (role == 0) {
        scatter(TensorAccessor(qo_a, get_arg_val<uint32_t>(0), tb));
    } else {
        scatter(TensorAccessor(ko_a, get_arg_val<uint32_t>(1), tb));
    }
    co.pop_front(Dt);
}
