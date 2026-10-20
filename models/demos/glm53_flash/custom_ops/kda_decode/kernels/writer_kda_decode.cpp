// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer of the fused KDA decode block: per work item, the updated recurrent state of every user of the part
// (in place); then the part's output block goes to the slot's part-0 core (gather CB + semaphore), where the
// part-0 writer stores the head's normed output tiles of the tile row (bf16 TILE output).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 10, cb_state_out = 11, cb_gather = 24, cb_ofin = 31;
constexpr uint32_t SEM_GATHER = 3;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t Bt = get_compile_time_arg_val(2);
    constexpr uint32_t B = get_compile_time_arg_val(3);
    constexpr uint32_t H = get_compile_time_arg_val(4);
    constexpr uint32_t parts = get_compile_time_arg_val(5);
    constexpr uint32_t HV_t = get_compile_time_arg_val(6);

    constexpr auto o_a = TensorAccessorArgs<7>();
    constexpr auto st_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();

    const uint32_t item0 = get_arg_val<uint32_t>(0);
    const uint32_t n_items = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t o_addr = get_arg_val<uint32_t>(3);
    const uint32_t st_addr = get_arg_val<uint32_t>(4);
    const uint32_t p0_x = get_arg_val<uint32_t>(5);
    const uint32_t p0_y = get_arg_val<uint32_t>(6);

    const uint32_t t32 = get_tile_size(cb_state_out);
    const uint32_t t16 = get_tile_size(cb_out);
    const auto o_acc = TensorAccessor(o_a, o_addr, t16);
    const auto st_acc = TensorAccessor(st_a, st_addr, t32);

    Noc noc;
    CircularBuffer out(cb_out), st(cb_state_out), ofin(cb_ofin), gather(cb_gather);
    for (uint32_t it = 0; it < n_items; it++) {
        const uint32_t item = item0 + it * stride;
        const uint32_t h = item / (Bt * parts);
        const uint32_t rem = item % (Bt * parts);
        const uint32_t tb = rem / parts;
        const uint32_t part = rem % parts;
        const uint32_t n_row = (B - tb * 32u < 32u) ? (B - tb * 32u) : 32u;
        const uint32_t up = (n_row + parts - 1) / parts;
        const uint32_t u0 = part * up;
        const uint32_t u1 = (u0 + up < n_row) ? (u0 + up) : n_row;
        if (u0 >= u1) {
            continue;
        }
        for (uint32_t r = u0; r < u1; r++) {
            st.wait_front(Kt * Vt);
            const uint32_t base = ((tb * 32 + r) * H + h) * Kt * Vt;
            for (uint32_t t = 0; t < Kt * Vt; t++) {
                noc.async_write(st, st_acc, t32, {.offset_bytes = t * t32}, {.page_id = base + t});
            }
            noc.async_write_barrier();
            st.pop_front(Kt * Vt);
        }
        if (part != 0) {
            // this part's (masked) output block -> slot part-1 of the part-0 core's gather CB
            ofin.wait_front(Vt);
            const uint64_t dst = get_noc_addr(p0_x, p0_y, gather.get_write_ptr() + (part - 1) * Vt * t32);
            noc_async_write(ofin.get_read_ptr(), dst, Vt * t32);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(p0_x, p0_y, get_semaphore(SEM_GATHER)), 1);
            ofin.pop_front(Vt);
        } else {
            out.wait_front(Vt);
            for (uint32_t n = 0; n < Vt; n++) {
                noc.async_write(out, o_acc, t16, {.offset_bytes = n * t16}, {.page_id = tb * HV_t + h * Vt + n});
            }
            noc.async_write_barrier();
            out.pop_front(Vt);
        }
    }
}
