// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer of the fused KDA decode block -- the FUSED T = K + 1 VARIANT (KdaDecode(fused_steps=T): the speculative
// loop of tt/spec_decode.py under GLM53_SPEC_COMMIT=fused; writer_kda_decode.cpp is the default-path source shared
// by v1 / v2 and must stay byte-identical to the shipped kernel). Per work item (head h, tile row 0, user part):
// in commit mode (DRY = 0) the final recurrent state of every user of the part with a non-zero step count (the
// step page: one more accessor block after the recurrent state's, runtime arg 7 = the address of the [Bt, 32]
// int32 ROW_MAJOR pages, word u = n_u; read per item into the writer's slot (+2048) of the scratch region after
// the Vt weight tiles of cb_onw, a region the CB protocol never reserves) -- one state write per user after its
// n_u steps, in place; an untouched user (n_u = 0) has no state in the CB at all. In dry mode (DRY = 1, the verify)
// no state is written and the step page is not read. Then the part's output block goes to the slot's part-0 core
// (gather CB + semaphore), where the part-0 writer stores the head's normed output tiles of the R-row tile row
// (bf16 TILE output), as in the default writer. USERS / STEPS / DRY: compile-time args 7 / 8 / 9; accessors from 10.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_onw = 6, cb_out = 10, cb_state_out = 11, cb_gather = 24, cb_ofin = 31;
constexpr uint32_t SEM_GATHER = 3;
constexpr uint32_t STEP_BYTES = 32 * 4;        // one step page: 32 int32 per-user step counts (op.py GATE_BYTES)
constexpr uint32_t STEP_WRITER_OFFSET = 2048;  // the writer's landing slot inside the scratch region's first page

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t Bt = get_compile_time_arg_val(2);
    constexpr uint32_t B = get_compile_time_arg_val(3);  // the R = USERS * STEPS candidate rows
    constexpr uint32_t H = get_compile_time_arg_val(4);
    constexpr uint32_t parts = get_compile_time_arg_val(5);
    constexpr uint32_t HV_t = get_compile_time_arg_val(6);
    constexpr uint32_t USERS = get_compile_time_arg_val(7);  // user slots (the state's batch)
    constexpr uint32_t STEPS = get_compile_time_arg_val(8);  // candidates per user: T = K + 1
    constexpr bool DRY = get_compile_time_arg_val(9) != 0;   // verify mode: no state written
    static_assert(
        Bt == 1 && USERS * STEPS == B && B <= 32 && USERS >= 1 && STEPS >= 1, "one tile row of USERS x STEPS rows");

    constexpr auto o_a = TensorAccessorArgs<10>();
    constexpr auto st_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();
    constexpr auto steps_a = TensorAccessorArgs<st_a.next_compile_time_args_offset()>();  // the step pages

    const uint32_t item0 = get_arg_val<uint32_t>(0);
    const uint32_t n_items = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t o_addr = get_arg_val<uint32_t>(3);
    const uint32_t st_addr = get_arg_val<uint32_t>(4);
    const uint32_t p0_x = get_arg_val<uint32_t>(5);
    const uint32_t p0_y = get_arg_val<uint32_t>(6);
    const uint32_t steps_addr = get_arg_val<uint32_t>(7);  // the step pages ([Bt, 32] int32)

    const uint32_t t32 = get_tile_size(cb_state_out);
    const uint32_t t16 = get_tile_size(cb_out);
    const auto o_acc = TensorAccessor(o_a, o_addr, t16);
    const auto st_acc = TensorAccessor(st_a, st_addr, t32);
    const auto steps_acc = TensorAccessor(steps_a, steps_addr, STEP_BYTES);
    // this core's copy of the step page: the writer's slot of the scratch region after the Vt weight tiles of cb_onw
    // (the reader's copy sits at +0 of the same region; this RISC never moves the cb_onw pointers)
    const uint32_t page_l1 = CircularBuffer(cb_onw).get_write_ptr() + Vt * t32 + STEP_WRITER_OFFSET;
    volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_l1);

    Noc noc;
    CircularBuffer out(cb_out), st(cb_state_out), ofin(cb_ofin), gather(cb_gather);
    for (uint32_t it = 0; it < n_items; it++) {
        const uint32_t item = item0 + it * stride;
        const uint32_t h = item / (Bt * parts);
        const uint32_t rem = item % (Bt * parts);
        const uint32_t tb = rem / parts;
        const uint32_t part = rem % parts;
        const uint32_t up = (USERS + parts - 1) / parts;
        const uint32_t u0 = part * up;
        const uint32_t u1 = (u0 + up < USERS) ? (u0 + up) : USERS;
        if (u0 >= u1) {
            continue;
        }
        if constexpr (!DRY) {
            noc_async_read(steps_acc.get_noc_addr(tb), page_l1, STEP_BYTES);  // this tile row's step counts
            noc_async_read_barrier();
            for (uint32_t r = u0; r < u1; r++) {
                if (page[r] == 0u) {
                    continue;  // untouched user: the compute packed no state for it
                }
                st.wait_front(Kt * Vt);
                const uint32_t base = ((tb * 32 + r) * H + h) * Kt * Vt;
                for (uint32_t t = 0; t < Kt * Vt; t++) {
                    noc.async_write(st, st_acc, t32, {.offset_bytes = t * t32}, {.page_id = base + t});
                }
                noc.async_write_barrier();
                st.pop_front(Kt * Vt);
            }
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
    // every NoC transaction this kernel issued (the cross-core semaphore increments above included) must be
    // acknowledged before the kernel returns: an atomic acked after the next kernel's noc_local_state_init
    // leaves that kernel's atomic counters unequal forever (its noc_async_atomic_barrier hangs; seen on the 8x4
    // o_proj matmul in0 receiver after kda_decode, 2026-10-20)
    noc_async_full_barrier();
}
