// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GDN decode step, writer (fused_step/gdn_step.py): the head's gated output tiles, its updated recurrent state (in
// place), and the conv shift register of the channels this core owns: st0 <- st1 <- st2 <- st3 <- x (the new input),
// tile by tile in that order, exactly what the composite decode's four ttnn.copy calls leave behind. The q/k channels
// of a shared key head are owned by one core; it first waits for the partner reader's semaphore increment.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_state = 26, cb_out = 28, cb_wscr = 30;

void kernel_main() {
    uint32_t i = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(i++);
    const uint32_t rec_addr = get_arg_val<uint32_t>(i++);
    const uint32_t st0_addr = get_arg_val<uint32_t>(i++);
    const uint32_t st1_addr = get_arg_val<uint32_t>(i++);
    const uint32_t st2_addr = get_arg_val<uint32_t>(i++);
    const uint32_t st3_addr = get_arg_val<uint32_t>(i++);
    const uint32_t proj_addr = get_arg_val<uint32_t>(i++);
    const uint32_t h = get_arg_val<uint32_t>(i++);
    const uint32_t kh = get_arg_val<uint32_t>(i++);
    const uint32_t own_qk = get_arg_val<uint32_t>(i++);  // 1: this core shifts the q/k history of key head kh
    const uint32_t wait_n = get_arg_val<uint32_t>(i++);  // partner reads to wait for before that

    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t k_off = get_compile_time_arg_val(2);
    constexpr uint32_t v_off = get_compile_time_arg_val(3);
    constexpr uint32_t sem_id = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();
    constexpr auto rec_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto st0_args = TensorAccessorArgs<rec_args.next_compile_time_args_offset()>();
    constexpr auto st1_args = TensorAccessorArgs<st0_args.next_compile_time_args_offset()>();
    constexpr auto st2_args = TensorAccessorArgs<st1_args.next_compile_time_args_offset()>();
    constexpr auto st3_args = TensorAccessorArgs<st2_args.next_compile_time_args_offset()>();
    constexpr auto proj_args = TensorAccessorArgs<st3_args.next_compile_time_args_offset()>();

    constexpr uint32_t BF = 2048, F32 = 4096;
    const auto out = TensorAccessor(out_args, out_addr, BF);
    const auto rec = TensorAccessor(rec_args, rec_addr, F32);
    const auto st0 = TensorAccessor(st0_args, st0_addr, BF);
    const auto st1 = TensorAccessor(st1_args, st1_addr, BF);
    const auto st2 = TensorAccessor(st2_args, st2_addr, BF);
    const auto st3 = TensorAccessor(st3_args, st3_addr, BF);
    const auto proj = TensorAccessor(proj_args, proj_addr, BF);

    // updated recurrent state of head h
    cb_wait_front(cb_state, Kt * Vt);
    {
        uint32_t src = get_read_ptr(cb_state);
        for (uint32_t t = 0; t < Kt * Vt; t++, src += F32) {
            noc_async_write_page(h * Kt * Vt + t, rec, src);
        }
    }
    // gated output
    cb_wait_front(cb_out, Vt);
    {
        uint32_t src = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < Vt; t++, src += BF) {
            noc_async_write_page(h * Vt + t, out, src);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_state, Kt * Vt);
    cb_pop_front(cb_out, Vt);

    // conv shift register of the owned channels (this core's reader finished long before cb_out arrived)
    if (own_qk && wait_n) {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
        noc_semaphore_wait_min(sem, wait_n);
        noc_semaphore_set(sem, 0);
    }
    cb_reserve_back(cb_wscr, 1);
    const uint32_t scr = get_write_ptr(cb_wscr);
    auto shift = [&](uint32_t c) {
        noc_async_read_page(c, st1, scr);
        noc_async_read_barrier();
        noc_async_write_page(c, st0, scr);
        noc_async_write_barrier();
        noc_async_read_page(c, st2, scr);
        noc_async_read_barrier();
        noc_async_write_page(c, st1, scr);
        noc_async_write_barrier();
        noc_async_read_page(c, st3, scr);
        noc_async_read_barrier();
        noc_async_write_page(c, st2, scr);
        noc_async_write_barrier();
        noc_async_read_page(c, proj, scr);
        noc_async_read_barrier();
        noc_async_write_page(c, st3, scr);
        noc_async_write_barrier();
    };
    if (own_qk) {
        for (uint32_t t = 0; t < Kt; t++) {
            shift(kh * Kt + t);
            shift(k_off + kh * Kt + t);
        }
    }
    for (uint32_t t = 0; t < Vt; t++) {
        shift(v_off + h * Vt + t);
    }
}
