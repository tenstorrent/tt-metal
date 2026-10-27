// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GDN decode step, reader (fused_step/gdn_step.py). Value head h on this core, key head kh = h / (Nv / Nk).
// Pushes, in the order the compute kernel consumes them:
//   cb_h   [st1 | st2 | st3] x this head's NC conv channel tiles (q of kh, k of kh, v of h)   bf16
//   cb_x   the in-projection's NC channel tiles                                              bf16
//   cb_tap [tap0 | tap1 | tap2 | tap3] x NC                                                  bf16
//   cb_z   Vt tiles of z                                                                     bf16
//   cb_one, cb_mcol                                                                          fp32 constants
//   cb_a, cb_b  fp32 tiles holding a_h / b_h at [0, 0] (zero elsewhere), from the in-projection's a|b tile
//   cb_na, cb_dt  fp32 constant tiles (-exp(A_log)[h], dt_bias[h] at [0, 0])
//   cb_w   Vt fp32 tiles of the gated-norm weight
//   cb_S   Kt*Vt fp32 tiles of the recurrent state of head h
// A core whose key head is shared reads the q/k conv history that the partner's writer then shifts: after its reads
// it increments the partner's semaphore (gdn_step_writer waits for it before writing).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_h = 0, cb_x = 1, cb_tap = 2, cb_z = 3, cb_a = 4, cb_b = 5, cb_na = 6, cb_dt = 7, cb_w = 8;
constexpr uint32_t cb_mcol = 9, cb_one = 10, cb_S = 11, cb_abraw = 29;

void kernel_main() {
    uint32_t i = 0;
    const uint32_t proj_addr = get_arg_val<uint32_t>(i++);
    const uint32_t st1_addr = get_arg_val<uint32_t>(i++);
    const uint32_t st2_addr = get_arg_val<uint32_t>(i++);
    const uint32_t st3_addr = get_arg_val<uint32_t>(i++);
    const uint32_t tap0_addr = get_arg_val<uint32_t>(i++);
    const uint32_t tap1_addr = get_arg_val<uint32_t>(i++);
    const uint32_t tap2_addr = get_arg_val<uint32_t>(i++);
    const uint32_t tap3_addr = get_arg_val<uint32_t>(i++);
    const uint32_t gconst_addr =
        get_arg_val<uint32_t>(i++);                          // fp32 [.., 2 Nv tiles]: (negA_h, dtb_h) at tiles 2h, 2h+1
    const uint32_t normw_addr = get_arg_val<uint32_t>(i++);  // fp32 Vt tiles
    const uint32_t cmask_addr = get_arg_val<uint32_t>(i++);  // fp32 2 tiles: [ones, mcol]
    const uint32_t rec_addr = get_arg_val<uint32_t>(i++);    // fp32 [Nv, Kt, Vt] tiles
    const uint32_t h = get_arg_val<uint32_t>(i++);
    const uint32_t kh = get_arg_val<uint32_t>(i++);
    const uint32_t signal = get_arg_val<uint32_t>(i++);  // 1: increment the partner's semaphore after reading
    const uint32_t partner_x = get_arg_val<uint32_t>(i++);
    const uint32_t partner_y = get_arg_val<uint32_t>(i++);

    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t k_off = get_compile_time_arg_val(3);    // tile offset of k in the qkv block
    constexpr uint32_t v_off = get_compile_time_arg_val(4);    // tile offset of v
    constexpr uint32_t z_off = get_compile_time_arg_val(5);    // tile offset of z in the projection
    constexpr uint32_t ab_tile = get_compile_time_arg_val(6);  // projection tile holding a | b
    constexpr uint32_t a_col = get_compile_time_arg_val(7);    // column of a_0 in that tile (b_0 = a_col + Nv)
    constexpr uint32_t Nv = get_compile_time_arg_val(8);
    constexpr uint32_t sem_id = get_compile_time_arg_val(9);
    constexpr auto proj_args = TensorAccessorArgs<10>();
    constexpr auto st1_args = TensorAccessorArgs<proj_args.next_compile_time_args_offset()>();
    constexpr auto st2_args = TensorAccessorArgs<st1_args.next_compile_time_args_offset()>();
    constexpr auto st3_args = TensorAccessorArgs<st2_args.next_compile_time_args_offset()>();
    constexpr auto tap0_args = TensorAccessorArgs<st3_args.next_compile_time_args_offset()>();
    constexpr auto tap1_args = TensorAccessorArgs<tap0_args.next_compile_time_args_offset()>();
    constexpr auto tap2_args = TensorAccessorArgs<tap1_args.next_compile_time_args_offset()>();
    constexpr auto tap3_args = TensorAccessorArgs<tap2_args.next_compile_time_args_offset()>();
    constexpr auto gconst_args = TensorAccessorArgs<tap3_args.next_compile_time_args_offset()>();
    constexpr auto normw_args = TensorAccessorArgs<gconst_args.next_compile_time_args_offset()>();
    constexpr auto cmask_args = TensorAccessorArgs<normw_args.next_compile_time_args_offset()>();
    constexpr auto rec_args = TensorAccessorArgs<cmask_args.next_compile_time_args_offset()>();

    constexpr uint32_t BF = 2048, F32 = 4096;
    const auto proj = TensorAccessor(proj_args, proj_addr, BF);
    const auto st1 = TensorAccessor(st1_args, st1_addr, BF);
    const auto st2 = TensorAccessor(st2_args, st2_addr, BF);
    const auto st3 = TensorAccessor(st3_args, st3_addr, BF);
    const auto tap0 = TensorAccessor(tap0_args, tap0_addr, BF);
    const auto tap1 = TensorAccessor(tap1_args, tap1_addr, BF);
    const auto tap2 = TensorAccessor(tap2_args, tap2_addr, BF);
    const auto tap3 = TensorAccessor(tap3_args, tap3_addr, BF);
    const auto gconst = TensorAccessor(gconst_args, gconst_addr, F32);
    const auto normw = TensorAccessor(normw_args, normw_addr, F32);
    const auto cmask = TensorAccessor(cmask_args, cmask_addr, F32);
    const auto rec = TensorAccessor(rec_args, rec_addr, F32);

    // the head's conv channel tiles inside the qkv block: q(kh), k(kh), v(h)
    uint32_t ch[NC];
    for (uint32_t t = 0; t < Kt; t++) {
        ch[t] = kh * Kt + t;
        ch[Kt + t] = k_off + kh * Kt + t;
    }
    for (uint32_t t = 0; t < Vt; t++) {
        ch[2 * Kt + t] = v_off + h * Vt + t;
    }

    // conv history (st1, st2, st3)
    cb_reserve_back(cb_h, 3 * NC);
    {
        uint32_t dst = get_write_ptr(cb_h);
        for (uint32_t t = 0; t < NC; t++, dst += BF) {
            noc_async_read_page(ch[t], st1, dst);
        }
        for (uint32_t t = 0; t < NC; t++, dst += BF) {
            noc_async_read_page(ch[t], st2, dst);
        }
        for (uint32_t t = 0; t < NC; t++, dst += BF) {
            noc_async_read_page(ch[t], st3, dst);
        }
    }
    // the new conv input (the in-projection's qkv columns: same tile numbering as the history)
    cb_reserve_back(cb_x, NC);
    {
        uint32_t dst = get_write_ptr(cb_x);
        for (uint32_t t = 0; t < NC; t++, dst += BF) {
            noc_async_read_page(ch[t], proj, dst);
        }
    }
    cb_reserve_back(cb_tap, 4 * NC);
    {
        uint32_t dst = get_write_ptr(cb_tap);
        for (uint32_t t = 0; t < NC; t++, dst += BF) {
            noc_async_read_page(ch[t], tap0, dst);
        }
        for (uint32_t t = 0; t < NC; t++, dst += BF) {
            noc_async_read_page(ch[t], tap1, dst);
        }
        for (uint32_t t = 0; t < NC; t++, dst += BF) {
            noc_async_read_page(ch[t], tap2, dst);
        }
        for (uint32_t t = 0; t < NC; t++, dst += BF) {
            noc_async_read_page(ch[t], tap3, dst);
        }
    }
    noc_async_read_barrier();
    if (signal) {
        // every read of the shared q/k history is done: the partner's writer may shift it
        noc_semaphore_inc(get_noc_addr(partner_x, partner_y, get_semaphore(sem_id)), 1);
    }
    cb_push_back(cb_h, 3 * NC);
    cb_push_back(cb_x, NC);
    cb_push_back(cb_tap, 4 * NC);

    cb_reserve_back(cb_z, Vt);
    {
        uint32_t dst = get_write_ptr(cb_z);
        for (uint32_t t = 0; t < Vt; t++, dst += BF) {
            noc_async_read_page(z_off + h * Vt + t, proj, dst);
        }
    }
    cb_reserve_back(cb_one, 1);
    cb_reserve_back(cb_mcol, 1);
    noc_async_read_page(0, cmask, get_write_ptr(cb_one));
    noc_async_read_page(1, cmask, get_write_ptr(cb_mcol));
    noc_async_read_barrier();
    cb_push_back(cb_z, Vt);
    cb_push_back(cb_one, 1);
    cb_push_back(cb_mcol, 1);

    // gates: a_h, b_h out of the a|b projection tile (bf16, face 0 row 0) into zeroed fp32 tiles at [0, 0]
    cb_reserve_back(cb_abraw, 1);
    noc_async_read_page(ab_tile, proj, get_write_ptr(cb_abraw));
    cb_reserve_back(cb_a, 1);
    cb_reserve_back(cb_b, 1);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t which = 0; which < 2; which++) {
        const uint32_t dst = get_write_ptr(which == 0 ? cb_a : cb_b);
        for (uint32_t z = 0; z < F32; z += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, dst + z, (F32 - z) < MEM_ZEROS_SIZE ? (F32 - z) : MEM_ZEROS_SIZE);
        }
    }
    cb_reserve_back(cb_na, 1);
    cb_reserve_back(cb_dt, 1);
    noc_async_read_page(2 * h, gconst, get_write_ptr(cb_na));
    noc_async_read_page(2 * h + 1, gconst, get_write_ptr(cb_dt));
    noc_async_read_barrier();
    {
        volatile tt_l1_ptr uint16_t* raw = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_abraw));
        volatile tt_l1_ptr uint32_t* pa = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_a));
        volatile tt_l1_ptr uint32_t* pb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_b));
        pa[0] = static_cast<uint32_t>(raw[a_col + h]) << 16;
        pb[0] = static_cast<uint32_t>(raw[a_col + Nv + h]) << 16;
    }
    cb_push_back(cb_a, 1);
    cb_push_back(cb_b, 1);
    cb_push_back(cb_na, 1);
    cb_push_back(cb_dt, 1);

    cb_reserve_back(cb_w, Vt);
    {
        uint32_t dst = get_write_ptr(cb_w);
        for (uint32_t t = 0; t < Vt; t++, dst += F32) {
            noc_async_read_page(t, normw, dst);
        }
    }
    cb_reserve_back(cb_S, Kt * Vt);
    {
        uint32_t dst = get_write_ptr(cb_S);
        for (uint32_t t = 0; t < Kt * Vt; t++, dst += F32) {
            noc_async_read_page(h * Kt * Vt + t, rec, dst);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_w, Vt);
    cb_push_back(cb_S, Kt * Vt);
}
