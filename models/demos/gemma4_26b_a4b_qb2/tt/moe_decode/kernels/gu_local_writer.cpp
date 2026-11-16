// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// moe_decode gate|up, bank-local form (RISCV_0, NOC_1). Roles as in gu_local_reader.cpp:
//   1 bank reader (bank b, reader r): x K slice b (ks 1x32 tiles from tile kofs of row 0 of the activation) ->
//     cb_x; then, per routed slot s, each of its pairs' (g, u) partial tiles (2 x 128 bytes, fp32) into the cb_recv
//     of the reducer that owns unit (s, pair), at (local unit, bank), and one semaphore increment there.
//   2 reducer: per unit (slot, pair) it owns: the slot's routing weight -> MATH thread (mailbox); the h tile ->
//     row (slot * pairs + pair) of the h buffer.
// compile-time args: [cb_x, cb_part, cb_recv, cb_hout, cb_scr, nr, pairs, k, route_page_bytes, sem_id, banks, cb_bias]
//                    + TensorAccessorArgs(x, h, route, bias)
// HAS_BIAS (reducer): per unit, the expert's gate and up bias rows of the pair (bias tensor [E, pairs * 64] bf16
// row-major: per pair [gate 32 | up 32]) -> cb_bias.
// runtime args: [x_addr, h_addr, route_addr, role, bank, reader, ks, kofs, units, bias_addr,
//                role 1: per unit (k * pairs, in (slot, pair) order) its reducer's NOC (x, y) and local index;
//                role 2: per local unit its (slot, pair)]
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t h_addr = get_arg_val<uint32_t>(1);
    const uint32_t route_addr = get_arg_val<uint32_t>(2);
    const uint32_t role = get_arg_val<uint32_t>(3);
    const uint32_t bank = get_arg_val<uint32_t>(4);
    const uint32_t reader = get_arg_val<uint32_t>(5);
    const uint32_t ks = get_arg_val<uint32_t>(6);
    const uint32_t kofs = get_arg_val<uint32_t>(7);
    const uint32_t units = get_arg_val<uint32_t>(8);
    const uint32_t bias_addr = get_arg_val<uint32_t>(9);
    constexpr uint32_t table = 10;
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_part = get_compile_time_arg_val(1);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(2);
    constexpr uint32_t cb_hout = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(4);
    constexpr uint32_t nr = get_compile_time_arg_val(5);
    constexpr uint32_t pairs = get_compile_time_arg_val(6);
    constexpr uint32_t k = get_compile_time_arg_val(7);
    constexpr uint32_t route_page_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t banks = get_compile_time_arg_val(10);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(11);
    constexpr auto x_args = TensorAccessorArgs<12>();
    constexpr auto h_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto r_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048, face_bytes = 512, row_bytes = 32, tiny_bytes = 64, part_bytes = 128;
    constexpr uint32_t dst_group = 4;  // the compute kernel pushes partials in groups of up to 4 tiles (2 pairs)
    if (role == 1) {
        const auto s_x = TensorAccessor(x_args, x_addr, tile_bytes);
        cb_reserve_back(cb_x, ks);
        uint32_t l1 = get_write_ptr(cb_x);
        for (uint32_t t = 0; t < ks; ++t) {
            const uint64_t src = s_x.get_noc_addr(kofs + t);
            noc_async_read(src, l1, row_bytes);
            noc_async_read(src + face_bytes, l1 + row_bytes, row_bytes);
            l1 += tiny_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, ks);
        const uint32_t recv = get_write_ptr(cb_recv);  // same address on every core
        const uint32_t sem = get_semaphore(sem_id);
        for (uint32_t s = 0; s < k; ++s) {
            for (uint32_t n0 = 0; n0 < nr; n0 += dst_group) {
                const uint32_t n1 = n0 + dst_group < nr ? n0 + dst_group : nr;
                cb_wait_front(cb_part, n1 - n0);
                const uint32_t src = get_read_ptr(cb_part);
                for (uint32_t q = n0 / 2; q < n1 / 2; ++q) {
                    const uint32_t u = s * pairs + reader * (nr / 2) + q;
                    const uint32_t x = get_arg_val<uint32_t>(table + 3 * u),
                                   y = get_arg_val<uint32_t>(table + 3 * u + 1);
                    const uint32_t local = get_arg_val<uint32_t>(table + 3 * u + 2);
                    noc_async_write(
                        src + (q - n0 / 2) * 2 * part_bytes,
                        get_noc_addr(x, y, recv + (local * banks + bank) * 2 * part_bytes),
                        2 * part_bytes);
                }
                noc_async_write_barrier();
                for (uint32_t q = n0 / 2; q < n1 / 2; ++q) {
                    const uint32_t u = s * pairs + reader * (nr / 2) + q;
                    noc_semaphore_inc(
                        get_noc_addr(
                            get_arg_val<uint32_t>(table + 3 * u), get_arg_val<uint32_t>(table + 3 * u + 1), sem),
                        1);
                }
                cb_pop_front(cb_part, n1 - n0);
            }
        }
        noc_async_atomic_barrier();
    } else if (role == 2) {
        const auto s_h = TensorAccessor(h_args, h_addr, tiny_bytes);
        const auto s_r = TensorAccessor(r_args, route_addr, route_page_bytes);
        const uint32_t scratch = get_write_ptr(cb_scr);
        noc_async_read(s_r.get_noc_addr(0), scratch, route_page_bytes);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint16_t* weights = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch);
        for (uint32_t u = 0; u < units; ++u) {
            const uint32_t slot = get_arg_val<uint32_t>(table + 2 * u), pair = get_arg_val<uint32_t>(table + 2 * u + 1);
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, static_cast<uint32_t>(weights[slot]) << 16);
#ifdef HAS_BIAS
            {
                union {
                    uint32_t u;
                    float f;
                } id_bits;
                id_bits.u = static_cast<uint32_t>(weights[k + slot]) << 16;
                const auto s_b = TensorAccessor(b_args, bias_addr, pairs * 2 * tiny_bytes);
                cb_reserve_back(cb_bias, 2);
                noc_async_read(
                    s_b.get_noc_addr(static_cast<uint32_t>(id_bits.f)) + pair * 2 * tiny_bytes,
                    get_write_ptr(cb_bias),
                    2 * tiny_bytes);
                noc_async_read_barrier();
                cb_push_back(cb_bias, 2);
            }
#endif
            cb_wait_front(cb_hout, 1);
            noc_async_write(get_read_ptr(cb_hout), s_h.get_noc_addr(slot * pairs + pair), tiny_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_hout, 1);
        }
    }
}
