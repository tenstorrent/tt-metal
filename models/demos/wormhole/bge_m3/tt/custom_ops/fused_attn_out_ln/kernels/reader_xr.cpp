// SPDX-License-Identifier: Apache-2.0
// Cross-core LN reduce PROBE reader. Reads this core's N-slice of x + gamma/beta
// slice, fills scalers/eps, then performs TWO all-gather rounds of the compute-
// produced partials across the P cores in the column. Each round: write local
// partial into slot my_slot of every peer's external CB, signal that round's
// semaphore on every peer, wait until it reaches P (each peer increments it
// exactly once -> equality wait is race-safe), then release the gathered block
// to compute. Single M_block (probe): no CB reuse across iterations.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

constexpr uint32_t cb_in = tt::CBIndex::c_0;
constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
constexpr uint32_t cb_beta = tt::CBIndex::c_6;
constexpr uint32_t cb_scaler = tt::CBIndex::c_7;
constexpr uint32_t cb_eps = tt::CBIndex::c_8;
constexpr uint32_t cb_ex_partial = tt::CBIndex::c_9;
constexpr uint32_t cb_ex_external = tt::CBIndex::c_10;
constexpr uint32_t cb_ex_partial2 = tt::CBIndex::c_13;
constexpr uint32_t cb_ex_external2 = tt::CBIndex::c_14;
constexpr uint32_t cb_scaler_g = tt::CBIndex::c_17;

FORCE_INLINE void fill_tile(uint32_t cb_id, uint32_t packed) {
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 512; i++) {
        p[i] = packed;
    }
    cb_push_back(cb_id, 1);
}

void kernel_main() {
    uint32_t a = 0;
    const uint32_t a_addr = get_arg_val<uint32_t>(a++);
    const uint32_t g_addr = get_arg_val<uint32_t>(a++);
    const uint32_t b_addr = get_arg_val<uint32_t>(a++);
    const uint32_t scaler_packed = get_arg_val<uint32_t>(a++);
    const uint32_t scaler_g_packed = get_arg_val<uint32_t>(a++);
    const uint32_t eps_packed = get_arg_val<uint32_t>(a++);
    const uint32_t n_start = get_arg_val<uint32_t>(a++);
    const uint32_t my_slot = get_arg_val<uint32_t>(a++);

    constexpr uint32_t N_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t Ns = get_compile_time_arg_val(1);
    constexpr uint32_t M_t = get_compile_time_arg_val(2);
    constexpr uint32_t tb = get_compile_time_arg_val(3);
    constexpr uint32_t P = get_compile_time_arg_val(4);
    constexpr uint32_t sem0_id = get_compile_time_arg_val(5);
    constexpr uint32_t sem1_id = get_compile_time_arg_val(6);
    constexpr uint32_t obn = M_t * Ns;

    // peer NoC coords: P pairs (x,y) as runtime args
    uint32_t px[16], py[16];
    for (uint32_t j = 0; j < P; j++) {
        px[j] = get_arg_val<uint32_t>(a++);
        py[j] = get_arg_val<uint32_t>(a++);
    }

    constexpr auto a_args = TensorAccessorArgs<7>();
    const auto a_acc = TensorAccessor(a_args, a_addr, tb);
    constexpr auto g_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto g_acc = TensorAccessor(g_args, g_addr, tb);
    constexpr auto b_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto b_acc = TensorAccessor(b_args, b_addr, tb);

    const uint32_t sem0 = get_semaphore(sem0_id);
    const uint32_t sem1 = get_semaphore(sem1_id);
    volatile tt_l1_ptr uint32_t* sem0_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem0);
    volatile tt_l1_ptr uint32_t* sem1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem1);

    wh_generate_reduce_scaler(cb_scaler, scaler_packed);
    wh_generate_reduce_scaler(cb_scaler_g, scaler_g_packed);
    fill_tile(cb_eps, eps_packed);

    // gamma/beta slice: Ns tiles at [n_start, n_start+Ns)
    cb_reserve_back(cb_gamma, Ns);
    uint32_t gw = get_write_ptr(cb_gamma);
    for (uint32_t n = 0; n < Ns; n++) {
        noc_async_read_page(n_start + n, g_acc, gw);
        gw += tb;
    }
    cb_reserve_back(cb_beta, Ns);
    uint32_t bw = get_write_ptr(cb_beta);
    for (uint32_t n = 0; n < Ns; n++) {
        noc_async_read_page(n_start + n, b_acc, bw);
        bw += tb;
    }
    // x slice: tiles (m, n_start+ns) = m*N_tiles + n_start + ns
    cb_reserve_back(cb_in, obn);
    uint32_t xw = get_write_ptr(cb_in);
    for (uint32_t m = 0; m < M_t; m++) {
        for (uint32_t n = 0; n < Ns; n++) {
            noc_async_read_page(m * N_tiles + n_start + n, a_acc, xw);
            xw += tb;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_gamma, Ns);
    cb_push_back(cb_beta, Ns);
    cb_push_back(cb_in, obn);

    // ---- gather round 1 (mean) ----
    {
        cb_wait_front(cb_ex_partial, M_t);
        uint32_t lp = get_read_ptr(cb_ex_partial);
        cb_reserve_back(cb_ex_external, P * M_t);
        uint32_t le = get_write_ptr(cb_ex_external);
        uint32_t slot_off = my_slot * M_t * tb;
        for (uint32_t j = 0; j < P; j++) {
            uint64_t dst = get_noc_addr(px[j], py[j], le + slot_off);
            noc_async_write(lp, dst, M_t * tb);
        }
        noc_async_write_barrier();
        for (uint32_t j = 0; j < P; j++) {
            uint64_t sd = get_noc_addr(px[j], py[j], sem0);
            noc_semaphore_inc(sd, 1);
        }
        noc_semaphore_wait(sem0_ptr, P);
        cb_push_back(cb_ex_external, P * M_t);
        cb_pop_front(cb_ex_partial, M_t);
    }

    // ---- gather round 2 (var) ----
    {
        cb_wait_front(cb_ex_partial2, M_t);
        uint32_t lp = get_read_ptr(cb_ex_partial2);
        cb_reserve_back(cb_ex_external2, P * M_t);
        uint32_t le = get_write_ptr(cb_ex_external2);
        uint32_t slot_off = my_slot * M_t * tb;
        for (uint32_t j = 0; j < P; j++) {
            uint64_t dst = get_noc_addr(px[j], py[j], le + slot_off);
            noc_async_write(lp, dst, M_t * tb);
        }
        noc_async_write_barrier();
        for (uint32_t j = 0; j < P; j++) {
            uint64_t sd = get_noc_addr(px[j], py[j], sem1);
            noc_semaphore_inc(sd, 1);
        }
        noc_semaphore_wait(sem1_ptr, P);
        cb_push_back(cb_ex_external2, P * M_t);
        cb_pop_front(cb_ex_partial2, M_t);
    }
}
