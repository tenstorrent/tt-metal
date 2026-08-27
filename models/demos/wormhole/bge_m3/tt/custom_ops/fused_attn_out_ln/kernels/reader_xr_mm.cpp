// SPDX-License-Identifier: Apache-2.0
// FUSED reader (integration step 1, single M_block): streams A [M_t,K] and this
// core's W N-slice [K,Ns] in K_block chunks for the matmul; reads residual/gamma/
// beta N-slice; then performs the two proven §13 all-gather rounds of the
// compute-produced partials across the P cores of this column.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
constexpr uint32_t cb_resid = tt::CBIndex::c_4;
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
    const uint32_t w_addr = get_arg_val<uint32_t>(a++);
    const uint32_t r_addr = get_arg_val<uint32_t>(a++);
    const uint32_t g_addr = get_arg_val<uint32_t>(a++);
    const uint32_t b_addr = get_arg_val<uint32_t>(a++);
    const uint32_t scaler_packed = get_arg_val<uint32_t>(a++);
    const uint32_t scaler_g_packed = get_arg_val<uint32_t>(a++);
    const uint32_t eps_packed = get_arg_val<uint32_t>(a++);
    const uint32_t n_start = get_arg_val<uint32_t>(a++);
    const uint32_t my_slot = get_arg_val<uint32_t>(a++);

    constexpr uint32_t K_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t Ns = get_compile_time_arg_val(2);
    constexpr uint32_t M_t = get_compile_time_arg_val(3);
    constexpr uint32_t K_block = get_compile_time_arg_val(4);
    constexpr uint32_t tb = get_compile_time_arg_val(5);
    constexpr uint32_t P = get_compile_time_arg_val(6);
    constexpr uint32_t sem0_id = get_compile_time_arg_val(7);
    constexpr uint32_t sem1_id = get_compile_time_arg_val(8);
    constexpr uint32_t K_num_blocks = K_tiles / K_block;
    constexpr uint32_t obn = M_t * Ns;

    uint32_t px[16], py[16];
    for (uint32_t j = 0; j < P; j++) {
        px[j] = get_arg_val<uint32_t>(a++);
        py[j] = get_arg_val<uint32_t>(a++);
    }

    constexpr auto a_args = TensorAccessorArgs<9>();
    const auto a_acc = TensorAccessor(a_args, a_addr, tb);
    constexpr auto w_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto w_acc = TensorAccessor(w_args, w_addr, tb);
    constexpr auto r_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto r_acc = TensorAccessor(r_args, r_addr, tb);
    constexpr auto g_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
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

    // gamma/beta slice
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
    noc_async_read_barrier();
    cb_push_back(cb_gamma, Ns);
    cb_push_back(cb_beta, Ns);

    // matmul inputs: stream K in K_block chunks
    for (uint32_t kb = 0; kb < K_num_blocks; kb++) {
        uint32_t k0 = kb * K_block;
        cb_reserve_back(cb_in0, M_t * K_block);
        uint32_t aw = get_write_ptr(cb_in0);
        for (uint32_t m = 0; m < M_t; m++) {
            for (uint32_t k = 0; k < K_block; k++) {
                noc_async_read_page(m * K_tiles + (k0 + k), a_acc, aw);
                aw += tb;
            }
        }
        cb_reserve_back(cb_in1, K_block * Ns);
        uint32_t ww = get_write_ptr(cb_in1);
        for (uint32_t k = 0; k < K_block; k++) {
            for (uint32_t n = 0; n < Ns; n++) {
                noc_async_read_page((k0 + k) * N_tiles + n_start + n, w_acc, ww);
                ww += tb;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0, M_t * K_block);
        cb_push_back(cb_in1, K_block * Ns);
    }

    // residual slice [M_t, Ns]
    cb_reserve_back(cb_resid, obn);
    uint32_t rw = get_write_ptr(cb_resid);
    for (uint32_t m = 0; m < M_t; m++) {
        for (uint32_t n = 0; n < Ns; n++) {
            noc_async_read_page(m * N_tiles + n_start + n, r_acc, rw);
            rw += tb;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_resid, obn);

    // ---- gather round 1 (mean) ----
    {
        cb_wait_front(cb_ex_partial, M_t);
        uint32_t lp = get_read_ptr(cb_ex_partial);
        cb_reserve_back(cb_ex_external, P * M_t);
        uint32_t le = get_write_ptr(cb_ex_external);
        uint32_t slot_off = my_slot * M_t * tb;
        for (uint32_t j = 0; j < P; j++) {
            noc_async_write(lp, get_noc_addr(px[j], py[j], le + slot_off), M_t * tb);
        }
        noc_async_write_barrier();
        for (uint32_t j = 0; j < P; j++) {
            noc_semaphore_inc(get_noc_addr(px[j], py[j], sem0), 1);
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
            noc_async_write(lp, get_noc_addr(px[j], py[j], le + slot_off), M_t * tb);
        }
        noc_async_write_barrier();
        for (uint32_t j = 0; j < P; j++) {
            noc_semaphore_inc(get_noc_addr(px[j], py[j], sem1), 1);
        }
        noc_semaphore_wait(sem1_ptr, P);
        cb_push_back(cb_ex_external2, P * M_t);
        cb_pop_front(cb_ex_partial2, M_t);
    }
}
