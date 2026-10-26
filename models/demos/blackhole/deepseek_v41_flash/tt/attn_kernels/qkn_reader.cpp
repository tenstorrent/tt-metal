// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// qk_norm (bit-exact to ttnn.rms_norm), reader (NCRISC).  Cores form groups (q group, kv group); core `kl` of a group
// owns the column tiles [j0, j0 + gpc) of y = [wq_a | wkv] output.  Only the sum of squares has to run in the reference
// order (one reduce_tile chain over all tiles of the row, on the reducer = core 0 of the group): every core squares its
// own tiles, the other cores write their bf16 square tiles into the reducer's SQALL buffer (data, barrier, semaphore 0
// increment); the reducer reduces all tiles in order, computes rs and writes the rs tile to the other cores (data,
// barrier, semaphore 1 increment); every core then finishes its own tiles.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(2);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(3);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sqall = get_compile_time_arg_val(5);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(6);
    constexpr uint32_t MX0 = get_compile_time_arg_val(7);
    constexpr uint32_t MY0 = get_compile_time_arg_val(8);
    constexpr uint32_t MW = get_compile_time_arg_val(9);
    constexpr auto x_args = TensorAccessorArgs<10>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    const uint32_t is_red = get_arg_val<uint32_t>(0);
    const uint32_t gb = get_arg_val<uint32_t>(1);  // global index of the reducer (first core of my group)
    const uint32_t ng = get_arg_val<uint32_t>(2);
    const uint32_t kl = get_arg_val<uint32_t>(3);
    const uint32_t j0 = get_arg_val<uint32_t>(4);
    const uint32_t gpc = get_arg_val<uint32_t>(5);
    const uint32_t eps_bits = get_arg_val<uint32_t>(6);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, 2048);
    const auto w_acc = TensorAccessor(w_args, w_addr, 2048);
    experimental::CB cbx(cb_x), cbw(cb_w), cbsc(cb_sc), cbeps(cb_eps), cbsq(cb_sq), cbsqall(cb_sqall), cbrs(cb_rs);

    volatile tt_l1_ptr uint32_t* sem_sq = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* sem_rs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    const uint32_t sem_sq_addr = get_semaphore(0), sem_rs_addr = get_semaphore(1);

    cbx.reserve_back(gpc);
    cbw.reserve_back(gpc);
    for (uint32_t g = 0; g < gpc; ++g) {
        noc.async_read(x_acc, cbx, 2048, {.page_id = j0 + g, .offset_bytes = 0}, {.offset_bytes = g * 2048});
    }
    for (uint32_t g = 0; g < gpc; ++g) {
        noc.async_read(w_acc, cbw, 2048, {.page_id = j0 + g, .offset_bytes = 0}, {.offset_bytes = g * 2048});
    }

    // scaler tile: 1.0 in row 0 of every face (bf16), zeros elsewhere (only the reducer reduces; harmless elsewhere)
    cbsc.reserve_back(1);
    {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbsc.get_write_ptr());
        for (uint32_t i = 0; i < 512; ++i) {
            p[i] = 0;
        }
        for (uint32_t f = 0; f < 4; ++f) {
            for (uint32_t c = 0; c < 8; ++c) {
                p[f * 128 + c] = 0x3F803F80u;
            }
        }
    }
    cbsc.push_back(1);

    // eps tile (generate_bcast_col_scalar): bf16 eps in column 0 of faces 0 and 2
    cbeps.reserve_back(1);
    {
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbeps.get_write_ptr());
        for (uint32_t i = 0; i < 1024; ++i) {
            p[i] = 0;
        }
        const uint16_t e = eps_bits >> 16;
        for (uint32_t k = 0; k < 4; k += 2) {
            const uint32_t idx = k << 8;
            for (uint32_t j = 0; j < 256; j += 16) {
                p[idx + j] = e;
            }
        }
    }
    cbeps.push_back(1);

    noc.async_read_barrier();
    cbx.push_back(gpc);
    cbw.push_back(gpc);

    const uint32_t wt = ng * gpc;
    cbsqall.reserve_back(wt);
    const uint32_t sqall_base = cbsqall.get_write_ptr();
    cbrs.reserve_back(1);
    const uint32_t rs_base = cbrs.get_write_ptr();

    cbsq.wait_front(gpc);
    const uint32_t sq_ptr = cbsq.get_read_ptr();
    const uint32_t red_x = MX0 + (gb % MW), red_y = MY0 + gb / MW;
    if (!is_red) {
        noc_async_write(sq_ptr, get_noc_addr(red_x, red_y, sqall_base + kl * gpc * 2048), gpc * 2048);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(red_x, red_y, sem_sq_addr), 1);
        noc_async_atomic_barrier();
        noc_semaphore_wait(sem_rs, 1);
        *sem_rs = 0;
        cbrs.push_back(1);
    } else {
        noc_async_write(sq_ptr, get_noc_addr(red_x, red_y, sqall_base), gpc * 2048);
        noc_async_write_barrier();
        noc_semaphore_wait(sem_sq, ng - 1);
        *sem_sq = 0;
        cbsqall.push_back(wt);
        cbrs.wait_front(1);
        const uint32_t rs_ptr = cbrs.get_read_ptr();
        for (uint32_t kk = 1; kk < ng; ++kk) {
            const uint32_t gi = gb + kk;
            const uint32_t dx = MX0 + (gi % MW), dy = MY0 + gi / MW;
            noc_async_write(rs_ptr, get_noc_addr(dx, dy, rs_base), 2048);
        }
        noc_async_write_barrier();
        for (uint32_t kk = 1; kk < ng; ++kk) {
            const uint32_t gi = gb + kk;
            noc_semaphore_inc(get_noc_addr(MX0 + (gi % MW), MY0 + gi / MW, sem_rs_addr), 1);
        }
        noc_async_atomic_barrier();
    }
}
