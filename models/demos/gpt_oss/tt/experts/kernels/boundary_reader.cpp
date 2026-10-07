// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode layer boundary, receiver (NCRISC) (tt/decode_boundary.py: DecodeBoundary).
//
// Feeds the boundary compute: the RMSNorm weight (`tiles` BF16 pages read from DRAM) and the reduce scaler (both
// once per call), the residual stream (cb_res) and, once every device's partial sum has arrived (the receive
// semaphore reaches `expected`: the local copy + the remote packets), the `slots` x `tiles` receive buffer (cb_recv).
// The semaphore is reset for the next use of this boundary before the slots are published. expected = 0: cb_recv
// holds an already all-reduced sum (no wait).
//
// residual source (res_mode):
//   0: cb_res is backed by the flat residual tensor (published as is);
//   1: entry: the residual is the [1, 1, 1, hidden] BF16 row-major embedding (one page), copied into cb_res and
//      zero padded to `tiles` pages.
//
// runtime args: [gamma_addr, sem_addr, res_src_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t gamma_addr = get_arg_val<uint32_t>(0);
    const uint32_t sem_addr = get_arg_val<uint32_t>(1);
    const uint32_t res_src_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_res = get_compile_time_arg_val(0);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t tiles = get_compile_time_arg_val(4);
    constexpr uint32_t slots = get_compile_time_arg_val(5);
    constexpr uint32_t expected = get_compile_time_arg_val(6);
    constexpr uint32_t res_mode = get_compile_time_arg_val(7);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(8);  // hidden * 2
    constexpr uint32_t do_norm = get_compile_time_arg_val(9);
    constexpr auto gamma_args = TensorAccessorArgs<10>();
    constexpr auto res_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    constexpr uint32_t page = 2048;

    if constexpr (do_norm) {
        const auto s_gamma = TensorAccessor(gamma_args, gamma_addr, tiles * page);
        cb_reserve_back(cb_gamma, tiles);
        noc_async_read(s_gamma.get_noc_addr(0), get_write_ptr(cb_gamma), tiles * page);
    }

    if constexpr (res_mode == 1) {
        const auto s_res = TensorAccessor(res_args, res_src_addr, row_bytes);
        cb_reserve_back(cb_res, tiles);
        const uint32_t dst = get_write_ptr(cb_res);
        noc_async_read(s_res.get_noc_addr(0), dst, row_bytes);
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t z = row_bytes; z < tiles * page; z += MEM_ZEROS_SIZE) {
            const uint32_t n = tiles * page - z < MEM_ZEROS_SIZE ? tiles * page - z : MEM_ZEROS_SIZE;
            noc_async_read(zeros, dst + z, n);
        }
    } else {
        cb_reserve_back(cb_res, tiles);
    }

    if constexpr (do_norm) {
        // Reduce scaler: 1.0 in the first row of each face of a BF16 tile (scalar sum of all 1024 values).
        cb_reserve_back(cb_scaler, 1);
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
        for (uint32_t i = 0; i < page / 4; ++i) {
            s[i] = 0;
        }
        for (uint32_t f = 0; f < 4; ++f) {
            for (uint32_t i = 0; i < 8; ++i) {
                s[f * 128 + i] = 0x3F803F80;
            }
        }
        cb_push_back(cb_scaler, 1);
    }

    noc_async_read_barrier();
    cb_push_back(cb_res, tiles);
    if constexpr (do_norm) {
        cb_push_back(cb_gamma, tiles);
    }

    if constexpr (slots > 0) {
        if constexpr (expected > 0) {
            DeviceZoneScopedN("BND_R_WAIT");
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
            noc_semaphore_wait_min(sem, expected);
            noc_semaphore_set(sem, 0);
        }
        cb_reserve_back(cb_recv, slots * tiles);
        cb_push_back(cb_recv, slots * tiles);
    }
}
