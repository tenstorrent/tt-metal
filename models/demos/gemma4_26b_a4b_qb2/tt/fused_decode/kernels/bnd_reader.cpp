// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Gemma decode layer boundary, reader (NCRISC) (fused_decode/boundary.py: GemmaBoundary).
//
// 1. Gathers row 0 of the `parts` producer outputs (BF16 TILE tensors [1, 1, 32, hidden], e.g. the o_proj or the
//    shared-MLP / routed-expert partial sums) into the flat partial buffer (cb_part, value h at byte 2 h of part p's
//    `tiles` pages) and hands it to the fabric sender (bnd_sender.cpp). entry: gathers the embedding row into the
//    residual (cb_res) instead.
// 2. Reads the flat norm weights (cb_g: `gammas` x `tiles` BF16 pages, one DRAM page), the FP32 vector (cb_vf, the
//    router input scale) and builds the reduce scaler.
// 3. Waits until every device's flat partial has arrived in the receive slots (the receive semaphore reaches
//    `expected`), resets the semaphore and publishes the slots (cb_recv) to the compute.
//
// runtime args: [part0_addr, part1_addr, gamma_addr, vf_addr, sem_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t part0_addr = get_arg_val<uint32_t>(0);
    const uint32_t part1_addr = get_arg_val<uint32_t>(1);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(2);
    const uint32_t vf_addr = get_arg_val<uint32_t>(3);
    const uint32_t sem_addr = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cb_res = get_compile_time_arg_val(1);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(2);
    constexpr uint32_t cb_g = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(4);
    constexpr uint32_t cb_vf = get_compile_time_arg_val(5);
    constexpr uint32_t tiles = get_compile_time_arg_val(6);  // flat pages per vector
    constexpr uint32_t parts = get_compile_time_arg_val(7);  // producer tensors gathered
    constexpr uint32_t slots = get_compile_time_arg_val(8);  // devices (0: no all-reduce)
    constexpr uint32_t expected = get_compile_time_arg_val(9);
    constexpr uint32_t entry = get_compile_time_arg_val(10);   // gather into the residual instead
    constexpr uint32_t htiles = get_compile_time_arg_val(11);  // hidden / 32
    constexpr uint32_t gammas = get_compile_time_arg_val(12);
    constexpr uint32_t has_vf = get_compile_time_arg_val(13);
    constexpr auto p0_args = TensorAccessorArgs<14>();
    constexpr auto p1_args = TensorAccessorArgs<p0_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<p1_args.next_compile_time_args_offset()>();
    constexpr auto vf_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

    constexpr uint32_t page = 2048;  // BF16 32x32 tile
    constexpr uint32_t face = 512;   // BF16 16x16 face

    // Row 0 of tile j: face 0 row 0 (values 32 j .. 32 j + 15) at byte 0, face 1 row 0 at byte 512.
    auto gather = [&](const auto& src, uint32_t dst) {
        for (uint32_t j = 0; j < htiles; ++j) {
            noc_async_read(src.get_noc_addr(j, 0), dst + 64 * j, 32);
            noc_async_read(src.get_noc_addr(j, face), dst + 64 * j + 32, 32);
        }
    };

    if constexpr (parts > 0) {
        const auto s0 = TensorAccessor(p0_args, part0_addr, page);
        const uint32_t cb = entry ? cb_res : cb_part;
        cb_reserve_back(cb, parts * tiles);
        const uint32_t dst = get_write_ptr(cb);
        gather(s0, dst);
        if constexpr (parts > 1) {
            const auto s1 = TensorAccessor(p1_args, part1_addr, page);
            gather(s1, dst + tiles * page);
        }
        noc_async_read_barrier();
        cb_push_back(cb, parts * tiles);
    }
    if constexpr (!entry) {
        cb_reserve_back(cb_res, tiles);
        cb_push_back(cb_res, tiles);
    }

    {
        const auto s_g = TensorAccessor(g_args, gamma_addr, gammas * tiles * page);
        cb_reserve_back(cb_g, gammas * tiles);
        noc_async_read(s_g.get_noc_addr(0), get_write_ptr(cb_g), gammas * tiles * page);
    }
    if constexpr (has_vf) {
        const auto s_vf = TensorAccessor(vf_args, vf_addr, tiles * 2 * page);
        cb_reserve_back(cb_vf, tiles);
        noc_async_read(s_vf.get_noc_addr(0), get_write_ptr(cb_vf), tiles * 2 * page);
    }
    {
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
    cb_push_back(cb_g, gammas * tiles);
    if constexpr (has_vf) {
        cb_push_back(cb_vf, tiles);
    }

    if constexpr (slots > 0) {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
        noc_semaphore_wait_min(sem, expected);
        noc_semaphore_set(sem, 0);
        cb_reserve_back(cb_recv, slots * parts * tiles);
        cb_push_back(cb_recv, slots * parts * tiles);
    }
}
