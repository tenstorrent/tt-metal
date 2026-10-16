// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// BRISC (NOC 1): every pair's h tiny tiles from the row-major activation (page p = pair p's kt [4, 32] tiles), then
// rows 0-3 of this core's output columns (tile column j * banks + bank) into the zeroed [1, 1, 32, hidden] output.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "moe_stream_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_h = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t num_pairs = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t act_page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t groups = get_compile_time_arg_val(5);
    constexpr uint32_t ct = get_compile_time_arg_val(6);
    constexpr uint32_t banks = get_compile_time_arg_val(7);
    constexpr uint32_t tiny_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t tiny_face_bytes = tiny_bytes / 2;  // rows 0-3 of one face
    constexpr uint32_t bf16_tile_bytes = 2048;

    const uint32_t act_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank = get_arg_val<uint32_t>(2);

    const InterleavedAddrGen<false> act_gen = {.bank_base_address = act_addr, .page_size = act_page_bytes};
    cb_reserve_back(cb_h, num_pairs * kt);
    const uint32_t h_l1 = get_write_ptr(cb_h);
    for (uint32_t p = 0; p < num_pairs; p++) {
        noc_async_read(act_gen.get_noc_addr(p, 0), h_l1 + p * act_page_bytes, act_page_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_h, num_pairs * kt);

    const InterleavedAddrGen<true> out_gen = {.bank_base_address = out_addr, .page_size = bf16_tile_bytes};
    for (uint32_t g = 0; g < groups; g++) {
        cb_wait_front(cb_out, (g + 1) * ct);
        const uint32_t base = get_read_ptr(cb_out);
        for (uint32_t c = 0; c < ct; c++) {
            const uint32_t j = g * ct + c;
            const uint32_t page = j * banks + bank;
            const uint32_t src = base + j * tiny_bytes;
            noc_async_write(src, out_gen.get_noc_addr(page, 0), tiny_face_bytes);
            noc_async_write(
                src + tiny_face_bytes, out_gen.get_noc_addr(page, mimo_moe::kBf16FaceBytes), tiny_face_bytes);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, groups * ct);
}
