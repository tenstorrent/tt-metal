// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused decode router, reader. Every core: the 8 x tiles of its K chunk and the 8 matching fp32 tiles of its
// router weight column (output tile (chunk, n) of the K-split router GEMM). The collector also reads the dense
// bias page, and hands the 9 partial-sum pages to compute once all cores have written their row into them.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t kt = get_named_compile_time_arg_val("kt");  // K tiles per chunk
    constexpr uint32_t nt = get_named_compile_time_arg_val("nt");  // N tiles (experts / 32)
    constexpr uint32_t chunks = get_named_compile_time_arg_val("chunks");
    constexpr uint32_t sem_id = get_named_compile_time_arg_val("sem");
    constexpr uint32_t cb_bias = 1, cb_x = 6, cb_w = 7, cb_parts = 9;

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t chunk = get_arg_val<uint32_t>(2);
    const uint32_t n = get_arg_val<uint32_t>(3);
    const uint32_t collector = get_arg_val<uint32_t>(4);
    const uint32_t bias_addr = get_arg_val<uint32_t>(5);
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto xa = TensorAccessor(x_args, x_addr, 2048);
    const auto wa = TensorAccessor(w_args, w_addr, 4096);

    cb_reserve_back(cb_x, kt);
    cb_reserve_back(cb_w, kt);
    const uint32_t xd = get_write_ptr(cb_x), wd = get_write_ptr(cb_w);
    for (uint32_t k = 0; k < kt; ++k) {
        noc_async_read(xa.get_noc_addr(chunk * kt + k), xd + k * 2048, 2048);
        noc_async_read(wa.get_noc_addr((chunk * kt + k) * nt + n), wd + k * 4096, 4096);
    }
    if (collector) {
        cb_reserve_back(cb_bias, 1);
        const InterleavedAddrGen<true> bias = {.bank_base_address = bias_addr, .page_size = 4096};
        noc_async_read(bias.get_noc_addr(0), get_write_ptr(cb_bias), 4096);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, kt);
    cb_push_back(cb_w, kt);
    if (!collector) {
        return;
    }
    cb_push_back(cb_bias, 1);
    cb_reserve_back(cb_parts, chunks);  // every writer targets this CB's base address on the collector
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    noc_semaphore_wait(sem, chunks * nt);
    cb_push_back(cb_parts, chunks);
}
