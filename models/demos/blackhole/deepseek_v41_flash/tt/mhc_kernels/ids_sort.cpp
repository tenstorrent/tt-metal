// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Ascending sort of the indexer's selected entry ids: one core per row, a 3-pass LSD radix sort (8-bit digits over the low 24 bits) of K uint32 keys
// (0xFFFFFFFF sentinels end up last). ids / out: uint32 ROW_MAJOR [R,1,1,K] pages of K * 4 bytes.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_c = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
    constexpr auto in_args = TensorAccessorArgs<4>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    const uint32_t t_first = get_arg_val<uint32_t>(0);
    const uint32_t t_stride = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);

    Noc noc;
    const auto in_acc = TensorAccessor(in_args, in_addr, K * 4);
    const auto out_acc = TensorAccessor(out_args, out_addr, K * 4);
    experimental::CB cba(cb_a), cbb(cb_b), cbc(cb_c);
    cba.reserve_back(1);
    cbb.reserve_back(1);
    cbc.reserve_back(1);
    volatile tt_l1_ptr uint32_t* pa = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cba.get_write_ptr());
    volatile tt_l1_ptr uint32_t* pb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbc.get_write_ptr());

    for (uint32_t it = 0; it < count; ++it) {
        const uint32_t t = t_first + it * t_stride;
        noc.async_read(in_acc, cba, K * 4, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        volatile tt_l1_ptr uint32_t* src = pa;
        volatile tt_l1_ptr uint32_t* dst = pb;
        for (uint32_t pass = 0; pass < 3; ++pass) {
            const uint32_t sh = pass * 8;
            for (uint32_t d = 0; d < 256; ++d) {
                cnt[d] = 0;
            }
            for (uint32_t i = 0; i < K; ++i) {
                cnt[(src[i] >> sh) & 255u]++;
            }
            uint32_t run = 0;
            for (uint32_t d = 0; d < 256; ++d) {
                const uint32_t c = cnt[d];
                cnt[d] = run;
                run += c;
            }
            for (uint32_t i = 0; i < K; ++i) {
                const uint32_t v = src[i];
                dst[cnt[(v >> sh) & 255u]++] = v;
            }
            volatile tt_l1_ptr uint32_t* tmp = src;
            src = dst;
            dst = tmp;
        }
        // three passes (the low 24 bits: ids < 2^19, and the sentinel 0xFFFFFFFF has all-ones low bits, so it ends up last): the sorted keys are in pb
        noc.async_write(cbb, out_acc, K * 4, {.offset_bytes = 0}, {.page_id = t, .offset_bytes = 0});
        noc.async_write_barrier();
    }
}
