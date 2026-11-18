// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B49 iHC prefill: generic tile writer (BRISC). Block 1: NJ1 groups of NS1 tiles popped from cb1, tile (j, s) -> page
// base1 + s * stride1 + j of tensor A. Block 2 (optional, cb2 != 0xFFFF): N2 tiles from cb2 -> pages base2 + t of
// tensor B. CT: 0 cb1, 1 page1 (bytes), 2 NJ1, 3 NS1, 4 stride1, 5 WB (groups per wait, divides NJ1), 6 cb2, 7 page2, 8
// N2, accessors A (9..), B. RT: base1, base2. Common: addrA, addrB.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb1 = get_compile_time_arg_val(0);
    constexpr uint32_t page1 = get_compile_time_arg_val(1);
    constexpr uint32_t NJ1 = get_compile_time_arg_val(2);
    constexpr uint32_t NS1 = get_compile_time_arg_val(3);
    constexpr uint32_t stride1 = get_compile_time_arg_val(4);
    constexpr uint32_t WB = get_compile_time_arg_val(5);
    constexpr uint32_t cb2 = get_compile_time_arg_val(6);
    constexpr uint32_t page2 = get_compile_time_arg_val(7);
    constexpr uint32_t N2 = get_compile_time_arg_val(8);
    constexpr auto a_args = TensorAccessorArgs<9>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const uint32_t base1 = get_arg_val<uint32_t>(0);
    const uint32_t base2 = get_arg_val<uint32_t>(1);
    const auto a_acc = TensorAccessor(a_args, get_common_arg_val<uint32_t>(0), page1);
    Noc noc;
    experimental::CB c1(cb1);
    for (uint32_t j0 = 0; j0 < NJ1; j0 += WB) {
        c1.wait_front(WB * NS1);
        for (uint32_t jj = 0; jj < WB; ++jj) {
            for (uint32_t s = 0; s < NS1; ++s) {
                noc.async_write(
                    c1,
                    a_acc,
                    page1,
                    {.offset_bytes = (jj * NS1 + s) * page1},
                    {.page_id = base1 + s * stride1 + j0 + jj, .offset_bytes = 0});
            }
        }
        noc.async_write_barrier();
        c1.pop_front(WB * NS1);
    }
    if constexpr (cb2 != 0xFFFFu) {
        const auto b_acc = TensorAccessor(b_args, get_common_arg_val<uint32_t>(1), page2);
        experimental::CB c2(cb2);
        c2.wait_front(N2);
        for (uint32_t t = 0; t < N2; ++t) {
            noc.async_write(c2, b_acc, page2, {.offset_bytes = t * page2}, {.page_id = base2 + t, .offset_bytes = 0});
        }
        noc.async_write_barrier();
        c2.pop_front(N2);
    }
}
