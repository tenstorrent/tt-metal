// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B35 (HY4_B35_VPUSH=1): the consumer half of a push-gather (hy4_qpush.Push): waits for the per-source flags, then
// assembles the gathered rows exactly like all_gather_async(dim=3) of a [S,1,1,PB/2] bf16 RM tensor:
//   out [S,1,1,8*PB/2] RM, page s = [source 0 row s | source 1 row s | ... | source 7 row s]
// landing tensor (Push): page c*NP + s/RP, byte offset (s % RP) * PB = row s of source c. One core per row s. Flags are
// monotonic, each core keeps a launch counter in a global semaphore (as the other B29 consumers).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t PB = get_compile_time_arg_val(2);
    constexpr uint32_t RP = get_compile_time_arg_val(3);
    constexpr uint32_t NP = get_compile_time_arg_val(4);
    constexpr auto l_args = TensorAccessorArgs<5>();
    constexpr auto o_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    constexpr uint32_t OB = 8 * PB;
    const uint32_t s = get_arg_val<uint32_t>(0);
    const uint32_t fx = get_arg_val<uint32_t>(1), fy = get_arg_val<uint32_t>(2);
    // args 3..10: the 8 flag addresses, 11: launch counter address
    const uint32_t a_l = get_common_arg_val<uint32_t>(0), a_o = get_common_arg_val<uint32_t>(1);
    Noc noc;
    const auto l_acc = TensorAccessor(l_args, a_l, RP * PB);
    const auto o_acc = TensorAccessor(o_args, a_o, OB);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();  // [row OB][flag scratch 8 x 16 B]
    const uint32_t scr = base + OB;
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(11));
    const uint32_t target = *cnt + NP;
    *cnt = target;
    bool done;
    do {
        for (uint32_t c = 0; c < 8; ++c) {
            noc_async_read(get_noc_addr(fx, fy, get_arg_val<uint32_t>(3 + c)), scr + 16 * c, 4);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        done = true;
        for (uint32_t c = 0; c < 8; ++c) {
            done &= ((int32_t)(*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 16 * c) - target) >= 0);
        }
    } while (!done);
    for (uint32_t c = 0; c < 8; ++c) {
        noc.async_read(
            l_acc, cb, PB, {.page_id = c * NP + s / RP, .offset_bytes = (s % RP) * PB}, {.offset_bytes = c * PB});
    }
    noc.async_read_barrier();
    noc.async_write(cb, o_acc, OB, {.offset_bytes = 0}, {.page_id = s, .offset_bytes = 0});
    noc.async_write_barrier();
}
