// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// EP-4 receive-side lists (ep4_layer.build_lists), BRISC (NOC0: the DRAM reads): the stream-position table and this
// core's range of the group's topk rows (grp, 128 B a token) into L1, then semaphore 4 + 1 on this core (e4_lists.cpp
// waits for it). grp: 4-KB pages of 32 tokens.
// CT: 0 tok_buf (L1), 1 pos table (L1), 2 EPC, then TensorAccessorArgs(grp), (pos).
// RT: 0 first group token, 1 tokens, 2 grp addr, 3 pos addr.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tok_buf = get_compile_time_arg_val(0);
    constexpr uint32_t pos_l1 = get_compile_time_arg_val(1);
    constexpr uint32_t EPC = get_compile_time_arg_val(2);
    constexpr auto gargs = TensorAccessorArgs<3>();
    constexpr auto pargs = TensorAccessorArgs<gargs.next_compile_time_args_offset()>();
    const uint32_t a = get_arg_val<uint32_t>(0), nt = get_arg_val<uint32_t>(1);
    const auto gs = TensorAccessor(gargs, get_arg_val<uint32_t>(2), 4096);  // (32 tokens a page)
    const auto ps = TensorAccessor(pargs, get_arg_val<uint32_t>(3), EPC * 4);
    noc_async_read(ps.get_noc_addr(0), pos_l1, EPC * 4);
    for (uint32_t i = 0; i < nt; ++i) {
        noc_async_read(gs.get_noc_addr((a + i) / 32, (a + i) % 32 * 128), tok_buf + i * 128, 128);
    }
    noc_async_read_barrier();
    noc_semaphore_inc(get_noc_addr(get_semaphore(4)), 1);
    noc_async_atomic_barrier();
}
