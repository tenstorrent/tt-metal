// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// One-token decode embedding on one core (decode.py embed): row ids[0] of the bf16 row-major table into row 0 of a
// [.., 32, D] bf16 TILE L1 output whose rows 1-31 stay zero.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t D = get_compile_time_arg_val(0);
    constexpr auto i_args = TensorAccessorArgs<1>();
    constexpr auto t_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    const uint32_t ids_addr = get_arg_val<uint32_t>(0);
    const uint32_t ids_page = get_arg_val<uint32_t>(1);
    const auto ids = TensorAccessor(i_args, ids_addr, ids_page);
    const auto table = TensorAccessor(t_args, get_arg_val<uint32_t>(2), D * 2);
    const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(3), 2048);
    const uint32_t buf = get_write_ptr(0);  // [64 B id | D * 2 B row]
    const uint32_t row = buf + 64;
    noc_async_read(ids.get_noc_addr(0), buf, 64);
#ifdef POS_INFO
    // the step's position and the page table, for the decode SDPA's K / V prefetch (sdpa_dec): into the same L1 record
    // ([0] position, [16 ..] page table) on each of the n decode cores (x << 16 | y) in the same round trip
    constexpr auto p_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    const uint32_t pt_bytes = get_arg_val<uint32_t>(6), rec = get_arg_val<uint32_t>(7), n = get_arg_val<uint32_t>(8);
    const uint32_t info = row + D * 2;  // [64 B position | page table]
    noc_async_read(TensorAccessor(p_args, get_arg_val<uint32_t>(4), 4).get_noc_addr(0), info, 64);
    noc_async_read(TensorAccessor(pt_args, get_arg_val<uint32_t>(5), pt_bytes).get_noc_addr(0), info + 64, pt_bytes);
#endif
    noc_async_read_barrier();
    const uint32_t id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf)[0];
    noc_async_read(table.get_noc_addr(id), row, D * 2);
#ifdef POS_INFO
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t v = get_arg_val<uint32_t>(9 + i);
        noc_async_write(info, get_noc_addr(v >> 16, v & 0xffff, rec), 64 + pt_bytes);
    }
#endif
    noc_async_read_barrier();
    // the output is a persistent L1 tensor zeroed once (decode.py): only row 0 of each tile is written
    for (uint32_t j = 0; j < D / 32; ++j) {
        const uint64_t t = out.get_noc_addr(j);
        noc_async_write(row + j * 64, t, 32);             // columns 0-15: face 0 row 0
        noc_async_write(row + j * 64 + 32, t + 512, 32);  // columns 16-31: face 1 row 0
    }
    noc_async_write_barrier();
}
