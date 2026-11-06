// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Prefill per-head RMSNorm + partial RoPE (qk_rope.py), writer (BRISC): per item the compute's tiles 2 .. Dt - 1, then
// 0 and 1, to the head-major output.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "../../ternary/kernels/core_span.hpp"

void kernel_main() {
    // common args: out, items, ncores, grid_x, H, Mt, Dt
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    uint32_t first, n_items;
    core_span(
        get_common_arg_val<uint32_t>(1),
        get_common_arg_val<uint32_t>(2),
        get_common_arg_val<uint32_t>(3),
        1,
        first,
        n_items);
    const uint32_t H = get_common_arg_val<uint32_t>(4), Mt = get_common_arg_val<uint32_t>(5);
    const uint32_t Dt = get_common_arg_val<uint32_t>(6);
    constexpr auto o_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_out = 16;
    const uint32_t TB = get_tile_size(cb_out);
    const auto os = TensorAccessor(o_args, o_addr, TB);
#ifdef TOKEN_MAJOR
    // Decode (T = 1): row 0 of head h's tiles goes to row h of a token-major [32, Dt * 32] L1 shard on one core (the
    // paged cache update's input, qk_rope.py kv_out), and head h of V (head-major, row 0 of its tiles) likewise into
    // a second one. Rows are 32 + 32 B (face 0 / face 1, or faces 2 / 3 for h >= 16).
    const uint32_t sh_x = get_common_arg_val<uint32_t>(7), sh_y = get_common_arg_val<uint32_t>(8);
    const uint32_t k_sh = get_common_arg_val<uint32_t>(9), v_sh = get_common_arg_val<uint32_t>(10);
    constexpr auto v_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto vs = TensorAccessor(v_args, get_common_arg_val<uint32_t>(11), TB);
    constexpr uint32_t cb_v = 17;
    cb_reserve_back(cb_v, 1);
    const uint32_t scratch = get_write_ptr(cb_v);
    for (uint32_t it = first; it < first + n_items; ++it) {
        const uint32_t h = it % H;
        const uint32_t ro = (h < 16 ? 0 : 1024) + (h % 16) * 32;
        for (uint32_t k = 0; k < Dt; ++k) {
            const uint32_t j = k < Dt - 2 ? k + 2 : k - (Dt - 2);
            cb_wait_front(cb_out, 1);
            const uint64_t d = get_noc_addr(sh_x, sh_y, k_sh + j * TB + ro);
            noc_async_write(get_read_ptr(cb_out), d, 32);
            noc_async_write(get_read_ptr(cb_out) + 512, d + 512, 32);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
        for (uint32_t j = 0; j < Dt; ++j) {
            const uint64_t s = vs.get_noc_addr(h * Dt + j);
            noc_async_read(s, scratch, 64);  // face 0 rows 0-1 (64-B DRAM reads)
            noc_async_read(s + 512, scratch + 64, 64);
            noc_async_read_barrier();
            const uint64_t d = get_noc_addr(sh_x, sh_y, v_sh + j * TB + ro);
            noc_async_write(scratch, d, 32);
            noc_async_write(scratch + 64, d + 512, 32);
            noc_async_writes_flushed();
        }
    }
    (void)os;
    (void)Mt;
#else
    for (uint32_t it = first; it < first + n_items; ++it) {
        const uint32_t r = it / H, h = it % H, base = (h * Mt + r) * Dt;
        for (uint32_t k = 0; k < Dt; ++k) {
            const uint32_t j = k < Dt - 2 ? k + 2 : k - (Dt - 2);
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), os.get_noc_addr(base + j), TB);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
#endif
    noc_async_write_barrier();
}
