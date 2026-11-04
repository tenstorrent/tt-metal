// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// otherlayers: writer of the key-rows program (index key append of the owner layers). One core per 32-column tile
// column c of the fp4-simulated key block k [1,1,R,128] (R = U * n rows, row u * n + j = block index j of user u). For
// every block index j and user u the key row (u * n + j) of this tile goes to row 0 of tile column c of the update tile
// of (j, u): n destination tensors [1,U,1,128] bf16 TILE, L1 height-sharded (one 32 x 128 shard = 4 tiles per user
// core), exactly what ``ttnn.to_memory_config(to_layout(slice(k_rm, j)), kcfg)`` of ``append_key_block`` produced
// (rows 1..31 zero). common args: [addr_0 .. addr_{n-1}, then (noc x, noc y) of the shard core of user u for u =
// 0..U-1].
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ol_zones.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(1);
    constexpr uint32_t U = get_compile_time_arg_val(2);
    constexpr uint32_t N = get_compile_time_arg_val(3);
    const uint32_t c = ol_core_idx(get_common_arg_val<uint32_t>(N + 2 * U));

    OLZ("Z_WALL");
    // zero tile (local scratch)
    cb_reserve_back(cb_zero, 1);
    const uint32_t zp = get_write_ptr(cb_zero);
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zp);
    for (uint32_t i = 0; i < 512; ++i) {
        z[i] = 0;
    }
    cb_wait_front(cb_out, 1);
    const uint32_t op = get_read_ptr(cb_out);
    for (uint32_t j = 0; j < N; ++j) {
        const uint32_t addr = get_common_arg_val<uint32_t>(j) + c * 2048;
        for (uint32_t u = 0; u < U; ++u) {
            const uint32_t nx = get_common_arg_val<uint32_t>(N + 2 * u);
            const uint32_t ny = get_common_arg_val<uint32_t>(N + 2 * u + 1);
            // zero everything except the two 32 B row-0 pieces (non-overlapping writes: no ordering assumption between
            // them)
            noc_async_write(zp, get_noc_addr(nx, ny, addr + 32), 480);
            noc_async_write(zp, get_noc_addr(nx, ny, addr + 544), 480);
            noc_async_write(zp, get_noc_addr(nx, ny, addr + 1024), 1024);
            const uint32_t rho = u * N + j;
            const uint32_t src = op + ((rho >> 4) << 10) + ((rho & 15) << 5);
            noc_async_write(src, get_noc_addr(nx, ny, addr), 32);
            noc_async_write(src + 512, get_noc_addr(nx, ny, addr + 512), 32);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
