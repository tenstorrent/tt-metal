// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// x52 streamed decode matmul, activation shard core (one per L1 width shard of the gathered hidden): the matmul reads
// ONE row of the activation tile (a row of the product depends on its own row only), so the core compacts its tiles'
// row 0 (face 0 row 0 at byte 0, face 1 row 0 at byte 512: 2 x 32 B) into 64-byte slots = a 1 x 32 tile each, and
// writes the chunk into the hub's compact buffer, then raises the hub's semaphore.  Runtime args: act address (this
// core's shard), hub NoC x y, the shard's index.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "../../kernels/zones.h"

void kernel_main() {
    constexpr uint32_t cb_stage = get_named_compile_time_arg_val("cb_stage");
    constexpr uint32_t shard_tiles = get_named_compile_time_arg_val("shard_tiles");
    const uint32_t act_addr = get_arg_val<uint32_t>(0);
    const uint32_t hub_x = get_arg_val<uint32_t>(1);
    const uint32_t hub_y = get_arg_val<uint32_t>(2);
    const uint32_t index =
        get_arg_val<uint32_t>(3);  // this shard's chunk in the hub's compact buffer (a CB at one address on both cores)
    const uint32_t hub_sem = get_semaphore(get_named_compile_time_arg_val("sem_hub"));
    CircularBuffer stage(cb_stage);
    FUSED_ZONE("fz_mm_shard");
    const uint32_t chunk = stage.get_write_ptr() + index * shard_tiles * 64;
    volatile tt_l1_ptr uint32_t* to = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(chunk);
    for (uint32_t t = 0; t < shard_tiles; ++t) {
        volatile tt_l1_ptr uint32_t* from = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_addr + t * 2048);
#pragma GCC unroll 8
        for (uint32_t w = 0; w < 8; ++w) {
            to[t * 16 + w] = from[w];
            to[t * 16 + 8 + w] = from[128 + w];
        }
    }
    noc_async_write(chunk, get_noc_addr(hub_x, hub_y, chunk), shard_tiles * 64);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(hub_x, hub_y, hub_sem), 1);
    noc_async_atomic_barrier();
}
