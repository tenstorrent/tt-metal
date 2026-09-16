// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Hybrid RVV/SFPU dual-stream writer (Architecture A, dedicated-CB streams).
// Identical to writer.cpp except each tile is drained from the CB its stream
// produced into — c_17 (RVV, raw-pushed by TRISC2's pack thread) or c_16
// (SFPU, llk-pushed) — using the shared parity
//   is_rvv(i) = (i % rvv_den) < rvv_num   (i = LOCAL tile index on this core).
// Tile i still lands in DRAM slot start_tile_id + i, so the output DRAM
// layout is bit-identical to the production path and DUMP_OUTPUT_CSV needs no
// changes. The plain dataflow cb_wait_front/cb_pop_front used here are the
// MMIO stream-register versions, which pair correctly with the raw producer.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

#if !defined(ARCH_BLACKHOLE) && !defined(ARCH_WORMHOLE)
#error "generic activation hybrid writer flush pipeline is certified only on Blackhole and Wormhole"
#endif

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_tile_id = get_arg_val<uint32_t>(2);  // Tile offset for multi-core
    uint32_t rvv_num = get_arg_val<uint32_t>(3);
    uint32_t rvv_den = get_arg_val<uint32_t>(4);
    if (rvv_den == 0) {
        rvv_den = 1;
        rvv_num = 0;
    }

    constexpr uint32_t cb_out = tt::CBIndex::c_16;      // SFPU stream
    constexpr uint32_t cb_rvv_out = tt::CBIndex::c_17;  // RVV stream
    const uint32_t tile_size_bytes = get_tile_size(cb_out);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out_accessor = TensorAccessor(out_args, out_addr, tile_size_bytes);

    for (uint32_t i = 0; i < n_tiles; i++) {
        const uint32_t cb = ((i % rvv_den) < rvv_num) ? cb_rvv_out : cb_out;  // only change vs writer.cpp
        cb_wait_front(cb, 1);
        uint32_t cb_addr = get_read_ptr(cb);
        noc_async_write_tile(start_tile_id + i, out_accessor, cb_addr);
        // Release the selected source CB only after the write has departed its
        // L1 page; defer remote acknowledgement completion to kernel exit.
        noc_async_writes_flushed();
        cb_pop_front(cb, 1);
    }
    noc_async_write_barrier();
}
