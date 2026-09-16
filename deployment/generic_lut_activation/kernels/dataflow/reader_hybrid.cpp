// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Hybrid RVV/SFPU dual-stream reader (Architecture A, dedicated-CB streams).
// Identical to reader.cpp except each tile is steered by the shared parity
//   is_rvv(i) = (i % rvv_den) < rvv_num   (i = LOCAL tile index on this core)
// into c_1 (RVV stream, consumed raw by TRISC2's pack thread) instead of c_0
// (SFPU stream). rvv_num == 0 reproduces the production single-stream flow on
// c_0 exactly. Selected by the host only when HYBRID_RVV_SHARE is set; the
// hybrid demo and FUSE_GRAD are mutually exclusive (host-asserted), so c_1 is
// free for the RVV stream.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t in_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_tile_id = get_arg_val<uint32_t>(2);  // Tile offset for multi-core
    uint32_t rvv_num = get_arg_val<uint32_t>(3);
    uint32_t rvv_den = get_arg_val<uint32_t>(4);
    if (rvv_den == 0) {
        rvv_den = 1;
        rvv_num = 0;
    }

    constexpr uint32_t cb_in = tt::CBIndex::c_0;      // SFPU stream
    constexpr uint32_t cb_rvv_in = tt::CBIndex::c_1;  // RVV stream
    const uint32_t tile_size_bytes = get_tile_size(cb_in);

    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto in_accessor = TensorAccessor(in_args, in_addr, tile_size_bytes);

    for (uint32_t i = 0; i < n_tiles; i++) {
        const uint32_t cb = ((i % rvv_den) < rvv_num) ? cb_rvv_in : cb_in;  // only change vs reader.cpp
        cb_reserve_back(cb, 1);
        uint32_t cb_addr = get_write_ptr(cb);
        noc_async_read_tile(start_tile_id + i, in_accessor, cb_addr);
        noc_async_read_barrier();
        cb_push_back(cb, 1);
    }
}
