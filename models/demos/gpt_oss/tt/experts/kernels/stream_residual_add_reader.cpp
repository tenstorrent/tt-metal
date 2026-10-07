// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Packed residual add, data movement (RISCV_1, NOC0) (experts/stream.py: PackedResidualAdd).
//
// The decode all-reduce runs on a packed [32, W] BF16 partial (W = hidden / residual cores): hidden value h sits at
// row h / W, column h % W, so residual core c (shard [32, W], row 0 = the token) needs exactly row c of the packed
// sum. This kernel publishes the core's residual shard (cb_res, backed by the residual tensor) and builds cb_p:
// `tiles` 32x32 BF16 tiles that are zero except row 0 = packed row c (a tile row is 16 values in face 0/2 and 16 in
// face 1/3; row 0 is bytes [0, 32) and [512, 544) of the tile).
//
// runtime args: [packed_addr, row]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t packed_addr = get_arg_val<uint32_t>(0);
    const uint32_t row = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_res = get_compile_time_arg_val(0);
    constexpr uint32_t cb_p = get_compile_time_arg_val(1);
    constexpr uint32_t tiles = get_compile_time_arg_val(2);
    constexpr auto packed_args = TensorAccessorArgs<3>();

    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t face_bytes = 512;
    constexpr uint32_t half_row = 32;
    const auto s_packed = TensorAccessor(packed_args, packed_addr, tile_bytes);

    cb_reserve_back(cb_res, tiles);
    cb_push_back(cb_res, tiles);

    cb_reserve_back(cb_p, tiles);
    const uint32_t dst = get_write_ptr(cb_p);
    // Row 0 of each tile (bytes [0, 32) of face 0 and [512, 544) of face 1) comes from the packed sum; everything
    // else is zero-filled from the local MEM_ZEROS region. The destinations are disjoint, so all reads are issued
    // together and waited for once.
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    const uint32_t off = row < 16 ? row * half_row : 2 * face_bytes + (row - 16) * half_row;
    for (uint32_t t = 0; t < tiles; ++t) {
        const uint32_t base = dst + t * tile_bytes;
        const uint64_t src = s_packed.get_noc_addr(t, off);
        noc_async_read(src, base, half_row);
        noc_async_read(src + face_bytes, base + face_bytes, half_row);
        noc_async_read(zeros, base + half_row, face_bytes - half_row);
        for (uint32_t z = face_bytes + half_row; z < tile_bytes; z += MEM_ZEROS_SIZE) {
            const uint32_t n = tile_bytes - z < MEM_ZEROS_SIZE ? tile_bytes - z : MEM_ZEROS_SIZE;
            noc_async_read(zeros, base + z, n);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_p, tiles);
}
