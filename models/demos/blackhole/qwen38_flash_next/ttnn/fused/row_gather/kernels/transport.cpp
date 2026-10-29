// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The packed row all-gather of the per-block hidden: one gather phase over the line (gr_fold/kernels/transport_phase.h,
// SOURCE 0: NCRISC reads this device's local tiles into the scratch CB and writes them, in full, into the device's own
// pages of the gathered tensor).  With ROW_MODE 3 only the valid rows of each tile travel: the tiles' valid chunks are
// packed into one packet per direction, land in the peers' transport-core L1 and are unpacked into the gathered
// tensor's pages at the same byte offsets a full-tile write uses.  Rows >= ROWS of a remote shard are not written.
// Compile-time args: 0 scratch cb, 1 TILES, 2 go semaphore id, 3 RING, 4 SOURCE, 5 scratch semaphore id, 6 done
//   semaphore id, 7 TILE_FIRST, 8 TILE_STEP, 9 PAGE_TILE_STRIDE, 10 PAGE_RANK_STRIDE, 11 ROWS, 12 ROW_MODE, 13 PACK_CB,
//   14 SLOT_BYTES, 15.. TensorAccessorArgs of the gathered tensor, then of the local tensor.
// Runtime args as gr_fold/kernels/transport.cpp.

#include "../../gr_fold/kernels/transport_phase.h"
#include "../../kernels/zones.h"

constexpr uint32_t SCRATCH_CB = get_compile_time_arg_val(0);
constexpr uint32_t TILES = get_compile_time_arg_val(1);
constexpr uint32_t SEM_GO = get_compile_time_arg_val(2);
constexpr uint32_t RING = get_compile_time_arg_val(3);
constexpr uint32_t SOURCE = get_compile_time_arg_val(4);
constexpr uint32_t SEM_SCRATCH = get_compile_time_arg_val(5);
constexpr uint32_t SEM_DONE = get_compile_time_arg_val(6);
constexpr uint32_t TILE_FIRST = get_compile_time_arg_val(7);
constexpr uint32_t TILE_STEP = get_compile_time_arg_val(8);
constexpr uint32_t PAGE_TILE_STRIDE = get_compile_time_arg_val(9);
constexpr uint32_t PAGE_RANK_STRIDE = get_compile_time_arg_val(10);
constexpr uint32_t ROWS = get_compile_time_arg_val(11);
constexpr uint32_t ROW_MODE = get_compile_time_arg_val(12);
constexpr uint32_t PACK_CB = get_compile_time_arg_val(13);
constexpr uint32_t SLOT_BYTES = get_compile_time_arg_val(14);
constexpr uint32_t ACCESSOR_BASE = 15;
constexpr uint32_t PHASE_RT = 3;

void kernel_main() {
    const uint32_t rank = get_arg_val<uint32_t>(0);
    size_t arg_idx = PHASE_RT + PHASE_RT_ARGS + 2 * get_arg_val<uint32_t>(PHASE_RT + 4);
    constexpr auto out_args = TensorAccessorArgs<ACCESSOR_BASE>();
    constexpr auto local_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    LineSender line;
    line.open<RING>(rank, arg_idx);
    {
        FUSED_ZONE("fz_rg_main");
        line.arrive(get_arg_val<uint32_t>(PHASE_RT + 2));
        transport_phase<
            SCRATCH_CB,
            TILES,
            TILE_FIRST,
            TILE_STEP,
            PAGE_TILE_STRIDE,
            PAGE_RANK_STRIDE,
            SOURCE,
            RING,
            SEM_GO,
            SEM_SCRATCH,
            SEM_DONE,
            true,
            ROWS,
            ROW_MODE,
            PACK_CB,
            SLOT_BYTES>(line, out_args, local_args, PHASE_RT, 0);
    }
    noc_async_full_barrier();
}
