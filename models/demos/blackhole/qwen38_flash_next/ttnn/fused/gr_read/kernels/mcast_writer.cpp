// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// A producer core's writer: multicasts its `num_tiles` packed tiles into the consumers' CB (the same L1 address on
// every core that declares that CB) at a tile offset, then raises each consumer's semaphore by one; optionally also
// writes the same tiles to a TILE tensor, and one extra CB stream to another tensor.  Consumers are the NoC
// rectangle [x0..x1] x [y0..y1] (NoC-0 virtual coordinates measured by noc_probe.cpp, top-left first).  A writer
// kernel runs on NOC_1 (WriterDataMovementConfig = BRISC + preferred_noc_for_dram_write = NOC_1 on every arch), and
// NOC_1 routes a multicast from the opposite corner, so the start/end corners are swapped for it exactly as the
// DRAM-sharded matmul factory and tests/tt_metal/.../one_to_all/kernels/sender_multicast.cpp do.
// Compile-time args: 0 src cb, 1 dst cb, 2 num_tiles, 3 write the tiles to DRAM (0/1), 4 extra stream cb (0xFF none),
//   5 semaphore id, 6.. two TensorAccessorArgs sets (tiles tensor, extra tensor; unused slots repeat).
// Runtime args: 0 dst tile offset, 1-4 NoC x0 y0 x1 y1, 5-7 tiles tensor (addr, first, stride), 8-12 extra stream
//   (addr, count, first, stride, batch).

#include "mcast_phase.h"
#include "../../kernels/zones.h"

constexpr uint32_t SRC_CB = get_compile_time_arg_val(0);
constexpr uint32_t DST_CB = get_compile_time_arg_val(1);
constexpr uint32_t NUM_TILES = get_compile_time_arg_val(2);
constexpr uint32_t WRITE_TILES = get_compile_time_arg_val(3);
constexpr uint32_t EXTRA_CB = get_compile_time_arg_val(4);
constexpr uint32_t SEM_ID = get_compile_time_arg_val(5);
constexpr uint32_t ACCESSOR_BASE = 6;

void kernel_main() {
    constexpr auto tiles_args = TensorAccessorArgs<ACCESSOR_BASE>();
    constexpr auto extra_args = TensorAccessorArgs<tiles_args.next_compile_time_args_offset()>();
    {
        FUSED_ZONE("fz_gr_mcw_main");
        mcast_phase<SRC_CB, DST_CB, NUM_TILES, WRITE_TILES, EXTRA_CB, SEM_ID>(tiles_args, extra_args, 0);
    }
}
