// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One gather phase per program (transport_phase.h): the standalone gather of the smoke tool (SOURCE 0) and the
// stage-(b)/(c) programs (SOURCE 1).
// Compile-time args: 0 scratch cb, 1 TILES, 2 go semaphore id, 3 RING, 4 SOURCE, 5 scratch semaphore id, 6 done
//   semaphore id, 7 TILE_FIRST, 8 TILE_STEP, 9 PAGE_TILE_STRIDE, 10 PAGE_RANK_STRIDE, 11.. TensorAccessorArgs of the
//   gathered tensor, then of the local tensor.
// Runtime args: 0 rank, 1 delay before the arrive, 2 delay after the reset, 3.. the phase block (gathered address,
//   local address, barrier address, data address, consumer count N, consumer semaphore id, N NoC (x, y) pairs), then
//   the fabric connection (ttnn.setup_fabric_connection) when this RISC's direction has peers.

#include "transport_phase.h"
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
constexpr uint32_t ACCESSOR_BASE = 11;
constexpr uint32_t PHASE_RT = 3;

void kernel_main() {
    const uint32_t rank = get_arg_val<uint32_t>(0);
    const uint32_t delay_before_arrive = get_arg_val<uint32_t>(1);
    const uint32_t delay_after_reset = get_arg_val<uint32_t>(2);
    size_t arg_idx = PHASE_RT + PHASE_RT_ARGS + 2 * get_arg_val<uint32_t>(PHASE_RT + 4);
    constexpr auto out_args = TensorAccessorArgs<ACCESSOR_BASE>();
    constexpr auto local_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    LineSender line;
    line.open<RING>(rank, arg_idx);
    if (delay_before_arrive != 0) {
        riscv_wait(delay_before_arrive);
    }
    {
        FUSED_ZONE("fz_gf_tr_main");
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
            true>(line, out_args, local_args, PHASE_RT, delay_after_reset);
    }
    noc_async_full_barrier();
}
