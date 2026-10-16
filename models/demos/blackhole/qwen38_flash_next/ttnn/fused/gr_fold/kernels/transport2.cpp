// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Two gather phases in sequence on one connection per RISC (transport_phase.h): the fused read's front gathers the
// stats (phase A, then it releases the norm cores through their gate semaphore) and the partials (phase B) with the
// same transport core per link, so the program holds one open sender per link per direction.  Both arrives go out at
// kernel start; phase B's wait, after phase A, finds its counter complete long before the partial tiles exist.  Both
// phases take their tiles from producer cores of the program (SOURCE 1): the go/done handshake that orders the reuse
// of `go` across the phases needs `done`, which SOURCE 0 does not raise.
// Compile-time args: 0 go semaphore id, 1 RING, 2 done semaphore id, 3 TILE_FIRST, 4 TILE_STEP, 5-10 phase A
//   (scratch cb, TILES, SOURCE, scratch semaphore id, PAGE_TILE_STRIDE, PAGE_RANK_STRIDE), 11-16 phase B, 17.. four
//   TensorAccessorArgs sets (A gathered, A local, B gathered, B local).
// Runtime args: 0 rank, 1 delay before the arrives, 2 delay after phase A's reset, 3.. phase A's block, then phase
//   B's block (transport_phase.h: 6 words + the consumers' (x, y) pairs each), then the fabric connection.

#include "transport_phase.h"
#include "../../kernels/zones.h"

constexpr uint32_t SEM_GO = get_compile_time_arg_val(0);
constexpr uint32_t RING = get_compile_time_arg_val(1);
constexpr uint32_t SEM_DONE = get_compile_time_arg_val(2);
constexpr uint32_t TILE_FIRST = get_compile_time_arg_val(3);
constexpr uint32_t TILE_STEP = get_compile_time_arg_val(4);
constexpr uint32_t A_SCRATCH_CB = get_compile_time_arg_val(5);
constexpr uint32_t A_TILES = get_compile_time_arg_val(6);
constexpr uint32_t A_SOURCE = get_compile_time_arg_val(7);
constexpr uint32_t A_SEM_SCRATCH = get_compile_time_arg_val(8);
constexpr uint32_t A_PAGE_TILE_STRIDE = get_compile_time_arg_val(9);
constexpr uint32_t A_PAGE_RANK_STRIDE = get_compile_time_arg_val(10);
constexpr uint32_t B_SCRATCH_CB = get_compile_time_arg_val(11);
constexpr uint32_t B_TILES = get_compile_time_arg_val(12);
constexpr uint32_t B_SOURCE = get_compile_time_arg_val(13);
constexpr uint32_t B_SEM_SCRATCH = get_compile_time_arg_val(14);
constexpr uint32_t B_PAGE_TILE_STRIDE = get_compile_time_arg_val(15);
constexpr uint32_t B_PAGE_RANK_STRIDE = get_compile_time_arg_val(16);
constexpr uint32_t ACCESSOR_BASE = 17;
constexpr uint32_t A_RT = 3;
static_assert(A_SOURCE == 1 && B_SOURCE == 1, "the two-phase transport takes both phases from producer cores");
static_assert(A_SEM_SCRATCH != B_SEM_SCRATCH, "each phase has its own scratch semaphore");

void kernel_main() {
    const uint32_t rank = get_arg_val<uint32_t>(0);
    const uint32_t delay_before_arrive = get_arg_val<uint32_t>(1);
    const uint32_t delay_after_reset = get_arg_val<uint32_t>(2);
    const uint32_t b_rt = A_RT + PHASE_RT_ARGS + 2 * get_arg_val<uint32_t>(A_RT + 4);
    size_t arg_idx = b_rt + PHASE_RT_ARGS + 2 * get_arg_val<uint32_t>(b_rt + 4);
    constexpr auto a_out_args = TensorAccessorArgs<ACCESSOR_BASE>();
    constexpr auto a_local_args = TensorAccessorArgs<a_out_args.next_compile_time_args_offset()>();
    constexpr auto b_out_args = TensorAccessorArgs<a_local_args.next_compile_time_args_offset()>();
    constexpr auto b_local_args = TensorAccessorArgs<b_out_args.next_compile_time_args_offset()>();
    LineSender line;
    line.open<RING>(rank, arg_idx);
    if (delay_before_arrive != 0) {
        riscv_wait(delay_before_arrive);
    }
    {
        FUSED_ZONE("fz_gf_tr2_stats");
        line.arrive(get_arg_val<uint32_t>(A_RT + 2));
        line.arrive(get_arg_val<uint32_t>(b_rt + 2));
        transport_phase<
            A_SCRATCH_CB,
            A_TILES,
            TILE_FIRST,
            TILE_STEP,
            A_PAGE_TILE_STRIDE,
            A_PAGE_RANK_STRIDE,
            A_SOURCE,
            RING,
            SEM_GO,
            A_SEM_SCRATCH,
            SEM_DONE,
            false>(line, a_out_args, a_local_args, A_RT, delay_after_reset);
    }
    {
        FUSED_ZONE("fz_gf_tr2_partials");
        transport_phase<
            B_SCRATCH_CB,
            B_TILES,
            TILE_FIRST,
            TILE_STEP,
            B_PAGE_TILE_STRIDE,
            B_PAGE_RANK_STRIDE,
            B_SOURCE,
            RING,
            SEM_GO,
            B_SEM_SCRATCH,
            SEM_DONE,
            true>(line, b_out_args, b_local_args, b_rt, 0);
    }
    noc_async_full_barrier();
}
