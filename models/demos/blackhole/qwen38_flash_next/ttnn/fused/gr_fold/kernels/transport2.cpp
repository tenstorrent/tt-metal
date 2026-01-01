// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Two gather phases in sequence on one connection per RISC (transport_phase.h): the fused read's front gathers the
// stats (the first phase, then it releases the norm cores through their gate semaphore) and the partials (the second)
// with the same transport core per link, so the program holds one open sender per link per direction.  Both phases
// take their tiles from producer cores (`source` 1): the handshake that orders the reuse of `go` across phases needs
// `done`, which `source` 0 does not raise.
// Compile-time args: 0 go semaphore id, 1 line_devices, 2 done semaphore id, 3 first_tile, 4 tile_step, 5-10 the stats
//   phase (scratch cb, tiles, source, scratch semaphore id, page tile stride, page rank stride), 11-16 the partials
//   phase, 17.. four TensorAccessorArgs sets (stats gathered, stats local, partials gathered, partials local).
// Runtime args: 0 rank, 1 delay before the arrives, 2 delay after the stats phase's reset, 3.. the stats phase's block,
//   then the partials phase's block (transport_phase.h: 6 words + the consumers' (x, y) pairs each), then the fabric
//   connection.

#include "transport_phase.h"

constexpr uint32_t go_semaphore_id = get_compile_time_arg_val(0);
constexpr uint32_t line_devices = get_compile_time_arg_val(1);
constexpr uint32_t done_semaphore_id = get_compile_time_arg_val(2);
constexpr uint32_t first_tile = get_compile_time_arg_val(3);
constexpr uint32_t tile_step = get_compile_time_arg_val(4);
constexpr uint32_t stats_scratch_cb = get_compile_time_arg_val(5);
constexpr uint32_t stats_tiles = get_compile_time_arg_val(6);
constexpr uint32_t stats_source = get_compile_time_arg_val(7);
constexpr uint32_t stats_scratch_semaphore_id = get_compile_time_arg_val(8);
constexpr uint32_t stats_page_tile_stride = get_compile_time_arg_val(9);
constexpr uint32_t stats_page_rank_stride = get_compile_time_arg_val(10);
constexpr uint32_t partials_scratch_cb = get_compile_time_arg_val(11);
constexpr uint32_t partials_tiles = get_compile_time_arg_val(12);
constexpr uint32_t partials_source = get_compile_time_arg_val(13);
constexpr uint32_t partials_scratch_semaphore_id = get_compile_time_arg_val(14);
constexpr uint32_t partials_page_tile_stride = get_compile_time_arg_val(15);
constexpr uint32_t partials_page_rank_stride = get_compile_time_arg_val(16);
constexpr uint32_t accessor_args_base = 17;
constexpr uint32_t stats_first_arg = 3;
static_assert(
    stats_source == 1 && partials_source == 1, "the two-phase transport takes both phases from producer cores");
static_assert(stats_scratch_semaphore_id != partials_scratch_semaphore_id, "each phase has its own scratch semaphore");

void kernel_main() {
    const uint32_t rank = get_arg_val<uint32_t>(0);
    const uint32_t delay_before_arrive = get_arg_val<uint32_t>(1);
    const uint32_t delay_after_reset = get_arg_val<uint32_t>(2);
    const uint32_t partials_first_arg =
        stats_first_arg + phase_runtime_args + 2 * get_arg_val<uint32_t>(stats_first_arg + 4);
    size_t connection_arg = partials_first_arg + phase_runtime_args + 2 * get_arg_val<uint32_t>(partials_first_arg + 4);
    constexpr auto stats_gathered_args = TensorAccessorArgs<accessor_args_base>();
    constexpr auto stats_local_args = TensorAccessorArgs<stats_gathered_args.next_compile_time_args_offset()>();
    constexpr auto partials_gathered_args = TensorAccessorArgs<stats_local_args.next_compile_time_args_offset()>();
    constexpr auto partials_local_args = TensorAccessorArgs<partials_gathered_args.next_compile_time_args_offset()>();
    LineSender line;
    line.open<line_devices>(rank, connection_arg);
    if (delay_before_arrive != 0) {
        riscv_wait(delay_before_arrive);
    }
    {
        line.arrive(get_arg_val<uint32_t>(stats_first_arg + 2));
        line.arrive(get_arg_val<uint32_t>(partials_first_arg + 2));
        transport_phase<
            stats_scratch_cb,
            stats_tiles,
            first_tile,
            tile_step,
            stats_page_tile_stride,
            stats_page_rank_stride,
            stats_source,
            line_devices,
            go_semaphore_id,
            stats_scratch_semaphore_id,
            done_semaphore_id,
            false>(line, stats_gathered_args, stats_local_args, stats_first_arg, delay_after_reset);
    }
    {
        transport_phase<
            partials_scratch_cb,
            partials_tiles,
            first_tile,
            tile_step,
            partials_page_tile_stride,
            partials_page_rank_stride,
            partials_source,
            line_devices,
            go_semaphore_id,
            partials_scratch_semaphore_id,
            done_semaphore_id,
            true>(line, partials_gathered_args, partials_local_args, partials_first_arg, 0);
    }
    const Noc noc;
    noc.async_full_barrier();
}
