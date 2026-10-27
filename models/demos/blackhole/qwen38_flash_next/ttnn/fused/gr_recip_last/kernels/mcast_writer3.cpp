// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Three multicast phases in sequence on one norm core of ttnn/fused/gr_recip_last (gr_read/kernels/mcast_phase.h
// each): the stats phase hands the stats tile to the transport core (its scratch slot + the device's own gathered
// page), then raises the core's OWN gate semaphore: mcast_phase signals the transport before the own-page write lands,
// so only this signal keeps the core's reader from streaming its own gathered page early. The u-row phase multicasts
// the u row into the down workers' CB; the recip phase multicasts the recip tile into the workers' recip CB and writes
// the normalized' row for the gate.
// Compile-time args: 0-5 the stats phase (src cb, dst cb, tiles, write tiles, extra cb, semaphore id), 6-11 the u-row
// phase, 12-17 the recip phase, 18 gate semaphore id (0xFF none), 19.. six TensorAccessorArgs sets (stats tiles, stats
// extra, u-row tiles, u-row extra, recip tiles, recip extra; unused slots repeat).  Runtime args: 0-12 the stats
// phase, 13-25 the u-row phase, 26-38 the recip phase (mcast_phase.h's layout each), 39-40 this core's NoC (x, y) for
// the gate increment.

#include "../../gr_read/kernels/mcast_phase.h"

constexpr uint32_t gate_semaphore_id = get_compile_time_arg_val(18);
constexpr uint32_t accessor_args_base = 19;

void kernel_main() {
    constexpr auto stats_tiles_args = TensorAccessorArgs<accessor_args_base>();
    constexpr auto stats_extra_args = TensorAccessorArgs<stats_tiles_args.next_compile_time_args_offset()>();
    constexpr auto u_row_tiles_args = TensorAccessorArgs<stats_extra_args.next_compile_time_args_offset()>();
    constexpr auto u_row_extra_args = TensorAccessorArgs<u_row_tiles_args.next_compile_time_args_offset()>();
    constexpr auto recip_tiles_args = TensorAccessorArgs<u_row_extra_args.next_compile_time_args_offset()>();
    constexpr auto recip_extra_args = TensorAccessorArgs<recip_tiles_args.next_compile_time_args_offset()>();
    {
        mcast_phase<
            get_compile_time_arg_val(0),
            get_compile_time_arg_val(1),
            get_compile_time_arg_val(2),
            get_compile_time_arg_val(3),
            get_compile_time_arg_val(4),
            get_compile_time_arg_val(5)>(stats_tiles_args, stats_extra_args, 0);
    }
    if constexpr (gate_semaphore_id != 0xFF) {
        const Noc noc;
        Semaphore<> gate(gate_semaphore_id);
        gate.up(noc, get_arg_val<uint32_t>(39), get_arg_val<uint32_t>(40), 1);
        noc.async_atomic_barrier();
    }
    {
        mcast_phase<
            get_compile_time_arg_val(6),
            get_compile_time_arg_val(7),
            get_compile_time_arg_val(8),
            get_compile_time_arg_val(9),
            get_compile_time_arg_val(10),
            get_compile_time_arg_val(11)>(u_row_tiles_args, u_row_extra_args, 13);
    }
    {
        mcast_phase<
            get_compile_time_arg_val(12),
            get_compile_time_arg_val(13),
            get_compile_time_arg_val(14),
            get_compile_time_arg_val(15),
            get_compile_time_arg_val(16),
            get_compile_time_arg_val(17)>(recip_tiles_args, recip_extra_args, 26);
    }
}
