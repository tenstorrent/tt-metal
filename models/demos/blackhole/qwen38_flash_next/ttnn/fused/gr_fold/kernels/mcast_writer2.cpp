// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Two multicast phases in sequence on one producer core (gr_read/kernels/mcast_phase.h): the fused read's norm core
// first hands its stats tile to the transport core (phase A: one tile into the transport's scratch slot + the
// device's own gathered page), then, once it has normalized, multicasts the normalized row into the down workers'
// CB (phase B).  Between them, with GATE_SEM set, the core raises its OWN gate semaphore by one (a NoC atomic, like
// the transport cores' increments of the same semaphore): phase A's own-page write is complete at its write
// barrier, so the core's reader, which waits for the transports' signals plus this one before it streams the
// gathered stats pages, never reads its own page early (mcast_phase raises the transport's scratch semaphore before
// it writes the page; inside one program only this signal orders the two).
// Compile-time args: 0-5 phase A (src cb, dst cb, tiles, write tiles, extra cb, semaphore id), 6-11 phase B,
// 12 gate semaphore id (0xFF none), 13.. four TensorAccessorArgs sets (A tiles, A extra, B tiles, B extra; unused
// slots repeat).  Runtime args: 0-12 phase A, 13-25 phase B (mcast_writer.cpp's layout each), 26-27 this core's
// NoC (x, y) for the gate increment.

#include "../../gr_read/kernels/mcast_phase.h"
#include "../../kernels/zones.h"

constexpr uint32_t GATE_SEM = get_compile_time_arg_val(12);
constexpr uint32_t ACCESSOR_BASE = 13;

void kernel_main() {
    constexpr auto a_tiles = TensorAccessorArgs<ACCESSOR_BASE>();
    constexpr auto a_extra = TensorAccessorArgs<a_tiles.next_compile_time_args_offset()>();
    constexpr auto b_tiles = TensorAccessorArgs<a_extra.next_compile_time_args_offset()>();
    constexpr auto b_extra = TensorAccessorArgs<b_tiles.next_compile_time_args_offset()>();
    {
        FUSED_ZONE("fz_gf_mw2_stats");
        mcast_phase<
            get_compile_time_arg_val(0),
            get_compile_time_arg_val(1),
            get_compile_time_arg_val(2),
            get_compile_time_arg_val(3),
            get_compile_time_arg_val(4),
            get_compile_time_arg_val(5)>(a_tiles, a_extra, 0);
    }
    if constexpr (GATE_SEM != 0xFF) {
        Noc noc;
        Semaphore<> gate(GATE_SEM);
        gate.up(noc, get_arg_val<uint32_t>(26), get_arg_val<uint32_t>(27), 1);
        noc.async_atomic_barrier();
    }
    {
        FUSED_ZONE("fz_gf_mw2_partials");
        mcast_phase<
            get_compile_time_arg_val(6),
            get_compile_time_arg_val(7),
            get_compile_time_arg_val(8),
            get_compile_time_arg_val(9),
            get_compile_time_arg_val(10),
            get_compile_time_arg_val(11)>(b_tiles, b_extra, 13);
    }
}
