// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Three multicast phases in sequence on one norm core of ttnn/fused/gr_recip_last (gr_read/kernels/mcast_phase.h
// each): A hands the stats tile to the transport core (its scratch slot + the device's own gathered page), then the
// core raises its OWN gate semaphore (the own-page write is complete at phase A's barrier; gr_fold's mcast_writer2.cpp
// has the argument); B multicasts the u row into the down workers' CB; C multicasts the recip tile into the workers'
// recip CB and, as its extra stream, writes the normalized' row to the normalized tensor for the gate.
// Compile-time args: 0-5 phase A (src cb, dst cb, tiles, write tiles, extra cb, semaphore id), 6-11 phase B, 12-17
// phase C, 18 gate semaphore id (0xFF none), 19.. six TensorAccessorArgs sets (A tiles, A extra, B tiles, B extra,
// C tiles, C extra; unused slots repeat).  Runtime args: 0-12 phase A, 13-25 phase B, 26-38 phase C
// (mcast_phase.h's layout each), 39-40 this core's NoC (x, y) for the gate increment.

#include "../../gr_read/kernels/mcast_phase.h"
#include "../../kernels/zones.h"

constexpr uint32_t GATE_SEM = get_compile_time_arg_val(18);
constexpr uint32_t ACCESSOR_BASE = 19;

void kernel_main() {
    constexpr auto a_tiles = TensorAccessorArgs<ACCESSOR_BASE>();
    constexpr auto a_extra = TensorAccessorArgs<a_tiles.next_compile_time_args_offset()>();
    constexpr auto b_tiles = TensorAccessorArgs<a_extra.next_compile_time_args_offset()>();
    constexpr auto b_extra = TensorAccessorArgs<b_tiles.next_compile_time_args_offset()>();
    constexpr auto c_tiles = TensorAccessorArgs<b_extra.next_compile_time_args_offset()>();
    constexpr auto c_extra = TensorAccessorArgs<c_tiles.next_compile_time_args_offset()>();
    {
        FUSED_ZONE("fz_gl_w_stats");
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
        gate.up(noc, get_arg_val<uint32_t>(39), get_arg_val<uint32_t>(40), 1);
        noc.async_atomic_barrier();
    }
    {
        FUSED_ZONE("fz_gl_w_u");
        mcast_phase<
            get_compile_time_arg_val(6),
            get_compile_time_arg_val(7),
            get_compile_time_arg_val(8),
            get_compile_time_arg_val(9),
            get_compile_time_arg_val(10),
            get_compile_time_arg_val(11)>(b_tiles, b_extra, 13);
    }
    {
        FUSED_ZONE("fz_gl_w_recip");
        mcast_phase<
            get_compile_time_arg_val(12),
            get_compile_time_arg_val(13),
            get_compile_time_arg_val(14),
            get_compile_time_arg_val(15),
            get_compile_time_arg_val(16),
            get_compile_time_arg_val(17)>(c_tiles, c_extra, 26);
    }
}
