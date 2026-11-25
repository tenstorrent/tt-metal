// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Combine lab accumulator, stage "take": polls its landing rings - one ring of D slots a (source chip, part): SL parts
// a row (SPLIT: a row's halves travel and land apart, one a link; else 1) - in arrival order a ring (slot (rows taken
// so far) % D), takes each row (part) once its last word (the arrival flag) is set (clears it: the slot is free), and
// counts it at the home port that returns that ring's credits (its rows taken per (source, accumulator) and per
// source: the port snapshots them back). READ_ROW: also reads the row (a NoC read of PB bytes into a local buffer).
// POOL: the rings on the pool cores - every pass reads each waiting ring's current slot's last 64 B over the NoC, takes
// the landed rows and clears their flags with a NoC write. Ends once every ring's rows of this call are taken.
// Cumulative over calls (GOT), no per-call zeroing.
//
// RT: 0 me (block index), 1 o (this accumulator's index), then N SL x (rows a call into ring r = s SL + part, the home
// port core returning its credits x | y << 8), then POOL x pool core (x | y << 8).
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PB = get_named_compile_time_arg_val("PB");
constexpr uint32_t SLOT = get_named_compile_time_arg_val("SLOT");
constexpr uint32_t D = get_named_compile_time_arg_val("D");
constexpr uint32_t N = get_named_compile_time_arg_val("N");
constexpr uint32_t AP = get_named_compile_time_arg_val("AP");
constexpr uint32_t SL = get_named_compile_time_arg_val("SL");  // parts a row (SPLIT: L, else 1)
constexpr uint32_t NR = N * SL;                                // rings
constexpr uint32_t ST = get_named_compile_time_arg_val("STATE");
constexpr uint32_t RINGS = get_named_compile_time_arg_val("RINGS");
constexpr uint32_t GOT = ST + get_named_compile_time_arg_val("O_GOT");  // [N SL] rows taken a ring (cumulative)
constexpr uint32_t RET = ST + get_named_compile_time_arg_val("O_RET");
constexpr uint32_t RTOT = ST + get_named_compile_time_arg_val("O_RTOT");
constexpr uint32_t SCR = ST + get_named_compile_time_arg_val("O_SRC");  // (READ_ROW: the row's local copy)
constexpr uint32_t READ_ROW = get_named_compile_time_arg_val("READ_ROW");
constexpr uint32_t POOL = get_named_compile_time_arg_val("POOL");  // (lab_port.cpp)
constexpr uint32_t NPOOL = POOL > 0 ? POOL : 1;  // (a divisor in branches compiled with POOL 0)
constexpr uint32_t PROBE = ST + get_named_compile_time_arg_val("O_PROBE");  // [NR] x 64 B trailer probes; 64 B zeros

void kernel_main() {
    const uint32_t o = get_arg_val<uint32_t>(1);
    uint32_t want[NR], port_xy[NR];
    uint32_t total = 0;
    for (uint32_t r = 0; r < NR; ++r) {
        want[r] = get_arg_val<uint32_t>(2 + 2 * r);
        port_xy[r] = get_arg_val<uint32_t>(3 + 2 * r);
        total += want[r];
    }
    volatile tt_l1_ptr uint32_t* got = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(GOT);
    uint32_t pool_xy[POOL > 0 ? POOL : 1];
    for (uint32_t i = 0; i < POOL; ++i) {
        pool_xy[i] = get_arg_val<uint32_t>(2 + 2 * NR + i);
    }
    auto count = [&](uint32_t r, uint32_t n) {  // (n rows of ring r taken: credited at its home port)
        const uint32_t s = r / SL, px = port_xy[r] & 0xFF, py = port_xy[r] >> 8;
        noc_semaphore_inc(get_noc_addr(px, py, RET + 4 * (s * AP + o)), n);
        noc_semaphore_inc(get_noc_addr(px, py, RTOT + 4 * s), n);
    };
    if constexpr (POOL > 0) {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROBE + NR * 64);
        for (uint32_t w = 0; w < 16; ++w) {
            z[w] = 0;
        }
        auto slot_noc = [&](uint32_t r) {
            const uint32_t g = (o * NR + r) * D + got[r] % D, xy = pool_xy[g % NPOOL];
            return get_noc_addr(xy & 0xFF, xy >> 8, RINGS + (g / NPOOL) * SLOT);
        };
        while (total > 0) {
            for (uint32_t r = 0; r < NR; ++r) {  // (every waiting ring's probe in flight at once)
                if (want[r]) {
                    noc_async_read(slot_noc(r) + PB - 64, PROBE + r * 64, 64);
                }
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t r = 0; r < NR; ++r) {
                if (want[r] == 0 || reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROBE + r * 64)[15] == 0) {
                    continue;
                }
                const uint64_t sn = slot_noc(r);
                if constexpr (READ_ROW) {
                    noc_async_read(sn, SCR, PB);
                    noc_async_read_barrier();
                }
                noc_async_write(PROBE + NR * 64, sn + PB - 64, 64);  // (the flag cleared: the slot is free)
                noc_async_writes_flushed();
                got[r] = got[r] + 1;
                --want[r];
                --total;
                count(r, 1);
            }
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        return;
    }
    while (total > 0) {
        invalidate_l1_cache();
        for (uint32_t r = 0; r < NR; ++r) {
            if (want[r] == 0) {
                continue;
            }
            const uint32_t slot = RINGS + (r * D + got[r] % D) * SLOT;
            volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + PB - 4);
            if (*flag == 0) {
                continue;
            }
            if constexpr (READ_ROW) {
                noc_async_read(get_noc_addr(slot), SCR, PB);
                noc_async_read_barrier();
            }
            *flag = 0;
            got[r] = got[r] + 1;
            --want[r];
            --total;
            count(r, 1);
        }
    }
    noc_async_atomic_barrier();
}
