// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Combine lab port reader (production's combine_port_reader hand-off): grants this port's injection ring slots to the
// W writers - every writer's GRANT[pid] bumped by the slots freed, GB at a time (or the call's last ones) - and marks
// rows ready once every writer's LANDED count covers them (READY: the port's sender reads it). The sender frees slots
// in order (FREED). Everything cumulative over calls.
//
// RT: 0 pid (this port's id), 1 rows this call, 2 nw (the writers feeding this port: W, SPLIT a group's), then nw x
// writer core (x | y << 8).
#include "api/dataflow/dataflow_api.h"
#ifdef LAB_TIMING
#include "api/debug/kernel_profiler.hpp"
#endif

constexpr uint32_t W = get_named_compile_time_arg_val("W");
constexpr uint32_t ISLOTS = get_named_compile_time_arg_val("ISLOTS");
constexpr uint32_t GB = get_named_compile_time_arg_val("GB");  // grant batch
constexpr uint32_t ST = get_named_compile_time_arg_val("STATE");
constexpr uint32_t GRANT = ST + get_named_compile_time_arg_val("O_GRANT");  // (on the writers) [8]
constexpr uint32_t LANDED = ST + get_named_compile_time_arg_val("O_LANDED");
constexpr uint32_t READY = ST + get_named_compile_time_arg_val("O_READY");
constexpr uint32_t FREED = ST + get_named_compile_time_arg_val("O_FREED");
constexpr uint32_t GRANTED = ST + get_named_compile_time_arg_val("O_GRANTED");

void kernel_main() {
    const uint32_t pid = get_arg_val<uint32_t>(0), n = get_arg_val<uint32_t>(1), nw = get_arg_val<uint32_t>(2);
    uint32_t wxy[W];
    for (uint32_t i = 0; i < nw; ++i) {
        wxy[i] = get_arg_val<uint32_t>(3 + i);
    }
    volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(LANDED);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(READY);
    volatile tt_l1_ptr uint32_t* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(FREED);
    volatile tt_l1_ptr uint32_t* granted_w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(GRANTED);
    uint32_t granted = *granted_w;
    const uint32_t target = *ready + n;  // (every earlier row is ready, sent and freed: the calls drain)
#ifdef LAB_TIMING
    uint32_t t_grant = 0, n_pass = 0, n_batch = 0;
    const uint32_t t0 = get_timestamp_32b();
#endif
    while (*ready < target) {
#ifdef LAB_TIMING
        ++n_pass;
#endif
        invalidate_l1_cache();
        if (granted < target) {
            const uint32_t room = ISLOTS - (granted - *freed);
            uint32_t g = room < target - granted ? room : target - granted;
            if (g >= GB || (g > 0 && granted + g == target)) {
#ifdef LAB_TIMING
                const uint32_t tg = get_timestamp_32b();
                ++n_batch;
#endif
                for (uint32_t i = 0; i < nw; ++i) {
                    noc_semaphore_inc(get_noc_addr(wxy[i] & 0xFF, wxy[i] >> 8, GRANT + 4 * pid), g);
                }
                granted += g;
#ifdef LAB_TIMING
                t_grant += get_timestamp_32b() - tg;
#endif
            }
        }
        uint32_t m = 0xFFFFFFFFu;
        for (uint32_t i = 0; i < nw; ++i) {
            m = landed[i] < m ? landed[i] : m;
        }
        if (m > *ready) {
            *ready = m;
        }
    }
    *granted_w = granted;
    noc_async_atomic_barrier();
#ifdef LAB_TIMING
    DeviceTimestampedData("LAB-RGRANT", t_grant);
    DeviceTimestampedData("LAB-RBATCH", n_batch);
    DeviceTimestampedData("LAB-RPASS", n_pass);
    DeviceTimestampedData("LAB-REND", get_timestamp_32b() - t0);
#endif
}
