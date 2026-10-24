// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Times each kernel-exit write-barrier variant for one scenario (see dfb_write_barrier_bench_common.h).
// Each iteration sets up the scenario, times one barrier variant, checks whether writes were still
// outstanding right after it returned, then drains untimed so the next iteration starts clean.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "dev_mem_map.h"  // MEM_L1_UNCACHED_BASE
#include "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_write_barrier_bench_common.h"

using namespace dfb_wb_bench;

namespace {

inline uint32_t cycles() {
    uint32_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
}

inline uint32_t outstanding_all() { return static_cast<uint32_t>(__builtin_riscv_ttrocc_scmdbuf_tr_ack()); }
inline uint32_t outstanding_trid(uint32_t trid) {
    return static_cast<uint32_t>(__builtin_riscv_ttrocc_scmdbuf_tr_ack_trid(trid));
}

inline void issue_explicit_writes(uint32_t src, uint64_t dst, uint32_t n, uint32_t bytes) {
    for (uint32_t i = 0; i < n; i++) {
        noc_async_write(src + i * bytes, dst + i * bytes, bytes);
    }
}

// The same call Noc::async_write<TXN_ID>(DataflowBuffer, ...) makes for an implicit-sync DFB write.
inline void issue_trid_writes(uint32_t src, uint64_t dst, uint32_t n, uint32_t bytes) {
    for (uint32_t i = 0; i < n; i++) {
        ncrisc_noc_fast_write_any_len<noc_mode, true, false>(
            noc_index,
            write_cmd_buf,
            src + i * bytes,
            dst + i * bytes,
            bytes,
            NOC_UNICAST_WRITE_VC,
            /*mcast*/ false,
            /*linked*/ false,
            /*num_dests*/ 1,
            /*multicast_path_reserve*/ true,
            /*posted*/ false,
            FIRST_TRID + (i % NUM_TRIDS));
    }
}

inline void barrier_trids() {
    for (uint32_t t = 0; t < NUM_TRIDS; t++) {
        noc_async_write_barrier_with_trid(FIRST_TRID + t);
    }
}

inline void drain_untimed() {
    while (outstanding_all() != 0) {
    }
    for (uint32_t t = 0; t < NUM_TRIDS; t++) {
        while (outstanding_trid(FIRST_TRID + t) != 0) {
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t scenario = get_arg(args::scenario);
    const uint32_t num_writes = get_arg(args::num_writes);
    const uint32_t write_bytes = get_arg(args::write_bytes);
    const uint32_t iters = get_arg(args::iters);
    const uint32_t src_addr = get_arg(args::src_addr);
    const uint64_t dst = get_noc_addr(get_arg(args::dst_x), get_arg(args::dst_y), get_arg(args::dst_addr));
    volatile uint32_t* result =
        reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(get_arg(args::result_addr)) + MEM_L1_UNCACHED_BASE);

    const bool needs_streamer = scenario == SCENARIO_COUPLED_IDLE || scenario == SCENARIO_CROSS_HART_TRID;
    if (needs_streamer) {
        while (result[FLAG_STREAMER_STARTED] == 0) {
        }
    }

    uint32_t seq = 0;
    for (uint32_t v = 0; v < NUM_VARIANTS; v++) {
        uint32_t mn = 0xFFFFFFFFu, mx = 0, sum = 0, leaked = 0, max_seen = 0;
        for (uint32_t it = 0; it < iters; it++) {
            seq++;
            bool wrote = false;
            if (scenario == SCENARIO_EXPLICIT_WRITES) {
                issue_explicit_writes(src_addr, dst, num_writes, write_bytes);
                wrote = true;
            } else if (scenario == SCENARIO_TRID_WRITES) {
                issue_trid_writes(src_addr, dst, num_writes, write_bytes);
                wrote = true;
            } else if (scenario == SCENARIO_CROSS_HART_TRID) {
                // Wait for the streamer's next burst, then see whether its writes show up in this DM's
                // per-txn-id counter.
                while (result[FLAG_ISSUED_SEQ] != seq) {
                }
            }
            const uint32_t seen = scenario == SCENARIO_CROSS_HART_TRID ? outstanding_trid(FIRST_TRID) : outstanding_all();
            max_seen = seen > max_seen ? seen : max_seen;

            const uint32_t t0 = cycles();
            switch (v) {
                case V_NONE: break;
                case V_CONDITIONAL:
                    if (wrote) {
                        noc_async_write_barrier();
                    }
                    break;
                case V_FULL: noc_async_write_barrier(); break;
                case V_TRID: barrier_trids(); break;
                default: break;
            }
            const uint32_t dt = cycles() - t0;

            // Writes this DM is responsible for that were still in flight after the barrier.
            uint32_t left = 0;
            if (scenario == SCENARIO_EXPLICIT_WRITES || scenario == SCENARIO_TRID_WRITES) {
                left = outstanding_all();
                for (uint32_t t = 0; t < NUM_TRIDS; t++) {
                    left += outstanding_trid(FIRST_TRID + t);
                }
            } else if (scenario == SCENARIO_CROSS_HART_TRID) {
                left = outstanding_trid(FIRST_TRID);
            }
            leaked += left != 0 ? 1u : 0u;

            mn = dt < mn ? dt : mn;
            mx = dt > mx ? dt : mx;
            sum += dt;
            if (scenario == SCENARIO_CROSS_HART_TRID) {
                result[FLAG_CONSUMED_SEQ] = seq;
            } else if (scenario != SCENARIO_COUPLED_IDLE) {
                drain_untimed();
            }
        }
        volatile uint32_t* r = result + v * WORDS_PER_VARIANT;
        r[R_MIN] = mn;
        r[R_MAX] = mx;
        r[R_SUM] = sum;
        r[R_ITERS] = iters;
        r[R_LEAKED] = leaked;
        r[R_MAX_OUTSTANDING] = max_seen;
    }
    result[FLAG_MEASURER_DONE] = 0xFFFFFFFFu;
}
