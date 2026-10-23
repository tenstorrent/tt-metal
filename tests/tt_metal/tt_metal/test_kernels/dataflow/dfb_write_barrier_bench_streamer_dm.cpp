// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Second DM for the write-barrier bench (see dfb_write_barrier_bench_common.h).
//   SCENARIO_COUPLED_IDLE:    stream plain writes until the measurer is done, so its idle barriers
//                             run alongside another DM's traffic.
//   SCENARIO_CROSS_HART_TRID: one txn-id burst per measured iteration, left in flight until the
//                             measurer has sampled and barriered on it.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "dev_mem_map.h"  // MEM_L1_UNCACHED_BASE
#include "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_write_barrier_bench_common.h"

using namespace dfb_wb_bench;

void kernel_main() {
    const uint32_t scenario = get_arg(args::scenario);
    const uint32_t num_writes = get_arg(args::num_writes);
    const uint32_t write_bytes = get_arg(args::write_bytes);
    const uint32_t iters = get_arg(args::iters);
    const uint32_t src_addr = get_arg(args::src_addr);
    const uint64_t dst = get_noc_addr(get_arg(args::dst_x), get_arg(args::dst_y), get_arg(args::dst_addr));
    volatile uint32_t* result =
        reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(get_arg(args::result_addr)) + MEM_L1_UNCACHED_BASE);

    if (scenario == SCENARIO_COUPLED_IDLE) {
        result[FLAG_STREAMER_STARTED] = 1;
        while (result[FLAG_MEASURER_DONE] == 0) {
            for (uint32_t i = 0; i < num_writes; i++) {
                noc_async_write(src_addr + i * write_bytes, dst + i * write_bytes, write_bytes);
            }
            noc_async_write_barrier();
        }
        return;
    }

    // SCENARIO_CROSS_HART_TRID
    result[FLAG_STREAMER_STARTED] = 1;
    uint32_t self_max = 0;
    const uint32_t total = iters * NUM_VARIANTS;
    for (uint32_t seq = 1; seq <= total; seq++) {
        for (uint32_t i = 0; i < num_writes; i++) {
            ncrisc_noc_fast_write_any_len<noc_mode, true, false>(
                noc_index,
                write_cmd_buf,
                src_addr + i * write_bytes,
                dst + i * write_bytes,
                write_bytes,
                NOC_UNICAST_WRITE_VC,
                /*mcast*/ false,
                /*linked*/ false,
                /*num_dests*/ 1,
                /*multicast_path_reserve*/ true,
                /*posted*/ false,
                FIRST_TRID);
        }
        const uint32_t seen = static_cast<uint32_t>(__builtin_riscv_ttrocc_scmdbuf_tr_ack_trid(FIRST_TRID));
        self_max = seen > self_max ? seen : self_max;
        result[FLAG_ISSUED_SEQ] = seq;
        while (result[FLAG_CONSUMED_SEQ] != seq) {
        }
        noc_async_write_barrier_with_trid(FIRST_TRID);
    }
    result[STREAMER_SELF_MAX_OUTSTANDING] = self_max;
    while (result[FLAG_MEASURER_DONE] == 0) {
    }
}
