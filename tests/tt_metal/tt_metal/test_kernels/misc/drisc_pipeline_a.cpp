// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Pipeline leg A: sweep the whole worker grid into a DRAM ping/pong buffer.
//
// Per sweep, A walks all cores in batches of kCoresPerBatch, ping-ponging two L1 buffers against the
// two DMA TX streams (the geometry measured fastest: DMA of batch N hides behind the reads of N+1).
// A full sweep lands in one of two DRAM buffers, alternating each round.
//
// Flow control with B is a 2-credit SPSC over the two DRAM buffers:
//   A publishes fill_idx  (sweeps completed) into B's L1 after finishing a sweep
//   B publishes drain_idx (sweeps drained)   into A's L1 after emptying one
//   A blocks while fill_idx - drain_idx >= 2, i.e. both DRAM buffers are outstanding
// Both DRISCs are in stream mode, so inbound NoC traffic terminates in L1 and plain local addresses
// are the correct targets for those flag writes.
//
// t_block is the whole point of the measurement: it is how long A stalls because B has not kept up.

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "experimental/drisc_mode.h"
#include "experimental/gddr_dma.h"
#include "internal/tt-1xx/risc_common.h"

void kernel_main() {
    constexpr uint32_t kCoresPerBatch = get_compile_time_arg_val(0);
    constexpr uint32_t kBytesPerCore = get_compile_time_arg_val(1);
    constexpr uint32_t kL1BufBase = get_compile_time_arg_val(2);
    constexpr uint32_t kResultsAddr = get_compile_time_arg_val(3);
    constexpr uint32_t kDrainFlagAddr = get_compile_time_arg_val(4);  // in A's L1, written by B
    constexpr uint32_t kSweepBytes = get_compile_time_arg_val(5);

    constexpr uint32_t kBatchBytes = kCoresPerBatch * kBytesPerCore;
    static_assert(kBytesPerCore <= NOC_MAX_BURST_SIZE, "per-core read must fit one NoC packet");
    static_assert((kBatchBytes & 0xF) == 0, "GDDR DMA transfers must be 16 B multiples");

    const uint32_t num_cores = get_arg_val<uint32_t>(0);
    const uint32_t num_sweeps = get_arg_val<uint32_t>(1);
    const uint32_t src_addr = get_arg_val<uint32_t>(2);
    const uint64_t dram_base = (static_cast<uint64_t>(get_arg_val<uint32_t>(4)) << 32) | get_arg_val<uint32_t>(3);
    const uint32_t b_xy = get_arg_val<uint32_t>(5);
    const uint32_t b_fill_flag_addr = get_arg_val<uint32_t>(6);
    volatile tt_l1_ptr uint32_t* coords = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(7));

    Noc noc;
    UnicastEndpoint src;

    experimental::drisc_set_stream_mode();

    volatile tt_l1_ptr uint32_t* drain_flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDrainFlagAddr);
    *drain_flag = 0;
    const uint64_t b_flag_noc_addr = get_noc_addr(b_xy & 0xFFFFu, b_xy >> 16, b_fill_flag_addr);

    uint64_t t_block = 0;
    uint32_t fill_idx = 0;
    uint32_t cur = 0;

    const uint64_t t_start = get_timestamp();
    for (uint32_t sweep = 0; sweep < num_sweeps; sweep++) {
        // Both DRAM buffers outstanding -> wait for B.
        const uint64_t b0 = get_timestamp();
        while ((fill_idx - *drain_flag) >= 2) {
            invalidate_l1_cache();
        }
        t_block += get_timestamp() - b0;

        const uint64_t dram_off = static_cast<uint64_t>(sweep & 1u) * kSweepBytes;
        uint32_t batch_off = 0;
        uint32_t core = 0;
        while (core < num_cores) {
            const uint32_t base = kL1BufBase + cur * kBatchBytes;
            experimental::dma_async_write_barrier(cur);

            uint32_t n = 0;
            for (; n < kCoresPerBatch && core < num_cores; n++, core++) {
                const uint32_t xy = coords[core];
                CoreLocalMem<uint32_t> dst(base + n * kBytesPerCore);
                noc.async_read<NocOptions::DEFAULT, kBytesPerCore>(
                    src, dst, kBytesPerCore, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = src_addr}, {});
            }
            noc.async_read_barrier();

            experimental::dma_async_write(cur, base, dram_base + dram_off + batch_off, n * kBytesPerCore);
            batch_off += n * kBytesPerCore;
            cur ^= 1;
        }
        // The sweep is not complete until both streams have drained into DRAM.
        experimental::dma_async_write_barrier(0);
        experimental::dma_async_write_barrier(1);

        fill_idx++;
        noc_inline_dw_write(b_flag_noc_addr, fill_idx);
    }
    const uint64_t t_end = get_timestamp();

    experimental::drisc_set_noc2axi_mode();

    const uint64_t cycles = t_end - t_start;
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kResultsAddr);
    out[0] = static_cast<uint32_t>(cycles & 0xFFFFFFFFu);
    out[1] = static_cast<uint32_t>(cycles >> 32);
    out[2] = static_cast<uint32_t>(t_block & 0xFFFFFFFFu);
    out[3] = static_cast<uint32_t>(t_block >> 32);
    out[4] = fill_idx;
    out[5] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kL1BufBase);  // liveness
}
