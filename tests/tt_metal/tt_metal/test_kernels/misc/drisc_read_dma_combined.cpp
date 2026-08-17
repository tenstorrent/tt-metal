// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Combined DRISC drainer leg: NoC-read whole cores out of worker L1, then DMA the batch to this
// DRISC's own GDDR bank. This is the configuration the DRAM-buffer design needs, and the one place
// where the two legs might interfere.
//
// A DRISC cannot land NoC traffic straight in GDDR -- in stream mode (required to initiate reads at
// all) NoC traffic terminates at L1, and DRAM is only reachable through the L1 + DMA path. So every
// byte crosses L1 twice: written by the NIU, read by the DMA engine. A buffer crossed twice can only
// sustain half its own bandwidth, hence this test.
//
// Structure: two L1 batch buffers ping-ponged against the two DMA TX streams, so the DMA of batch N
// overlaps the NoC reads of batch N+1.
//
//   loop:
//     dma_async_write_barrier(cur)      wait for the DMA that last used buffer[cur]
//     NoC-read kCoresPerBatch cores into buffer[cur], one barrier
//     dma_async_write(cur, buffer[cur] -> GDDR ring)
//     cur ^= 1
//
// kDoRead / kDoDma compile out either leg so the same batching and buffer layout can be measured
// three ways -- read only, DMA only, both -- making the difference attributable to the interaction
// rather than to a change in access pattern.

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
    constexpr uint32_t kBytesPerCore = get_compile_time_arg_val(1);  // 10240 = whole core, 5 rings
    constexpr uint32_t kBufBase = get_compile_time_arg_val(2);
    constexpr uint32_t kResultsAddr = get_compile_time_arg_val(3);
    constexpr uint32_t kDoRead = get_compile_time_arg_val(4);
    constexpr uint32_t kDoDma = get_compile_time_arg_val(5);
    constexpr uint32_t kGddrRingBytes = get_compile_time_arg_val(6);
    // 2 = ping-pong (DMA of batch N overlaps the reads of N+1, but read depth is capped at
    // kCoresPerBatch by the L1 split). 1 = one buffer, so reads and DMA serialize but the whole
    // budget goes to a single deeper batch. Which wins is a measurement, not an argument.
    constexpr uint32_t kNumBuffers = get_compile_time_arg_val(7);
    static_assert(kNumBuffers == 1 || kNumBuffers == 2, "one or two buffers");

    constexpr uint32_t kBatchBytes = kCoresPerBatch * kBytesPerCore;
    static_assert(kBytesPerCore <= NOC_MAX_BURST_SIZE, "per-core read must fit one NoC packet");
    static_assert((kBatchBytes & 0xF) == 0, "GDDR DMA transfers must be 16 B multiples");
    static_assert(kBatchBytes <= 262128u, "GDDR DMA transfer size field is 14 bits of words");

    const uint32_t num_cores = get_arg_val<uint32_t>(0);
    const uint32_t iters = get_arg_val<uint32_t>(1);
    const uint32_t src_addr = get_arg_val<uint32_t>(2);
    const uint64_t gddr_base = (static_cast<uint64_t>(get_arg_val<uint32_t>(4)) << 32) | get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t* coords = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(5));

    Noc noc;
    UnicastEndpoint src;

    experimental::drisc_set_stream_mode();

    uint64_t t_dma_wait = 0;
    uint64_t t_read_issue = 0;
    uint64_t t_read_wait = 0;
    uint64_t t_dma_issue = 0;
    uint32_t cur = 0;
    uint64_t gddr_off = 0;

    const uint64_t t_start = get_timestamp();
    for (uint32_t iter = 0; iter < iters; iter++) {
        uint32_t core = 0;
        while (core < num_cores) {
            const uint32_t base = kBufBase + cur * kBatchBytes;

            const uint64_t d0 = get_timestamp();
            if constexpr (kDoDma) {
                // Buffer[cur] is only safe to refill once its previous DMA has drained.
                experimental::dma_async_write_barrier(cur);
            }
            const uint64_t d1 = get_timestamp();

            uint32_t n = 0;
            if constexpr (kDoRead) {
                for (; n < kCoresPerBatch && core < num_cores; n++, core++) {
                    const uint32_t xy = coords[core];
                    CoreLocalMem<uint32_t> dst(base + n * kBytesPerCore);
                    noc.async_read<NocOptions::DEFAULT, kBytesPerCore>(
                        src, dst, kBytesPerCore, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = src_addr}, {});
                }
            } else {
                n = (num_cores - core) < kCoresPerBatch ? (num_cores - core) : kCoresPerBatch;
                core += n;
            }
            const uint64_t d2 = get_timestamp();

            if constexpr (kDoRead) {
                noc.async_read_barrier();
            }
            const uint64_t d3 = get_timestamp();

            if constexpr (kDoDma) {
                experimental::dma_async_write(cur, base, gddr_base + gddr_off, n * kBytesPerCore);
                gddr_off += n * kBytesPerCore;
                if (gddr_off + kBatchBytes > kGddrRingBytes) {
                    gddr_off = 0;
                }
            }
            const uint64_t d4 = get_timestamp();

            t_dma_wait += d1 - d0;
            t_read_issue += d2 - d1;
            t_read_wait += d3 - d2;
            t_dma_issue += d4 - d3;
            if constexpr (kNumBuffers == 2) {
                cur ^= 1;
            }
        }
    }
    if constexpr (kDoDma) {
        experimental::dma_async_write_barrier(0);
        experimental::dma_async_write_barrier(1);
    }
    const uint64_t t_end = get_timestamp();

    experimental::drisc_set_noc2axi_mode();

    uint32_t checksum = 0;
    for (uint32_t b = 0; b < kNumBuffers; b++) {
        checksum += *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kBufBase + b * kBatchBytes);
    }

    const uint64_t cycles = t_end - t_start;
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kResultsAddr);
    out[0] = static_cast<uint32_t>(cycles & 0xFFFFFFFFu);
    out[1] = static_cast<uint32_t>(cycles >> 32);
    out[2] = checksum;
    out[3] = num_cores * iters;
    out[4] = static_cast<uint32_t>(t_dma_wait & 0xFFFFFFFFu);
    out[5] = static_cast<uint32_t>(t_read_issue & 0xFFFFFFFFu);
    out[6] = static_cast<uint32_t>(t_read_wait & 0xFFFFFFFFu);
    out[7] = static_cast<uint32_t>(t_dma_issue & 0xFFFFFFFFu);
    out[8] = kCoresPerBatch;
}
