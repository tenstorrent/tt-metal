// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Persistent H2D kernel — runs on NCRISC (RISCV_1, NOC1)
//
// Data path per command (DRAM mode):
//   host pinned mem ──[noc_read_with_state × batch]──► L1 buf[0..7]
//   L1 buf[0..7]    ──[noc_async_write × batch]──►    DRAM pages
//
// H2D PCIe reads are non-posted (request→completion round-trip), so
// throughput depends on keeping many requests in the PCIe pipeline
// simultaneously.  noc_async_read_barrier() is global ("wait for ALL
// outstanding reads"), so we batch kDmaH2DNumBufs=8 reads into
// separate L1 buffers, barrier once, then drain all 8 to DRAM.
//
// Pipeline depth: 8 × 16 KB = 128 KB per core in-flight.
// With 16 cores: 2 MB total in the PCIe read pipeline.
//
// Compile-time args:
//   0: pcie_xy_enc — PCIe tile NOC XY encoding (fixed for session)

#include "persistent_dma_cmd.hpp"

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "internal/dataflow/dataflow_api_addrgen.h"

void kernel_main() {
    const uint32_t pcie_xy_enc = get_arg_val<uint32_t>(0);

    // Build buffer address table from the 8-buffer pool
    uint32_t bufs[kDmaH2DNumBufs];
    for (uint32_t i = 0; i < kDmaH2DNumBufs; ++i) {
        bufs[i] = kDmaH2DBufStart + i * kDmaChunkSize;
    }

    uint32_t last_seq = 0u;
    uint32_t slot_idx = 0u;

    volatile tt_l1_ptr uint32_t* progress = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDmaH2DProgress);
    progress[0] = 0u;

    volatile tt_l1_ptr uint32_t* ready_flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDmaH2DReadyFlag);
    ready_flag[0] = 1u;

    while (true) {
        volatile tt_l1_ptr uint32_t* slot =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDmaH2DCmdBase + slot_idx * kDmaSlotBytes);

        while ((int32_t)(slot[0] - last_seq) <= 0) { /* spin */
        }

        const uint32_t new_seq = slot[0];
        const uint32_t opcode = slot[1];
        const uint32_t src_lo = slot[2];
        const uint32_t src_hi = slot[3];
        const uint32_t dram_base = slot[4];
        const uint32_t page_start = slot[5];
        const uint32_t num_chunks = slot[6];

        last_seq = new_seq;
        slot_idx = (slot_idx + 1u) & (kDmaRingDepth - 1u);

        if (opcode == DMA_OP_EXIT) {
            break;
        }

        progress[0] = 1u;

        const uint64_t pcie_src = (static_cast<uint64_t>(src_hi) << 32) | src_lo;
        const uint32_t B = kDmaH2DNumBufs;  // batch size = 8

        if (opcode == DMA_OP_H2D_PUSH || opcode == DMA_OP_H2D_PUSH_L1) {
            // ── Host-push H2D: host writes data to L1 via BAR, kernel optionally drains to DRAM ──
            const bool skip_dram = (opcode == DMA_OP_H2D_PUSH_L1);

            volatile tt_l1_ptr uint32_t* push_seq_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDmaH2DPushSeq);
            volatile tt_l1_ptr uint32_t* push_ack_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDmaH2DPushAck);

            InterleavedAddrGen<true> dram_gen = {
                .bank_base_address = dram_base,
                .page_size = kDmaChunkSize,
            };

            uint32_t last_push_seq = push_seq_ptr[0];
            progress[0] = 2u;

            for (uint32_t base = 0u; base < num_chunks; base += B) {
                uint32_t batch_end = base + B;
                if (batch_end > num_chunks) {
                    batch_end = num_chunks;
                }
                uint32_t batch_count = batch_end - base;

                uint32_t expected_push = last_push_seq + 1u;
                while ((int32_t)(push_seq_ptr[0] - expected_push) < 0) { /* spin on L1 */
                }
                last_push_seq = expected_push;

                if (!skip_dram) {
                    for (uint32_t i = 0u; i < batch_count; ++i) {
                        noc_async_write(bufs[i], dram_gen.get_noc_addr(page_start + base + i), kDmaChunkSize);
                    }
                    noc_async_write_barrier();
                }

                push_ack_ptr[0] = last_push_seq;
            }

            progress[0] = 3u;
        } else if (opcode == DMA_OP_TRANSFER || opcode == DMA_OP_L1_ONLY) {
            // ── Device-pull H2D: 4+4 double-buffered PCIe reads ──────────
            // Split 8 L1 buffers into two groups of 4.  While group A's
            // data drains to DRAM, group B fills from PCIe — overlapping
            // the two independent NOC transactions for higher throughput.
            const bool skip_dram = (opcode == DMA_OP_L1_ONLY);

            InterleavedAddrGen<true> dram_gen = {
                .bank_base_address = dram_base,
                .page_size = kDmaChunkSize,
            };

            const uint32_t HALF = B / 2;  // 4 bufs per group
            uint32_t group = 0;

            // Prime: read first batch into group 0 (bufs[0..3])
            uint32_t prime_count = (num_chunks < HALF) ? num_chunks : HALF;
            for (uint32_t i = 0; i < prime_count; ++i) {
                noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
                    NOC_INDEX,
                    pcie_xy_enc,
                    pcie_src + static_cast<uint64_t>(i) * kDmaChunkSize,
                    bufs[i],
                    kDmaChunkSize);
            }
            noc_async_read_barrier();
            progress[0] = 2u;

            uint32_t prev_count = prime_count;
            uint32_t prev_page_base = 0;

            for (uint32_t base = HALF; base < num_chunks; base += HALF) {
                uint32_t prev_group = group;
                group ^= 1;

                uint32_t rd_count = ((num_chunks - base) < HALF) ? (num_chunks - base) : HALF;

                // PCIe reads into new group (uses read_cmd_buf)
                for (uint32_t i = 0; i < rd_count; ++i) {
                    noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
                        NOC_INDEX,
                        pcie_xy_enc,
                        pcie_src + static_cast<uint64_t>(base + i) * kDmaChunkSize,
                        bufs[group * HALF + i],
                        kDmaChunkSize);
                }

                // DRAM writes from previous group (uses write_cmd_buf, concurrent)
                if (!skip_dram) {
                    for (uint32_t i = 0; i < prev_count; ++i) {
                        noc_async_write(
                            bufs[prev_group * HALF + i],
                            dram_gen.get_noc_addr(page_start + prev_page_base + i),
                            kDmaChunkSize);
                    }
                }

                noc_async_read_barrier();
                if (!skip_dram) {
                    noc_async_write_barrier();
                }

                prev_count = rd_count;
                prev_page_base = base;
            }

            // Drain: write the last group to DRAM
            if (!skip_dram) {
                for (uint32_t i = 0; i < prev_count; ++i) {
                    noc_async_write(
                        bufs[group * HALF + i], dram_gen.get_noc_addr(page_start + prev_page_base + i), kDmaChunkSize);
                }
                noc_async_write_barrier();
            }

            progress[0] = 3u;
        }
        progress[0] = 4u;

        volatile tt_l1_ptr uint32_t* completion_seq =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDmaH2DCompletionSeq);
        completion_seq[0] = last_seq;
        progress[0] = 5u;
    }
}
