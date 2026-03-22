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

        volatile tt_l1_ptr uint32_t* h2d_diag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDmaH2DDiagBase);

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
            // ── Device-pull H2D: streaming circular buffer with per-read HW tracking ──
            //
            // Replaces the old 4+4 double-buffer with a B-deep circular ring.
            // Uses NIU_MST_RD_RESP_RECEIVED (increments by 1 per noc_read_with_state
            // completion) for fine-grained tracking instead of noc_async_read_barrier().
            // Benefits: all B buffers in flight at once; no pipeline bubble between
            // batches; DRAM writes overlap with PCIe reads naturally.
            const bool skip_dram = (opcode == DMA_OP_L1_ONLY);

            InterleavedAddrGen<true> dram_gen = {
                .bank_base_address = dram_base,
                .page_size = kDmaChunkSize,
            };

            uint32_t acc_rd_issue = 0;
            uint32_t acc_rd_barrier = 0;
            uint32_t acc_wr_issue = 0;
            uint32_t acc_wr_barrier = 0;
            uint32_t rd_waits = 0;

            uint32_t t_xfer_start = get_timestamp_32b();

            uint32_t rd_resp_base = NOC_STATUS_READ_REG(NOC_INDEX, NIU_MST_RD_RESP_RECEIVED);
            uint32_t issued = 0;
            uint32_t drained = 0;

            // Prime: fill all B buffers to maximize in-flight PCIe reads.
            uint32_t prime = (num_chunks < B) ? num_chunks : B;
            uint32_t t0 = get_timestamp_32b();
            for (uint32_t i = 0; i < prime; i++) {
                noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
                    NOC_INDEX,
                    pcie_xy_enc,
                    pcie_src + static_cast<uint64_t>(i) * kDmaChunkSize,
                    bufs[i],
                    kDmaChunkSize);
            }
            issued = prime;
            uint32_t t1 = get_timestamp_32b();
            acc_rd_issue += (t1 - t0);
            progress[0] = 2u;

            // Steady-state: drain each read as it completes, reissue into freed buffer.
            while (drained < num_chunks) {
                t0 = get_timestamp_32b();
                uint32_t target = rd_resp_base + drained + 1;
                while ((int32_t)(NOC_STATUS_READ_REG(NOC_INDEX, NIU_MST_RD_RESP_RECEIVED) - target) < 0) {
                }
                t1 = get_timestamp_32b();
                acc_rd_barrier += (t1 - t0);
                rd_waits++;

                uint32_t buf_idx = drained % B;

                if (!skip_dram) {
                    uint32_t tw0 = get_timestamp_32b();
                    noc_async_write(bufs[buf_idx], dram_gen.get_noc_addr(page_start + drained), kDmaChunkSize);
                    uint32_t tw1 = get_timestamp_32b();
                    acc_wr_issue += (tw1 - tw0);
                }

                drained++;

                if (issued < num_chunks) {
                    uint32_t tr0 = get_timestamp_32b();
                    noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
                        NOC_INDEX,
                        pcie_xy_enc,
                        pcie_src + static_cast<uint64_t>(issued) * kDmaChunkSize,
                        bufs[issued % B],
                        kDmaChunkSize);
                    uint32_t tr1 = get_timestamp_32b();
                    acc_rd_issue += (tr1 - tr0);
                    issued++;
                }
            }

            // Sync NOC state so next command starts clean.
            noc_async_read_barrier();
            if (!skip_dram) {
                t0 = get_timestamp_32b();
                noc_async_write_barrier();
                t1 = get_timestamp_32b();
                acc_wr_barrier += (t1 - t0);
            }

            uint32_t t_xfer_end = get_timestamp_32b();

            h2d_diag[0] = acc_rd_issue;
            h2d_diag[1] = acc_rd_barrier;
            h2d_diag[2] = acc_wr_issue;
            h2d_diag[3] = acc_wr_barrier;
            h2d_diag[4] = rd_waits;
            h2d_diag[5] = num_chunks;
            h2d_diag[6] = t_xfer_end - t_xfer_start;
            h2d_diag[7] = NOC_STATUS_READ_REG(NOC_INDEX, NIU_MST_RD_RESP_RECEIVED);

            progress[0] = 3u;
        }
        progress[0] = 4u;

        volatile tt_l1_ptr uint32_t* completion_seq =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDmaH2DCompletionSeq);
        completion_seq[0] = last_seq;
        progress[0] = 5u;
    }
}
