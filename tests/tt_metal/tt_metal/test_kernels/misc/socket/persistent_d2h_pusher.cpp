// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Persistent D2H kernel — runs on BRISC (RISCV_0, NOC0)
//
// Data path per command:
//   DRAM[dram_page_start + c] ──[noc_async_read]──► L1 ping/pong
//   L1 ping/pong ──[noc_wwrite_with_state → PCIe]──► host pinned memory
//
// Control flow:
//   1. Zero the D2H command ring at startup (L1 state is undefined on launch).
//   2. Spin on slot[0] (seq field) — local L1 read, zero cost.
//   3. When seq changes: read command fields, execute double-buffered transfer.
//   4. Write completion_seq to L1.
//   5. Advance ring slot (mod kDmaRingDepth), loop.
//   6. On DMA_OP_EXIT: break.
//
// Compile-time args:
//   0: pcie_xy_enc — PCIe tile NOC XY encoding (fixed for session)
//
// All L1 addresses and ring geometry come from persistent_dma_cmd.hpp.

#include "persistent_dma_cmd.hpp"

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "internal/dataflow/dataflow_api_addrgen.h"

void kernel_main() {
    const uint32_t pcie_xy_enc = get_arg_val<uint32_t>(0);

    const uint32_t buf[2] = {kDmaD2HPingBase, kDmaD2HPongBase};
    uint32_t last_seq = 0u;
    uint32_t slot_idx = 0u;
    uint32_t cmd_count = 0u;

    volatile tt_l1_ptr uint32_t* progress = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDmaD2HProgress);
    progress[0] = 0u;  // 0 = idle / spinning

    volatile tt_l1_ptr uint32_t* diag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDmaD2HDiagBase);

    // ── Signal ready — after local state init ──────────────────────────────
    volatile tt_l1_ptr uint32_t* ready_flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDmaD2HReadyFlag);
    ready_flag[0] = 1u;

    // DPRINT omitted — DPRINT buffer blocks when no server is attached, which
    // stalls the kernel and causes host-side done-flag timeouts.

    while (true) {
        // ── Spin on seq field of current command slot ─────────────────────
        volatile tt_l1_ptr uint32_t* slot =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDmaD2HCmdBase + slot_idx * kDmaSlotBytes);

        // Spin until the host writes a new command to this slot.
        // Use serial-number comparison: any seq <= last_seq (including 0 in an
        // uninitialized / not-yet-filled slot) is treated as "not ready".
        // The old `slot[0] == last_seq` was wrong: after processing seq=1, moving
        // to an uninitialized slot (seq=0) would immediately exit the spin because
        // 0 != 1, causing the kernel to process a garbage all-zero command.
        while ((int32_t)(slot[0] - last_seq) <= 0) { /* spin — local L1 read, no NOC */
        }

        // ── Read command fields ───────────────────────────────────────────
        const uint32_t new_seq = slot[0];
        const uint32_t opcode = slot[1];
        const uint32_t dram_base = slot[2];   // DRAM bank_base_address
        const uint32_t page_start = slot[3];  // first interleaved page index
        const uint32_t dst_lo = slot[4];      // host phys addr [31:0]
        const uint32_t dst_hi = slot[5];      // host phys addr [63:32]
        const uint32_t num_chunks = slot[6];
        const uint32_t done_flag_lo = slot[7];  // host done-flag addr [31:0]
        const uint32_t done_flag_hi = slot[8];  // host done-flag addr [63:32]

        last_seq = new_seq;
        slot_idx = (slot_idx + 1u) & (kDmaRingDepth - 1u);

        if (opcode == DMA_OP_EXIT) {
            break;
        }

        progress[0] = 1u;  // 1 = cmd_received

        const uint64_t pcie_dst = (static_cast<uint64_t>(dst_hi) << 32) | dst_lo;
        const uint64_t done_flag_phys = (static_cast<uint64_t>(done_flag_hi) << 32) | done_flag_lo;
        const bool l1_only = (opcode == DMA_OP_L1_ONLY);

        if (!l1_only) {
            // ── DRAM → L1 → PCIe path ────────────────────────────────────
            InterleavedAddrGen<true> dram_gen = {
                .bank_base_address = dram_base,
                .page_size = kDmaChunkSize,
            };

            noc_async_read(dram_gen.get_noc_addr(page_start), buf[0], kDmaChunkSize);
            noc_async_read_barrier();
            progress[0] = 2u;

            uint32_t cur = 0u;
            for (uint32_t c = 1u; c < num_chunks; ++c) {
                noc_async_read(dram_gen.get_noc_addr(page_start + c), buf[cur ^ 1u], kDmaChunkSize);

                noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
                noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
                    NOC_INDEX,
                    buf[cur],
                    pcie_xy_enc,
                    pcie_dst + static_cast<uint64_t>(c - 1u) * kDmaChunkSize,
                    kDmaChunkSize,
                    1);

                noc_async_read_barrier();
                noc_async_write_barrier();
                cur ^= 1u;
            }
            progress[0] = 3u;

            noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
            noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
                NOC_INDEX,
                buf[cur],
                pcie_xy_enc,
                pcie_dst + static_cast<uint64_t>(num_chunks - 1u) * kDmaChunkSize,
                kDmaChunkSize,
                1);
            noc_async_write_barrier();
        } else {
            // ── L1-only → PCIe path (skip DRAM, pure PCIe write BW) ──────
            // Ping buffer already contains data from prior transfer (or garbage).
            // Push the same L1 buffer for every chunk — data content doesn't matter
            // for bandwidth measurement; host verifies done-flag only.
            progress[0] = 2u;

            for (uint32_t c = 0u; c < num_chunks; ++c) {
                noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
                noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
                    NOC_INDEX,
                    buf[0],
                    pcie_xy_enc,
                    pcie_dst + static_cast<uint64_t>(c) * kDmaChunkSize,
                    kDmaChunkSize,
                    1);
                noc_async_write_barrier();
            }
            progress[0] = 3u;
        }
        progress[0] = 4u;

        // ── Signal host: write last_seq to HOST done-flag via PCIe ──────
        volatile tt_l1_ptr uint32_t* l1_staging = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf[0]);
        l1_staging[0] = last_seq;

        volatile tt_l1_ptr uint32_t* completion_seq =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDmaD2HCompletionSeq);
        completion_seq[0] = last_seq;

        cmd_count++;

        noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
        noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            NOC_INDEX, buf[0], pcie_xy_enc, done_flag_phys, sizeof(uint32_t), 1);
        noc_async_write_barrier();

        diag[0] = done_flag_lo;
        diag[1] = done_flag_hi;
        diag[2] = last_seq;
        diag[3] = cmd_count;
        diag[4] = noc_nonposted_writes_acked[NOC_INDEX];
        diag[5] = NOC_STATUS_READ_REG(NOC_INDEX, NIU_MST_WR_ACK_RECEIVED);
        diag[6] = noc_nonposted_writes_num_issued[NOC_INDEX];
        diag[7] = NOC_STATUS_READ_REG(NOC_INDEX, NIU_MST_NONPOSTED_WR_REQ_SENT);

        progress[0] = 5u;  // 5 = done_flag written to host
    }
}
