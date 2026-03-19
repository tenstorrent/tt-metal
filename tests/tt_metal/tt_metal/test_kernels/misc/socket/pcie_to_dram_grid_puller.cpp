// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Phase 2/3 grid kernel: PCIe → L1 (double-buffered) → DRAM, no socket flow control
//
// This is the grid-parallel counterpart to pcie_to_dram_puller.cpp (Phase 1).
// Phase 1 used the H2D socket protocol (socket_wait_for_pages / socket_notify_sender)
// for per-chunk flow control, which adds per-chunk PCIe round-trip overhead (~2 µs).
//
// This kernel eliminates all flow control:
//   • Host pre-fills the ENTIRE pinned buffer before launching kernels.
//   • Each core receives its slice of the buffer via runtime args and reads
//     directly — no wait, no ACK, no per-chunk synchronisation.
//   • Completion is signalled by writing a uint32_t done-flag to a second
//     pinned region in host memory.
//
// This makes each core's bottleneck pure PCIe read latency with no protocol
// overhead.  With N cores all issuing reads simultaneously, the aggregate
// outstanding data easily exceeds the bandwidth-delay product (31 KB at 500 ns
// RTT), keeping the PCIe link saturated.
//
// ── Data path ─────────────────────────────────────────────────────────────
//
//   Host pinned buf [pcie_src .. pcie_src+per_core_bytes]
//       ──[noc_read_with_state, PCIe tile]──► L1 buf[cur^1]
//                   L1 buf[cur] ──[noc_async_write]──► DRAM[dram_page_start+c-1]
//   (end) noc_wwrite_with_state ──► done_flag in host pinned memory
//
// ── Compile-time args ─────────────────────────────────────────────────────
//   0: l1_buf_a_addr  — ping buffer (chunk_size bytes)
//   1: l1_buf_b_addr  — pong buffer (chunk_size bytes)
//   2: chunk_size     — bytes per chunk; PCIe-aligned, ≤ 16 KB
//
// ── Runtime args (differ per core) ────────────────────────────────────────
//   0: pcie_xy_enc     — PCIe tile NOC XY encoding from PinnedMemory::NocAddr
//   1: pcie_src_lo     — this core's host physical address, bits [31:0]
//   2: pcie_src_hi     — bits [63:32]
//   3: num_chunks      — chunks for this core (= per_core_bytes / chunk_size)
//   4: dram_base_addr  — InterleavedAddrGen bank_base_address (same on all cores)
//   5: dram_page_start — first page index in the global interleaved DRAM layout
//   6: done_flag_lo    — host done-flag physical address, bits [31:0]
//   7: done_flag_hi    — bits [63:32]
//
// ── DPRINT checkpoints ────────────────────────────────────────────────────
//   [A] Kernel entry   — prints all key addresses
//   [B] Prime read OK  — first PCIe read landed
//   [C] Loop progress  — first 3 iterations only
//   [E] Done flag sent

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "internal/dataflow/dataflow_api_addrgen.h"

void kernel_main() {
    // ── Compile-time parameters ────────────────────────────────────────────
    constexpr uint32_t l1_buf_a_addr = get_compile_time_arg_val(0);
    constexpr uint32_t l1_buf_b_addr = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_size = get_compile_time_arg_val(2);

    // ── Runtime parameters (per core) ──────────────────────────────────────
    const uint32_t pcie_xy_enc = get_arg_val<uint32_t>(0);
    const uint32_t pcie_src_lo = get_arg_val<uint32_t>(1);
    const uint32_t pcie_src_hi = get_arg_val<uint32_t>(2);
    const uint32_t num_chunks = get_arg_val<uint32_t>(3);
    const uint32_t dram_base_addr = get_arg_val<uint32_t>(4);
    const uint32_t dram_page_start = get_arg_val<uint32_t>(5);
    const uint32_t done_flag_lo = get_arg_val<uint32_t>(6);
    const uint32_t done_flag_hi = get_arg_val<uint32_t>(7);

    const uint64_t pcie_src = (static_cast<uint64_t>(pcie_src_hi) << 32) | static_cast<uint64_t>(pcie_src_lo);
    const uint64_t done_flag_phys = (static_cast<uint64_t>(done_flag_hi) << 32) | static_cast<uint64_t>(done_flag_lo);

    DPRINT << "[A] grid_puller" << ENDL();
    DPRINT << "  pcie_xy=0x" << HEX() << pcie_xy_enc << " src_hi=0x" << HEX() << pcie_src_hi << ENDL();
    DPRINT << "  chunks=" << DEC() << num_chunks << " pg_start=" << DEC() << dram_page_start << " csz=" << DEC()
           << chunk_size << ENDL();

    // ── DRAM address generator ─────────────────────────────────────────────
    InterleavedAddrGen<true> dram_gen = {
        .bank_base_address = dram_base_addr,
        .page_size = chunk_size,
    };

    // ── Ping-pong buffers ─────────────────────────────────────────────────
    const uint32_t buf[2] = {l1_buf_a_addr, l1_buf_b_addr};
    uint32_t cur = 0;

    // ── Prime: fetch chunk 0 into buf[0] ────────────────────────────────────
    // No flow control wait — host has pre-filled the entire buffer.
    noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
        NOC_INDEX,
        pcie_xy_enc,
        pcie_src,  // host physical address of chunk 0 for this core
        buf[0],
        chunk_size);
    noc_async_read_barrier();
    DPRINT << "[B] prime OK" << ENDL();

    // ── Double-buffered streaming loop ─────────────────────────────────────
    for (uint32_t c = 1; c < num_chunks; ++c) {
        if (c <= 3) {
            DPRINT << "[C] c=" << DEC() << c << ENDL();
        }

        // (A) PCIe read chunk c into idle buffer — uses read_cmd_buf.
        noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
            NOC_INDEX, pcie_xy_enc, pcie_src + static_cast<uint64_t>(c) * chunk_size, buf[cur ^ 1], chunk_size);

        // (B) DRAM write chunk c-1 — uses write_cmd_buf, runs concurrently with (A).
        noc_async_write(buf[cur], dram_gen.get_noc_addr(dram_page_start + c - 1), chunk_size);

        // (C) Wait for both.
        noc_async_read_barrier();
        noc_async_write_barrier();

        cur ^= 1;
    }

    // ── Drain: write the last chunk to DRAM ──────────────────────────────
    noc_async_write(buf[cur], dram_gen.get_noc_addr(dram_page_start + num_chunks - 1), chunk_size);
    noc_async_write_barrier();

    // ── Signal host: write 1 to done_flag in host pinned memory ─────────────
    // Reuse buf[cur] (ping-pong buffers are no longer needed) as a 4-byte
    // staging area for the value 1.  Then write it to host via PCIe NOC write.
    volatile tt_l1_ptr uint32_t* l1_staging = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_buf_a_addr);
    l1_staging[0] = 1u;
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
    noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
        NOC_INDEX, l1_buf_a_addr, pcie_xy_enc, done_flag_phys, sizeof(uint32_t), 1);
    noc_async_write_barrier();

    DPRINT << "[E] done" << ENDL();
}
