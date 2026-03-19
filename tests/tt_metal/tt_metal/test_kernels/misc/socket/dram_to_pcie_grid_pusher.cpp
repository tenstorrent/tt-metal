// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// D2H grid kernel: DRAM → L1 (double-buffered) → host pinned memory
//
// Mirror of pcie_to_dram_grid_puller.cpp (H2D direction).
// Replaces the socket-based dram_stream_sender.cpp with a no-flow-control
// version: host pre-allocates the full destination buffer, device writes
// directly to host physical addresses via PCIe NOC writes (posted TLPs).
//
// Key difference from dram_stream_sender:
//   • No socket_reserve_pages / socket_notify_receiver — eliminates per-chunk
//     host round-trip overhead (~2 µs/chunk) that limited D2H to ~3 GB/s.
//   • Runtime args carry the destination physical address directly.
//   • Completion signalled by a done-flag PCIe write (same as H2D Phase 2/3).
//
// ── Data path ─────────────────────────────────────────────────────────────
//
//   DRAM[dram_page_start+c] ──[noc_async_read]──► L1 buf[cur^1]
//                     L1 buf[cur] ──[noc_wwrite_with_state → PCIe]──► host pinned
//
// noc_async_read uses read_cmd_buf; noc_wwrite_with_state uses write_cmd_buf.
// Both run concurrently, hiding DRAM read latency (~100 ns) behind PCIe write
// latency (~200–500 ns).
//
// ── Write ordering ────────────────────────────────────────────────────────
//
// PCIe posted writes from the same NOC endpoint are strongly ordered (PCIe
// spec §2.4.1).  The done-flag write is issued after noc_async_write_barrier()
// on the last data write, so the host observing done_flag==1 guarantees all
// preceding data writes have been committed to host memory.
//
// ── Compile-time args ─────────────────────────────────────────────────────
//   0: l1_buf_a_addr  — ping buffer (chunk_size bytes)
//   1: l1_buf_b_addr  — pong buffer (chunk_size bytes)
//   2: chunk_size     — bytes per chunk; ≤ NOC_MAX_BURST_SIZE (16 KB on BH)
//
// ── Runtime args (per core) ───────────────────────────────────────────────
//   0: pcie_xy_enc     — PCIe tile NOC XY encoding (from PinnedMemory::NocAddr)
//   1: pcie_dst_lo     — host destination physical address, bits [31:0]
//   2: pcie_dst_hi     — bits [63:32]
//   3: num_chunks      — chunks for this core
//   4: dram_base_addr  — InterleavedAddrGen bank_base_address
//   5: dram_page_start — first page index in the global interleaved layout
//   6: done_flag_lo    — host done-flag physical address, bits [31:0]
//   7: done_flag_hi    — bits [63:32]
//
// ── DPRINT checkpoints ────────────────────────────────────────────────────
//   [A] Kernel entry
//   [B] First DRAM read landed
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
    const uint32_t pcie_dst_lo = get_arg_val<uint32_t>(1);
    const uint32_t pcie_dst_hi = get_arg_val<uint32_t>(2);
    const uint32_t num_chunks = get_arg_val<uint32_t>(3);
    const uint32_t dram_base_addr = get_arg_val<uint32_t>(4);
    const uint32_t dram_page_start = get_arg_val<uint32_t>(5);
    const uint32_t done_flag_lo = get_arg_val<uint32_t>(6);
    const uint32_t done_flag_hi = get_arg_val<uint32_t>(7);

    const uint64_t pcie_dst = (static_cast<uint64_t>(pcie_dst_hi) << 32) | static_cast<uint64_t>(pcie_dst_lo);
    const uint64_t done_flag_phys = (static_cast<uint64_t>(done_flag_hi) << 32) | static_cast<uint64_t>(done_flag_lo);

    DPRINT << "[A] dram_to_pcie_pusher" << ENDL();
    DPRINT << "  pcie_xy=0x" << HEX() << pcie_xy_enc << " dst_hi=0x" << HEX() << pcie_dst_hi << ENDL();
    DPRINT << "  chunks=" << DEC() << num_chunks << " pg_start=" << DEC() << dram_page_start << " csz=" << DEC()
           << chunk_size << ENDL();

    // ── DRAM address generator ─────────────────────────────────────────────
    // InterleavedAddrGen<true> uses firmware-global num_dram_banks (correct
    // after harvesting).  Never derive bank count from the host.
    InterleavedAddrGen<true> dram_gen = {
        .bank_base_address = dram_base_addr,
        .page_size = chunk_size,
    };

    // ── Ping-pong buffers ─────────────────────────────────────────────────
    const uint32_t buf[2] = {l1_buf_a_addr, l1_buf_b_addr};
    uint32_t cur = 0;

    // ── Prime: fetch chunk 0 from DRAM into buf[0] ───────────────────────
    noc_async_read(dram_gen.get_noc_addr(dram_page_start), l1_buf_a_addr, chunk_size);
    noc_async_read_barrier();
    DPRINT << "[B] prime read OK" << ENDL();

    // ── Double-buffered streaming loop ────────────────────────────────────
    for (uint32_t c = 1; c < num_chunks; ++c) {
        if (c <= 3) {
            DPRINT << "[C] c=" << DEC() << c << ENDL();
        }

        // (A) DRAM read chunk c into idle buffer (read_cmd_buf, runs concurrently).
        noc_async_read(dram_gen.get_noc_addr(dram_page_start + c), buf[cur ^ 1], chunk_size);

        // (B) PCIe write chunk c-1 to host pinned memory (write_cmd_buf).
        //     noc_write_init_state must be re-armed before each noc_wwrite_with_state
        //     (mirrors dram_stream_sender.cpp line 124 — "re-arm after reserve").
        noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
        noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            NOC_INDEX, buf[cur], pcie_xy_enc, pcie_dst + static_cast<uint64_t>(c - 1) * chunk_size, chunk_size, 1);

        // (C) Wait for DRAM read to land; check PCIe write was issued to NOC.
        //     noc_async_writes_flushed() (not write_barrier) allows the PCIe
        //     write to be in-flight while the next DRAM read begins.
        noc_async_read_barrier();
        noc_async_writes_flushed();

        cur ^= 1;
    }

    // ── Drain: send last chunk to host ────────────────────────────────────
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
    noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
        NOC_INDEX, buf[cur], pcie_xy_enc, pcie_dst + static_cast<uint64_t>(num_chunks - 1) * chunk_size, chunk_size, 1);

    // Full write barrier: ensures ALL data is committed to host memory before
    // the done flag.  PCIe strong ordering then guarantees host sees data
    // before the done flag.
    noc_async_write_barrier();

    // ── Signal host: write 1 to done_flag in host pinned memory ──────────
    volatile tt_l1_ptr uint32_t* l1_staging = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_buf_a_addr);
    l1_staging[0] = 1u;
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
    noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
        NOC_INDEX, l1_buf_a_addr, pcie_xy_enc, done_flag_phys, sizeof(uint32_t), 1);
    noc_async_write_barrier();

    DPRINT << "[E] done" << ENDL();
}
