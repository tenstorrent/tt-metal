// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Phase 1 kernel: H2D PCIe → L1 (double-buffered) → DRAM
//
// This is the inverted analogue of dram_stream_sender.cpp.
// That kernel does DRAM → L1 → PCIe (D2H).
// This kernel does PCIe → L1 → DRAM (H2D) using the same double-buffer pattern.
//
// ── Data path ─────────────────────────────────────────────────────────────
//
//   Host pinned buf ──[noc_read_with_state / PCIe NOC read]──► L1 buf[cur^1]
//                              L1 buf[cur] ──[noc_async_write]──► DRAM (interleaved)
//
// PCIe reads use read_cmd_buf; DRAM writes use write_cmd_buf.
// Both run concurrently on separate NOC command buffers, hiding PCIe read
// latency (typically 300–600 ns) behind DRAM write latency (~100 ns).
//
// ── Flow control ──────────────────────────────────────────────────────────
//
// H2D socket (DEVICE_PULL mode): host writes data to a pinned ring buffer and
// increments bytes_sent in device L1 via TLB write.  The kernel spins on
// bytes_sent in socket_wait_for_pages, then issues the PCIe read directly into
// its own L1 ping-pong buffer (bypassing the socket's L1 FIFO — the FIFO
// ring pointers are kept in sync for correct flow control arithmetic only).
// After each chunk the kernel writes bytes_acked back to host pinned memory
// via socket_notify_sender so the host can advance its write pointer.
//
// ── Root-cause note (mirrors dram_stream_sender.cpp) ──────────────────────
//
// InterleavedAddrGen<true> is used for DRAM address generation because the
// firmware-global num_dram_banks correctly reflects harvested channels.
// Using a host-provided bank count (get_num_dram_channels()) can diverge after
// harvesting and produce garbage NOC addresses → noc_async_write_barrier hang.
//
// ── Compile-time args ─────────────────────────────────────────────────────
//   0: socket_config_addr  — H2D receiver socket config struct in L1
//   1: l1_buf_a_addr       — ping buffer (chunk_size bytes)
//   2: l1_buf_b_addr       — pong buffer (chunk_size bytes)
//   3: chunk_size          — bytes per chunk; PCIe-aligned, ≤ 16 KB (NOC_MAX_BURST_SIZE)
//   4: total_bytes         — total bytes to pull; must be a multiple of chunk_size
//
// ── Runtime args ──────────────────────────────────────────────────────────
//   0: dram_base_addr — per-bank base address of the destination DRAM buffer
//
// ── DPRINT checkpoints ────────────────────────────────────────────────────
//   [A] Kernel entry — prints socket metadata.  Zeros in pcie_xy / data_hi
//       indicate the socket config addr overlaps an L1 ping-pong buffer.
//   [B] First PCIe read completed — confirms NOC PCIe read path is live.
//   [C] Per-chunk counter (first 3 chunks only) — confirms loop progress.
//   [E] Kernel done.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/socket_api.h"
#include "internal/dataflow/dataflow_api_addrgen.h"

void kernel_main() {
    // ── Compile-time parameters ────────────────────────────────────────────
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t l1_buf_a_addr = get_compile_time_arg_val(1);
    constexpr uint32_t l1_buf_b_addr = get_compile_time_arg_val(2);
    constexpr uint32_t chunk_size = get_compile_time_arg_val(3);
    constexpr uint32_t total_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t num_chunks = total_bytes / chunk_size;

    // ── Runtime parameters ─────────────────────────────────────────────────
    const uint32_t dram_base_addr = get_arg_val<uint32_t>(0);

    // ── H2D socket (receiver side) ─────────────────────────────────────────
    SocketReceiverInterface receiver = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver, chunk_size);

    // PCIe tile NOC XY encoding (bit-60 set for PCIe routing on BH).
    const uint32_t pcie_xy_enc = receiver.h2d.pcie_xy_enc;

    // Full 64-bit physical address of the host-side pinned ring buffer.
    // The NOC PCIe read source is: pcie_fifo_base + (read_ptr - fifo_addr).
    // read_ptr - fifo_addr gives the byte offset within the ring, matching
    // the offset at which the host wrote the data via H2DSocket::write().
    const uint64_t pcie_fifo_base =
        (static_cast<uint64_t>(receiver.h2d.data_addr_hi) << 32) | static_cast<uint64_t>(receiver.h2d.data_addr_lo);

    DPRINT << "[A] pcie_to_dram_puller started" << ENDL();
    DPRINT << "  cfg=0x" << HEX() << socket_config_addr << " buf_a=0x" << HEX() << l1_buf_a_addr << " buf_b=0x" << HEX()
           << l1_buf_b_addr << ENDL();
    DPRINT << "  pcie_xy=0x" << HEX() << pcie_xy_enc << " data_hi=0x" << HEX() << receiver.h2d.data_addr_hi << ENDL();
    DPRINT << "  dram=0x" << HEX() << dram_base_addr << " chunks=" << DEC() << num_chunks << " csz=" << DEC()
           << chunk_size << ENDL();

    // ── DRAM address generator ─────────────────────────────────────────────
    // InterleavedAddrGen<true> uses the firmware-global num_dram_banks, which
    // accounts for harvested channels.  Do NOT derive bank count from the host.
    InterleavedAddrGen<true> dram_gen = {
        .bank_base_address = dram_base_addr,
        .page_size = chunk_size,
    };

    // ── Ping-pong state ────────────────────────────────────────────────────
    // buf[cur]   = chunk ready to drain to DRAM (was fetched in the previous step)
    // buf[cur^1] = idle buffer, destination for the next PCIe read
    const uint32_t buf[2] = {l1_buf_a_addr, l1_buf_b_addr};
    uint32_t cur = 0;

    // ── Prime: fetch chunk 0 into buf[0] ────────────────────────────────────
    // No DRAM write yet — just get the pipeline started.
    socket_wait_for_pages(receiver, 1);
    noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
        NOC_INDEX, pcie_xy_enc, pcie_fifo_base + (receiver.read_ptr - receiver.fifo_addr), buf[0], chunk_size);
    noc_async_read_barrier();
    // If execution never reaches [B], noc_async_read_barrier hung:
    //   • pcie_xy_enc is wrong (socket config addr overlaps L1 bufs)
    //   • host never issued notify_receiver (bytes_sent not advanced)
    //   • PCIe NOC address encoding wrong for this arch
    socket_pop_pages(receiver, 1);
    socket_notify_sender(receiver);  // ACK chunk 0 to host; also arms write_cmd_buf

    DPRINT << "[B] first PCIe read OK" << ENDL();

    // ── Double-buffered streaming loop ─────────────────────────────────────
    for (uint32_t c = 1; c < num_chunks; ++c) {
        if (c <= 3) {
            DPRINT << "[C] c=" << DEC() << c << ENDL();
        }

        // (A) Issue PCIe read for chunk c into the idle buffer.
        //     noc_read_with_state uses read_cmd_buf — fully independent of
        //     the DRAM write below which uses write_cmd_buf.
        socket_wait_for_pages(receiver, 1);
        noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
            NOC_INDEX,
            pcie_xy_enc,
            pcie_fifo_base + (receiver.read_ptr - receiver.fifo_addr),
            buf[cur ^ 1],
            chunk_size);

        // (B) Issue DRAM write for chunk c-1 (the buffer filled in the
        //     previous iteration or the priming step).
        //     noc_async_write uses write_cmd_buf — runs concurrently with (A).
        noc_async_write(buf[cur], dram_gen.get_noc_addr(c - 1), chunk_size);

        // (C) Wait for both to complete before flipping buffers.
        //     Barriers are independent: read barrier ≠ write barrier on BH.
        noc_async_read_barrier();   // chunk c landed in buf[cur^1]
        noc_async_write_barrier();  // chunk c-1 committed to DRAM

        // (D) ACK chunk c to host so it can refill that slot in the ring.
        //     socket_notify_sender writes bytes_acked to host pinned memory via
        //     a non-blocking NOC PCIe write.  The write completes well before
        //     the next socket_wait_for_pages returns (host memcpy + TLB write
        //     takes ~0.5 µs, longer than the notify PCIe write latency).
        socket_pop_pages(receiver, 1);
        socket_notify_sender(receiver);

        cur ^= 1;
    }

    // ── Drain: write the last fetched chunk (buf[cur]) to DRAM ───────────────
    // After the loop, buf[cur] holds chunk num_chunks-1 which has not yet been
    // written to DRAM (the loop body writes chunk c-1 for iteration c, so the
    // final chunk is drained here).
    noc_async_write(buf[cur], dram_gen.get_noc_addr(num_chunks - 1), chunk_size);
    noc_async_write_barrier();

    // Persist updated bytes_acked and read_ptr back to the L1 socket config struct.
    update_socket_config(receiver);

    DPRINT << "[E] done" << ENDL();
}
