// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Streaming kernel: DRAM (interleaved) → L1 (double-buffered) → PCIe host pinned memory
//
// ──────────────────────────────────────────────────────────────────────────────
// ROOT CAUSE OF HANG (fixed here)
// ──────────────────────────────────────────────────────────────────────────────
// The previous version passed num_dram_banks from the host via runtime arg and
// computed bank_id = chunk_id % host_num_dram_banks. This is WRONG because the
// device firmware populates a separate global `num_dram_banks` (reflecting actual
// harvesting state) that may differ from the host's get_num_dram_channels().
// A mismatched bank_id produces a garbage NOC address → noc_async_read_barrier()
// waits forever for a transaction that never completes.
//
// Fix: use InterleavedAddrGen<true> which reads the firmware-correct num_dram_banks
// internally and handles the bank → NOC coordinate mapping for any architecture.
//
// ──────────────────────────────────────────────────────────────────────────────
// Data path (double-buffered):
//   DRAM page[c] ──[noc_async_read via InterleavedAddrGen]──► L1 buf[cur^1]
//                                                              L1 buf[cur] ──[noc_wwrite]──► PCIe → host
//
// Compile-time args:
//   0: socket_config_addr  — D2H sender socket config in L1 (must not overlap L1 bufs!)
//   1: l1_buf_a_addr       — first  ping-pong L1 buffer
//   2: l1_buf_b_addr       — second ping-pong L1 buffer
//   3: chunk_size          — bytes per chunk (== DRAM page_size, PCIe-aligned)
//   4: total_bytes         — total bytes; must be multiple of chunk_size
//
// Runtime args:
//   0: dram_base_addr  — per-bank base address (dram_buf->address() from host)
//
// DPRINT checkpoints (set TT_METAL_DPRINT_CORES=0,0 before running):
//   [A] Kernel entry  — prints socket metadata; if pcie_xy_enc/data_addr_hi are
//                       0 or garbage, the socket config addr overlaps an L1 buf
//   [B] First DRAM read returned — confirms InterleavedAddrGen+noc_async_read work
//   [C/D] Per-chunk — shows which chunk and whether socket_reserve_pages stalls
//   [E] Done

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/socket_api.h"
#include "internal/dataflow/dataflow_api_addrgen.h"

void kernel_main() {
    // ---------- compile-time parameters ----------
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t l1_buf_a_addr = get_compile_time_arg_val(1);
    constexpr uint32_t l1_buf_b_addr = get_compile_time_arg_val(2);
    constexpr uint32_t chunk_size = get_compile_time_arg_val(3);
    constexpr uint32_t total_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t num_chunks = total_bytes / chunk_size;

    // ---------- runtime parameters ----------
    const uint32_t dram_base_addr = get_arg_val<uint32_t>(0);

    // ---------- socket setup ----------
    SocketSenderInterface sender = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender, chunk_size);

    const uint32_t pcie_xy_enc = sender.d2h.pcie_xy_enc;
    const uint32_t data_addr_hi = sender.d2h.data_addr_hi;

    // CHECKPOINT [A] — if pcie_xy_enc or data_addr_hi look wrong (0 or garbage)
    // the socket config buffer address overlaps with the L1 ping-pong buffers.
    // socket_config_addr must NOT be in [l1_buf_a_addr, l1_buf_b_addr + chunk_size).
    DPRINT << "[A] dram_stream_sender started" << ENDL();
    DPRINT << "  cfg=0x" << HEX() << socket_config_addr << " buf_a=0x" << HEX() << l1_buf_a_addr << " buf_b=0x" << HEX()
           << l1_buf_b_addr << ENDL();
    DPRINT << "  pcie_xy=0x" << HEX() << pcie_xy_enc << " addr_hi=0x" << HEX() << data_addr_hi << ENDL();
    DPRINT << "  dram_base=0x" << HEX() << dram_base_addr << " chunks=" << DEC() << num_chunks << " chunk_sz=" << DEC()
           << chunk_size << ENDL();

    // ---------- DRAM address generator ----------
    // InterleavedAddrGen<true> uses the device-side firmware global num_dram_banks,
    // which correctly reflects harvested channels. This is the authoritative source —
    // do NOT recompute bank_id manually from a host-provided bank count.
    InterleavedAddrGen<true> dram_gen = {
        .bank_base_address = dram_base_addr,
        .page_size = chunk_size,
    };

    // ---------- ping-pong buffers ----------
    const uint32_t buf[2] = {l1_buf_a_addr, l1_buf_b_addr};
    uint32_t cur = 0;

    // ---------- prefetch first chunk into buf[0] ----------
    noc_async_read(dram_gen.get_noc_addr(0), l1_buf_a_addr, chunk_size);
    noc_async_read_barrier();
    // If we never reach [B], the noc_async_read hung:
    //   • dram_base_addr is wrong (buffer not allocated/written before kernel launch)
    //   • InterleavedAddrGen returns an invalid NOC address on this SOC variant
    //   • NOC is congested / not initialised for this core
    DPRINT << "[B] first DRAM read OK" << ENDL();

    // Arm stateful PCIe write state once before the loop.
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);

    // ---------- main double-buffered streaming loop ----------
    for (uint32_t c = 1; c < num_chunks; ++c) {
        // (A) Prefetch next chunk into the idle buffer while we process the current one.
        noc_async_read(dram_gen.get_noc_addr(c), buf[cur ^ 1], chunk_size);

        // (B) Stall until the socket FIFO has space.
        if (c <= 3) {
            DPRINT << "[C] c=" << DEC() << c << " reserve..." << ENDL();
        }
        socket_reserve_pages(sender, 1);
        if (c <= 3) {
            DPRINT << "[D] c=" << DEC() << c << " wptr=0x" << HEX() << sender.write_ptr << ENDL();
        }

        // (C) Destination in host pinned FIFO.
        uint64_t dst = ((static_cast<uint64_t>(data_addr_hi) << 32) | sender.downstream_fifo_addr) + sender.write_ptr;
        if (c <= 3) {
            DPRINT << "[D1] dst_lo=0x" << HEX() << (uint32_t)dst << " fifo=0x" << HEX() << sender.downstream_fifo_addr
                   << ENDL();
        }

        // (D) Re-arm stateful write state after socket_reserve_pages.
        noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
        if (c <= 3) {
            DPRINT << "[D2] init_state done" << ENDL();
        }

        // (E) Stream buf[cur] → PCIe endpoint → host pinned memory.
        noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            NOC_INDEX, buf[cur], pcie_xy_enc, dst, chunk_size, 1);
        if (c <= 3) {
            DPRINT << "[D3] wwrite done" << ENDL();
        }

        // (F) Advance socket write pointer and notify host.
        socket_push_pages(sender, 1);
        if (c <= 3) {
            DPRINT << "[D4] push_pages done" << ENDL();
        }
        socket_notify_receiver(sender);
        if (c <= 3) {
            DPRINT << "[D5] notify done" << ENDL();
        }

        // (G) Wait for DRAM prefetch to land before flipping buffers.
        // Note: if [D3] printed but this hangs, the concurrent DRAM read issued in (A)
        // did not complete — the read and write may be contending for the same NOC cmd buf.
        noc_async_read_barrier();
        if (c <= 3) {
            DPRINT << "[D6] read_barrier done" << ENDL();
        }
        noc_async_writes_flushed();
        if (c <= 3) {
            DPRINT << "[D7] writes_flushed done" << ENDL();
        }

        cur ^= 1;
    }

    // ---------- flush the last prefetched chunk (buf[cur]) ----------
    socket_reserve_pages(sender, 1);
    {
        uint64_t dst = ((static_cast<uint64_t>(data_addr_hi) << 32) | sender.downstream_fifo_addr) + sender.write_ptr;
        noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
        noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            NOC_INDEX, buf[cur], pcie_xy_enc, dst, chunk_size, 1);
    }
    socket_push_pages(sender, 1);
    socket_notify_receiver(sender);

    noc_async_write_barrier();
    socket_barrier(sender);
    update_socket_config(sender);

    DPRINT << "[E] done" << ENDL();
}
