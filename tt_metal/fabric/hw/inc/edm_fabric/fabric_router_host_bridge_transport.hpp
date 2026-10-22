// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-bridged link transport. Same interface as EthTxqLinkTransport, but every operation becomes one
// frame in a D2H socket ring on this core (hostdevcommon/fabric_host_bridge.h); the host relays the
// ring to the peer host, which replays the frames on the far router's core.
//
// The socket config is written by the host after the router starts, so the socket opens lazily once
// the host arms the block. Until then is_busy() is true and nothing is sent.
#pragma once

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "hostdevcommon/fabric_host_bridge.h"
#include "hostdev/socket.h"
#include "api/alignment.h"

#include <cstdint>

// NOC / cmd buf are chosen by the builder so they never collide with the router's own stateful cmd bufs.
template <uint32_t SOCKET_CONFIG_ADDR, uint32_t STATUS_ADDR, uint32_t PAGE_BYTES, uint8_t NOC, uint8_t CMD_BUF>
struct HostBridgeLinkTransport {
    using Header = tt::tt_fabric::host_bridge::FrameHeader;
    static constexpr uint32_t header_scratch_addr = SOCKET_CONFIG_ADDR + tt::tt_fabric::host_bridge::kSocketConfigBytes;
    // Same layout D2HSocket writes and create_sender_socket_interface() reads: md, bytes_acked, d2h encoding.
    static constexpr uint32_t md_bytes = align(sizeof(sender_socket_md), L1_ALIGNMENT);
    static constexpr uint32_t bytes_acked_addr = SOCKET_CONFIG_ADDR + md_bytes;
    static constexpr uint32_t d2h_md_addr = bytes_acked_addr + align(sizeof(uint32_t), L1_ALIGNMENT);

    // Socket cursors, cached off L1 once opened.
    static inline bool opened = false;
    static inline uint32_t write_ptr = 0;
    static inline uint32_t bytes_sent = 0;
    static inline uint32_t fifo_total_size = 0;
    static inline uint32_t fifo_curr_size = 0;  // page-aligned part of the ring
    static inline uint32_t fifo_addr_lo = 0;
    static inline uint32_t fifo_addr_hi = 0;
    static inline uint32_t bytes_sent_addr_lo = 0;
    static inline uint32_t bytes_sent_addr_hi = 0;
    static inline uint32_t pcie_xy_enc = 0;
    static inline uint32_t seq = 0;

    static FORCE_INLINE volatile tt_l1_ptr tt::tt_fabric::host_bridge::Status* status() {
        return reinterpret_cast<volatile tt_l1_ptr tt::tt_fabric::host_bridge::Status*>(STATUS_ADDR);
    }
    static constexpr bool is_tx = (SOCKET_CONFIG_ADDR - STATUS_ADDR) == tt::tt_fabric::host_bridge::kTxSocketOffset;
    static FORCE_INLINE void note_opened() {
        if constexpr (is_tx) {
            status()->tx_opened = 1;
        } else {
            status()->rx_opened = 1;
        }
    }
    static FORCE_INLINE void note_unarmed() {
        if constexpr (is_tx) {
            status()->tx_unarmed++;
        } else {
            status()->rx_unarmed++;
        }
    }
    static FORCE_INLINE void note_ring_full() {
        if constexpr (is_tx) {
            status()->tx_ring_full++;
        } else {
            status()->rx_ring_full++;
        }
    }

    static FORCE_INLINE bool ensure_open() {
        if (opened) {
            return true;
        }
        invalidate_l1_cache();
        if (status()->armed != tt::tt_fabric::host_bridge::kArmed) {
            note_unarmed();
            return false;
        }
        volatile tt_l1_ptr sender_socket_md* md =
            reinterpret_cast<volatile tt_l1_ptr sender_socket_md*>(SOCKET_CONFIG_ADDR);
        volatile tt_l1_ptr d2h_sender_socket_md* d2h =
            reinterpret_cast<volatile tt_l1_ptr d2h_sender_socket_md*>(d2h_md_addr);
        ASSERT(md->is_d2h);
        write_ptr = md->write_ptr;
        bytes_sent = md->bytes_sent;
        fifo_total_size = md->downstream_fifo_total_size;
        fifo_curr_size = fifo_total_size - fifo_total_size % PAGE_BYTES;
        fifo_addr_lo = md->downstream_fifo_addr;
        fifo_addr_hi = d2h->data_addr_hi;
        bytes_sent_addr_lo = md->downstream_bytes_sent_addr;
        bytes_sent_addr_hi = d2h->bytes_sent_addr_hi;
        pcie_xy_enc = d2h->pcie_xy_enc;
        ASSERT(PAGE_BYTES <= fifo_curr_size);
        opened = true;
        note_opened();
        return true;
    }

    // Non-blocking socket_reserve_pages(1).
    static FORCE_INLINE bool slot_free() {
        invalidate_l1_cache();
        const uint32_t acked = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bytes_acked_addr);
        return fifo_total_size - (bytes_sent - acked) >= PAGE_BYTES;
    }

    static FORCE_INLINE bool is_busy() {
        if (!ensure_open()) {
            return true;
        }
        if (!slot_free()) {
            note_ring_full();
            return true;
        }
        return false;
    }

    // L1 -> host, NOC_MAX_BURST_SIZE at a time, on this transport's own cmd buf.
    static FORCE_INLINE void write_to_host(uint32_t src_l1, uint64_t dst_host, uint32_t size) {
        while (size) {
            const uint32_t chunk = size > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : size;
            // The router's own NOC counters are left alone (update_counter=false): flush_writes() waits
            // on the hardware counters instead.
            noc_wwrite_with_state<noc_mode, CMD_BUF, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, false, false>(
                NOC, src_l1, pcie_xy_enc, dst_host, chunk, 1);
            src_l1 += chunk;
            dst_host += chunk;
            size -= chunk;
        }
    }

    // Every nonposted write this RISC issued on NOC has been acked. Hardware counters only: the
    // receiver RISC's software counters do not cover its transaction-id writes, so the generic
    // noc_async_write_barrier() would never return there.
    static FORCE_INLINE void flush_writes() {
        while (NOC_STATUS_READ_REG(NOC, NIU_MST_NONPOSTED_WR_REQ_SENT) !=
               NOC_STATUS_READ_REG(NOC, NIU_MST_WR_ACK_RECEIVED)) {
        }
    }

    // Assumes !is_busy(). Header then payload to the ring slot, flush, then bytes_sent to the host.
    static FORCE_INLINE void write_frame(
        uint32_t kind, uint32_t addr, uint32_t value, uint32_t src_l1, uint32_t bytes) {
        const uint64_t slot = ((static_cast<uint64_t>(fifo_addr_hi) << 32) | fifo_addr_lo) + write_ptr;
        volatile tt_l1_ptr Header* h = reinterpret_cast<volatile tt_l1_ptr Header*>(header_scratch_addr);
        h->kind = kind;
        h->addr = addr;
        h->value = value;
        h->seq = ++seq;

        noc_write_init_state<CMD_BUF>(NOC, NOC_UNICAST_WRITE_VC);
        write_to_host(header_scratch_addr, slot, sizeof(Header));
        if (bytes != 0) {
            write_to_host(
                src_l1,
                slot + tt::tt_fabric::host_bridge::kFrameHeaderBytes,
                tt::tt_fabric::host_bridge::align_up(bytes, NOC_PCIE_WRITE_ALIGNMENT_BYTES));
        }
        flush_writes();

        // socket_push_pages(1)
        if (write_ptr + PAGE_BYTES >= fifo_curr_size) {
            write_ptr = write_ptr + PAGE_BYTES - fifo_curr_size;
            bytes_sent += PAGE_BYTES + fifo_total_size - fifo_curr_size;
        } else {
            write_ptr += PAGE_BYTES;
            bytes_sent += PAGE_BYTES;
        }
        // socket_notify_receiver() on our own cmd buf, and the cursors back to L1 for the host to read.
        volatile tt_l1_ptr sender_socket_md* md =
            reinterpret_cast<volatile tt_l1_ptr sender_socket_md*>(SOCKET_CONFIG_ADDR);
        md->bytes_sent = bytes_sent;
        md->write_ptr = write_ptr;
        const uint64_t bytes_sent_host = (static_cast<uint64_t>(bytes_sent_addr_hi) << 32) | bytes_sent_addr_lo;
        write_to_host(SOCKET_CONFIG_ADDR, bytes_sent_host, sizeof(uint32_t));
        noc_async_write_clear_pcie_state(NOC, CMD_BUF);
    }

    // Assumes !is_busy()
    static FORCE_INLINE void send_packet(uint32_t src_addr, uint32_t dest_addr, uint32_t num_bytes) {
        write_frame(tt::tt_fabric::host_bridge::PACKET, dest_addr, num_bytes, src_addr, num_bytes);
        status()->packets++;
    }

    // Assumes !is_busy()
    template <uint32_t stream_id>
    static FORCE_INLINE void remote_update_ptr_val(int32_t val) {
        remote_update_ptr_val(stream_id, val);
    }

    // Assumes !is_busy()
    static FORCE_INLINE void remote_update_ptr_val(uint32_t stream_id, int32_t val) {
        write_frame(
            tt::tt_fabric::host_bridge::REG_WRITE,
            get_stream_reg_write_addr(static_cast<uint8_t>(stream_id)),
            static_cast<uint32_t>(val << REMOTE_DEST_BUF_WORDS_FREE_INC),
            0,
            0);
        status()->reg_writes++;
    }
};
