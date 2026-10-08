// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Device side: places one [packet | BridgeDescriptor] frame into a D2H socket slot in host
// memory. Nothing here is fabric-aware, so it runs on an ETH core or a Tensix proxy alike.
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "hostdevcommon/erisc_host_bridge.h"

namespace tt::tt_fabric {

// L1 scratch the caller must reserve for the descriptor staged ahead of its PCIe write.
inline constexpr std::uint32_t kBridgeDescScratchBytes = kBridgeDescriptorBytes;

// The guard sits at offset 24 and 24 % 16 != 0, so it cannot be written alone. The tail carries
// `elapsed` with it to land 16 B aligned -- which is why 32 B is the descriptor's floor.
inline constexpr std::uint32_t kBridgeDescTailBytes = 16;
inline constexpr std::uint32_t kBridgeDescHeadBytes = kBridgeDescriptorBytes - kBridgeDescTailBytes;
static_assert(kBridgeDescHeadBytes % NOC_PCIE_WRITE_ALIGNMENT_BYTES == 0, "head transfer must be PCIe aligned");
static_assert(kBridgeDescTailBytes % NOC_PCIE_WRITE_ALIGNMENT_BYTES == 0, "tail transfer must be PCIe aligned");
// Only the guard may carry PROTOCOL meaning in the tail: the two words share a transfer and so
// have no order between them. elapsed is instrumentation, read only after the guard validates.
static_assert(offsetof(BridgeDescriptor, guard) >= kBridgeDescHeadBytes, "guard must land in the tail");

// Transfer sizes as well as addresses carry the PCIe write alignment.
constexpr std::uint32_t bridge_align_up(std::uint32_t v, std::uint32_t a) { return (v + a - 1) / a * a; }

// L1 -> PCIe in NOC_MAX_BURST_SIZE chunks. Local copy of the idiom in the socket tests'
// pcie_noc_utils.h so this header carries no dependency on test code.
inline void bridge_noc_write_chunked(
    uint8_t noc, uint32_t pcie_xy_enc, uint32_t src_l1, uint64_t dst_pcie, uint32_t size) {
    noc_write_init_state<write_cmd_buf>(noc, NOC_UNICAST_WRITE_VC);
    while (size) {
        uint32_t chunk = size > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : size;
        noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            noc, src_l1, pcie_xy_enc, dst_pcie, chunk, 1);
        src_l1 += chunk;
        dst_pcie += chunk;
        size -= chunk;
    }
}

// Everything the sender needs, held on the stack. The socket's own cursors live inside and are
// written back to L1 on every frame, so a caller that stops mid-loop leaves no stale config.
struct EriscHostBridgeSender {
    SocketSenderInterface socket;
    std::uint32_t capacity;   // packet bytes per slot; descriptor sits directly after
    std::uint32_t desc_addr;  // L1 scratch, kBridgeDescScratchBytes
    // Absolute packets sent, never a delta, stamped into both ordering_cntr and the guard's seq.
    // open() runs once per router lifetime, so it never restarts in flight.
    std::uint32_t seq;
};

// Page size is fixed by the wire format: a socket page and a bridge slot are the same object.
// The host must have configured this socket's FIFO page to bridge_socket_page_bytes(capacity).
inline EriscHostBridgeSender erisc_host_bridge_open(
    std::uint32_t socket_config_addr, std::uint32_t capacity, std::uint32_t desc_scratch_addr) {
    EriscHostBridgeSender b;
    b.socket = create_sender_socket_interface(socket_config_addr);
    // The descriptor is written as its own transfer at slot+capacity, so capacity carries the
    // alignment for both halves of the frame.
    ASSERT(b.socket.is_d2h);
    // PCIE_ALIGNMENT, not L1: D2HSocket refuses a page that is not a multiple of it, and the
    // descriptor's two transfers inherit their alignment from capacity.
    ASSERT(capacity % PCIE_ALIGNMENT == 0);
    ASSERT(desc_scratch_addr % L1_ALIGNMENT == 0);
    ASSERT(bridge_socket_page_bytes(capacity) <= b.socket.downstream_fifo_total_size);
    set_sender_socket_page_size(b.socket, bridge_socket_page_bytes(capacity));
    b.capacity = capacity;
    b.desc_addr = desc_scratch_addr;
    b.seq = 0;
    return b;
}

// The proper host memory location for the next frame: the D2H FIFO base, which is a 64-bit host
// address split across the socket config, plus the socket's own relative write cursor.
inline std::uint64_t erisc_host_bridge_slot_addr(const EriscHostBridgeSender& b) {
    return ((static_cast<std::uint64_t>(b.socket.d2h.data_addr_hi) << 32) | b.socket.downstream_fifo_addr) +
           b.socket.write_ptr;
}

// Non-blocking form of socket_reserve_pages. The bridge is a fail-safe on the router's own
// core: it must be able to decline a frame rather than spin while the host is behind.
inline bool erisc_host_bridge_slot_free(const EriscHostBridgeSender& b, std::uint32_t num_pages = 1) {
    const std::uint32_t num_bytes = num_pages * b.socket.page_size;
    std::uint32_t addr = b.socket.bytes_acked_base_addr;
    const std::uint32_t end = addr + b.socket.num_downstreams * bytes_acked_size_bytes;
    invalidate_l1_cache();
    while (addr < end) {
        const std::uint32_t acked = *reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(addr);
        // bytes_acked never runs ahead of bytes_sent, so the difference is safe unsigned.
        if (b.socket.downstream_fifo_total_size - (b.socket.bytes_sent - acked) < num_bytes) {
            return false;
        }
        addr += bytes_acked_size_bytes;
    }
    return true;
}

// Blocking reserve, for a caller that owns its core and would rather wait than drop.
inline void erisc_host_bridge_reserve(const EriscHostBridgeSender& b, std::uint32_t num_pages = 1) {
    socket_reserve_pages(b.socket, num_pages);
}

// SPLIT_GUARD writes the guard in its own 16 B tail transfer after the rest of the descriptor.
// A single 32 B transfer has no defined internal order, so the host could see the guard early.
template <bool SPLIT_GUARD = true>
inline void erisc_host_bridge_write_frame(
    EriscHostBridgeSender& b,
    std::uint32_t src_l1_addr,
    std::uint32_t length,
    std::uint16_t slot_idx,
    std::uint32_t stall_cycles = 0,
    std::uint8_t noc = noc_index) {
    ASSERT(length != 0 && length <= b.capacity);
    const std::uint32_t pcie_xy_enc = b.socket.d2h.pcie_xy_enc;
    const std::uint64_t slot = erisc_host_bridge_slot_addr(b);
    const std::uint64_t t0 = get_timestamp();

    // Payload first, and a barrier before anything that validates it. Rounded up because the
    // transfer size is aligned too; the slack is inside the slot and length bounds the reader.
    bridge_noc_write_chunked(
        noc, pcie_xy_enc, src_l1_addr, slot, bridge_align_up(length, NOC_PCIE_WRITE_ALIGNMENT_BYTES));
    noc_async_write_barrier();

    b.seq++;
    volatile tt_l1_ptr BridgeDescriptor* d = reinterpret_cast<volatile tt_l1_ptr BridgeDescriptor*>(b.desc_addr);
    d->length = length;
    d->ordering_cntr = b.seq;  // ordering key; the guard carries the same value as its lap tag
    d->slot_idx = slot_idx;
    d->reserved0 = 0;  // v1 must zero the alignment hole or a later version reads garbage
    // RELEASE, not the write's start: t0 was taken before the payload transfer, and what the
    // host's arrival pairs with is the moment this frame became visible to it.
    d->release_cyc = static_cast<std::uint32_t>(get_timestamp());
    // Low half the payload write and its barrier, high half the stall the caller measured.
    d->elapsed = bridge_elapsed_pack(get_timestamp() - t0, stall_cycles);
    const std::uint64_t armed = bridge_guard(kBridgeVersion, b.seq);
    d->guard = SPLIT_GUARD ? 0 : armed;

    const std::uint64_t desc = slot + bridge_desc_offset_in_slot(b.capacity);
    bridge_noc_write_chunked(
        noc, pcie_xy_enc, b.desc_addr, desc, SPLIT_GUARD ? kBridgeDescHeadBytes : kBridgeDescriptorBytes);
    noc_async_write_barrier();

    if constexpr (SPLIT_GUARD) {
        d->guard = armed;
        bridge_noc_write_chunked(
            noc, pcie_xy_enc, b.desc_addr + kBridgeDescHeadBytes, desc + kBridgeDescHeadBytes, kBridgeDescTailBytes);
        noc_async_write_barrier();
    }

    socket_push_pages(b.socket, 1);
    socket_notify_receiver(b.socket, noc);
    // Cursors were cached on the stack; write them back or the host reads a stale write_ptr.
    update_socket_config(b.socket);
}

// Declines instead of stalling when the host has not drained. Returns false having touched
// nothing, so the caller can fall back to the cable or retry on a later poll.
template <bool SPLIT_GUARD = true>
inline bool erisc_host_bridge_try_write_frame(
    EriscHostBridgeSender& b,
    std::uint32_t src_l1_addr,
    std::uint32_t length,
    std::uint16_t slot_idx,
    std::uint32_t stall_cycles = 0,
    std::uint8_t noc = noc_index) {
    if (!erisc_host_bridge_slot_free(b, 1)) {
        return false;
    }
    erisc_host_bridge_write_frame<SPLIT_GUARD>(b, src_l1_addr, length, slot_idx, stall_cycles, noc);
    return true;
}

// Waits for the host to ack every frame sent. Only for a caller whose host reader runs
// concurrently -- against a host that reads after the workload completes this deadlocks.
inline void erisc_host_bridge_drain(const EriscHostBridgeSender& b) { socket_barrier(b.socket); }

}  // namespace tt::tt_fabric
