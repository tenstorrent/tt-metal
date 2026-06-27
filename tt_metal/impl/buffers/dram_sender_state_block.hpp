// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per-GCB "sender state block" that lives in DRISC L1 for DRAM-sender
// GlobalCircularBuffers. The GCB constructor pre-initializes one block per sender
// core so the long-running Tensor prefetcher kernel can switch between multiple
// GCBs across successive requests, resuming each GCB's ring-buffer write pointer
// (fifo_wr_ptr) where the previous request to that GCB left off.
//
// Each request the kernel loads these fields into its static cb_interface[] slot
// (a RemoteSenderCBInterface), runs the chunk loop, and writes fifo_wr_ptr back at
// request end. fifo_wr_ptr is the only field that must persist across requests to
// the same GCB; the rest are static config the host stamps once. The per-tensor
// fifo geometry (fifo_limit_page_aligned / fifo_page_size) is NOT stored here — the
// kernel recomputes it from config_ptr[3] (fifo size) + fifo_start_addr + the
// request's page size via resize_remote_sender_cb_interface before any use.
//
// Shared by host (composes the bytes) and the DRISC kernel (reads/writes via the
// struct). Keep it packed so the L1 byte layout is identical on both sides.

#pragma once

#include <cstddef>
#include <cstdint>

namespace tt::tt_metal {

struct DramSenderStateBlock {
    // ----- Fields the kernel loads into its RemoteSenderCBInterface (24 B) -----
    uint32_t config_ptr;              // -> the config block below (is_sender .. fifo_size_per_receiver)
    uint32_t fifo_start_addr;         // receiver buffer base
    uint32_t fifo_wr_ptr;             // persists across requests to this GCB
    uint32_t receiver_noc_xy_ptr;     // -> the receiver NOC XY table below
    uint32_t aligned_pages_sent_ptr;  // DRISC-side per-receiver pages_sent slot base
    uint32_t num_receivers_and_remote_pages_sent_ptr;  // packed; see remote_cb_pack()
    // ----- Sender config block, pointed to by config_ptr (20 B) -----
    // Read by the remote-CB kernel helpers (fifo_size_per_receiver at word [3]) and by
    // the prefetcher kernel (num_receivers).
    uint32_t is_sender;
    uint32_t num_receivers;
    uint32_t buffer_address;
    uint32_t fifo_size_per_receiver;
    // Bank-local slab index of this sender's first receiver. Lets two DRISC cores split
    // one bank's receiver set: the kernel reads slab (recv_index_base + r) for its local
    // receiver r. 0 for a single sender / the first of a pair.
    uint32_t recv_index_base;
    // ----- Followed in L1 by the receiver NOC XY table -----
    // 2 * num_receivers uint32s (x0, y0, x1, y1, ...), appended by the caller after
    // this struct's bytes since its length is dynamic; pointed to by receiver_noc_xy_ptr.
} __attribute__((packed));

static_assert(sizeof(DramSenderStateBlock) == 11 * sizeof(uint32_t), "DramSenderStateBlock layout drift");
static_assert(
    offsetof(DramSenderStateBlock, is_sender) == 6 * sizeof(uint32_t),
    "config block must stay contiguous right after the loaded interface fields");

// Total L1 footprint of one block: the fixed struct plus the variable-length
// receiver NOC XY table.
inline constexpr uint32_t dram_sender_state_block_size(uint32_t num_receivers) {
    return sizeof(DramSenderStateBlock) + 2u * num_receivers * sizeof(uint32_t);
}

}  // namespace tt::tt_metal
