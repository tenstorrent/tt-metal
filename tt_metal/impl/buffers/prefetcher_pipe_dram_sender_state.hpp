// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRISC-L1 sender block for a DRAM-sender PrefetcherPipe.
//
// A PrefetcherPipe sender endpoint is fully described by its config page (the 9-word
// remote-DFB header, followed by the receiver NOC XY table and the per-receiver
// entries_sent/entries_acked pairs). For a DRAM-sender PrefetcherPipe that page lives in
// DRISC L1 rather than in a persistent worker-L1 arena allocation, so the DRISC kernel can
// build a sender interface with setup_prefetcher_pipe_interface() without any
// launch-message state -- DRAM cores are not dispatched to and therefore never Attach.
//
// Two facts the kernel cannot read out of the config page precede it:
//
//   recv_index_base     Bank-local slab index of this sender's first receiver, so two
//                       DRISC cores can split one bank's receiver set (the kernel reads
//                       slab recv_index_base + r for its local receiver r).
//   max_num_receivers   Largest receiver count across the sibling pipes the Tensor
//                       prefetcher created together. The prefetcher request page uses it
//                       as the uniform per-tensor layout-slot stride; it is constant
//                       across senders, unlike the per-sender receiver count in the config
//                       page's word[1].
//
// Unlike DramSenderStateBlock (the GlobalCircularBuffer equivalent) there is no
// fifo_wr_ptr here: a PrefetcherPipe sender derives each receiver's write cursor from that
// receiver's durable entries_sent counter, which already lives in the config page and
// survives across programs. Nothing has to be written back at request end.
//
// Shared by host (composes the bytes) and the DRISC kernel (reads the prefix), so keep it
// packed and keep the config page at a fixed offset both sides agree on.

#pragma once

#include <cstdint>

namespace tt::tt_metal {

struct PrefetcherPipeDramSenderState {
    uint32_t recv_index_base;
    uint32_t max_num_receivers;
    uint32_t reserved0;
    uint32_t reserved1;
} __attribute__((packed));

// Byte offset from the allocation base to the PrefetcherPipe sender config page. Held at a
// fixed 16 bytes -- one L1_ALIGNMENT unit on Wormhole and Blackhole -- so the config page
// stays L1-aligned (the per-receiver counters inside it are NOC-atomic targets) and both
// host and kernel can name the offset without consulting the HAL. The host asserts the
// device's actual L1 alignment divides it.
inline constexpr uint32_t kPrefetcherPipeSenderPrefixBytes = 16;

inline constexpr uint32_t prefetcher_pipe_config_page_offset() { return kPrefetcherPipeSenderPrefixBytes; }

static_assert(
    sizeof(PrefetcherPipeDramSenderState) == kPrefetcherPipeSenderPrefixBytes,
    "PrefetcherPipeDramSenderState must exactly fill the prefix reserved ahead of the config page");

}  // namespace tt::tt_metal
