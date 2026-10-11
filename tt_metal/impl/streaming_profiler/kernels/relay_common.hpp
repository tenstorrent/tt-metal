// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Frame packing and host writes for the relays that ship profiler data to the host.

#pragma once

#include <algorithm>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "hostdev/streaming_profiler_common.h"

constexpr uint32_t kRingWords = kernel_profiler::PROFILER_L1_VECTOR_SIZE;
constexpr uint32_t kPrefixWords = kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
constexpr uint32_t kWireControlWords = kernel_profiler::SPSC_SPAN_WIRE_CTRL_WORDS;
constexpr uint32_t kPageBytes = kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4u;

// Rounds `bytes` up to whole socket pages, because a frame occupies whole pages on the wire.
FORCE_INLINE constexpr uint32_t page_round(uint32_t bytes) { return (bytes + kPageBytes - 1u) & ~(kPageBytes - 1u); }

// Whether a wrapping run of `take` words ships as the whole ring image. It is spsc_span_wrap_image's test written as
// one subtraction, which saves two instructions per wrapping lane.
constexpr bool wrap_image(uint32_t take) {
    return take - (kRingWords - kernel_profiler::SPSC_SPAN_WRAP_IMAGE_MAX_PAD_WORDS) <=
           kernel_profiler::SPSC_SPAN_WRAP_IMAGE_MAX_PAD_WORDS;
}
static_assert(
    [] {
        for (uint32_t take = 1; take < 2 * kRingWords; take++) {
            if (wrap_image(take) != kernel_profiler::spsc_span_wrap_image(kRingWords - 1u, take, kRingWords)) {
                return false;
            }
        }
        return true;
    }(),
    "the inline image test drifted from spsc_span_wrap_image");

// Places one lane's run in the frame staged at L1 address `frame`, whose first `frame_words` words are already used,
// and returns the frame's new length in words. `read(src, dst, bytes, first)` copies ring offset `src` to L1 `dst`,
// with `first` false only for a wrapped run's second piece. The pads keep every read's source and destination congruent
// mod 16 B, including the second piece of a wrapped run.
template <typename Read>
FORCE_INLINE uint32_t place_run(uint32_t start, uint32_t take, uint32_t frame_words, uint32_t frame, Read&& read) {
    const uint32_t ring_start = start & (kRingWords - 1u);
    if (ring_start + take <= kRingWords) {
        frame_words += kernel_profiler::spsc_span_pack_pad(start, frame_words);
        read(ring_start * 4u, frame + frame_words * 4u, take * 4u, true);
        return frame_words + take;
    }
    if (wrap_image(take)) {
        // A nearly full run that wraps is shipped as the whole ring image in one read, and the decoder reorders it
        // using the head. Coalescing adjacent images into a single read starves the producer's L1 port by about 70x.
        frame_words += kernel_profiler::spsc_span_pack_pad(0u, frame_words);
        read(0u, frame + frame_words * 4u, kRingWords * 4u, true);
        return frame_words + kRingWords;
    }
    // A small run that wraps is shipped as two exact pieces, since most of its ring image would be unused.
    frame_words += kernel_profiler::spsc_span_pack_pad(start, frame_words);
    const uint32_t first = kRingWords - ring_start;
    uint32_t dst = frame + frame_words * 4u;
    const uint32_t first_bytes = first * 4u;
    read(ring_start * 4u, dst, first_bytes, true);
    // The second piece's operands are computed after the first read is sent, so there is enough work between that send
    // and the next command-buffer poll. An earlier poll arrives before the NIU has assigned the VC and costs an extra
    // spin.
    uint32_t rest = take;
    asm volatile("" : "+r"(dst), "+r"(rest), "+r"(frame_words));
    frame_words += rest;
    rest -= first;
    read(0u, dst + first_bytes, rest * 4u, false);
    return frame_words;
}

// Resets the back-off when the sweep did work, and otherwise grows it and waits that long, calling `idle` meanwhile.
// The back-off grows to at most 5 us, less than a lane's fill time at high rates. The wait compares 32-bit low words
// because a 64-bit wall-clock read can pair the high word from after a low-word wrap with the low word from before it,
// a value 2^32 cycles ahead.
template <typename Idle>
FORCE_INLINE void idle_wait(uint32_t& gap, bool worked, Idle&& idle) {
    constexpr uint32_t kCyclesPerUs = 1350;  // at the 1.35 GHz AICLK
    constexpr uint32_t kMinGapStep = 256, kMaxGap = 5 * kCyclesPerUs;
    if (worked) {
        gap = 0;
        return;
    }
    gap = std::min(gap + std::max(gap / 2u, kMinGapStep), kMaxGap);
    const uint32_t wait_start = get_timestamp_32b();
    while (get_timestamp_32b() - wait_start < gap) {
        idle();
    }
}

// Writes `size` bytes from L1 to host memory through write_cmd_buf, which is programmed once at relay init and which
// nothing else on the core touches.
inline void write_to_host(uint32_t pcie_xy_enc, uint32_t src_l1, uint64_t dst_pcie, uint32_t size) {
    noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
        NOC_INDEX, src_l1, pcie_xy_enc, dst_pcie, size, 1);
}

// Writes `len` bytes from L1 `src` to the host FIFO at offset `dst`, splitting a piece that crosses the FIFO's end,
// because socket_push_pages only wraps the pointer. fifo_size is a whole number of pages, so the split keeps the pads'
// NoC alignment.
inline void push_fifo(const SocketSenderInterface& sender, uint32_t src, uint32_t dst, uint32_t len) {
    const uint32_t fifo_size = sender.downstream_fifo_curr_size;
    if (dst >= fifo_size) {
        dst -= fifo_size;
    }
    const uint64_t base = kernel_profiler::join_words(sender.d2h.data_addr_hi, sender.downstream_fifo_addr);
    const uint32_t first = (dst + len > fifo_size) ? fifo_size - dst : len;
    write_to_host(sender.d2h.pcie_xy_enc, src, base + dst, first);
    if (first < len) {
        write_to_host(sender.d2h.pcie_xy_enc, src + first, base, len - first);
    }
}

// Makes the core's staged stores visible before the NIU or DMA engine reads them, because Blackhole stores can reach
// SRAM out of order. A bare `fence` also orders the device-register writes that start those reads, which
// std::atomic_thread_fence's `fence rw,rw` does not.
FORCE_INLINE void staged_store_fence() { asm volatile("fence" ::: "memory"); }

// Sends the socket's bytes_sent to the host once the PCIe tile has acked every push before it. socket_notify_receiver
// moves write_cmd_buf to another VC, so bytes_sent could overtake its data. Using the same VC isn't enough either,
// because the PCIe tile turns each NoC write into separate PCIe transactions with no ordering between them (a 4 B
// notify can land ahead of the 15 KB pushed before it).
inline void notify_bytes_sent(const SocketSenderInterface& sender) {
    while (!ncrisc_noc_nonposted_writes_flushed(NOC_INDEX)) {
    }
    reinterpret_cast<volatile tt_l1_ptr sender_socket_md*>(sender.config_addr)->bytes_sent = sender.bytes_sent;
    staged_store_fence();
    const uint64_t dst = kernel_profiler::join_words(sender.d2h.bytes_sent_addr_hi, sender.downstream_bytes_sent_addr);
    write_to_host(sender.d2h.pcie_xy_enc, sender.config_addr, dst, 4u);
}
