// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "hostdev/streaming_profiler_common.h"

constexpr uint32_t kRingWords = kernel_profiler::PROFILER_L1_VECTOR_SIZE;
constexpr uint32_t kPrefix = kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
constexpr uint32_t kWireCtrl = kernel_profiler::SPSC_SPAN_WIRE_CTRL_WORDS;
constexpr uint32_t kPageBytes = kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4u;

// A frame occupies whole socket pages on the wire.
FORCE_INLINE constexpr uint32_t page_round(uint32_t bytes) { return (bytes + kPageBytes - 1u) & ~(kPageBytes - 1u); }

// The wrap-image rule as one subtraction on the take already computed (spsc_span_wrap_image costs two more instructions
// per wrapping lane). The check keeps it equal to the shared rule.
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

// The pads keep every read's source and destination congruent mod 16 B, wrap continuations included.
template <typename Read>
FORCE_INLINE uint32_t place_run(uint32_t start, uint32_t take, uint32_t frame_words, uint32_t frame, Read&& read) {
    const uint32_t ring_start = start & (kRingWords - 1u);
    if (ring_start + take <= kRingWords) {
        frame_words += kernel_profiler::spsc_span_pack_pad(start, frame_words);
        read(ring_start * 4u, frame + frame_words * 4u, take * 4u, true);
        return frame_words + take;
    }
    if (wrap_image(take)) {
        // A nearly full run that wraps ships as its whole ring image in one read, and the decoder linearises it by
        // head. Coalescing adjacent images into one read starves the producer's L1 port about 70x.
        frame_words += kernel_profiler::spsc_span_pack_pad(0u, frame_words);
        read(0u, frame + frame_words * 4u, kRingWords * 4u, true);
        return frame_words + kRingWords;
    }
    // A small run that wraps ships as two exact pieces, since most of its ring image would be dead space.
    frame_words += kernel_profiler::spsc_span_pack_pad(start, frame_words);
    const uint32_t first = kRingWords - ring_start;
    uint32_t dst = frame + frame_words * 4u;
    const uint32_t first_bytes = first * 4u;
    read(ring_start * 4u, dst, first_bytes, true);
    // Compute the second piece only after the first send, so the next poll trails the send by enough work. Otherwise
    // the poll lands before the NIU has assigned the VC and costs an extra spin per command.
    uint32_t rest = take;
    asm volatile("" : "+r"(dst), "+r"(rest), "+r"(frame_words));
    frame_words += rest;
    rest -= first;
    read(0u, dst + first_bytes, rest * 4u, false);
    return frame_words;
}

// A lane below its ship gate waits for a sweep that ships nothing else, but for at most this many sweeps (~5-10 ms when
// idle), so a trickling lane still reaches the host and a sparse grid sends at most one frame per core in that time.
constexpr uint32_t kMaxDeferSweeps = 2048;
// A relay sweeps back to back while it has work. After an idle sweep, it waits half again as long as it last did, at
// least 256 cycles more and at most 5 us, under a lane's fill time at high rates. The wait is a 32-bit low-word delta
// because a 64-bit wall-clock read on Blackhole can return the next epoch's high half with a pre-wrap low word, 2^32
// cycles (3.2 s) ahead.
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
FORCE_INLINE constexpr uint64_t host_addr(uint32_t hi, uint32_t lo) { return (static_cast<uint64_t>(hi) << 32) | lo; }

// write_cmd_buf is programmed once at relay init, and nothing else on the core touches it.
inline void write_to_host(uint32_t pcie_xy_enc, uint32_t src_l1, uint64_t dst_pcie, uint32_t size) {
    noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
        NOC_INDEX, src_l1, pcie_xy_enc, dst_pcie, size, 1);
}

// socket_push_pages only wraps the pointer, so a piece that crosses the FIFO's end is split here. fifo_size is whole
// pages, so the split keeps the pads' NoC alignment.
inline void push_fifo(const SocketSenderInterface& sender, uint32_t src, uint32_t dst, uint32_t len) {
    const uint32_t fifo_size = sender.downstream_fifo_curr_size;
    if (dst >= fifo_size) {
        dst -= fifo_size;
    }
    const uint64_t base = host_addr(sender.d2h.data_addr_hi, sender.downstream_fifo_addr);
    const uint32_t first = (dst + len > fifo_size) ? fifo_size - dst : len;
    write_to_host(sender.d2h.pcie_xy_enc, src, base + dst, first);
    if (first < len) {
        write_to_host(sender.d2h.pcie_xy_enc, src + first, base, len - first);
    }
}

// Blackhole stores can reach SRAM out of order, and the NIU and DMA engine read what the core staged. A bare fence also
// orders the device-register writes that start them, which std::atomic_thread_fence's `fence rw,rw` does not.
FORCE_INLINE void staged_store_fence() { asm volatile("fence" ::: "memory"); }

// Not socket_notify_receiver, which moves write_cmd_buf to another VC so bytes_sent can overtake its data. The same VC
// isn't enough either: the PCIe tile turns each NoC write into separate PCIe transactions with no ordering between them
// (a 4 B notify has landed ahead of the 15 KB pushed before it). So bytes_sent only goes out once the tile has acked
// every push.
inline void notify_bytes_sent(const SocketSenderInterface& sender) {
    while (!ncrisc_noc_nonposted_writes_flushed(NOC_INDEX)) {
    }
    reinterpret_cast<volatile tt_l1_ptr sender_socket_md*>(sender.config_addr)->bytes_sent = sender.bytes_sent;
    staged_store_fence();
    const uint64_t dst = host_addr(sender.d2h.bytes_sent_addr_hi, sender.downstream_bytes_sent_addr);
    write_to_host(sender.d2h.pcie_xy_enc, sender.config_addr, dst, 4u);
}
