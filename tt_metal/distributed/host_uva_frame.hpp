// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The wire frame: one message is [payload | trailer] occupying one socket FIFO page.
// Trailer last so the same bytes double as the H2H arrival flag when forwarded.
#pragma once

#include <stdint.h>

#include "tt_metal/distributed/host_uva.hpp"

namespace tt::tt_metal::experimental {

// 64 B keeps payload + trailer PCIe-aligned whenever the payload alone is.
constexpr uint32_t kFrameTrailerBytes = 64;

constexpr uint64_t kFrameMagic = 0x5556ull;  // 'UV'
// 2 gave the reserved words meaning. A v1 sender never cleared them, so its garbage would
// read as a signal opcode.
constexpr uint32_t kFrameVersion = 2;
constexpr uint32_t kFrameGuardMagicShift = 16;

// Armed by the sender in stage(), checked before the trailer is trusted. A consumer that
// reuses it as an arrival flag clears it; the D2H ring gates slot reuse on credits instead.
constexpr uint64_t tt_uva_frame_guard(uint32_t version) {
    return (kFrameMagic << kFrameGuardMagicShift) | static_cast<uint64_t>(version);
}
constexpr bool tt_uva_frame_armed(uint64_t guard) { return guard == tt_uva_frame_guard(kFrameVersion); }

// Cycles on the sender's own clock; the host applies a measured ns/cycle rate. Low half is
// the payload write and its barrier, high half the wait for a free slot ahead of it.
constexpr uint64_t kFrameElapsedMask = 0xFFFFFFFFull;
constexpr uint64_t tt_uva_frame_elapsed_pack(uint64_t issue, uint64_t stall) {
    return ((stall > kFrameElapsedMask ? kFrameElapsedMask : stall) << 32) |
           (issue > kFrameElapsedMask ? kFrameElapsedMask : issue);
}
constexpr uint32_t tt_uva_frame_elapsed_issue(uint64_t packed) {
    return static_cast<uint32_t>(packed & kFrameElapsedMask);
}
constexpr uint32_t tt_uva_frame_elapsed_stall(uint64_t packed) { return static_cast<uint32_t>(packed >> 32); }

// How the receiver updates the signal word once the payload has landed. SET stamps a value,
// ADD accumulates -- so N senders can drive one counter.
enum UvaSignalOp : uint32_t { kSignalNone = 0, kSignalSet = 1, kSignalAdd = 2 };

// Written by the device, read by both hosts, forwarded unrewritten.
struct FrameTrailer {
    uint64_t guard;
    uint64_t dst;     // tt_uva_t bits
    uint32_t length;  // payload bytes ahead of this trailer
    uint32_t origin;  // sender's tt_uva_t6_global_selector
    uint64_t elapsed;
    // The signal rides the frame rather than racing it, so data-before-signal needs no
    // fence: the payload and these words arrive in one page.
    // Symmetric: the host hands both sides the same l1_base through tt_uva_ini(), so this
    // offset names one word that sender and target each own at the same place.
    uint32_t sig_off;
    uint32_t sig_val;
    uint32_t sig_op;  // UvaSignalOp
    uint32_t reserved0;
    uint64_t reserved[2];
};
static_assert(sizeof(FrameTrailer) == kFrameTrailerBytes, "the trailer must fill its slot");

// A peer's bytes name the address this core stores to, so the span is bounded before use.
constexpr bool tt_uva_frame_signal_ok(uint32_t sig_op, uint32_t sig_off, uint32_t l1_size) {
    return sig_op == kSignalNone ||
           (sig_op <= kSignalAdd && sig_off % alignof(uint32_t) == 0 &&
            static_cast<uint64_t>(sig_off) + sizeof(uint32_t) <= l1_size);
}

constexpr uint32_t tt_uva_frame_page_size(uint32_t payload_bytes) { return payload_bytes + kFrameTrailerBytes; }

}  // namespace tt::tt_metal::experimental
