// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/host_l1_map.hpp"

#include <fmt/format.h>

namespace tt::tt_metal::experimental {

namespace {

constexpr uint32_t align64(uint32_t v) { return (v + 0x3Fu) & ~0x3Fu; }

}  // namespace

// 64 B alignment throughout: the NOC needs source and destination to agree in bits [3:0].
// Both slots are sized for a whole FRAME; under !bidirectional they are the same bytes.
L1MapNew L1MapNew::compute(uint32_t l1_base, uint32_t l1_size, uint32_t payload_bytes, bool bidirectional) {
    L1MapNew m;
    m.l1_size = l1_size;
    m.l1_base = l1_base;
    m.bidirectional = bidirectional;
    const uint32_t slot = align64(tt_uva_frame_page_size(payload_bytes));
    m.payload_addr = align64(l1_base);
    m.stage_addr = m.payload_addr + slot;
    m.stop_addr = m.stage_addr + kFrameTrailerBytes;
    m.consumed_addr = m.stop_addr + kDoorbellBytes;
    m.dest_word_addr = m.consumed_addr + kDoorbellBytes;
    // Its own line: stop_addr is polled by the host mid-run, so sharing would false-share.
    m.verify_addr = m.dest_word_addr + kDestWordBytes;
    if (bidirectional) {
        m.deliver_addr = align64(m.verify_addr + kDoorbellBytes);
        m.deliver_end = m.deliver_addr + slot;
    } else {
        m.deliver_addr = m.payload_addr;
        m.deliver_end = 0;
    }
    return m;
}

// Bounds the map's top, not just the payload span: every word above the payload is written
// by someone -- the sender kernel, the pull kernel, or host delivery.
std::string L1MapNew::fits(uint32_t payload_bytes) const {
    if (end() <= l1_size && deliver_addr + tt_uva_frame_page_size(payload_bytes) <= l1_size) {
        return {};
    }
    return fmt::format(
        "payload {} B does not fit this core's L1: needs {} x {} B plus {} B of control words, in {} B",
        payload_bytes,
        payload_copies(),
        payload_bytes,
        control_bytes(),
        l1_size);
}

std::string L1MapNew::describe() const {
    return fmt::format(
        "payload {:#x} stage {:#x} stop {:#x} verify {:#x} consumed {:#x} dest_word {:#x} deliver {:#x} "
        "(L1 {} B, payload span {} B)",
        payload_addr,
        stage_addr,
        stop_addr,
        verify_addr,
        consumed_addr,
        dest_word_addr,
        deliver_addr,
        l1_size,
        stage_addr - payload_addr);
}

}  // namespace tt::tt_metal::experimental
