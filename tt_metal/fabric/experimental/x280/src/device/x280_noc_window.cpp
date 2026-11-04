// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280/device/x280_noc_window.hpp"

#include "x280/device/x280_state.hpp"
#include "x280_format.hpp"

#include "rt/x280_noc_map.h"

#include <stdexcept>

namespace x280 {

namespace {

// Field positions within the 128-bit entry: noc.h's shifts for noc_tlb_2m_t.
// The first two words carry the 43-bit address; the coordinate and flag fields
// start on the second 64-bit unit, which is word 2.
constexpr uint32_t kAddrShift = NOC_TLB_2M_LOCAL_OFFSET_SHIFT;  // entry stores target_addr >> 21
constexpr uint32_t kAddrBits = NOC_TLB_2M_LOCAL_OFFSET_BITS;
constexpr uint64_t kAddrMask = (1ull << kAddrBits) - 1;
constexpr uint32_t kAddrHighWordMask = (1u << (kAddrBits - 32)) - 1;

constexpr uint32_t kCoordMask = (1u << NOC_TLB_COORD_BITS) - 1;
static_assert(kCoordMask == X280_NOC_COORD_MAX, "TranslatedCoord's range must fill the window's coordinate fields");

uint32_t field(uint32_t word, uint32_t shift, uint32_t mask) { return (word >> shift) & mask; }

x280_noc_coord_t end_coord(const NocWindowWords& words) {
    return {
        field(words.data[2], NOC_TLB_X_END_SHIFT, kCoordMask), field(words.data[2], NOC_TLB_Y_END_SHIFT, kCoordMask)};
}

x280_noc_coord_t start_coord(const NocWindowWords& words) {
    return {
        field(words.data[2], NOC_TLB_X_START_SHIFT, kCoordMask),
        field(words.data[2], NOC_TLB_Y_START_SHIFT, kCoordMask)};
}

std::string coord_text(x280_noc_coord_t c) { return "(" + std::to_string(c.x) + "," + std::to_string(c.y) + ")"; }

}  // namespace

const char* noc_ordering_name(noc_ordering_t ordering) {
    switch (ordering) {
        case NOC_ORDER_DEFAULT: return "default";
        case NOC_ORDER_STRICT: return "strict_axi";
        case NOC_ORDER_POSTED: return "posted_writes";
    }
    throw std::invalid_argument(
        "NOC ordering " + std::to_string(static_cast<uint32_t>(ordering)) + " is not a noc_ordering_t member");
}

NocWindowEncodeResult encode(const TranslatedCoord& target, const NocWindow2M& window) {
    NocWindowEncodeResult result;
    result.target = target.noc();
    result.window = window;

    // TranslatedCoord's constructors already refuse this; a wider value would be
    // masked into another tile's coordinate below.
    if (!x280_noc_coord_valid(target.noc())) {
        result.fault = NocWindowEncodeFault::CoordinateInvalid;
        return result;
    }
    if (window.target_addr > kNoc2mMaxTargetBase) {
        result.fault = NocWindowEncodeFault::TargetAboveMaxBase;
        return result;
    }
    if ((window.target_addr & (kNoc2mWindowStride - 1)) != 0) {
        result.fault = NocWindowEncodeFault::TargetMisaligned;
        return result;
    }
    if (!noc_ordering_valid(window.ordering)) {
        result.fault = NocWindowEncodeFault::OrderingInvalid;
        return result;
    }

    const uint64_t addr = window.target_addr >> kAddrShift;
    result.words.data[0] = static_cast<uint32_t>(addr);
    result.words.data[1] = static_cast<uint32_t>(addr >> 32);

    // Unicast: the start and end of the destination range are the same tile.
    result.words.data[2] = (target.x() << NOC_TLB_X_END_SHIFT) | (target.y() << NOC_TLB_Y_END_SHIFT) |
                           (target.x() << NOC_TLB_X_START_SHIFT) | (target.y() << NOC_TLB_Y_START_SHIFT) |
                           (static_cast<uint32_t>(window.ordering) << NOC_TLB_ORDERING_SHIFT) |
                           (window.noc1 ? (1u << NOC_TLB_NOC_SEL_SHIFT) : 0u);
    return result;
}

std::string describe(const NocWindowEncodeResult& result) {
    switch (result.fault) {
        case NocWindowEncodeFault::None: return "ok";
        case NocWindowEncodeFault::CoordinateInvalid:
            return "NOC coordinate " + coord_text(result.target) + " fails x280_noc_coord_valid";
        case NocWindowEncodeFault::TargetAboveMaxBase:
            return "NOC target base " + hex(result.window.target_addr) + " exceeds the largest representable base " +
                   hex(kNoc2mMaxTargetBase);
        case NocWindowEncodeFault::TargetMisaligned:
            return "NOC target base " + hex(result.window.target_addr) + " is not 2 MiB aligned";
        case NocWindowEncodeFault::OrderingInvalid:
            return "NOC ordering " + std::to_string(static_cast<uint32_t>(result.window.ordering)) +
                   " is not a noc_ordering_t member";
    }
    return "unknown encode fault " + std::to_string(static_cast<uint32_t>(result.fault));
}

NocWindowDecodeResult decode(const NocWindowWords& words) {
    NocWindowDecodeResult result;
    result.words = words;
    if (is_unusable(words)) {
        result.fault = NocWindowDecodeFault::Unusable;
        return result;
    }

    const auto end = end_coord(words);
    const auto start = start_coord(words);
    if (field(words.data[2], NOC_TLB_MCAST_SHIFT, 1u) != 0) {
        result.fault = NocWindowDecodeFault::Multicast;
        return result;
    }
    if (start.x != end.x || start.y != end.y) {
        result.fault = NocWindowDecodeFault::StartIsNotEnd;
        return result;
    }
    if ((words.data[1] & ~kAddrHighWordMask) != 0) {
        result.fault = NocWindowDecodeFault::ReservedAddressBits;
        return result;
    }
    const uint32_t ordering = field(words.data[2], NOC_TLB_ORDERING_SHIFT, NOC_TLB_ORDERING_MASK);
    if (!noc_ordering_valid(static_cast<noc_ordering_t>(ordering))) {
        result.fault = NocWindowDecodeFault::CountedWrites;
        return result;
    }

    const uint64_t addr = static_cast<uint64_t>(words.data[0]) | (static_cast<uint64_t>(words.data[1]) << 32);
    result.window.target_addr = (addr & kAddrMask) << kAddrShift;
    result.coord = end;
    result.window.ordering = static_cast<noc_ordering_t>(ordering);
    result.window.noc1 = field(words.data[2], NOC_TLB_NOC_SEL_SHIFT, 1u) != 0;
    return result;
}

std::string describe(const NocWindowDecodeResult& result) {
    switch (result.fault) {
        case NocWindowDecodeFault::None: return "ok";
        case NocWindowDecodeFault::Unusable: return "NOC window payload read all ones";
        case NocWindowDecodeFault::Multicast: return "NOC window is multicast, not unicast";
        case NocWindowDecodeFault::StartIsNotEnd:
            return "NOC window coordinate range starts at " + coord_text(start_coord(result.words)) + " and ends at " +
                   coord_text(end_coord(result.words));
        case NocWindowDecodeFault::ReservedAddressBits:
            return "NOC window has non-zero reserved address bits in word 1 " + hex(result.words.data[1]);
        case NocWindowDecodeFault::CountedWrites: return "NOC window ordering is 3, Counted Writes";
    }
    return "unknown decode fault " + std::to_string(static_cast<uint32_t>(result.fault));
}

bool is_unusable(const NocWindowWords& words) {
    return is_all_ones(words.data[0]) && is_all_ones(words.data[1]) && is_all_ones(words.data[2]);
}

uint64_t noc_window_config_offset(uint32_t index) { return static_cast<uint64_t>(index) * kNoc2mConfigStride; }

uint64_t noc_window_aperture_offset(uint32_t index, uint64_t target_addr) {
    return static_cast<uint64_t>(index) * kNoc2mWindowStride + (target_addr & (kNoc2mWindowStride - 1));
}

NocWindowCheck validate_host_window_index(uint32_t index) {
    NocWindowCheck check;
    constexpr uint32_t kFirst = X280_TLB_HOST_BASE;
    constexpr uint32_t kLast = X280_TLB_HOST_BASE + X280_TLB_HOST_COUNT - 1;

    if (index >= kNoc2mWindowCount) {
        check.reason = "NOC window index " + std::to_string(index) + " is outside the hardware range 0.." +
                       std::to_string(kNoc2mWindowCount - 1);
        return check;
    }
    if (index < kFirst || index > kLast) {
        check.reason = "NOC window index " + std::to_string(index) + " is outside the host-programmable block " +
                       std::to_string(kFirst) + ".." + std::to_string(kLast) +
                       ". Every other index belongs to a firmware image (rt/x280_noc_map.h), and both sides "
                       "programming one entry is a silent collision.";
        return check;
    }
    check.ok = true;
    return check;
}

}  // namespace x280
