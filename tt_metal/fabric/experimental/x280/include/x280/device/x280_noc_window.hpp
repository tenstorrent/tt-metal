// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Encoding for the X280's outbound 2 MiB NOC TLB windows.
//
// These are the windows an X280 writes *through* to reach a peer tile, not the
// host's own TLBs into the chip. Firmware programs them with the inline helpers
// in x280/include/noc.h; this is the same encoding for a host that wants to hand
// a tile a ready window rather than have a kernel hard-code a peer coordinate.
//
// Pure: no hardware, no UMD, no Metal. The config register's base address is the
// device component's business; what lives here is the bit layout, the offset
// within the config block, and which indices a host may touch.
//
// The 128 GiB windows are deliberately absent. They exist for DMA NOC-to-NOC
// transfers whose address split is a firmware-side recipe (see
// noc_configure_tlb_128g), and nothing host-side consumes them.

#pragma once

#include "x280/device/translated_coord.hpp"
#include "noc.h"
#include "x280_noc_coord.h"

#include <cstdint>
#include <string>

namespace x280 {

// Hardware-defined. Indices are 0..kNoc2mWindowCount-1.
constexpr uint32_t kNoc2mWindowCount = NOC_2M_WINDOW_COUNT;
constexpr uint64_t kNoc2mWindowStride = NOC_2M_WINDOW_STRIDE;  // 2 MiB per window
constexpr uint64_t kNoc2mConfigStride = 0x10ull;               // 4 x u32 per entry
constexpr uint64_t kNoc2mMaxTargetBase = UINT64_MAX - (kNoc2mWindowStride - 1);

// The window's own words, in the order firmware stores them.
struct NocWindowWords {
    uint32_t data[4] = {0, 0, 0, 0};

    bool operator==(const NocWindowWords& other) const {
        return data[0] == other.data[0] && data[1] == other.data[1] && data[2] == other.data[2] &&
               data[3] == other.data[3];
    }
};

// A unicast window onto one peer tile, less the tile's coordinate: encode takes
// it as a TranslatedCoord and decode reports it as an x280_noc_coord_t.
//
// Only the fields noc_configure_tlb_2m exposes. The wider bitfield —
// multicast, striding, exclusion, virtual channel — is reachable from firmware
// via noc_configure_tlb_2m_ext and has no host-side consumer.
struct NocWindow2M {
    // Aligned base of the destination's 2 MiB NOC address span.
    uint64_t target_addr = 0;

    noc_ordering_t ordering = NOC_ORDER_DEFAULT;
    bool noc1 = false;  // NOC1 rather than NOC0
};

/** TLBWindows.md's name for `ordering`, as records spell it: "default", "strict_axi" or
 * "posted_writes". Throws std::invalid_argument for a value outside noc_ordering_t. */
const char* noc_ordering_name(noc_ordering_t ordering);

// Why encode refused an input; None when it did not.
enum class NocWindowEncodeFault : uint8_t {
    None,
    CoordinateInvalid,   // the target fails x280_noc_coord_valid
    TargetAboveMaxBase,  // target_addr is above kNoc2mMaxTargetBase
    TargetMisaligned,    // target_addr is not 2 MiB aligned
    OrderingInvalid,     // ordering is not a noc_ordering_t member
};

// Why decode refused the words; None when it did not.
enum class NocWindowDecodeFault : uint8_t {
    None,
    Unusable,             // the payload words read all ones (is_unusable)
    Multicast,            // the multicast bit is set
    StartIsNotEnd,        // the start and end coordinates differ
    ReservedAddressBits,  // word 1 has bits set above the 43-bit address
    CountedWrites,        // ordering mode 3, which noc_ordering_t excludes
};

/** What encode() produced, or why it refused; describe() renders a refusal. */
struct NocWindowEncodeResult {
    NocWindowEncodeFault fault = NocWindowEncodeFault::None;
    // The inputs encode() judged, so a refusal names the value it refused.
    x280_noc_coord_t target = {0, 0};
    NocWindow2M window;
    NocWindowWords words;  // meaningful only when ok()

    bool ok() const { return fault == NocWindowEncodeFault::None; }
};

/** One line naming the refused value; "ok" for a result that is ok(). */
std::string describe(const NocWindowEncodeResult& result);

// Bit layout is written out rather than taken from the noc_tlb_2m_t union in
// noc.h: that union's field order is what the rv64 compiler happens to pack, and
// this has to produce the identical words from a host compiler. A mismatch would
// program a window that points somewhere plausible and wrong.
//
// The coordinate is accepted only as a TranslatedCoord, resolved by the caller
// for the NoC window.noc1 selects; it must pass x280_noc_coord_valid. target_addr
// must be a representable 2 MiB base, and window.ordering a noc_ordering_t
// member. Invalid inputs are not truncated.
NocWindowEncodeResult encode(const TranslatedCoord& target, const NocWindow2M& window);

/** What decode() read, or why it refused the words; describe() renders a refusal. */
struct NocWindowDecodeResult {
    NocWindowDecodeFault fault = NocWindowDecodeFault::None;
    NocWindowWords words;  // the words decoded
    /** What the entry holds, read back rather than resolved, so not a TranslatedCoord. Always valid. */
    x280_noc_coord_t coord = {0, 0};
    NocWindow2M window;  // meaningful only when ok()

    bool ok() const { return fault == NocWindowDecodeFault::None; }
};

/** One line naming what the words hold that decode() refused; "ok" for a result that is ok(). */
std::string describe(const NocWindowDecodeResult& result);

// Rejects unusable reads and entries whose start/end coordinates or multicast
// bit do not describe a unicast window, or whose ordering is Counted Writes.
// Other extended fields are omitted.
NocWindowDecodeResult decode(const NocWindowWords& words);

// True when a readback cannot be believed.
//
// An unclocked tile returns all ones for the three payload words. Reserved word
// 3 may return either all ones or zero. No single all-ones word is sufficient.
bool is_unusable(const NocWindowWords& words);

// Byte offset of an entry within the 2 MiB TLB config block.
uint64_t noc_window_config_offset(uint32_t index);

// Offset of a window within the outbound aperture, including the low bits of
// target_addr. Add the System Port or Memory Port base to get an address the
// tile can dereference.
uint64_t noc_window_aperture_offset(uint32_t index, uint64_t target_addr);

struct NocWindowCheck {
    bool ok = false;
    std::string reason;  // empty when ok
};

// A host may program only the block reserved for it in rt/x280_noc_map.h.
// Every other index belongs to a firmware image, and both sides writing one
// config register is a silent collision: the loser's window still resolves, just
// to the wrong place.
NocWindowCheck validate_host_window_index(uint32_t index);

}  // namespace x280
