// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Pure X280 state classification and validation.
//
// Nothing here touches hardware, UMD, or Metal: every function is a decision
// over values already read. That is what makes the rules testable without
// silicon, and it is where the two rules that are easiest to get wrong live —
// rejecting an all-ones read before masking it, and bounding a resident-idle
// image before writing LIM.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "rt/x280_pinned_input.h"

namespace x280 {

inline constexpr uint8_t kMaxLimWays = 15;
inline constexpr uint64_t kLimWayBytes = 128 * 1024;

/** LIM ways at the prepared profile, WayEnable X280_PINNED_INPUT_WAYENABLE: the
 * shape the pinned-input suites and the PacketInjection layout open with. */
inline constexpr uint8_t kPreparedLimWays = kMaxLimWays - X280_PINNED_INPUT_WAYENABLE;
static_assert(
    X280_PINNED_INPUT_WAYENABLE < kMaxLimWays,
    "X280_PINNED_INPUT_WAYENABLE must leave at least one LIM way: WayEnable N backs 15 - N ways");

// Opt-in component memory profile. Legacy reserves the prefix through the
// per-hart protocol block. PacketInjection establishes a dedicated lower arena
// between image text and active stacks.
enum class LimLayout : uint32_t {
    Legacy = 0,
    PacketInjection = 1,
};

// Number of 128 KiB ways exposed as contiguous LIM from X280_LIM_BASE.
// The default is the reset shape: one cache way and fifteen LIM ways.
struct LimShape {
    uint8_t lim_ways = kMaxLimWays;

    bool operator==(const LimShape&) const = default;
};

// A half-open byte range in one tile's LIM.
struct LimRange {
    uint64_t begin = 0;
    uint64_t end = 0;

    /**
     * Byte count in the range, zero for an inverted range.
     */
    uint64_t size() const { return end >= begin ? end - begin : 0; }
    /**
     * @param bytes  compared against the extent remaining from `addr`, so a
     *               length whose sum would overflow is rejected rather than
     *               wrapping.
     */
    bool contains(uint64_t addr, uint64_t bytes) const {
        return end >= begin && addr >= begin && addr <= end && bytes <= end - addr;
    }
};

inline constexpr uint64_t kLimCacheLineBytes = 64;

struct LimBlockCheck {
    bool ok = false;
    std::string reason;
};

// Pure range calculation for a given layout profile and backed byte capacity.
LimRange available_lim_for_layout(LimLayout layout, uint64_t backed_bytes);

// Reads follow Metal's 4-byte block transport. Writes are whole cache lines so
// high LIM with uninitialised ECC never takes a partial-line read/merge path.
LimBlockCheck validate_available_lim_read(LimRange available, uint64_t address, uint64_t size);
LimBlockCheck validate_available_lim_write(LimRange available, uint64_t address, uint64_t size);

// Placement of an ActiveRunRequest argument block. A zero address with zero size
// means the image takes no arguments and is accepted. Anything else must be a
// non-empty, cache-line-aligned, whole-line span inside the tile's available
// LIM: the same rule as a write, because a case can only have populated the
// block through write_available_lim.
LimBlockCheck validate_argument_block(LimRange available, uint64_t address, uint64_t size);

struct LimShapeCheck {
    bool ok = false;
    uint32_t target_wayenable = 0;
    uint64_t backed_bytes = 0;
    std::string reason;
};

// Validates both the hardware range and the caller's lowest acceptable LIM
// capacity. The target WayEnable is 15 - lim_ways.
LimShapeCheck validate_lim_shape(LimShape shape, uint64_t required_backed_bytes);

enum class LimShapeTransition {
    AlreadyConfigured,
    ApplyWhileReset,
    ResetRequired,
};

// WayEnable is increase-only. A larger value may be applied only while every
// L2CPU on the chip is held in reset; a smaller value requires an ASIC reset.
LimShapeTransition classify_lim_shape_transition(
    uint32_t current_wayenable, uint32_t target_wayenable, bool all_tiles_held_in_reset);

// An unclocked or unreachable L2CPU answers reads with all ones. Both widths
// occur: 32-bit for CSRs, 64-bit for the LIM mailboxes assembled from a block
// read.
bool is_all_ones(uint32_t value);
bool is_all_ones(uint64_t value);

// ---------------------------------------------------------------------------
// LIM ECC
// ---------------------------------------------------------------------------

enum class LimEccState {
    // Any raw read was all ones: the tile is unclocked or the transport is
    // broken. Never a statement about ECC.
    Invalid,
    // Marker present and at least one LIM way remains backed.
    Primed,
    // WayEnable=15 leaves no LIM, so the marker and boot protocol cannot be
    // observed. Recovering any LIM requires an ASIC reset.
    LimUnavailable,
    // No marker: LIM has never been primed in this chip-reset lifetime, so a
    // narrow write can silently corrupt an uninitialised ECC line.
    Unprimed,
};

const char* to_string(LimEccState state);

struct LimEccClassification {
    LimEccState state = LimEccState::Invalid;
    // Bytes from X280_LIM_BASE the prime walked, as the word beside the marker
    // records it. Set only when state is Primed; empty for a marker seeded
    // without an extent word, which validate_active_image bounds by
    // kLimEccUnrecordedPrimeBytes.
    std::optional<uint64_t> primed_bytes;
};

// Classification order is load-bearing. Checking WayEnable first and masking
// it with 0xF turns an unclocked tile's 0xFFFFFFFF into a false committed 0xF,
// so the all-ones guard goes first.
//
// The prime record is the marker and the extent word beside it. A magic
// marker with a zero extent is Primed with no recorded extent; a magic marker
// whose extent is not a whole number of cache lines within one L3 way is not
// a prime record and classifies Unprimed.
LimEccClassification classify_lim_ecc(uint32_t raw_wayenable, uint64_t raw_marker, uint64_t raw_extent);

// Backed LIM capacity is (15 - WayEnable) * 128 KiB. The *address* range is a
// fixed 2 MiB that the static TLB covers exactly, so scratch cannot fall
// outside the window — but the backing shrinks as ways convert to cache.
uint64_t lim_backed_bytes(uint32_t wayenable);

// ---------------------------------------------------------------------------
// Image placement
// ---------------------------------------------------------------------------

struct ImageCheck {
    bool ok = false;
    std::string reason;  // empty when ok
};

// Bound a resident-idle image against the reserved region before writing LIM,
// independently of the linker script that produced it.
//
// Two comparisons: the image starts at the region base, and it ends at or below
// the ECC prime marker. The second bound is x280-lim-idle.ld's own tighter
// ASSERT (_bss_end <= ECC_MARKER_ADDR), so clearing it also clears the idle
// stack and the trap-diagnostic block at the top of the region. Clobbering the
// marker or the stack can wedge the tile until an ASIC reset.
ImageCheck validate_idle_image(uint64_t load_addr, size_t size);

// Bound an active image against its reserved region before writing LIM.
//
// Two upper bounds, whichever is lower: the selected layout's image boundary
// and the end of the ECC-primed LIM window, past which a hart's instruction
// fetch faults even though the host can write and read those bytes.
// primed_bytes is the tile's LimEccClassification::primed_bytes; empty bounds
// by kLimEccUnrecordedPrimeBytes.
//
// Checks file bytes, which exclude NOLOAD .bss, so this is necessary and not
// sufficient: the linker script is what bounds the memory footprint.
ImageCheck validate_active_image(
    uint64_t load_addr, size_t size, std::optional<uint64_t> primed_bytes, LimLayout layout = LimLayout::Legacy);

// ---------------------------------------------------------------------------
// Tile numbering
// ---------------------------------------------------------------------------

// Hardware L2CPU numbering, fixed by tt-isa-documentation
// BlackholeA0/L2CPUTile/README.md: reset bit 4+N, hart group 4N..4N+3, and
// telemetry L2CPUCLK<N>. UMD sorts its L2CPU cores by Y, so a UMD array index
// is NOT this number; convert explicitly or select the wrong tile while
// looking plausible.
constexpr int kNumL2cpuTiles = 4;
constexpr uint32_t kHartsPerTile = 4;

struct Noc0Coord {
    uint32_t x = 0;
    uint32_t y = 0;
};

// Returns -1 for any coordinate outside the documented set.
int hw_l2cpu_from_noc0(uint32_t noc_x, uint32_t noc_y);

// Returns nullopt for a hardware index outside 0..3.
std::optional<Noc0Coord> noc0_from_hw_l2cpu(int hw_index);

// Bit position of a tile in L2CPU_RESET. Set means out of reset.
int l2cpu_reset_bit(int hw_index);

}  // namespace x280
