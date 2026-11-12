// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280/device/x280_state.hpp"
#include "x280_format.hpp"
#include "x280_hw_addrs.hpp"

// The firmware ABI header. Macros only as far as this file is concerned — its
// `extern uint64_t` declarations are linker-script symbols that exist only in
// an X280 image, so nothing here may reference them; the host reads a hart's
// trap block through X280_TRAP_DIAG_ADDR(X280_IDLE_TRAP_DIAG_ADDR, hart).
#include "x280.h"
#include "x280_hart_protocol.h"

#include <array>

namespace x280 {

namespace {

// The idle image may not extend past the ECC prime marker. x280-lim-idle.ld
// ASSERTs the same bound on _bss_end; enforcing it here covers images that
// script did not produce.
static_assert(X280_ECC_PRIME_MARKER_OFFSET <= UINT64_MAX - X280_IDLE_FW_LOAD_ADDR);
constexpr uint64_t kIdleImageEnd = X280_IDLE_FW_LOAD_ADDR + X280_ECC_PRIME_MARKER_OFFSET;
static_assert(X280_IDLE_FW_LOAD_ADDR <= kIdleImageEnd);
constexpr uint64_t kMaxIdleImageBytes = kIdleImageEnd - X280_IDLE_FW_LOAD_ADDR;

// entry.S gives hart N a stack topped at _stack_top - N * HART_STACK_SIZE.
static_assert(kHartsPerTile <= X280_ACTIVE_TRAP_DIAG_ADDR / HART_STACK_SIZE);
constexpr uint64_t kStackBoundedImageEnd = X280_ACTIVE_TRAP_DIAG_ADDR - kHartsPerTile * HART_STACK_SIZE;
static_assert(X280_LIM_BASE < kStackBoundedImageEnd);

constexpr uint64_t kPacketInjectionImageEnd = X280_LIM_PACKET_INJECTION_IMAGE_END;
static_assert(X280_ACTIVE_FW_LOAD_ADDR <= kPacketInjectionImageEnd);

constexpr std::array<const char*, 4> kLimEccStateNames = {
    "invalid",
    "primed",
    "lim_unavailable",
    "unprimed",
};

LimBlockCheck validate_lim_block(
    LimRange available, uint64_t address, uint64_t size, uint64_t alignment, const char* operation) {
    LimBlockCheck check;
    if (size == 0) {
        check.reason = std::string(operation) + " size is zero";
        return check;
    }
    if (address % alignment != 0 || size % alignment != 0) {
        check.reason =
            std::string(operation) + " address and size must be aligned to " + std::to_string(alignment) + " bytes";
        return check;
    }
    if (!available.contains(address, size)) {
        check.reason = std::string(operation) + " range [" + hex(address) + ", " +
                       (size <= UINT64_MAX - address ? hex(address + size) : std::string("overflow")) +
                       ") is outside available LIM [" + hex(available.begin) + ", " + hex(available.end) + ")";
        return check;
    }
    check.ok = true;
    return check;
}

}  // namespace

bool is_all_ones(uint32_t value) { return value == 0xFFFFFFFFu; }

bool is_all_ones(uint64_t value) { return value == 0xFFFFFFFFFFFFFFFFull; }

LimBlockCheck validate_available_lim_read(LimRange available, uint64_t address, uint64_t size) {
    return validate_lim_block(available, address, size, sizeof(uint32_t), "LIM read");
}

LimBlockCheck validate_available_lim_write(LimRange available, uint64_t address, uint64_t size) {
    return validate_lim_block(available, address, size, kLimCacheLineBytes, "LIM write");
}

LimBlockCheck validate_argument_block(LimRange available, uint64_t address, uint64_t size) {
    LimBlockCheck check;
    if (address == 0 && size == 0) {
        check.ok = true;
        return check;
    }
    // Half-specified is a caller mistake either way round: an address with no
    // extent cannot be bounds-checked, and an extent with no address would be
    // published as "no arguments" while the caller believed otherwise.
    if (address == 0 || size == 0) {
        check.reason = "argument block needs both an address and a size, or neither; got address " + hex(address) +
                       " and size " + std::to_string(size);
        return check;
    }
    return validate_lim_block(available, address, size, kLimCacheLineBytes, "argument block");
}

LimShapeCheck validate_lim_shape(LimShape shape, uint64_t required_backed_bytes) {
    LimShapeCheck check;
    if (shape.lim_ways > kMaxLimWays) {
        check.reason = "LIM shape requests " + std::to_string(shape.lim_ways) + " ways; hardware supports at most " +
                       std::to_string(kMaxLimWays);
        return check;
    }

    check.target_wayenable = kMaxLimWays - shape.lim_ways;
    check.backed_bytes = static_cast<uint64_t>(shape.lim_ways) * kLimWayBytes;
    if (check.backed_bytes < required_backed_bytes) {
        check.reason = "LIM shape leaves " + std::to_string(check.backed_bytes) + " backed bytes, below the required " +
                       std::to_string(required_backed_bytes);
        return check;
    }

    check.ok = true;
    return check;
}

LimShapeTransition classify_lim_shape_transition(
    uint32_t current_wayenable, uint32_t target_wayenable, bool all_tiles_held_in_reset) {
    const uint32_t current = current_wayenable & 0xFu;
    if (current == target_wayenable) {
        return LimShapeTransition::AlreadyConfigured;
    }
    if (current > target_wayenable || !all_tiles_held_in_reset) {
        return LimShapeTransition::ResetRequired;
    }
    return LimShapeTransition::ApplyWhileReset;
}

const char* to_string(LimEccState state) { return enum_name(state, kLimEccStateNames); }

LimEccClassification classify_lim_ecc(uint32_t raw_wayenable, uint64_t raw_marker, uint64_t raw_extent) {
    LimEccClassification ecc;
    if (is_all_ones(raw_wayenable)) {
        return ecc;
    }
    if (lim_backed_bytes(raw_wayenable) == 0) {
        ecc.state = LimEccState::LimUnavailable;
        return ecc;
    }
    if (is_all_ones(raw_marker)) {
        return ecc;
    }
    if (raw_marker != X280_ECC_PRIME_MARKER_MAGIC) {
        ecc.state = LimEccState::Unprimed;
        return ecc;
    }
    if (raw_extent == 0) {
        ecc.state = LimEccState::Primed;
        return ecc;
    }
    if (raw_extent % kLimEccPrimeLineSize != 0 || raw_extent > kLimWayBytes) {
        ecc.state = LimEccState::Unprimed;
        return ecc;
    }
    ecc.state = LimEccState::Primed;
    ecc.primed_bytes = raw_extent;
    return ecc;
}

uint64_t lim_backed_bytes(uint32_t wayenable) {
    const uint32_t ways = wayenable & 0xFu;
    return static_cast<uint64_t>(kMaxLimWays - ways) * kLimWayBytes;
}

static_assert(X280_HART_PROTOCOL_END == 0x08131B00ULL, "hart protocol end is the runtime reservation bound");

LimRange available_lim_for_layout(LimLayout layout, uint64_t backed_bytes) {
    LimRange range;
    if (backed_bytes == 0) {
        return range;
    }
    const uint64_t backed_end = X280_LIM_BASE + backed_bytes;
    if (layout == LimLayout::PacketInjection) {
        if (backed_end < X280_HART_PROTOCOL_END) {
            return range;
        }
        range.begin = X280_LIM_PACKET_INJECTION_ARENA_BASE;
        range.end = X280_LIM_PACKET_INJECTION_ARENA_END;
        return range;
    }

    range.begin = X280_HART_PROTOCOL_END;
    range.end = backed_end > range.begin ? backed_end : range.begin;
    return range;
}

ImageCheck validate_idle_image(uint64_t load_addr, size_t size) {
    ImageCheck check;
    if (size == 0) {
        check.reason = "resident-idle image is empty";
        return check;
    }
    if (load_addr != X280_IDLE_FW_LOAD_ADDR) {
        check.reason = "resident-idle image load address " + hex(load_addr) + " is not the reserved region base " +
                       hex(X280_IDLE_FW_LOAD_ADDR);
        return check;
    }
    // load_addr is the region base by the check above, so bound the size alone:
    // load_addr + size wraps for a size near SIZE_MAX and lands under the bound.
    if (size > kMaxIdleImageBytes) {
        check.reason =
            "resident-idle image is " + std::to_string(size) +
            " bytes and would overrun the reserved region: it must end at or below the ECC prime marker at " +
            hex(kIdleImageEnd) + " (max " + std::to_string(kMaxIdleImageBytes) + " bytes)";
        return check;
    }
    check.ok = true;
    return check;
}

ImageCheck validate_active_image(
    uint64_t load_addr, size_t size, std::optional<uint64_t> primed_bytes, LimLayout layout) {
    ImageCheck check;
    if (size == 0) {
        check.reason = "active image is empty";
        return check;
    }
    if (load_addr != X280_ACTIVE_FW_LOAD_ADDR) {
        check.reason = "active image load address " + hex(load_addr) + " is not the reserved region base " +
                       hex(X280_ACTIVE_FW_LOAD_ADDR);
        return check;
    }
    // Instruction fetch from unprimed LIM faults even when host NoC write/readback
    // succeeds, so the active image must remain inside the primed window.
    const uint64_t layout_image_end =
        layout == LimLayout::PacketInjection ? kPacketInjectionImageEnd : kStackBoundedImageEnd;
    const uint64_t primed = primed_bytes.value_or(kLimEccUnrecordedPrimeBytes);
    const bool primed_window_binds = primed < layout_image_end - X280_LIM_BASE;
    const uint64_t active_image_end = primed_window_binds ? X280_LIM_BASE + primed : layout_image_end;
    const uint64_t max_active_image_bytes =
        active_image_end > X280_ACTIVE_FW_LOAD_ADDR ? active_image_end - X280_ACTIVE_FW_LOAD_ADDR : 0;
    // Bound the size alone rather than load_addr + size, which wraps for a size
    // near SIZE_MAX and lands under the bound.
    if (size > max_active_image_bytes) {
        std::string bound;
        if (primed_window_binds) {
            bound = "the end of the ECC-primed LIM window" +
                    (primed_bytes.has_value() ? std::string(" recorded beside the prime marker")
                                              : ", assumed at " + std::to_string(kLimEccUnrecordedPrimeBytes) +
                                                    " bytes because the prime marker records no extent") +
                    "; a hart cannot fetch instructions past it even though the host can write and read those bytes";
        } else if (layout == LimLayout::PacketInjection) {
            bound = "the PacketInjection layout's image boundary";
        } else {
            bound = "the trap block at " + hex(X280_ACTIVE_TRAP_DIAG_ADDR) + " less all " +
                    std::to_string(kHartsPerTile) + " per-hart stacks";
        }
        check.reason = "active image is " + std::to_string(size) + " bytes and must end at or below " +
                       hex(active_image_end) + " (max " + std::to_string(max_active_image_bytes) +
                       " bytes), which is " + bound;
        return check;
    }
    check.ok = true;
    return check;
}

int hw_l2cpu_from_noc0(uint32_t noc_x, uint32_t noc_y) {
    if (noc_x != 8) {
        return -1;
    }
    switch (noc_y) {
        case 3: return 0;
        case 9: return 1;
        case 5: return 2;
        case 7: return 3;
        default: return -1;
    }
}

std::optional<Noc0Coord> noc0_from_hw_l2cpu(int hw_index) {
    switch (hw_index) {
        case 0: return Noc0Coord{8, 3};
        case 1: return Noc0Coord{8, 9};
        case 2: return Noc0Coord{8, 5};
        case 3: return Noc0Coord{8, 7};
        default: return std::nullopt;
    }
}

int l2cpu_reset_bit(int hw_index) {
    if (hw_index < 0 || hw_index >= kNumL2cpuTiles) {
        return -1;
    }
    return hw_index + 4;
}

}  // namespace x280
