// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file addrgen_state.hpp
 * @brief Side-generic address generator helpers, and save/restore of one side's walk
 *
 * An address generator has two independent sides, source and destination, with identical loop/bank registers (they
 * share only the MISC register: one bank shift, one bank order field per side). These helpers take the side as a
 * template argument so a walk can be programmed, popped, saved and restored on either.
 *
 * Save/restore lets more walks than there are address-generator sides share them: a parked walk's registers are read
 * back (save_state_addrgen) and later written into any side of any address generator (restore_state_addrgen), which
 * then continues the walk exactly where it stopped. FACE_SIZE is not part of the state: callers that program it must
 * save it themselves.
 */

#pragma once

#include "addrgen_api.hpp"

namespace overlay {

enum class Side : uint32_t { Src = 0, Dest = 1 };

template <AddrGen ADDRGEN, Side SIDE>
inline __attribute__((always_inline)) void setup_banking_addrgen(const BankingConfig& cfg) {
    if constexpr (SIDE == Side::Src) {
        setup_src_banking_addrgen<ADDRGEN>(cfg);
    } else {
        setup_dest_banking_addrgen<ADDRGEN>(cfg);
    }
}

template <AddrGen ADDRGEN, Side SIDE>
inline __attribute__((always_inline)) void setup_inner_loop_addrgen(uint64_t stride, uint64_t end, uint64_t start) {
    if constexpr (SIDE == Side::Src) {
        setup_src_inner_loop_addrgen<ADDRGEN>(stride, end, start);
    } else {
        setup_dest_inner_loop_addrgen<ADDRGEN>(stride, end, start);
    }
}

template <AddrGen ADDRGEN, Side SIDE>
inline __attribute__((always_inline)) void setup_outer_loop_addrgen(uint64_t stride, uint64_t end, uint64_t start) {
    if constexpr (SIDE == Side::Src) {
        setup_src_outer_loop_addrgen<ADDRGEN>(stride, end, start);
    } else {
        setup_dest_outer_loop_addrgen<ADDRGEN>(stride, end, start);
    }
}

// Returns the side's current address and advances it by `amount` addresses (1 = next).
template <AddrGen ADDRGEN, Side SIDE>
inline __attribute__((always_inline)) uint64_t pop_addrgen(uint64_t amount) {
    if constexpr (SIDE == Side::Src) {
        return pop_src_addrgen<ADDRGEN>(amount);
    } else {
        return pop_dest_addrgen<ADDRGEN>(amount);
    }
}

// Kept compact (56 bytes): callers hold these in thread-local storage, which shares a DM core's 8 KB with its stack.
struct AddrgenState {
    uint64_t inner_stride, inner_end, inner_address;
    uint64_t outer_stride, outer_end, outer_address;
    // BANK_CURRENT, BANK_BASE, BANK_SIZE, BANK_SKIP and MISC's bank_offset (6 bits each) and this side's bank order
    // (2 bits).
    uint32_t banking;

    static constexpr uint32_t pack(
        uint32_t current, uint32_t base, uint32_t size, uint32_t skip, uint32_t offset, uint32_t order) {
        return (current & 0x3F) | (base & 0x3F) << 6 | (size & 0x3F) << 12 | (skip & 0x3F) << 18 |
               (offset & 0x3F) << 24 | (order & 0x3) << 30;
    }
    constexpr uint32_t bank_current() const { return banking & 0x3F; }
    constexpr uint32_t bank_base() const { return (banking >> 6) & 0x3F; }
    constexpr uint32_t bank_size() const { return (banking >> 12) & 0x3F; }
    constexpr uint32_t bank_skip() const { return (banking >> 18) & 0x3F; }
    constexpr uint32_t bank_offset() const { return (banking >> 24) & 0x3F; }
    constexpr uint32_t bank_order() const { return banking >> 30; }
};

// Register read, then a fence. Back-to-back rd_reg instructions hung the address generator on emu-quasar-2x3 (the same
// reads spaced apart returned correct values); a fence between them avoids it. Only save pays this.
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) uint64_t read_reg_fenced(uint32_t reg_offset) {
    const uint64_t value = __builtin_riscv_ttrocc_addrgen_rd_reg(ADDRGEN, reg_offset / 8);
    asm volatile("fence" ::: "memory");
    return value;
}

// SRC_/DEST_ register offset by side: OVERLAY_AG_REG(SIDE, BANK_CURRENT) etc.
#define OVERLAY_AG_REG(SIDE, name)                                                                     \
    ((SIDE) == ::overlay::Side::Src ? TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_##name##_REG_OFFSET \
                                    : TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_##name##_REG_OFFSET)

// Fills `s` in place: no temporary copy of the struct, which matters where stack is tight (see AddrgenState).
template <AddrGen ADDRGEN, Side SIDE>
inline __attribute__((always_inline)) void save_state_addrgen(AddrgenState& s) {
    const uint32_t current = read_reg_fenced<ADDRGEN>(OVERLAY_AG_REG(SIDE, BANK_CURRENT));
    const uint32_t base = read_reg_fenced<ADDRGEN>(OVERLAY_AG_REG(SIDE, BANK_BASE));
    const uint32_t size = read_reg_fenced<ADDRGEN>(OVERLAY_AG_REG(SIDE, BANK_SIZE));
    const uint32_t skip = read_reg_fenced<ADDRGEN>(OVERLAY_AG_REG(SIDE, BANK_SKIP));
    s.inner_stride = read_reg_fenced<ADDRGEN>(OVERLAY_AG_REG(SIDE, INNER_STRIDE));
    s.inner_end = read_reg_fenced<ADDRGEN>(OVERLAY_AG_REG(SIDE, INNER_END));
    s.inner_address = read_reg_fenced<ADDRGEN>(OVERLAY_AG_REG(SIDE, INNER_ADDRESS));
    s.outer_stride = read_reg_fenced<ADDRGEN>(OVERLAY_AG_REG(SIDE, OUTER_STRIDE));
    s.outer_end = read_reg_fenced<ADDRGEN>(OVERLAY_AG_REG(SIDE, OUTER_END));
    s.outer_address = read_reg_fenced<ADDRGEN>(OVERLAY_AG_REG(SIDE, OUTER_ADDRESS));
    TT_ROCC_ADDRESS_GEN_MISC_reg_u misc;
    misc.val = read_reg_fenced<ADDRGEN>(TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_MISC_REG_OFFSET);
    s.banking = AddrgenState::pack(
        current,
        base,
        size,
        skip,
        misc.f.bank_offset,
        SIDE == Side::Src ? misc.f.src_bank_order : misc.f.dst_bank_order);
}

template <AddrGen ADDRGEN, Side SIDE>
inline __attribute__((always_inline)) void restore_state_addrgen(const AddrgenState& s) {
    setup_banking_addrgen<ADDRGEN, SIDE>(BankingConfig{
        .endpoint_id_shift = s.bank_offset(),
        .size = s.bank_size(),
        .skip = s.bank_skip(),
        .base = s.bank_base(),
        .current = s.bank_current(),
        .bank_order = static_cast<bank_order_e>(s.bank_order()),
    });
    setup_inner_loop_addrgen<ADDRGEN, SIDE>(s.inner_stride, s.inner_end, s.inner_address);
    setup_outer_loop_addrgen<ADDRGEN, SIDE>(s.outer_stride, s.outer_end, s.outer_address);
}

#undef OVERLAY_AG_REG

}  // namespace overlay
