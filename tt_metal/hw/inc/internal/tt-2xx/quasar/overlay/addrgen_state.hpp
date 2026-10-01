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
 * Save/restore lets more walks than there are address-generator sides share them. A walk is what software programmed
 * (AddrgenProgram) plus how far the hardware has advanced it (AddrgenPosition). Hardware moves only the position
 * registers -- BANK_CURRENT, INNER_ADDRESS, OUTER_ADDRESS -- so a save reads back just those three
 * (save_position_addrgen); the program is the caller's to keep, as it wrote it. restore_addrgen writes both into any
 * side of any address generator, which then continues the walk exactly where it stopped. This is the address-generator
 * part of the HW team's command-buffer context-switch table ("Qsr cmd buffers"). FACE_SIZE is not part of the program:
 * callers that program it must restore it themselves.
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

// Everything software programs for a walk on one side; hardware never changes it. banking.current is not part of it
// (that is position). banking.endpoint_id_shift lands in MISC, which both sides of an address generator share.
struct AddrgenProgram {
    BankingConfig banking;
    uint64_t inner_stride, inner_end;
    uint64_t outer_stride, outer_end;
};

// The registers hardware advances while a walk runs.
struct AddrgenPosition {
    uint64_t inner_address;
    uint64_t outer_address;
    uint32_t bank_current;
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

template <AddrGen ADDRGEN, Side SIDE>
inline __attribute__((always_inline)) void save_position_addrgen(AddrgenPosition& p) {
    p.bank_current = read_reg_fenced<ADDRGEN>(OVERLAY_AG_REG(SIDE, BANK_CURRENT));
    p.inner_address = read_reg_fenced<ADDRGEN>(OVERLAY_AG_REG(SIDE, INNER_ADDRESS));
    p.outer_address = read_reg_fenced<ADDRGEN>(OVERLAY_AG_REG(SIDE, OUTER_ADDRESS));
}

template <AddrGen ADDRGEN, Side SIDE>
inline __attribute__((always_inline)) void restore_addrgen(const AddrgenProgram& prog, const AddrgenPosition& pos) {
    BankingConfig banking = prog.banking;
    banking.current = pos.bank_current;
    setup_banking_addrgen<ADDRGEN, SIDE>(banking);
    setup_inner_loop_addrgen<ADDRGEN, SIDE>(prog.inner_stride, prog.inner_end, pos.inner_address);
    setup_outer_loop_addrgen<ADDRGEN, SIDE>(prog.outer_stride, prog.outer_end, pos.outer_address);
}

#undef OVERLAY_AG_REG

}  // namespace overlay
