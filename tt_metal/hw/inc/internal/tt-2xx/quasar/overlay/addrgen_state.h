// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * An address generator has two independent sides, source and destination, with identical loop/bank registers (they
 * share only the MISC register: one bank shift, one bank order field per side). These helpers take the side as a
 * template argument so a walk can be programmed, popped, saved and restored on either.
 *
 * Save/restore lets more walks than there are address-generator sides share them. A walk is what software programmed
 * (AddrgenProgram) plus how far the hardware has advanced it (AddrgenPosition). Hardware moves only the position
 * registers -- BANK_CURRENT, INNER_ADDRESS, OUTER_ADDRESS -- so a save reads back just those three
 * (save_position_addrgen). The caller keeps the program. restore_addrgen writes both into any
 * side of any address generator, which then continues the walk exactly where it stopped.
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

// Address-generator register reads (rd_reg) must not be in flight together: two value-returning RoCC instructions
// outstanding at once can hang the core (AIHWE-6506). The hardware workaround is a no-result RoCC
// instruction between them (rocc_nop); a fence after each read also works but costs more.
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) uint64_t read_reg_addrgen(uint32_t reg_offset) {
    return __builtin_riscv_ttrocc_addrgen_rd_reg(ADDRGEN, reg_offset / 8);
}

// A register read, then rocc_nop (AIHWE-6506): the form every read here uses.
template <AddrGen ADDRGEN>
inline __attribute__((always_inline)) uint64_t read_reg_separated(uint32_t reg_offset) {
    const uint64_t value = read_reg_addrgen<ADDRGEN>(reg_offset);
    rocc_nop();
    return value;
}

// SRC_/DEST_ register offset by side: OVERLAY_AG_REG(SIDE, BANK_CURRENT) etc.
#define OVERLAY_AG_REG(SIDE, name)                                                                     \
    ((SIDE) == ::overlay::Side::Src ? TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_SRC_##name##_REG_OFFSET \
                                    : TT_ROCC_ACCEL_TT_ROCC_CPU0_ADDRESS_GEN_R_DEST_##name##_REG_OFFSET)

template <AddrGen ADDRGEN, Side SIDE>
inline __attribute__((always_inline)) void save_position_addrgen(AddrgenPosition& p) {
    p.bank_current = read_reg_separated<ADDRGEN>(OVERLAY_AG_REG(SIDE, BANK_CURRENT));
    p.inner_address = read_reg_separated<ADDRGEN>(OVERLAY_AG_REG(SIDE, INNER_ADDRESS));
    p.outer_address = read_reg_separated<ADDRGEN>(OVERLAY_AG_REG(SIDE, OUTER_ADDRESS));
}

// Move a side to another position of its current program: write only the three position registers. The program (loop
// strides and ends, banking) is untouched, so this is a cheap restart of the same walk elsewhere.
template <AddrGen ADDRGEN, Side SIDE>
inline __attribute__((always_inline)) void set_position_addrgen(const AddrgenPosition& p) {
    __builtin_riscv_ttrocc_addrgen_wr_reg(ADDRGEN, OVERLAY_AG_REG(SIDE, BANK_CURRENT) / 8, p.bank_current);
    __builtin_riscv_ttrocc_addrgen_wr_reg(ADDRGEN, OVERLAY_AG_REG(SIDE, INNER_ADDRESS) / 8, p.inner_address);
    __builtin_riscv_ttrocc_addrgen_wr_reg(ADDRGEN, OVERLAY_AG_REG(SIDE, OUTER_ADDRESS) / 8, p.outer_address);
}

// Point a single-bank walk at another bank: write only BANK_BASE (the bank loop's first endpoint). With
// set_position_addrgen this moves a walk to another shard without reprogramming its loops.
template <AddrGen ADDRGEN, Side SIDE>
inline __attribute__((always_inline)) void set_bank_base_addrgen(uint32_t base) {
    __builtin_riscv_ttrocc_addrgen_wr_reg(ADDRGEN, OVERLAY_AG_REG(SIDE, BANK_BASE) / 8, base);
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
