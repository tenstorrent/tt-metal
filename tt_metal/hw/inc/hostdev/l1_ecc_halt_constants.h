// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Values of mailboxes_t::l1_ecc_halt.reason and the Blackhole Tensix ECC register fields. Shared by host and
// firmware.
constexpr uint32_t L1_ECC_HALT_REASON_DBE = 0xECC00001;
constexpr uint32_t L1_ECC_HALT_REASON_RISC_MEMORY = 0xECC00002;
constexpr uint32_t L1_ECC_HALT_REASON_UNEXPECTED = 0xECC00003;

// Blackhole Tensix ECC_CTRL fields. The two clears hold while 1 and re-arm on 0, so they are always pulsed.
enum EccCtrl : uint32_t {
    ECC_CTRL_VIEW_MASK = 0xFu,         // [3:0]: ECC_STATUS view select
    ECC_CTRL_COUNTS_CLR = 1u << 4,     // [4]
    ECC_CTRL_IRQ_CLR = 1u << 5,        // [5]
    ECC_CTRL_L1_DBE_IRQ_EN = 1u << 9,  // [9]
    // [19:10]: local data memory DBE [14:10] and instruction cache parity [19:15], each for BRISC, TRISC0-2, NCRISC.
    ECC_CTRL_RISC_MEM_IRQ_EN = 0x3FFu << 10,
};

// ECC_STATUS views, selected by ECC_CTRL[3:0]. There is one last L1 error, in one of the 32 banks; its valid bit
// and row are in both views 3 and 6, and its bank is in whichever view covers it.
enum class EccView : uint32_t {
    // view 1: L1 counts as in view 2 [15:0]; local memory DBE counts for BRISC, TRISC0-2 [31:16]
    L1_AND_LOCAL_MEM_COUNTS = 1,
    // view 2: L1 DBE count [5:0], DBE overflow [6], SBE count [14:7]
    L1_COUNTS = 2,
    // view 3: last L1 error valid [31], banks 0-15 one-hot [27:12], row [11:0]
    LAST_L1_ERROR = 3,
    // view 5: NCRISC local memory DBE count [3:0]; icache parity counts for BRISC, TRISC0-2, NCRISC [23:4]
    ICACHE_AND_NCRISC_LOCAL_MEM_COUNTS = 5,
    // view 6: last L1 error banks 16-31 one-hot [27:12]
    LAST_L1_ERROR_BANKS_16_31 = 6,
};

// ECC_STATUS fields. Masks are in place; a field's value is (status & MASK) >> SHIFT.
enum EccStatus : uint32_t {
    // Views 1 and 2: L1 counts.
    ECC_DBE_COUNT_MASK = 0x3Fu,      // [5:0]
    ECC_DBE_COUNT_OVF_MASK = 0x7Fu,  // [6:0]: count and its overflow bit
    ECC_SBE_COUNT_SHIFT = 7,
    ECC_SBE_COUNT_MASK = 0xFFu << 7,  // [14:7]
    // Views 1 [31:16] and 5 [23:0]: one count (3-bit counter, overflow) per RISC memory.
    ECC_BRISC_LOCAL_MEM_MASK = 0xFu << 16,      // view 1
    ECC_TRISC0_LOCAL_MEM_MASK = 0xFu << 20,     // view 1
    ECC_TRISC1_LOCAL_MEM_MASK = 0xFu << 24,     // view 1
    ECC_TRISC2_LOCAL_MEM_MASK = 0xFu << 28,     // view 1
    ECC_LOCAL_MEM_COUNTS_MASK = 0xFFFFu << 16,  // view 1: all four above
    ECC_NCRISC_LOCAL_MEM_MASK = 0xFu << 0,      // view 5
    ECC_BRISC_ICACHE_MASK = 0xFu << 4,          // view 5
    ECC_TRISC0_ICACHE_MASK = 0xFu << 8,         // view 5
    ECC_TRISC1_ICACHE_MASK = 0xFu << 12,        // view 5
    ECC_TRISC2_ICACHE_MASK = 0xFu << 16,        // view 5
    ECC_NCRISC_ICACHE_MASK = 0xFu << 20,        // view 5
    // Views 3 and 6: last L1 error.
    ECC_LAST_ERR_VALID_MASK = 1u << 31,
    ECC_LAST_ERR_BANKS_SHIFT = 12,
    ECC_LAST_ERR_BANKS_MASK = 0xFFFFu << 12,  // [27:12], one-hot
    ECC_LAST_ERR_ROW_MASK = 0xFFFu,           // [11:0]
};

// L1 has 32 banks of 16 B SECDED words. Rows are 32 words apart in the default bank interleave.
constexpr uint32_t l1_ecc_banks = 32;
constexpr uint32_t l1_ecc_word_bytes = 16;
