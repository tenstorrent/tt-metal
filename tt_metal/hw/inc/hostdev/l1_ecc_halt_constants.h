// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Values of l1_ecc_halt_msg_t::reason and the Blackhole Tensix ECC and ARC PMM register fields. Shared by host and
// firmware.
constexpr uint32_t L1_ECC_HALT_REASON_DBE = 0xECC00001;
constexpr uint32_t L1_ECC_HALT_REASON_UNEXPECTED = 0xECC00003;

// Blackhole Tensix ECC_CTRL fields. The two clears act while held at 1, so they are always pulsed.
enum EccCtrl : uint32_t {
    ECC_CTRL_VIEW_MASK = 0xFu,         // [3:0]: ECC_STATUS view select
    ECC_CTRL_COUNTS_CLR = 1u << 4,     // [4]: also clears the last error's valid bit
    ECC_CTRL_IRQ_CLR = 1u << 5,        // [5]
    ECC_CTRL_L1_DBE_IRQ_EN = 1u << 9,  // [9]
};

// ECC_STATUS views, selected by ECC_CTRL[3:0]. There is one last L1 error, in one of the 32 banks; its valid bit
// and row are in both views 3 and 6, and its bank is in whichever view covers it.
enum class EccView : uint32_t {
    // view 1: L1 counts as in view 2 [15:0]; local memory DBE counts for BRISC, TRISC0-2 [31:16]
    L1_AND_LOCAL_MEM_COUNTS = 1,
    // view 2: L1 DBE count [5:0], DBE overflow [6], SBE count [14:7]
    L1_COUNTS = 2,
    // view 3: last L1 error valid [31], banks 0-15 one-hot [27:12], row [11:0]
    LAST_L1_ERROR_BANKS_0_15 = 3,
    // view 5: NCRISC local memory DBE count [3:0]; icache parity counts for BRISC, TRISC0-2, NCRISC [23:4]
    ICACHE_AND_NCRISC_LOCAL_MEM_COUNTS = 5,
    // view 6: last L1 error banks 16-31 one-hot [27:12]
    LAST_L1_ERROR_BANKS_16_31 = 6,
};

// ECC_STATUS fields. Masks are in place; a field's value is (status & MASK) >> SHIFT.
enum EccStatus : uint32_t {
    // Views 1 and 2: L1 counts.
    ECC_DBE_COUNT_MASK = 0x3Fu,  // [5:0]
    ECC_SBE_COUNT_SHIFT = 7,
    ECC_SBE_COUNT_MASK = 0xFFu << 7,  // [14:7]
    // Views 3 and 6: last L1 error.
    ECC_LAST_ERR_VALID_MASK = 1u << 31,
    ECC_LAST_ERR_BANKS_SHIFT = 12,
    ECC_LAST_ERR_BANKS_MASK = 0xFFFFu << 12,  // [27:12], one-hot
    ECC_LAST_ERR_ROW_MASK = 0xFFFu,           // [11:0]
};

// Blackhole PIC: the handler table (RISC_PIC_INT_PCS) and BRISC_INT_NO number the software IRQs first, then the
// hardware IRQs.
constexpr uint32_t pic_num_sw_ints = 32;
constexpr uint32_t pic_num_hw_ints = 4;
constexpr uint32_t pic_hw_int_err = 3;  // hardware IRQ raised by the tile's ECC errors and its watchdog

constexpr uint32_t riscv_mret_insn = 0x30200073;    // mret: return from the interrupt handler
constexpr uint32_t riscv_j_back_insn = 0xffdff06f;  // j .-4: jump back to the previous instruction
// Returns from an interrupt handler without using the stack. The jump retries an mret the branch predictor skipped.
constexpr uint32_t brisc_int_release_code[2] = {riscv_mret_insn, riscv_j_back_insn};

// L1 has 32 banks of 16 B SECDED words. Rows are 32 words apart in the default bank interleave.
constexpr uint32_t l1_ecc_banks = 32;
constexpr uint32_t l1_ecc_word_bytes = 16;

// l1_ecc_halt_msg_t is an ARC tile event record. ARC forwards one with this magic to the host event ring.
constexpr uint32_t l1_ecc_halt_record_magic = 0x54455654;  // "TEVT"
constexpr uint16_t l1_ecc_halt_event_type = 0x0002;        // provisional: not yet reserved in ARC firmware
// The PMM message carries the record address shifted right by 9.
constexpr uint32_t l1_ecc_halt_record_alignment = 512;

// ARC's PMM (power management messaging) block. With NOC translation on, ARC has these coordinates on both NOCs. A
// message selects its sender's mailbox word by the NOC0 x/y it carries; the word keeps the message's low 16 bits.
constexpr uint32_t arc_noc_x = 8;
constexpr uint32_t arc_noc_y = 0;
constexpr uint32_t arc_pmm_message_addr = 0x80400000;
constexpr uint32_t arc_pmm_mailbox_addr = 0x80400800;  // word for (x, y) at + 4 * (arc_pmm_grid_x * y + x)
constexpr uint32_t arc_pmm_grid_x = 17;
constexpr uint32_t arc_pmm_mailbox_mask = 0xFFFF;
constexpr uint32_t arc_pmm_flag0 = 1u << 12;

// The message announcing a halt record at record_addr (flags = arc_pmm_flag0), or, with flags = 0 and record_addr =
// 0, the one clearing (x, y)'s mailbox word.
constexpr uint32_t arc_pmm_message(uint32_t x, uint32_t y, uint32_t flags, uint32_t record_addr) {
    return x << 26 | y << 20 | flags | record_addr / l1_ecc_halt_record_alignment;
}
