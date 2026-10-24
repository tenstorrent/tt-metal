// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// dbg_read_cfgreg(bank, addr) reads one backend configuration word through the RISC debug
// selector, whose index space is the RISC's own word view of TENSIX_CFG_BASE: the two state-CFG
// banks, the dual-write window, then the three thread-CFG banks. The bank bases it adds to
// `addr` must therefore follow this architecture's CFG_STATE_SIZE / THD_STATE_SIZE, or a read of
// "state bank 1" or "thread n" returns some other register's word.
//
// The math thread plants a known value where each read must land and reads it back:
//   - a state-CFG word in bank 1 (the inactive bank; written through the bank-1 MMIO window,
//     expected = the MMIO read-back of the same word)
//   - a second bank-1 word far from the first, so a wrong base cannot pass by coincidence
//   - the same word in bank 0 as a control (base 0 on every layout)
//   - its own thread-CFG word, set with SETC16 (thread 1 is the math TRISC)
// State words are restored afterwards; the thread word is reset to its boot value 0. No LLK runs on
// any thread, so the words are free to use.
//
// Result words (math thread only):
//   0          magic
//   1          number of probes
//   8 + 4p     probe p: identifier (bank << PROBE_ID_BANK_SHIFT | addr)
//   9 + 4p     probe p: expected value
//   10 + 4p    probe p: value dbg_read_cfgreg returned

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "ckernel_debug.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    // idle
}

#endif

#ifdef LLK_TRISC_MATH

using namespace ckernel;

namespace
{

// Must match python_tests/test_dbg_read_cfgreg_layout.py
constexpr std::uint32_t RESULT_MAGIC        = 0x43464744; // "CFGD"
constexpr std::uint32_t RESULT_WORDS        = 64;
constexpr std::uint32_t PROBE_BASE          = 8;
constexpr std::uint32_t WORDS_PER_PROBE     = 4;
constexpr std::uint32_t PROBE_ID_BANK_SHIFT = 16;

// Word offset of state bank 1 in the RISC's view of the configuration space.
constexpr std::uint32_t BANK1_OFFSET = CFG_STATE_SIZE * 4;

// The word the dprint helpers read most (ALU format and accumulation control) and an unpacker-1
// stride word no idle unit consults; both are banked (below GLOBAL_CFGREG_BASE_ADDR32).
constexpr std::uint32_t ALU_WORD    = ALU_ACC_CTRL_Fp32_enabled_ADDR32;
constexpr std::uint32_t STRIDE_WORD = UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32;
static_assert(ALU_WORD < GLOBAL_CFGREG_BASE_ADDR32 && STRIDE_WORD < GLOBAL_CFGREG_BASE_ADDR32, "probe words must be banked");

// Patterns no reset value or neighbouring register is likely to hold.
constexpr std::uint32_t ALU_PATTERN    = 0x5A3C96E1;
constexpr std::uint32_t STRIDE_PATTERN = 0xC3A51E96;
constexpr std::uint32_t BANK0_PATTERN  = 0x3C5AE196;
// Within the 12-bit DEST_TARGET_REG_CFG_MATH_Offset field, the only field of that thread word.
constexpr std::uint32_t THREAD_PATTERN = 0xABC;
static_assert((THREAD_PATTERN & ~DEST_TARGET_REG_CFG_MATH_Offset_MASK) == 0, "thread pattern must fit its field");
// What every LLK op re-establishes before it uses the dest offset.
constexpr std::uint32_t THREAD_BOOT_VALUE = 0;
// A thread-CFG word is a 16-bit value and 16 bits of padding; only the value half is compared, as
// the HAL's thread-word read does.
constexpr std::uint32_t THREAD_WORD_MASK = 0xffff;

// This TRISC's thread-CFG bank, as dbg_read_cfgreg numbers them.
constexpr std::uint32_t OWN_THREAD_CFG = dbg_cfgreg::THREAD_1_CFG;

inline volatile std::uint32_t tt_reg_ptr* cfg_mmio()
{
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(TENSIX_CFG_BASE);
}

inline void record(
    volatile std::uint32_t* out,
    const std::uint32_t probe,
    const std::uint32_t bank,
    const std::uint32_t addr,
    const std::uint32_t expected,
    const std::uint32_t got)
{
    out[PROBE_BASE + WORDS_PER_PROBE * probe]     = (bank << PROBE_ID_BANK_SHIFT) | addr;
    out[PROBE_BASE + WORDS_PER_PROBE * probe + 1] = expected;
    out[PROBE_BASE + WORDS_PER_PROBE * probe + 2] = got;
}

// Plant `pattern` in the state-CFG word at RISC word index `mmio_index`, read it through the debug
// selector as (`bank`, `addr`), then put the original back. Expected is what the MMIO window reads
// back, so bits the register does not implement are not counted against the debug read.
inline void probe_state_word(
    volatile std::uint32_t* out,
    const std::uint32_t probe,
    const std::uint32_t bank,
    const std::uint32_t addr,
    const std::uint32_t mmio_index,
    const std::uint32_t pattern)
{
    const std::uint32_t original = load_blocking(&cfg_mmio()[mmio_index]);
    cfg_mmio()[mmio_index]       = pattern;
    const std::uint32_t expected = load_blocking(&cfg_mmio()[mmio_index]);

    const std::uint32_t got = dbg_read_cfgreg(bank, addr);

    cfg_mmio()[mmio_index] = original;
    (void)load_blocking(&cfg_mmio()[mmio_index]);

    record(out, probe, bank, addr, expected, got);
}

} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
    volatile std::uint32_t* out = reinterpret_cast<volatile std::uint32_t*>(params.buffer_Res[0]);
    for (std::uint32_t i = 0; i < RESULT_WORDS; i++)
    {
        out[i] = 0;
    }
    out[0] = RESULT_MAGIC;

    std::uint32_t probe = 0;

    // State bank 1 is idle (this thread never flips its state ID), so its words are safe to plant in.
    probe_state_word(out, probe++, dbg_cfgreg::HW_CFG_1, ALU_WORD, BANK1_OFFSET + ALU_WORD, ALU_PATTERN);
    probe_state_word(out, probe++, dbg_cfgreg::HW_CFG_1, STRIDE_WORD, BANK1_OFFSET + STRIDE_WORD, STRIDE_PATTERN);
    // Control: bank 0 starts at index 0 on every layout, so this read agrees with MMIO regardless.
    probe_state_word(out, probe++, dbg_cfgreg::HW_CFG_0, STRIDE_WORD, STRIDE_WORD, BANK0_PATTERN);

    // Own thread-CFG word: SETC16 is the only writer, and nothing else runs on this thread, so it is
    // put back to its boot value rather than read first.
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, THREAD_PATTERN);
    tensix_sync();
    const std::uint32_t thread_got = dbg_read_cfgreg(OWN_THREAD_CFG, DEST_TARGET_REG_CFG_MATH_Offset_ADDR32) & THREAD_WORD_MASK;
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, THREAD_BOOT_VALUE);
    tensix_sync();
    record(out, probe++, OWN_THREAD_CFG, DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, THREAD_PATTERN, thread_got);

    out[1] = probe;
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    // idle
}

#endif
