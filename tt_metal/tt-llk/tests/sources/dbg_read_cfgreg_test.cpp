// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// dbg_read_cfgreg(bank, addr) reads one backend configuration word through the debug read
// registers. The selector it writes is a 32-bit word index into the memory-mapped configuration
// window. On Wormhole that window holds the two thread-agnostic banks (CFG_STATE_SIZE * 4 words
// each) followed by the three thread banks (THD_STATE_SIZE entries each, one 16-byte entry per
// field with the 16-bit value in the low half of the entry's first word). Every bank except
// HW_CFG_0 therefore depends on those sizes being right, and no in-tree caller reads any bank
// but HW_CFG_0. The Blackhole window is laid out differently (a dual-write bank follows the two
// thread-agnostic banks and a thread entry is 4 bytes; see
// tt_llk_blackhole/llk_lib/hal/cfg/detail/register_layout.h), so the layout described here is
// Wormhole's; the kernel itself only names fields both architectures define.
//
// This driver plants known values in every bank and reads each back through the helper:
//
//   HW_CFG_0   control: the active bank, compared against the RISC's own read of the word
//   HW_CFG_1   a sentinel stored through the cfg pointer with cfg_state_id = 1; the word before
//              it holds a second sentinel so an off-by-one selector returns a recognisable value
//   THREAD_n   each TRISC writes two entries of its own thread bank with SETC16 (which can only
//              target the issuing thread's bank): entry 1 with a per-thread value, and the bank's
//              last entry, so a selector with the right bank base but the wrong per-entry stride
//              is caught as well; MATH reads all six
//
// Everything written is put back before the kernel ends: the two bank-1 words are restored to
// the values read before the test, and both thread fields are returned to 0, the value every
// math LLK that uses them programs before consuming them (and, for the first, the value its
// software mirror starts each kernel at).

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "ckernel_debug.h"
#include "ckernel_defs.h"

using namespace ckernel;

// Globals the harness links against.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

namespace
{
// The thread fields under test. Their sentinels stay inside the field's mask so a consumer that
// narrows the stored value could not change what is read back.
constexpr std::uint32_t THREAD_FIELD = DEST_TARGET_REG_CFG_MATH_Offset_ADDR32;
constexpr std::uint16_t T0_VALUE     = 0x0A5;
constexpr std::uint16_t T1_VALUE     = 0x123;
constexpr std::uint16_t T2_VALUE     = 0x5C3;

// The highest thread entry (the last one on Wormhole: THD_STATE_SIZE - 1). Its field is one bit
// wide, so the sentinel is 1 for every thread; telling the threads apart is entry 1's job. The
// only in-tree writer (the reduce LLK) sets it and puts it back to 0.
constexpr std::uint32_t THREAD_FIELD_LAST = FP16A_FORCE_Enable_ADDR32;
constexpr std::uint16_t LAST_VALUE        = 0x1;
static_assert(THREAD_FIELD < THREAD_FIELD_LAST && THREAD_FIELD_LAST < THD_STATE_SIZE, "the second thread probe must be a later entry of the same bank");

// The thread-agnostic word under test, in the inactive bank, and the word before it.
constexpr std::uint32_t BANK1_WORD        = ALU_FORMAT_SPEC_REG2_Dstacc_ADDR32;
constexpr std::uint32_t BANK1_WORD_BEFORE = BANK1_WORD - 1;
constexpr std::uint32_t BANK1_VALUE       = 0x5A1B0C0D;
constexpr std::uint32_t BANK1_BEFORE      = 0x3E0F1E2D;
static_assert(BANK1_WORD >= 1, "the sentinel word needs a predecessor in the same bank");

constexpr std::uint32_t READY = 0x1;
constexpr std::uint32_t DONE  = 0x2;

// Result layout: res[0] = NUM_CHECKS, then NUM_CHECKS quads of {bank, addr, expected, read}.
constexpr std::uint32_t NUM_CHECKS = 8;

// Plants this thread's sentinels, holds them while MATH reads, then clears them.
template <ThreadId thread_id>
inline void hold_thread_fields(const std::uint16_t value)
{
    TT_SETC16(THREAD_FIELD, value);
    TTI_SETC16(THREAD_FIELD_LAST, LAST_VALUE);
    tensix_sync();
    mailbox_write(ThreadId::MathThreadId, READY);
    (void)mailbox_read(ThreadId::MathThreadId);
    TTI_SETC16(THREAD_FIELD, 0); // the firmware drains the thread after run_kernel returns
    TTI_SETC16(THREAD_FIELD_LAST, 0);
}
} // namespace

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS)
{
    hold_thread_fields<ThreadId::UnpackThreadId>(T0_VALUE);
}

#endif

#ifdef LLK_TRISC_MATH

namespace
{
inline void record(std::uint32_t* res, std::uint32_t& n, const std::uint32_t bank, const std::uint32_t addr, const std::uint32_t expected)
{
    res[1 + 4 * n + 0] = bank;
    res[1 + 4 * n + 1] = addr;
    res[1 + 4 * n + 2] = expected;
    res[1 + 4 * n + 3] = dbg_read_cfgreg(bank, addr);
    n++;
}
} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
    auto* res = reinterpret_cast<std::uint32_t*>(params.buffer_Res[0]);

    // Wait for the other two threads to have planted their sentinels.
    (void)mailbox_read(ThreadId::UnpackThreadId);
    (void)mailbox_read(ThreadId::PackThreadId);

    TT_SETC16(THREAD_FIELD, T1_VALUE);
    TTI_SETC16(THREAD_FIELD_LAST, LAST_VALUE);
    tensix_sync();

    // Bank 1 through the LLK's own accessor for it. The RISC may reorder stores to different
    // memory regions, so each store blocks until it has landed, before the debug read register
    // (another region) is written.
    cfg_state_id                                 = 1;
    volatile std::uint32_t tt_reg_ptr* bank1_cfg = get_cfg_pointer();
    reset_cfg_state_id();

    const std::uint32_t saved_word   = bank1_cfg[BANK1_WORD];
    const std::uint32_t saved_before = bank1_cfg[BANK1_WORD_BEFORE];
    store_blocking(&bank1_cfg[BANK1_WORD], BANK1_VALUE);
    store_blocking(&bank1_cfg[BANK1_WORD_BEFORE], BANK1_BEFORE);

    std::uint32_t n = 0;
    record(res, n, dbg_cfgreg::HW_CFG_0, BANK1_WORD, cfg_read(BANK1_WORD));
    record(res, n, dbg_cfgreg::HW_CFG_1, BANK1_WORD, BANK1_VALUE);
    record(res, n, dbg_cfgreg::THREAD_0_CFG, THREAD_FIELD, T0_VALUE);
    record(res, n, dbg_cfgreg::THREAD_1_CFG, THREAD_FIELD, T1_VALUE);
    record(res, n, dbg_cfgreg::THREAD_2_CFG, THREAD_FIELD, T2_VALUE);
    record(res, n, dbg_cfgreg::THREAD_0_CFG, THREAD_FIELD_LAST, LAST_VALUE);
    record(res, n, dbg_cfgreg::THREAD_1_CFG, THREAD_FIELD_LAST, LAST_VALUE);
    record(res, n, dbg_cfgreg::THREAD_2_CFG, THREAD_FIELD_LAST, LAST_VALUE);
    res[0] = n;

    // Put everything back.
    store_blocking(&bank1_cfg[BANK1_WORD], saved_word);
    store_blocking(&bank1_cfg[BANK1_WORD_BEFORE], saved_before);
    TTI_SETC16(THREAD_FIELD, 0); // the firmware drains the thread after run_kernel returns
    TTI_SETC16(THREAD_FIELD_LAST, 0);

    mailbox_write(ThreadId::UnpackThreadId, DONE);
    mailbox_write(ThreadId::PackThreadId, DONE);
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS)
{
    hold_thread_fields<ThreadId::PackThreadId>(T2_VALUE);
}

#endif
