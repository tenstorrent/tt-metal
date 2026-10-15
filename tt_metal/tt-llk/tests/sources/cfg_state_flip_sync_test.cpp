// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Does a configuration-bank flip (SETC16 of the thread's state ID) need a tensix_sync() behind it
// before the thread's next configuration instruction, or do the two NOPs that follow the flip
// already cover every consumer?
//
// Each TRISC seeds the same scratch word in both banks through the RISC (bank chosen by address),
// flips the hardware state ID to bank 1, pads with 0..2 NOPs, optionally drains, and issues a
// read-modify-write of that word. It then flips back to bank 0 the same way and reads the word
// into a GPR with RDCFG. Both consumers pick their bank from the hardware state ID, so a consumer
// that sampled the state ID too early lands in, or reads from, the wrong bank. A MOP of
// read-modify-writes to a second scratch word can be queued ahead of each flip, so the flip and
// its consumer wait behind a long stream of configuration-unit work; those must all land in the
// bank that was selected when they were issued.
//
// Result words per thread, at word offset 256 * thread (0 unpack, 1 math, 2 pack):
//   0        magic
//   1        trials per combination
//   2        1 if instruction gathering was enabled for the run
//   8 + 4c   combination c: trials where the read-modify-write, or a backlog MOP, hit the wrong bank
//   9 + 4c   combination c: trials whose RDCFG read the wrong bank
//   64 + 4c  combination c: the values read in trial 0 (bank 0 word, bank 1 word, RDCFG result)
// with c = 6 * backlog + 2 * nops + sync, backlog in {0,1}, nops in {0,1,2}, sync in {0,1}.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_template.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

namespace
{

using namespace ckernel;

constexpr std::uint32_t RESULT_MAGIC     = 0x43464753;
constexpr std::uint32_t WORDS_PER_THREAD = 256;
constexpr std::uint32_t TRIALS           = 512;
constexpr std::uint32_t COUNT_WORDS      = 8;  // 4 words per combination: rmw wrong, rdcfg wrong, reserved, reserved
constexpr std::uint32_t RAW_WORDS        = 64; // 4 words per combination: trial-0 bank 0, bank 1, RDCFG value, reserved
constexpr std::uint32_t WORDS_PER_COMBO  = 4;

// No LLK runs on any thread in this test, so every GPR is free; 62 receives the RDCFG result.
constexpr std::uint32_t RESULT_GPR = 62;

// The configuration banks are shared by the three TRISCs, so each thread probes its own word. The
// words are address strides that only an unpacker or packer reads, and none runs in this test. They
// sit below GLOBAL_CFGREG_BASE_ADDR32, so each has a separate value per bank. The read-modify-write
// under test touches byte 0 of the word; the backlog MOPs touch byte 1 of the same word.
#ifdef LLK_TRISC_UNPACK
constexpr std::uint32_t PROBE_WORD = UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32;
#endif
#ifdef LLK_TRISC_MATH
constexpr std::uint32_t PROBE_WORD = UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32;
#endif
#ifdef LLK_TRISC_PACK
constexpr std::uint32_t PROBE_WORD = PCK0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32;
#endif
static_assert(PROBE_WORD < GLOBAL_CFGREG_BASE_ADDR32, "the probe word must be banked");

// Word offset of bank 1 in the RISC's view of the configuration space.
constexpr std::uint32_t BANK1_OFFSET = CFG_STATE_SIZE * 4;

constexpr std::uint32_t BANK0          = 0;
constexpr std::uint32_t BANK1          = 1;
constexpr std::uint32_t BACKLOG_OUTER  = 16;
constexpr std::uint32_t BACKLOG_INNER  = 4;
constexpr std::uint32_t BYTE_MASK      = 0xFF;
constexpr std::uint32_t RMW_BYTE       = 0xCC;
constexpr std::uint32_t BACKLOG_BYTE_A = 0x11;
constexpr std::uint32_t BACKLOG_BYTE_B = 0x22;

inline volatile std::uint32_t tt_reg_ptr* cfg_mmio()
{
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(TENSIX_CFG_BASE);
}

inline void seed_word(const std::uint32_t word, const std::uint32_t bank0_value, const std::uint32_t bank1_value)
{
    cfg_mmio()[word]                = bank0_value;
    cfg_mmio()[BANK1_OFFSET + word] = bank1_value;
    // The reads return only after the writes have landed.
    (void)load_blocking(&cfg_mmio()[word]);
    (void)load_blocking(&cfg_mmio()[BANK1_OFFSET + word]);
}

template <std::uint32_t NOPS, bool SYNC>
inline void after_flip()
{
    if constexpr (NOPS >= 1)
    {
        TTI_NOP;
    }
    if constexpr (NOPS >= 2)
    {
        TTI_NOP;
    }
    if constexpr (SYNC)
    {
        tensix_sync();
    }
}

template <bool BACKLOG, std::uint32_t NOPS, bool SYNC>
inline void run_combination(volatile std::uint32_t* out)
{
    ckernel_template backlog_a(BACKLOG_OUTER, BACKLOG_INNER, TT_OP_RMWCIB1(BYTE_MASK /*Mask*/, BACKLOG_BYTE_A /*Data*/, PROBE_WORD));
    ckernel_template backlog_b(BACKLOG_OUTER, BACKLOG_INNER, TT_OP_RMWCIB1(BYTE_MASK /*Mask*/, BACKLOG_BYTE_B /*Data*/, PROBE_WORD));

    std::uint32_t rmw_wrong   = 0;
    std::uint32_t rdcfg_wrong = 0;

    for (std::uint32_t trial = 0; trial < TRIALS; trial++)
    {
        // Bytes 0 and 1 start clear: byte 0 receives the probe read-modify-write (bank 1 only) and
        // byte 1 the backlog MOPs (first run into bank 0, second into bank 1).
        const std::uint32_t seed0 = 0x5A000000u | (trial << 16);
        const std::uint32_t seed1 = 0x25000000u | (trial << 16);
        seed_word(PROBE_WORD, seed0, seed1);
        TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 0 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, LO_16(RESULT_GPR));
        TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 0 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, HI_16(RESULT_GPR));
        tensix_sync();

        // Bank 0 is selected here. Flip to bank 1 and read-modify-write the probe word there.
        if constexpr (BACKLOG)
        {
            backlog_a.program();
            backlog_a.run();
        }
        TTI_SETC16(CFG_STATE_ID_StateID_ADDR32, BANK1);
        after_flip<NOPS, SYNC>();
        TTI_RMWCIB0(BYTE_MASK /*Mask*/, RMW_BYTE /*Data*/, PROBE_WORD);

        // Flip back to bank 0 and read the probe word from there.
        if constexpr (BACKLOG)
        {
            backlog_b.program();
            backlog_b.run();
        }
        TTI_SETC16(CFG_STATE_ID_StateID_ADDR32, BANK0);
        after_flip<NOPS, SYNC>();
        TTI_RDCFG(RESULT_GPR, PROBE_WORD);

        tensix_sync();
        const std::uint32_t probe_bank0 = cfg_mmio()[PROBE_WORD];
        const std::uint32_t probe_bank1 = cfg_mmio()[BANK1_OFFSET + PROBE_WORD];
        const std::uint32_t read_back   = regfile[RESULT_GPR];
        const std::uint32_t expect0     = seed0 | (BACKLOG ? (BACKLOG_BYTE_A << 8) : 0u);
        const std::uint32_t expect1     = seed1 | RMW_BYTE | (BACKLOG ? (BACKLOG_BYTE_B << 8) : 0u);

        if (trial == 0)
        {
            constexpr std::uint32_t c                = 6 * (BACKLOG ? 1 : 0) + 2 * NOPS + (SYNC ? 1 : 0);
            out[RAW_WORDS + WORDS_PER_COMBO * c]     = probe_bank0;
            out[RAW_WORDS + WORDS_PER_COMBO * c + 1] = probe_bank1;
            out[RAW_WORDS + WORDS_PER_COMBO * c + 2] = read_back;
        }
        if (probe_bank0 != expect0 || probe_bank1 != expect1)
        {
            rmw_wrong++;
        }
        if (read_back != expect0)
        {
            rdcfg_wrong++;
        }
    }

    constexpr std::uint32_t combination                  = 6 * (BACKLOG ? 1 : 0) + 2 * NOPS + (SYNC ? 1 : 0);
    out[COUNT_WORDS + WORDS_PER_COMBO * combination]     = rmw_wrong;
    out[COUNT_WORDS + WORDS_PER_COMBO * combination + 1] = rdcfg_wrong;
}

template <bool BACKLOG, std::uint32_t NOPS>
inline void run_nops(volatile std::uint32_t* out)
{
    run_combination<BACKLOG, NOPS, false>(out);
    run_combination<BACKLOG, NOPS, true>(out);
}

template <bool BACKLOG>
inline void run_backlog(volatile std::uint32_t* out)
{
    run_nops<BACKLOG, 0>(out);
    run_nops<BACKLOG, 1>(out);
    run_nops<BACKLOG, 2>(out);
}

inline void run_probe(volatile std::uint32_t* out)
{
    for (std::uint32_t i = 0; i < WORDS_PER_THREAD; i++)
    {
        out[i] = 0;
    }
    out[0] = RESULT_MAGIC;
    out[1] = TRIALS;
    out[2] = GATHERING_ON ? 1u : 0u;

    // Start from a known bank.
    TTI_SETC16(CFG_STATE_ID_StateID_ADDR32, BANK0);
    tensix_sync();

    if constexpr (GATHERING_ON)
    {
        enable_gathering();
    }
    run_backlog<false>(out);
    run_backlog<true>(out);
    if constexpr (GATHERING_ON)
    {
        disable_gathering();
    }

    TTI_SETC16(CFG_STATE_ID_StateID_ADDR32, BANK0);
    tensix_sync();
}

} // namespace

#ifdef LLK_TRISC_UNPACK
constexpr std::uint32_t THREAD_RESULT_OFFSET = 0 * WORDS_PER_THREAD;
#endif
#ifdef LLK_TRISC_MATH
constexpr std::uint32_t THREAD_RESULT_OFFSET = 1 * WORDS_PER_THREAD;
#endif
#ifdef LLK_TRISC_PACK
constexpr std::uint32_t THREAD_RESULT_OFFSET = 2 * WORDS_PER_THREAD;
#endif

void run_kernel(RUNTIME_PARAMETERS params)
{
    volatile std::uint32_t* out = reinterpret_cast<volatile std::uint32_t*>(params.buffer_Res[0]) + THREAD_RESULT_OFFSET;
    run_probe(out);
}
