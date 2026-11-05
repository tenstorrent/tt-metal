// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Regression test for a Blackhole TRISC hang in ckernel::disable_gathering().
//
// Turning gathering off while fetch is on a wrong path into code the TRISC has already gathered hangs the
// TRISC for good. MATH sets that up on purpose against the library's disable_gathering()/enable_gathering():
//  1. It runs a block of 64 Tensix NOPs twice, so it is gathered.
//  2. It trains a stale branch prediction for the instruction right after disable_gathering()'s gathering-off
//     write, with the NOP block as its target. The prediction comes from a `jal` written into spare L1 at an
//     address that the branch predictor cannot tell apart from that instruction.
//  3. It stalls execute on GFR_LEAD dependent L1 loads just before disable_gathering(), so fetch runs ahead
//     and meets the stale prediction early. Without a fix, fetch follows it into the NOP block as gathering
//     goes off, the TRISC never gets back, and the test times out.

#include <cstdint>

#include "ckernel.h"
#include "llk_assert.h"
#include "params.h"

std::uint32_t unp_cfg_context        = 0;
std::uint32_t pack_sync_tile_dst_ptr = 0;

#ifdef LLK_TRISC_UNPACK
void run_kernel(RUNTIME_PARAMETERS)
{
}
#endif

#ifdef LLK_TRISC_PACK
void run_kernel(RUNTIME_PARAMETERS)
{
}
#endif

#ifdef LLK_TRISC_MATH

#define GFR_STR2(x) #x
#define GFR_STR(x)  GFR_STR2(x)
#define GFR_TTNOP4  ".4byte 0x08000000\n .4byte 0x08000000\n .4byte 0x08000000\n .4byte 0x08000000\n"
#define GFR_TTNOP16 GFR_TTNOP4 GFR_TTNOP4 GFR_TTNOP4 GFR_TTNOP4

// gfr_hot: 64 Tensix NOPs, then ret.
asm(".section .text.gfr,\"ax\",@progbits\n"
    ".option push\n"
    ".option norvc\n"
    ".balign 64\n"
    ".global gfr_hot\n"
    "gfr_hot:\n" GFR_TTNOP16 GFR_TTNOP16 GFR_TTNOP16 GFR_TTNOP16
    "ret\n"
    ".option pop\n"
    ".text\n");

extern "C" void gfr_hot();

#ifndef GFR_LEAD
#define GFR_LEAD 0
#endif

// A word that holds its own address: a chain of loads through it stalls execute while fetch runs ahead.
volatile std::uint32_t gfr_self __attribute__((used));

// The bracket under test, as load_replay_buf() emits it around a record, after GFR_LEAD dependent loads.
[[gnu::noinline, gnu::aligned(64)]] void gfr_bracket()
{
    std::uint32_t p = reinterpret_cast<std::uint32_t>(&gfr_self);
    asm volatile(".rept " GFR_STR(GFR_LEAD) "\n lw %0, 0(%0)\n .endr\n addi %0, %0, 0\n" : "+r"(p) : : "memory");
    ckernel::disable_gathering();
    ckernel::enable_gathering();
}

// RISC-V jal x0, off (J-type).
static std::uint32_t encode_jal_x0(std::int32_t off)
{
    std::uint32_t imm = static_cast<std::uint32_t>(off);
    return (((imm >> 20) & 1) << 31) | (((imm >> 1) & 0x3ff) << 21) | (((imm >> 11) & 1) << 20) | (((imm >> 12) & 0xff) << 12) | 0x6f;
}

// csrrs/csrrc zero, 0x7c0, <any rs1>.
static bool is_cfg0_csr(std::uint32_t insn, std::uint32_t funct3)
{
    return (insn & 0xfff07fffu) == ((0x7c0u << 20) | (funct3 << 12) | 0x73u);
}

// The PC right after gathering goes off: the last csrrs 0x7c0 before the first csrrc 0x7c0 is the bit-18 write.
static std::uint32_t after_gathering_off()
{
    const volatile std::uint32_t *code = reinterpret_cast<const volatile std::uint32_t *>(&gfr_bracket);
    std::uint32_t last_set             = 0;
    for (std::uint32_t i = 0; i < 256; i++)
    {
        const std::uint32_t insn = code[i];
        if (is_cfg0_csr(insn, 0b010))
        {
            last_set = reinterpret_cast<std::uint32_t>(&code[i + 1]);
        }
        else if (is_cfg0_csr(insn, 0b011) && last_set != 0)
        {
            return last_set;
        }
    }
    return 0;
}

volatile std::uint32_t gfr_debug[4] __attribute__((used));

void run_kernel(RUNTIME_PARAMETERS)
{
    const std::uint32_t x   = after_gathering_off();
    const std::uint32_t hot = reinterpret_cast<std::uint32_t>(&gfr_hot);
    gfr_debug[0]            = x;
    LLK_ASSERT(x != 0, "gathering-off write not found in gfr_bracket");

    // TRISC branch predictor: 64 sets x 4 ways, index pc[7:2], tag pc[15:8] ^ pc[23:16] ^ pc[31:24].
    // Alias of x in spare L1 at 0x100000..0x10ffff: pc[23:16] = 0x10, pc[15:8] = tag(x) ^ 0x10, pc[7:2] = x's.
    const std::uint32_t tag_x = ((x >> 8) ^ (x >> 16) ^ (x >> 24)) & 0xff;
    const std::uint32_t y     = (0x10u << 16) | (((tag_x ^ 0x10u) & 0xff) << 8) | (x & 0xfc);
    gfr_debug[1]              = y;
    gfr_debug[2]              = hot;
    gfr_self                  = reinterpret_cast<std::uint32_t>(&gfr_self);

    // The alias: a jal at y whose target is the NOP block (which returns to our caller via ra).
    *reinterpret_cast<volatile std::uint32_t *>(y) = encode_jal_x0(static_cast<std::int32_t>(hot - y));
    asm volatile("fence" ::: "memory");

    // Warm the instruction cache on the bracket: fetch must run ahead of its first CSR write for the stale
    // prediction to be made in time.
    gfr_bracket();

    for (std::uint32_t i = 0; i < 64; i++)
    {
        // Gather the NOP block (each disable below flushes what was gathered).
        gfr_hot();
        gfr_hot();
        // Train the alias (the misprediction in the bracket drops it each time).
        reinterpret_cast<void (*)()>(y)();
        reinterpret_cast<void (*)()>(y)();
        gfr_bracket();
        gfr_debug[3] = i;
    }
    gfr_debug[3] = 0x600d;
}

#endif
