// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Raises a software interrupt on TRISC0 right after a taken jump whose branch predictor entry also matches the
// handler's mret, so the predictor predicts that mret as taken. The handler retries a skipped mret and counts it.
// arg scratch: scratch L1 address, arg scratch_bytes: its size, arg iterations: interrupts to raise.
// Writes {interrupts handled, mrets skipped, iterations not finished, jump address} to the first four scratch
// words; a jump address of 0 means no scratch address aliases the mret, and nothing else is written.

#include <cstdint>

#include "api/compute/common.h"
#include "experimental/kernel_args.h"
#include "internal/hw_thread.h"
#include "tensix.h"

#ifdef TRISC_UNPACK

extern "C" {
volatile uint32_t* sw_int_reg;
volatile uint32_t interrupts_handled;
volatile uint32_t mrets_skipped;
extern const uint32_t mret_alias_isr_mret[];
}

// Called from mret_alias_isr.
extern "C" __attribute__((noinline, used)) void mret_alias_handle() {
    (void)*sw_int_reg;  // reading SW_INT clears the interrupt
    interrupts_handled = interrupts_handled + 1;
}

// SW interrupt 0 handler. Saves the caller-saved registers around mret_alias_handle. If the mret is skipped, the
// code after it counts the skip and runs the mret again.
__attribute__((naked)) void mret_alias_isr() {
    asm volatile(
        "addi sp, sp, -(16 * 4)\n"
        "sw ra, 0 * 4(sp)\n"
        "sw t0, 1 * 4(sp)\n"
        "sw t1, 2 * 4(sp)\n"
        "sw t2, 3 * 4(sp)\n"
        "sw t3, 4 * 4(sp)\n"
        "sw t4, 5 * 4(sp)\n"
        "sw t5, 6 * 4(sp)\n"
        "sw t6, 7 * 4(sp)\n"
        "sw a0, 8 * 4(sp)\n"
        "sw a1, 9 * 4(sp)\n"
        "sw a2, 10 * 4(sp)\n"
        "sw a3, 11 * 4(sp)\n"
        "sw a4, 12 * 4(sp)\n"
        "sw a5, 13 * 4(sp)\n"
        "sw a6, 14 * 4(sp)\n"
        "sw a7, 15 * 4(sp)\n"
        "call mret_alias_handle\n"
        "lw ra, 0 * 4(sp)\n"
        "lw t0, 1 * 4(sp)\n"
        "lw t1, 2 * 4(sp)\n"
        "lw t2, 3 * 4(sp)\n"
        "lw t3, 4 * 4(sp)\n"
        "lw t4, 5 * 4(sp)\n"
        "lw t5, 6 * 4(sp)\n"
        "lw t6, 7 * 4(sp)\n"
        "lw a0, 8 * 4(sp)\n"
        "lw a1, 9 * 4(sp)\n"
        "lw a2, 10 * 4(sp)\n"
        "lw a3, 11 * 4(sp)\n"
        "lw a4, 12 * 4(sp)\n"
        "lw a5, 13 * 4(sp)\n"
        "lw a6, 14 * 4(sp)\n"
        "lw a7, 15 * 4(sp)\n"
        "addi sp, sp, 16 * 4\n"
        ".global mret_alias_isr_mret\n"
        "mret_alias_isr_mret:\n"
        "mret\n"
        "addi sp, sp, -(2 * 4)\n"
        "sw t0, 0 * 4(sp)\n"
        "sw t1, 1 * 4(sp)\n"
        "la t0, mrets_skipped\n"
        "lw t1, 0(t0)\n"
        "addi t1, t1, 1\n"
        "sw t1, 0(t0)\n"
        "lw t0, 0 * 4(sp)\n"
        "lw t1, 1 * 4(sp)\n"
        "addi sp, sp, 2 * 4\n"
        "j mret_alias_isr_mret\n");
}

namespace {

// Scratch bytes kept for the four result words; the jump is placed after them.
constexpr uint32_t results_bytes = 16;
// Loop passes to wait for one interrupt before giving up.
constexpr uint32_t spin_cap = 100000;

// Quasar TRISC branch predictor (16 sets): PCs with the same set and tag share an entry.
uint32_t btb_set_and_tag(uint32_t pc) {
    uint32_t set = ((pc >> 2) ^ (pc >> 5) ^ (pc >> 8) ^ (pc >> 11)) & 0xf;
    uint32_t tag = ((pc >> 2) ^ (pc >> 4) ^ (pc >> 6) ^ (pc >> 9)) & 0xff;
    return (tag << 4) | set;
}

}  // namespace

#endif

void kernel_main() {
#ifdef TRISC_UNPACK
    uint32_t scratch = get_arg(args::scratch);
    uint32_t scratch_bytes = get_arg(args::scratch_bytes);
    uint32_t iterations = get_arg(args::iterations);

    volatile uint32_t* results = reinterpret_cast<volatile uint32_t*>(scratch);
    results[3] = 0;

    // Bits 16:2 of the PC set the hash, so any 128 KB holds a match.
    uint32_t mret_key = btb_set_and_tag(reinterpret_cast<uint32_t>(mret_alias_isr_mret));
    uint32_t alias = 0;
    for (uint32_t a = scratch + results_bytes; a + 12 <= scratch + scratch_bytes; a += 4) {
        if (btb_set_and_tag(a) == mret_key) {
            alias = a;
            break;
        }
    }
    if (alias == 0) {
        return;
    }

    volatile uint32_t* code = reinterpret_cast<volatile uint32_t*>(alias);
    code[0] = 0x0080006f;  // jal x0, 8
    code[1] = 0x00000013;  // nop
    code[2] = 0x00008067;  // ret
    // Read the code back so the stores have landed before it is fetched.
    uint32_t written = code[0] + code[1] + code[2];
    asm volatile("addi zero, %0, 0" ::"r"(written));

    uint32_t trisc_id = internal_::get_trisc_id();
    sw_int_reg = RISC_PIC_BRISC_SW_INT_REG(trisc_id);
    interrupts_handled = 0;
    mrets_skipped = 0;
    RISC_PIC_BRISC_SW_IVT_BASE(trisc_id)[0] = reinterpret_cast<uint32_t>(mret_alias_isr);
    *RISC_PIC_BRISC_SW_INT_EN(trisc_id) = 1;

    auto aliased_jump = reinterpret_cast<void (*)()>(alias);
    for (; iterations > 0; iterations--) {
        aliased_jump();
        uint32_t before = interrupts_handled;
        *sw_int_reg = 1;
        uint32_t spins = spin_cap;
        while (interrupts_handled == before && --spins != 0) {
        }
        if (spins == 0) {
            break;
        }
    }

    *RISC_PIC_BRISC_SW_INT_EN(trisc_id) = 0;

    results[0] = interrupts_handled;
    results[1] = mrets_skipped;
    results[2] = iterations;
    results[3] = alias;
#endif
}
