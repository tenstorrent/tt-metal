// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TRISC1 runs a taken jump whose branch predictor entry also matches the firmware error handler's mret, in a loop
// that counts. TRISC0 then runs an illegal Tensix instruction. The error interrupts every TRISC in the NEO: TRISC0
// is the culprit and halts in the handler, and TRISC1 must return from the handler and keep counting.
// arg scratch: scratch L1 address, arg scratch_bytes: its size.
// Scratch words: {ready, count, mret address, jump address}. TRISC1 sets ready after its jump has run once, or early,
// leaving the mret address 0 if the handler has no mret or the jump address 0 if no scratch address aliases it.

#include <cstdint>

#include "api/compute/common.h"
#include "experimental/kernel_args.h"
#include "internal/firmware_common.h"
#include "internal/hw_thread.h"
#include "tensix.h"

namespace {

enum : uint32_t { ready_idx, count_idx, mret_address_idx, jump_address_idx, num_results };
// Scratch bytes kept for the result words; the jump is placed after them.
constexpr uint32_t results_bytes = num_results * sizeof(uint32_t);
constexpr uint32_t riscv_mret_insn = 0x30200073;

// Quasar TRISC branch predictor (16 sets): PCs with the same set and tag share an entry.
uint32_t btb_set_and_tag(uint32_t pc) {
    uint32_t set = ((pc >> 2) ^ (pc >> 5) ^ (pc >> 8) ^ (pc >> 11)) & 0xf;
    uint32_t tag = ((pc >> 2) ^ (pc >> 4) ^ (pc >> 6) ^ (pc >> 9)) & 0xff;
    return (tag << 4) | set;
}

// Lets the program complete while this TRISC keeps running.
void signal_done() {
    uint32_t hw_idx = internal_::get_hw_thread_idx();
    volatile tt_l1_ptr uint8_t* const trisc_run =
        &((tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE))->subordinate_sync.map[hw_idx];
    *trisc_run = RUN_SYNC_MSG_DONE;
}

}  // namespace

void kernel_main() {
#if defined(TRISC_MATH)
    uint32_t scratch = get_arg(args::scratch);
    uint32_t scratch_bytes = get_arg(args::scratch_bytes);
    volatile tt_l1_ptr uint32_t* results = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    // The error handler and its mret are in TRISC1 firmware.
    constexpr uint32_t firmware_end = MEM_TRISC1_FIRMWARE_BASE + MEM_TRISC_FIRMWARE_SIZE;
    uint32_t mret_pc = RISC_PIC_BRISC_HW_IVT_BASE(internal_::get_trisc_id())[HW_ERROR_INTERRUPT_INDEX];
    while (mret_pc < firmware_end && *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mret_pc) != riscv_mret_insn) {
        mret_pc += 4;
    }
    if (mret_pc >= firmware_end) {
        results[ready_idx] = 1;
        return;
    }
    results[mret_address_idx] = mret_pc;

    uint32_t alias = 0;
    for (uint32_t a = scratch + results_bytes; a + 12 <= scratch + scratch_bytes; a += 4) {
        if (btb_set_and_tag(a) == btb_set_and_tag(mret_pc)) {
            alias = a;
            break;
        }
    }
    results[jump_address_idx] = alias;
    if (alias == 0) {
        results[ready_idx] = 1;
        return;
    }

    volatile uint32_t* code = reinterpret_cast<volatile uint32_t*>(alias);
    code[0] = 0x0080006f;  // jal x0, 8
    code[1] = 0x00000013;  // nop
    code[2] = 0x00008067;  // ret
    // Read the code back so the stores have landed before it is fetched.
    uint32_t written = code[0] + code[1] + code[2];
    asm volatile("addi zero, %0, 0" ::"r"(written));

    signal_done();
    auto aliased_jump = reinterpret_cast<void (*)()>(alias);
    // The jump installs the predictor entry, so it must run before TRISC0 raises the error.
    aliased_jump();
    results[ready_idx] = 1;
    while (true) {
        aliased_jump();
        results[count_idx] = results[count_idx] + 1;
    }
#elif defined(TRISC_UNPACK)
    volatile tt_l1_ptr uint32_t* results = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::scratch));
    signal_done();
    while (results[ready_idx] == 0) {
    }
    TTI_INSN(TT_OP(0xbc, 0));  // illegal Tensix instruction
#endif
}
