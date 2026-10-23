// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Raises false L1 ECC errors with a NoC read past the end of L1, each right after a taken jump whose branch predictor
// entry also matches the L1 ECC handler's mret, so the predictor predicts that mret as taken.
// arg 0: source L1 address past the end of L1, arg 1: destination L1 address, arg 2: bytes per read,
// arg 3: scratch L1 address, arg 4: scratch bytes, arg 5: iterations.
// Writes the mret address and the jump address to the first two scratch words; 0 means not found, and the kernel
// then returns without raising any error.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "hostdev/l1_ecc_halt_constants.h"

namespace {

// RISC_PIC_INT_PCS index of the IRQ 3 handler address.
constexpr uint32_t pic_irq3_handler = pic_num_sw_ints + pic_hw_int_err;
// Scratch bytes kept for the two result words; the jump is placed after them.
constexpr uint32_t results_bytes = 16;

// Blackhole branch predictor: set pc[7:2], tag pc[31:24] ^ pc[23:16] ^ pc[15:8].
uint32_t btb_tag(uint32_t pc) { return ((pc >> 24) ^ (pc >> 16) ^ (pc >> 8)) & 0xff; }

}  // namespace

void kernel_main() {
    uint32_t src = get_arg_val<uint32_t>(0);
    uint32_t dst = get_arg_val<uint32_t>(1);
    uint32_t bytes = get_arg_val<uint32_t>(2);
    uint32_t scratch = get_arg_val<uint32_t>(3);
    uint32_t scratch_bytes = get_arg_val<uint32_t>(4);
    uint32_t iterations = get_arg_val<uint32_t>(5);

    volatile tt_l1_ptr uint32_t* results = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    results[0] = 0;
    results[1] = 0;

    // The IRQ 3 handler and its mret are in BRISC firmware.
    constexpr uint32_t firmware_end = MEM_BRISC_FIRMWARE_BASE + MEM_BRISC_FIRMWARE_SIZE;
    uint32_t mret_pc = RISC_PIC_INT_PCS[pic_irq3_handler];
    while (mret_pc < firmware_end && *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mret_pc) != riscv_mret_insn) {
        mret_pc += 4;
    }
    if (mret_pc >= firmware_end) {
        return;
    }
    results[0] = mret_pc;

    // Same set (same low byte) and same tag, inside the scratch area.
    uint32_t alias = 0;
    uint32_t first = scratch + results_bytes;
    for (uint32_t a = (first & ~0xffu) | (mret_pc & 0xff); a + 12 <= scratch + scratch_bytes; a += 0x100) {
        if (a >= first && btb_tag(a) == btb_tag(mret_pc)) {
            alias = a;
            break;
        }
    }
    if (alias == 0) {
        return;
    }
    results[1] = alias;

    volatile tt_l1_ptr uint32_t* code = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(alias);
    code[0] = 0x0080006f;  // jal x0, 8
    code[1] = 0x00000013;  // nop
    code[2] = 0x00008067;  // ret
    // Read the code back so the stores have landed before it is fetched.
    uint32_t written = code[0] + code[1] + code[2];
    asm volatile("addi zero, %0, 0" ::"r"(written));
    auto aliased_jump = reinterpret_cast<void (*)()>(alias);

    Noc noc;
    UnicastEndpoint self;
    CoreLocalMem<uint32_t> dst_mem(dst);
    for (uint32_t n = 0; n < iterations; n++) {
        aliased_jump();
        noc.async_read(
            self, dst_mem, bytes, {.noc_x = my_x[noc.get_noc_id()], .noc_y = my_y[noc.get_noc_id()], .addr = src}, {});
        noc.async_read_barrier();
    }
}
