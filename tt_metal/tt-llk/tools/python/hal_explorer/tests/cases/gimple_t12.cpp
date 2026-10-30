// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

using namespace ckernel;
void mm1(unsigned a, unsigned i) {
    regfile[i] = a;                    // GPR write runtime idx
    regfile[3] = a + 1;
    unsigned x = regfile[5];
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, ALU_FORMAT_SPEC_REG0_SrcA_SHAMT, ALU_FORMAT_SPEC_REG0_SrcA_MASK>(a);
    TT_SETDMAREG(0, x & 0xff, 0, 0);
    volatile uint32_t *p = (volatile uint32_t *)(0x1000);
    p[2] = a;
    reg_write(RISCV_DEBUG_REG_DBG_BUS_CTRL, a);
}
void mm2(unsigned a) {
    asm volatile("nop\n\tnop\n\taddi x0, x0, 0" ::: "memory");
    asm volatile("fence" ::: "memory");
    TTI_NOP;
    asm volatile("addi %0, %1, 0" : "=r"(a) : "r"(a));
    TT_SETDMAREG(0, a, 0, 0);
}
