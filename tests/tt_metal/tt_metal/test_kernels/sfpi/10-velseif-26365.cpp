// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ckernel.h"
#include "compute_kernel_api.h"
#include <sfpi.h>

// various v_if/v_elseif/v_else tests

using namespace sfpi;
namespace NAMESPACE {
volatile uint32_t global __attribute__((used)) = 0x12345678;
volatile uint32_t zero __attribute__((used));

// Do not let compiler propagate knowledge of the initialized value;
static uint32_t __attribute__((noinline)) get(volatile uint32_t* ptr) { return *ptr; }

static void __attribute__((noinline)) compc_26365() {
    vFloat val = l_reg[LRegs::LReg3];
    vUInt result = 0;

    v_if(val < 1.0f) { result = 1; }
    v_elseif(val <= 2.0f) { result = 2; }
    v_endif;

    l_reg[LRegs::LReg3] = result;
}

void MAIN {
#if COMPILE_FOR_TRISC == 1  // compute
#include "pre.inc"
    {
        l_reg[LRegs::LReg3] = vFloat(0f);
        compc_26365() vUInt res = l_reg[LRegs::LReg3];
        FAIL_IF(res != 1);
    }
    {
        l_reg[LRegs::LReg3] = vFloat(1f);
        compc_26365() vUInt res = l_reg[LRegs::LReg3];
        FAIL_IF(res != 2);
    }
    {
        l_reg[LRegs::LReg3] = vFloat(2f);
        compc_26365() vUInt res = l_reg[LRegs::LReg3];
        FAIL_IF(res != 2);
    }
    {
        l_reg[LRegs::LReg3] = vFloat(3f);
        compc_26365() vUInt res = l_reg[LRegs::LReg3];
        FAIL_IF(res != 0);
    }
#include "post.inc"
#endif
}
}  // namespace NAMESPACE
