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
        vFloat v = 0.0f;
        l_reg[LRegs::LReg3] = v;
        compc_26365();
        vUInt res = l_reg[LRegs::LReg3];
        FAIL_IF(res != 1);
    }
    {
        vFloat v = 1.0f;
        l_reg[LRegs::LReg3] = v;
        compc_26365();
        vUInt res = l_reg[LRegs::LReg3];
        FAIL_IF(res != 2);
    }
    {
        vFloat v = 2.0f;
        l_reg[LRegs::LReg3] = v;
        compc_26365();
        vUInt res = l_reg[LRegs::LReg3];
        FAIL_IF(res != 2);
    }
    {
        vFloat v = 3.0f;
        l_reg[LRegs::LReg3] = v;
        compc_26365();
        vUInt res = l_reg[LRegs::LReg3];
        FAIL_IF(res != 0);
    }
#include "post.inc"
#endif
}
}  // namespace NAMESPACE
