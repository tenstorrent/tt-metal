// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Runs only the SrcA/SrcB unpack init of a binary op (default) or a matmul (MATMUL_INIT) and returns.
// With LLK asserts on and an operand marked UnpackToDestFp32 the unpack thread trips inside the init;
// the math and pack threads issue no FPU or pack work and finish, so the core is left with no Tensix
// instruction stalled behind the halted unpack thread and the device recovers for the next program.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#ifdef MATMUL_INIT
#include "api/compute/matmul.h"
#else
#include "api/compute/eltwise_binary.h"
#endif

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;
#ifdef MATMUL_INIT
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_init(cb_in0, cb_in1);
#else
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);
    binary_tiles_init<true, EltwiseBinaryType::ELWADD>(cb_in0, cb_in1);
#endif
}
