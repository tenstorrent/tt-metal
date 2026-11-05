// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Runs only the SrcA/SrcB unpack init of one FPU op and returns: a binary add (default), a matmul (MATMUL_INIT),
// a column reduce with the scaler CB as operand B (REDUCE_INIT), or a tilize-A/unpack-B reduce with the scaler
// CB as operand B (TILIZE_A_B_INIT; the host defines REDUCE_OP and REDUCE_DIM).
// With LLK asserts on and an operand marked UnpackToDestFp32 the unpack thread trips inside the init;
// the math and pack threads issue no FPU or pack work and finish, so the core is left with no Tensix
// instruction stalled behind the halted unpack thread and the device recovers for the next program.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#if defined(MATMUL_INIT)
#include "api/compute/matmul.h"
#elif defined(REDUCE_INIT)
#include "api/compute/reduce.h"
#elif defined(TILIZE_A_B_INIT)
#include "api/compute/reduce.h"
#include "api/compute/tilize.h"
#else
#include "api/compute/eltwise_binary.h"
#endif

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;
#if defined(MATMUL_INIT)
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_init(cb_in0, cb_in1);
#elif defined(REDUCE_INIT)
    // A column reduce does not swap the operands: in0 (data) is operand A, in1 (scaler) is operand B.
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_in0, cb_in1, cb_out);
#elif defined(TILIZE_A_B_INIT)
    constexpr uint32_t block_ct_dim = 1;
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);
    tilizeA_B_reduce_init<true /*neginf_srcA*/, false /*zero_srcA_reduce*/>(cb_in0, cb_in1, block_ct_dim);
#else
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);
    binary_tiles_init<true /*full_init*/, EltwiseBinaryType::ELWADD>(cb_in0, cb_in1);
#endif
}
