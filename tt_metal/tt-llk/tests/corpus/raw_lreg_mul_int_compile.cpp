// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compile-only instantiation of the production Blackhole multiply LLK.
namespace ckernel {
inline volatile unsigned long instrn_buffer[1];
}
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "llk_defs.h"
#include "sfpu/ckernel_sfpu_mul_int.h"

void compile_mul_int_llk()
{
    ckernel::sfpu::_init_mul_int_<false>();
    ckernel::sfpu::_mul_int_<false, 1>(0, 1, 2);
}
