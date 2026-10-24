// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compile with the stock and aligned SFPI header/compiler tuples, WH and BH.
namespace ckernel
{
volatile unsigned long* instrn_buffer;
}

#if defined(ARCH_BLACKHOLE)
#include "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_bf16_sfpi_isa.h"
#else
#include "tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_bf16_sfpi_isa.h"
#endif

using namespace ckernel::sfpu::bf16_sfpi;

void bf16_float_round()
{
    sfp_stoch_rnd(0, 0, 1, 0, 0, 1);
}

void bf16_integer_immediate_descale()
{
    sfp_stoch_rnd(0, 2, 1, 0, 0, 12);
}

void bf16_integer_vector_descale()
{
    sfp_stoch_rnd(0, 0, 1, 0, 0, 4);
}
