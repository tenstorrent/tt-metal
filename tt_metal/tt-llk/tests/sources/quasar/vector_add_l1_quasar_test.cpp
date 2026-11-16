// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

constexpr std::uint32_t VECTOR_ELEMS = 32;

#ifdef LLK_TRISC_UNPACK

#include "ckernel_vector.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifdef SPEED_OF_LIGHT
    (void)params;
#else
    const std::int32_t* src_a = reinterpret_cast<const std::int32_t*>(params.buffer_A[0]);
    const std::int32_t* src_b = reinterpret_cast<const std::int32_t*>(params.buffer_B[0]);
    std::int32_t* dst         = reinterpret_cast<std::int32_t*>(params.buffer_Res[0]);

    (void)vsetvl<vdatasz::E32, vregmult::M8, VECTOR_ELEMS>();
    vector_load<vreg::V0>(src_a);
    vector_load<vreg::V8>(src_b);
    vector_add<vreg::V16, vreg::V0, vreg::V8>();
    vector_store<vreg::V16>(dst);
#endif
}

#endif

#ifdef LLK_TRISC_MATH

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
}

#endif

#ifdef LLK_TRISC_PACK

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
}

#endif
