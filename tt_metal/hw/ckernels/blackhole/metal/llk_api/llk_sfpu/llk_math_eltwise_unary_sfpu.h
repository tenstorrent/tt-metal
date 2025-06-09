#pragma once
#include "sfpu_macros.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

#include "ckernel_sfpu_cumsum.h"
SFPU_CUSTOM_UNARY_KERNEL(cumsum, RC_custom, bool first, first)

#include "ckernel_sfpu_i0.h"
SFPU_UNARY_KERNEL(i0)

#include "ckernel_sfpu_negative.h"
SFPU_UNARY_KERNEL(negative)

#include "ckernel_sfpu_bitwise_or.h"
SFPU_UNARY_KERNEL(bitwise_or)

#include "ckernel_sfpu_cast_fp32_to_fp16a.h"
SFPU_UNARY_KERNEL(cast_fp32_to_fp16a)

#include "ckernel_sfpu_unary_max_min.h"
SFPU_CUSTOM_UNARY_KERNEL(unary_max, RC, uint param0, param0)
SFPU_CUSTOM_UNARY_KERNEL(unary_min,  RC, uint param0, param0)

#include "ckernel_sfpu_hardtanh.h"
SFPU_CUSTOM_UNARY_KERNEL(
    hardtanh,
    RC,
    uint param0, uint param1, uint param2,
    param0, param1, param2)

#include "ckernel_sfpu_bitwise_not.h"
SFPU_UNARY_KERNEL(bitwise_not)

#include "ckernel_sfpu_heaviside.h"
SFPU_CUSTOM_UNARY_KERNEL(heaviside, RC, uint param0, param0)

#include "ckernel_sfpu_unary_comp.h"
// int32 variants
SFPU_COMP_INT32_KERNEL(ne, unary_ne)
SFPU_COMP_INT32_KERNEL(eq, unary_eq)

// normal variants
SFPU_COMP_KERNEL(ne)
SFPU_COMP_KERNEL(eq)
SFPU_COMP_KERNEL(gt)
SFPU_COMP_KERNEL(lt)

#include "llk_math_eltwise_unary_sfpu_log.h"

#include "ckernel_sfpu_clamp.h"
SFPU_CUSTOM_UNARY_KERNEL(clamp, RC, uint param0, uint param1, uint param2, param0, param1, param2)

#include "ckernel_sfpu_alt_complex_rotate90.h"
SFPU_UNARY_KERNEL(alt_complex_rotate90)
