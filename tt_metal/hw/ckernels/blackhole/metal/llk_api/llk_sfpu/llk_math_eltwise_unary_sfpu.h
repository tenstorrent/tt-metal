#pragma once
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

SFPU_ROUNDING_KERNEL(floor, 8, false)
SFPU_ROUNDING_KERNEL(floor_float32, 8, true)
SFPU_ROUNDING_KERNEL(ceil, 8, false)
SFPU_ROUNDING_KERNEL(ceil_float32, 8, true)
SFPU_ROUNDING_KERNEL(trunc, 8, false)
SFPU_ROUNDING_KERNEL(trunc_float32, 8, true):
SFPU_UNARY_PARAMS_KERNEL(round, RC, int decimals, decimals)

#include "ckernel_sfpu_cumsum.h"
SFPU_UNARY_PARAMS_KERNEL(cumsum, RC_custom, bool first, first)

#include "ckernel_sfpu_i0.h"
SFPU_UNARY_KERNEL(i0)

#include "ckernel_sfpu_negative.h"
SFPU_UNARY_KERNEL(negative)

#include "ckernel_sfpu_bitwise_or.h"
SFPU_UNARY_KERNEL(bitwise_or)

#include "ckernel_sfpu_cast_fp32_to_fp16a.h"
SFPU_UNARY_KERNEL(cast_fp32_to_fp16a)

#include "ckernel_sfpu_unary_max_min.h"
SFPU_UNARY_PARAMS_KERNEL(unary_max, RC, uint param0, param0)
SFPU_UNARY_PARAMS_KERNEL(unary_min, RC, uint param0, param0)
SFPU_UNARY_INT32_KERNEL(unary_max)

#include "ckernel_sfpu_hardtanh.h"
SFPU_UNARY_PARAMS_KERNEL(hardtanh, RC, uint param0, uint param1, uint param2, param0, param1, param2)

#include "ckernel_sfpu_bitwise_not.h"
SFPU_UNARY_KERNEL(bitwise_not)

#include "ckernel_sfpu_heaviside.h"
SFPU_UNARY_PARAMS_KERNEL(heaviside, RC, uint param0, param0)

#include "ckernel_sfpu_unary_comp.h"
// int32 variants
SFPU_COMP_INT32_KERNEL(ne, unary_ne)
SFPU_COMP_INT32_KERNEL(eq, unary_eq)
SFPU_COMP_INT32_KERNEL(gt, unary_gt)
SFPU_COMP_INT32_KERNEL(lt, unary_lt)

// normal variants
SFPU_COMP_KERNEL(ne)
SFPU_COMP_KERNEL(eq)
SFPU_COMP_KERNEL(gt)
SFPU_COMP_KERNEL(lt)

#include "ckernel_sfpu_clamp.h"
SFPU_UNARY_PARAMS_KERNEL(clamp, RC, uint param0, uint param1, uint param2, param0, param1, param2)

#include "ckernel_sfpu_alt_complex_rotate90.h"
SFPU_UNARY_KERNEL(alt_complex_rotate90)

#include "ckernel_sfpu_abs.h"
SFPU_UNARY_KERNEL(abs)
SFPU_UNARY_INT32_KERNEL(abs)

#include "ckernel_sfpu_comp.h"
SFPU_ZERO_KERNEL(eqz, equal_zero, 8)
SFPU_ZERO_KERNEL(nez, not_equal_zero, 8)
SFPU_ZERO_KERNEL(ltz, less_than_zero, 8)
SFPU_ZERO_KERNEL(gtz, greater_than_zero, 8)
SFPU_ZERO_KERNEL(lez, less_than_equal_zero, 8)
SFPU_ZERO_KERNEL(gez, greater_than_equal_zero, 8)

#include "ckernel_sfpu_exp2.h"
SFPU_INIT_KERNEL(exp2, sfpu::exp2_init)

#include "ckernel_sfpu_expm1.h"
SFPU_INIT_KERNEL(expm1, sfpu::expm1_init)

#include "ckernel_sfpu_log.h"
SFPU_INIT_LITERAL_KERNEL(log, sfpu::log_init, 0)
SFPU_UNARY_PARAMS_KERNEL(log_with_base, RC, uint base_scale, base_scale)

#include "ckernel_sfpu_log1p.h"
SFPU_INIT_KERNEL(log1p, sfpu::log1p_init)

#include "ckernel_sfpu_max.h"
SFPU_UNARY_KERNEL(max)

#include "ckernel_sfpu_power_iterative.h"
SFPU_UNARY_PARAMS_KERNEL(power, RC, int pow, pow)

#include "ckernel_sfpu_rsqrt.h"
SFPU_INIT_KERNEL(rsqrt, sfpu::rsqrt_init)

#include "ckernel_sfpu_sigmoid.h"
SFPU_INIT_KERNEL(sigmoid, sfpu::sigmoid_init)

#include "ckernel_sfpu_sign.h"
SFPU_UNARY_ONE_PARAM_KERNEL(sign, RC, uint exponent_size_8, exponent_size_8)

#include "ckernel_sfpu_signbit.h"
SFPU_UNARY_KERNEL(signbit)

#include "ckernel_sfpu_silu.h"
SFPU_UNARY_KERNEL(silu)

#include "ckernel_sfpu_square.h"
SFPU_UNARY_KERNEL(square)

#include "ckernel_sfpu_tanh.h"
SFPU_INIT_KERNEL(tanh, sfpu::tanh_init)

#include "ckernel_sfpu_tiled_prod.h"
SFPU_UNARY_KERNEL(tiled_prod)

#include "ckernel_sfpu_topk.h"
SFPU_INIT_KERNEL(topk, sfpu::topk_init)
SFPU_UNARY_PARAMS_KERNEL(topk_local_sort, RC_custom,
                         int idir, int i_end_phase, int i_start_phase,
                         int i_end_step, int i_start_step,
                         idir, i_end_phase, i_start_phase, i_end_step, i_start_step)
SFPU_UNARY_PARAMS_KERNEL(topk_merge, RC_custom,
                         int m_iter, int k,
                         m_iter, k)
SFPU_UNARY_PARAMS_KERNEL(topk_rebuild, RC_custom,
                         bool idir, int m_iter, int k, int logk, int skip_second,
                         idir, m_iter, k, logk, skip_second)

#include "ckernel_sfpu_trigonometry.h"
SFPU_TRIG_KERNEL(sine)
SFPU_TRIG_KERNEL(cosine)
SFPU_TRIG_KERNEL(tan)

SFPU_UNARY_KERNEL(asin)
SFPU_UNARY_KERNEL(acos)

SFPU_INVERSE_HYPERBOLIC_KERNEL(acosh, 8)

SFPU_INIT_KERNEL(atan, sfpu::atan_init)
SFPU_UNARY_KERNEL(atan)

SFPU_INVERSE_HYPERBOLIC_KERNEL(asinh, 8)

#include "ckernel_sfpu_fill.h"
SFPU_UNARY_ONE_PARAM_KERNEL(fill, RC, float param0, param0)
SFPU_UNARY_ONE_PARAM_KERNEL(fill_bitcast, RC, uint32_t param0, param0)

#include "ckernel_sfpu_prelu.h"
SFPU_UNARY_ONE_PARAM_KERNEL(prelu, RC, uint param0, param0)

#include "ckernel_sfpu_i1.h"
SFPU_SIMPLE_OP_KERNEL(i1)

#include "ckernel_sfpu_softplus.h"
SFPU_UNARY_PARAMS_KERNEL(
    softplus,
    RC,
    uint param0, uint param1, uint param2,
    param0, param1, param2
)
