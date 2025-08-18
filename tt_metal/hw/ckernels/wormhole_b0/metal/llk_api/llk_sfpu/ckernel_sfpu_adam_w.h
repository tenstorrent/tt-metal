#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {
namespace moreh {

template <bool APPROXIMATION_MODE, int RECIPROCAL_ITERATIONS>
sfpi_inline sfpi::vFloat _calculate_sqrt_body_(sfpi::vFloat val)
{
    sfpi::vFloat result;
    if constexpr (APPROXIMATION_MODE)
    {
        sfpi::vUInt magic = sfpi::vConstIntPrgm0;
        sfpi::vUInt val_s = magic + sfpi::reinterpret<sfpi::vUInt>(val);
        val_s >>= 1;
        result = sfpi::reinterpret<sfpi::vFloat>(val_s);
    }
    else
    {
        v_if (val != 0.0f)
        {
            sfpi::vUInt magic   = sfpi::vConstIntPrgm0;
            sfpi::vFloat approx = sfpi::reinterpret<sfpi::vFloat>(magic - (sfpi::reinterpret<sfpi::vUInt>(val) >> 1));
            for (int r = 0; r < RECIPROCAL_ITERATIONS; r++)
            {
                approx = ((approx * approx) * (val * -0.5f) + 1.5f) * approx;
            }
            result = approx * val;
        }
        v_else
        {
            result = val;
        }
        v_endif;
    }
    return result;
}

template <int max_iter = 3>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in)
{
    sfpi::vFloat val = sfpi::setsgn(in, 1);
    val = setexp(val, 126);
    sfpi::vFloat vConstLn2Recip = sfpi::vConstFloatPrgm1;
    sfpi::vFloat two            = sfpi::vConstFloatPrgm2;
    sfpi::vFloat result         = vConstLn2Recip * (val * vConstLn2Recip + two);
    for (int s_iter = 0; s_iter < (max_iter - 1); s_iter++)
    {
        result = result * (val * result + two);
    }
    sfpi::vInt orig_exp = exexp(in);
    sfpi::vInt new_exp  = exexp(result);
    new_exp -= orig_exp;
    new_exp += 126;
    v_if (new_exp < 0)
    {
        result  = 0.0F;
        new_exp = 0;
    }
    v_endif;
    vFloat ret = setexp(result, new_exp);
    return ret;
}

template <bool APPROXIMATION_MODE>
void adam_w_init() {
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_adam_w(float lr, float weight_decay, float beta1, float beta2,
  float recip_bias_correction1, float recip_bias_correction2, float eps) {
#pragma GCC unroll 8
  for (int d = 0; d < ITERATIONS; d++) {
    vFloat grad = dst_reg[32];
    vFloat exp_avg = dst_reg[64];
    vFloat exp_avg_sq = dst_reg[96];
    vFloat m_coef0 = beta1;
    vFloat m_coef1 = 1 - beta1;
    vFloat v_coef0 = beta2;
    vFloat v_coef1 = 1 - beta2;
    vFloat m_new = m_coef0 * exp_avg + m_coef1 * grad;
    vFloat v_new = v_coef0 * exp_avg_sq + v_coef1 * (grad * grad);
    vFloat p_coef = 1 - lr * weight_decay;
    vFloat param_in = dst_reg[0];
    vFloat param_new = param_in * p_coef;
    vFloat recip_bias_correction1 = recip_bias_correction1;
    vFloat recip_bias_correction2 = recip_bias_correction2;
    vFloat m_hat = m_new * recip_bias_correction1;
    vFloat v_hat = v_new * recip_bias_correction2;
    vFloat sqrt_v_hat = _calculate_sqrt_body_<APPROXIMATION_MODE, /*RECIPROCAL_ITERATIONS=*/2>(v_hat);
    vFloat denom = sqrt_v_hat + eps;
    vFloat recip_denom = _sfpu_reciprocal_<APPROXIMATION_MODE ? 2 : 3>(denom);
    vFloat param_out = param_new - (lr * m_hat) * recip_denom;
    dst_reg[0] = m_new;
    dst_reg[32] = v_new;
    dst_reg++;
  }
}

}  // namespace moreh
}  // namespace sfpu
}  // namespace ckernel
