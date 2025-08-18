#pragma once

#include "ckernel_sfpu_adam_w.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel {
namespace moreh {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_adam_w_init() {
  llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu::moreh::adam_w_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_adam_w(
    uint dst_index,float lr, float weight_decay,  float beta1, float beta2,
  float recip_bias_correction1, float recip_bias_correction2,
  float eps, int vector_mode = (int)VectorMode::RC) {
  _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
      ckernel::sfpu::moreh::calculate_adam_w<APPROXIMATE>,
      dst_index, vector_mode, lr, weight_decay, beta1, beta2, recip_bias_correction1, recip_bias_correction2, eps);
}

}  // namespace moreh
}  // namespace ckernel
