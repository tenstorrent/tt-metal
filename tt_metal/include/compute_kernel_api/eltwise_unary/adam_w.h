#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {
namespace moreh {

ALWI void adam_w_init() {
  MATH((llk_math_eltwise_unary_sfpu_adam_w_init<APPROX>()));
}

ALWI void adam_w(std::uint32_t idst,float lr, float weight_decay,  float beta1, float beta2,
  float recip_bias_correction1, float recip_bias_correction2, float eps ) {
  MATH((llk_math_eltwise_unary_sfpu_adam_w<APPROX>(
      idst, lr, weight_decay, beta1, beta2, recip_bias_correction1, recip_bias_correction2, eps)));
}

}  // namespace moreh
}  // namespace ckernel
