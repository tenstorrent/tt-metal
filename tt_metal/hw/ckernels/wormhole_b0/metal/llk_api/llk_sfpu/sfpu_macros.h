#pragma once
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

#define SFPU_UNARY_KERNEL(OP)                                            \
namespace ckernel {                                                      \
  template<bool APPROXIMATE>                                             \
  inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>();       \
  }                                                                      \
  template<bool APPROXIMATE>                                             \
  inline void llk_math_eltwise_unary_sfpu_##OP(                          \
      uint dst_index, int vector_mode = (int)VectorMode::RC) {          \
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                     \
      ckernel::sfpu::calculate_##OP<APPROXIMATE>,                        \
      dst_index, vector_mode);                                           \
  }                                                                      \
}
