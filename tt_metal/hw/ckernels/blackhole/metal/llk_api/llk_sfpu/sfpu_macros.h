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

#define SFPU_CUSTOM_UNARY_KERNEL(OP, MODE, EXTRA_ARG_DECL, EXTRA_ARG_PASS) \
namespace ckernel {                                                      \
  template<bool APPROXIMATE>                                             \
  inline void llk_math_eltwise_unary_sfpu_##OP##_init() {                \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(         \
      sfpu::OP##_init<APPROXIMATE>);                                     \
  }                                                                      \
  template<bool APPROXIMATE>                                             \
  inline void llk_math_eltwise_unary_sfpu_##OP(                          \
      uint dst_index, EXTRA_ARG_DECL,                                    \
      int vector_mode = (int)VectorMode::MODE) {                        \
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                     \
      ckernel::sfpu::calculate_##OP<APPROXIMATE>,                        \
      dst_index, vector_mode, EXTRA_ARG_PASS);                           \
  }                                                                      \
}

// For the int32 comparison variants (uses calculate_comp_unary_int<…, SfpuType>)
#define SFPU_COMP_INT32_KERNEL(OP, TYPE)                                  \
namespace ckernel {                                                      \
  template<bool APPROXIMATE>                                             \
  inline void llk_math_eltwise_unary_sfpu_unary_##OP##_int32(             \
      uint dst_index, uint param0,                                       \
      int vector_mode = (int)VectorMode::RC) {                           \
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                     \
      ckernel::sfpu::calculate_comp_unary_int<                           \
        APPROXIMATE, SfpuType::TYPE>,                                    \
      dst_index, vector_mode, param0);                                   \
  }                                                                       \
}

// For the “normal” comparison ops
#define SFPU_COMP_KERNEL(OP)                                              \
namespace ckernel {                                                      \
  template<bool APPROXIMATE>                                             \
  inline void llk_math_eltwise_unary_sfpu_unary_##OP##_init() {           \
    llk_math_eltwise_unary_sfpu_init<SfpuType::unary_##OP,                \
                                      APPROXIMATE>();                    \
  }                                                                       \
  template<bool APPROXIMATE>                                             \
  inline void llk_math_eltwise_unary_sfpu_unary_##OP(                     \
      uint dst_index, uint param0,                                       \
      int vector_mode = (int)VectorMode::RC) {                           \
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(                     \
      ckernel::sfpu::calculate_unary_##OP<APPROXIMATE>,                  \
      dst_index, vector_mode, param0);                                   \
  }                                                                       \
}

