// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Common TensorShape coverage definitions shared by TRISC-specific coverage tables.
//
// Regenerate by running the functional pytests with --logging-level=DEBUG
// and feeding the per-worker test_run_gw*.log files through
// tests/python_tests/helpers/tensor_shape_coverage_parser.py.
//

#pragma once

#include <cstdint>

#include "tensor_shape.h"

// Match the coverage call-site gate so production kernel builds do not see this table.
#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)

namespace ckernel::coverage
{

constexpr TensorShape TENSOR_SHAPE_FR1_NF1x1  = {1, MAX_FACE_C_DIM, 1, 1};  ///<  1x16
constexpr TensorShape TENSOR_SHAPE_FR1_NF1x2  = {1, MAX_FACE_C_DIM, 1, 2};  ///<  1x32
constexpr TensorShape TENSOR_SHAPE_FR1_NF2x1  = {1, MAX_FACE_C_DIM, 2, 1};  ///<  2x16
constexpr TensorShape TENSOR_SHAPE_FR1_NF2x2  = {1, MAX_FACE_C_DIM, 2, 2};  ///<  2x32
constexpr TensorShape TENSOR_SHAPE_FR2_NF1x1  = {2, MAX_FACE_C_DIM, 1, 1};  ///<  2x16
constexpr TensorShape TENSOR_SHAPE_FR2_NF1x2  = {2, MAX_FACE_C_DIM, 1, 2};  ///<  2x32
constexpr TensorShape TENSOR_SHAPE_FR2_NF2x1  = {2, MAX_FACE_C_DIM, 2, 1};  ///<  4x16
constexpr TensorShape TENSOR_SHAPE_FR2_NF2x2  = {2, MAX_FACE_C_DIM, 2, 2};  ///<  4x32
constexpr TensorShape TENSOR_SHAPE_FR4_NF1x1  = {4, MAX_FACE_C_DIM, 1, 1};  ///<  4x16
constexpr TensorShape TENSOR_SHAPE_FR4_NF1x2  = {4, MAX_FACE_C_DIM, 1, 2};  ///<  4x32
constexpr TensorShape TENSOR_SHAPE_FR4_NF2x1  = {4, MAX_FACE_C_DIM, 2, 1};  ///<  8x16
constexpr TensorShape TENSOR_SHAPE_FR4_NF2x2  = {4, MAX_FACE_C_DIM, 2, 2};  ///<  8x32
constexpr TensorShape TENSOR_SHAPE_FR8_NF1x1  = {8, MAX_FACE_C_DIM, 1, 1};  ///<  8x16
constexpr TensorShape TENSOR_SHAPE_FR8_NF1x2  = {8, MAX_FACE_C_DIM, 1, 2};  ///<  8x32
constexpr TensorShape TENSOR_SHAPE_FR8_NF2x1  = {8, MAX_FACE_C_DIM, 2, 1};  ///< 16x16
constexpr TensorShape TENSOR_SHAPE_FR8_NF2x2  = {8, MAX_FACE_C_DIM, 2, 2};  ///< 16x32
constexpr TensorShape TENSOR_SHAPE_FR16_NF1x1 = {16, MAX_FACE_C_DIM, 1, 1}; ///< 16x16
constexpr TensorShape TENSOR_SHAPE_FR16_NF1x2 = {16, MAX_FACE_C_DIM, 1, 2}; ///< 16x32
constexpr TensorShape TENSOR_SHAPE_FR16_NF2x1 = {16, MAX_FACE_C_DIM, 2, 1}; ///< 32x16
constexpr TensorShape TENSOR_SHAPE_FR16_NF2x2 = DEFAULT_TENSOR_SHAPE;       ///< 32x32

enum class TensorShapeFunctionCoverage
{
    _llk_math_eltwise_binary_standard_,
    _llk_math_eltwise_binary_standard_init_,
    _llk_math_eltwise_binary_with_dest_reuse_,
    _llk_math_eltwise_binary_with_dest_reuse_init_,
    _llk_math_reduce_,
    _llk_math_reduce_init_,
    _llk_unpack_AB_init_,
    _llk_unpack_AB_mop_config_,
    _llk_unpack_AB_reduce_init_,
    _llk_unpack_reduce_init_,
    _llk_unpack_AB_reduce_mop_config_,
    _llk_unpack_A_init_,
    _llk_unpack_A_mop_config_,
    eltwise_binary_configure_mop_standard,
    eltwise_binary_configure_mop_with_dest_reuse,
};

constexpr const char* tensor_shape_function_name(const TensorShapeFunctionCoverage fn)
{
    using Function = TensorShapeFunctionCoverage;
    switch (fn)
    {
        case Function::_llk_math_eltwise_binary_standard_:
            return "_llk_math_eltwise_binary_standard_";
        case Function::_llk_math_eltwise_binary_standard_init_:
            return "_llk_math_eltwise_binary_standard_init_";
        case Function::_llk_math_eltwise_binary_with_dest_reuse_:
            return "_llk_math_eltwise_binary_with_dest_reuse_";
        case Function::_llk_math_eltwise_binary_with_dest_reuse_init_:
            return "_llk_math_eltwise_binary_with_dest_reuse_init_";
        case Function::_llk_math_reduce_:
            return "_llk_math_reduce_";
        case Function::_llk_math_reduce_init_:
            return "_llk_math_reduce_init_";
        case Function::_llk_unpack_AB_init_:
            return "_llk_unpack_AB_init_";
        case Function::_llk_unpack_AB_mop_config_:
            return "_llk_unpack_AB_mop_config_";
        case Function::_llk_unpack_AB_reduce_init_:
            return "_llk_unpack_AB_reduce_init_";
        case Function::_llk_unpack_reduce_init_:
            return "_llk_unpack_reduce_init_";
        case Function::_llk_unpack_AB_reduce_mop_config_:
            return "_llk_unpack_AB_reduce_mop_config_";
        case Function::_llk_unpack_A_init_:
            return "_llk_unpack_A_init_";
        case Function::_llk_unpack_A_mop_config_:
            return "_llk_unpack_A_mop_config_";
        case Function::eltwise_binary_configure_mop_standard:
            return "eltwise_binary_configure_mop_standard";
        case Function::eltwise_binary_configure_mop_with_dest_reuse:
            return "eltwise_binary_configure_mop_with_dest_reuse";
    }
    return "unknown";
}

constexpr bool tensor_shape_eq(const TensorShape& lhs, const TensorShape& rhs)
{
    return lhs.face_r_dim == rhs.face_r_dim && lhs.face_c_dim == rhs.face_c_dim && lhs.num_faces_r_dim == rhs.num_faces_r_dim &&
           lhs.num_faces_c_dim == rhs.num_faces_c_dim;
}

constexpr const char* tensor_shape_dim_name(const std::uint8_t dim)
{
    switch (dim)
    {
        case 1:
            return "1";
        case 2:
            return "2";
        case 4:
            return "4";
        case 8:
            return "8";
        case 16:
            return "16";
        default:
            return "unknown";
    }
}

__attribute__((noinline, cold)) inline void assert_tensor_shape_coverage_unobserved_()
{
    LLK_ASSERT(false, "TensorShape not observed before, please add it to the coverage table.");
}

} // namespace ckernel::coverage

#ifdef DEBUG_PRINT_ENABLED
#ifdef ENV_LLK_INFRA
#include "dprint.h"
#else
#include "api/debug/dprint.h"
#endif

// Concatenate fn_name into the literal instead of using CTSTR(fn_name); CTSTR's
// COMDAT string object conflicts with DEVICE_PRINT's own string-section metadata
// at inline template call sites.
#define LLK_VALIDATE_TENSOR_SHAPE_EMIT_(fn, ts)                                                   \
    DEVICE_PRINT(                                                                                 \
        "[{}] tensor_shape: face_r_dim={} face_c_dim={} num_faces_r_dim={} num_faces_c_dim={}\n", \
        ::ckernel::coverage::tensor_shape_function_name(fn),                                      \
        ::ckernel::coverage::tensor_shape_dim_name((ts).face_r_dim),                              \
        ::ckernel::coverage::tensor_shape_dim_name((ts).face_c_dim),                              \
        ::ckernel::coverage::tensor_shape_dim_name((ts).num_faces_r_dim),                         \
        ::ckernel::coverage::tensor_shape_dim_name((ts).num_faces_c_dim))
#else
#define LLK_VALIDATE_TENSOR_SHAPE_EMIT_(fn, ts) ((void)0)
#endif

#define LLK_VALIDATE_TENSOR_SHAPE_WITH_CHECKER(checker, fn, ts)              \
    do                                                                       \
    {                                                                        \
        if (!checker(fn, ts))                                                \
        {                                                                    \
            LLK_VALIDATE_TENSOR_SHAPE_EMIT_(fn, ts);                         \
            ::ckernel::coverage::assert_tensor_shape_coverage_unobserved_(); \
        }                                                                    \
    } while (0)

#define LLK_VALIDATE_TENSOR_SHAPE_UNPACK(fn, ts) LLK_VALIDATE_TENSOR_SHAPE_WITH_CHECKER(::ckernel::coverage::is_unpack_tensor_shape_covered, fn, ts)
#define LLK_VALIDATE_TENSOR_SHAPE_MATH(fn, ts)   LLK_VALIDATE_TENSOR_SHAPE_WITH_CHECKER(::ckernel::coverage::is_math_tensor_shape_covered, fn, ts)
#define LLK_VALIDATE_TENSOR_SHAPE_PACK(fn, ts)   LLK_VALIDATE_TENSOR_SHAPE_WITH_CHECKER(::ckernel::coverage::is_pack_tensor_shape_covered, fn, ts)

#else

#define LLK_VALIDATE_TENSOR_SHAPE_WITH_CHECKER(checker, fn, ts) ((void)0)
#define LLK_VALIDATE_TENSOR_SHAPE_UNPACK(fn, ts)                ((void)0)
#define LLK_VALIDATE_TENSOR_SHAPE_MATH(fn, ts)                  ((void)0)
#define LLK_VALIDATE_TENSOR_SHAPE_PACK(fn, ts)                  ((void)0)

#endif // defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)
