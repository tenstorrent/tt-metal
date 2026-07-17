// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Common TensorShape coverage definitions shared by TRISC-specific coverage tables.
//
// Coverage is TRISC-scoped (math / unpack / pack): call sites only pass a string
// literal for DPRINT tagging. There is no central API enum to update when a new
// LLK gains TensorShape validation.
//
// Regenerate the TRISC tables (tensor_shape_coverage_{math,unpack,pack}.h) from
// tests/python_tests with:
//   TT_LLK_DISABLE_ASSERTS=1 pytest --logging-level=DEBUG <tests>
//   python3 helpers/tensor_shape_coverage_parser.py harvest <label>
//   python3 helpers/tensor_shape_coverage_parser.py emit
// Optional: `seed` bootstraps coverage.json from the checked-in headers first.
// Harvest state defaults to tests/python_tests/tensor_shape_coverage/coverage.json
// (override with --coverage-json).
//

#pragma once

#include <cstdint>

#include "tensor_shape.h"

// Match the coverage call-site gate so production kernel builds do not see this table.
#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)

namespace ckernel::coverage
{

// Named shapes referenced by the TRISC coverage tables. Keep this list limited
// to shapes that appear in tensor_shape_coverage_{math,unpack}.h; add a new
// entry only when a harvested shape needs a named constant.
constexpr TensorShape TENSOR_SHAPE_FR1_NF1x1  = {1, MAX_FACE_C_DIM, 1, 1};  ///<  1x16
constexpr TensorShape TENSOR_SHAPE_FR1_NF1x2  = {1, MAX_FACE_C_DIM, 1, 2};  ///<  1x32
constexpr TensorShape TENSOR_SHAPE_FR2_NF1x1  = {2, MAX_FACE_C_DIM, 1, 1};  ///<  2x16
constexpr TensorShape TENSOR_SHAPE_FR2_NF1x2  = {2, MAX_FACE_C_DIM, 1, 2};  ///<  2x32
constexpr TensorShape TENSOR_SHAPE_FR4_NF1x2  = {4, MAX_FACE_C_DIM, 1, 2};  ///<  4x32
constexpr TensorShape TENSOR_SHAPE_FR8_NF1x1  = {8, MAX_FACE_C_DIM, 1, 1};  ///<  8x16
constexpr TensorShape TENSOR_SHAPE_FR8_NF1x2  = {8, MAX_FACE_C_DIM, 1, 2};  ///<  8x32
constexpr TensorShape TENSOR_SHAPE_FR16_NF1x1 = {16, MAX_FACE_C_DIM, 1, 1}; ///< 16x16
constexpr TensorShape TENSOR_SHAPE_FR16_NF1x2 = {16, MAX_FACE_C_DIM, 1, 2}; ///< 16x32
constexpr TensorShape TENSOR_SHAPE_FR16_NF2x1 = {16, MAX_FACE_C_DIM, 2, 1}; ///< 32x16
constexpr TensorShape TENSOR_SHAPE_FR16_NF2x2 = DEFAULT_TENSOR_SHAPE;       ///< 32x32

constexpr bool tensor_shape_eq(const TensorShape& lhs, const TensorShape& rhs)
{
    return lhs.face_r_dim == rhs.face_r_dim && lhs.face_c_dim == rhs.face_c_dim && lhs.num_faces_r_dim == rhs.num_faces_r_dim &&
           lhs.num_faces_c_dim == rhs.num_faces_c_dim;
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

// Concatenate fn_name into the format literal instead of using CTSTR(fn_name);
// CTSTR's COMDAT string object conflicts with DEVICE_PRINT's own string-section
// metadata at inline template call sites. fn_name must be a string literal.
// Dims are printed as integers so DEVICE_PRINT stringifies them (no hand-rolled
// dim->string table / std::to_string on device); the coverage parser harvests
// those digit fields.
#define LLK_VALIDATE_TENSOR_SHAPE_EMIT_(fn_name, ts)                                                       \
    DEVICE_PRINT(                                                                                          \
        "[" fn_name "] tensor_shape: face_r_dim={} face_c_dim={} num_faces_r_dim={} num_faces_c_dim={}\n", \
        static_cast<std::uint32_t>((ts).face_r_dim),                                                       \
        static_cast<std::uint32_t>((ts).face_c_dim),                                                       \
        static_cast<std::uint32_t>((ts).num_faces_r_dim),                                                  \
        static_cast<std::uint32_t>((ts).num_faces_c_dim))
#else
#define LLK_VALIDATE_TENSOR_SHAPE_EMIT_(fn_name, ts) ((void)0)
#endif

#define LLK_VALIDATE_TENSOR_SHAPE_WITH_CHECKER(checker, fn_name, ts)         \
    do                                                                       \
    {                                                                        \
        if (!checker(ts))                                                    \
        {                                                                    \
            LLK_VALIDATE_TENSOR_SHAPE_EMIT_(fn_name, ts);                    \
            ::ckernel::coverage::assert_tensor_shape_coverage_unobserved_(); \
        }                                                                    \
    } while (0)

#define LLK_VALIDATE_TENSOR_SHAPE_UNPACK(fn_name, ts) LLK_VALIDATE_TENSOR_SHAPE_WITH_CHECKER(::ckernel::coverage::is_unpack_tensor_shape_covered, fn_name, ts)
#define LLK_VALIDATE_TENSOR_SHAPE_MATH(fn_name, ts)   LLK_VALIDATE_TENSOR_SHAPE_WITH_CHECKER(::ckernel::coverage::is_math_tensor_shape_covered, fn_name, ts)
#define LLK_VALIDATE_TENSOR_SHAPE_PACK(fn_name, ts)   LLK_VALIDATE_TENSOR_SHAPE_WITH_CHECKER(::ckernel::coverage::is_pack_tensor_shape_covered, fn_name, ts)

#else

#define LLK_VALIDATE_TENSOR_SHAPE_WITH_CHECKER(checker, fn_name, ts) ((void)0)
#define LLK_VALIDATE_TENSOR_SHAPE_UNPACK(fn_name, ts)                ((void)0)
#define LLK_VALIDATE_TENSOR_SHAPE_MATH(fn_name, ts)                  ((void)0)
#define LLK_VALIDATE_TENSOR_SHAPE_PACK(fn_name, ts)                  ((void)0)

#endif // defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)
