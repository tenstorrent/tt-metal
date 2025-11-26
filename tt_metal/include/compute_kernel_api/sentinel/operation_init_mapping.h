// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

/**
 * @file operation_init_mapping.h
 * @brief Template-based mapping system for automatically calling init functions for operations.
 *
 * This system maps operations to their corresponding init functions using compile-time templates.
 * Only operations actually used in a kernel will be instantiated, ensuring minimal overhead.
 *
 * Usage:
 *   call_init_for<OperationType::SIGMOID_TILE>();
 *   call_init_for<OperationType::ADD_TILES>(icb0, icb1);
 *   call_init_for<OperationType::EMA_TILE>(alpha, beta);
 */

/**
 * @brief Enumeration of all compute kernel operations.
 * Each operation maps to its corresponding init function.
 */
enum class OperationType : uint32_t {
    // Operations with no init
    NONE = 0,

    // Mathematical Functions
    SIGMOID_TILE,
    LOG_TILE,
    LOG_WITH_BASE_TILE,
    TANH_TILE,
    EXP2_TILE,
    EXPM1_TILE,

    // Sign and Absolute Value
    SIGNBIT_TILE,
    SIGNBIT_TILE_INT32,
    ABS_TILE,
    ABS_TILE_INT32,
    SIGN_TILE,

    // Power and Square
    SQUARE_TILE,
    POWER_TILE,
    TILED_PROD_TILE,

    // Max/Min Operations
    MAX_TILE,
    UNARY_MAX_TILE,
    UNARY_MAX_INT32_TILE,
    UNARY_MIN_TILE,
    UNARY_MIN_INT32_TILE,

    // Special Functions
    HEAVISIDE_TILE,
    SILU_TILE,
    ALT_COMPLEX_ROTATE90_TILE,

    // TopK Operations
    TOPK_LOCAL_SORT,
    TOPK_MERGE,
    TOPK_REBUILD,

    // Reduction Operations
    MAX_REDUCE_WITH_INDICES,
    SFPU_REDUCE,
    SFPU_ADD_TOP_ROW,

    // Add Int SFPU
    ADD_INT32_TILE,
    ADD_UINT16_TILE,
    ADD_UINT32_TILE,

    // Broadcast Operations
    UNARY_BCAST,
    SUB_TILES_BCAST_COLS,
    SUB_TILES_BCAST_SCALAR,
    MUL_TILES_BCAST_COLS,
    MUL_TILES_BCAST_ROWS,
    ADD_TILES_BCAST_ROWS,
    ADD_TILES_BCAST_COLS,
    ADD_TILES_BCAST_SCALAR,
    MUL_TILES_BCAST_SCALAR,

    // Binary Bitwise SFPU
    BITWISE_AND_BINARY_TILE,
    BITWISE_AND_UINT32_BINARY_TILE,
    BITWISE_AND_UINT16_BINARY_TILE,
    BITWISE_OR_BINARY_TILE,
    BITWISE_OR_UINT32_BINARY_TILE,
    BITWISE_OR_UINT16_BINARY_TILE,
    BITWISE_XOR_BINARY_TILE,
    BITWISE_XOR_UINT32_BINARY_TILE,
    BITWISE_XOR_UINT16_BINARY_TILE,

    // Binary Comparison
    LT_INT32_TILE,
    GT_INT32_TILE,
    GE_INT32_TILE,
    LE_INT32_TILE,

    // Binary Max/Min
    BINARY_MAX_INT32_TILE,
    BINARY_MAX_TILE,
    BINARY_MIN_INT32_TILE,
    BINARY_MIN_TILE,

    // Binary Shift
    BINARY_LEFT_SHIFT_TILE,
    BINARY_LEFT_SHIFT_UINT32_TILE,
    BINARY_LEFT_SHIFT_INT32_TILE,
    BINARY_RIGHT_SHIFT_TILE,
    BINARY_RIGHT_SHIFT_UINT32_TILE,
    BINARY_RIGHT_SHIFT_INT32_TILE,
    BINARY_LOGICAL_RIGHT_SHIFT_TILE,
    BINARY_LOGICAL_RIGHT_SHIFT_UINT32_TILE,
    BINARY_LOGICAL_RIGHT_SHIFT_INT32_TILE,

    // Copy Dest Values
    COPY_DEST_VALUES,

    // Cumsum
    CUMSUM_TILE,

    // Div Int32
    DIV_INT32_FLOOR_TILE,
    DIV_INT32_TRUNC_TILE,
    DIV_INT32_TILE,

    // Eltwise Binary SFPU
    ADD_BINARY_TILE,
    SUB_BINARY_TILE,
    MUL_BINARY_TILE,
    DIV_BINARY_TILE,
    RSUB_BINARY_TILE,
    POWER_BINARY_TILE,

    // Eltwise Binary
    MUL_TILES,
    ADD_TILES,
    SUB_TILES,
    BINARY_DEST_REUSE_TILES,

    // EMA
    EMA_TILE,

    // GCD/LCM
    GCD_TILE,
    LCM_TILE,

    // Logsigmoid
    LOGSIGMOID_TILE,

    // Mask
    MASK_TILE,
    MASK_POSINF_TILE,

    // Matmul
    MATMUL_TILES,
    MATMUL_TILES_MATH,
    MATMUL_BLOCK,
    MATMUL_BLOCK_MATH_DYNAMIC_THROTTLE,

    // Mul Int
    MUL_UINT16_TILE,
    MUL_INT32_TILE,
    MUL_UINT32_TILE,

    // Pack Untilize
    PACK_UNTILIZE_BLOCK,
    PACK_UNTILIZE_DEST,

    // Pack Rows
    PACK_ROWS,

    // Quantization
    QUANT_TILE,
    REQUANT_TILE,
    DEQUANT_TILE,

    // Reduce
    REDUCE_TILE,
    REDUCE_TILE_MATH,
    REDUCE_BLOCK_MAX_ROW,

    // Reshuffle
    RESHUFFLE_ROWS_TILE,

    // Sub Int SFPU
    SUB_INT32_TILE,
    SUB_UINT32_TILE,
    SUB_UINT16_TILE,
    RSUB_INT32_TILE,
    RSUB_UINT32_TILE,
    RSUB_UINT16_TILE,

    // Tile Move Copy
    COPY_TILE,
    COPY_BLOCK_MATMUL_PARTIALS,

    // Tilize
    TILIZE_BLOCK,
    TILIZE_BLOCK_NO_PACK,
    UNPACK_TILIZEA_B_BLOCK,
    FAST_TILIZE_BLOCK,

    // Transpose
    TRANSPOSE_WH_DEST,
    TRANSPOSE_WH_TILE,

    // Untilize
    UNTILIZE_BLOCK,

    // Welford
    WELFORD_UPDATE,
    WELFORD_UPDATE_ROWS,
    WELFORD_SAVE_STATE,
    WELFORD_RESTORE_STATE,
    WELFORD_FINALIZE_TO_ROW,
    WELFORD_FINALIZE_TO_FACE,

    // Xlogy
    XLOGY_BINARY_TILE,

    // Eltwise Unary - Activations
    HARDSIGMOID_TILE,
    SOFTSIGN_TILE,
    CELU_TILE,
    SOFTSHRINK_TILE,

    // Eltwise Unary - Binop with Scalar
    ADD_UNARY_TILE,
    SUB_UNARY_TILE,
    MUL_UNARY_TILE,
    DIV_UNARY_TILE,
    RSUB_UNARY_TILE,
    ADD_UNARY_TILE_INT32,
    SUB_UNARY_TILE_INT32,

    // Eltwise Unary - Exp
    EXP_TILE,

    // Eltwise Unary - Recip
    RECIP_TILE,

    // Eltwise Unary - Sqrt
    SQRT_TILE,

    // Eltwise Unary - Rsqrt
    RSQRT_TILE,

    // Eltwise Unary - Trigonometry
    SIN_TILE,
    COS_TILE,
    TAN_TILE,
    ASIN_TILE,
    ACOS_TILE,
    ATAN_TILE,
    SINH_TILE,
    COSH_TILE,
    ASINH_TILE,
    ACOSH_TILE,
    ATANH_TILE,

    // Eltwise Unary - GELU
    GELU_TILE,

    // Eltwise Unary - ReLU
    RELU_TILE,
    RELU_TILE_INT32,
    RELU_MAX_TILE,
    RELU_MAX_TILE_INT32,
    RELU_MIN_TILE,
    RELU_MIN_TILE_INT32,
    LEAKY_RELU_TILE,

    // Eltwise Unary - Fill
    FILL_TILE,
    FILL_TILE_INT,
    FILL_TILE_BITCAST,

    // Eltwise Unary - Typecast
    TYPECAST_TILE,

    // Eltwise Unary - Comparison
    UNARY_EQ_TILE,
    UNARY_EQ_TILE_INT32,
    UNARY_NE_TILE,
    UNARY_NE_TILE_INT32,
    UNARY_GT_TILE,
    UNARY_GT_TILE_INT32,
    UNARY_GE_TILE,
    UNARY_GE_TILE_INT32,
    UNARY_LT_TILE,
    UNARY_LT_TILE_INT32,
    UNARY_LE_TILE,
    UNARY_LE_TILE_INT32,
    GTZ_TILE,
    GTZ_TILE_INT32,
    GEZ_TILE,
    GEZ_TILE_INT32,
    LTZ_TILE,
    LTZ_TILE_INT32,
    LEZ_TILE,
    LEZ_TILE_INT32,
    EQZ_TILE,
    EQZ_TILE_INT32,
    EQZ_TILE_UINT16,
    EQZ_TILE_UINT32,
    NEZ_TILE,
    NEZ_TILE_INT32,
    NEZ_TILE_UINT16,
    NEZ_TILE_UINT32,

    // Eltwise Unary - Bitwise
    BITWISE_AND_TILE,
    BITWISE_NOT_TILE,
    BITWISE_OR_TILE,
    BITWISE_XOR_TILE,

    // Eltwise Unary - Where
    WHERE_TILE,
    WHERE_FP32_TILE,
    WHERE_INT32_TILE,
    WHERE_UINT32_TILE,

    // Eltwise Unary - Rsub
    RSUB_TILE,
    RSUB_UNARY_INT32_TILE,

    // Eltwise Unary - Rdiv
    RDIV_TILE,

    // Eltwise Unary - Rounding
    CEIL_TILE,
    FLOOR_TILE,
    TRUNC_TILE,
    ROUND_TILE,
    ROUND_TILE_WITH_DECIMALS,
    STOCHASTIC_ROUND_TILE,
    FRAC_TILE,

    // Eltwise Unary - Softplus
    SOFTPLUS_TILE,

    // Eltwise Unary - Other
    ISINF_TILE,
    ISPOSINF_TILE,
    ISNEGINF_TILE,
    ISNAN_TILE,
    ISFINITE_TILE,
    IDENTITY_TILE,
    IDENTITY_TILE_UINT32,
    NEGATIVE_TILE,
    NEGATIVE_TILE_INT32,
    LOGICAL_NOT_UNARY_TILE,
    LOG1P_TILE,
    LEFT_SHIFT_TILE,
    RIGHT_SHIFT_TILE,
    REMAINDER_TILE,
    FMOD_TILE,
    PRELU_TILE,
    RAND_TILE,
    I0_TILE,
    I1_TILE,
    ERFINV_TILE,
    ERF_TILE,
    ERFC_TILE,
    ELU_TILE,
    DROPOUT_TILE,
    CLAMP_TILE,
    HARDTANH_TILE,
    HARDMISH_TILE,
    RPOW_TILE,
    THRESHOLD_TILE,
    SELU_TILE,
    CBRT_TILE,
    SFPU_SUM_INT_COL,
    SFPU_SUM_INT_ROW,
    SFPU_ADD_INT,
};

/**
 * @brief Primary template for operation init/uninit mapping.
 * Specializations will be provided for each operation type.
 *
 * @tparam OpType The operation type enum value
 * @tparam Args Variadic template for init/uninit function arguments
 */
template <OperationType OpType, typename... Args>
class OperationDispatcher {
public:
    // Default: no-op for operations without init
    static ALWI void call(Args... /*args*/) {
        // No init function for this operation
    }

    // Default: no-op for operations without uninit
    static ALWI void call_uninit(Args... /*args*/) {
        // No uninit function for this operation
    }
};

}  // namespace ckernel
