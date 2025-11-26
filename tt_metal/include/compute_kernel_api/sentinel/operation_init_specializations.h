// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/sentinel/operation_init_mapping.h"

// ============================================================================
// IMPORTANT: Header Inclusion Strategy
// ============================================================================
//
// Users must include the appropriate compute_kernel_api headers for operations they use.
// For example:
//   - For exp2_tile_init(): include "compute_kernel_api.h"
//   - For relu_tile_init(): include "compute_kernel_api/eltwise_unary/relu.h"
//   - For add_tiles_init(): include "compute_kernel_api/eltwise_binary.h"
//
// Only specializations that are actually used will be instantiated by the compiler,
// ensuring minimal compile-time overhead. The template system ensures that unused
// operations don't contribute to binary size.
//
// For operations with template parameters (e.g., sigmoid_tile_init<fast_and_approx>()),
// users should call the init function directly rather than using this mapping system.
// ============================================================================

namespace ckernel {

// Helper type for passing compile-time template parameters to OperationDispatcher
template <uint32_t Value>
struct TemplateValue {
    static constexpr uint32_t value = Value;
};

// Helper type for passing compile-time bool template parameters to OperationDispatcher
template <bool Value>
struct TemplateBool {
    static constexpr bool value = Value;
};

// ============================================================================
// Mathematical Functions
// ============================================================================

// SIGMOID_TILE - sigmoid_tile_init<fast_and_approx>()
// Requires: #include "compute_kernel_api.h"
template <bool fast_and_approx>
class OperationDispatcher<OperationType::SIGMOID_TILE, TemplateBool<fast_and_approx>> {
public:
    static ALWI void call() { sigmoid_tile_init<fast_and_approx>(); }
};

// LOG_TILE - log_tile_init<fast_and_approx>()
// Requires: #include "compute_kernel_api.h"
template <bool fast_and_approx>
class OperationDispatcher<OperationType::LOG_TILE, TemplateBool<fast_and_approx>> {
public:
    static ALWI void call() { log_tile_init<fast_and_approx>(); }
};

// LOG_WITH_BASE_TILE - log_with_base_tile_init<fast_and_approx>()
// Requires: #include "compute_kernel_api.h"
template <bool fast_and_approx>
class OperationDispatcher<OperationType::LOG_WITH_BASE_TILE, TemplateBool<fast_and_approx>> {
public:
    static ALWI void call() { log_with_base_tile_init<fast_and_approx>(); }
};

// TANH_TILE - tanh_tile_init<fast_and_approx>()
// Requires: #include "compute_kernel_api.h"
template <bool fast_and_approx>
class OperationDispatcher<OperationType::TANH_TILE, TemplateBool<fast_and_approx>> {
public:
    static ALWI void call() { tanh_tile_init<fast_and_approx>(); }
};

// EXP2_TILE - exp2_tile_init()
template <>
class OperationDispatcher<OperationType::EXP2_TILE> {
public:
    static ALWI void call() { exp2_tile_init(); }
    // No uninit for this operation
};

// ============================================================================
// Sign and Absolute Value
// ============================================================================

// SIGNBIT_TILE, SIGNBIT_TILE_INT32 - signbit_tile_init()
// Requires: #include "compute_kernel_api.h"
template <>
class OperationDispatcher<OperationType::SIGNBIT_TILE> {
public:
    static ALWI void call() { signbit_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::SIGNBIT_TILE_INT32> {
public:
    static ALWI void call() { signbit_tile_init(); }
};

// ABS_TILE, ABS_TILE_INT32 - abs_tile_init()
// Requires: #include "compute_kernel_api.h"
template <>
class OperationDispatcher<OperationType::ABS_TILE> {
public:
    static ALWI void call() { abs_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::ABS_TILE_INT32> {
public:
    static ALWI void call() { abs_tile_init(); }
};

// SIGN_TILE - sign_tile_init()
// Requires: #include "compute_kernel_api.h"
template <>
class OperationDispatcher<OperationType::SIGN_TILE> {
public:
    static ALWI void call() { sign_tile_init(); }
};

// ============================================================================
// Power and Square
// ============================================================================

// SQUARE_TILE - square_tile_init()
// Requires: #include "compute_kernel_api.h"
template <>
class OperationDispatcher<OperationType::SQUARE_TILE> {
public:
    static ALWI void call() { square_tile_init(); }
};

// POWER_TILE - power_tile_init()
// Requires: #include "compute_kernel_api.h"
template <>
class OperationDispatcher<OperationType::POWER_TILE> {
public:
    static ALWI void call() { power_tile_init(); }
};

// TILED_PROD_TILE - tiled_prod_tile_init()
// Requires: #include "compute_kernel_api.h"
template <>
class OperationDispatcher<OperationType::TILED_PROD_TILE> {
public:
    static ALWI void call() { tiled_prod_tile_init(); }
};

// ============================================================================
// Max/Min Operations
// ============================================================================

// MAX_TILE - max_tile_init()
// Requires: #include "compute_kernel_api.h"
template <>
class OperationDispatcher<OperationType::MAX_TILE> {
public:
    static ALWI void call() { max_tile_init(); }
};

// UNARY_MAX_TILE, UNARY_MAX_INT32_TILE - unary_max_tile_init()
// Requires: #include "compute_kernel_api.h"
template <>
class OperationDispatcher<OperationType::UNARY_MAX_TILE> {
public:
    static ALWI void call() { unary_max_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::UNARY_MAX_INT32_TILE> {
public:
    static ALWI void call() { unary_max_tile_init(); }
};

// UNARY_MIN_TILE, UNARY_MIN_INT32_TILE - unary_min_tile_init()
template <>
class OperationDispatcher<OperationType::UNARY_MIN_TILE> {
public:
    static ALWI void call() { unary_min_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::UNARY_MIN_INT32_TILE> {
public:
    static ALWI void call() { unary_min_tile_init(); }
};

// ============================================================================
// Special Functions
// ============================================================================

// HEAVISIDE_TILE - heaviside_tile_init()
template <>
class OperationDispatcher<OperationType::HEAVISIDE_TILE> {
public:
    static ALWI void call() { heaviside_tile_init(); }
};

// SILU_TILE - silu_tile_init()
template <>
class OperationDispatcher<OperationType::SILU_TILE> {
public:
    static ALWI void call() { silu_tile_init(); }
};

// ALT_COMPLEX_ROTATE90_TILE - alt_complex_rotate90_tile_init()
template <>
class OperationDispatcher<OperationType::ALT_COMPLEX_ROTATE90_TILE> {
public:
    static ALWI void call() { alt_complex_rotate90_tile_init(); }
};

// ============================================================================
// TopK Operations
// ============================================================================

// TOPK_LOCAL_SORT, TOPK_MERGE, TOPK_REBUILD - topk_tile_init()
template <>
class OperationDispatcher<OperationType::TOPK_LOCAL_SORT> {
public:
    static ALWI void call() { topk_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::TOPK_MERGE> {
public:
    static ALWI void call() { topk_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::TOPK_REBUILD> {
public:
    static ALWI void call() { topk_tile_init(); }
};

// ============================================================================
// Reduction Operations
// ============================================================================

// Note: Reduction operations have template parameters and specific signatures
// Users should call these directly or use appropriate template specializations

// MAX_REDUCE_WITH_INDICES - max_reduce_with_indices_init<layout>()
// SFPU_REDUCE - sfpu_reduce_init<pool_type, format>()
// SFPU_ADD_TOP_ROW - sfpu_add_top_row_init()
template <>
class OperationDispatcher<OperationType::SFPU_ADD_TOP_ROW> {
public:
    static ALWI void call() { sfpu_add_top_row_init(); }
};

// ============================================================================
// Add Int SFPU
// ============================================================================

// ADD_INT32_TILE, ADD_UINT16_TILE, ADD_UINT32_TILE - add_int_tile_init()
// Requires: #include "compute_kernel_api/add_int_sfpu.h"
template <>
class OperationDispatcher<OperationType::ADD_INT32_TILE> {
public:
    static ALWI void call() { add_int_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::ADD_UINT16_TILE> {
public:
    static ALWI void call() { add_int_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::ADD_UINT32_TILE> {
public:
    static ALWI void call() { add_int_tile_init(); }
};

// ============================================================================
// Broadcast Operations
// ============================================================================

// Note: Broadcast operations have complex signatures with template parameters
// UNARY_BCAST - unary_bcast_init<BroadcastType bcast_type>(...)
// Other bcast operations use init_bcast<...>() or *_init_short() variants
// These are complex and users should call them directly

// ============================================================================
// Binary Bitwise SFPU
// ============================================================================

// All bitwise operations use binary_bitwise_tile_init()
template <>
class OperationDispatcher<OperationType::BITWISE_AND_BINARY_TILE> {
public:
    static ALWI void call() { binary_bitwise_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BITWISE_AND_UINT32_BINARY_TILE> {
public:
    static ALWI void call() { binary_bitwise_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BITWISE_AND_UINT16_BINARY_TILE> {
public:
    static ALWI void call() { binary_bitwise_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BITWISE_OR_BINARY_TILE> {
public:
    static ALWI void call() { binary_bitwise_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BITWISE_OR_UINT32_BINARY_TILE> {
public:
    static ALWI void call() { binary_bitwise_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BITWISE_OR_UINT16_BINARY_TILE> {
public:
    static ALWI void call() { binary_bitwise_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BITWISE_XOR_BINARY_TILE> {
public:
    static ALWI void call() { binary_bitwise_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BITWISE_XOR_UINT32_BINARY_TILE> {
public:
    static ALWI void call() { binary_bitwise_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BITWISE_XOR_UINT16_BINARY_TILE> {
public:
    static ALWI void call() { binary_bitwise_tile_init(); }
};

// ============================================================================
// Binary Comparison
// ============================================================================

template <>
class OperationDispatcher<OperationType::LT_INT32_TILE> {
public:
    static ALWI void call() { lt_int32_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::GT_INT32_TILE> {
public:
    static ALWI void call() { gt_int32_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::GE_INT32_TILE> {
public:
    static ALWI void call() { ge_int32_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::LE_INT32_TILE> {
public:
    static ALWI void call() { le_int32_tile_init(); }
};

// ============================================================================
// Binary Max/Min
// ============================================================================

template <>
class OperationDispatcher<OperationType::BINARY_MAX_INT32_TILE> {
public:
    static ALWI void call() { binary_max_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BINARY_MAX_TILE> {
public:
    static ALWI void call() { binary_max_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BINARY_MIN_INT32_TILE> {
public:
    static ALWI void call() { binary_min_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BINARY_MIN_TILE> {
public:
    static ALWI void call() { binary_min_tile_init(); }
};

// ============================================================================
// Binary Shift
// ============================================================================

// All shift operations use binary_shift_tile_init()
template <>
class OperationDispatcher<OperationType::BINARY_LEFT_SHIFT_TILE> {
public:
    static ALWI void call() { binary_shift_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BINARY_LEFT_SHIFT_UINT32_TILE> {
public:
    static ALWI void call() { binary_shift_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BINARY_LEFT_SHIFT_INT32_TILE> {
public:
    static ALWI void call() { binary_shift_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BINARY_RIGHT_SHIFT_TILE> {
public:
    static ALWI void call() { binary_shift_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BINARY_RIGHT_SHIFT_UINT32_TILE> {
public:
    static ALWI void call() { binary_shift_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BINARY_RIGHT_SHIFT_INT32_TILE> {
public:
    static ALWI void call() { binary_shift_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BINARY_LOGICAL_RIGHT_SHIFT_TILE> {
public:
    static ALWI void call() { binary_shift_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BINARY_LOGICAL_RIGHT_SHIFT_UINT32_TILE> {
public:
    static ALWI void call() { binary_shift_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BINARY_LOGICAL_RIGHT_SHIFT_INT32_TILE> {
public:
    static ALWI void call() { binary_shift_tile_init(); }
};

// ============================================================================
// Copy Dest Values
// ============================================================================

// COPY_DEST_VALUES - copy_dest_values_init()
// Requires: #include "compute_kernel_api/copy_dest_values.h"
template <>
class OperationDispatcher<OperationType::COPY_DEST_VALUES> {
public:
    static ALWI void call() { copy_dest_values_init(); }
};

// ============================================================================
// Cumsum
// ============================================================================

template <>
class OperationDispatcher<OperationType::CUMSUM_TILE> {
public:
    static ALWI void call() { cumsum_tile_init(); }
};

// ============================================================================
// Div Int32
// ============================================================================

template <>
class OperationDispatcher<OperationType::DIV_INT32_FLOOR_TILE> {
public:
    static ALWI void call() { div_int32_floor_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::DIV_INT32_TRUNC_TILE> {
public:
    static ALWI void call() { div_int32_trunc_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::DIV_INT32_TILE> {
public:
    static ALWI void call() { div_int32_tile_init(); }
};

// ============================================================================
// Eltwise Binary SFPU
// ============================================================================

template <>
class OperationDispatcher<OperationType::ADD_BINARY_TILE> {
public:
    static ALWI void call() { add_binary_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::SUB_BINARY_TILE> {
public:
    static ALWI void call() { sub_binary_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::MUL_BINARY_TILE> {
public:
    static ALWI void call() { mul_binary_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::DIV_BINARY_TILE> {
public:
    static ALWI void call() { div_binary_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::RSUB_BINARY_TILE> {
public:
    static ALWI void call() { rsub_binary_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::POWER_BINARY_TILE> {
public:
    static ALWI void call() { power_binary_tile_init(); }
};

// ============================================================================
// Eltwise Binary
// ============================================================================

// ADD_TILES - add_tiles_init(icb0, icb1, acc_to_dest, call_line)
// Requires: #include "compute_kernel_api/eltwise_binary.h"
template <>
class OperationDispatcher<OperationType::ADD_TILES, uint32_t, uint32_t> {
public:
    static ALWI void call(uint32_t icb0, uint32_t icb1) { add_tiles_init(icb0, icb1); }
};

template <>
class OperationDispatcher<OperationType::ADD_TILES, uint32_t, uint32_t, bool> {
public:
    static ALWI void call(uint32_t icb0, uint32_t icb1, bool acc_to_dest) { add_tiles_init(icb0, icb1, acc_to_dest); }
};

// SUB_TILES - sub_tiles_init(icb0, icb1, acc_to_dest, call_line)
template <>
class OperationDispatcher<OperationType::SUB_TILES, uint32_t, uint32_t> {
public:
    static ALWI void call(uint32_t icb0, uint32_t icb1) { sub_tiles_init(icb0, icb1); }
};

template <>
class OperationDispatcher<OperationType::SUB_TILES, uint32_t, uint32_t, bool> {
public:
    static ALWI void call(uint32_t icb0, uint32_t icb1, bool acc_to_dest) { sub_tiles_init(icb0, icb1, acc_to_dest); }
};

// MUL_TILES - mul_tiles_init(icb0, icb1, acc_to_dest, call_line)
template <>
class OperationDispatcher<OperationType::MUL_TILES, uint32_t, uint32_t> {
public:
    static ALWI void call(uint32_t icb0, uint32_t icb1) { mul_tiles_init(icb0, icb1); }
};

template <>
class OperationDispatcher<OperationType::MUL_TILES, uint32_t, uint32_t, bool> {
public:
    static ALWI void call(uint32_t icb0, uint32_t icb1, bool acc_to_dest) { mul_tiles_init(icb0, icb1, acc_to_dest); }
};

// BINARY_DEST_REUSE_TILES - binary_dest_reuse_tiles_init<...>(...)
// Note: Has template parameters, users should call directly or use appropriate template specializations
// This is a placeholder - actual usage requires template parameters

// ============================================================================
// EMA
// ============================================================================

// EMA_TILE - ema_init(alpha, beta)
template <>
class OperationDispatcher<OperationType::EMA_TILE, uint32_t, uint32_t> {
public:
    static ALWI void call(uint32_t alpha, uint32_t beta) { ema_init(alpha, beta); }
};

// ============================================================================
// GCD/LCM
// ============================================================================

template <>
class OperationDispatcher<OperationType::GCD_TILE> {
public:
    static ALWI void call() { gcd_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::LCM_TILE> {
public:
    static ALWI void call() { lcm_tile_init(); }
};

// ============================================================================
// Logsigmoid
// ============================================================================

template <>
class OperationDispatcher<OperationType::LOGSIGMOID_TILE> {
public:
    static ALWI void call() { logsigmoid_tile_init(); }
};

// ============================================================================
// Mask
// ============================================================================

// MASK_TILE, MASK_POSINF_TILE - mask_tile_init()
template <>
class OperationDispatcher<OperationType::MASK_TILE> {
public:
    static ALWI void call() { mask_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::MASK_POSINF_TILE> {
public:
    static ALWI void call() { mask_tile_init(); }
};

// ============================================================================
// Matmul
// ============================================================================

// Note: Matmul has multiple init variants (mm_init, mm_init_short, etc.)
// Users should call the appropriate variant directly

// ============================================================================
// Mul Int
// ============================================================================

template <>
class OperationDispatcher<OperationType::MUL_UINT16_TILE> {
public:
    static ALWI void call() { mul_int_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::MUL_INT32_TILE> {
public:
    static ALWI void call() { mul_int32_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::MUL_UINT32_TILE> {
public:
    static ALWI void call() { mul_int32_tile_init(); }
};

// ============================================================================
// Pack Untilize
// ============================================================================

// Note: Pack untilize has template parameters and uninit - users should call directly

// ============================================================================
// Pack Rows
// ============================================================================

// PACK_ROWS - pack_rows_init(num_rows)
// PACK_ROWS - pack_rows_init(num_rows) / pack_rows_uninit()
// Requires: #include "compute_kernel_api/pack.h"
template <>
class OperationDispatcher<OperationType::PACK_ROWS, uint32_t> {
public:
    static ALWI void call(uint32_t num_rows) { pack_rows_init(num_rows); }
    static ALWI void call_uninit() { pack_rows_uninit(); }
};

// ============================================================================
// Quantization
// ============================================================================

// QUANT_TILE - quant_tile_init(zero_point)
template <>
class OperationDispatcher<OperationType::QUANT_TILE, uint32_t> {
public:
    static ALWI void call(uint32_t zero_point) { quant_tile_init(zero_point); }
};

// REQUANT_TILE - requant_tile_init(zero_point)
template <>
class OperationDispatcher<OperationType::REQUANT_TILE, uint32_t> {
public:
    static ALWI void call(uint32_t zero_point) { requant_tile_init(zero_point); }
};

// DEQUANT_TILE - dequant_tile_init(zero_point)
template <>
class OperationDispatcher<OperationType::DEQUANT_TILE, uint32_t> {
public:
    static ALWI void call(uint32_t zero_point) { dequant_tile_init(zero_point); }
};

// ============================================================================
// Reduce
// ============================================================================

// REDUCE_TILE, REDUCE_TILE_MATH - reduce_init(...) / reduce_uninit<enforce_fp32_accumulation>()
// Requires: #include "compute_kernel_api/reduce.h"
// Note: Init has template parameters with defaults, uninit can be called with default template parameter
template <>
class OperationDispatcher<OperationType::REDUCE_TILE, uint32_t, uint32_t, uint32_t> {
public:
    static ALWI void call(uint32_t icb, uint32_t icb_scaler, uint32_t ocb) { reduce_init(icb, icb_scaler, ocb); }
    static ALWI void call_uninit() { reduce_uninit(); }
};

template <>
class OperationDispatcher<OperationType::REDUCE_TILE_MATH, uint32_t, uint32_t, uint32_t> {
public:
    static ALWI void call(uint32_t icb, uint32_t icb_scaler, uint32_t ocb) { reduce_init(icb, icb_scaler, ocb); }
    static ALWI void call_uninit() { reduce_uninit(); }
};

// REDUCE_BLOCK_MAX_ROW - reduce_block_max_row_init<block_ct_dim>() /
// reduce_block_max_row_uninit<clear_fp32_accumulation>() Requires: #include "compute_kernel_api/reduce_custom.h" Note:
// Init requires compile-time template parameter block_ct_dim Specialization with template parameter support
template <uint32_t block_ct_dim>
class OperationDispatcher<OperationType::REDUCE_BLOCK_MAX_ROW> {
public:
    static ALWI void call() { reduce_block_max_row_init<block_ct_dim>(); }
    static ALWI void call_uninit() { reduce_block_max_row_uninit(); }
};

// Fallback specialization for uninit-only usage (when init is called directly)
template <>
class OperationDispatcher<OperationType::REDUCE_BLOCK_MAX_ROW> {
public:
    // Init must be called directly: reduce_block_max_row_init<block_ct_dim>()
    // where block_ct_dim is a compile-time constant (e.g., reduce_block_max_row_init<cols>())
    // Or use: OperationDispatcher<OperationType::REDUCE_BLOCK_MAX_ROW, TemplateValue<block_ct_dim>>::call()
    static ALWI void call_uninit() { reduce_block_max_row_uninit(); }
};

// ============================================================================
// Reshuffle
// ============================================================================

template <>
class OperationDispatcher<OperationType::RESHUFFLE_ROWS_TILE> {
public:
    static ALWI void call() { reshuffle_rows_tile_init(); }
};

// ============================================================================
// Sub Int SFPU
// ============================================================================

// All use sub_int_tile_init() or rsub_int_tile_init()
template <>
class OperationDispatcher<OperationType::SUB_INT32_TILE> {
public:
    static ALWI void call() { sub_int_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::SUB_UINT32_TILE> {
public:
    static ALWI void call() { sub_int_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::SUB_UINT16_TILE> {
public:
    static ALWI void call() { sub_int_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::RSUB_INT32_TILE> {
public:
    static ALWI void call() { rsub_int_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::RSUB_UINT32_TILE> {
public:
    static ALWI void call() { rsub_int_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::RSUB_UINT16_TILE> {
public:
    static ALWI void call() { rsub_int_tile_init(); }
};

// ============================================================================
// Tile Move Copy
// ============================================================================

// COPY_TILE - copy_tile_init(cbid) or variants
// Requires: #include "compute_kernel_api/tile_move_copy.h"
template <>
class OperationDispatcher<OperationType::COPY_TILE, uint32_t> {
public:
    static ALWI void call(uint32_t cbid) { copy_tile_init(cbid); }
};

// COPY_BLOCK_MATMUL_PARTIALS - copy_tile_init(...)
// Requires: #include "compute_kernel_api/tile_move_copy.h"
// Note: Uses same init as COPY_TILE, but may have different signatures
// Users should call copy_tile_init directly with appropriate parameters

// ============================================================================
// Tilize
// ============================================================================

// Note: Tilize has multiple init variants and uninit
// Users should call these directly

// ============================================================================
// Transpose
// ============================================================================

// TRANSPOSE_WH_DEST - transpose_wh_dest_init_short<is_32bit>()
// Requires: #include "compute_kernel_api/transpose_wh_dest.h"
// Note: Has template parameter is_32bit, users should call directly
// This is a placeholder - actual usage requires template parameter

// TRANSPOSE_WH_TILE - transpose_wh_init(...) or transpose_wh_init_short(...)
// Requires: #include "compute_kernel_api/transpose_wh.h"
// Note: Multiple variants exist, users should call directly

// ============================================================================
// Untilize
// ============================================================================

// UNTILIZE_BLOCK - untilize_init(icb) / untilize_uninit(icb)
// Requires: #include "compute_kernel_api/untilize.h"
template <>
class OperationDispatcher<OperationType::UNTILIZE_BLOCK, uint32_t> {
public:
    static ALWI void call(uint32_t icb) { untilize_init(icb); }
    static ALWI void call_uninit(uint32_t icb) { untilize_uninit(icb); }
};

// ============================================================================
// Welford
// ============================================================================

// All welford operations use welford_init()
template <>
class OperationDispatcher<OperationType::WELFORD_UPDATE> {
public:
    static ALWI void call() { welford_init(); }
};

template <>
class OperationDispatcher<OperationType::WELFORD_UPDATE_ROWS> {
public:
    static ALWI void call() { welford_init(); }
};

template <>
class OperationDispatcher<OperationType::WELFORD_SAVE_STATE> {
public:
    static ALWI void call() { welford_init(); }
};

template <>
class OperationDispatcher<OperationType::WELFORD_RESTORE_STATE> {
public:
    static ALWI void call() { welford_init(); }
};

template <>
class OperationDispatcher<OperationType::WELFORD_FINALIZE_TO_ROW> {
public:
    static ALWI void call() { welford_init(); }
};

template <>
class OperationDispatcher<OperationType::WELFORD_FINALIZE_TO_FACE> {
public:
    static ALWI void call() { welford_init(); }
};

// ============================================================================
// Xlogy
// ============================================================================

template <>
class OperationDispatcher<OperationType::XLOGY_BINARY_TILE> {
public:
    static ALWI void call() { xlogy_binary_tile_init(); }
};

// ============================================================================
// Eltwise Unary - Activations
// ============================================================================

template <>
class OperationDispatcher<OperationType::HARDSIGMOID_TILE> {
public:
    static ALWI void call() { hardsigmoid_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::SOFTSIGN_TILE> {
public:
    static ALWI void call() { softsign_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::CELU_TILE> {
public:
    static ALWI void call() { celu_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::SOFTSHRINK_TILE> {
public:
    static ALWI void call() { softshrink_tile_init(); }
};

// ============================================================================
// Eltwise Unary - Binop with Scalar
// ============================================================================

// All use binop_with_scalar_tile_init()
template <>
class OperationDispatcher<OperationType::ADD_UNARY_TILE> {
public:
    static ALWI void call() { binop_with_scalar_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::SUB_UNARY_TILE> {
public:
    static ALWI void call() { binop_with_scalar_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::MUL_UNARY_TILE> {
public:
    static ALWI void call() { binop_with_scalar_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::DIV_UNARY_TILE> {
public:
    static ALWI void call() { binop_with_scalar_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::RSUB_UNARY_TILE> {
public:
    static ALWI void call() { binop_with_scalar_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::ADD_UNARY_TILE_INT32> {
public:
    static ALWI void call() { binop_with_scalar_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::SUB_UNARY_TILE_INT32> {
public:
    static ALWI void call() { binop_with_scalar_tile_init(); }
};

// ============================================================================
// Eltwise Unary - Exp
// ============================================================================

// EXP_TILE - exp_tile_init<approx, fast_and_approx, scale>()
// Requires: #include "compute_kernel_api/eltwise_unary/exp.h"
template <bool approx, bool fast_and_approx, uint32_t scale>
class OperationDispatcher<
    OperationType::EXP_TILE,
    TemplateBool<approx>,
    TemplateBool<fast_and_approx>,
    TemplateValue<scale>> {
public:
    static ALWI void call() { exp_tile_init<approx, fast_and_approx, scale>(); }
};

// ============================================================================
// Eltwise Unary - Recip
// ============================================================================

// RECIP_TILE - recip_tile_init<legacy_compat>()
// Requires: #include "compute_kernel_api/eltwise_unary/recip.h"
template <bool legacy_compat>
class OperationDispatcher<OperationType::RECIP_TILE, TemplateBool<legacy_compat>> {
public:
    static ALWI void call() { recip_tile_init<legacy_compat>(); }
};

// ============================================================================
// Eltwise Unary - Sqrt
// ============================================================================

// SQRT_TILE - sqrt_tile_init()
template <>
class OperationDispatcher<OperationType::SQRT_TILE> {
public:
    static ALWI void call() { sqrt_tile_init(); }
};

// ============================================================================
// Eltwise Unary - Rsqrt
// ============================================================================

// RSQRT_TILE - rsqrt_tile_init<legacy_compat>()
// Requires: #include "compute_kernel_api/eltwise_unary/rsqrt.h"
template <bool legacy_compat>
class OperationDispatcher<OperationType::RSQRT_TILE, TemplateBool<legacy_compat>> {
public:
    static ALWI void call() { rsqrt_tile_init<legacy_compat>(); }
};

// ============================================================================
// Eltwise Unary - Trigonometry
// ============================================================================

template <>
class OperationDispatcher<OperationType::SIN_TILE> {
public:
    static ALWI void call() { sin_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::COS_TILE> {
public:
    static ALWI void call() { cos_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::TAN_TILE> {
public:
    static ALWI void call() { tan_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::ASIN_TILE> {
public:
    static ALWI void call() { asin_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::ACOS_TILE> {
public:
    static ALWI void call() { acos_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::ATAN_TILE> {
public:
    static ALWI void call() { atan_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::SINH_TILE> {
public:
    static ALWI void call() { sinh_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::COSH_TILE> {
public:
    static ALWI void call() { cosh_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::ASINH_TILE> {
public:
    static ALWI void call() { asinh_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::ACOSH_TILE> {
public:
    static ALWI void call() { acosh_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::ATANH_TILE> {
public:
    static ALWI void call() { atanh_tile_init(); }
};

// ============================================================================
// Eltwise Unary - GELU
// ============================================================================

// GELU_TILE - gelu_tile_init<fast_and_approx>()
// Requires: #include "compute_kernel_api/eltwise_unary/gelu.h"
template <bool fast_and_approx>
class OperationDispatcher<OperationType::GELU_TILE, TemplateBool<fast_and_approx>> {
public:
    static ALWI void call() { gelu_tile_init<fast_and_approx>(); }
};

// ============================================================================
// Eltwise Unary - ReLU
// ============================================================================

// RELU_TILE, RELU_TILE_INT32 - relu_tile_init()
// Requires: #include "compute_kernel_api/eltwise_unary/relu.h"
template <>
class OperationDispatcher<OperationType::RELU_TILE> {
public:
    static ALWI void call() { relu_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::RELU_TILE_INT32> {
public:
    static ALWI void call() { relu_tile_init(); }
};

// RELU_MAX_TILE, RELU_MAX_TILE_INT32 - relu_max_tile_init()
template <>
class OperationDispatcher<OperationType::RELU_MAX_TILE> {
public:
    static ALWI void call() { relu_max_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::RELU_MAX_TILE_INT32> {
public:
    static ALWI void call() { relu_max_tile_init(); }
};

// RELU_MIN_TILE, RELU_MIN_TILE_INT32 - relu_min_tile_init()
template <>
class OperationDispatcher<OperationType::RELU_MIN_TILE> {
public:
    static ALWI void call() { relu_min_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::RELU_MIN_TILE_INT32> {
public:
    static ALWI void call() { relu_min_tile_init(); }
};

// LEAKY_RELU_TILE - leaky_relu_tile_init()
template <>
class OperationDispatcher<OperationType::LEAKY_RELU_TILE> {
public:
    static ALWI void call() { leaky_relu_tile_init(); }
};

// ============================================================================
// Eltwise Unary - Fill
// ============================================================================

// FILL_TILE, FILL_TILE_INT, FILL_TILE_BITCAST - fill_tile_init()
template <>
class OperationDispatcher<OperationType::FILL_TILE> {
public:
    static ALWI void call() { fill_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::FILL_TILE_INT> {
public:
    static ALWI void call() { fill_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::FILL_TILE_BITCAST> {
public:
    static ALWI void call() { fill_tile_init(); }
};

// ============================================================================
// Eltwise Unary - Typecast
// ============================================================================

// TYPECAST_TILE - typecast_tile_init<IN_DTYPE, OUT_DTYPE>()
// Requires: #include "compute_kernel_api/eltwise_unary/typecast.h"
template <uint32_t IN_DTYPE, uint32_t OUT_DTYPE>
class OperationDispatcher<OperationType::TYPECAST_TILE, TemplateValue<IN_DTYPE>, TemplateValue<OUT_DTYPE>> {
public:
    static ALWI void call() { typecast_tile_init<IN_DTYPE, OUT_DTYPE>(); }
};

// ============================================================================
// Eltwise Unary - Comparison
// ============================================================================

// All comparison operations use their respective init functions
// Many share the same init (e.g., unary_eq_tile and unary_eq_tile_int32)

template <>
class OperationDispatcher<OperationType::UNARY_EQ_TILE> {
public:
    static ALWI void call() { unary_eq_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::UNARY_EQ_TILE_INT32> {
public:
    static ALWI void call() { unary_eq_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::UNARY_NE_TILE> {
public:
    static ALWI void call() { unary_ne_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::UNARY_NE_TILE_INT32> {
public:
    static ALWI void call() { unary_ne_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::UNARY_GT_TILE> {
public:
    static ALWI void call() { unary_gt_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::UNARY_GT_TILE_INT32> {
public:
    static ALWI void call() { unary_gt_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::UNARY_GE_TILE> {
public:
    static ALWI void call() { unary_ge_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::UNARY_GE_TILE_INT32> {
public:
    static ALWI void call() { unary_ge_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::UNARY_LT_TILE> {
public:
    static ALWI void call() { unary_lt_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::UNARY_LT_TILE_INT32> {
public:
    static ALWI void call() { unary_lt_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::UNARY_LE_TILE> {
public:
    static ALWI void call() { unary_le_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::UNARY_LE_TILE_INT32> {
public:
    static ALWI void call() { unary_le_tile_init(); }
};

// Zero comparison operations - all use their respective init functions
template <>
class OperationDispatcher<OperationType::GTZ_TILE> {
public:
    static ALWI void call() { gtz_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::GTZ_TILE_INT32> {
public:
    static ALWI void call() { gtz_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::GEZ_TILE> {
public:
    static ALWI void call() { gez_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::GEZ_TILE_INT32> {
public:
    static ALWI void call() { gez_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::LTZ_TILE> {
public:
    static ALWI void call() { ltz_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::LTZ_TILE_INT32> {
public:
    static ALWI void call() { ltz_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::LEZ_TILE> {
public:
    static ALWI void call() { lez_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::LEZ_TILE_INT32> {
public:
    static ALWI void call() { lez_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::EQZ_TILE> {
public:
    static ALWI void call() { eqz_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::EQZ_TILE_INT32> {
public:
    static ALWI void call() { eqz_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::EQZ_TILE_UINT16> {
public:
    static ALWI void call() { eqz_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::EQZ_TILE_UINT32> {
public:
    static ALWI void call() { eqz_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::NEZ_TILE> {
public:
    static ALWI void call() { nez_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::NEZ_TILE_INT32> {
public:
    static ALWI void call() { nez_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::NEZ_TILE_UINT16> {
public:
    static ALWI void call() { nez_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::NEZ_TILE_UINT32> {
public:
    static ALWI void call() { nez_tile_init(); }
};

// ============================================================================
// Eltwise Unary - Bitwise
// ============================================================================

template <>
class OperationDispatcher<OperationType::BITWISE_AND_TILE> {
public:
    static ALWI void call() { bitwise_and_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BITWISE_NOT_TILE> {
public:
    static ALWI void call() { bitwise_not_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BITWISE_OR_TILE> {
public:
    static ALWI void call() { bitwise_or_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::BITWISE_XOR_TILE> {
public:
    static ALWI void call() { bitwise_xor_tile_init(); }
};

// ============================================================================
// Eltwise Unary - Where
// ============================================================================

// All where operations use where_tile_init()
template <>
class OperationDispatcher<OperationType::WHERE_TILE> {
public:
    static ALWI void call() { where_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::WHERE_FP32_TILE> {
public:
    static ALWI void call() { where_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::WHERE_INT32_TILE> {
public:
    static ALWI void call() { where_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::WHERE_UINT32_TILE> {
public:
    static ALWI void call() { where_tile_init(); }
};

// ============================================================================
// Eltwise Unary - Rsub
// ============================================================================

template <>
class OperationDispatcher<OperationType::RSUB_TILE> {
public:
    static ALWI void call() { rsub_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::RSUB_UNARY_INT32_TILE> {
public:
    static ALWI void call() { rsub_unary_int32_tile_init(); }
};

// ============================================================================
// Eltwise Unary - Rdiv
// ============================================================================

// RDIV_TILE - rdiv_tile_init()
template <>
class OperationDispatcher<OperationType::RDIV_TILE> {
public:
    static ALWI void call() { rdiv_tile_init(); }
};

// ============================================================================
// Eltwise Unary - Rounding
// ============================================================================

// All rounding operations use rounding_op_tile_init()
template <>
class OperationDispatcher<OperationType::CEIL_TILE> {
public:
    static ALWI void call() { rounding_op_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::FLOOR_TILE> {
public:
    static ALWI void call() { rounding_op_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::TRUNC_TILE> {
public:
    static ALWI void call() { rounding_op_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::ROUND_TILE> {
public:
    static ALWI void call() { rounding_op_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::ROUND_TILE_WITH_DECIMALS> {
public:
    static ALWI void call() { rounding_op_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::STOCHASTIC_ROUND_TILE> {
public:
    static ALWI void call() { rounding_op_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::FRAC_TILE> {
public:
    static ALWI void call() { rounding_op_tile_init(); }
};

// ============================================================================
// Eltwise Unary - Softplus
// ============================================================================

template <>
class OperationDispatcher<OperationType::SOFTPLUS_TILE> {
public:
    static ALWI void call() { softplus_tile_init(); }
};

// ============================================================================
// Eltwise Unary - Other
// ============================================================================

// Note: Many of these operations have their own init functions
// Adding a representative subset - others follow the same pattern

template <>
class OperationDispatcher<OperationType::ISINF_TILE> {
public:
    static ALWI void call() { isinf_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::ISPOSINF_TILE> {
public:
    static ALWI void call() { isposinf_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::ISNEGINF_TILE> {
public:
    static ALWI void call() { isneginf_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::ISNAN_TILE> {
public:
    static ALWI void call() { isnan_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::ISFINITE_TILE> {
public:
    static ALWI void call() { isfinite_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::IDENTITY_TILE> {
public:
    static ALWI void call() { identity_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::IDENTITY_TILE_UINT32> {
public:
    static ALWI void call() { identity_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::NEGATIVE_TILE> {
public:
    static ALWI void call() { negative_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::NEGATIVE_TILE_INT32> {
public:
    static ALWI void call() { negative_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::LOGICAL_NOT_UNARY_TILE> {
public:
    static ALWI void call() { logical_not_unary_tile_init(); }
};

// LOG1P_TILE - log1p_tile_init<fast_and_approx>()
// Requires: #include "compute_kernel_api/eltwise_unary/log1p.h"
template <bool fast_and_approx>
class OperationDispatcher<OperationType::LOG1P_TILE, TemplateBool<fast_and_approx>> {
public:
    static ALWI void call() { log1p_tile_init<fast_and_approx>(); }
};

// LEFT_SHIFT_TILE - left_shift_tile_init()
template <>
class OperationDispatcher<OperationType::LEFT_SHIFT_TILE> {
public:
    static ALWI void call() { left_shift_tile_init(); }
};

// RIGHT_SHIFT_TILE - right_shift_tile_init()
template <>
class OperationDispatcher<OperationType::RIGHT_SHIFT_TILE> {
public:
    static ALWI void call() { right_shift_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::REMAINDER_TILE> {
public:
    static ALWI void call() { remainder_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::FMOD_TILE> {
public:
    static ALWI void call() { fmod_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::PRELU_TILE> {
public:
    static ALWI void call() { prelu_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::RAND_TILE> {
public:
    static ALWI void call() { rand_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::I0_TILE> {
public:
    static ALWI void call() { i0_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::I1_TILE> {
public:
    static ALWI void call() { i1_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::ERFINV_TILE> {
public:
    static ALWI void call() { erfinv_tile_init(); }
};

// ERF_TILE - erf_tile_init<fast_and_approx>()
// Requires: #include "compute_kernel_api/eltwise_unary/erf_erfc.h"
template <bool fast_and_approx>
class OperationDispatcher<OperationType::ERF_TILE, TemplateBool<fast_and_approx>> {
public:
    static ALWI void call() { erf_tile_init<fast_and_approx>(); }
};

// ERFC_TILE - erfc_tile_init<fast_and_approx>()
// Requires: #include "compute_kernel_api/eltwise_unary/erf_erfc.h"
template <bool fast_and_approx>
class OperationDispatcher<OperationType::ERFC_TILE, TemplateBool<fast_and_approx>> {
public:
    static ALWI void call() { erfc_tile_init<fast_and_approx>(); }
};

template <>
class OperationDispatcher<OperationType::ELU_TILE> {
public:
    static ALWI void call() { elu_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::DROPOUT_TILE> {
public:
    static ALWI void call() { dropout_kernel_init(); }
};

template <>
class OperationDispatcher<OperationType::CLAMP_TILE> {
public:
    static ALWI void call() { clamp_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::HARDTANH_TILE> {
public:
    static ALWI void call() { hardtanh_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::HARDMISH_TILE> {
public:
    static ALWI void call() { hardmish_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::RPOW_TILE> {
public:
    static ALWI void call() { rpow_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::THRESHOLD_TILE> {
public:
    static ALWI void call() { threshold_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::SELU_TILE> {
public:
    static ALWI void call() { selu_tile_init(); }
};

template <>
class OperationDispatcher<OperationType::CBRT_TILE> {
public:
    static ALWI void call() { cbrt_tile_init(); }
};

// SFPU_SUM_INT_COL, SFPU_SUM_INT_ROW, SFPU_ADD_INT - sfpu_sum_int_init()
template <>
class OperationDispatcher<OperationType::SFPU_SUM_INT_COL> {
public:
    static ALWI void call() { sfpu_sum_int_init(); }
};

template <>
class OperationDispatcher<OperationType::SFPU_SUM_INT_ROW> {
public:
    static ALWI void call() { sfpu_sum_int_init(); }
};

template <>
class OperationDispatcher<OperationType::SFPU_ADD_INT> {
public:
    static ALWI void call() { sfpu_sum_int_init(); }
};

// ============================================================================
// Uninit Specializations (for operations with template-parameterized init)
// ============================================================================

// PACK_UNTILIZE_BLOCK - pack_untilize_init(...) / pack_untilize_uninit(ocb)
// Requires: #include "compute_kernel_api/pack_untilize.h"
// Note: Init has template parameters, but uninit just needs ocb
template <>
class OperationDispatcher<OperationType::PACK_UNTILIZE_BLOCK, uint32_t> {
public:
    // Init is handled by template specializations with template params
    static ALWI void call_uninit(uint32_t ocb) { pack_untilize_uninit(ocb); }
};

// PACK_UNTILIZE_DEST - pack_untilize_dest_init(...) / pack_untilize_uninit(ocb)
template <>
class OperationDispatcher<OperationType::PACK_UNTILIZE_DEST, uint32_t> {
public:
    // Init is handled by template specializations with template params
    static ALWI void call_uninit(uint32_t ocb) { pack_untilize_uninit(ocb); }
};

// TILIZE_BLOCK - tilize_init(...) / tilize_uninit(icb, ocb)
// Requires: #include "compute_kernel_api/tilize.h"
template <>
class OperationDispatcher<OperationType::TILIZE_BLOCK, uint32_t, uint32_t> {
public:
    // Init is handled by template specializations with template params
    static ALWI void call_uninit(uint32_t icb, uint32_t ocb) { tilize_uninit(icb, ocb); }
};

// TILIZE_BLOCK_NO_PACK - tilize_init_no_pack(...) / tilize_uninit_no_pack(icb)
template <>
class OperationDispatcher<OperationType::TILIZE_BLOCK_NO_PACK, uint32_t> {
public:
    // Init is handled by template specializations with template params
    static ALWI void call_uninit(uint32_t icb) { tilize_uninit_no_pack(icb); }
};

// FAST_TILIZE_BLOCK - fast_tilize_init(...) / fast_tilize_uninit(icb, ocb)
template <>
class OperationDispatcher<OperationType::FAST_TILIZE_BLOCK, uint32_t, uint32_t> {
public:
    // Init is handled by template specializations with template params
    static ALWI void call_uninit(uint32_t icb, uint32_t ocb) { fast_tilize_uninit(icb, ocb); }
};

// UNPACK_TILIZEA_B_BLOCK - tilizeA_B_reduce_init(...) / unpack_tilizeA_B_uninit(icb)
// Requires: #include "compute_kernel_api/tilize.h"
template <>
class OperationDispatcher<OperationType::UNPACK_TILIZEA_B_BLOCK, uint32_t> {
public:
    // Init is handled by template specializations with template params
    static ALWI void call_uninit(uint32_t icb) { unpack_tilizeA_B_uninit(icb); }
};

// REDUCE_TILE, REDUCE_TILE_MATH - reduce_init(...) / reduce_uninit<enforce_fp32_accumulation>()
// Requires: #include "compute_kernel_api/reduce.h"
// Note: Both init and uninit have template parameters, users should call directly

// REDUCE_BLOCK_MAX_ROW - reduce_block_max_row_init(...) / reduce_block_max_row_uninit<clear_fp32_accumulation>()
// Requires: #include "compute_kernel_api/reduce_custom.h"
// Note: Both init and uninit have template parameters, users should call directly

}  // namespace ckernel
