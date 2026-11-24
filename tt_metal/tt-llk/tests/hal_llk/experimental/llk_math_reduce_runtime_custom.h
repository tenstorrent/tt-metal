// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_reduce_runtime_custom.h. Same Tensix instruction
// sequence as production, expressed through the HAL with the helpers of llk_math_reduce_custom.h
// (hal_port::reduce_block_max_math). That header is included here because _llk_math_reduce_block_max_row_reinit_short_runtime_
// calls its reduce_max_row_configure_addrmod(), as production does.

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "experimental/llk_math_reduce_custom.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "lltt.h"
#include "tensor_shape.h"

using namespace ckernel;

/**
 * Configures address modifiers for specialized reduce_max_row operations.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block/tile operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for native reduce LLK configuration.
 * Use the standard reduce configuration functions for general-purpose reduction operations.
 */
inline void reduce_max_row_configure_addrmod_runtime()
{
    namespace rbm = hal_port::reduce_block_max_math;
    rbm::set_addr_mod_6(); // ADDR_MOD_6: default mode used with pool operations (no auto-increment)
    rbm::set_addr_mod_1(); // ADDR_MOD_1: face-to-face advancement used with GMPOOL (SrcA += 16, SrcB cleared)
    rbm::set_addr_mod_2(); // ADDR_MOD_2: sequential 4-row write mode used with MOVB2D (Dest += 4)
    rbm::set_addr_mod_3(); // ADDR_MOD_3: face-boundary jump used with the final MOVB2D in each face (Dest += 20)
}

inline void reduce_max_row_configure_addrmod_reinit_runtime()
{
    namespace rbm = hal_port::reduce_block_max_math;
    rbm::set_addr_mod_6();
    rbm::set_addr_mod_1();
    rbm::set_addr_mod_2();
    rbm::set_addr_mod_3();
}

// Minimal reinit: ADDR_MOD_1 + ADDR_MOD_2 (clobbered by matmul) and ADDR_MOD_6
// (clobbered by sub_exp runtime). Skips ADDR_MOD_3 which is preserved from the
// previous reduce (matmul doesn't touch 3, sub_exp custom doesn't touch 3,
// copy_tile_custom_v2 doesn't touch 3).
inline void reduce_max_row_configure_addrmod_reinit_minimal_runtime()
{
    namespace rbm = hal_port::reduce_block_max_math;
    rbm::set_addr_mod_6();
    rbm::set_addr_mod_1();
    rbm::set_addr_mod_2();
}

/**
 * Configures MOP (Macro Operation) for block-based reduce_max_row operations.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32 (num_faces=4) or 16x32 (num_faces=2, a single face-row)
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * For num_faces=2 the per-tile MOP reduces only F0&F1 (no F2 jump, no second reduce) and the
 * recorded transpose block covers the single face-row.
 *
 * This function should NOT be used as a substitute for native reduce LLK MOP configuration.
 * Use the standard reduce MOP configuration with _llk_math_reduce_init_ for general-purpose reduction.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_mop_config_runtime_(std::uint32_t block_ct_dim, const ckernel::TensorShape tensor_shape)
{
    // Constraint on the outerloop and innerloop dim
    // static_assert(block_ct_dim < 128, "block_ct_dim must be less than 128");
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    LLK_ASSERT(!(tensor_shape.num_faces_r_dim == 1 && is_fp32_dest_acc_en), "16x32 reduce_block_max_row not supported in FP32 dest mode yet");

    hal_port::reduce_block_max_math::record_replay<is_fp32_dest_acc_en>(tensor_shape);
    hal_port::reduce_block_max_math::program_mop(block_ct_dim, tensor_shape);
}

/**
 * Reprograms only the MOP registers for reduce_max_row, without re-recording the replay buffer.
 * The replay buffer at positions 0-14 must still contain the reduce GMPOOL+transpose sequence
 * from the original _llk_math_reduce_block_max_row_mop_config_ call.
 * Use when the MOP was clobbered (e.g., by eltwise binary ops) but the replay buffer is intact.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_mop_reprogram_only_runtime_(std::uint32_t block_ct_dim, const ckernel::TensorShape& tensor_shape)
{
    hal_port::reduce_block_max_math::program_mop(block_ct_dim, tensor_shape);
}

/**
 * Initializes block-based reduce_max_row operation for processing multiple tiles.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native _llk_math_reduce_init_ LLK.
 * Use the standard _llk_math_reduce_init_<PoolType::MAX, ReduceDim::REDUCE_ROW>() with multiple
 * _llk_math_reduce_() calls in a loop for general-purpose block reduction.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_init_runtime_(std::uint32_t block_ct_dim, const ckernel::TensorShape& tensor_shape)
{
    reduce_max_row_configure_addrmod_runtime();

    hal_port::reduce_block_max_math::enable_clear_dvalid_and_reset_counters();

    _llk_math_reduce_block_max_row_mop_config_runtime_<is_fp32_dest_acc_en>(block_ct_dim, tensor_shape);

    // The reduce runs on the FPU pool (GMPOOL). Restore the operand-driven zero-flag baseline so a
    // preceding datacopy/copy_init that left PRESERVE (keep denormals) does not leak into it, mirroring
    // the standard _llk_math_reduce_init_.
    hal_port::_configure_default_zero_flag_state_();
}

template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_uninit_runtime_()
{
}

/**
 * Performs block-based reduce_max_row operation across multiple tiles in the width dimension.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native _llk_math_reduce_ LLK.
 * Use the standard _llk_math_reduce_<PoolType::MAX, ReduceDim::REDUCE_ROW>() in a loop
 * for general-purpose block reduction across multiple tiles.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_runtime_(const std::uint32_t dst_index, const ckernel::TensorShape tensor_shape)
{
    LLK_ASSERT(!(tensor_shape.num_faces_r_dim == 1 && is_fp32_dest_acc_en), "16x32 reduce_block_max_row not supported in FP32 dest mode yet");

    hal_port::reduce_block_max_math::run_tile<is_fp32_dest_acc_en>(dst_index, tensor_shape);
}

/**
 * Reinitializes the block-based reduce_max_row operation after a matmul.
 *
 * This LLK API function is used only to re-initialize the address modifiers after a
 * matmul operation in an SDPA inner loop. Please don't use this function as a substitute for
 * the native llk_math_reduce_block_max_row_init LLK. This function is highly specialized
 * for a certain use case and the LLK team does not guarantee any degree of generality.
 */
inline void _llk_math_reduce_block_max_row_reinit_runtime_()
{
    reduce_max_row_configure_addrmod_reinit_runtime();
}

/**
 * Short reinitialization for block-based reduce_max_row operation after a matmul.
 * Reprograms address modifiers and MOP configuration. Used when MOP was clobbered
 * but full init is not needed.
 *
 * This LLK API function is used only to re-initialize the address modifiers and MOP
 * after a matmul operation in an SDPA inner loop.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_reinit_short_runtime_(std::uint32_t block_ct_dim, const ckernel::TensorShape& tensor_shape)
{
    reduce_max_row_configure_addrmod();
    hal_port::reduce_block_max_math::enable_clear_dvalid_and_reset_counters();
    _llk_math_reduce_block_max_row_mop_reprogram_only_runtime_<is_fp32_dest_acc_en>(block_ct_dim, tensor_shape);
}

/**
 * Minimal reinitialization for block-based reduce_max_row operation.
 * Only reconfigures ADDR_MOD_1, ADDR_MOD_2, and ADDR_MOD_6 (preserves ADDR_MOD_3).
 * Used when only specific addrmods were clobbered by previous operations.
 */
inline void _llk_math_reduce_block_max_row_reinit_minimal_runtime_()
{
    reduce_max_row_configure_addrmod_reinit_minimal_runtime();
    hal_port::reduce_block_max_math::enable_clear_dvalid_and_reset_counters();
}
