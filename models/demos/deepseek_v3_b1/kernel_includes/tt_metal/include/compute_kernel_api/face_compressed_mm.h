// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common.h"
#ifdef TRISC_MATH
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_math_face_compressed_mm_api.h"
#endif
#ifdef TRISC_UNPACK
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_face_compressed_mm_api.h"
#endif
namespace ckernel {

// clang-format off
/**
 * Full initialization for face_compressed_mm_block operation. Must be called before face_compressed_mm_block and only once at the beginning of the kernel.
 * For initializing face_compressed_mm_block in the middle of the kernel, please use face_compressed_mm_block_init_short.
 *
 * Face-granular (16x16) variant of compressed_custom_mm: B is BFP-compressed per 16x16 face and
 * streamed from the meta buffer rather than read from in1, so ct_dim is a compile-time template
 * parameter (not a runtime argument). Otherwise the same limitations apply:
 * in0 tile shape: [{1, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: any integer from 1 to 16 (compile-time)
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Return value: None
 *
 * | Argument       | Description                                                                            | Type     | Valid Range                           | Required              |
 * |----------------|----------------------------------------------------------------------------------------|----------|---------------------------------------|-----------------------|
 * | ct_dim         | The width of the output matrix in tiles                                                | uint32_t | 1 to 16 (compile-time)                | False (default 1)     |
 * | transpose      | The transpose flag for performing transpose operation on in1                           | bool     | true/false                            | False (default false) |
 * | dense_packing  | Whether to pack consecutive tiles 32 rows apart (instead of 64, doubles dest capacity) | bool     | true/false                            | False (default false) |
 * | in0_cb_id      | The identifier of the input activation circular buffer (CB)                            | uint32_t | 0 to 31                               | True                  |
 * | in1_cb_id      | The identifier of the compressed-B circular buffer (CB, used for hw_configure)         | uint32_t | 0 to 31                               | True                  |
 * | out_cb_id      | The identifier of the output circular buffer (CB)                                      | uint32_t | 0 to 31                               | True                  |
 */
// clang-format on
template <
    std::uint32_t ct_dim = 1,
    bool transpose = false,
    bool dense_packing = false,
    bool fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void face_compressed_mm_block_init(
    const std::uint32_t in0_cb_id, const std::uint32_t in1_cb_id, const std::uint32_t out_cb_id) {
    // Intentionally swap in0 and in1 as operation specific hw_configures are deprecated
    UNPACK((llk_unpack_hw_configure<fp32_dest_acc_en>(in1_cb_id, in0_cb_id)));
    UNPACK((llk_unpack_AB_face_compressed_mm_init<transpose>(in0_cb_id, in1_cb_id)));

    MATH((llk_math_pack_sync_init<fp32_dest_acc_en>()));
    MATH((llk_math_hw_configure<fp32_dest_acc_en>(in0_cb_id, in1_cb_id)));
    MATH((llk_math_face_compressed_mm_init<ct_dim>(in0_cb_id, in1_cb_id)));

    PACK((llk_pack_dest_init<fp32_dest_acc_en, PackMode::Default>()));
    PACK((llk_pack_hw_configure<fp32_dest_acc_en>(out_cb_id)));
    PACK((llk_pack_init<PackMode::Default, false /* zero_output */>(out_cb_id)));
    if constexpr (dense_packing) {
        // Reduce packing stride from tile to tile to 32 rows instead of 64
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(
            (TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2)));
    }
}

// clang-format off
/**
 * Short initialization for face_compressed_mm_block operation. Must be called before face_compressed_mm_block and is safe to call at any point in the kernel.
 * For initializing face_compressed_mm_block at the beginning of the kernel, please use face_compressed_mm_block_init.
 *
 * Face-granular (16x16) variant of compressed_custom_mm: B is BFP-compressed per 16x16 face and
 * streamed from the meta buffer rather than read from in1, so ct_dim is a compile-time template
 * parameter (not a runtime argument). Otherwise the same limitations apply:
 * in0 tile shape: [{1, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: any integer from 1 to 16 (compile-time)
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Return value: None
 *
 * | Argument       | Description                                                                            | Type     | Valid Range                           | Required              |
 * |----------------|----------------------------------------------------------------------------------------|----------|---------------------------------------|-----------------------|
 * | ct_dim         | The width of the output matrix in tiles                                                | uint32_t | 1 to 16 (compile-time)                | False (default 1)     |
 * | transpose      | The transpose flag for performing transpose operation on in1                           | bool     | true/false                            | False (default false) |
 * | dense_packing  | Whether to pack consecutive tiles 32 rows apart (instead of 64, doubles dest capacity) | bool     | true/false                            | False (default false) |
 * | in0_cb_id      | The identifier of the input activation circular buffer (CB)                            | uint32_t | 0 to 31                               | True                  |
 * | in1_cb_id      | The identifier of the compressed-B circular buffer (CB)                                | uint32_t | 0 to 31                               | True                  |
 * | out_cb_id      | The identifier of the output circular buffer (CB)                                      | uint32_t | 0 to 31                               | True                  |
 */
// clang-format on
template <std::uint32_t ct_dim = 1, bool transpose = false, bool dense_packing = false>
ALWI void face_compressed_mm_block_init_short(
    const std::uint32_t in0_cb_id, const std::uint32_t in1_cb_id, const std::uint32_t out_cb_id) {
    UNPACK((llk_unpack_AB_face_compressed_mm_init<transpose>(in0_cb_id, in1_cb_id)));

    MATH((llk_math_face_compressed_mm_init<ct_dim>(in0_cb_id, in1_cb_id)));

    if constexpr (dense_packing) {
        // Reduce packing stride from tile to tile to 32 rows instead of 64
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(
            (TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2)));
    }
}

// clang-format off
/**
 * Performs block-sized matrix multiplication *C=A\*B* between the in0 activation block and the
 * BFP-compressed B streamed from the meta buffer, writing the result to DST. The DST register
 * buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Face-granular (16x16) variant of compressed_custom_mm: B is BFP-compressed per 16x16 face and
 * streamed from the meta buffer rather than read from in1, so ct_dim is a compile-time template
 * parameter (not a runtime argument). Otherwise the same limitations apply:
 * in0 tile shape: [{1, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: any integer from 1 to 16 (compile-time)
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Return value: None
 *
 * | Argument           | Description                                                                                    | Type     | Valid Range                                      | Required              |
 * |--------------------|------------------------------------------------------------------------------------------------|----------|--------------------------------------------------|-----------------------|
 * | ct_dim             | The width of the output matrix in tiles                                                        | uint32_t | 1 to 16 (compile-time)                           | False (default 1)     |
 * | finalize           | Whether to merge the split-accumulation partials (applied only when ct_dim == 1)               | bool     | true/false                                       | False (default true)  |
 * | clear_src          | Whether to clear SrcB before unpacking (the activation fills only part of SrcB)                | bool     | true/false                                       | False (default true)  |
 * | in0_cb_id          | The identifier of the input activation circular buffer (CB)                                    | uint32_t | 0 to 31                                          | True                  |
 * | in1_cb_id          | The identifier of the compressed-B circular buffer (CB)                                        | uint32_t | 0 to 31                                          | True                  |
 * | base_address_meta  | The L1 address of the compressed-B meta buffer                                                 | uint32_t | Valid L1 address                                 | True                  |
 * | dst_index          | The index of the tile in DST REG to which the result C will be written                         | uint32_t | Must be less than the acquired size of DST REG   | True                  |
 * | kt_dim             | The inner dimension in tiles                                                                   | uint32_t | Must be an even number from 2 to 256 (inclusive) | True                  |
 */
// clang-format on
template <std::uint32_t ct_dim = 1, bool finalize = true, bool clear_src = true>
ALWI void face_compressed_mm_block(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t base_address_meta,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim) {
    UNPACK((llk_unpack_AB_face_compressed_mm<ct_dim, clear_src, finalize>(
        in0_cb_id, in1_cb_id, base_address_meta, kt_dim)));
    MATH((llk_math_face_compressed_mm<ct_dim, finalize>(in0_cb_id, in1_cb_id, base_address_meta, dst_index, kt_dim)));
}

// clang-format off
/**
 * Performs the unpack part of the block-sized matrix multiplication *C=A\*B* (see face_compressed_mm_block).
 * This call is blocking and is only available on the compute engine.
 *
 * Face-granular (16x16) variant of compressed_custom_mm: B is BFP-compressed per 16x16 face and
 * streamed from the meta buffer rather than read from in1, so ct_dim is a compile-time template
 * parameter (not a runtime argument). Otherwise the same limitations apply:
 * in0 tile shape: [{1, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: any integer from 1 to 16 (compile-time)
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Return value: None
 *
 * | Argument           | Description                                                                                    | Type     | Valid Range                                      | Required              |
 * |--------------------|------------------------------------------------------------------------------------------------|----------|--------------------------------------------------|-----------------------|
 * | ct_dim             | The width of the output matrix in tiles                                                        | uint32_t | 1 to 16 (compile-time)                           | False (default 1)     |
 * | clear_src          | Whether to clear SrcB before unpacking (the activation fills only part of SrcB)                | bool     | true/false                                       | False (default true)  |
 * | finalize           | Whether this unpack performs the split-accumulation finalize (ct_dim == 1)                     | bool     | true/false                                       | False (default true)  |
 * | in0_cb_id          | The identifier of the input activation circular buffer (CB)                                    | uint32_t | 0 to 31                                          | True                  |
 * | in1_cb_id          | The identifier of the compressed-B circular buffer (CB)                                        | uint32_t | 0 to 31                                          | True                  |
 * | base_address_meta  | The L1 address of the compressed-B meta buffer                                                 | uint32_t | Valid L1 address                                 | True                  |
 * | kt_dim             | The inner dimension in tiles                                                                   | uint32_t | Must be an even number from 2 to 256 (inclusive) | True                  |
 */
// clang-format on
template <std::uint32_t ct_dim = 1, bool clear_src = true, bool finalize = true>
ALWI void face_compressed_mm_block_unpack(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t base_address_meta,
    const std::uint32_t kt_dim) {
    UNPACK((llk_unpack_AB_face_compressed_mm<ct_dim, clear_src, finalize>(
        in0_cb_id, in1_cb_id, base_address_meta, kt_dim)));
}

// clang-format off
/**
 * Performs the math part of the block-sized matrix multiplication *C=A\*B* (see face_compressed_mm_block).
 * This call is blocking and is only available on the compute engine.
 *
 * Face-granular (16x16) variant of compressed_custom_mm: B is BFP-compressed per 16x16 face and
 * streamed from the meta buffer rather than read from in1, so ct_dim is a compile-time template
 * parameter (not a runtime argument). Otherwise the same limitations apply:
 * in0 tile shape: [{1, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: any integer from 1 to 16 (compile-time)
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Return value: None
 *
 * | Argument           | Description                                                                                    | Type     | Valid Range                                      | Required              |
 * |--------------------|------------------------------------------------------------------------------------------------|----------|--------------------------------------------------|-----------------------|
 * | ct_dim             | The width of the output matrix in tiles                                                        | uint32_t | 1 to 16 (compile-time)                           | False (default 1)     |
 * | finalize           | Whether to merge the split-accumulation partials (applied only when ct_dim == 1)               | bool     | true/false                                       | False (default true)  |
 * | in0_cb_id          | The identifier of the input activation circular buffer (CB)                                    | uint32_t | 0 to 31                                          | True                  |
 * | in1_cb_id          | The identifier of the compressed-B circular buffer (CB)                                        | uint32_t | 0 to 31                                          | True                  |
 * | base_address_meta  | The L1 address of the compressed-B meta buffer                                                 | uint32_t | Valid L1 address                                 | True                  |
 * | dst_index          | The index of the tile in DST REG to which the result C will be written                         | uint32_t | Must be less than the acquired size of DST REG   | True                  |
 * | kt_dim             | The inner dimension in tiles                                                                   | uint32_t | Must be an even number from 2 to 256 (inclusive) | True                  |
 */
// clang-format on
template <std::uint32_t ct_dim = 1, bool finalize = true>
ALWI void face_compressed_mm_block_math(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t base_address_meta,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim) {
    MATH((llk_math_face_compressed_mm<ct_dim, finalize>(in0_cb_id, in1_cb_id, base_address_meta, dst_index, kt_dim)));
}

// clang-format off
/**
 * Uninitializes the face_compressed_mm_block operation, must be called after the final face_compressed_mm_block call in a sequence and before initializing another operation.
 * Restores the in1 (compressed-B) tile descriptor that the init forced to a single face.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                            | Type     | Valid Range                  | Required              |
 * |----------------|----------------------------------------------------------------------------------------|----------|------------------------------|-----------------------|
 * | dense_packing  | Whether to pack consecutive tiles 32 rows apart (instead of 64, doubles dest capacity) | bool     | true/false                   | False (default false) |
 * | in0_cb_id      | The identifier of the input activation circular buffer (CB)                            | uint32_t | 0 to 31                      | True                  |
 * | in1_cb_id      | The identifier of the compressed-B circular buffer (CB)                                | uint32_t | 0 to 31                      | True                  |
 */
// clang-format on
template <bool dense_packing = false>
ALWI void face_compressed_mm_block_uninit(const std::uint32_t in0_cb_id, const std::uint32_t in1_cb_id) {
    UNPACK((llk_unpack_AB_face_compressed_mm_uninit(in0_cb_id, in1_cb_id)));
    if constexpr (dense_packing) {
        // Restore default packing stride of 64 rows between tiles
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2)));
    }
}

}  // namespace ckernel
