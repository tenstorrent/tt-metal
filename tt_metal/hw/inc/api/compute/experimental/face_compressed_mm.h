// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common.h"
// Blackhole-only: both LLK halves and the mutexed pack API exist only in the Blackhole trees, so the
// includes are gated on the arch as well as the TRISC role -- otherwise merely including this header
// breaks a Wormhole or Quasar kernel build.
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_math_face_compressed_mm_api.h"
#endif
#if defined(TRISC_UNPACK) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_unpack_AB_face_compressed_mm_api.h"
#endif
#if defined(TRISC_PACK) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_pack_custom_api.h"
#endif
// =============================================================================================
// FACE-GRANULAR (16x16) COMPRESSED MATMUL
// =============================================================================================
//
// C = in0 * in1, with in1 compressed ahead of time per 16x16 face: each face is independently bfp0
// (structurally zero, stored as nothing and skipped), bfp2 or bfp4. Matmul swaps the operands on the way
// in, so in0, the activation, is unpacked into SrcB and in1, the compressed weights, into SrcA.
//
// An ordinary matmul cannot express a per-face format, because the unpacker reads one data format per
// config context. Three mechanisms get around that:
//
//   1. The unpack init programs cntx0 and cntx2 as bfp2, cntx1 and cntx3 as bfp4, so the context an UNPACR
//      issues through is what picks that one face's precision. The two pairs also double buffer, letting
//      the RISC stage the next chunk's base addresses while the unpacker streams from the current pair.
//   2. in1 is never read as an operand. Everything about it comes from one L1 meta buffer laid out as
//      math metas | iters | address words | index words, the first section for the math thread and the
//      rest for unpack. in1_cb_id is there only to keep the (in0, in1) pair normalized and to let the
//      uninit restore the tile descriptor the init changed.
//   3. Each thread records a code sequence into its replay buffer at init and then pushes two REPLAY
//      handles per meta, so the per-face decision is a table index rather than a branch.
//
// The details live with the code they describe: the index-word bit layout and the unpack code sequence
// above the decode tables in llk_unpack_AB_face_compressed_mm.h, the math code sequence and the addrmods
// it names in llk_math_face_compressed_mm.h, the replay and table machinery in ckernel_code_sequence.h,
// and the producer that builds the meta buffer in encode_meta in
// tt_metal/tt-llk/tests/python_tests/test_matmul_face_compressed.py.
// =============================================================================================

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

// clang-format off
/**
 * Short initialization for face_compressed_mm_block operation. Must be called before face_compressed_mm_block and is safe to call at any point in the kernel.
 * At the beginning of a kernel, bring the op up by hand: the unpack, math and pack hw configures, the math
 * pack-sync init and the pack dest init, then pack_init_mutex_ADC as the only pack init.
 * deepseek_compute_kernel_hw_startup cannot stand in for that until it takes a mutex_ADC option.
 *
 * Face-granular (16x16) variant of compressed_custom_mm: in1 is BFP-compressed per 16x16 face and
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
 * | in0_cb_id      | The identifier of the input activation circular buffer (CB)                            | uint32_t | 0 to 31                               | True                  |
 * | in1_cb_id      | The identifier of the compressed-weight circular buffer (CB)                           | uint32_t | 0 to 31                               | True                  |
 * | out_cb_id      | The identifier of the output circular buffer (CB)                                      | uint32_t | 0 to 31                               | True                  |
 */
// clang-format on
template <std::uint32_t ct_dim = 1, bool transpose = false>
ALWI void face_compressed_mm_block_init_short(
    const std::uint32_t in0_cb_id, const std::uint32_t in1_cb_id, const std::uint32_t out_cb_id) {
    UNPACK((llk_unpack_AB_face_compressed_mm_init<transpose>(in0_cb_id, in1_cb_id)));

    MATH((llk_math_face_compressed_mm_init<ct_dim>(in0_cb_id, in1_cb_id)));

    constexpr bool dense_packing = true;  // only dense packing is supported
    if constexpr (dense_packing) {
        // Reduce packing stride from tile to tile to 32 rows instead of 64
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(
            (TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2)));
    }
}

// clang-format off
/**
 * Performs block-sized matrix multiplication *C=A\*B* between the in0 activation block and the
 * BFP-compressed weights streamed from the meta buffer, writing the result to DST. The DST register
 * buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Face-granular (16x16) variant of compressed_custom_mm: in1 is BFP-compressed per 16x16 face and
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
 * | in1_cb_id          | The identifier of the compressed-weight circular buffer (CB)                                   | uint32_t | 0 to 31                                          | True                  |
 * | base_address_meta  | The L1 address of the compressed-weight meta buffer                                            | uint32_t | Valid L1 address                                 | True                  |
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
 * Face-granular (16x16) variant of compressed_custom_mm: in1 is BFP-compressed per 16x16 face and
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
 * | in1_cb_id          | The identifier of the compressed-weight circular buffer (CB)                                   | uint32_t | 0 to 31                                          | True                  |
 * | base_address_meta  | The L1 address of the compressed-weight meta buffer                                            | uint32_t | Valid L1 address                                 | True                  |
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
 * Face-granular (16x16) variant of compressed_custom_mm: in1 is BFP-compressed per 16x16 face and
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
 * | in1_cb_id          | The identifier of the compressed-weight circular buffer (CB)                                   | uint32_t | 0 to 31                                          | True                  |
 * | base_address_meta  | The L1 address of the compressed-weight meta buffer                                            | uint32_t | Valid L1 address                                 | True                  |
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
 * Restores the in1 (compressed-weight) tile descriptor that the init forced to a single face.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                            | Type     | Valid Range                  | Required              |
 * |----------------|----------------------------------------------------------------------------------------|----------|------------------------------|-----------------------|
 * | in0_cb_id      | The identifier of the input activation circular buffer (CB)                            | uint32_t | 0 to 31                      | True                  |
 * | in1_cb_id      | The identifier of the compressed-weight circular buffer (CB)                           | uint32_t | 0 to 31                      | True                  |
 */
// clang-format on
ALWI void face_compressed_mm_block_uninit(const std::uint32_t in0_cb_id, const std::uint32_t in1_cb_id) {
    UNPACK((llk_unpack_AB_face_compressed_mm_uninit(in0_cb_id, in1_cb_id)));
    constexpr bool dense_packing = true;  // only dense packing is supported
    if constexpr (dense_packing) {
        // Restore default packing stride of 64 rows between tiles
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2)));
    }
}

// The mutexed pack path this op requires. The unpack thread borrows the pack thread's address
// counters (ADCs) for the whole op, and two SETADC* instructions reaching the MISC unit in the same
// cycle corrupt each other's counters, so every SETADC* the pack thread issues has to be serialized
// against a hardware mutex. PackMode::Default only.
//
// Arm the packer through pack_init_mutex_ADC only, in place of any other pack init, and pack every tile
// with pack_tile_mutex_ADC. pack_tile, pack_block, pack_untilize and pack_rows are not covered.

// clang-format off
/**
 * Initializes the packer for single-tile packing with ADC-mutexed SETADC issue.
 *
 * Must be paired with `pack_tile_mutex_ADC`; see the note above this function.
 *
 * Return value: None
 *
 * | Param Type | Name | Description                                       | Type     | Valid Range | Required |
 * |------------|------|---------------------------------------------------|----------|-------------|----------|
 * | Function   | ocb  | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void pack_init_mutex_ADC(std::uint32_t ocb) { PACK((llk_pack_init_mutex_ADC(ocb))); }

// clang-format off
/**
 * Copies a single tile from the DEST register buffer to the output CB with ADC-mutexed SETADC issue.
 * Otherwise identical to `pack_tile` - see that function for the CB reservation and write-pointer
 * semantics, including the meaning of `out_of_order_output`.
 *
 * Must be preceded by `pack_init_mutex_ADC`; see the note above
 * this function.
 *
 * Return value: None
 *
 * | Param Type | Name                | Description                                       | Type     | Valid Range                                          | Required |
 * |------------|---------------------|---------------------------------------------------|----------|------------------------------------------------------|----------|
 * | Template   | out_of_order_output | Whether to allow out-of-order output              | bool     | true/false                                           | False    |
 * | Function   | ifrom_dst           | The index of the tile in the DEST register        | uint32_t | Must be less than the size of the DEST register (16) | True     |
 * | Function   | icb                 | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31                                              | True     |
 * | Function   | output_tile_index   | The index of the tile in the output CB to copy to | uint32_t | Must be less than the size of the CB                 | False    |
 */
// clang-format on
template <bool out_of_order_output = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void pack_tile_mutex_ADC(std::uint32_t ifrom_dst, std::uint32_t icb, std::uint32_t output_tile_index = 0) {
    PACK((llk_pack_mutex_ADC<is_fp32_dest_acc_en, out_of_order_output>(ifrom_dst, icb, output_tile_index)));
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
