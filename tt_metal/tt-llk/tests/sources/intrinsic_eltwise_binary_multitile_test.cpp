
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* Multi-tile elementwise binary test.  As in the single-tile oracle
 * (intrinsic_eltwise_binary_test.cpp), the MATH thread drives the compute
 * through the compiler-managed Tensix compute intrinsics
 * (__builtin_rvtt_*_elwmul) and deliberately does NOT call the LLK math
 * configure/reconfig functions: the compiler's config pass emits the ALU
 * hw_configure baseline + per-compute reconfig.  What this thread DOES own is
 * dest-walking across a block of tiles (the agreed programming model):
 *
 *   - per-tile dest base: math::set_dst_write_addr<Tile32x32>(tile) -- the
 *     LLK's TT_SETC16 to DEST_TARGET_REG_CFG_MATH_Offset = tile<<6.  This is
 *     the instrn-buffer (LLK) channel, because SETC16 data is an immediate and
 *     a runtime per-tile base is not expressible as a compiler-emitted config.
 *     The compiler's baseline dest-base=0 is never re-emitted (its state stays
 *     "known 0", satisfied), so it does not fight the per-tile bases.
 *   - intra-tile row advance: TTI_INCRWC (the LLK partial-face MOP loop_op1).
 *   - per-tile cleanup: TTI_SETRWC(CLR_AB, ..., SET_ABD) -- clears the SrcA/B
 *     valid bits so the unpacker can write the next tile into the bank, and
 *     resets the RWC A/B/D counters to 0 (they sit at 8 after a 16-row face).
 *     This mirrors the MOP end_op (CLR_AB, SET_AB) + clear_dst_reg_addr().
 *
 * Each 16x16 tile is one 16-row face = two TTELWMULs (8 rows each,
 * MAX_FPU_ROWS) with an INCRWC(0,8,8,8) between, exactly as the single-tile
 * oracle.  scope: NUM_TILES_IN_BLOCK tiles in one block.
 */

#include <cstdint>

#include "build.h"
#include "llk_defs.h"
#include "tensor_shape.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Cache volatile values to local variables first
    const std::uint8_t face_r_dim           = static_cast<std::uint8_t>(params.TEST_FACE_R_DIM);
    const std::uint8_t face_c_dim           = static_cast<std::uint8_t>(params.TEST_FACE_C_DIM);
    const std::uint8_t num_faces_r_dim      = static_cast<std::uint8_t>(params.num_faces_r_dim_A);
    const std::uint8_t num_faces_c_dim      = static_cast<std::uint8_t>(params.num_faces_c_dim_A);
    const ckernel::TensorShape tensor_shape = {face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim};
    const ckernel::Transpose transpose      = params.UNPACK_TRANSPOSE_FACES
                                                  ? (params.UNPACK_TRANSPOSE_WITHIN_FACE ? ckernel::Transpose::Both : ckernel::Transpose::InterFace)
                                                  : (params.UNPACK_TRANSPOSE_WITHIN_FACE ? ckernel::Transpose::IntraFace : ckernel::Transpose::None);

    // Configure hardware for unpacking, no broadcast, no transpose
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        tensor_shape.face_r_dim,
        tensor_shape.face_r_dim,
        tensor_shape.total_num_faces(),
        tensor_shape.total_num_faces(),
        params.TILE_SIZE_UNPACK_A,
        params.TILE_SIZE_UNPACK_B);

    // Must come after _llk_unpack_hw_configure_, otherwise the ALU stoch-rnd
    // bits programmed here are overwritten by configure_unpack_AB().
    _llk_unpack_configure_stoch_rnd_<StochRndType::None>();

    _llk_unpack_AB_init_<BROADCAST_TYPE>(tensor_shape, transpose);

    const std::uint32_t num_total_tiles = params.NUM_TILES_IN_BLOCK * params.NUM_BLOCKS;

    for (std::uint32_t i = 0; i < num_total_tiles; ++i)
    {
        _llk_unpack_AB_<BROADCAST_TYPE>(L1_ADDRESS(params.buffer_A[i]), L1_ADDRESS(params.buffer_B[i]));
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "params.h"

using namespace ckernel;

// Compute intrinsic, arch-prefixed (the Tensix mnemonics are identical; the
// J-format field widths differ per arch).  Selected by the harness's
// -DARCH_* define.
#if defined(ARCH_WORMHOLE)
#define INTR_ELWMUL __builtin_rvtt_wh_elwmul
#else
#define INTR_ELWMUL __builtin_rvtt_bh_elwmul
#endif

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Semaphore + dest-section sync, then the LLK's one-time math config
    // (zeroacc/INT8/Fp32/SFPU_Fp32/override-clear/zero-flag/dest-base) issued
    // through the config-write intrinsics; pass_rvtt_config coalesces those.
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(MATH_FORMAT, MATH_FORMAT);

    const std::uint32_t tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks     = params.NUM_BLOCKS;

    for (std::uint32_t block = 0; block < num_blocks; block++)
    {
        _llk_math_wait_for_dest_available_<dest_sync>();
        for (std::uint32_t tile = 0; tile < tiles_in_block; tile++)
        {
            // Dest base for this tile (author-owned; LLK instrn-buffer channel).
            math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(tile);

            // One 16-row face: two TTELWMULs (8 rows each) with an INCRWC
            // row-advance between (the pass coalesces the config established
            // once above; TTELWMUL fires twice per tile).
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
            _llk_math_reconfig_data_format_<is_fp32_dest_acc_en>(
                ckernel::to_underlying(formats.math), ckernel::to_underlying(formats.math));
            INTR_ELWMUL(
                0 /*clr_src*/, 0 /*acc_to_dest*/, 0 /*broadcast*/, 0 /*addr_mod*/, 0 /*dst*/);
            TTI_INCRWC(0 /*cr*/, 8 /*dest*/, 8 /*srcb*/, 8 /*srca*/);
            INTR_ELWMUL(
                0 /*clr_src*/, 0 /*acc_to_dest*/, 0 /*broadcast*/, 0 /*addr_mod*/, 0 /*dst*/);
#else
            INTR_ELWMUL(0, 0, 0, 0, 0);
            TTI_INCRWC(0 /*cr*/, 8 /*dest*/, 8 /*srcb*/, 8 /*srca*/);
            INTR_ELWMUL(0, 0, 0, 0, 0);
#endif
            // Release the source banks for the next tile and reset the RWC
            // A/B/D counters (they sit at 8 after the face).
            TTI_SETRWC(p_setrwc::CLR_AB, 0 /*cr*/, 0 /*dest*/, 0 /*srcb*/, 0 /*srca*/, p_setrwc::SET_ABD);
        }
        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Cache volatile values to local variables first
    const std::uint8_t face_r_dim           = static_cast<std::uint8_t>(params.TEST_FACE_R_DIM);
    const std::uint8_t face_c_dim           = static_cast<std::uint8_t>(params.TEST_FACE_C_DIM);
    const std::uint8_t num_faces_r_dim      = static_cast<std::uint8_t>(params.num_faces_r_dim_A);
    const std::uint8_t num_faces_c_dim      = static_cast<std::uint8_t>(params.num_faces_c_dim_A);
    const ckernel::TensorShape tensor_shape = {face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim};

    const std::uint32_t tile_size = tensor_shape.total_tensor_size();

    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;

    const bool narrow_tile = (tensor_shape.num_faces_c_dim == 1);

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);

    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
        formats.pack_dst, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);

    _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(tensor_shape.face_r_dim, narrow_tile);

    const std::uint32_t output_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t output_num_blocks     = params.NUM_BLOCKS;

    for (std::uint32_t block = 0; block < output_num_blocks; block++)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < output_tiles_in_block; tile++)
        {
            std::uint32_t res_tile_idx = (block * output_tiles_in_block) + tile;
            LLK_ASSERT(
                (static_cast<std::uint32_t>(tile) < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[res_tile_idx]));
        }
        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif
