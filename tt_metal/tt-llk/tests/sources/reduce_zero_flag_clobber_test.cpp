// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Regression guard for the Src zero-substitution flag (ALU_ACC_CTRL_Zero_Flag_disabled_src) across a
// REDUCE_ROW MAX.
//
// That reduce is the only reduce path with a mov phase: reduce_row_perform_transpose moves the pooled
// row DEST -> SrcB (MOVD2B/TRNSPSRCB) and adds it back with ELWADD. Those readers need the flag SET
// (PRESERVE / no zero substitution) or a datum whose low byte is zero is flushed to 0 mid-reduction --
// this is what made layernorm drift when the flag was unpack-owned (#46511, tt-llk #960/#966).
//
// #46511 built the tracker so that "the op-need is (re)asserted in the EXECUTE path so it survives an
// llk_math_hw_configure that runs after the op init". This kernel exercises exactly that: it clobbers
// the flag AFTER _llk_math_reduce_init_ the way a real tt-metal compute kernel would, then reduces. If
// _llk_math_reduce_ re-asserts PRESERVE per tile, the result matches golden; if PRESERVE is only
// established at init, the clobber wins and the reduction is silently wrong.
//
// ZERO_FLAG_CLOBBER selects the pollution (see the ZERO_FLAG_CLOBBER docstring in
// python_tests/helpers/test_variant_parameters.py); ZERO_FLAG_CLOBBER_PER_TILE repeats it before every
// tile instead of once after init.

#include <algorithm>
#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"
#include "tensor_shape.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_reduce.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(params.in0_face_r_dim),
        static_cast<std::uint8_t>(params.in0_face_c_dim),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};

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
    _llk_unpack_AB_reduce_init_<POOL_TYPE, REDUCE_DIM>(tensor_shape);
    for (int i = 0; i < params.INPUT_TILE_CNT; ++i)
    {
        _llk_unpack_AB_reduce_<POOL_TYPE, REDUCE_DIM>(L1_ADDRESS(params.buffer_A[i]), L1_ADDRESS(params.buffer_B[0]));
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"

// Pollute the Src zero-substitution flag the way a tt-metal compute kernel would between reduce_init
// and reduce_tile. Every mode ends with the flag zero-substituting (flushing), which is what the
// REDUCE_ROW MAX mov phase must not see.
//
// Returns whether the pollution actually reached the hardware, read back from the config register.
// Without that read-back a passing test is ambiguous: it could mean "the reduce recovered" or "the
// clobber never moved the bit". The caller skips the reduce when this returns false, so an
// ineffective clobber fails the comparison instead of silently vouching for the op.
template <int Mode>
inline bool clobber_zero_flag_state(const std::uint32_t math_format)
{
    if constexpr (Mode == 1)
    {
        // reconfig_data_format(icb, icb_scaler) -- by far the most common real pattern. On Blackhole
        // this touches only the int8 bit and the zero flag, so it cannot disturb the reduce MOP.
        _llk_math_reconfig_data_format_<is_fp32_dest_acc_en>(math_format, math_format);
    }
    else if constexpr (Mode == 2)
    {
        // The case #46511's execute-path rule was written for: an llk_math_hw_configure after the op
        // init (e.g. a fused kernel re-running init_common between phases).
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
    }
    else if constexpr (Mode == 3)
    {
        // copy_tile_to_dst_init_short(cb) where cb is fp8: tile_move_copy.h routes that to
        // _configure_copy_zero_flag_state_, which flushes for fp8 sources.
        ckernel::math::_configure_copy_zero_flag_state_(static_cast<std::uint32_t>(DataFormat::Fp8_e4m3));
    }
    else if constexpr (Mode == 4)
    {
        // Ground truth: force the flag to zero-substitute irrespective of tracker/format state, so a
        // pass cannot be explained by "the clobber never actually moved the bit".
        ckernel::math::_configure_src_zero_flag_(false);
    }

    if constexpr (Mode == 0)
    {
        return true; // control: nothing was clobbered, nothing to verify
    }
    else
    {
        // cfg_reg_rmw_tensix issues RMWCIB (a Tensix instruction), so drain the pipe before reading
        // the register back from the RISC or we would observe the pre-clobber value.
        ckernel::tensix_sync();
        // Validated by inverting this comparison: with the polarity flipped the gate below fires and
        // the whole matrix fails, so the read-back does observe the clobber rather than a stale value.
        return (ckernel::cfg_read(ALU_ACC_CTRL_Zero_Flag_disabled_src_ADDR32) & ALU_ACC_CTRL_Zero_Flag_disabled_src_MASK) == 0;
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    constexpr bool is_int_fpu_en            = false;
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(params.in0_face_r_dim),
        static_cast<std::uint8_t>(params.in0_face_c_dim),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY>(tensor_shape);

    // The pollution under test: after the op init, before any reduce.
    bool clobber_landed = true;
    if constexpr (!ZERO_FLAG_CLOBBER_PER_TILE)
    {
        clobber_landed = clobber_zero_flag_state<ZERO_FLAG_CLOBBER>(formats.math);
    }

    int remaining_tiles = params.INPUT_TILE_CNT;
    while (remaining_tiles != 0)
    {
        int tiles_to_dest = std::min(remaining_tiles, static_cast<int>(params.NUM_TILES_IN_BLOCK));
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (int i = 0; i < tiles_to_dest; ++i)
        {
            if constexpr (ZERO_FLAG_CLOBBER_PER_TILE)
            {
                clobber_landed = clobber_zero_flag_state<ZERO_FLAG_CLOBBER>(formats.math) && clobber_landed;
            }
            // Skipping the reduce leaves DEST unwritten, so an ineffective clobber shows up as a
            // failed comparison rather than as a pass that proves nothing.
            if (clobber_landed)
            {
                _llk_math_reduce_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY, is_int_fpu_en>(i, tensor_shape);
            }
        }
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        remaining_tiles -= tiles_to_dest;
    }

    _llk_math_reduce_uninit_();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(params.in0_face_r_dim),
        static_cast<std::uint8_t>(params.in0_face_c_dim),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};

    const std::uint32_t tile_size = tensor_shape.total_tensor_size();
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;
    const bool narrow_tile        = tensor_shape.num_faces_c_dim == 1;

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);

    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
        formats.pack_dst, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);

    _llk_pack_reduce_mask_config_<REDUCE_DIM>(tensor_shape.face_r_dim);

    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>(tensor_shape.face_r_dim, narrow_tile);

    int remaining_tiles = params.OUTPUT_TILE_CNT;
    while (remaining_tiles != 0)
    {
        int tiles_from_dest = std::min(remaining_tiles, static_cast<int>(params.NUM_TILES_IN_BLOCK));
        _llk_packer_wait_for_math_done_();
        for (int i = 0; i < tiles_from_dest; ++i)
        {
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                i, L1_ADDRESS(params.buffer_Res[params.OUTPUT_TILE_CNT - remaining_tiles + i]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        remaining_tiles -= tiles_from_dest;
    }
    _llk_pack_reduce_mask_clear_();
}

#endif
