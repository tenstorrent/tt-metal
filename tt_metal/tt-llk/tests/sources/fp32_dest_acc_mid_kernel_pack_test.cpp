// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// One bf16 tile, built in one dest-acc mode and run in the other.
// All three threads init for the build's mode, then _llk_set_fp32_dest_acc_ switches Dst to the opposite mode.
// MID_KERNEL_DATACOPY selects an A2D datacopy, otherwise the tile is an ELWMUL HiFi4 of A and B.

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "llk_defs.h"
#include "llk_fp32_dest_acc.h"
#include "tensor_shape.h"

// Globals the harness links against.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

namespace
{
// The mode Dst runs in after the switch.
constexpr bool run_fp32_dest_acc_en = !is_fp32_dest_acc_en;

constexpr ckernel::DstSync dest_sync_mode       = ckernel::DstSync::SyncHalf;
constexpr ckernel::MathFidelity math_fidelity   = ckernel::MathFidelity::HiFi4;
constexpr ckernel::BroadcastType broadcast_type = ckernel::BroadcastType::NONE;

// A full 32x32 tile of four 16x16 faces.
constexpr ckernel::TensorShape tensor_shape = {16, 16, 2, 2};

// Single tile, always at Dst index 0.
constexpr std::uint32_t dst_index = 0;
} // namespace

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    const Operand& buffer_A     = params.buffer_A;
    const Operand& buffer_B     = params.buffer_B;

    // Each arm repeats the unpack init of the kernel it is compared with.
    if constexpr (MID_KERNEL_DATACOPY)
    {
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            tensor_shape.face_r_dim,
            tensor_shape.face_r_dim,
            tensor_shape.total_num_faces(),
            tensor_shape.total_num_faces());
        _llk_unpack_A_init_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);
    }
    else
    {
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
        // Must follow HW configure, which overwrites the ALU stoch-rnd bits.
        _llk_unpack_configure_stoch_rnd_<StochRndType::None>();
        _llk_unpack_AB_init_<broadcast_type>(tensor_shape, ckernel::Transpose::None);
    }

    _llk_set_fp32_dest_acc_<ThreadId::UnpackThreadId>();

    if constexpr (MID_KERNEL_DATACOPY)
    {
        _llk_unpack_A_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
    }
    else
    {
        _llk_unpack_AB_<broadcast_type>(L1_ADDRESS(buffer_A[0]), L1_ADDRESS(buffer_B[0]));
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;

    if constexpr (MID_KERNEL_DATACOPY)
    {
        // Takes the run mode because the copy instruction follows the Dst width, MOVA2D for 16 bit and ELWADD for 32 bit.
        // It only programs the MOP and address mods, so it can keep the native kernel's place ahead of the HW configure.
        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            run_fp32_dest_acc_en,
            BroadcastType::NONE,
            false /* is_int_fpu_en */,
            PackMode::Default>(tensor_shape.total_num_faces(), formats.math);
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    }
    else
    {
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWMUL, broadcast_type, math_fidelity, EltwiseBinaryReuseDestType::NONE>(
            tensor_shape, false /* acc_to_dest */);
    }

    _llk_set_fp32_dest_acc_<ThreadId::MathThreadId>(run_fp32_dest_acc_en);

    // The sync init reads neither the template mode nor tile size, so placing it after the switch addresses tile 0 the same way.
    // It is documented as safe to rerun after other work, and here it carries the mode Dst now runs in.
    _llk_math_pack_sync_init_<dest_sync_mode, run_fp32_dest_acc_en>();

    LLK_ASSERT(
        (dst_index < get_dest_max_tiles<dest_sync_mode, run_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Tile index exceeds maximum destination tiles");
    _llk_math_wait_for_dest_available_<dest_sync_mode>();
    if constexpr (MID_KERNEL_DATACOPY)
    {
        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, dest_sync_mode, run_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            dst_index, formats.math, formats.math, tensor_shape.total_num_faces());
    }
    else
    {
        _llk_math_eltwise_binary_<
            EltwiseBinaryType::ELWMUL,
            broadcast_type,
            dest_sync_mode,
            run_fp32_dest_acc_en,
            math_fidelity,
            EltwiseBinaryReuseDestType::NONE>(tensor_shape, dst_index, false /* clear_fp32_dst_acc */);
    }
    _llk_math_dest_section_done_<dest_sync_mode, run_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats   = params.formats;
    const Operand& buffer_Res     = params.buffer_Res;
    constexpr bool partial_face   = false;
    constexpr bool narrow_tile    = false;
    const std::uint32_t tile_size = tensor_shape.total_tensor_size();
    const std::uint32_t num_faces = tensor_shape.total_num_faces();

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
        formats.pack_dst, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);

    _llk_set_fp32_dest_acc_<ThreadId::PackThreadId>();

    // Same placement as the math sync init: the dest init does not depend on the mode and may run again mid kernel.
    _llk_pack_dest_init_wrapper_<dest_sync_mode, run_fp32_dest_acc_en, PackMode::Default>(tensor_shape.face_r_dim, narrow_tile);

    LLK_ASSERT(
        (dst_index < get_dest_max_tiles<dest_sync_mode, run_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Tile index exceeds maximum destination tiles");
    _llk_packer_wait_for_math_done_();
    _llk_pack_<dest_sync_mode, run_fp32_dest_acc_en, PackMode::Default>(dst_index, L1_ADDRESS(buffer_Res[0]));
    _llk_pack_dest_section_done_<dest_sync_mode, run_fp32_dest_acc_en>();
}

#endif
