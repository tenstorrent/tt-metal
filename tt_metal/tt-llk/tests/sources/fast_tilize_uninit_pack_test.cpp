// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Fast tilize -> uninit -> a normal pack of Float32 data, all in one kernel (fp32_dest_acc_en only).
//
// Phase 1 fast-tilizes one row of BLOCK_CT_DIM Float16_b tiles (it must fit a single dest bank) into
// Float32 tiles buffer_Res[0 .. BLOCK_CT_DIM). Every thread then runs its fast-tilize uninit, and phase 2
// unpacks BLOCK_CT_DIM Float32 tiles of buffer_B straight to dest and packs them with a normal _llk_pack_
// into the next BLOCK_CT_DIM tiles of buffer_Res. The pack thread does no init, reconfig or hw configure
// between the two phases, so phase 2 packs with exactly the state _llk_pack_fast_tilize_uninit_ leaves
// behind (packer input/output formats, Dstacc, strides, Dest read control, tile header, dest offsets,
// addr mods and MOP). Phase 2 must copy its input bit for bit. Its data uses all 32 bits of each datum,
// so a packer left with the Float16_b input format that fast tilize switches to would drop the low 16 bits.
//
// Supports 32x32 tiles (num_faces 4, BLOCK_CT_DIM 1..4) and 16x32 tiles (num_faces 2, BLOCK_CT_DIM 2 or 4).
// With 16x32 tiles the result operand's tile size is a full 32x32 tile, i.e. two dense 16x32 tiles, as in
// the fast tilize tiny-tile test.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

static_assert(is_fp32_dest_acc_en, "phase 2 unpacks Float32 straight to dest, which needs fp32_dest_acc_en");

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Splits a single-bank row of tiles into fast tilize calls the way fast_tilize_test.cpp does for its
// last dest bank: unit_dim 1 for a single tile, otherwise unit_dim 2 units plus at most one unit_dim 3 unit.
template <typename Call>
inline void fast_tilize_single_bank(const std::uint32_t block_ct_dim, Call call)
{
    const std::uint32_t unit_dim = block_ct_dim == 1 ? 1 : 2;
    if (block_ct_dim % 2 == 0 || unit_dim == 1)
    {
        call(0 /* first_tile */, unit_dim, block_ct_dim / unit_dim /* num_units */);
    }
    else if (block_ct_dim == 3)
    {
        call(0 /* first_tile */, 3 /* unit_dim */, 1 /* num_units */);
    }
    else
    {
        call(0 /* first_tile */, unit_dim, (block_ct_dim - 3) / unit_dim /* num_units */);
        call(block_ct_dim - 3 /* first_tile */, 3 /* unit_dim */, 1 /* num_units */);
    }
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t num_faces    = params.num_faces;
    const std::uint32_t BLOCK_CT_DIM = params.BLOCK_CT_DIM;
    const Operand& buffer_A          = params.buffer_A;
    const Operand& buffer_B          = params.buffer_B;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);

    // Phase 1: fast tilize
    _llk_unpack_fast_tilize_init_(formats.unpack_A_dst, BLOCK_CT_DIM);
    fast_tilize_single_bank(
        BLOCK_CT_DIM,
        [&](const std::uint32_t first_tile, const std::uint32_t unit_dim, const std::uint32_t num_units)
        { _llk_unpack_fast_tilize_block_(L1_ADDRESS(buffer_A[0]), first_tile, formats.unpack_A_src, unit_dim, num_units, BLOCK_CT_DIM, num_faces); });
    _llk_unpack_fast_tilize_uninit_<is_fp32_dest_acc_en>();

    // Phase 2: Float32 tiles of buffer_B unpacked straight to dest
    constexpr std::uint32_t phase2_format = ckernel::to_underlying(DataFormat::Float32);
    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::FACE_ROW_MAJOR>(
        phase2_format, phase2_format, SCALE_DATUM_SIZE(phase2_format, num_faces * FACE_R_DIM * FACE_C_DIM) /*tile_size*/, FACE_R_DIM, num_faces);
    _llk_unpack_A_init_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, true /* unpack_to_dest */>(
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
        phase2_format,
        phase2_format);
    for (std::uint32_t j = 0; j < BLOCK_CT_DIM; j++)
    {
        _llk_unpack_A_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, true /* unpack_to_dest */>(
            L1_ADDRESS(buffer_B[j]), phase2_format, phase2_format);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_fast_tilize.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t num_faces    = params.num_faces;
    const std::uint32_t BLOCK_CT_DIM = params.BLOCK_CT_DIM;

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    // Phase 1: fast tilize
    _llk_math_fast_tilize_init_(formats.math, BLOCK_CT_DIM == 1 ? 1 : 2);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    fast_tilize_single_bank(
        BLOCK_CT_DIM,
        [&](const std::uint32_t first_tile, const std::uint32_t unit_dim, const std::uint32_t num_units)
        { _llk_math_fast_tilize_block_(first_tile, formats.math, unit_dim, num_units, num_faces); });
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_fast_tilize_uninit_<is_fp32_dest_acc_en>(formats.math);

    // Phase 2: Float32 tiles unpacked straight to dest, math only hands dest over to the packer
    constexpr std::uint32_t phase2_format   = ckernel::to_underlying(DataFormat::Float32);
    const ckernel::TensorShape tensor_shape = ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces);
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en>(tensor_shape, phase2_format);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    for (std::uint32_t j = 0; j < BLOCK_CT_DIM; j++)
    {
        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, true /* unpack_to_dest */>(
            j, phase2_format, phase2_format, tensor_shape);
    }
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "llk_pack_fast_tilize.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t num_faces    = params.num_faces;
    const std::uint32_t BLOCK_CT_DIM = params.BLOCK_CT_DIM;
    const Operand& buffer_Res        = params.buffer_Res;

    // number of tiles per buffer_Res slot (a 16x32 tile is half of the full-tile slot)
    const std::uint32_t tiles_per_slot = num_faces == 2 ? 2 : 1;
    const std::uint32_t tile_bytes     = (buffer_Res[1] - buffer_Res[0]) / tiles_per_slot;
    const auto res_tile_address        = [&](const std::uint32_t tile) { return L1_ADDRESS(buffer_Res[0] + tile * tile_bytes); };

    const std::uint32_t use_32bit_dest = formats.unpack_A_dst == ckernel::to_underlying(DataFormat::Tf32);

    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>();
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(
        formats.pack_src, formats.pack_dst, SCALE_DATUM_SIZE(formats.pack_dst, TILE_C_DIM * TILE_R_DIM));

    // Phase 1: fast tilize into tiles [0, BLOCK_CT_DIM)
    _llk_pack_fast_tilize_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>(
        use_32bit_dest, formats.pack_dst, BLOCK_CT_DIM == 1 ? 1 : 2, num_faces, TILE_C_DIM * TILE_R_DIM /*l1_tile_elements*/, formats.pack_src);
    _llk_packer_wait_for_math_done_();
    fast_tilize_single_bank(
        BLOCK_CT_DIM,
        [&](const std::uint32_t first_tile, const std::uint32_t unit_dim, const std::uint32_t num_units)
        { _llk_pack_fast_tilize_block_(first_tile, res_tile_address(first_tile), unit_dim, num_units, num_faces); });
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_pack_fast_tilize_uninit_<DstSync::SyncHalf, is_fp32_dest_acc_en>(
        formats.pack_dst, FACE_R_DIM, num_faces, false /* partial_face */, false /* narrow_tile */, formats.pack_src, use_32bit_dest);

    // Phase 2: normal pack of the Float32 dest tiles into tiles [BLOCK_CT_DIM, 2 * BLOCK_CT_DIM), with only the state the uninit left
    _llk_packer_wait_for_math_done_();
    for (std::uint32_t j = 0; j < BLOCK_CT_DIM; j++)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(j, res_tile_address(BLOCK_CT_DIM + j));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
