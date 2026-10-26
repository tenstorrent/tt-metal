// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Regression test: rand must store the whole tile when another op's init (topk, slot 6 = dest incr 32)
// last programmed its dest address mode; seed once, sentinel into Dest, topk init, rand with no rand init.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;
// The sentinel copy and the draw must hit the same Dest tile.
static constexpr std::uint32_t dst_index = 0;
static constexpr bool acc_to_dest        = false;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);

    _llk_unpack_A_init_<BroadcastType::NONE, acc_to_dest, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /*transpose_of_faces*/, 0 /*within_face_16x16_transpose*/, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    _llk_unpack_A_<BroadcastType::NONE, acc_to_dest, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_sfpu/ckernel_sfpu_rand.h"

using namespace ckernel;

// APPROX_MODE, VECTOR_MODE, RAND_FROM_BITS / RAND_SCALE_BITS (the draw interval, which the stimulus
// sentinel lies outside of) and RAND_SEED come from the test's template parameters.

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /*is_int_fpu_en*/, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    // The row mapping the test checks holds only with Dest address remapping off; an earlier pack_untilize
    // test on the same core leaves it on (Blackhole; a no-op on Wormhole).
    _llk_math_reconfig_remap_wrapper_(false /*remap_enable*/);

    // Seed once, as a host-set seed's kernel does; no rand init follows after this point.
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::rand_init<APPROX_MODE>(RAND_SEED);

    // Sentinel tile into Dest.
    _llk_math_wait_for_dest_available_<DST_SYNC>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        dst_index, formats.math, formats.math);

    // Another op's init between the seed and the draw: programs slot 6 to dest incr 32.
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::topk_local_sort>();

    // The draw. rand must overwrite the whole tile regardless of what slot 6 held.
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::rand<APPROX_MODE>, dst_index, VECTOR_MODE, RAND_FROM_BITS, RAND_SCALE_BITS);
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES /*tile_size*/);
    _llk_pack_init_wrapper_<PackMode::Default, false /*zero_output*/>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /*tile_index*/, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
