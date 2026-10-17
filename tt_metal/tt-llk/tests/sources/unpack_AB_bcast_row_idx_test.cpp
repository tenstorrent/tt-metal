// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// One 32x32 tile: Res = A <ELTWISE_BINARY_OP> row_bcast(B, BCAST_ROW_IDX).
// Exercises the bcast_row_idx / srcb_format arguments of _llk_unpack_AB_<ROW>, which the
// eltwise_binary tests leave at their defaults. Math and pack wait on an L1 word (UNPACK_ISSUED, the
// host writes 0 before the first run) that the unpack thread sets once its unpack call has returned, so
// a case whose unpack thread stops in an LLK_ASSERT leaves no Tensix instruction pending on the
// other threads (a RISC spinning on L1 is reset cleanly by the next soft reset, a stalled Tensix
// instruction is not).

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// L1 copy of the runtime parameters (host-written before the first run); the UNPACK_ISSUED gate lives here.
extern const volatile struct RuntimeParams __runtime_args_start[];

#if defined(LLK_TRISC_MATH) || defined(LLK_TRISC_PACK)
// Spin until the unpack thread has issued its unpack. Another RISC writes the word, so the data cache
// is invalidated on every poll, as the harness's other cross-RISC polls do.
inline void wait_for_unpack_issued()
{
    while (__runtime_args_start[0].UNPACK_ISSUED == 0)
    {
        ckernel::invalidate_data_cache();
    }
}
#endif

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        FACE_R_DIM,
        FACE_R_DIM,
        TILE_NUM_FACES /*unpA_num_faces*/,
        TILE_NUM_FACES /*unpB_num_faces*/);

    _llk_unpack_AB_init_<BROADCAST_TYPE>(ckernel::DEFAULT_TENSOR_SHAPE, ckernel::Transpose::None);
    _llk_unpack_AB_<BROADCAST_TYPE>(L1_ADDRESS(params.buffer_A[0]), L1_ADDRESS(params.buffer_B[0]), params.BCAST_ROW_IDX, formats.unpack_B_src);
    _llk_unpack_AB_uninit_();
    *const_cast<volatile std::uint32_t*>(&__runtime_args_start[0].UNPACK_ISSUED) = 1;
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    wait_for_unpack_issued();
    constexpr auto sync_mode = DstSync::SyncHalf;

    _llk_math_pack_sync_init_<sync_mode, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(DEFAULT_TENSOR_SHAPE, 0 /*acc_to_dest*/);

    _llk_math_wait_for_dest_available_<sync_mode>();
    _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BROADCAST_TYPE, sync_mode, is_fp32_dest_acc_en, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
        DEFAULT_TENSOR_SHAPE, 0 /*dst_index*/, false /*clear_fp32_dst_acc*/);
    _llk_math_dest_section_done_<sync_mode, is_fp32_dest_acc_en>();
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
    wait_for_unpack_issued();
    constexpr auto sync_mode = ckernel::DstSync::SyncHalf;

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES /*tile_size*/, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES /*num_faces*/);
    _llk_pack_init_wrapper_<PackMode::Default, false /*zero_output*/>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES /*num_faces*/);
    _llk_pack_dest_init_wrapper_<sync_mode, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<sync_mode, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /*tile_index*/, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<sync_mode, is_fp32_dest_acc_en>();
}

#endif
