// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// lane FS  FP-3 architectural model: shared datacopy-based vehicle.
//
// Full unpack -> math -> pack pipeline (cloned from eltwise_unary_datacopy_
// test.cpp) so DEST is a properly acquired, 32-bit, pack-visible bank.  After
// datacopy fills DEST with the input tile, the math body injects a payload
// that overwrites DEST tile 0 with the sentinel float 123.5f
// (BF16 0x42F7 -> FP32 0x42F70000) through the Replay Expander in one of four
// modes selected by FS_MODE:
//
//   0 INLINE        execute sfploadi+sfpstore directly (no replay) -- vehicle
//                   control: sentinel MUST appear.
//   1 RECORD_LAUNCH record the payload (no-exec) in one noinline function then
//                   launch it in another (cross-function, one launch) -- the
//                   pfj1 within-launch reassembly the FP-3 reach rule misses.
//   2 RECORD_ONLY   record the payload (no-exec), never launch -- kernel A of
//                   the cross-invocation experiment; produces plain datacopy.
//   3 LAUNCH_ONLY   launch slots 0..1 with NO record in this kernel -- kernel B
//                   of the cross-invocation experiment; sentinel appears iff the
//                   Replay buffer persisted from a prior kernel A invocation.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, params.num_faces), formats.unpack_A_src, formats.unpack_A_dst);

    const std::uint32_t num_tiles = params.NUM_BLOCKS * params.NUM_TILES_IN_BLOCK;
    for (std::uint32_t i = 0; i < num_tiles; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#ifdef FORMAT_INT32
const bool is_int_fpu_en = true;
#else
const bool is_int_fpu_en = false;
#endif

#include "ckernel_ops.h"
#include "llk_lib_math_wrappers.h"
#include "params.h"

using namespace ckernel;

// Sentinel: SFPLOADI FLOATB (BF16 imm) 0x42F7 -> FP32 0x42F70000 == 123.5f.
static constexpr std::uint32_t FS_SENTINEL_BF16 = 0x42F7;

#if FS_MODE == 1 || FS_MODE == 2
static __attribute__((noinline)) void fs_record_arm()
{
    // NO-EXEC record of slots 0..1 (execute_while_loading=0, Load=1).
    TTI_REPLAY(0, 2, 0, 1);
    TTI_SFPLOADI(p_sfpu::LREG0, 0 /* FLOATB */, FS_SENTINEL_BF16);
    TTI_SFPSTORE(p_sfpu::LREG0, 3 /* MOD0_FMT_FP32 */, 0, 0);
}
#endif

#if FS_MODE == 1 || FS_MODE == 3
static __attribute__((noinline)) void fs_launch_arm()
{
    TTI_REPLAY(0, 2, 0, 0); // launch slots 0..1
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SFPU1);
    TTI_NOP;
}
#endif

static inline void fs_inject_payload()
{
#if FS_MODE == 0 // INLINE
    TTI_SFPLOADI(p_sfpu::LREG0, 0 /* FLOATB */, FS_SENTINEL_BF16);
    TTI_SFPSTORE(p_sfpu::LREG0, 3 /* MOD0_FMT_FP32 */, 0, 0);
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SFPU1);
    TTI_NOP;
#elif FS_MODE == 1 // RECORD_LAUNCH (cross-function, within one launch)
    fs_record_arm();
    fs_launch_arm();
#elif FS_MODE == 2 // RECORD_ONLY (kernel A: record, never launch)
    fs_record_arm();
#elif FS_MODE == 3 // LAUNCH_ONLY (kernel B: launch, no record here)
    fs_launch_arm();
#endif
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_eltwise_unary_datacopy_init_wrapper_<
        DataCopyType::A2D,
        is_fp32_dest_acc_en,
        BroadcastType::NONE,
        is_int_fpu_en,
        llk_test_pack_mode_v<false, false>>(params.num_faces, formats.math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    for (int block_num = 0; block_num < params.NUM_BLOCKS; ++block_num)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (std::uint32_t tile_num = 0; tile_num < params.NUM_TILES_IN_BLOCK; ++tile_num)
        {
            _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                params.DST_INDEX + tile_num, formats.math, formats.math, params.num_faces);
        }
        // Inject the replay payload while DEST is still acquired by math.
        fs_inject_payload();
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<false, false>>(
        formats.pack_src, formats.pack_dst, 16 * 16 * 4, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_init_wrapper_<llk_test_pack_mode_v<false, false>, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    for (int block_num = 0; block_num < params.NUM_BLOCKS; ++block_num)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile_num = 0; tile_num < params.NUM_TILES_IN_BLOCK; ++tile_num)
        {
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                params.DST_INDEX + tile_num, L1_ADDRESS(params.buffer_Res[block_num * params.NUM_TILES_IN_BLOCK + tile_num]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}
#endif
