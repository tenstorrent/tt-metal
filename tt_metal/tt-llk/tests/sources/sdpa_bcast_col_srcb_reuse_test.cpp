// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ADVANCE TEST: covers experimental LLK sdpa_bcast_col_srcb_reuse (+ unpack_A_sdpa) (tt-metal#47554 /
// tt-blaze#1971), promoted into tt_llk_blackhole/llk_lib/experimental/ on main by #53295. The includes below resolve
// through the canonical -I; the demo-fork shadow tree this test was first written against no longer exists.
//
// What the op does (tt-blaze#1971): eltwise ADD/SUB/MUL of a per-tile operand (SrcA) with a *column* broadcast
// (SrcB) where the column source is DEST reused as a source register (DEST -> SrcB via MOVD2B), reused across every
// SrcA row (the softmax scale / normalize step). This advance test exercises all three instantiations -- MUL (the
// softmax-scale path, which also has high-fidelity ELWMUL support; LoFi is pinned here for the small bf16 grid), ADD
// and SUB -- selected per variant through ELTWISE_BINARY_OP from the generated build header.
//
// SrcB-reuse + set_srcb_dummy_valid handshake modeled here (same shape as transpose_dest_test.cpp's dest-reuse
// handshake; the external reference for the op itself is tt-blaze#1971, cited above):
//   1. Seed DEST[SRC_INDEX] with the column-source tile (buffer_B) via an ordinary A2D datacopy. In the real op this
//      tile is the per-column softmax scale produced by a prior reduce; here we feed it directly so the golden is a
//      plain column broadcast.
//   2. Call _llk_unpack_A_sdpa_set_srcb_dummy_valid_() -- BEFORE the operand unpacks -- to inject STALL_UNPACK plus a
//      UNPACR_NOP SET_DVALID on SrcB with no real data, so the downstream dual-source eltwise op sees SrcB "ready". This
//      is the dummy SrcB the math preamble's STALLWAIT(SRCB_VLD) waits on before its MOVD2B reads DEST into SrcB.
//   3. Unpack TWO operand groups of NUM_TILES tiles each (buffer_A[0..] and buffer_A[NUM_TILES..]) into SrcA under
//      the SrcA-only llk_unpack_A_sdpa MOP, one llk_unpack_A call per group. The execute runs the MOP twice and every
//      eltwise op carries CLR_A, so it retires 2 * NUM_TILES * num_faces SrcA dvalids; one group stalls MATH forever.
//      The demo pairs the same way (compute_kernel_api/sdpa.h:56-57).
//   4. _llk_math_sdpa_bcast_col_srcb_reuse_preamble_ MOVD2Bs DEST rows -> SrcB (P1 and P2), then
//      _llk_math_sdpa_bcast_col_srcb_reuse_ runs the SRCB_BCAST_COL eltwise MOP and the postamble clears SrcB. With
//      A_t the t-th operand tile of the first group and A_{N+t} its partner in the second group (N == NUM_TILES):
//      DEST[DST_INDEX + t] = (A_t op broadcast_col(P1)) + (A_{N+t} op broadcast_col(P2))   (both passes accumulate).
//
// This kernel covers unpack_A_sdpa as well, because exercising either primitive in isolation is impossible:
// unpack_A_sdpa is init/mop-config plus the dummy-SrcB-valid helper and has no execute of its own, and the math op
// cannot run without them. All three of its symbols are driven here: _llk_unpack_A_sdpa_init_ (the SrcA-only UNPACR
// MOP), the base llk_unpack_A execute under that MOP, and _llk_unpack_A_sdpa_set_srcb_dummy_valid_. The test that
// pinned unpack_A_sdpa by name (test_unpack_A_sdpa.py) is owned by #53361.
//
// Blackhole-only. The golden is verified on Blackhole silicon (p100a), not compile-green only.

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

using namespace ckernel;

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr DstSync DST_SYNC = DstSync::SyncHalf;

// Single 8x32 tile: two 8x16 faces (num_faces == 2). The SDPA mop config LLK_ASSERTs num_faces == 2 (the init's
// assert allows 1/2/4, but sdpa_bcast_col_srcb_reuse_configure_mop hard-asserts == 2), so a 2-face tile is the only
// shape this primitive instantiates -- and 8 rows is the only face_r_dim that matches what it writes. Verified on
// p100a: the MOP is one eltwise op per face and each op covers 8 dest rows, so the op's output tile is 8x32. This is
// also the demo's tile ("Each tile is 8x32, which is the same as a full 16x16 face" -- sdpa.h:317).
//
// Two column sources are needed, not one: the preamble MOVD2Bs DEST rows 0-7 into SrcB rows 0-7 (P1) and DEST rows
// 64-71 -- the first 8 rows of the NEXT 32x32 dest tile -- into SrcB rows 8-15 (P2). So P1 lives in DEST[SRC_INDEX]
// and P2 in DEST[SRC_INDEX + 1]. Reuse-in-place: the output goes back to DEST[DST_INDEX] == DEST[SRC_INDEX] (and,
// with NUM_TILES == 2, over P2 in DEST[SRC_INDEX + 1]), which is safe because both MOVD2Bs run before the first
// eltwise op.
//
// NUM_TILES is the MOP's outer-loop count (SDPA_BCAST_NUM_TILES from the build header): one llk_unpack_A call walks
// this many tiles and the math MOP writes them to DEST[DST_INDEX ..]; 2 is what reaches the MOP's last_inner_loop slot.
static constexpr std::uint32_t NUM_TILES = SDPA_BCAST_NUM_TILES;
static constexpr std::uint32_t SRC_INDEX = 0;
static constexpr std::uint32_t DST_INDEX = 0;

// num_faces MUST be a compile-time constant on the math thread: sdpa_bcast_col_srcb_reuse_configure_addrmod feeds
// (16 + (num_dest_faces - num_faces)*16) into the ADDR_MOD dest.incr, which lands in a SETC16 whose immediate takes
// the "n" (integer-constant) asm constraint, so a runtime value trips "impossible constraint in 'asm'". That is why
// the python driver passes NUM_FACES as a TEMPLATE parameter rather than a runtime one: `num_faces` below is the
// constexpr the generated build header declares (helpers/sdpa_bcast_utils.py), so the value lives in python only and
// the unpack/pack sides read the same constant the math mop does.

// The op (ELWMUL / ELWADD / ELWSUB) is SDPA_OP, bound below from the build header's ELTWISE_BINARY_OP; LoFi fidelity.
static constexpr MathFidelity SDPA_FIDELITY = MathFidelity::LoFi;

// The MOP's two passes accumulate into DEST for every op (ADD/SUB set the dest-accumulate bit; MUL always does), so
// the output tiles must be zero on entry. clear_dest == true makes the preamble ZEROACC the dest half *after* its
// MOVD2Bs have latched P1/P2 into SrcB; without it the column-source seeds still sitting in the output tiles (P1 in
// DEST[0], and P2 in DEST[1] when NUM_TILES == 2) are added in (verified on p100a for MUL). The demo takes the same
// branch on its normalize path (sdpa.h: sdpa_bcast_col_reuse_preamble<normalize>()).
static constexpr bool CLEAR_DEST = true;

#ifdef LLK_TRISC_UNPACK

// PRIMITIVE symbols under test (NOT the forked _api.h wrapper / compute_kernel_api entry).
// Resolved from the promoted experimental/ copy (landed on main via #53295); the demo-fork shadow tree this test was
// originally written against is gone.
//
// The promoted header still takes transpose_of_faces / unpack_src_format / unpack_dst_format params that the
// num_faces == 2 path we instantiate does not read, and the harness compiles with -Werror -Wunused-parameter. The
// offending params are on template bodies, so an include-only push/pop does not reach the instantiation point --
// suppress at file scope. Drop this once the promoted header is warning-clean (tracked in #53295).
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_unpack_A_sdpa.h"
// Base unpack_A supplies the per-tile execute for both the DEST seed and the operand stream; llk_unpack_A_sdpa.h is
// init/mop-config + the dummy-SrcB-valid helper only.
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

static inline void run_unpack_iteration(
    const Operand& buffer_A, const Operand& buffer_B, const FormatConfig& formats, const ckernel::TensorShape& tensor_shape, std::uint32_t face_r_dim)
{
    _llk_unpack_A_init_<BroadcastType::NONE>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE>(L1_ADDRESS(buffer_B[0]), formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE>(L1_ADDRESS(buffer_B[1]), formats.unpack_A_src, formats.unpack_A_dst);

    _llk_unpack_A_sdpa_init_<NUM_TILES, BroadcastType::NONE>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, face_r_dim, num_faces, formats.unpack_A_src, formats.unpack_A_dst);
    // The preamble waits for this SrcB token before reading the column sources from Dest.
    _llk_unpack_A_sdpa_set_srcb_dummy_valid_();
    _llk_unpack_A_<BroadcastType::NONE>(L1_ADDRESS(buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE>(L1_ADDRESS(buffer_A[NUM_TILES]), formats.unpack_A_src, formats.unpack_A_dst);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t TEST_FACE_R_DIM = params.in0_face_r_dim;
    const Operand& buffer_A             = params.buffer_A;
    const Operand& buffer_B             = params.buffer_B;
#else
    constexpr std::uint32_t TEST_FACE_R_DIM = in0_face_r_dim;
#endif
    // 8x32 tile: one face-row (num_faces_r_dim == 1) of two 8x16 faces (num_faces_c_dim == 2).
    // The op writes 8 rows per face, so 8 is the natural face_r_dim -- see the SDPA_TILE_R_DIM note above.
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(TEST_FACE_R_DIM), static_cast<std::uint8_t>(FACE_C_DIM), 1 /* num_faces_r_dim */, 2 /* num_faces_c_dim */};

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, TEST_FACE_R_DIM, TEST_FACE_R_DIM, num_faces, num_faces);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                // The two generic unpack_A seed copies publish paired SrcA and zero-SrcB valids per face.
                _perf_unpack_stream_set_valid<true /* set_a */, true /* set_b */, PERF_RUN_TYPE>(2 * num_faces);
                _perf_unpack_set_valid(ckernel::SrcB);
                _perf_unpack_stream_set_valid<true /* set_a */, false /* set_b */, PERF_RUN_TYPE>(2 * NUM_TILES * num_faces);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                run_unpack_iteration(buffer_A, buffer_B, formats, tensor_shape, TEST_FACE_R_DIM);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

// PRIMITIVE symbol under test (NOT the forked _api.h wrapper / compute_kernel_api entry), from the promoted
// experimental/ copy.
//
// Its configure_mop declares addr_mod / innerloop / outerloop (llk_math_sdpa_bcast_col_srcb_reuse.h:23-25) without
// reading them on the path we instantiate, which trips -Werror=unused-variable. Same file-scope suppression rationale
// as the unpack side: the declarations are inside template bodies, so an include-only push/pop does not reach the
// instantiation point. These are the symbols #53361 asked @pmilenkovicTT to drop in #53295; this shim goes when they do.
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "experimental/llk_math_sdpa_bcast_col_srcb_reuse.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

// ELTWISE_BINARY_OP is the per-variant op the python driver emits into the build header (MATH_OP template).
static constexpr EltwiseBinaryType SDPA_OP = ELTWISE_BINARY_OP;

static inline void run_math_iteration(const FormatConfig& formats, const ckernel::TensorShape& tensor_shape)
{
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE>(tensor_shape, formats.math);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en>(SRC_INDEX, formats.math, formats.math, tensor_shape);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en>(SRC_INDEX + 1, formats.math, formats.math, tensor_shape);
    _llk_math_sdpa_bcast_col_srcb_reuse_init_<SDPA_OP, NUM_TILES, SDPA_FIDELITY>(num_faces);
    _llk_math_sdpa_bcast_col_srcb_reuse_preamble_<DST_SYNC, is_fp32_dest_acc_en, CLEAR_DEST>();
    _llk_math_sdpa_bcast_col_srcb_reuse_<SDPA_OP, NUM_TILES, DST_SYNC, is_fp32_dest_acc_en, SDPA_FIDELITY, CLEAR_DEST>(DST_INDEX);
    _llk_math_sdpa_bcast_col_srcb_reuse_postamble_();
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t TEST_FACE_R_DIM = params.in0_face_r_dim;
#else
    constexpr std::uint32_t TEST_FACE_R_DIM = in0_face_r_dim;
#endif
    // Same 8x32 shape as the unpack side.
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(TEST_FACE_R_DIM), static_cast<std::uint8_t>(FACE_C_DIM), 1 /* num_faces_r_dim */, 2 /* num_faces_c_dim */};

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _perf_math_stream_clear_valid<true /* clear_a */, true /* clear_b */, PERF_RUN_TYPE>(2 * num_faces);
                _perf_math_stream_clear_valid<true /* clear_a */, false /* clear_b */, PERF_RUN_TYPE>(2 * NUM_TILES * num_faces);
                _perf_math_clear_valid(ckernel::SrcB);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_wait_for_dest_available_<DST_SYNC>();
                }
                run_math_iteration(formats, tensor_shape);
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

static inline void run_pack_iteration(const Operand& buffer_Res, std::uint32_t tile_cnt)
{
    for (std::uint32_t tile = 0; tile < tile_cnt; ++tile)
    {
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(buffer_Res[tile]));
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT        = params.TILE_CNT;
    const std::uint32_t TILE_SIZE_PACK  = params.TILE_SIZE_PACK;
    const std::uint32_t TEST_FACE_R_DIM = params.in0_face_r_dim;
    const Operand& buffer_Res           = params.buffer_Res;
#else
    constexpr std::uint32_t TEST_FACE_R_DIM = in0_face_r_dim;
#endif
    {
        START_PERF_MEASURE("INIT")
        // Partial-face packing preserves the two 8-row faces in this 8x32 tile.
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, TILE_SIZE_PACK, TEST_FACE_R_DIM, TILE_C_DIM, num_faces, true /* partial_face */);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, TEST_FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_packer_wait_for_math_done_();
                }
                run_pack_iteration(buffer_Res, TILE_CNT);
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
