// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// wint4_aint8 linear, integer matmul + parallel group-wise SFPU dequant.
// TRISC0-1: int8 x int4 matmul -> Int32 Dest (FPU); every GROUP_KT_DIM K tiles the
//           group's partial sum is handed to the SFPU in its own Dest section.
// TRISC3:   per group, loads the weight scales s_w[g, n] of each tile column into LREGs
//           (one SrcS fill), then rewrites the Dest partials in place as
//           float(partial) * s_w with a replayed per-face sequence.
// TRISC2:   PACK0 writes group 0 and accumulates later groups onto it in L1 (fp32).
// Epilogue: once PACK0 has written the last group, TRISC3 reads each result tile back
//           through SrcS, multiplies by s_a[m] and rewrites it with PACK1.
//
// Perf run types: MATH_ISOLATE times the FPU matmul alone, SFPU_ISOLATE the per-group SFPU
// pass alone (no epilogue), PACK_ISOLATE the group packs alone, L1_TO_L1 the full pipeline.

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_bfd_alloc.h"
#include "llk_unpack_matmul.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t CT_DIM      = params.CT_DIM;
    const std::uint32_t RT_DIM      = params.RT_DIM;
    const std::uint32_t KT_DIM      = params.KT_DIM;
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;
    const Operand& buffer_B         = params.buffer_B;
    const ckernel::TensorShape tensor_shape_A =
        ckernel::make_tensor_shape(params.in0_face_r_dim, params.in0_face_c_dim, params.num_faces_r_dim_A, params.num_faces_c_dim_A);
    const ckernel::TensorShape tensor_shape_B =
        ckernel::make_tensor_shape(params.in1_face_r_dim, params.in1_face_c_dim, params.num_faces_r_dim_B, params.num_faces_c_dim_B);
#else
    constexpr ckernel::TensorShape tensor_shape_A = ckernel::make_tensor_shape(in0_face_r_dim, in0_face_c_dim, num_faces_r_dim_A, num_faces_c_dim_A);
    constexpr ckernel::TensorShape tensor_shape_B = ckernel::make_tensor_shape(in1_face_r_dim, in1_face_c_dim, num_faces_r_dim_B, num_faces_c_dim_B);
#endif

    {
        START_PERF_MEASURE("INIT")
        set_ttsync_enables<TRACK_ALL>(ckernel::TRISC_ID);
        // Matmul flips the unpacker roles: operand A drives UNPACR1/SrcB, operand B drives UNPACR0/SrcA.
        const auto bfd_a =
            ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp1>(tensor_shape_A, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
        const auto bfd_b =
            ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(tensor_shape_B, L1_ADDRESS(buffer_B[0]), formats.unpack_B_src);
        _llk_unpack_hw_configure_<ckernel::p_unpacr::UNP_B>(static_cast<DataFormat>(formats.unpack_A_dst));
        _llk_unpack_hw_configure_<ckernel::p_unpacr::UNP_A>(static_cast<DataFormat>(formats.unpack_B_dst));
        _llk_unpack_matmul_init_<UNPACK_TRANSPOSE_FACES>(bfd_a, bfd_b, CT_DIM, RT_DIM, KT_DIM, tensor_shape_A, tensor_shape_B);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::SFPU_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_matmul_mock(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t j = 0; j < KT_DIM; j++)
                {
                    _llk_unpack_matmul_(CT_DIM, RT_DIM, KT_DIM, j, j * CT_DIM, tensor_shape_A, tensor_shape_B);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t CT_DIM       = params.CT_DIM;
    const std::uint32_t RT_DIM       = params.RT_DIM;
    const std::uint32_t KT_DIM       = params.KT_DIM;
    const std::uint32_t GROUP_KT_DIM = params.GROUP_KT_DIM;
    const std::uint32_t LOOP_FACTOR  = params.LOOP_FACTOR;
    const ckernel::TensorShape tensor_shape_A =
        ckernel::make_tensor_shape(params.in0_face_r_dim, params.in0_face_c_dim, params.num_faces_r_dim_A, params.num_faces_c_dim_A);
    const ckernel::TensorShape tensor_shape_B =
        ckernel::make_tensor_shape(params.in1_face_r_dim, params.in1_face_c_dim, params.num_faces_r_dim_B, params.num_faces_c_dim_B);
#else
    constexpr ckernel::TensorShape tensor_shape_A = ckernel::make_tensor_shape(in0_face_r_dim, in0_face_c_dim, num_faces_r_dim_A, num_faces_c_dim_A);
    constexpr ckernel::TensorShape tensor_shape_B = ckernel::make_tensor_shape(in1_face_r_dim, in1_face_c_dim, num_faces_r_dim_B, num_faces_c_dim_B);
#endif

    {
        START_PERF_MEASURE("INIT")
        // A partial last group would stay in a Dest section that is never handed to the SFPU.
        LLK_ASSERT(GROUP_KT_DIM > 0 && KT_DIM % GROUP_KT_DIM == 0, "KT_DIM must be a whole number of quantization groups");
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            set_up_fpu_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
        }

        const DataFormat math_format = static_cast<DataFormat>(formats.math);
        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en>(math_format, math_format);
        _llk_math_matmul_init_<(ckernel::MathFidelity)MATH_FIDELITY, ENABLE_DIRECT_INDEXING, ENABLE_2X_FORMAT>(CT_DIM, RT_DIM, tensor_shape_A, tensor_shape_B);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::SFPU_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            _perf_math_matmul_mock(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // FPU alone: no Dest handoff, so the inactive SFPU never blocks it.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < KT_DIM; i++)
                {
                    _llk_math_matmul_block_(CT_DIM, RT_DIM);
                }
            }
        }
        else
        {
            // One Dest section per quantization group: the SFPU scales group g while the FPU runs g+1.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < KT_DIM; i++)
                {
                    _llk_math_matmul_block_(CT_DIM, RT_DIM);
                    if ((i + 1) % GROUP_KT_DIM == 0)
                    {
                        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_bfd_alloc.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_pack_common.h"
#include "llk_srcs.h"
#include "llk_sync.h"
#include "llk_unpack_common.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

// TRISC2 <-> TRISC3 handshake around the s_a epilogue (both indices are unused by this kernel's
// dest-dvalid pipeline).
constexpr std::uint8_t ACC_READY_SEM     = semaphore::PACK_UNPACK;
constexpr std::uint8_t EPILOGUE_DONE_SEM = semaphore::UNPACK_MATH;

// Replay slots for the dequant sequence over half a face (8 rows): one per 16-column half of
// the tile, since faces 0/2 (columns 0-15) and 1/3 (columns 16-31) use different scale LREGs.
// Two 16-entry sequences fit the replay buffer where two full-face (32-entry) ones did not.
// A pass covers 4 rows of the even or odd columns: 8 rows x 2 column parities = 4 passes, each
// one LOAD, CAST, MUL and STORE.
constexpr std::uint32_t HALF_FACE_ROWS            = 8;
constexpr std::uint32_t HALF_FACES_PER_FACE       = 2;
constexpr std::uint32_t SFPU_PASSES_PER_HALF_FACE = 4;
constexpr std::uint32_t INSTRS_PER_PASS           = 4;
constexpr std::uint32_t FACE_SEQ_LEN              = SFPU_PASSES_PER_HALF_FACE * INSTRS_PER_PASS;
constexpr std::uint32_t FACE_SEQ_SLOT[2]          = {0, FACE_SEQ_LEN};
// Each pass owns one of LREG0-3 (LREG4-7 hold the scales), and TRISC3's replay buffer holds 32.
static_assert(SFPU_PASSES_PER_HALF_FACE <= p_sfpu::LREG4, "passes would overwrite the scale LREGs");
static_assert(2 * FACE_SEQ_LEN <= 32, "both half-face sequences must fit TRISC3's replay buffer");

// s_w is stored as one SrcS slice per 16-column half of each (group, tile column).
constexpr std::uint32_t W_SCALE_SLICES_PER_TILE_COL = 2;

// Rewrites 8 rows of a 16x16 Dest face in place: float(int32 partial) * column scale. A pass
// covers 4 rows of the even (addr +0) or odd (addr +2) columns, so the scale LREG alternates.
// Recorded stage-major (4 loads, 4 casts, 4 multiplies, 4 stores), each pass in its own LREG0-3,
// so independent work sits between every dependent pair instead of stalling on each result.
template <std::uint32_t SCALE_EVEN, std::uint32_t SCALE_ODD>
inline void record_face_seq(const std::uint32_t slot)
{
    load_replay_buf(
        slot,
        FACE_SEQ_LEN,
        false /*exec_while_loading*/,
        0 /*set_mutex*/,
        0 /*last*/,
        []
        {
#pragma GCC unroll SFPU_PASSES_PER_HALF_FACE
            for (std::uint32_t pass = 0; pass < SFPU_PASSES_PER_HALF_FACE; pass++)
            {
                TT_SFPLOAD(pass, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /*done*/, pass << 1);
            }
#pragma GCC unroll SFPU_PASSES_PER_HALF_FACE
            for (std::uint32_t pass = 0; pass < SFPU_PASSES_PER_HALF_FACE; pass++)
            {
                TT_SFPCAST(pass, pass, 0 /* int32 (sign-magnitude) -> fp32, RNE */);
            }
#pragma GCC unroll SFPU_PASSES_PER_HALF_FACE
            for (std::uint32_t pass = 0; pass < SFPU_PASSES_PER_HALF_FACE; pass++)
            {
                // Scale (SrcS-loaded) as the multiplier operand.
                TT_SFPMUL((pass & 1) ? SCALE_ODD : SCALE_EVEN, pass, p_sfpu::LCONST_0, pass, 0 /*instr_mod1*/);
            }
#pragma GCC unroll SFPU_PASSES_PER_HALF_FACE
            for (std::uint32_t pass = 0; pass < SFPU_PASSES_PER_HALF_FACE; pass++)
            {
                TT_SFPSTORE(pass, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0 /*done*/, pass << 1);
            }
        });
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t CT_DIM      = params.CT_DIM;
    const std::uint32_t RT_DIM      = params.RT_DIM;
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_C         = params.buffer_C;
    const Operand& buffer_S         = params.buffer_S;
    const Operand& buffer_T         = params.buffer_T;
    const Operand& buffer_Res       = params.buffer_Res;
    const std::uint32_t num_groups  = params.KT_DIM / params.GROUP_KT_DIM;
#else
    constexpr std::uint32_t num_groups = KT_DIM / GROUP_KT_DIM;
#endif

    // Scales and results are fp32, so SrcS runs in 32-bit mode: 4-row slices, 16 per tile.
    // The BFDs below all address L1 in SrcS-slice units (4 rows x 16 columns).
    constexpr DataFormat fp32       = DataFormat::Float32;
    constexpr bool srcs_32bit_mode  = true;
    const std::uint32_t srcs_ydim   = srcs_dims::ydim(srcs_32bit_mode);
    const std::uint32_t slice_count = srcs_dims::slice_count(srcs_32bit_mode);
    const int in0_base              = ckernel::math::SFPU_SRCS_BASE_ADDR;
    const int in1_base              = ckernel::math::SFPU_SRCS_BASE_ADDR + srcs_ydim;
    const int out_base              = ckernel::math::SFPU_SRCS_BASE_ADDR + 2 * srcs_ydim;

    constexpr bool active = PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::SFPU_ISOLATE;
    constexpr bool full   = PERF_RUN_TYPE == PerfRunType::L1_TO_L1;

    std::uint8_t bfd_w_scale_lo = 0, bfd_w_scale_hi = 0, bfd_acc = 0, bfd_a_scale = 0, bfd_pack = 0, bfd_scratch = 0;
    {
        START_PERF_MEASURE("INIT")
        if constexpr (full)
        {
            set_up_fpu_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::SFPU>();
            _llk_sync_init_(ACC_READY_SEM, 2 /*max*/, 0 /*init*/);
            _llk_sync_init_(EPILOGUE_DONE_SEM, 2 /*max*/, 0 /*init*/);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::SFPU_ISOLATE)
        {
            // SFPU alone: read Dest without waiting for the inactive FPU. CFG persists across run types.
            set_up_zero_dest_dvalid_handshake_for_sfpu();
        }

        if constexpr (active)
        {
            const ckernel::TensorShape srcs_shape =
                ckernel::make_tensor_shape(static_cast<std::uint8_t>(srcs_ydim), srcs_dims::XDIM, srcs_dims::ZDIM, srcs_dims::ZDIM);

            // Weight scales s_w[g, n]: one slice per (group, tile column, 16-column half), the 16
            // column scales repeated down 4 rows. The two halves get their own BFD (offset by one
            // slice) so one L1 index addresses both: changing the index between the two UNPACR2s,
            // by SET or by L1 increment, made both read the second half.
            constexpr std::uint32_t slice_bytes_16B = 4 /*rows*/ * 16 /*cols*/ * 4 /*bytes*/ / 16;
            bfd_w_scale_lo                          = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp2_Slice0>(
                srcs_shape, L1_ADDRESS(buffer_C[0]), ckernel::to_underlying(fp32));
            bfd_w_scale_hi = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp2_Slice1>(
                srcs_shape, L1_ADDRESS(buffer_C[0]) + slice_bytes_16B, ckernel::to_underlying(fp32));
            // Epilogue inputs: the accumulated result read back, and s_a[m] expanded to one tile per
            // output tile.
            bfd_acc = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp2_Slice0>(
                srcs_shape, L1_ADDRESS(buffer_Res[0]), ckernel::to_underlying(fp32));
            bfd_a_scale = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp2_Slice1>(
                srcs_shape, L1_ADDRESS(buffer_T[0]), ckernel::to_underlying(fp32));
            _llk_unpack_configure_unary_<p_unpacr::UNP_S>(fp32);

            // SrcS slice 2 -> L1: the epilogue result, or a scratch slice that only closes the SrcS
            // bank cycle (unpack -> SFPU -> PACK1) after the scale loads.
            bfd_pack =
                ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack1>(srcs_shape, L1_ADDRESS(buffer_Res[0]), ckernel::to_underlying(fp32));
            bfd_scratch =
                ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack1>(srcs_shape, L1_ADDRESS(buffer_S[0]), ckernel::to_underlying(fp32));
            _llk_pack_hw_configure_<p_pacr::PACK1, false /*EN_32BIT_DEST*/>(fp32, ckernel::ReluConfig::none());

            cfg[DISABLE_IMPLIED_SRCS_FORMAT_ADDR32 + TRISC_ID] = !IMPLIED_MATH_FORMAT;

            // UNPACR2 is issued explicitly below, so the auto-loop must not replay it.
            _llk_unpack_srcs_config_<1 /*INSTRN_COUNT*/, 1 /*INSTRN_LOOP_COUNT*/>();
            _llk_pack_srcs_config_<1 /*INSTRN_COUNT*/, 1 /*INSTRN_LOOP_COUNT*/>();
            _llk_math_eltwise_sfpu_init_();

            // LREG4/5: columns 0-15 even/odd, LREG6/7: columns 16-31 even/odd.
            record_face_seq<p_sfpu::LREG4, p_sfpu::LREG5>(FACE_SEQ_SLOT[0]);
            record_face_seq<p_sfpu::LREG6, p_sfpu::LREG7>(FACE_SEQ_SLOT[1]);
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (active)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                // ---- Per group: Dest partial -> float(partial) * s_w[g, n], in place ----
                for (std::uint32_t group = 0; group < num_groups; ++group)
                {
                    for (std::uint32_t col_tile = 0; col_tile < CT_DIM; ++col_tile)
                    {
                        // One SrcS fill brings both 16-column halves' scales; LREG4-7 then hold them
                        // for every row tile of this column.
                        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_S, (group * CT_DIM + col_tile) * W_SCALE_SLICES_PER_TILE_COL);
                        TT_UNPACR2_TILE_INC(0b1 /*SrcS tile inc*/, 0b0 /*no L1 inc*/, bfd_w_scale_lo, 0b0 /*no dvalid*/);
                        TT_UNPACR2_TILE_INC(0b0 /*no SrcS tile inc*/, 0b0 /*no L1 inc*/, bfd_w_scale_hi, 0b1 /*Set dvalid*/);
                        _llk_pack_srcs_<1 /*INSTRN_COUNT*/>(bfd_scratch, 0);
                        TT_SFPLOAD(p_sfpu::LREG4, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0 /*done*/, in0_base);
                        TT_SFPLOAD(p_sfpu::LREG5, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0 /*done*/, in0_base + 2);
                        TT_SFPLOAD(p_sfpu::LREG6, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0 /*done*/, in1_base);
                        TT_SFPLOAD(p_sfpu::LREG7, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0 /*done*/, in1_base + 2);
                        _llk_math_eltwise_sfpu_srcs_clear_vlds_<true /*SRCS_RD_DONE*/, true /*SRCS_WR_DONE*/>();

                        for (std::uint32_t row_tile = 0; row_tile < RT_DIM; ++row_tile)
                        {
                            _llk_math_eltwise_sfpu_start_(row_tile * CT_DIM + col_tile);
#pragma GCC unroll 4
                            for (std::uint32_t face = 0; face < 4; ++face)
                            {
#pragma GCC unroll HALF_FACES_PER_FACE
                                for (std::uint32_t half = 0; half < HALF_FACES_PER_FACE; ++half)
                                {
                                    TTI_REPLAY(
                                        FACE_SEQ_SLOT[face & 1], FACE_SEQ_LEN, 0 /*last*/, 0 /*set_mutex*/, 0 /*execute_while_loading*/, 0 /*load_mode*/);
                                    _inc_dst_addr_<HALF_FACE_ROWS>();
                                }
                            }
                            _llk_math_eltwise_sfpu_done_();
                        }
                    }
                    if constexpr (full)
                    {
                        _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
                    }
                }

                if constexpr (full)
                {
                    // ---- Epilogue: out = acc * s_a[m], written over the accumulated tiles ----
                    // PACK0 posts once it has written the last group of this loop. STALL_SYNC holds
                    // the SEMGET (it runs on the sync unit), STALL_UNPACK holds the epilogue reads.
                    _llk_sync_wait_<p_stall::STALL_SYNC | p_stall::STALL_UNPACK, p_stall::STALL_ON_ZERO>(ACC_READY_SEM);
                    _llk_sync_get_(ACC_READY_SEM);

                    // s_a is laid out per output tile, so both reads share one L1 index (no SET between them).
                    for (std::uint32_t tile = 0; tile < CT_DIM * RT_DIM; ++tile)
                    {
                        for (std::uint32_t slice = 0; slice < slice_count; ++slice)
                        {
                            TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_S, tile * slice_count + slice);
                            TT_UNPACR2_TILE_INC(0b1 /*SrcS tile inc*/, 0b0 /*no L1 inc*/, bfd_acc, 0b0 /*no dvalid*/);
                            TT_UNPACR2_TILE_INC(0b0 /*no SrcS tile inc*/, 0b0 /*no L1 inc*/, bfd_a_scale, 0b1 /*Set dvalid*/);
                            _llk_pack_srcs_<1 /*INSTRN_COUNT*/>(bfd_pack, tile * slice_count + slice);

#pragma GCC unroll 2
                            for (int col = 0; col < 4; col += 2)
                            {
                                TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0 /*done*/, in0_base + col);
                                TT_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0 /*done*/, in1_base + col);
                                TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /*instr_mod1*/);
                                TT_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0 /*done*/, out_base + col);
                            }
                            _llk_math_eltwise_sfpu_srcs_clear_vlds_<true /*SRCS_RD_DONE*/, true /*SRCS_WR_DONE*/>();
                        }
                    }

                    // Let PACK0 overwrite the tiles (next loop's group 0) only once PACK1 is done.
                    t6_semaphore_post<p_stall::PACK1>(EPILOGUE_DONE_SEM);
                }
            }
        }
        wait_sfpu_idle();
        wait_pack_idle();
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_bfd_alloc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "llk_pack_matmul.h"
#include "llk_sync.h"
#include "params.h"

constexpr std::uint8_t ACC_READY_SEM     = semaphore::PACK_UNPACK;
constexpr std::uint8_t EPILOGUE_DONE_SEM = semaphore::UNPACK_MATH;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t CT_DIM      = params.CT_DIM;
    const std::uint32_t RT_DIM      = params.RT_DIM;
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_groups  = params.KT_DIM / params.GROUP_KT_DIM;
    const Operand& buffer_Res       = params.buffer_Res;
    const ckernel::TensorShape output_shape =
        ckernel::make_tensor_shape(params.in0_face_r_dim, params.in1_face_c_dim, params.num_faces_r_dim_A, params.num_faces_c_dim_B);
#else
    constexpr std::uint32_t num_groups          = KT_DIM / GROUP_KT_DIM;
    constexpr ckernel::TensorShape output_shape = ckernel::make_tensor_shape(in0_face_r_dim, in1_face_c_dim, num_faces_r_dim_A, num_faces_c_dim_B);
#endif

    constexpr bool active = PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION;
    {
        START_PERF_MEASURE("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            set_up_fpu_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
        }
        else if constexpr (active)
        {
            set_up_zero_dest_dvalid_handshake_for_pack();
        }

        if constexpr (active)
        {
            // The SFPU leaves fp32 in Dest; the Int32 format inferred from the matmul does not apply.
            const auto bfd_pack =
                ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(output_shape, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
            _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(DataFormat::Float32, ckernel::ReluConfig::none());
            _llk_pack_matmul_init_(bfd_pack, RT_DIM, CT_DIM, 1 /*num_subblocks_c_dim*/, output_shape);
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (active)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    // The previous loop's epilogue rewrites these tiles through PACK1. STALL_SYNC holds
                    // the SEMGET, STALL_PACK holds the packs.
                    if (loop > 0)
                    {
                        _llk_sync_wait_<p_stall::STALL_SYNC | p_stall::STALL_PACK, p_stall::STALL_ON_ZERO>(EPILOGUE_DONE_SEM);
                        _llk_sync_get_(EPILOGUE_DONE_SEM);
                    }
                }
                for (std::uint32_t group = 0; group < num_groups; ++group)
                {
                    // Group 0 writes the result tiles; later groups accumulate onto them in L1 (fp32).
                    TTI_STALLWAIT(p_stall::STALL_CFG, 0, 0, p_stall::PACK0);
                    _llk_pack_set_l1_acc_<p_pacr::PACK0>(group > 0 /*l1_acc_en*/);
                    _llk_pack_matmul_(0 /*start_math_dest_tile_idx*/, 0 /*start_l1_tile_idx*/, output_shape);
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
                }
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    t6_semaphore_post<p_stall::PACK0>(ACC_READY_SEM);
                }
            }
            TTI_STALLWAIT(p_stall::STALL_CFG, 0, 0, p_stall::PACK0);
            _llk_pack_set_l1_acc_<p_pacr::PACK0>(false /*l1_acc_en*/);
        }
        PROFILER_SYNC();
    }
}

#endif
