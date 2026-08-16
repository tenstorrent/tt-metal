// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================
// End-to-end cost of a NEGATIVE-threshold Top-K filter pass (Blackhole)
// ============================================================================
//
// WHAT THIS SETTLES
// -----------------
// The zero-SFPU Top-K path (packer MIN_THRESHOLD_RELU + zero-compression) is
// 4.175 cyc/vector end to end against _topk_xl_merge_'s 6.930 in the same
// kernel, but it cannot express a negative threshold (Packers/ReLU.md:41 makes
// signbit(Threshold) UndefinedBehavior; measured on silicon this session, the
// packer compares against |Threshold| instead). Signed logits need one, so the
// fallback is an SFPU filter -- and its cost decides whether the win survives.
//
// The filter itself is sources/topk_negfilter_common.h: two SFPLOADMACROs per
// 32-element vector computing Dst[i] = (Dst[i] > T) ? Dst[i] : +0.0, bitwise
// only. Two issues per vector is the analytical floor for a value-preserving
// filter here (see that header); this kernel measures what the floor costs in a
// real three-thread pipeline.
//
// THE ARMS, and why each one is carried
// -------------------------------------
//   none      SFPU off. The floor -- unpack + pack of the same stream.
//   ctrlload  CONTROL. Replay+MOP-fed stream of plain TTI_SFPLOAD, one per
//             vector. Establishes the 1-instruction/vector rate in THIS kernel.
//   ctrlswap  CONTROL / tripwire. Same structure with TTI_SFPSWAP, which is 2
//             backend cycles with a hardware bubble that cannot be filled from
//             this thread (SFPSWAP.md). It MUST come out at ~2.0x ctrlload; if
//             it does not, the run is invalid.
//   mask1     CALIBRATION. The 1-macro/vector MaskStore probe (SFPGT SET_VD
//             then SFPSTORE). NOT a usable filter -- it stores the -1/0 mask
//             and destroys the values -- but it is the published 1.003
//             MATH_ISOLATE / +1.5 L1_TO_L1 datapoint, so carrying it here
//             converts "the SFPU costs X" into a difference of differences.
//   negfilter THE CANDIDATE. 2 macros/vector, value-preserving, bit-exact
//             (test_topk_negfilter.py, 6/6).
//   xlmerge   THE COMPETITION. _topk_xl_merge_<512,false,true> on the math
//             thread of this same pipeline, against the same streamed operand
//             and the same packer.
//
// MEASUREMENT: L1_TO_L1, and why not an isolate
// ---------------------------------------------
// L1_TO_L1 timestamps unpack's ZONE_START against pack's ZONE_END, i.e. the
// whole three-thread pipeline. That is the basis that matters here, because
// PACK overlaps the unpacker but the SFPU does not -- math and unpack both
// drive the Dst register file -- so SFPU work ADDS to the 3.938 cyc/vector
// unpack floor rather than hiding under it. A two-point slope over TILE_CNT
// cancels the pipeline fill/drain and every one-time cost. MATH_ISOLATE is
// reported alongside as the component number.
//
// Compile-time knobs (emitted into build.h by the python driver):
//   FILTER_ARM             - 0 none / 1 mask1 / 2 negfilter / 3 ctrlload / 4 ctrlswap
//   XL_MERGE_EN            - run _topk_xl_merge_ instead (preprocessor: it gates an #include)
//   COMPRESS_EN            - clear THCON_SEC0_REG1_Disable_zero_compress
//   ROW_START_SECTION_SIZE - THCON_SEC0_REG1_Row_start_section_size, 16 B units
//   DOWNSAMPLE_MASK        - ALWAYS written: set_packer_config never touches
//                            THCON_SEC0_REG1 word 3, so a mask left behind by an
//                            earlier kernel survives an ELF reload and silently
//                            decimates this pack
//   THR_BITS               - raw 32-bit threshold for the SFPU compare
//   RES_SLOTS / SRC_SLOTS  - ring sizes for the L1 buffers (power of two)

#include <algorithm>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "counters.h"
#include "llk_defs.h"
#include "lltt.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr std::uint32_t MAX_TILES_DEST          = is_fp32_dest_acc_en ? 4 : 8;
static constexpr ckernel::DstSync DST_SYNC_MODE        = ckernel::DstSync::SyncHalf;
static constexpr ckernel::BroadcastType BROADCAST_TYPE = ckernel::BroadcastType::NONE;

static constexpr std::uint32_t ARM_NONE      = 0;
static constexpr std::uint32_t ARM_MASK1     = 1;
static constexpr std::uint32_t ARM_NEGFILTER = 2;
static constexpr std::uint32_t ARM_CTRLLOAD  = 3;
static constexpr std::uint32_t ARM_CTRLSWAP  = 4;
// The per-tile SFPU envelope on its own: _llk_math_eltwise_sfpu_start_/_done_
// plus the drain SFPNOPs, with NO MOP body. Every other SFPU arm pays this, so
// subtracting it is what turns a raw per-vector slope into an issue rate that
// can be checked against the published SFPLOAD=1.000 / SFPSWAP=2.000 controls.
static constexpr std::uint32_t ARM_CTRLENV = 5;

static constexpr bool SFPU_EN = (FILTER_ARM != ARM_NONE);

// 32 SFPLOADs cover one 32x32 tile.
static constexpr std::uint32_t VECTORS_PER_TILE = 32;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t num_faces = params.num_faces;
    const std::uint32_t TILE_CNT  = params.TILE_CNT;
    const auto& buffer_A          = params.buffer_A;
#endif
    const EltwiseBinaryReuseDestType reuse_dest_type = EltwiseBinaryReuseDestType::NONE;

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
        _llk_unpack_A_init_<BROADCAST_TYPE, false /* acc_to_dest */, reuse_dest_type, unpack_to_dest>(
            0 /* transpose_of_faces */,
            0 /* within_face_16x16_transpose */,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
            formats.unpack_A_src,
            formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE && PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t i = 0; i < TILE_CNT; ++i)
            {
                _llk_unpack_A_<BROADCAST_TYPE, false, reuse_dest_type, unpack_to_dest>(
                    L1_ADDRESS(buffer_A[i & (SRC_SLOTS - 1)]), formats.unpack_A_src, formats.unpack_A_dst);
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "topk_negfilter_common.h"

#if XL_MERGE_EN
// Included only under this arm: the experimental SFPU trees are separate and
// pulling in more than one at a time is not supported.
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"
#endif

namespace
{
// ---------------------------------------------------------------------------
// mask1 -- the published MaskStore probe, carried for calibration only.
// One macro per vector: Load + Simple(SFPGT, SET_VD, delay 0) + Store(delay 1).
// The SFPGT overwrites the loaded register with its -1/0 mask and the store
// writes that mask back, so this DESTROYS the values. It is a timing probe.
// ---------------------------------------------------------------------------
constexpr std::uint32_t M1_L_A   = 0;
constexpr std::uint32_t M1_L_B   = 1;
constexpr std::uint32_t M1_L_THR = 3;

constexpr std::uint32_t M1_ADDR_MOD = ckernel::ADDR_MOD_6;

constexpr std::uint32_t M1_SEQ_SIMPLE = 0x80 | (0u << 3) | 4u;
constexpr std::uint32_t M1_SEQ_STORE  = (1u << 3) | 3u;
constexpr std::uint32_t M1_SEQUENCE   = (M1_SEQ_STORE << 24) | M1_SEQ_SIMPLE;
constexpr std::uint32_t M1_MISC       = 0x900 | 0x40; // UnitDelayKind Simple+Store; macro 2 store inherits load Mod0

inline void configure_mask1()
{
    ckernel::addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(M1_ADDR_MOD);

    TTI_SFPENCC(0, 0, 0, topk_negfilter::SFPENCC_MOD1_EI);
    ckernel::sfpu::_sfpu_load_imm32_(M1_L_THR, THR_BITS);

    TTI_SFPGT(0, M1_L_THR, 12, topk_negfilter::SFPGT_MOD1_SET_VD);

    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, M1_SEQUENCE & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (M1_SEQUENCE >> 16) & 0xFFFF);
    TTI_SFPCONFIG(0, 4 + 2, 0);

    TTI_SFPCONFIG(M1_MISC, 8, topk_negfilter::SFPCFG_IMM16_IS_VALUE);
    TTI_SFPNOP;
    TTI_SFPNOP;
}

#define MASK1_MACRO(vd) TTI_SFPLOADMACRO((2u << 2) | ((vd) & 3u), ckernel::InstrModLoadStore::INT32, M1_ADDR_MOD, ((vd) >> 2))

constexpr std::uint32_t M1_PASSES_PER_TILE = VECTORS_PER_TILE / 2;

#if XL_MERGE_EN
// 512 is the smallest legal K and the configuration whose two-tile Dest window
// and 32-vector body match the published 2.844 cyc/vector figure.
constexpr std::uint32_t XL_K = 512;
constexpr bool XL_FUSED      = true;
constexpr bool XL_APPROX     = false;
#endif

// One SFPU body per streamed tile, whichever body it is: every arm's body walks
// exactly 32 vectors, and one _topk_xl_merge_ invocation consumes 32 distinct
// vectors too, so "per streamed tile" is the same work unit for all of them.
inline void run_math_body([[maybe_unused]] std::uint32_t block_tile)
{
    if constexpr (SFPU_EN)
    {
        _llk_math_eltwise_sfpu_start_(block_tile);
        if constexpr (FILTER_ARM == ARM_NEGFILTER)
        {
            topk_negfilter::run_tile();
        }
        else if constexpr (FILTER_ARM == ARM_CTRLENV)
        {
            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;
        }
        else
        {
            ckernel::ckernel_unpack_template::run(M1_PASSES_PER_TILE);
            // Drain the scheduled Simple (t+1) and Store (t+2) of the final
            // macro before the Dst base moves.
            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;
        }
        _llk_math_eltwise_sfpu_done_();
    }
#if XL_MERGE_EN
    // Always dst_index 0: the merge works a fixed four-tile window and restores
    // the Dst write pointer itself. The operand is garbage under every run type
    // here, which is sound because the body is data-independent.
    _llk_math_eltwise_sfpu_start_(0);
    ckernel::sfpu::_topk_xl_merge_<XL_K, XL_APPROX, XL_FUSED>(0);
    _llk_math_eltwise_sfpu_done_();
#endif
}
} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t num_faces = params.num_faces;
    const std::uint32_t TILE_CNT  = params.TILE_CNT;
#endif
    const DataCopyType data_copy_type = DataCopyType::A2D;

    {
        START_PERF_MEASURE("INIT")

        _llk_math_eltwise_unary_datacopy_init_<data_copy_type, is_fp32_dest_acc_en>(num_faces, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

        if constexpr (SFPU_EN)
        {
            // Establishes the SFPU config register and clears LaneConfig, which
            // is the precondition for the VD >= 12 backdoor template writes.
            _llk_math_eltwise_unary_sfpu_init_once_();

            if constexpr (FILTER_ARM == ARM_NEGFILTER)
            {
                topk_negfilter::configure(THR_BITS);
                topk_negfilter::program_replay();
            }
            else if constexpr (FILTER_ARM == ARM_MASK1)
            {
                configure_mask1();
                ckernel::load_replay_buf<ckernel::NoExec>(
                    0,
                    2,
                    []
                    {
                        MASK1_MACRO(M1_L_A);
                        MASK1_MACRO(M1_L_B);
                    });
                ckernel::ckernel_unpack_template::lA(lltt::replay_insn(0, 2), TT_OP_NOP).program();
            }
            else if constexpr (FILTER_ARM == ARM_CTRLLOAD)
            {
                // CONTROL -- frontend floor. Deliberately a PLAIN load and not
                // SFPLOADMACRO: a macro issue would run against whatever
                // LoadMacroConfig the previous kernel left behind.
                ckernel::load_replay_buf<ckernel::NoExec>(
                    0,
                    2,
                    []
                    {
                        TTI_SFPLOAD(M1_L_A, ckernel::InstrModLoadStore::INT32, ckernel::ADDR_MOD_7, 0);
                        TTI_SFPLOAD(M1_L_B, ckernel::InstrModLoadStore::INT32, ckernel::ADDR_MOD_7, 2);
                    });
                ckernel::ckernel_unpack_template::lA(lltt::replay_insn(0, 2), TT_OP_NOP).program();
            }
            else if constexpr (FILTER_ARM == ARM_CTRLSWAP)
            {
                // CONTROL -- the tripwire. SFPSWAP is 2 backend cycles with a
                // hardware bubble that cannot be filled from this thread, so
                // this arm MUST come out at ~2.0x ctrlload.
                ckernel::load_replay_buf<ckernel::NoExec>(
                    0,
                    2,
                    []
                    {
                        TTI_SFPSWAP(0, M1_L_A, M1_L_B, ckernel::p_sfpswap::ALL_ROWS_MAX);
                        TTI_SFPSWAP(0, M1_L_B, M1_L_A, ckernel::p_sfpswap::ALL_ROWS_MAX);
                    });
                ckernel::ckernel_unpack_template::lA(lltt::replay_insn(0, 2), TT_OP_NOP).program();
            }
        }

#if XL_MERGE_EN
        // Programs ADDR_MOD_5/6 and the merge MOP template. _topk_xl_merge_
        // fires that template with ckernel_unpack_template::run(), so skipping
        // this would run whatever MOP the previous kernel left behind.
        _llk_math_eltwise_unary_sfpu_init_once_();
        ckernel::sfpu::_topk_xl_init_<XL_K, XL_FUSED>();
        _llk_math_eltwise_sfpu_start_(0);
#endif

        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Unpack does nothing in this run type, so math must NOT call the
            // datacopy: with unpack_to_dest that call is a handshake and would
            // block forever. The stimulus is whatever happens to be in Dest,
            // which is the right trade for an issue-rate number.
            if constexpr (SFPU_EN || XL_MERGE_EN)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        run_math_body(block_tile);
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
            {
                const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();
                }

                for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                {
                    // unpack_to_dest: the unpacker wrote Dest directly, so this
                    // call is pure synchronization and copies nothing.
                    _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                        block_tile, formats.math, formats.math);

                    if constexpr (PERF_RUN_TYPE != PerfRunType::UNPACK_ISOLATE)
                    {
                        run_math_body(block_tile);
                    }
                }

                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t num_faces = params.num_faces;
    const std::uint32_t TILE_CNT  = params.TILE_CNT;
    const int RELU_CONFIG         = params.RELU_CONFIG;
    const auto& buffer_Res        = params.buffer_Res;
#endif

    {
        START_PERF_MEASURE("INIT")

        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src,
            formats.pack_dst,
            FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES /* tile_size */,
            FACE_R_DIM,
            TILE_C_DIM,
            num_faces,
            false /* partial_face */,
            false /* narrow_tile */,
            RELU_CONFIG);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();

        {
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK | p_stall::THCON);
            if constexpr (ROW_START_SECTION_SIZE != 0)
            {
                cfg_reg_rmw_tensix<THCON_SEC0_REG1_Row_start_section_size_RMW>(ROW_START_SECTION_SIZE);
            }
            if constexpr (COMPRESS_EN)
            {
                cfg_reg_rmw_tensix<THCON_SEC0_REG1_Disable_zero_compress_RMW>(0);
            }
            // Written unconditionally, including the disabled (0) case.
            cfg_reg_rmw_tensix<THCON_SEC0_REG1_Downsample_mask_RMW>(DOWNSAMPLE_MASK);
        }

        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION || PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
            {
                const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_packer_wait_for_math_done_();
                }

                for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                {
                    _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, PackMode::Default>(
                        block_tile, L1_ADDRESS(buffer_Res[(block_start + block_tile) & (RES_SLOTS - 1)]));
                }

                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
            // The RISC runs far ahead of the packer; without this the zone-end
            // timestamp could land before the last PACRs have retired.
            TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK);
        }
        PROFILER_SYNC();
    }

    // Outside every timed zone: report how many bytes the last pack actually
    // emitted. A build whose compression config write silently failed measures
    // identically to the baseline and would read as "compression is free".
    {
        TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK);
        tensix_sync();
        volatile std::uint32_t* diag = reinterpret_cast<volatile std::uint32_t*>(buffer_Res[RES_SLOTS]);
        diag[0]                      = 0xC0DEBA5E;
        diag[1]                      = reg_read(RISCV_TDMA_REG_PACKED_SIZE + 0x080); // PackerTileSize(0, T2), 16 B units
        diag[2]                      = COMPRESS_EN ? 1u : 0u;
        diag[3]                      = FILTER_ARM;
    }
}

#endif // LLK_TRISC_PACK
