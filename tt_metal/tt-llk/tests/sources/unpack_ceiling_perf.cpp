// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================
// Is 3.938 cyc/vector the Blackhole unpacker's ceiling, or the LLK's handshake?
// ============================================================================
//
// THE CLAIM UNDER ATTACK
// ----------------------
// perf_topk_pipeline.py measures UNPACK_ISOLATE at 126.0 cycles per 32x32 FP32
// tile (3.9375 cyc / 32-element vector) for `_llk_unpack_A_<..., unpack_to_dest>`,
// and it has been treated as a hardware floor. 4096 B / 126 cyc = 32.5 B/cycle.
// The documented single-unpacker L1 read rate (WormholeB0/TensixTile/L1.md:37,
// "four 128-bit reads from L1 per cycle") is 64 B/cycle, and Blackhole is
// documented only as having "more L1 bandwidth" (BlackholeA0/TensixTile/README.md:19)
// with 32 L1 banks instead of 16 (BlackholeA0/.../MemoryOrdering.md:36). So the
// measured rate is at best half, and plausibly a quarter, of the ceiling.
//
// Reading the LLK shows where the rest goes. Per tile, `_llk_unpack_A_` with
// unpack_to_dest brackets a 4-UNPACR MOP with:
//
//   unpack: SETADCZW, wait_for_next_context, cfg[base_addr]=, SEMPOST,
//           set_dst_write_addr  -> mailbox_read(MathThreadId)   (BLOCKING)
//                                  SETC16, SETDMAREG, WRCFG,
//                                  2x cfg_reg_rmw_tensix
//           wait_for_dest_available -> SEMWAIT(UNPACK_TO_DEST, ON_MAX)
//           STALLWAIT(STALL_UNPACK, TRISC_CFG)   <- drains the unpacker
//           [MOP: 4x UNPACR]
//           SEMGET, unpack_to_dest_tile_done -> SEMPOST, SETDMAREG, WRCFG,
//                                               2x cfg_reg_rmw_tensix, SETC16
//   math:   math_unpack_to_dest_math_ready -> STALLWAIT(STALL_SYNC, MATH|WAIT_SFPU)
//                                             SEMPOST, RISC-V spin on
//                                             semaphore_read(MATH_DONE), SEMGET
//           set_dst_write_addr -> mailbox_write(UnpackThreadId, dst_index)
//           math_unpack_to_dest_tile_ready -> SEMWAIT(UNPACK_TO_DEST, ON_ZERO), SEMGET
//           4x ZEROACC
//
// UNPACK_TO_DEST is initialised max=1 (blackhole/c_tensix_core.h:419) and the
// mailbox is a single slot, so the in-flight depth is exactly one tile. Worse,
// `math_unpack_to_dest_math_ready` stalls on MATH|WAIT_SFPU -- a FULL drain of
// every in-flight FPU and SFPU instruction -- before every single tile. That is
// a sufficient explanation both for the 32.5 B/cycle and for the observation
// that adding SFPU work moves UNPACK_ISOLATE from 126 to 176 cycles/tile: the
// SFPU time cannot hide under the unpacker because the unpacker is parked in
// mailbox_read waiting for a math thread that is required to be idle first.
//
// THE ARMS
// --------
// Raw* arms hoist the entire bracket out of the loop. The MOP body is unchanged
// (`unpack_srca_to_dest`, AddrMode 0b00010001 so ch0_z walks L1 and ch1_z walks
// Dest); only the per-tile handshake is removed. Every 8 tiles one SETADCZW
// rewinds both counters, which keeps the L1 reads inside the 16-tile stimulus
// ring and the Dest writes inside the 8 FP32 tiles Dest holds.
//
//   LlkDest         reproduce 3.938 in THIS kernel                UNPACK_ISOLATE
//   LlkDestSfpu     reproduce 5.438 in THIS kernel                UNPACK_ISOLATE
//   RawDest         handshake removed -> unpacker->Dest ceiling   UNPACK_ISOLATE
//   RawDestSfpu     RawDest + SFPU macro on a DIFFERENT Dest tile UNPACK_ISOLATE (unpack view)
//                                                                 MATH_ISOLATE   (math view)
//   RawDestSfpuSame RawDest + SFPU macro on the SAME Dest tiles   both
//   SfpuOnly        SFPU macro alone                              MATH_ISOLATE
//   CtrlLoad        SFPLOAD issue-rate tripwire, must be 1.000    MATH_ISOLATE
//   CtrlSwap        SFPSWAP issue-rate tripwire, must be 2.000    MATH_ISOLATE
//   LlkSrcA         `_llk_unpack_A_` to SrcA, per-tile LLK path   UNPACK_ISOLATE
//   RawSrcA         MOP-only stream to SrcA, no DVALID            UNPACK_ISOLATE
//
// RawDestSfpu deliberately reports under BOTH isolate run types: in this kernel
// the "isolate" run types select which thread's zone the report exposes, not
// which threads run, so the same binary answers "does the SFPU slow the
// unpacker" and "does the unpacker slow the SFPU". Neither thread synchronises
// with the other in these arms, so any slowdown is genuine port contention.
//
// RawDestSfpuSame vs RawDestSfpu is the structural question: if concurrency is
// recovered only when the SFPU works a different Dest tile, the limit is a
// same-region dependency; if both serialise, the Dest register file has one
// arbitrated port and the answer is no.
//
// READING THE NUMBER
// ------------------
// Two-point slope over TILE_CNT, exactly as perf_topk_pipeline.py:
//   cyc_per_vector = (mean@hi - mean@lo) / (hi - lo) / 32   [report is per-tile]
// and B/cycle = tile_bytes / (32 * cyc_per_vector).

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

// Arm ids. MUST match UnpArm in perf_unpack_ceiling.py.
#define ARM_LLK_DEST           0
#define ARM_LLK_DEST_SFPU      1
#define ARM_RAW_DEST           2
#define ARM_RAW_DEST_SFPU      3
#define ARM_RAW_DEST_SFPU_SAME 4
#define ARM_SFPU_ONLY          5
#define ARM_CTRL_LOAD          6
#define ARM_CTRL_SWAP          7
#define ARM_LLK_SRCA           8
#define ARM_RAW_SRCA           9

#ifndef UNP_ARM
#define UNP_ARM ARM_LLK_DEST
#endif

// Arms whose unpack thread streams to Dest without the per-tile LLK bracket.
#define RAW_DEST_ARM (UNP_ARM == ARM_RAW_DEST || UNP_ARM == ARM_RAW_DEST_SFPU || UNP_ARM == ARM_RAW_DEST_SFPU_SAME)
// Arms whose unpack thread uses the stock LLK unpack-to-dest path.
#define LLK_DEST_ARM (UNP_ARM == ARM_LLK_DEST || UNP_ARM == ARM_LLK_DEST_SFPU)
// Arms whose math thread runs the SFPLOADMACRO filter body.
#define SFPU_ARM (UNP_ARM == ARM_LLK_DEST_SFPU || UNP_ARM == ARM_RAW_DEST_SFPU || UNP_ARM == ARM_RAW_DEST_SFPU_SAME || UNP_ARM == ARM_SFPU_ONLY)
// Arms whose math thread runs one of the two issue-rate tripwires.
#define CTRL_ARM (UNP_ARM == ARM_CTRL_LOAD || UNP_ARM == ARM_CTRL_SWAP)

static constexpr ckernel::DstSync DST_SYNC_MODE        = ckernel::DstSync::SyncHalf;
static constexpr ckernel::BroadcastType BROADCAST_TYPE = ckernel::BroadcastType::NONE;
static constexpr std::uint32_t MAX_TILES_DEST          = is_fp32_dest_acc_en ? 4 : 8;

// 32 SFPLOADs cover one 32x32 tile: an SFPLOAD reads 4 consecutive Dst rows and
// the addr_mod dest field is in u10 Addr units where one SFPLOAD advances by 2.
static constexpr std::uint32_t VECTORS_PER_TILE = 32;

// Dest holds 8 FP32 tiles (16 FP16). Rewinding both ADC Z counters every
// DEST_WRAP_TILES tiles keeps the raw stream's Dest writes inside Dest and its
// L1 reads inside the stimulus ring, for one instruction per wrap.
//
// The wrap is MAX_TILES_DEST, not the full Dest depth, on purpose: it confines
// the raw stream to Dest tiles 0..3 so that SFPU_DEST_TILE = 4 addresses a
// genuinely DISJOINT set of Dest rows. If the stream covered all 8 tiles there
// would be no "different region" arm to run.
static constexpr std::uint32_t DEST_WRAP_TILES = MAX_TILES_DEST;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

namespace
{
#if RAW_DEST_ARM || UNP_ARM == ARM_RAW_SRCA

// Same MOP body the LLK programs for unpack-to-dest (llk_unpack_A.h:70-71):
// SetDatValid = 0, so no SrcA bank handshake; AddrMode 0b00010001 increments
// ch0_z (L1 source) and ch1_z (Dest destination) by one face each.
constexpr std::uint32_t UNPACK_SRCA_TO_DEST =
    TT_OP_UNPACR(ckernel::SrcA, 0b00010001, 0, 0, 0, 1 /* OvrdThreadId */, 0 /* SetDatValid */, ckernel::p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

// SrcA variant: ch0_z only, and still no DVALID, so the math thread has nothing
// to clear and the write rate is measured with no handshake in the path.
constexpr std::uint32_t UNPACK_SRCA_NO_DVALID =
    TT_OP_UNPACR(ckernel::SrcA, 0b1, 0, 0, 0, 1 /* OvrdThreadId */, 0 /* SetDatValid */, ckernel::p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

// Rewind ch0_y/z and ch1_y/z on both unpackers -- the same instruction
// `_llk_unpack_A_` issues at the top of every call (llk_unpack_A.h:419).
constexpr std::uint32_t REWIND_ADC = TT_OP_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

// Hoisted, non-blocking twin of cunpack_common.h:1000 set_dst_write_addr. The
// only functional difference is that the Dest tile index is a parameter instead
// of a blocking mailbox_read(MathThreadId) -- which is precisely the dependency
// under test.
inline void raw_arm_point_unpacker_at_dest(const std::uint32_t dst_tile_index, const std::uint32_t unpack_dst_format)
{
    using namespace ckernel;
    // 4*16 is the fixed Dest offset the LLK applies; << 6 is DstTileSizeLog2[Tile32x32].
    const std::uint32_t dst_byte_addr = 16 * (4 + (dst_tile_index << 6));

    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x0); // disable address bit swizzle
    TT_SETDMAREG(0, LOWER_HALFWORD(canonical_unpA_z_stride(unpack_dst_format) << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT), 0, LO_16(p_gpr_unpack::TMP_LO));
    TTI_WRCFG(p_gpr_unpack::TMP_LO, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(1);
    cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(dst_byte_addr);
}
#endif
} // namespace

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
    constexpr bool UNPACK_TO_DEST_ARM                = (RAW_DEST_ARM || LLK_DEST_ARM);

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
        _llk_unpack_A_init_<BROADCAST_TYPE, false /* acc_to_dest */, reuse_dest_type, UNPACK_TO_DEST_ARM>(
            0 /* transpose_of_faces */,
            0 /* within_face_16x16_transpose */,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
            formats.unpack_A_src,
            formats.unpack_A_dst);

#if RAW_DEST_ARM
        // Everything the per-tile bracket would have done, done once. The L1
        // source base is set here and never rewritten: ch0_z walks the stimulus
        // ring by itself and REWIND_ADC returns it to slot 0.
        {
            volatile std::uint32_t tt_reg_ptr* cfg   = ckernel::get_cfg_pointer();
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = L1_ADDRESS(buffer_A[0]);
        }
        raw_arm_point_unpacker_at_dest(0, formats.unpack_A_dst);
        // UNPACR_Regular.md:640-652 -- Throttle_mode picks the unpacker's L1
        // fetch rate: 0 = x1 (16 B/cyc), 1 = x2 (32), 2 = x4 (64). Blackhole
        // additionally decodes 3 = x8 (128) and upgrades x4 to 128 for >=2-byte
        // datums, but BOTH of those live inside the `!UnpackToDst` SrcA-burst
        // branch (UNPACR_Regular.md:315-328), so on this path 2 should already
        // be the ceiling. Sweeping it is how we find out rather than assume.
        // cunpack_common.h:892 programs 2 for every LLK unpack.
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Throttle_mode_RMW>(THROTTLE_MODE);
        TTI_STALLWAIT(ckernel::p_stall::STALL_UNPACK, ckernel::p_stall::TRISC_CFG);
#elif UNP_ARM == ARM_RAW_SRCA
        {
            volatile std::uint32_t tt_reg_ptr* cfg   = ckernel::get_cfg_pointer();
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = L1_ADDRESS(buffer_A[0]);
        }
        // Reprogram the MOP to the no-DVALID SrcA body: the stock NONE-broadcast
        // MOP publishes DVALID, which would stall the unpacker on a math thread
        // that does nothing in this arm.
        {
            ckernel::ckernel_template tmp(num_faces, 1, UNPACK_SRCA_NO_DVALID);
            tmp.program();
        }
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Throttle_mode_RMW>(THROTTLE_MODE);
        TTI_STALLWAIT(ckernel::p_stall::STALL_UNPACK, ckernel::p_stall::TRISC_CFG);
#endif
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        // PACK_ISOLATE is never requested by this test's driver, but the guard
        // keeps the kernel honest if it ever is.
        if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
#if LLK_DEST_ARM || UNP_ARM == ARM_LLK_SRCA
            // Stock LLK path. Under MATH_ISOLATE the math thread does not run
            // the datacopy handshake, so the unpacker must not either -- it
            // would block forever in mailbox_read.
            if constexpr (PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_unpack_A_<BROADCAST_TYPE, false, reuse_dest_type, UNPACK_TO_DEST_ARM>(
                        L1_ADDRESS(buffer_A[i & (SRC_SLOTS - 1)]), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
#elif RAW_DEST_ARM || UNP_ARM == ARM_RAW_SRCA
            // One MOP issue per tile expands to num_faces UNPACRs in hardware,
            // so the RISC-V is never the limiter.
            for (std::uint32_t i = 0; i < TILE_CNT; ++i)
            {
                if ((i & (DEST_WRAP_TILES - 1)) == 0)
                {
                    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
                }
                ckernel::ckernel_template::run();
            }
            TTI_STALLWAIT(ckernel::p_stall::STALL_SYNC, ckernel::p_stall::UNPACK0);
#else
            // SfpuOnly / CtrlLoad / CtrlSwap: unpack contributes nothing.
            (void)0;
#endif
        }
        PROFILER_SYNC();
    }
    // Silence unused warnings in arms that read neither.
    (void)TILE_CNT;
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu/ckernel_sfpu_load_config.h"

namespace
{
// LReg map for the filter macro, identical to sources/topk_pipeline_perf.cpp.
constexpr std::uint32_t L_A   = ckernel::p_sfpu::LREG0;
constexpr std::uint32_t L_B   = ckernel::p_sfpu::LREG1;
constexpr std::uint32_t L_THR = ckernel::p_sfpu::LREG3;

constexpr std::uint32_t SFPGT_MOD1_SET_VD     = 8; // SFPGT.md:53
constexpr std::uint32_t SFPENCC_MOD1_EI       = 2; // SFPENCC.md:41
constexpr std::uint32_t SFPCFG_IMM16_IS_VALUE = 1; // SFPCONFIG.md:108

constexpr std::uint32_t ADDR_MOD_WALK = ckernel::ADDR_MOD_6;

constexpr std::uint32_t SEQ_SIMPLE  = 0x80 | (0u << 3) | 4u;
constexpr std::uint32_t SEQ_STORE   = (1u << 3) | 3u;
constexpr std::uint32_t SEQUENCE_2  = (SEQ_STORE << 24) | SEQ_SIMPLE;
constexpr std::uint32_t MISC_WORD_2 = 0x900 | 0x40;

inline void configure_filter_macro()
{
    TTI_SFPGT(0, L_THR, 12, SFPGT_MOD1_SET_VD);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, SEQUENCE_2 & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (SEQUENCE_2 >> 16) & 0xFFFF);
    TTI_SFPCONFIG(0, 4 + 2, 0);
    TTI_SFPCONFIG(MISC_WORD_2, 8, SFPCFG_IMM16_IS_VALUE);
    TTI_SFPNOP;
    TTI_SFPNOP;
}

#define FILTER_MACRO(vd, addr_mod, off) TTI_SFPLOADMACRO((2u << 2) | ((vd) & 3u), ckernel::InstrModLoadStore::INT32, (addr_mod), (off) | ((vd) >> 2))

constexpr std::uint32_t PASSES_PER_TILE = VECTORS_PER_TILE / 2;
static_assert(PASSES_PER_TILE <= 128, "MOP loop_count is 7 bits");
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
    const DataCopyType data_copy_type                  = DataCopyType::A2D;
    [[maybe_unused]] constexpr bool UNPACK_TO_DEST_ARM = (RAW_DEST_ARM || LLK_DEST_ARM);

    {
        START_PERF_MEASURE("INIT")

        _llk_math_eltwise_unary_datacopy_init_<data_copy_type, is_fp32_dest_acc_en>(num_faces, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

#if SFPU_ARM || CTRL_ARM
        _llk_math_eltwise_unary_sfpu_init_once_();

        ckernel::addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ADDR_MOD_WALK);

        TTI_SFPENCC(0, 0, 0, SFPENCC_MOD1_EI);
        ckernel::sfpu::_sfpu_load_imm32_(L_THR, THR_BITS);
        configure_filter_macro();
#endif

#if SFPU_ARM
        ckernel::load_replay_buf<ckernel::NoExec>(
            0,
            2,
            []
            {
                FILTER_MACRO(L_A, ADDR_MOD_WALK, 0);
                FILTER_MACRO(L_B, ADDR_MOD_WALK, 0);
            });
        ckernel::ckernel_unpack_template::lA(lltt::replay_insn(0, 2), TT_OP_NOP).program();
#elif UNP_ARM == ARM_CTRL_LOAD
        // Pure SFPLOAD issue rate: two loads per replay pass, one per vector.
        ckernel::load_replay_buf<ckernel::NoExec>(
            0,
            2,
            []
            {
                TTI_SFPLOAD(L_A, ckernel::InstrModLoadStore::INT32, ADDR_MOD_WALK, 0);
                TTI_SFPLOAD(L_B, ckernel::InstrModLoadStore::INT32, ADDR_MOD_WALK, 0);
            });
        ckernel::ckernel_unpack_template::lA(lltt::replay_insn(0, 2), TT_OP_NOP).program();
#elif UNP_ARM == ARM_CTRL_SWAP
        // Pure SFPSWAP issue rate: SFPSWAP.md:110 gives 2 cycles with a
        // non-fillable bubble, so this arm MUST land at exactly 2.00x CtrlLoad.
        // Operands alternate exactly as in sources/topk_merge_macro_perf.cpp so
        // the two harnesses' controls are the same instruction stream.
        ckernel::load_replay_buf<ckernel::NoExec>(
            0,
            2,
            []
            {
                TTI_SFPSWAP(0, L_A, L_B, ckernel::p_sfpswap::ALL_ROWS_MAX);
                TTI_SFPSWAP(0, L_B, L_A, ckernel::p_sfpswap::ALL_ROWS_MAX);
            });
        ckernel::ckernel_unpack_template::lA(lltt::replay_insn(0, 2), TT_OP_NOP).program();
#endif
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
#if LLK_DEST_ARM || UNP_ARM == ARM_LLK_SRCA
            // Stock LLK path: the datacopy call IS the handshake. Skipped under
            // MATH_ISOLATE, where the unpack thread is idle and the handshake
            // would never complete.
            if constexpr (PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, UNPACK_TO_DEST_ARM>(
                            block_tile, formats.math, formats.math);
#if SFPU_ARM
                        _llk_math_eltwise_sfpu_start_(block_tile);
                        ckernel::ckernel_unpack_template::run(PASSES_PER_TILE);
                        TTI_SFPNOP;
                        TTI_SFPNOP;
                        TTI_SFPNOP;
                        _llk_math_eltwise_sfpu_done_();
#endif
                    }
                }
            }
#elif SFPU_ARM
            // Free-running: no handshake with the unpack thread in any arm here,
            // so a slowdown relative to the solo arm is genuine contention.
            //
            // SFPU_DEST_TILE selects whether the macro walks the Dest tiles the
            // raw unpack stream is writing (0) or a disjoint set (4).
            for (std::uint32_t i = 0; i < TILE_CNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(SFPU_DEST_TILE + (i & (MAX_TILES_DEST - 1)));
                ckernel::ckernel_unpack_template::run(PASSES_PER_TILE);
                TTI_SFPNOP;
                TTI_SFPNOP;
                TTI_SFPNOP;
                _llk_math_eltwise_sfpu_done_();
            }
#elif CTRL_ARM
            // The tripwires must land at exactly 1.000 and 2.000 cyc/vector, so
            // there is no per-tile envelope at all: no _llk_math_eltwise_sfpu_start_
            // / _done_ pair, matching sources/topk_merge_macro_perf.cpp's control
            // arms. The SFPU arms above deliberately keep the envelope per-tile,
            // because there it is part of the cost being compared.
            for (std::uint32_t i = 0; i < TILE_CNT; ++i)
            {
                ckernel::ckernel_unpack_template::run(PASSES_PER_TILE);
            }
#else
            (void)0;
#endif
        }
        PROFILER_SYNC();
    }
    (void)TILE_CNT;
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

using namespace ckernel;

// With PACK_EN == 0 the packer does no work: the test isolates the unpacker and
// the SFPU. It still declares both zones, because the profiler's per-thread zone
// tables are indexed by zone id and a missing zone shifts every id after it.
//
// With PACK_EN == 1 the packer runs the shipping Top-K filter -- MIN_THRESHOLD_RELU
// plus zero-compression, the `relucomp` arm of perf_topk_pipeline.py -- FREE
// RUNNING, with no _llk_packer_wait_for_math_done_. That is deliberate and it is
// the same thing PACK_ISOLATE already does in that kernel: the packer's rate was
// measured flat in survivor density, so a handshake-free packer over garbage Dest
// contents has the right steady-state rate. PACK_DEST_TILE decides whether it
// reads the Dest tiles the raw unpack stream is writing (0) or a disjoint set (4),
// which is the packer-side twin of the SFPU_DEST_TILE question.
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

#if PACK_EN
        // set_packer_config forces config.f.uncompress = 1, so compression has to
        // be re-enabled behind the LLK's back. Downsample_mask is written
        // unconditionally: set_packer_config never touches THCON_SEC0_REG1 word 3,
        // so a mask left behind by an earlier kernel survives an ELF reload and
        // would silently decimate this pack.
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
            cfg_reg_rmw_tensix<THCON_SEC0_REG1_Downsample_mask_RMW>(0);
        }
#endif
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
#if PACK_EN
        if constexpr (PERF_RUN_TYPE != PerfRunType::UNPACK_ISOLATE && PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
            {
                const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                {
                    _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, PackMode::Default>(
                        PACK_DEST_TILE + block_tile, L1_ADDRESS(buffer_Res[(block_start + block_tile) & (RES_SLOTS - 1)]));
                }
            }
            // The RISC runs far ahead of the packer; without this the zone-end
            // timestamp could land before the last PACRs have retired.
            TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK);
        }
#endif
        PROFILER_SYNC();
    }

#if PACK_EN
    // Outside every timed zone: report how many bytes the last pack emitted. A
    // build whose compression config write silently failed measures identically
    // to the baseline and would read as "compression is free".
    {
        TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK);
        tensix_sync();
        volatile std::uint32_t* diag = reinterpret_cast<volatile std::uint32_t*>(buffer_Res[RES_SLOTS]);
        diag[0]                      = 0xC0DEBA5E;
        diag[1]                      = reg_read(RISCV_TDMA_REG_PACKED_SIZE + 0x080); // PackerTileSize(0, T2), 16 B units
        diag[2]                      = COMPRESS_EN ? 1u : 0u;
        diag[3]                      = UNP_ARM;
    }
#endif
    (void)TILE_CNT;
}

#endif // LLK_TRISC_PACK
