// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================
// End-to-end pipelined Top-K filter pass (Blackhole)
// ============================================================================
//
// WHAT THIS SETTLES
// -----------------
// Two numbers were measured in isolation earlier this session:
//
//   SFPU  (MATH_ISOLATE) MaskStore macro, Load+SFPGT+SFPSTORE   1.003 cyc/vector
//   PACK  (PACK_ISOLATE) 32-bit datums, zero-compression on     1.648 cyc/vector
//
// and the competition, _topk_xl_merge_ at K=512 fused, is 2.844 cyc/vector.
// The SFPU and the packer are separate backend ports off the same frontend mux
// (Diagrams/Src/TensixFrontend.lua:126-136), so the steady state of a kernel
// that runs both SHOULD be max(1.003, 1.648) and not the sum 2.651. An isolate
// measurement cannot tell those apart. This kernel runs the real thing -- a
// multi-tile stream, unpack -> (SFPU filter) -> compressed pack -- so the
// steady state is observed rather than argued.
//
// THE 2x2, AND WHY IT IS A 2x2
// ----------------------------
// SFPU_EN and COMPRESS_EN are independent compile-time flags, so the sweep is
// a full factorial and each effect is a difference of differences:
//
//   sfpu=0 compress=0   baseline pack of a stream
//   sfpu=0 compress=1   cost of compression alone
//   sfpu=1 compress=0   cost of the SFPU filter alone
//   sfpu=1 compress=1   both. If this equals max() the ports are concurrent;
//                       if it equals the sum they are not.
//
// The RELU arm is orthogonal and runtime-configured (RELU_CONFIG): with
// MIN_THRESHOLD_RELU the PACKER does the threshold compare itself, so the
// filter costs ZERO SFPU instructions. That is only expressible for a
// non-negative threshold (Packers/ReLU.md:41, signbit(Threshold) is
// UndefinedBehavior) -- but negative DATA is fine, it simply falls below the
// threshold. Verified this session: a dense tile of fused FP32 sort keys
// [bf16 value (high 16) | u16 index (low 16)] packs down from 4096 B to 640 B
// with relu+compression on, which is the same emitted size the SFPU-prezeroed
// Int32 arm produces. The fused word is a well-formed FP32 whose FP32 ordering
// is exactly "by value, ties broken by index", so the packer's float compare is
// the right compare.
//
// MEASUREMENT: L1_TO_L1, and why not an isolate
// ---------------------------------------------
// L1_TO_L1 timestamps unpack's ZONE_START against pack's ZONE_END
// (helpers/profiler.py::_stats_l1_to_l1), i.e. the whole three-thread pipeline
// end to end, which is the only run type that can show max-vs-sum. A two-point
// slope over TILE_CNT cancels the pipeline fill/drain and the marker pair and
// leaves the steady-state per-tile rate. L1_CONGESTION is reported alongside as
// a cross-check: it times each thread's own zone while all three run, so
// L1_CONGESTION[PACK] is the packer's view of the same steady state.
//
// Compile-time knobs (emitted into build.h by the python driver):
//   SFPU_EN                - run the macro filter on the math thread
//   COMPRESS_EN            - clear THCON_SEC0_REG1_Disable_zero_compress
//   ROW_START_SECTION_SIZE - THCON_SEC0_REG1_Row_start_section_size, 16 B units
//   DOWNSAMPLE_MASK        - ALWAYS written: set_packer_config never touches
//                            THCON_SEC0_REG1 word 3, so a mask left behind by an
//                            earlier kernel survives an ELF reload and silently
//                            decimates this pack
//   THR_BITS               - raw 32-bit threshold for the SFPU SFPGT
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

// 32 SFPLOADs cover one 32x32 tile: an SFPLOAD reads 4 consecutive Dst rows and
// the addr_mod dest field is in u10 Addr units where one SFPLOAD advances by 2,
// so 32 loads walk the tile's 64 Dst rows.
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

#if XL_MERGE_EN
// THE COMPETITION, measured in this kernel rather than quoted from another one.
// _topk_xl_merge_<512, false, true> is the shipping topk_xl merge body; it
// consumes 32 distinct 32-element vectors (= one 32x32 tile) per invocation and
// measures 2.844 cyc/vector under MATH_ISOLATE. Running it on the MATH thread of
// THIS pipeline, against the same streamed operand and the same packer, is the
// only way to compare it with the packer-resident filter on equal terms:
// isolate-vs-isolate cannot show whether the SFPU work hides under the
// unpacker or adds to it.
//
// Included only under this arm: the experimental SFPU trees are separate and
// pulling in more than one at a time is not supported.
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"
#endif

namespace
{
// LReg map for the filter macro. A/B ping-pong as the macro load target; the
// scheduled SFPGT overwrites the loaded register with its own -1/0 mask and the
// scheduled SFPSTORE writes that mask back to the same Dst address.
constexpr std::uint32_t L_A   = ckernel::p_sfpu::LREG0;
constexpr std::uint32_t L_B   = ckernel::p_sfpu::LREG1;
constexpr std::uint32_t L_THR = ckernel::p_sfpu::LREG3;

constexpr std::uint32_t SFPGT_MOD1_SET_VD     = 8; // SFPGT.md:53
constexpr std::uint32_t SFPENCC_MOD1_EI       = 2; // SFPENCC.md:41
constexpr std::uint32_t SFPCFG_IMM16_IS_VALUE = 1; // SFPCONFIG.md:108

// ADDR_MOD_6 is the slot the SFPU unary path itself uses for its Dst walk
// (llk_math_eltwise_unary_sfpu.h:52-57); ADDR_MOD_0/1/2 are taken by the A2D
// datacopy this kernel also runs.
constexpr std::uint32_t ADDR_MOD_WALK = ckernel::ADDR_MOD_6;

// Macro 2 -- Load + Simple(SFPGT, delay 0) + Store(SFPSTORE, delay 1).
// Byte layout per SFPLOADMACRO.md:
//   Simple 0x80 -> Insn.VB = macroVD, so the compare is (loaded > L_THR);
//                  0x40 clear -> Insn.VD = macroVD, so the mask lands in the
//                  loaded register, which is what the Store then reads.
//   Store  selector 3 = SFPSTORE, delay 1 -> fires one cycle after the compare.
constexpr std::uint32_t SEQ_SIMPLE = 0x80 | (0u << 3) | 4u;
constexpr std::uint32_t SEQ_STORE  = (1u << 3) | 3u;
constexpr std::uint32_t SEQUENCE_2 = (SEQ_STORE << 24) | SEQ_SIMPLE;

// Misc (SFPLOADMACRO.md:53-57). Bit 4+2 = 0x40 is UsesLoadMod0ForStore for macro
// 2, so the store inherits the load's INT32 mode -- the mask is a raw bit
// pattern and must not be format-converted. Bits 8 (Simple) and 11 (Store) put
// both on WaitForElapsedInstructions, which counts SFPU issues rather than wall
// cycles so a frontend bubble cannot slide a scheduled instruction off its slot.
constexpr std::uint32_t MISC_WORD_2 = 0x900 | 0x40;

inline void configure_filter_macro()
{
    // InstructionTemplate[0] via the VD>=12 backdoor: an instruction with
    // VD >= 12 is stored rather than executed while LaneConfig.DISABLE_BACKDOOR_LOAD
    // is false (SFPCONFIG.md:45-46). _llk_math_eltwise_unary_sfpu_init_once_()
    // clears LaneConfig, so this MUST run after it.
    TTI_SFPGT(0, L_THR, 12, SFPGT_MOD1_SET_VD);

    // Sequence[2] needs the Store byte in bits 24..31, which does not fit the
    // 16-bit immediate path -- stage the full word through LReg[0] and write
    // with Mod1 = 0, the idiom of _init_mul_int_.
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, SEQUENCE_2 & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (SEQUENCE_2 >> 16) & 0xFFFF);
    TTI_SFPCONFIG(0, 4 + 2, 0);

    TTI_SFPCONFIG(MISC_WORD_2, 8, SFPCFG_IMM16_IS_VALUE);
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// SFPLOADMACRO field packing (ckernel_ops.h:683, SFPLOADMACRO.md:20-26,45):
//   lreg_ind      = (MacroIndex << 2) | (VD & 3)
//   dest_reg_addr = (Imm9 << 1) | (VD >> 2)
#define FILTER_MACRO(vd, addr_mod, off) TTI_SFPLOADMACRO((2u << 2) | ((vd) & 3u), ckernel::InstrModLoadStore::INT32, (addr_mod), (off) | ((vd) >> 2))

// The recorded body covers two vectors (A/B ping-pong), so one tile is
// VECTORS_PER_TILE/2 replay passes. TT_OP_MOP's loop_count field is 7 bits, so
// a pass count above 128 would silently truncate; 16 is comfortably inside it.
constexpr std::uint32_t PASSES_PER_TILE = VECTORS_PER_TILE / 2;
static_assert(PASSES_PER_TILE <= 128, "MOP loop_count is 7 bits");

#if XL_MERGE_EN
// 512 is the smallest legal K (the kernel static_asserts K in {512, 1024, 2048})
// and the configuration whose two-tile Dest window and 32-vector body match the
// already-published 2.844 cyc/vector figure, so the numbers are comparable.
constexpr std::uint32_t XL_K = 512;
constexpr bool XL_FUSED      = true;
constexpr bool XL_APPROX     = false;
#endif

// One SFPU body per streamed tile, whichever body it is: the filter macro walks
// a tile's 32 vectors, and one _topk_xl_merge_ invocation consumes exactly 32
// distinct vectors too. So "per streamed tile" is the same work unit for both
// and the per-vector figures divide out identically.
inline void run_math_body([[maybe_unused]] std::uint32_t block_tile)
{
    if constexpr (SFPU_EN)
    {
        _llk_math_eltwise_sfpu_start_(block_tile);
        ckernel::ckernel_unpack_template::run(PASSES_PER_TILE);
        // Drain the scheduled Simple (t+1) and Store (t+2) of the final macro
        // before the Dst base moves.
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPNOP;
        _llk_math_eltwise_sfpu_done_();
    }
#if XL_MERGE_EN
    // Always dst_index 0: the merge works a fixed four-tile window and restores
    // the Dst write pointer itself. The operand is garbage under every run type
    // here, which is sound because the body is data-independent -- its loop
    // bounds come from (K, m_iter) and its compare network is a fixed SFPSWAP
    // lattice with no data-dependent timing.
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
            // is the precondition for the backdoor template write below.
            _llk_math_eltwise_unary_sfpu_init_once_();

            // Hardware Dst advance for the replayed body. Recorded instruction
            // words are immutable, so the walk cannot use sfpi::dst_reg++.
            // dest.incr = 2, not 4: the addr_mod dest field is in u10 Addr units
            // where bits [9:2] pick the 4-row group and bit 1 picks even-vs-odd
            // columns, so one SFPLOAD advances by 2.
            ckernel::addr_mod_t {
                .srca = {.incr = 0},
                .srcb = {.incr = 0},
                .dest = {.incr = 2},
            }
                .set(ADDR_MOD_WALK);

            // Clear stale lane predication: SFPGT's SET_VD write is gated on
            // LaneEnabled, so a mask left behind by an earlier kernel would
            // silently suppress the compare in some lanes.
            TTI_SFPENCC(0, 0, 0, SFPENCC_MOD1_EI);

            ckernel::sfpu::_sfpu_load_imm32_(L_THR, THR_BITS);
            configure_filter_macro();

            // Record once, replay per tile. One MOP issue then feeds the backend
            // at a guaranteed one instruction per cycle with the RISC-V off the
            // critical path.
            ckernel::load_replay_buf<ckernel::NoExec>(
                0,
                2,
                []
                {
                    FILTER_MACRO(L_A, ADDR_MOD_WALK, 0);
                    FILTER_MACRO(L_B, ADDR_MOD_WALK, 0);
                });
            ckernel::ckernel_unpack_template::lA(lltt::replay_insn(0, 2), TT_OP_NOP).program();
        }

#if XL_MERGE_EN
        // Programs ADDR_MOD_5/6 and the merge MOP template. _topk_xl_merge_ fires
        // that template with ckernel_unpack_template::run(), so skipping this
        // would run whatever MOP the previous kernel left behind -- undefined,
        // and a documented way to hang the math thread rather than fail loudly.
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
            // block forever waiting for an unpacker that never runs. The
            // stimulus is whatever happens to be in Dest, which is the right
            // trade for an issue-rate number.
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
            // UNPACK_ISOLATE, L1_TO_L1, L1_CONGESTION. In UNPACK_ISOLATE math
            // runs only the handshake the unpacker needs and must add no work of
            // its own; in the other two it also runs the filter.
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

        // set_packer_config forces config.f.uncompress = 1, so compression has to
        // be re-enabled behind the LLK's back. Row_start_section_size reserves
        // room for the row-start index array the compressed layout puts before
        // the data.
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
            // Written unconditionally, including the disabled (0) case -- see the
            // header comment on DOWNSAMPLE_MASK.
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
        diag[3]                      = SFPU_EN ? 1u : 0u;
    }
}

#endif // LLK_TRISC_PACK
