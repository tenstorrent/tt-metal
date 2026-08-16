// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================
// Threshold-count correctness kernel  (Blackhole only)
// ============================================================================
//
// WHAT THIS IS
// ------------
// The checkable twin of ARM_COUNT_D1 in sfpu_count_above_perf.cpp. It runs the
// SAME inner loop -- a replay-recorded body of (SFPLOADMACRO carrying a
// macro-scheduled SFPGT) + (software-issued SFPIADD), ping-ponged across two
// LRegs and driven by a MOP -- but bounded to whole Dest tiles, with the
// prologue/epilogue the perf arm deliberately omits, and it packs a result the
// host can check.
//
// It exists so the perf numbers come from a loop that is known to compute the
// right answer. In particular it is the only way to catch the hazard in
// SFPLOADMACRO.md:149: "If an instruction scheduled via SFPLOADMACRO arrives at
// a sub-unit on the same cycle as software issues a regular Vector Unit (SFPU)
// instruction to that sub-unit, then the scheduled instruction takes priority
// and the regular instruction is silently discarded." No fault is raised, so a
// dropped SFPIADD reads out only as a low count. An all-above stimulus checked
// against an exact count == N is the tripwire.
//
// WHY THE MASK IS ACCUMULATED WITH INTEGER ARITHMETIC
// ---------------------------------------------------
// SFPGT with Mod1 = SFPGT_MOD1_SET_VD writes `LReg[VD].i32 = IsVcSmaller ? -1
// : 0` (SFPGT.md:28-30), i.e. the bit patterns 0xFFFFFFFF / 0x00000000.
// 0xFFFFFFFF read as FP32 is a NaN, so a float MAD would accumulate NaN, not a
// count. SFPIADD is the only correct consumer.
//
// WHY THE ACCUMULATOR RUNS NEGATIVE AND IS NEGATED ONCE
// -----------------------------------------------------
// Each hit contributes -1, so the running total is -count. It is negated once,
// at the end, with SFPIADD in its 2's-complement-subtract mode against
// LCONST_0 (LReg[9], the hardwired zero), so the value that reaches SFPSTORE
// is non-negative and therefore has the SAME bit pattern under two's
// complement and under sign-magnitude. That matters because
// InstrModLoadStore::INT32_2S_COMP is a NO-OP on Blackhole
// (ckernel_sfpu_add_int.h:28-29) -- there is no store-time conversion to lean
// on, so the only safe contract is "never store a negative".
//
// WHY THE HOST SUMS THE WHOLE OUTPUT TILE
// ----------------------------------------
// One SFPSTORE writes 32 datums scattered over 4 Dst rows x 8 even-or-odd
// columns (SFPSTORE.md, cross-lane pattern). Modelling which packed tile
// element each SFPU lane lands on would duplicate that permutation in the
// golden for no benefit, so instead:
//   * the whole result tile is zeroed first (32 SFPSTOREs of a zeroed LReg,
//     which is exactly 32 x 32 = 1024 datums = the entire tile), then
//   * the 32 per-lane partials are dropped in with a single SFPSTORE, and
//   * the host sums the entire packed tile.
// The sum is permutation-invariant, so the check is exact without the host
// knowing the lane map. Zeroing is mandatory: without it the other 992
// elements are whatever the unpacked input left in Dest.
//
// THE Dst WALK
// ------------
// SFPLOAD/SFPSTORE addressing (SFPLOAD.md:86-107): the top 8 bits of the u10
// Addr select an aligned group of FOUR Dst rows, the next bit (bit 1) selects
// even or odd columns, and bit 0 is unused. A Dst row holds 16 32-bit datums,
// so a 32x32 tile in a 32-bit format is 64 rows, and one load covers
// 4 rows x 8 columns = 32 datums. That is 32 loads per tile, at Addr
// 0, 2, 4, ... 62 -- i.e. an advance of 2 per load, and a tile span of 64.
//
// The reference integer walk (_add_int_, ckernel_sfpu_add_int.h:41-58) gets
// that advance from `sfpi::dst_reg++`, which is RISC-V-side arithmetic folded
// into the emitted instruction word. Recorded instructions are immutable
// words, so a replayed body cannot walk Dst that way; the advance has to come
// from hardware, via an ADDR_MOD with dest.incr = 2 applied by the SFPLOAD
// itself. This matches the in-tree SFPU walk mod
// (llk_math_eltwise_unary_sfpu.h:52-57, `.dest = {.incr = 2}`).
//
// ADDR_MOD_6 is used for the walk. ADDR_MOD_0/2/3 are taken by the A2D
// datacopy this kernel also runs (llk_math_eltwise_unary_datacopy.h:344-409)
// and ADDR_MOD_7 is the invariant {0,0,0} established by the SFPU init.
//
// PIPELINE SHAPE (and why the body needs a prologue and an epilogue)
// -------------------------------------------------------------------
// Recorded body, one pass = two 32-datum vectors:
//
//   0 | SFPLOADMACRO -> L_A   (schedules SFPGT on L_A, delay 1)
//   1 | SFPIADD  L_ACC += L_B (mask produced one vector earlier)
//   2 | SFPLOADMACRO -> L_B   (schedules SFPGT on L_B, delay 1)
//   3 | SFPIADD  L_ACC += L_A
//
// Misc.UnitDelayKind selects WaitForElapsedInstructions for the Simple
// sub-unit, so a scheduled instruction's counter ticks on every SFPU
// instruction issued, not on every cycle (SFPLOADMACRO.md:149-151). With
// delay 1, the SFPGT scheduled at slot k executes on the cycle after slot k+1
// -- i.e. alongside slot k+2, which in this body is always a SFPLOADMACRO
// (Load sub-unit). The software SFPIADDs sit at the odd slots. Scheduled work
// therefore never shares a cycle with software work on the Simple sub-unit,
// which is precisely what keeps the silent-discard hazard away.
//
// The accumulate is one vector behind its compare, so:
//   * PROLOGUE: L_B is zeroed before each tile, otherwise the first SFPIADD
//     of the first pass folds in a stale LReg.
//   * EPILOGUE: the final mask has been produced but not yet accumulated, so
//     one extra SFPIADD (after three SFPNOPs, to let the last SFPGT land)
//     folds it in.
// The perf arm skips both and states the resulting one-vector bias; a
// correctness kernel cannot.
//
// NAMESPACES: this file is compiled at global scope. The `using namespace
// ckernel;` that lets LLK headers write these names bare arrives only with the
// LLK headers included inside each per-thread block below, so everything
// declared ABOVE them is explicitly ckernel::-qualified.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "llk_defs.h"
#include "lltt.h"
#include "params.h"
#include "sfpu/ckernel_sfpu_load_config.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

namespace
{
// LReg map. L_A / L_B ping-pong as the macro's load target; the scheduled
// SFPGT overwrites the loaded register with its own -1/0 mask, which the
// software SFPIADD then folds into L_ACC.
constexpr std::uint32_t L_A   = ckernel::p_sfpu::LREG0;
constexpr std::uint32_t L_B   = ckernel::p_sfpu::LREG1;
constexpr std::uint32_t L_ACC = ckernel::p_sfpu::LREG2;
constexpr std::uint32_t L_THR = ckernel::p_sfpu::LREG3;

// Threshold, as a raw 32-bit pattern. Bits rather than a float so that -0.0,
// the infinities and the NaNs are expressible exactly -- SFPGT orders by the
// sign-magnitude total order -NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN
// (SFPGT.md:3), which disagrees with IEEE on precisely those values, so they
// are the interesting thresholds.
//
// Delivered through the existing SFPU_UNARY_SCALAR template parameter
// (helpers/test_variant_parameters.py), documented there as "raw fp32 bits",
// which is exactly this shape. The perf twin takes the same quantity as
// `#define THR_BITS` because it guards the symbol with #ifndef and supplies a
// fallback; this kernel has no fallback to protect, so the ordinary constexpr
// parameter is the simpler wiring. It must be compile-time either way: the
// threshold is loaded into L_THR before the replay body is recorded.
constexpr std::uint32_t THRESHOLD_BITS = SFPU_UNARY_SCALAR;

// Instruction modifier bits, named from the ISA docs rather than spelled as
// bare integers at the call sites.
constexpr std::uint32_t SFPGT_MOD1_SET_VD              = 8; // SFPGT.md:53
constexpr std::uint32_t SFPIADD_MOD1_ARG_LREG_DST      = 0; // SFPIADD.md:48
constexpr std::uint32_t SFPIADD_MOD1_ARG_2SCOMP_LREG_D = 2; // SFPIADD.md:50
constexpr std::uint32_t SFPIADD_MOD1_CC_NONE           = 4; // SFPIADD.md:52
constexpr std::uint32_t SFPENCC_MOD1_EI                = 2; // SFPENCC.md:41
constexpr std::uint32_t SFPCFG_IMM16_IS_VALUE          = 1; // SFPCONFIG.md:108

// Dst geometry, in the u10 Addr units SFPLOAD/SFPSTORE take (SFPLOAD.md:86).
constexpr std::uint32_t DST_ADDR_PER_SFPLOAD = 2;                     // 4 rows, even or odd columns
constexpr std::uint32_t SFPLOADS_PER_TILE    = 32;                    // 32 datums each -> 1024 = one tile
constexpr std::uint32_t PASSES_PER_TILE      = SFPLOADS_PER_TILE / 2; // recorded body covers two loads
constexpr std::uint32_t REPLAY_BODY_LEN      = 4;                     // instructions in the recorded body
constexpr std::uint32_t ADDR_MOD_WALK        = ckernel::ADDR_MOD_6;

// Macro 0: Load + Simple(SFPGT). No MAD, no Round, no Store.
//
// simple_bits = 0x80 | (1 << 3) | 4
//   0x80        -> Insn.VB = macroVD, so the SFPGT computes (loaded > L_THR).
//                  MANDATORY: with the bit clear the macro puts macroVD in VC
//                  and leaves VB as the template's own VD, silently comparing
//                  against the wrong register (SFPLOADMACRO.md:112-119).
//   0x40 clear  -> Insn.VD = macroVD, so the mask lands back in the loaded
//                  register where the SFPIADD reads it.
//   delay 1     -> the field that separates scheduled work from software work;
//                  see the pipeline table in the file header.
//   selector 4  -> InstructionTemplate[0] (SFPLOADMACRO.md:83).
constexpr std::uint32_t SEQ_SIMPLE = 0x80u | (1u << 3) | 4u; // 0x8C
constexpr std::uint32_t SEQ_MAD    = 0;                      // nothing scheduled
constexpr std::uint32_t SEQUENCE_0 = (SEQ_MAD << 8) | SEQ_SIMPLE;

// Misc (SFPLOADMACRO.md:53-57): StoreMod0 [0:3], UsesLoadMod0ForStore [4:7],
// UnitDelayKind [8:11]. Bit 8 = Simple = WaitForElapsedInstructions.
//
// Instruction-counting rather than cycle-counting is required: the delay must
// track SFPU issues, not wall cycles, or a frontend bubble would slide the
// SFPGT out from under its load. Every in-tree user sets it --
// ckernel_sfpu_mul_int.h writes 0x330, ckernel_sfpu_where.h writes 0x770.
constexpr std::uint32_t MISC_WORD = 0x100;

// SFPLOADMACRO field packing (ckernel_ops.h, SFPLOADMACRO.md:20-26,45):
//   lreg_ind      = (MacroIndex << 2) | (VD & 3)
//   dest_reg_addr = (Imm9 << 1) | (VD >> 2)
// VD is constrained to 0..7 (bit 3 is hard-zeroed), so the split is exact.
// Mirrors the ckernel_sfpu_mul_int.h idiom.
#define LOADMACRO(vd, addr_mod, off) TTI_SFPLOADMACRO((0u << 2) | ((vd) & 3u), ckernel::InstrModLoadStore::INT32, (addr_mod), (off) | ((vd) >> 2))

} // namespace

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
    _llk_unpack_A_init_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"

using namespace ckernel;

// Write InstructionTemplate[0] and Sequence[0], then the Misc word.
//
// The template is written through the VD >= 12 backdoor: an instruction with
// VD >= 12 is stored rather than executed, provided
// LaneConfig.DISABLE_BACKDOOR_LOAD is false (SFPCONFIG.md:45-46, :120). The
// SFPU init clears LaneConfig, which is what establishes that -- so this must
// run after it, never before.
static inline void configure_macro0()
{
    TTI_SFPGT(0, L_THR, 12, SFPGT_MOD1_SET_VD);

    TTI_SFPCONFIG(SEQUENCE_0, 4 + 0, SFPCFG_IMM16_IS_VALUE); // Sequence[0]
    TTI_SFPCONFIG(MISC_WORD, 8, SFPCFG_IMM16_IS_VALUE);      // Misc
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// Count one whole Dest tile into L_ACC. Assumes the caller has already pointed
// DEST_TARGET_REG_CFG_MATH_Offset at the tile and zeroed the Dst RWC.
static inline void count_one_tile()
{
    // PROLOGUE: the first SFPIADD of the first pass reads L_B one vector
    // before anything has written it.
    ckernel::sfpu::_sfpu_load_imm32_(L_B, 0);

    ckernel_unpack_template::run(PASSES_PER_TILE);

    // EPILOGUE: the last mask has been produced but not accumulated. Three
    // SFPNOPs let the final scheduled SFPGT land (and, being NOPs, are
    // harmless even if one of them is the instruction that gets discarded),
    // then one SFPIADD folds the mask in.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPIADD(0, L_B, L_ACC, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    // Establishes the SFPU config register and the invariant ADDR_MOD_7 =
    // {srca:0, srcb:0, dest:0}. Also clears LaneConfig, which is the
    // precondition for the backdoor template write below. Do not reorder.
    _llk_math_eltwise_unary_sfpu_init_once_();

    // Hardware Dst advance for the replayed body -- see the file header for
    // why a recorded body cannot use sfpi::dst_reg++.
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = DST_ADDR_PER_SFPLOAD},
    }
        .set(ADDR_MOD_WALK);

    // Clear stale lane predication. SFPGT's SET_VD write is gated on
    // LaneEnabled (SFPGT.md:27-33), so a mask left behind by a previous kernel
    // would silently suppress the compare in some lanes -- an undercount that
    // looks like a correct-but-slow result. Nothing after this point touches
    // LaneFlags: the scheduled SFPGT sets only SET_VD, and every SFPIADD uses
    // CC_NONE, so all 32 lanes stay enabled through to the final SFPSTOREs.
    TTI_SFPENCC(0, 0, 0, SFPENCC_MOD1_EI);

    ckernel::sfpu::_sfpu_load_imm32_(L_THR, THRESHOLD_BITS);
    ckernel::sfpu::_sfpu_load_imm32_(L_ACC, 0x00000000);

    configure_macro0();

    // The body the perf arm measures, recorded once.
    load_replay_buf<NoExec>(
        0,
        REPLAY_BODY_LEN,
        []
        {
            LOADMACRO(L_A, ADDR_MOD_WALK, 0);
            TTI_SFPIADD(0, L_B, L_ACC, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
            LOADMACRO(L_B, ADDR_MOD_WALK, 0);
            TTI_SFPIADD(0, L_A, L_ACC, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
        });
    ckernel_unpack_template::lA(lltt::replay_insn(0, REPLAY_BODY_LEN), TT_OP_NOP).program();

    const std::uint32_t num_tiles = params.NUM_TILES_IN_BLOCK;
    // Re-walk the resident tiles LOOP_FACTOR times. The per-lane count is at
    // most 32 per tile pass, so this is the only way to drive the accumulator
    // past 2^16 and expose a 16-bit truncation anywhere in the accumulate,
    // store or pack path.
    const std::uint32_t repeat = params.LOOP_FACTOR;

    LLK_ASSERT(
        (num_tiles <= get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "NUM_TILES_IN_BLOCK exceeds max dest tiles");

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

    for (std::uint32_t tile = 0; tile < num_tiles; ++tile)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            tile, formats.math, formats.math);
    }

    for (std::uint32_t r = 0; r < repeat; ++r)
    {
        for (std::uint32_t tile = 0; tile < num_tiles; ++tile)
        {
            // Points DEST_TARGET_REG_CFG_MATH_Offset at tile * 64 (plus the
            // SyncHalf bank base); the RWC reset puts the walk back at the
            // start of that tile.
            _llk_math_eltwise_sfpu_start_(tile);
            math::clear_dst_reg_addr();
            count_one_tile();
        }
    }

    // L_ACC holds -(count). Negate once, here, so the stored word is
    // non-negative and therefore bit-identical under two's complement and
    // sign-magnitude. LCONST_0 is the hardwired zero (LReg[9]).
    TTI_SFPIADD(0, p_sfpu::LCONST_0, L_ACC, SFPIADD_MOD1_ARG_2SCOMP_LREG_D | SFPIADD_MOD1_CC_NONE);

    // Result goes over Dest tile 0. The inputs have already been consumed, so
    // overwriting them costs nothing and keeps num_tiles free to use the whole
    // Dest section.
    _llk_math_eltwise_sfpu_start_(0);
    math::clear_dst_reg_addr();

    // Zero the entire tile first: the single SFPSTORE below writes only 32 of
    // its 1024 elements, and the host sums all of them.
    ckernel::sfpu::_sfpu_load_imm32_(L_B, 0x00000000);
    for (std::uint32_t i = 0; i < SFPLOADS_PER_TILE; ++i)
    {
        TT_SFPSTORE(L_B, InstrModLoadStore::INT32, ADDR_MOD_7, i * DST_ADDR_PER_SFPLOAD);
    }
    TT_SFPSTORE(L_ACC, InstrModLoadStore::INT32, ADDR_MOD_7, 0);

    _llk_math_eltwise_sfpu_done_();
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();

    // Only Dest tile 0 carries the answer; the other tiles are spent input.
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
