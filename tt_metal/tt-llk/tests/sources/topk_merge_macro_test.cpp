// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// CORRECTNESS HARNESS FOR THE SFPLOADMACRO MERGE.
//
// A verbatim clone of `sources/topk_xl_test.cpp` with exactly one thing changed:
// when MERGE_USE_MACRO is 1, the FUSED merge step dispatches to
// `macro_merge_<K>` (defined below) instead of
// `ckernel::sfpu::_topk_xl_merge_<K, APPROX, true>`. Everything else -- the
// unpack path, copy_sort, add_lsb_indices, local_sort, rebuild, the index split,
// the pack path, and the torch golden in test_topk_merge_macro.py -- is
// untouched, so a PASS is end-to-end evidence that the macro-scheduled
// compare-exchange is functionally identical to the shipping SFPSWAP body
// inside the real K=512/1024/2048 pipeline.
//
// WHY THIS TEST IS NOT OPTIONAL. The perf arm alone cannot prove the macro
// works. A misconfigured (or all-zero, i.e. "schedule nothing")
// LoadMacroConfig.Sequence degenerates an SFPLOADMACRO into a plain SFPLOAD --
// which measures the SAME 1.000 cyc/vector as a correctly configured macro,
// because the scheduled SFPSWAP and SFPSTORE ride free sub-units either way.
// The issue rate is therefore blind to whether any compare-exchange happened at
// all. Only a golden comparison distinguishes them.
//
// MERGE_USE_MACRO = 0 rebuilds this file as a plain copy of topk_xl_test.cpp and
// is the control: it establishes that the clone itself did not break anything.
//
// ---------------------------------------------------------------------------
//
// SFPU topk_xl test: bitonic sort / merge / rebuild top-K for K = 512, 1024, 2048.
// Blackhole-only.
//
// This test covers every LLK referenced by the Metal topk_xl headers
// (llk_api/experimental/llk_{unpack_A,math}_topk_xl_copy_api.h and
// llk_math_eltwise_unary_sfpu_topk_xl.h), which wrap the llk_lib entry points:
//   * ckernel::_llk_unpack_topk_xl_copy_init_ / _llk_unpack_topk_xl_copy_
//   * ckernel::_llk_math_topk_xl_copy_init_ / _llk_math_topk_xl_copy_
//   * ckernel::sfpu::_topk_xl_init_ / _topk_xl_local_sort_ / _topk_xl_merge_ /
//     _topk_xl_rebuild_ / _topk_xl_add_lsb_indices_(_init_) /
//     _topk_xl_separate_indices_row_major_(_init_static_ / _reinit_ /
//     _advance_chunk_base_) / _topk_xl_separate_indices_(_init_) /
//     _topk_xl_remove_msb_values_(_init_)
//
// The shared core is copy_sort (copy -> add_lsb_indices -> init(fused) ->
// local_sort); TOPK_XL_INDEX_OP picks the terminal index step. Op 0 mirrors the
// topk_large_indices compute kernel
// (ttnn/.../topk_large_indices/device/kernels/compute.cpp); ops 1 and 2 cover the
// two LLKs that the op does not use.
//
// Fused word: bf16 value (bits 31:16) | u16 index (bits 15:0). Dest is fp32,
// full sync; K=2048 fills tiles 0-7.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Shared compile-time derived constants:
//   TOPK_XL_K:             512 | 1024 | 2048
//   TOPK_XL_NUM_CHUNKS:    number of K-element windows per row (row-major only)
//   TOPK_XL_TAIL_ELEMENTS: valid element count of the last chunk (1 .. K)
//   TOPK_XL_NUM_ROWS:      number of independent top-K problems
//   TOPK_XL_INDEX_OP:      terminal index step:
//     0 row-major   separate_indices_row_major per chunk, then merge/rebuild the
//                   chunks into slot0 -> value region + row-major u32 index region
//     1 separate    generic separate_indices<group_id> -> value region [value|0]
//                   plus index region [group_id<<shift | raw], raw being the
//                   add_lsb tile coordinate, undecoded. Single chunk.
//     2 remove_msb  zero the bf16 value half in place -> [0|raw], the fused region
//                   only. Issued from PACK, where the compute API puts it. Single
//                   chunk.
//   TOPK_XL_GROUP_ID / TOPK_XL_GROUP_SHIFT : generic separate_indices params
//   TOPK_XL_CORE_ID:         add_lsb_indices core_id, index bits [15:11] (0 .. 31)
//   TOPK_XL_ASCENDING:       rebuild direction (false descending, true ascending)
//   TOPK_XL_FUSED_REDUCE:    false unfused merge/rebuild (op path), true fused
//   TOPK_XL_CHUNK_BASE_MODE: three ways to init chunk_base:
//                            0 init_static<hi,lo> | 1 init_upper<hi>(lo) | 2 init(runtime)
//   TOPK_XL_CHUNK_BASE:      starting chunk_base (must be a multiple of K)

constexpr std::uint32_t ELEMENTS_PER_TILE = ckernel::TILE_R_DIM * ckernel::TILE_C_DIM;
constexpr std::uint32_t TILES_PER_SEQ     = (TOPK_XL_K + ELEMENTS_PER_TILE - 1) / ELEMENTS_PER_TILE;
constexpr std::uint32_t SLOT0             = 0;
constexpr bool APPROX                     = false; // The wrappers take this param but never use it.

constexpr bool INDEX_OP_ROW_MAJOR  = (TOPK_XL_INDEX_OP == 0);
constexpr bool INDEX_OP_SEPARATE   = (TOPK_XL_INDEX_OP == 1);
constexpr bool INDEX_OP_REMOVE_MSB = (TOPK_XL_INDEX_OP == 2);

constexpr bool FUSED_REDUCE = TOPK_XL_FUSED_REDUCE;

// Second merge operand. `_topk_xl_merge_` reads it at a fixed distance from the
// first: 64 dest units (one tile) per sequence-tile when fused, 128 (value +
// index region) when unfused, so the slot stride follows the mode.
constexpr std::uint32_t SLOT1 = FUSED_REDUCE ? TILES_PER_SEQ : (2 * TILES_PER_SEQ);

// A lone chunk has nothing to merge with, but the row-major path still rebuilds it;
// the fused path merges/rebuilds only when there is a second operand.
// Both TRISCs use this: MATH issues the rebuild, UNPACK the SrcB dummy valid feeding it.
// Fused variants leave TOPK_XL_INDEX_OP at its 0 default and ignore it, hence the !FUSED_REDUCE.
constexpr bool REBUILD_LONE_CHUNK = !FUSED_REDUCE && INDEX_OP_ROW_MAJOR && TOPK_XL_NUM_CHUNKS == 1;

constexpr std::uint32_t CHUNK_BASE_HI16 = (TOPK_XL_CHUNK_BASE >> 16) & 0xFFFF;
constexpr std::uint32_t CHUNK_BASE_LO16 = TOPK_XL_CHUNK_BASE & 0xFFFF;

// Active element count of chunk `c` (last chunk is the tail).
inline constexpr std::uint32_t chunk_active_elements(std::uint32_t c)
{
    return (c == TOPK_XL_NUM_CHUNKS - 1) ? TOPK_XL_TAIL_ELEMENTS : TOPK_XL_K;
}

inline constexpr std::uint32_t tile_active_elements(std::uint32_t active, std::uint32_t t)
{
    return (t == 0) ? ((active < ELEMENTS_PER_TILE) ? active : ELEMENTS_PER_TILE) : ((active > ELEMENTS_PER_TILE) ? (active - ELEMENTS_PER_TILE) : 0);
}

// Global index of input tile `t` of chunk `c` in row `r`.
inline constexpr std::uint32_t input_tile_index(std::uint32_t r, std::uint32_t c, std::uint32_t t)
{
    return ((r * TOPK_XL_NUM_CHUNKS + c) * TILES_PER_SEQ) + t;
}

#ifdef LLK_TRISC_UNPACK

#include "ckernel_template.h" // ckernel_template used by the topk_xl copy MOP below
#include "experimental/llk_unpack_A_topk_xl_copy.h"
#include "llk_unpack_common.h"

// Replicates llk_unpack_topk_xl_copy_one_tile_unpack() from the metal API wrapper:
// program the partial-tile element count then run the TopK-XL copy MOP for one tile.
inline void unpack_copy_one_tile(std::uint32_t l1_tile_address, std::uint32_t src_format, std::uint32_t dst_format, std::uint32_t elements_this_tile)
{
    const std::uint32_t adc_count = (elements_this_tile == 0) ? (ELEMENTS_PER_TILE - 1) : (elements_this_tile - 1);
    TT_SETADCXX(p_setadc::UNP_A, adc_count, 0x0);
    ckernel::_llk_unpack_topk_xl_copy_(l1_tile_address, src_format, dst_format, elements_this_tile);
}

// Replicates topk_xl_copy_tile<K>() unpack half: 1 tile for K<=1024, 2 tiles for K=2048.
// Marked noinline to ensure K=2048 stays under the TRISC code budget. Emitting one body
// per call site becomes problematic because of loop unrolling.
__attribute__((noinline)) void unpack_copy_tile(RUNTIME_PARAMETERS params, std::uint32_t r, std::uint32_t c, std::uint32_t src_format, std::uint32_t dst_format)
{
    const std::uint32_t active = chunk_active_elements(c);
    for (std::uint32_t t = 0; t < TILES_PER_SEQ; t++)
    {
        unpack_copy_one_tile(L1_ADDRESS(params.buffer_A[input_tile_index(r, c, t)]), src_format, dst_format, tile_active_elements(active, t));
    }
    // Restore the unpacker element count to a full face row (mirrors the trailing
    // TTI_SETADCXX in topk_xl_copy_tile()).
    TTI_SETADCXX(p_setadc::UNP_A, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t src_format = formats.unpack_A_src;
    const std::uint32_t dst_format = formats.unpack_A_dst;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        src_format, src_format, dst_format, dst_format, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES /* unpA_num_faces */, TILE_NUM_FACES /* unpB_num_faces */);
    ckernel::_llk_unpack_topk_xl_copy_init_(src_format, dst_format);
    for (std::uint32_t r = 0; r < TOPK_XL_NUM_ROWS; r++)
    {
        for (std::uint32_t c = 0; c < TOPK_XL_NUM_CHUNKS; c++)
        {
            unpack_copy_tile(params, r, c, src_format, dst_format); // chunk 0 -> slot0, the rest -> slot1
            _llk_unpack_set_srcb_dummy_valid_();                    // local_sort
            if (c > 0)
            {
                _llk_unpack_set_srcb_dummy_valid_(); // rebuild(slot0)
            }
        }
        if constexpr (REBUILD_LONE_CHUNK)
        {
            _llk_unpack_set_srcb_dummy_valid_();
        }
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

// TRISC1 code region overflows by well over 4K under the default -O3.
#pragma GCC optimize("O2")

#include "experimental/llk_math_eltwise_unary_datacopy_topk_xl_copy.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"

using namespace ckernel;

// ===========================================================================
//  SFPLOADMACRO merge -- the thing under test
// ===========================================================================
//
// Replaces `_topk_xl_merge_`'s 16-instruction fused body
//
//     8 SFPLOAD + 4 SFPSWAP (2 cyc each) + 4 SFPSTORE  = 20 cycles / 8 vectors
//
// with an 8-instruction one that produces bit-identical output:
//
//     SFPLOAD      L_B[m] <- B[i]
//     SFPLOADMACRO L_A[m] <- A[i]  + Simple(SFPSWAP) + MAD(SFPNOP) + Store
//                                                     = 8 cycles / 8 vectors
//
// Full derivation, operand plumbing and the cycle-by-cycle collision analysis
// live in sources/topk_merge_macro_perf.cpp's header; the load-bearing points
// are repeated inline below. Kept in this test file rather than in
// ckernel_sfpu_topk_xl.h so the header stays untouched while the idea is being
// evaluated.
#ifndef MERGE_USE_MACRO
#define MERGE_USE_MACRO 1
#endif

#if MERGE_USE_MACRO

namespace macro_merge
{
// LREG0..3 = the A rotation (macroVD targets, receive the max);
// LREG4..7 = the B holders, one per macro index, baked into the four
// InstructionTemplates as SFPSWAP's VC. VD is u3 (SFPLOADMACRO.md:45), so all
// eight rotation slots must be LREG0..LREG7 -- which leaves no LReg for
// constants, and the merge needs none.
constexpr std::uint32_t L_A0 = ckernel::p_sfpu::LREG0;
constexpr std::uint32_t L_B0 = ckernel::p_sfpu::LREG4;

// SFPSWAP Mod1 = 9: "In all lanes, VD = max and VC = min" (SFPSWAP.md:31).
// NOT p_sfpswap::ALL_ROWS_MAX (= 1 = SFPSWAP_MOD1_VEC_MIN_MAX), which is the
// opposite assignment. The direction is forced, not chosen: SFPLOADMACRO always
// overrides Insn.VD (SFPLOADMACRO.md:111-115), so macroVD -- the only register
// the Store slot can reach -- must be the one that receives the max.
// `p_sfpswap` defines no enum for 9 because no shipping kernel wants the max in
// VD; the value is architectural, its own `case 9:` in SFPSWAP.md's functional
// model rather than the `default:` NonContractualBehavior branch.
constexpr std::uint32_t SFPSWAP_MOD1_VD_GETS_MAX = 9;

constexpr std::uint32_t SFPCFG_IMM16_IS_VALUE = 1; // SFPCONFIG.md:108

// Simple byte: selector 4+m -> InstructionTemplate[m].
//   0x80 SET   -> Insn.VB = macroVD, leaving Insn.VC at the template's
//                 LREG(4+m). MANDATORY: with it clear the macro assigns
//                 Insn.VC = macroVD and the SFPSWAP degenerates to comparing
//                 the loaded value against itself.
//   0x40 CLEAR -> Insn.VD = macroVD, where the max lands under Mod1 = 9.
//   delay 0    -> executes the cycle after the SFPLOADMACRO, consuming the
//                 value that same macro just loaded.
constexpr std::uint32_t seq_simple(std::uint32_t m)
{
    return 0x80u | (0u << 3) | (4u + m);
}

// MAD byte: SFPNOP at the same delay. Required by SFPLOADMACRO.md:11 footnote
// (‡): "If SFPSWAP is scheduled to the Simple sub-unit, then SFPNOP needs to be
// scheduled to the MAD sub-unit for the same time".
constexpr std::uint32_t SEQ_MAD = (0u << 3) | 2u;

// Round byte: schedule nothing. Also discharges the second half of (‡) (Simple
// and Round idle on the next cycle) and sidesteps the (†) Simple/Round VD == 16
// exclusivity rule entirely.
constexpr std::uint32_t SEQ_ROUND = 0u;

// Store byte: selector 3 = built-in SFPSTORE, 0x40/0x80 clear -> Insn.VD =
// macroVD, i.e. store the max the SFPSWAP just wrote. Delay 2: the SFPSWAP
// writes on its second cycle (macro+2), so the store fires at macro+3.
constexpr std::uint32_t SEQ_STORE = (2u << 3) | 3u;

constexpr std::uint32_t sequence_word(std::uint32_t m)
{
    return (SEQ_STORE << 24) | (SEQ_ROUND << 16) | (SEQ_MAD << 8) | seq_simple(m);
}

// Misc (SFPLOADMACRO.md:53-57): UsesLoadMod0ForStore bits 4..7 all SET so every
// macro's store inherits the load's INT32 mode -- the fused
// [bf16 value | u16 index] word is an opaque sort key and a format-converting
// store would destroy the index in its low half. UnitDelayKind bits 8/9/11 SET
// puts Simple, MAD and Store on WaitForElapsedInstructions so the delay-2
// producer chain cannot slide if the frontend bubbles at a MOP boundary.
constexpr std::uint32_t MISC_WORD = 0xB00u | 0xF0u;

// SFPLOADMACRO field packing (ckernel_ops.h:689, SFPLOADMACRO.md:20-26,45):
//   lreg_ind = (MacroIndex << 2) | (VD & 3), dest_reg_addr = (Imm9 << 1) | (VD >> 2).
// VD >> 2 lands in bit 0 of the 10-bit Dst address, which SFPLOAD.md:83 states
// "goes unused" -- so rotating VD across 0..7 does not perturb the address.
#define MACRO_MERGE_LOADMACRO(macro_idx, vd, addr_mod, off) \
    TTI_SFPLOADMACRO(((macro_idx) << 2) | ((vd) & 3u), ckernel::InstrModLoadStore::INT32, (addr_mod), (off) | ((vd) >> 2))

// Program the four templates + four Sequence words + Misc. Idempotent.
//
// MUST run after `_init_sfpu_config_reg()` (which `topk_xl_init` issues via
// `_llk_math_eltwise_unary_sfpu_init_`): that clears LaneConfig, and the
// VD >= 12 backdoor -- an instruction with VD >= 12 is STORED into
// InstructionTemplate[VD - 12] rather than executed -- is gated on
// LaneConfig.DISABLE_BACKDOOR_LOAD being false (SFPCONFIG.md:45-46, :120).
inline void configure()
{
    // InstructionTemplate[m] = SFPSWAP whose VC is the m-th B register. The
    // 12..15 in the VD field is the backdoor slot selector, not an operand: the
    // macro overrides Insn.VD with macroVD at issue time.
    TTI_SFPSWAP(0, L_B0 + 0, 12, SFPSWAP_MOD1_VD_GETS_MAX);
    TTI_SFPSWAP(0, L_B0 + 1, 13, SFPSWAP_MOD1_VD_GETS_MAX);
    TTI_SFPSWAP(0, L_B0 + 2, 14, SFPSWAP_MOD1_VD_GETS_MAX);
    TTI_SFPSWAP(0, L_B0 + 3, 15, SFPSWAP_MOD1_VD_GETS_MAX);

    // The Store byte lives in bits 24..31, so Sequence[] does not fit the
    // 16-bit immediate path -- stage the 32-bit word through LReg[0] and write
    // with Mod1 = 0, the idiom of ckernel_sfpu_mul_int.h's _init_mul_int_.
    // Clobbering LReg[0] is harmless: the A rotation is seeded by its own loads.
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, sequence_word(0) & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (sequence_word(0) >> 16) & 0xFFFF);
    TTI_SFPCONFIG(0, 4 + 0, 0);

    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, sequence_word(1) & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (sequence_word(1) >> 16) & 0xFFFF);
    TTI_SFPCONFIG(0, 4 + 1, 0);

    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, sequence_word(2) & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (sequence_word(2) >> 16) & 0xFFFF);
    TTI_SFPCONFIG(0, 4 + 2, 0);

    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, sequence_word(3) & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (sequence_word(3) >> 16) & 0xFFFF);
    TTI_SFPCONFIG(0, 4 + 3, 0);

    TTI_SFPCONFIG(MISC_WORD, 8, SFPCFG_IMM16_IS_VALUE);
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// The body: 8 instructions, 8 input vectors, 4 output vectors. Address layout
// is byte-for-byte `_topk_xl_merge_`'s fused body -- run A at Dst + {0,4,8,12},
// run B at Dst + distance + {0,4,8,12}, merged max written back over run A, Dst
// advanced +16 at the end.
//
// The +16 rides the LAST SFPLOADMACRO's own load, not a store: a
// macro-scheduled SFPSTORE skips ApplyPartialAddrMod entirely
// (SFPLOADMACRO.md:139) and its address was already resolved at SFPLOADMACRO
// time (:140), so the store still lands at +12 rather than at +28.
template <int distance>
inline void body()
{
    TTI_SFPLOAD(L_B0 + 0, ckernel::InstrModLoadStore::INT32, ckernel::ADDR_MOD_7, distance + 0);
    MACRO_MERGE_LOADMACRO(0u, L_A0 + 0, ckernel::ADDR_MOD_7, 0);
    TTI_SFPLOAD(L_B0 + 1, ckernel::InstrModLoadStore::INT32, ckernel::ADDR_MOD_7, distance + 4);
    MACRO_MERGE_LOADMACRO(1u, L_A0 + 1, ckernel::ADDR_MOD_7, 4);
    TTI_SFPLOAD(L_B0 + 2, ckernel::InstrModLoadStore::INT32, ckernel::ADDR_MOD_7, distance + 8);
    MACRO_MERGE_LOADMACRO(2u, L_A0 + 2, ckernel::ADDR_MOD_7, 8);
    TTI_SFPLOAD(L_B0 + 3, ckernel::InstrModLoadStore::INT32, ckernel::ADDR_MOD_7, distance + 12);
    MACRO_MERGE_LOADMACRO(3u, L_A0 + 3, ckernel::ADDR_MOD_5, 12);
}
} // namespace macro_merge

// Drop-in replacement for `ckernel::sfpu::_topk_xl_merge_<K, APPROX, true>`.
// Same signature, same Dst window, same two-column split, same per-column MOP
// trip counts -- see ckernel_sfpu_topk_xl.h:1683-1739, which this mirrors line
// for line. Only the recorded body differs (8 instructions vs 16).
template <std::uint32_t K>
inline void macro_merge_(const std::uint32_t dst_index)
{
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");

    constexpr int num_tiles_per_sequence = (K == 2048) ? 2 : 1;
    constexpr int row_scale_factor       = (K == 512) ? 1 : (K == 1024) ? 2 : 4;
    constexpr int distance               = 64 * num_tiles_per_sequence;
    constexpr std::uint32_t n_iters      = row_scale_factor * 2;

    const std::uint32_t tile_offset = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];

    // The body is 8 instructions, not the 16 `_topk_xl_init_`'s
    // `topk_mop_config<true>()` programmed. Reprogrammed here rather than at
    // init because `_topk_xl_rebuild_<2048>` also owns the MOP template and
    // restores it to REPLAY(0, 16) when it is done.
    ckernel_unpack_template::lA(lltt::replay_insn(0, 8), TT_OP_NOP).program();

    // Recording IS iter 0 of col=0 (Exec), so col=0 fires n_iters - 1 more.
    load_replay_buf<Exec>(0, 8, [] { macro_merge::body<distance>(); });

    ckernel_unpack_template::run(n_iters - 1);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    ckernel::sfpu::set_dst_write_addr_offset(tile_offset + 2);

    ckernel_unpack_template::run(n_iters);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    ckernel::sfpu::set_dst_write_addr_offset(tile_offset + 0);

    // Retire the last two scheduled SFPSTOREs. Their delay counters are
    // WaitForElapsedInstructions, so they only move when this thread issues an
    // SFPU instruction -- and TTI_SETRWC / TT_SETC16 are not SFPU instructions.
    // Without this the final two output vectors of the merge would not be
    // written until the NEXT SFPU op (the rebuild) happened to issue two
    // instructions, which is well after the rebuild has already read them.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

#endif // MERGE_USE_MACRO

template <std::uint32_t K, bool fused>
inline void topk_xl_init()
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::_topk_xl_init_<K, fused>();
#if MERGE_USE_MACRO
    if constexpr (fused)
    {
        // After _init_sfpu_config_reg() cleared LaneConfig, so the VD >= 12
        // template backdoor is open.
        macro_merge::configure();
    }
#endif
}

template <std::uint32_t K>
inline void topk_xl_local_sort(std::uint32_t dst_index, bool ascending)
{
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_local_sort_<K, APPROX>, dst_index, VectorMode::RC_custom, dst_index, ascending);
}

template <std::uint32_t K, bool fused>
inline void topk_xl_merge(std::uint32_t dst_index)
{
#if MERGE_USE_MACRO
    // Only the FUSED merge is swapped. The unfused path keeps values and
    // indices in two separate Dst regions and stores all 8 LREGs, so it has no
    // dead half to exploit and the macro rewrite does not apply to it; leaving
    // it on the shipping body also keeps the unfused variants of this test as a
    // second control.
    if constexpr (fused)
    {
        _llk_math_eltwise_unary_sfpu_params_(macro_merge_<K>, dst_index, VectorMode::RC_custom, dst_index);
        return;
    }
#endif
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_merge_<K, APPROX, fused>, dst_index, VectorMode::RC_custom, dst_index);
}

template <std::uint32_t K, bool fused>
inline void topk_xl_rebuild(std::uint32_t dst_index, bool ascending)
{
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_rebuild_<K, APPROX, fused>, dst_index, VectorMode::RC_custom, dst_index, ascending);
}

inline void topk_xl_add_lsb_indices_init()
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::_topk_xl_add_lsb_indices_init_();
}

template <std::uint32_t K, std::uint32_t core_id>
inline void topk_xl_add_lsb_indices(std::uint32_t dst_index)
{
    static_assert(core_id < 32, "core_id occupies index bits [15:11]");
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_add_lsb_indices_<K, APPROX, core_id>, dst_index, VectorMode::RC_custom);
}

// --- Row-major index split (topk_large_indices op path) ---
template <std::uint32_t chunk_base_upper16, std::uint32_t chunk_base_lower16>
inline void topk_xl_separate_indices_row_major_init_static()
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::_topk_xl_separate_indices_row_major_init_static_<chunk_base_upper16, chunk_base_lower16>();
}

// Same chunk_base latch, runtime value. The flavor a caller uses when the base
// is only known at runtime.
inline void topk_xl_separate_indices_row_major_init(std::uint32_t chunk_base)
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::_topk_xl_separate_indices_row_major_init_(chunk_base);
}

// Hybrid flavor: high half static, low half runtime.
template <std::uint32_t chunk_base_upper16>
inline void topk_xl_separate_indices_row_major_init_upper(std::uint32_t chunk_base_low16)
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::_topk_xl_separate_indices_row_major_init_upper_<chunk_base_upper16>(chunk_base_low16);
}

inline void topk_xl_separate_indices_row_major_reinit()
{
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    ckernel::sfpu::_topk_xl_separate_indices_row_major_reinit_();
}

template <std::uint32_t K>
inline void topk_xl_separate_indices_row_major(std::uint32_t dst_index)
{
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_separate_indices_row_major_<K, APPROX>, dst_index, VectorMode::RC_custom);
}

template <std::uint32_t K>
inline void topk_xl_separate_indices_row_major_advance_chunk_base()
{
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    ckernel::sfpu::_topk_xl_separate_indices_row_major_advance_chunk_base_<K>();
}

// Generic separate_indices (keeps the tile coordinate and prepends group_id).
inline void topk_xl_separate_indices_init(std::uint32_t group_id_bit_shift)
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::_topk_xl_separate_indices_init_(group_id_bit_shift);
}

template <std::uint32_t K, std::uint32_t group_id>
inline void topk_xl_separate_indices(std::uint32_t dst_index)
{
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_separate_indices_<K, APPROX, group_id>, dst_index, VectorMode::RC_custom);
}

// Shared core: copy the chunk into `slot`, stamp indices, fused local-sort.
// Marked noinline to avoid overflowing the code region.
template <std::uint32_t K>
__attribute__((noinline)) void copy_sort(std::uint32_t slot, std::uint32_t active_elements, bool ascending, std::uint32_t dst_format)
{
    ckernel::_llk_math_topk_xl_copy_init_(dst_format);
    for (std::uint32_t t = 0; t < TILES_PER_SEQ; t++)
    {
        ckernel::_llk_math_topk_xl_copy_(slot + t, dst_format, tile_active_elements(active_elements, t));
    }

    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices<K, TOPK_XL_CORE_ID>(slot);

    topk_xl_init<K, true>();
    topk_xl_local_sort<K>(slot, ascending);
}

// Row-major process_chunk: copy_sort then split into unfused values + row-major
// uint32 indices, ready for the merge tree.
template <std::uint32_t K>
__attribute__((noinline)) void process_chunk_math(std::uint32_t slot, std::uint32_t active_elements, bool ascending, std::uint32_t dst_format)
{
    copy_sort<K>(slot, active_elements, ascending, dst_format);
    topk_xl_separate_indices_row_major_reinit();
    topk_xl_separate_indices_row_major<K>(slot);
    topk_xl_separate_indices_row_major_advance_chunk_base<K>();
}

// Init + (optional) merge of slot1 into slot0 + rebuild, in either fused mode.
// `fused` selects the whole merge/rebuild code family: operand distance, MOP body
// length and iteration count all differ (see topk_mop_config / _topk_xl_merge_).
// `_topk_xl_merge_` always keeps the max half, so TOPK_XL_ASCENDING changes only
// the order the surviving top-K is rebuilt into, not which elements survive.
template <std::uint32_t K, bool fused>
__attribute__((noinline)) void merge_and_rebuild(bool do_merge)
{
    topk_xl_init<K, fused>();
    if (do_merge)
    {
        topk_xl_merge<K, fused>(SLOT0);
    }
    topk_xl_rebuild<K, fused>(SLOT0, TOPK_XL_ASCENDING);
}

// Save the starting chunk_base through the requested init flavor.
// All three save into LREG12, but the value is split differently.
inline void topk_xl_chunk_base_init()
{
    if constexpr (TOPK_XL_CHUNK_BASE_MODE == 0)
    {
        topk_xl_separate_indices_row_major_init_static<CHUNK_BASE_HI16, CHUNK_BASE_LO16>();
    }
    else if constexpr (TOPK_XL_CHUNK_BASE_MODE == 1)
    {
        topk_xl_separate_indices_row_major_init_upper<CHUNK_BASE_HI16>(CHUNK_BASE_LO16);
    }
    else
    {
        topk_xl_separate_indices_row_major_init(TOPK_XL_CHUNK_BASE);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t math_format = formats.math;

    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);

    for (std::uint32_t r = 0; r < TOPK_XL_NUM_ROWS; r++)
    {
        _llk_math_wait_for_dest_available_<dest_sync>();

        if constexpr (FUSED_REDUCE)
        {
            // Fused reduction: chunks stay in the fused [value|index] form all the
            // way through merge/rebuild, and the index split happens once at the end.
            copy_sort<TOPK_XL_K>(SLOT0, chunk_active_elements(0), false /* ascending */, math_format);

            for (std::uint32_t c = 1; c < TOPK_XL_NUM_CHUNKS; c++)
            {
                copy_sort<TOPK_XL_K>(SLOT1, chunk_active_elements(c), true /* ascending */, math_format);
                merge_and_rebuild<TOPK_XL_K, true /* fused */>(true /* do_merge */);
            }

            topk_xl_separate_indices_init(TOPK_XL_GROUP_SHIFT);
            topk_xl_separate_indices<TOPK_XL_K, TOPK_XL_GROUP_ID>(SLOT0);
        }
        else if constexpr (INDEX_OP_ROW_MAJOR)
        {
            topk_xl_chunk_base_init();

            // chunk 0 -> slot0, local-sort descending.
            process_chunk_math<TOPK_XL_K>(SLOT0, chunk_active_elements(0), false /* ascending */, math_format);

            for (std::uint32_t c = 1; c < TOPK_XL_NUM_CHUNKS; c++)
            {
                // chunk c -> slot1, local-sort ascending, then merge into slot0.
                process_chunk_math<TOPK_XL_K>(SLOT1, chunk_active_elements(c), true /* ascending */, math_format);
                merge_and_rebuild<TOPK_XL_K, false /* fused */>(true /* do_merge */);
            }
            if constexpr (REBUILD_LONE_CHUNK)
            {
                merge_and_rebuild<TOPK_XL_K, false /* fused */>(false /* do_merge */);
            }
        }
        else
        {
            // Single-chunk terminal ops: copy_sort, then the index step. For separate
            // the split is here (MATH). For remove_msb the value-half zero runs on
            // PACK (as the op does), so MATH just leaves the fused [value|index] in slot0.
            static_assert(INDEX_OP_ROW_MAJOR || TOPK_XL_NUM_CHUNKS == 1, "terminal index ops (INDEX_OP 1/2) are single-chunk only");
            copy_sort<TOPK_XL_K>(SLOT0, chunk_active_elements(0), false /* fused */, math_format);

            if constexpr (INDEX_OP_SEPARATE)
            {
                topk_xl_separate_indices_init(TOPK_XL_GROUP_SHIFT);
                topk_xl_separate_indices<TOPK_XL_K, TOPK_XL_GROUP_ID>(SLOT0);
            }
        }

        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"

// remove_msb_values on PACK: verbatim reproduction of the Metal wrapper
// llk_math_eltwise_unary_sfpu_topk_xl_remove_msb_values, which Compute API invokes through
// PACK(...). Zeros the bf16 value half of the fused Dest words in place: [0 | index].
template <std::uint32_t K>
inline void pack_remove_msb_values(std::uint32_t dst_index)
{
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index + get_dest_buffer_base());
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH | p_stall::PACK);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // The SFPU drain on the way in and the pack drain on the way out both live
    // inside `_topk_xl_remove_msb_values_`. The LLK static_asserts SyncFull.
    ckernel::sfpu::_topk_xl_remove_msb_values_<K, dest_sync>();
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t pack_src_format = formats.pack_src;
    const std::uint32_t pack_dst_format = formats.pack_dst; // UInt32: raw 32-bit words.

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        pack_src_format, pack_dst_format, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, 4 /* num_faces */);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(pack_dst_format, FACE_R_DIM, TILE_C_DIM, 4 /* num_faces */);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();

    if constexpr (INDEX_OP_REMOVE_MSB)
    {
        ckernel::sfpu::_topk_xl_remove_msb_values_init_();
    }

    // remove_msb: the in-place fused region [0|index] (TILES_PER_SEQ). Otherwise the
    // value region then the index region (2*TILES_PER_SEQ).
    constexpr std::uint32_t RESULT_TILES_PER_ROW = INDEX_OP_REMOVE_MSB ? TILES_PER_SEQ : (2 * TILES_PER_SEQ);

    for (std::uint32_t r = 0; r < TOPK_XL_NUM_ROWS; r++)
    {
        _llk_packer_wait_for_math_done_();

        if constexpr (INDEX_OP_REMOVE_MSB)
        {
            // Zero the value half on PACK, then pack the fused region as [0|index].
            // Includes the trailing SFPU drain before the pack below.
            pack_remove_msb_values<TOPK_XL_K>(SLOT0);
        }

        std::uint32_t res = r * RESULT_TILES_PER_ROW;
        // Value / fused region of slot0: Dest tiles [SLOT0 .. SLOT0 + TILES_PER_SEQ - 1].
        for (std::uint32_t t = 0; t < TILES_PER_SEQ; t++)
        {
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(SLOT0 + t, L1_ADDRESS(params.buffer_Res[res++]));
        }
        if constexpr (!INDEX_OP_REMOVE_MSB)
        {
            // Index region of slot0: Dest tiles [SLOT0 + TILES_PER_SEQ .. +2*TILES_PER_SEQ - 1].
            for (std::uint32_t t = 0; t < TILES_PER_SEQ; t++)
            {
                _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(SLOT0 + TILES_PER_SEQ + t, L1_ADDRESS(params.buffer_Res[res++]));
            }
        }

        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif // LLK_TRISC_PACK
