// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// SFPLOADMACRO + SFPU INDEX-TRACKING PROBE (Blackhole).
//
// THE QUESTION THIS FILE ANSWERS. The measured topk_xl merge/rebuild macro win
// (SFPSWAP scheduled into an SFPLOADMACRO Simple slot, store riding the load's
// address) exists only on the FUSED path. The shipping consumer —
// ttnn.experimental.topk_large_indices — runs merge/rebuild UNFUSED, where
// values live in LREG0..3 and their indices ride along in LREG4..7 through the
// SFPU index-tracking mode (LaneConfig.ENABLE_DEST_INDEX, bit [2], the 0x4 that
// `_topk_xl_init_<K, false>` writes). Porting the macro to the unfused path is
// only possible if a MACRO-SCHEDULED SFPSWAP still performs the argmin/argmax
// companion swap of LReg[4+(VC&3)] <-> LReg[4+(VD&3)] that SFPSWAP.md:58-70
// specifies for a software SFPSWAP.
//
// Nothing in the ISA documentation states whether the companion swap survives
// macro scheduling, and two documented hazards sit right next to it:
//
//   * SFPSWAP.md CAUTION: a macro-scheduled SFPSWAP is exempt from the
//     automatic 1-cycle stall and occupies the Simple sub-unit for 2 cycles.
//     Index tracking adds a companion write (ckernel_sfpu_topk.h:974 notes an
//     extra 1-cycle stall for the software case), so the fused path's
//     "2 issue slots apart + 2 drain SFPNOPs" rule may be insufficient here.
//   * TEN-2932 (SFPCONFIG.md LaneConfig bit [2]): while ENABLE_DEST_INDEX is
//     set, instructions OTHER THAN SFPLOAD / SFPLOADI / SFPSWAP / SFPTRANSP
//     that write LReg[4..7] are UnsupportedFunctionality. SFPLOADMACRO is not
//     in the allowed list, so a macro whose macroVD is an index register
//     (LREG4..7) may or may not deliver a clean load.
//
// Every arm of this probe runs the same tiny fragment of the unfused merge
// geometry — two value regions (run A / run B) in Dst tile 0, their index
// regions at the same offsets in Dst tile 1 — and packs both tiles out
// unmodified except for the compare-exchange under test. The python driver
// (test_topk_unfused_macro_probe.py) compares arms against each other and
// against a positional torch golden.
//
// PROBE_ARM (emitted into build.h by the python driver):
//   0  SW_BOTH        both pairs via software SFPSWAP under index tracking —
//                     the shipping unfused primitive; reference for arms 1-5.
//   1  MACRO_SINGLE   pair (L0,L2) on a macro-scheduled SFPSWAP, pair (L1,L3)
//                     in software after a generous drain. THE yes/no arm:
//                     arm1 == arm0 iff the macro honours index tracking.
//   2  MACRO_DUAL_2A  both pairs on macros, 2 issue slots apart (the fused
//                     path's interleave rule) + 2 drain SFPNOPs.
//   3  MACRO_DUAL_3A  both pairs on macros, 3 issue slots apart + 2 drains.
//                     2 vs 3 localises any tracking-induced Simple-occupancy
//                     growth: arm2 wrong + arm3 right => 3-slot rule unfused.
//   4  MACRO_FULL     arm2 plus the value stores riding the macros' Store
//                     slots (delay 2) — the merge's full trick, value half.
//   5  MACRO_MUTATE   arm2 with BOTH swap macros' Sequence words zeroed —
//                     "schedule nothing", the documented degeneration of
//                     SFPLOADMACRO into a plain SFPLOAD. No compare-exchange
//                     happens and the software stores write back the loaded
//                     registers, so the output must equal the RAW INPUT and
//                     must NOT equal arm0 — proves arms 1-4 are sensitive to
//                     the thing they claim to test. (An earlier mutation that
//                     cleared only the Simple byte's 0x80 bit produced a
//                     same-register SFPSWAP whose 2-cycle read-modify-write of
//                     one LReg is not architecturally modelable — silicon
//                     emitted garbage, not a no-op — so it cannot serve as a
//                     control. Measured 2026-08-16: 0xBF2CC4C7 broadcast.)
//   6  IDX_STORE_MACRO single pair (L0,L2) on a swap macro, AND the run-B index
//                     load (L6) carried by a store-only macro whose store is
//                     delayed (WaitForElapsedInstructions, delay 6) past the
//                     companion swap. Tests two things at once: whether a
//                     macro LOAD into LREG4..7 is clean under TEN-2932, and
//                     whether a deferred macro store emits the POST-companion-
//                     swap index (which would let unfused index stores ride
//                     macros for free).
//   7  SW_SINGLE      software reference for arm 6: same single pair, all
//                     loads/stores software.
//
// Geometry (Dst offsets, dest units; one unit = one 16-datum Dst row, one
// SFPU load/store covers 2 units = 32 lanes):
//   values  (Dst tile 0): run A at {0, 4}, run B at {16, 20}     — FP32 mode
//   indices (Dst tile 1): run A at {64, 68}, run B at {80, 84}   — INT32 mode
// Register map (the shipping unfused ascending layout, bitonic_sort_len_k):
//   LREG0 <- A0 vals   LREG1 <- A1 vals   LREG2 <- B0 vals   LREG3 <- B1 vals
//   LREG4 <- A0 idx    LREG5 <- A1 idx    LREG6 <- B0 idx    LREG7 <- B1 idx
// Compare-exchange (ascending, VD gets min — SFPSWAP_MOD1_VEC_MIN_MAX):
//   SFPSWAP(VC=LREG2, VD=LREG0)  companions LREG6 <-> LREG4
//   SFPSWAP(VC=LREG3, VD=LREG1)  companions LREG7 <-> LREG5

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr std::uint32_t RES_TILES = 3; // values, indices, diag
constexpr std::uint32_t DIAG_TILE = 2;

#ifdef LLK_TRISC_UNPACK

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
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, params.num_faces),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    // Tile 0 (values) -> Dst tile 0, tile 1 (indices) -> Dst tile 1.
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[1]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

#include "ckernel_addrmod.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "params.h"
#include "sfpu/ckernel_sfpu_load_config.h"

using namespace ckernel;

namespace probe
{

// Dst offsets (dest units, relative to the SFPU dest base set by
// _llk_math_eltwise_sfpu_start_(0)). Tile 0 = units 0..63, tile 1 = 64..127.
constexpr int VAL_A0 = 0;
constexpr int VAL_A1 = 4;
constexpr int VAL_B0 = 16;
constexpr int VAL_B1 = 20;
constexpr int IDX_A0 = 64;
constexpr int IDX_A1 = 68;
constexpr int IDX_B0 = 80;
constexpr int IDX_B1 = 84;

constexpr std::uint32_t SFPCFG_IMM16_IS_VALUE = 1; // SFPCONFIG.md:108

// --- Sequence words --------------------------------------------------------
//
// Simple byte: selector 4+m -> InstructionTemplate[m], delay 0.
//   0x80 SET   -> Insn.VB = macroVD, leaving Insn.VC at the template's value
//                 (SFPSWAP ignores VB, so this is purely "don't clobber VC").
//   0x40 CLEAR -> Insn.VD = macroVD.
//
// MAD byte: SFPNOP at delay 0 — required whenever SFPSWAP is scheduled to the
// Simple sub-unit (SFPLOADMACRO.md:11 footnote ‡).
constexpr std::uint32_t SEQ_MAD_NOP = (0u << 3) | 2u;

// Store byte for the swap macros: only PROBE_ARM == 4 schedules the built-in
// SFPSTORE (selector 3) at delay 2 — the SFPSWAP writes macroVD on its second
// cycle (macro+2), the store fires at macro+3, exactly the fused merge's
// timing. 0x40/0x80 clear -> Insn.VD = macroVD.
constexpr std::uint32_t SWAP_STORE_BYTE = (PROBE_ARM == 4) ? ((2u << 3) | 3u) : 0u;

// THE MUTATION (PROBE_ARM == 5) zeroes both swap macros' ENTIRE Sequence
// words: "schedule nothing" degenerates each SFPLOADMACRO into a plain
// SFPLOAD (the exact failure mode the branch's macro work documents as
// timing-invisible), so no compare-exchange runs and the body's software
// stores write back the loaded registers — a provable identity. Do NOT
// mutate by clearing only the 0x80 bit: that yields SFPSWAP(VC == VD), a
// same-register 2-cycle read-modify-write that silicon resolves as garbage
// (measured 0xBF2CC4C7 broadcast), not as a modelable no-op.
constexpr std::uint32_t SEQ_SWAP_BODY_M0 = (SWAP_STORE_BYTE << 24) | (SEQ_MAD_NOP << 8) | (0x80u | (0u << 3) | 4u);
constexpr std::uint32_t SEQ_SWAP_BODY_M1 = (SWAP_STORE_BYTE << 24) | (SEQ_MAD_NOP << 8) | (0x80u | (0u << 3) | 5u);
constexpr std::uint32_t SEQ_M0           = (PROBE_ARM == 5) ? 0u : SEQ_SWAP_BODY_M0;
constexpr std::uint32_t SEQ_M1           = (PROBE_ARM == 5) ? 0u : SEQ_SWAP_BODY_M1;

// Macro 2 (PROBE_ARM == 6 only): store-only, nothing on Simple/MAD/Round.
// Delay 6 on WaitForElapsedInstructions puts the store 6 issued instructions
// after its SFPLOADMACRO — past the companion swap the value macro triggers.
constexpr std::uint32_t SEQ_M2 = ((6u << 3) | 3u) << 24;

// Misc (SFPLOADMACRO.md:53-57):
//   0xF0  -> UsesLoadMod0ForStore for all four macros: a scheduled store
//            inherits its load's Mod0 (FP32 for the value macros, INT32 for
//            the index macro) — raw, format-preserving writes.
//   0xB00 -> Simple (bit 8), MAD (bit 9) and Store (bit 11) on
//            WaitForElapsedInstructions so no scheduled slot can slide if the
//            frontend bubbles.
constexpr std::uint32_t MISC_WORD = 0xB00u | 0xF0u;

// SFPLOADMACRO field packing (ckernel_ops.h:689, SFPLOADMACRO.md:20-26,45):
//   lreg_ind = (MacroIndex << 2) | (VD & 3), dest_reg_addr = (Imm9 << 1) | (VD >> 2).
// The VD >> 2 bit lands in bit 0 of the Dst address, which SFPLOAD.md:83 says
// is unused — so VD in 0..7 never perturbs the address.
#define PROBE_LOADMACRO(macro_idx, vd, mod0, off) TTI_SFPLOADMACRO(((macro_idx) << 2) | ((vd) & 3u), (mod0), ckernel::ADDR_MOD_7, (off) | ((vd) >> 2))

// Write LoadMacroConfig::Sequence[idx] — the 32-bit word does not fit the
// 16-bit immediate path, so stage it through LReg[0] (the _init_mul_int_
// idiom). LReg[0] is reloaded by the body's own loads afterwards.
template <std::uint32_t IDX, std::uint32_t VALUE>
inline void write_sequence()
{
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, VALUE & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (VALUE >> 16) & 0xFFFF);
    TTI_SFPCONFIG(0, 4 + IDX, 0);
}

// One-time setup. MUST run after _llk_math_eltwise_unary_sfpu_init_once_(),
// which clears LaneConfig and so leaves the VD >= 12 backdoor open for the
// InstructionTemplate writes (SFPCONFIG.md:45-46).
inline void configure()
{
    // Reset the lane-enable CC state (the in-tree idiom, cf.
    // ckernel_sfpu_rounding_ops.h). _llk_math_eltwise_unary_sfpu_init_once_
    // does NOT touch CC, and a stale mask with disabled lanes would suppress
    // the swap in exactly those lanes — indistinguishable from a semantics
    // difference unless ruled out here.
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPNOP;

    // ADDR_MOD_7 — zero advance; every load/store in this probe addresses Dst
    // absolutely through its immediate.
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);

    // InstructionTemplate[0/1] = the two ascending compare-exchanges, VC baked,
    // VD field 12/13 = backdoor slot selector (the macro overrides VD with
    // macroVD at issue time). Mod1 stays ALL_ROWS_MAX: both halves are kept
    // and the shipping operand order already puts min where macroVD is.
    TTI_SFPSWAP(0, p_sfpu::LREG2, 12, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, 13, p_sfpswap::ALL_ROWS_MAX);

    write_sequence<0, SEQ_M0>();
    write_sequence<1, SEQ_M1>();
    write_sequence<2, SEQ_M2>();
    TTI_SFPCONFIG(MISC_WORD, 8, SFPCFG_IMM16_IS_VALUE);
    TTI_SFPNOP;
    TTI_SFPNOP;

    // Enable SFPU index-tracking mode LAST (bit [2] of LaneConfig — exactly
    // what _topk_xl_init_<K, fused=false> writes). Bit [1]
    // (DISABLE_BACKDOOR_LOAD) stays 0, which is fine: everything issued from
    // here on has VD < 12.
    ckernel::sfpu::_sfpu_load_config32_(0xF, 0x0, 0x4);
    TTI_SFPNOP;
}

// --- Shared fragments -------------------------------------------------------

inline void load_indices_all()
{
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, IDX_A0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, IDX_A1);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, IDX_B0);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, IDX_B1);
}

inline void store_values_all()
{
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_A0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_A1);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_B0);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_B1);
}

inline void store_indices_all()
{
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, IDX_A0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, IDX_A1);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, IDX_B0);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, IDX_B1);
}

// --- The probe body ----------------------------------------------------------

inline void body()
{
    if constexpr (PROBE_ARM == 0)
    {
        // SW_BOTH — the shipping unfused primitive, verbatim semantics:
        // plain loads, two software SFPSWAPs (hardware auto-stalls after
        // each), software stores. Companion swaps come from index tracking.
        load_indices_all();
        TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_B0);
        TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_B1);
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_A0);
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_A1);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        // Extra settle before the stores read LREG4..7: the companion write of
        // the second swap may land one cycle behind its value write.
        TTI_SFPNOP;
        TTI_SFPNOP;
        store_values_all();
        store_indices_all();
    }
    else if constexpr (PROBE_ARM == 1)
    {
        // MACRO_SINGLE — pair (L0,L2) macro-scheduled, pair (L1,L3) software
        // after a generous drain. Separation effects are arm 2/3's business;
        // this arm isolates the yes/no question.
        load_indices_all();
        TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_B0);
        PROBE_LOADMACRO(0u, p_sfpu::LREG0, InstrModLoadStore::FP32, VAL_A0);
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_B1);
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_A1);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPNOP;
        TTI_SFPNOP;
        store_values_all();
        store_indices_all();
    }
    else if constexpr (PROBE_ARM == 2 || PROBE_ARM == 5)
    {
        // MACRO_DUAL_2APART (and its mutation, arm 5) — the fused merge's
        // interleave rule: macros 2 issue slots apart, each plain load
        // supplying the next macro's VC, then 2 drain SFPNOPs.
        load_indices_all();
        TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_B0);
        PROBE_LOADMACRO(0u, p_sfpu::LREG0, InstrModLoadStore::FP32, VAL_A0);
        TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_B1);
        PROBE_LOADMACRO(1u, p_sfpu::LREG1, InstrModLoadStore::FP32, VAL_A1);
        TTI_SFPNOP;
        TTI_SFPNOP;
        store_values_all();
        store_indices_all();
    }
    else if constexpr (PROBE_ARM == 3)
    {
        // MACRO_DUAL_3APART — one extra slot between the macros. If arm 2 is
        // wrong and this is right, index tracking grows the Simple occupancy
        // and the unfused port needs a 3-slot interleave.
        load_indices_all();
        TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_B0);
        PROBE_LOADMACRO(0u, p_sfpu::LREG0, InstrModLoadStore::FP32, VAL_A0);
        TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_B1);
        TTI_SFPNOP;
        PROBE_LOADMACRO(1u, p_sfpu::LREG1, InstrModLoadStore::FP32, VAL_A1);
        TTI_SFPNOP;
        TTI_SFPNOP;
        store_values_all();
        store_indices_all();
    }
    else if constexpr (PROBE_ARM == 4)
    {
        // MACRO_FULL — arm 2 plus the value stores riding the macros' Store
        // slots (delay 2, address = the load's own). Software stores for the
        // VC halves and the indices are placed off the cycles the two macro
        // stores fire on (macro at i, store at i+3 counted in issues), the
        // same NOP pattern macro_ce_body uses.
        load_indices_all();
        TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_B0);
        PROBE_LOADMACRO(0u, p_sfpu::LREG0, InstrModLoadStore::FP32, VAL_A0);
        TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_B1);
        PROBE_LOADMACRO(1u, p_sfpu::LREG1, InstrModLoadStore::FP32, VAL_A1);
        TTI_SFPNOP; // m0's store fires here — keep the cycle store-free
        TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_B0);
        TTI_SFPNOP; // m1's store fires here
        TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_B1);
        store_indices_all();
    }
    else if constexpr (PROBE_ARM == 6)
    {
        // IDX_STORE_MACRO — single pair (L0,L2); the run-B index load (L6)
        // rides macro 2, store-only, delay 6. Issue timeline (i = issue idx):
        //   i0 L4 plain           i4 M2: L6 <- IDX_B0 (store scheduled, d6)
        //   i1..i2 (value VC)     i5 M0: L0 <- VAL_A0 (swap scheduled, d0)
        //   ...                   i6..i11 SFPNOP — the swap completes and the
        //                         companion LREG4<->LREG6 write lands early in
        //                         this window; m2's store counts down through
        //                         it and fires INSIDE it, reading LREG6's
        //                         post-companion-swap content.
        // Software stores for L0/L2/L4 follow; L6's Dst word is written ONLY
        // by the macro store.
        TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, IDX_A0);
        TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_B0);
        PROBE_LOADMACRO(2u, p_sfpu::LREG6, InstrModLoadStore::INT32, IDX_B0);
        PROBE_LOADMACRO(0u, p_sfpu::LREG0, InstrModLoadStore::FP32, VAL_A0);
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_A0);
        TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_B0);
        TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, IDX_A0);
    }
    else if constexpr (PROBE_ARM == 7)
    {
        // SW_SINGLE — software reference for arm 6: same single pair, all
        // loads/stores software.
        TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, IDX_A0);
        TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, IDX_B0);
        TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_B0);
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_A0);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_A0);
        TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, VAL_B0);
        TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, IDX_A0);
        TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, IDX_B0);
    }
    else
    {
        static_assert(PROBE_ARM <= 7, "PROBE_ARM must be 0..7");
    }

    // Retire any still-outstanding scheduled instructions: their delay
    // counters are WaitForElapsedInstructions, and only SFPU issues move them.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

} // namespace probe

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
        params.num_faces, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_wait_for_dest_available_<dest_sync>();

    // Values -> Dst tile 0, indices -> Dst tile 1 (raw 32-bit copies).
    _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0, formats.math, formats.math, params.num_faces);
    _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        1, formats.math, formats.math, params.num_faces);

    // Clears LaneConfig — precondition for the template backdoor writes.
    _llk_math_eltwise_unary_sfpu_init_once_();
    probe::configure();

    _llk_math_eltwise_sfpu_start_(0);
    probe::body();
    _llk_math_eltwise_sfpu_done_();

    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(1, L1_ADDRESS(params.buffer_Res[1]));
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();

    TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK);
    tensix_sync();

    // Ran-to-completion sentinel + arm echo + freshness nonce, read by the
    // python driver. The nonce is a RUNTIME argument (marshalled into the
    // params buffer at execution time, no rebuild), so a stale/cached result
    // buffer cannot echo the value the driver chose for THIS run — asserting
    // diag[2] == nonce proves the kernel executed fresh on device. The driver
    // repurposes the RELU_CONFIG runtime slot as the nonce carrier.
    volatile std::uint32_t* diag = reinterpret_cast<volatile std::uint32_t*>(params.buffer_Res[DIAG_TILE]);
    diag[0]                      = 0xC0DEBA5E;
    diag[1]                      = PROBE_ARM;
    diag[2]                      = params.RELU_CONFIG;
    diag[12]                     = 0xC0DEE0D1;
}

#endif // LLK_TRISC_PACK
