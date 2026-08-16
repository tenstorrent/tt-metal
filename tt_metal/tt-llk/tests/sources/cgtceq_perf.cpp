// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================
// (Cgt, Ceq) exact-count engine microbenchmark  (Blackhole only)
// ============================================================================
//
// HONESTY GUARD (from the cgtceq research report / RADX audit IMPL-2,
// RADIX_BUCKET_GPU.md:609-615): this bench prices the Gate-2 correctness
// ORACLE only. Even a perfect 25-cycle rendezvous does not create a win region
// before Gate 4 (candidate materialization) -- at N = 32768 the bisection
// costs ~decisions x 2-4k cycles on top of the unpack floor while the
// incumbent bitonic path pays zero threshold-search cost. The value of these
// numbers is (a) closing dependency-map open dep #1 with measured constants
// and (b) giving the dual-RISC BF16 histogram alternative an honest SFPU-side
// comparator for the Gate-3 shootout. Nothing here is a claimed speedup.
//
// WHAT IS MEASURED (three deliverables, one kernel):
//
//   (i)  ADDITIVITY. Streamed arms in the topk_negfilter_perf.cpp pipeline
//        shape: a single strict count (Cgt) and a dual strict count
//        (Cgt at T, Cgt at pred(T), i.e. (Cgt, Cge) and Ceq = Cge - Cgt) per
//        streamed tile, under L1_TO_L1 / UNPACK_ISOLATE / MATH_ISOLATE. The
//        Gate-2 question: is L1_TO_L1(single) - L1_TO_L1(none) equal to
//        MATH_ISOLATE(single)? (SFPU work adds to the ~3.94 cyc/vec fp32
//        unpack_to_dest floor because math and unpack share Dst; PACK does
//        not.) Priors: single ~2.0 cyc/vec, dual ~4.0.
//
//   (ii) RENDEZVOUS. Segmented-restart arm (the sfpu_count_above_perf.cpp
//        ARM_PASS_SYNC method, VECTORS_PER_SEGMENT = 64): after every
//        segment the count is folded across lanes, SFPSTOREd to a scratch Dst
//        row, ordered, and READ BACK by this same TRISC1 through the
//        memory-mapped Dst window at 0xFFBD8000 (Dst.md:103;
//        ckernel_dest.h::configure_dest_access; read precedent
//        dprint_tensix.h). The next threshold is a function of the count that
//        was just read -- a real control dependency, which is what
//        ARM_PASS_SYNC deliberately left out. 3x3 arms:
//            FOLD_DEPTH 0: full fold  (SFPTRANSP + 3xSFPIADD +
//                          7x(SFPSHFT2 ROR1 + SFPNOP + SFPIADD)), read 1 word
//            FOLD_DEPTH 1: partial    (SFPTRANSP + 3xSFPIADD), read 16 words
//            FOLD_DEPTH 2: none       (store raw lane partials), read 64 words
//            SYNC_PRIM  0: tensix_sync()             (control, >= 25.1 cyc)
//            SYNC_PRIM  1: t6_semaphore_post<WAIT_SFPU> + pc_buf poll
//            SYNC_PRIM  2: polled Dst sentinel (RISC pre-writes 0xFFFFFFFF)
//        cycles/decision = (slope_rendezvous - slope_rate) * 64, two-point
//        slope over ITER_COUNT {512, 2048}. ARM_RATE is the in-kernel
//        subtraction partner (plain CountD1 shape, no restarts).
//        NEVER a bare STALLWAIT: the Wait Gate gates Tensix instructions
//        only; a RISCV lw from 0xFFBD8000 never consults it (report 1.6).
//
//  (iii) BISECTION. TRISC1-driven <= 16-decision bisection over the 16-bit
//        sign-magnitude key space to the exact K-th threshold of Dst-resident
//        rows (1 row = 1 tile = 1024 bf16-pattern fp32 words), certified by a
//        dual (Cgt, Cge) count at the found key and its predecessor key.
//        Success modes: CERT (Cgt < K <= Cgt+Ceq, key == exact K-th value) or
//        VALIDSET (an interior probe hit Cgt == K exactly). Per-row decision
//        counts and wall-clock cycles land in a diagnostics block the host
//        reads back; the python driver checks every count against an exact
//        sign-magnitude golden -- any mismatch fails the test.
//
// SELF-CHECKING: the rendezvous arm counts KNOWN data (2 tiles unpacked into
// Dst during INIT) and reports an XOR checksum of the per-segment counts plus
// the data-dependent threshold automaton's trace; python simulates the same
// automaton exactly. The bisection arm is checked per row. The streamed arms
// are timing-only (accumulators roll across tiles); their loop body is
// byte-identical to the checked ones.
//
// FOLD PROHIBITION: the fold sequence is software-issued TTI_*, never hosted
// inside an SFPLOADMACRO sequence -- SFPSHFT2.md:166-171 makes the ROR1
// bubble automatic only for software-issued sequences; inside a macro it is
// UB. In-tree ROR1 precedent: ckernel_sfpu_binary_bcast.h:209-218.
//
// SFPGT footnote (report 1.1): SET_VD writes VD only if VD < 8 (or 16), so
// every mask target here stays in LREG0..7. SFPGT orders by the
// sign-magnitude total order -NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN,
// so bisection runs on raw fp32/bf16 bit patterns with no premap.
//
// Representation trap (report 1.2): the accumulator holds -(count) in two's
// complement; it is negated ONCE (SFPIADD 2SCOMP vs LCONST_0) before any
// store, because InstrModLoadStore::INT32_2S_COMP conversion is a no-op on
// Blackhole (ckernel_sfpu_add_int.h:28-29). Non-negative values are
// bit-identical under two's complement and sign-magnitude, which also makes
// the MMIO read's sign-magnitude -> two's-complement conversion (fmt=INT32,
// Dst.md:105-113) an identity on them.
//
// Run the consumer phase under `flock /tmp/tt-device.lock`. ttsim cannot
// substitute: it does not implement SFPLOADMACRO and is not cycle-accurate.

#include <algorithm>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_dest.h" // RISCV_DEST_START_ADDR, configure_dest_access<>
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_structs.h" // semaphore::
#include "ckernel_template.h"
#include "counters.h"
#include "llk_defs.h"
#include "lltt.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "sfpu/ckernel_sfpu_load_config.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// ---------------------------------------------------------------------------
// Parameters. All #define (not constexpr): each is guarded with #ifndef and a
// constexpr would not satisfy the preprocessor guard -- the fallback would
// fire and every swept variant would silently compile identically
// (the sfpu_count_above_perf.cpp lesson, verbatim).
// ---------------------------------------------------------------------------
#ifndef CGTCEQ_ARM
#define CGTCEQ_ARM 5
#endif

// 32-element vectors in the timed region of the rate/rendezvous arms.
#ifndef ITER_COUNT
#define ITER_COUNT 512
#endif

// Raw fp32 bit patterns (NOT floats: -0.0 / +-Inf / NaN must be exact).
#ifndef THR_BITS
#define THR_BITS 0x3F800000 // 1.0f
#endif
#ifndef THR2_BITS
#define THR2_BITS 0x3F000000 // 0.5f
#endif

// Rendezvous crossing.
#ifndef FOLD_DEPTH
#define FOLD_DEPTH 0
#endif
#ifndef SYNC_PRIM
#define SYNC_PRIM 0
#endif

// Bisection.
#ifndef BISECT_K
#define BISECT_K 32
#endif
#ifndef BISECT_ROWS
#define BISECT_ROWS 3
#endif

// Which word of the scratch 4-row window carries lane 0 after a full fold.
// Pinned by the python correctness check rather than derived from the
// cross-lane diagram (the pack_exp_histogram "prove by construction" method):
// if the lane map differs from the model, the count check fails loudly and
// this define is the one-line fix.
#ifndef R0_WORD
#define R0_WORD 0
#endif

namespace
{
constexpr std::uint32_t ARM_STREAM_NONE   = 0; // stream floor (C0)
constexpr std::uint32_t ARM_STREAM_SINGLE = 1; // C1 (MATH_ISOLATE) / C3 (L1_TO_L1)
constexpr std::uint32_t ARM_STREAM_DUAL   = 2; // C2 / C4
constexpr std::uint32_t ARM_CTRL_LOAD     = 3; // control: 1.0 cyc/vec floor
constexpr std::uint32_t ARM_CTRL_SWAP     = 4; // control/tripwire: 2.0 cyc/vec
constexpr std::uint32_t ARM_RATE          = 5; // plain CountD1 slope partner
constexpr std::uint32_t ARM_RENDEZVOUS    = 6; // fold+store+order+read per segment
constexpr std::uint32_t ARM_BISECT        = 7; // resident K-th threshold search

constexpr bool IS_STREAM_ARM = (CGTCEQ_ARM <= ARM_CTRL_SWAP);
constexpr bool IS_FILL_ARM   = (CGTCEQ_ARM == ARM_RENDEZVOUS || CGTCEQ_ARM == ARM_BISECT);

// Tiles unpacked into Dst during INIT for the self-checking arms.
constexpr std::uint32_t FILL_TILES = (CGTCEQ_ARM == ARM_BISECT) ? BISECT_ROWS : 2;

// ---------------------------------------------------------------------------
// LReg map. Chosen so the fold's SFPTRANSP groups work out:
// SFPTRANSP transposes {LREG0..3} and {LREG4..7} independently, so each
// accumulator owns one group and its three partners are zeroed before a fold
// (the thresholds/ping-pong registers live in the partner slots and are
// reloaded after every fold -- that reload is part of the restart cost a real
// decision pays anyway).
//
// Explicitly ckernel::-qualified: this block sits above the LLK includes
// (sfpu_count_above_perf.cpp discipline).
// ---------------------------------------------------------------------------
constexpr std::uint32_t L_ACC1 = ckernel::p_sfpu::LREG0; // -(Cgt) accumulator
constexpr std::uint32_t L_THR  = ckernel::p_sfpu::LREG1; // threshold T
constexpr std::uint32_t L_A    = ckernel::p_sfpu::LREG2; // ping (mask vs T)
constexpr std::uint32_t L_B    = ckernel::p_sfpu::LREG3; // pong (mask vs T)
constexpr std::uint32_t L_ACC2 = ckernel::p_sfpu::LREG4; // -(Cge) accumulator
constexpr std::uint32_t L_THR2 = ckernel::p_sfpu::LREG5; // predecessor key
constexpr std::uint32_t L_A2   = ckernel::p_sfpu::LREG6; // ping (mask vs T2)
constexpr std::uint32_t L_B2   = ckernel::p_sfpu::LREG7; // pong (mask vs T2)

constexpr std::uint32_t SFPGT_MOD1_SET_VD              = 8; // SFPGT.md:53
constexpr std::uint32_t SFPIADD_MOD1_ARG_LREG_DST      = 0; // SFPIADD.md:48
constexpr std::uint32_t SFPIADD_MOD1_ARG_2SCOMP_LREG_D = 2; // SFPIADD.md:50
constexpr std::uint32_t SFPIADD_MOD1_CC_NONE           = 4; // SFPIADD.md:52
constexpr std::uint32_t SFPENCC_MOD1_EI                = 2; // SFPENCC.md:41
constexpr std::uint32_t SFPCFG_IMM16_IS_VALUE          = 1; // SFPCONFIG.md:108

// SFPSHFT2 mode 3: rotate right by one within each 8-lane sub-vector.
// Spelled out the same way ckernel_sfpu_binary_bcast.h:145 does.
constexpr std::uint32_t SFPSHFT2_MOD1_SUBVEC_SHFLROR1 = 3;

// Dst walk: one SFPLOAD covers 4 rows x 8 even-or-odd columns = 32 datums and
// the hardware advance per load is 2 u10-Addr units (SFPLOAD.md:86-107).
// ADDR_MOD_6 is the slot the SFPU unary path itself uses for this walk;
// ADDR_MOD_7 is the invariant {0,0,0} from the SFPU init.
constexpr std::uint32_t ADDR_MOD_WALK       = ckernel::ADDR_MOD_6;
constexpr std::uint32_t DST_ADDR_PER_LOAD   = 2;
constexpr std::uint32_t VECTORS_PER_TILE    = 32; // 32 loads x 32 datums = 1024
constexpr std::uint32_t TILE_ADDR_SPAN      = 64; // u10 Addr units per 32x32 fp32 tile
constexpr std::uint32_t VECTORS_PER_SEGMENT = 64; // rendezvous segment (= PassSync)

// ---------------------------------------------------------------------------
// Macros. Macro index 0 = Load + Simple(SFPGT vs L_THR); macro index 1 =
// Load + Simple(SFPGT vs L_THR2). Byte layout and rationale copied verbatim
// from sfpu_count_above_perf.cpp / sfpu_count_above_test.cpp (measured 1.998
// cyc/vec, correctness-proven 15/15):
//   0x80        -> Insn.VB = macroVD (compare the LOADED datum, mandatory)
//   0x40 clear  -> Insn.VD = macroVD (mask lands in the loaded register)
//   delay 1     -> fires alongside slot k+2, always a Load slot in these
//                  bodies, dodging the silent software-Simple discard
//                  (SFPLOADMACRO.md:149)
//   selector 4/5 -> InstructionTemplate[0]/[1] (SFPLOADMACRO.md:83)
// ---------------------------------------------------------------------------
constexpr std::uint32_t SEQ_GT_T   = 0x80u | (1u << 3) | 4u; // 0x8C, template[0]
constexpr std::uint32_t SEQ_GT_T2  = 0x80u | (1u << 3) | 5u; // 0x8D, template[1]
constexpr std::uint32_t SEQUENCE_0 = SEQ_GT_T;               // Simple byte only
constexpr std::uint32_t SEQUENCE_1 = SEQ_GT_T2;

// Misc: UnitDelayKind bit 8 = Simple counts SFPU issues, not wall cycles
// (mandatory -- a frontend bubble under cycle-counting slides the SFPGT onto
// a software Simple slot and the software instruction is silently discarded).
constexpr std::uint32_t MISC_WORD = 0x100;

// SFPLOADMACRO field packing (ckernel_ops.h:683, SFPLOADMACRO.md:20-26,45):
//   lreg_ind      = (MacroIndex << 2) | (VD & 3)
//   dest_reg_addr = (Imm << 1)... -- VD bit 2 rides in dest_reg_addr bit 0.
#define LOADMACRO0(vd, addr_mod, off) TTI_SFPLOADMACRO((0u << 2) | ((vd) & 3u), ckernel::InstrModLoadStore::INT32, (addr_mod), (off) | ((vd) >> 2))
#define LOADMACRO1(vd, addr_mod, off) TTI_SFPLOADMACRO((1u << 2) | ((vd) & 3u), ckernel::InstrModLoadStore::INT32, (addr_mod), (off) | ((vd) >> 2))

// Replay-buffer layout (32 slots per thread): the single-count body lives at
// [0,4); the dual-count body at [4,12). Both are recorded in INIT for arms
// that need both (bisect); other arms record only what they run.
constexpr std::uint32_t RB_SINGLE     = 0;
constexpr std::uint32_t RB_SINGLE_LEN = 4;
constexpr std::uint32_t RB_DUAL       = 4;
constexpr std::uint32_t RB_DUAL_LEN   = 8;

// MOP loop_count is 7 bits: count <= 128 per ckernel_unpack_template::run.
constexpr std::uint32_t MOP_MAX_ITERS = 128;

constexpr std::uint32_t SINGLE_PASSES_PER_TILE = VECTORS_PER_TILE / 2; // 16
constexpr std::uint32_t DUAL_PASSES_PER_TILE   = VECTORS_PER_TILE / 2; // 16 (4 loads/pass)

// Rendezvous segmentation.
constexpr std::uint32_t SEGMENTS           = ITER_COUNT / VECTORS_PER_SEGMENT;
constexpr std::uint32_t PASSES_PER_SEGMENT = VECTORS_PER_SEGMENT / 2; // 32

static_assert(ITER_COUNT % 2 == 0, "ping-pong body covers two vectors");
static_assert(ITER_COUNT % VECTORS_PER_SEGMENT == 0, "whole segments only");
static_assert(PASSES_PER_SEGMENT <= MOP_MAX_ITERS, "segment exceeds MOP ceiling");
static_assert(BISECT_ROWS >= 1 && BISECT_ROWS <= 3, "BISECT_ROWS+scratch must fit the 4-tile 32-bit SyncHalf Dest");

// Scratch Dst location for the folded counts (rendezvous arm): the 4-row
// group at u10 Addr 128 = physical rows 128..131 = the start of tile 2. The
// counted data lives in tiles 0..1, so the store never clobbers it.
constexpr std::uint32_t RDV_SCRATCH_ADDR = 128;
constexpr std::uint32_t RDV_MMIO_BASE    = 128 * 16; // 16 32-bit words per Dst row

// Bisection scratch = the tile after the data rows.
constexpr std::uint32_t BIS_SCRATCH_TILE = BISECT_ROWS;
constexpr std::uint32_t BIS_MMIO_BASE    = BIS_SCRATCH_TILE * 64 * 16;

// Sentinel for SYNC_PRIM 2. Impossible for a non-negative count. The MMIO
// write/read conversions (2's-comp <-> sign-magnitude) are symmetric, so the
// RISC reads back exactly what it wrote until the SFPSTORE lands.
constexpr std::uint32_t DST_SENTINEL = 0xFFFFFFFFu;

// Bounded polls so a wrong lane-map model or a dead semaphore reads out as a
// flagged, checkable failure instead of a device hang.
constexpr std::uint32_t POLL_LIMIT = 1u << 22;

// Rendezvous/bisect diagnostics block (written by TRISC1 straight into L1 at
// buffer_Res[0], outside every timed zone). Word layout is mirrored by the
// python driver.
constexpr std::uint32_t DIAG_MAGIC = 0xC67C0DE1u;

// pc_buf-visible semaphore for SYNC_PRIM 1. UNPACK_MATH_DONE is documented
// "for recording perf events and inserting delay ... only for either unpack
// or math" (ckernel_structs.h:21-23); the perf-zone barrier uses FPU_SFPU and
// UNPACK_TO_DEST (counters.h:531-532), and no LLK path this kernel calls
// touches index 6 -- so it is free here, on the math thread only.
constexpr std::uint8_t RDV_SEM = ckernel::semaphore::UNPACK_MATH_DONE;

} // namespace

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"

using namespace ckernel;

namespace
{

// --- macro configuration (must run after _llk_math_eltwise_unary_sfpu_init_once_,
// --- which clears LaneConfig and thereby enables the VD>=12 backdoor) -------
inline void configure_macro0_gt()
{
    TTI_SFPGT(0, L_THR, 12, SFPGT_MOD1_SET_VD); // template[0]
    TTI_SFPCONFIG(SEQUENCE_0, 4 + 0, SFPCFG_IMM16_IS_VALUE);
    TTI_SFPCONFIG(MISC_WORD, 8, SFPCFG_IMM16_IS_VALUE);
    TTI_SFPNOP;
    TTI_SFPNOP;
}

inline void configure_macro1_gt2()
{
    TTI_SFPGT(0, L_THR2, 13, SFPGT_MOD1_SET_VD); // template[1]
    TTI_SFPCONFIG(SEQUENCE_1, 4 + 1, SFPCFG_IMM16_IS_VALUE);
    TTI_SFPCONFIG(MISC_WORD, 8, SFPCFG_IMM16_IS_VALUE);
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// --- replay bodies ----------------------------------------------------------
// Single strict count, two vectors per pass (the proven CountD1 shape, only
// the register numbers differ from sfpu_count_above_test.cpp):
//   0 LOADMACRO0 -> L_A   (SFPGT vs T fires at slot 2)
//   1 SFPIADD L_ACC1 += L_B
//   2 LOADMACRO0 -> L_B   (fires at next slot 0)
//   3 SFPIADD L_ACC1 += L_A
inline void record_single_body()
{
    load_replay_buf<NoExec>(
        RB_SINGLE,
        RB_SINGLE_LEN,
        []
        {
            LOADMACRO0(L_A, ADDR_MOD_WALK, 0);
            TTI_SFPIADD(0, L_B, L_ACC1, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
            LOADMACRO0(L_B, ADDR_MOD_WALK, 0);
            TTI_SFPIADD(0, L_A, L_ACC1, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
        });
}

// Dual strict count: each vector is loaded twice -- once compared against T
// (macro 0) and once against pred(T) (macro 1). Only the second load advances
// the Dst walk. Two vectors per pass, eight instructions, ~4 cyc/vec:
//   0 LOADMACRO0 -> L_A  (no advance;  SFPGT vs T  fires slot 2, writes L_A)
//   1 SFPIADD L_ACC1 += L_B    (prev vector's T mask)
//   2 LOADMACRO1 -> L_A2 (advance;     SFPGT vs T2 fires slot 4, writes L_A2)
//   3 SFPIADD L_ACC2 += L_B2   (prev vector's T2 mask)
//   4 LOADMACRO0 -> L_B  (no advance;  fires slot 6, writes L_B)
//   5 SFPIADD L_ACC1 += L_A
//   6 LOADMACRO1 -> L_B2 (advance;     fires next slot 0, writes L_B2)
//   7 SFPIADD L_ACC2 += L_A2
// Scheduled SFPGTs land only on Load-slot cycles; software SFPIADDs sit at
// odd slots; no same-cycle double-write on any LReg (checked pairwise).
inline void record_dual_body()
{
    load_replay_buf<NoExec>(
        RB_DUAL,
        RB_DUAL_LEN,
        []
        {
            LOADMACRO0(L_A, ckernel::ADDR_MOD_7, 0);
            TTI_SFPIADD(0, L_B, L_ACC1, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
            LOADMACRO1(L_A2, ADDR_MOD_WALK, 0);
            TTI_SFPIADD(0, L_B2, L_ACC2, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
            LOADMACRO0(L_B, ckernel::ADDR_MOD_7, 0);
            TTI_SFPIADD(0, L_A, L_ACC1, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
            LOADMACRO1(L_B2, ADDR_MOD_WALK, 0);
            TTI_SFPIADD(0, L_A2, L_ACC2, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
        });
}

inline void program_mop_single()
{
    ckernel_unpack_template::lA(lltt::replay_insn(RB_SINGLE, RB_SINGLE_LEN), TT_OP_NOP).program();
}

inline void program_mop_dual()
{
    ckernel_unpack_template::lA(lltt::replay_insn(RB_DUAL, RB_DUAL_LEN), TT_OP_NOP).program();
}

// --- exact single count of the tile the Dst walk currently points at --------
// Prologue/epilogue close the 1-deep software pipeline (sfpu_count_above_test
// count_one_tile, verbatim): without them the count is off by one vector.
inline void count_tile_single(const std::uint32_t thr_bits)
{
    ckernel::sfpu::_sfpu_load_imm32_(L_THR, thr_bits);
    ckernel::sfpu::_sfpu_load_imm32_(L_ACC1, 0);
    ckernel::sfpu::_sfpu_load_imm32_(L_B, 0); // prologue
    math::clear_dst_reg_addr();
    ckernel_unpack_template::run(SINGLE_PASSES_PER_TILE);
    TTI_SFPNOP; // let the last scheduled SFPGT land
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPIADD(0, L_B, L_ACC1, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE); // epilogue
}

// Dual (Cgt, Cge) count of the current tile.
inline void count_tile_dual(const std::uint32_t thr_bits, const std::uint32_t thr2_bits)
{
    ckernel::sfpu::_sfpu_load_imm32_(L_THR, thr_bits);
    ckernel::sfpu::_sfpu_load_imm32_(L_THR2, thr2_bits);
    ckernel::sfpu::_sfpu_load_imm32_(L_ACC1, 0);
    ckernel::sfpu::_sfpu_load_imm32_(L_ACC2, 0);
    ckernel::sfpu::_sfpu_load_imm32_(L_B, 0); // prologue x2
    ckernel::sfpu::_sfpu_load_imm32_(L_B2, 0);
    math::clear_dst_reg_addr();
    ckernel_unpack_template::run(DUAL_PASSES_PER_TILE);
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPIADD(0, L_B, L_ACC1, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE); // epilogue x2
    TTI_SFPIADD(0, L_B2, L_ACC2, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
}

// --- fold (software-issued, NEVER macro-hosted; see file header) ------------
// The 8-partial -> scalar broadcast within one transpose group: 7 x
// { rotate-right-1 within each 8-lane subvector; accumulate }. In-tree
// ROR1+SFPNOP idiom: ckernel_sfpu_binary_bcast.h:209-218 (line 217 proves
// in-place ROR1, VD == VB, is legal).
template <std::uint32_t r0, std::uint32_t r1>
inline void ror1_broadcast_sum()
{
    TTI_SFPSHFT2(0, r0, r1, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPIADD(0, r1, r0, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
    for (std::uint32_t i = 0; i < 6; ++i)
    {
        TTI_SFPSHFT2(0, r1, r1, SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPNOP;
        TTI_SFPIADD(0, r1, r0, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
    }
}

// Single-accumulator fold on the LREG0..3 group. Negates -(count) -> count,
// zeroes the accumulator's three transpose partners (thresholds/ping-pong --
// they are reloaded by the next probe anyway), then folds per `depth`:
//   depth 2: negate only (store raw 32 lane partials; RISC reads 64 words)
//   depth 1: + TRANSP + 3 IADD (8 partials in lanes 0-7; RISC reads 16 words)
//   depth 0: + 7x(ROR1 + SFPNOP + IADD) (scalar in lanes 0-7; RISC reads 1)
// NOTE: SFPTRANSP transposes BOTH register groups; the LREG4..7 group is
// dont-care in every single-fold caller.
template <std::uint32_t depth>
inline void fold_single()
{
    TTI_SFPIADD(0, p_sfpu::LCONST_0, L_ACC1, SFPIADD_MOD1_ARG_2SCOMP_LREG_D | SFPIADD_MOD1_CC_NONE);

    if constexpr (depth <= 1)
    {
        ckernel::sfpu::_sfpu_load_imm32_(L_THR, 0);
        ckernel::sfpu::_sfpu_load_imm32_(L_A, 0);
        ckernel::sfpu::_sfpu_load_imm32_(L_B, 0);
        TTI_SFPNOP; // let the last SFPLOADI retire before the transpose reads it
        TTI_SFPTRANSP(0, 0, 0, 0);
        TTI_SFPIADD(0, L_THR, L_ACC1, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(0, L_A, L_ACC1, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(0, L_B, L_ACC1, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
    }
    if constexpr (depth == 0)
    {
        ror1_broadcast_sum<L_ACC1, L_THR>();
    }
}

// Dual fold, full depth, for the certification step: BOTH accumulators under
// ONE SFPTRANSP (calling a per-group fold twice would transpose twice and the
// second TRANSP would scramble the first group's folded state).
inline void fold_dual_full()
{
    TTI_SFPIADD(0, p_sfpu::LCONST_0, L_ACC1, SFPIADD_MOD1_ARG_2SCOMP_LREG_D | SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0, p_sfpu::LCONST_0, L_ACC2, SFPIADD_MOD1_ARG_2SCOMP_LREG_D | SFPIADD_MOD1_CC_NONE);
    ckernel::sfpu::_sfpu_load_imm32_(L_THR, 0);
    ckernel::sfpu::_sfpu_load_imm32_(L_A, 0);
    ckernel::sfpu::_sfpu_load_imm32_(L_B, 0);
    ckernel::sfpu::_sfpu_load_imm32_(L_THR2, 0);
    ckernel::sfpu::_sfpu_load_imm32_(L_A2, 0);
    ckernel::sfpu::_sfpu_load_imm32_(L_B2, 0);
    TTI_SFPNOP; // let the last SFPLOADI retire before the transpose reads it
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPIADD(0, L_THR, L_ACC1, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0, L_A, L_ACC1, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0, L_B, L_ACC1, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0, L_THR2, L_ACC2, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0, L_A2, L_ACC2, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0, L_B2, L_ACC2, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
    ror1_broadcast_sum<L_ACC1, L_THR>();
    ror1_broadcast_sum<L_ACC2, L_THR2>();
}

// --- RISC-side MMIO Dst window ----------------------------------------------
// A function, not a namespace-scope pointer: reinterpret_cast is not a
// constant initializer and these kernels must not rely on static-init code.
inline volatile std::uint32_t* dst_mmio_ptr()
{
    return reinterpret_cast<volatile std::uint32_t*>(RISCV_DEST_START_ADDR);
}

// Words the RISC reads per fold depth (see file header; sums over the fixed
// scratch window are permutation-invariant because INIT zeroed every word the
// stores do not write).
constexpr std::uint32_t READ_WORDS = (FOLD_DEPTH == 0) ? 1 : (FOLD_DEPTH == 1) ? 16 : 64;

inline std::uint32_t read_count(const std::uint32_t mmio_base, std::uint32_t& flags)
{
    volatile std::uint32_t* const dst_mmio = dst_mmio_ptr();
    if constexpr (READ_WORDS == 1)
    {
        const std::uint32_t v = dst_mmio[mmio_base + R0_WORD];
        // A sentinel surviving into the read means the rendezvous failed.
        if (v == DST_SENTINEL)
        {
            flags |= 0x4;
            return 0;
        }
        return v;
    }
    else
    {
        std::uint32_t sum = 0;
        for (std::uint32_t w = 0; w < READ_WORDS; ++w)
        {
            sum += dst_mmio[mmio_base + w];
        }
        return sum;
    }
}

// --- ordering primitives ------------------------------------------------
// Issued between the SFPSTORE of the folded count and the RISC's MMIO read.
inline void rendezvous_order(const std::uint32_t sentinel_word, std::uint32_t& flags)
{
    if constexpr (SYNC_PRIM == 0)
    {
        tensix_sync(); // full drain -- the measured >= 25.1 cyc control
    }
    else if constexpr (SYNC_PRIM == 1)
    {
        // In-stream STALLWAIT(STALL_SYNC, WAIT_SFPU) orders the SEMPOST after
        // SFPU completion (which includes our SFPSTORE); the RISC then sees
        // the post through the pc_buf. Reset with SEMGET for the next use.
        t6_semaphore_post<p_stall::WAIT_SFPU>(RDV_SEM);
        std::uint32_t spins = 0;
        while (semaphore_read(RDV_SEM) == 0)
        {
            if (++spins > POLL_LIMIT)
            {
                flags |= 0x1;
                break;
            }
        }
        if ((flags & 0x1) == 0)
        {
            semaphore_get(RDV_SEM);
        }
    }
    else
    {
        // Sentinel poll: wait for the store itself, not the whole pipe. The
        // sentinel was pre-written by the RISC before the counting pass.
        volatile std::uint32_t* const dst_mmio = dst_mmio_ptr();
        std::uint32_t spins                    = 0;
        while (dst_mmio[sentinel_word] == DST_SENTINEL)
        {
            if (++spins > POLL_LIMIT)
            {
                flags |= 0x2;
                break;
            }
        }
    }
}

inline void arm_sentinel(const std::uint32_t sentinel_word)
{
    if constexpr (SYNC_PRIM == 2)
    {
        dst_mmio_ptr()[sentinel_word] = DST_SENTINEL;
    }
}

// --- 16-bit sign-magnitude key space (bisection) -----------------------------
// Monotone map to plain unsigned order and back:
//   -NaN(0xFFFF) -> 0x0000 ... -0(0x8000) -> 0x7FFF, +0(0x0000) -> 0x8000,
//   ... +NaN(0x7FFF) -> 0xFFFF.
inline std::uint32_t key_to_ord(const std::uint32_t k)
{
    return (k & 0x8000u) ? ((~k) & 0xFFFFu) : (k | 0x8000u);
}

inline std::uint32_t ord_to_key(const std::uint32_t m)
{
    return (m & 0x8000u) ? (m & 0x7FFFu) : ((~m) & 0xFFFFu);
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
    const auto& buffer_Res        = params.buffer_Res;
#endif
    constexpr DstSync DST_SYNC_MODE        = DstSync::SyncHalf;
    constexpr std::uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;

    // Diagnostics accumulated on the RISC, dumped after the timed zone.
    [[maybe_unused]] std::uint32_t diag_flags    = 0;
    [[maybe_unused]] std::uint32_t diag_checksum = 0;
    [[maybe_unused]] std::uint32_t diag_last     = 0;
    // Bring-up probe: raw MMIO Dst words captured right after the INIT fill
    // (diag[9..15]) so a mis-placed fill / wrong MMIO base reads out as data
    // instead of requiring another theory. Dumped with the diag block.
    [[maybe_unused]] std::uint32_t diag_probe[7] = {};
    // Bisection per-row records, staged on the stack (<= 3 rows x 8 words).
    [[maybe_unused]] std::uint32_t row_rec[BISECT_ROWS][8] = {};

    {
        START_PERF_MEASURE("INIT")

        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
            num_faces, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

        // Establishes the SFPU config register, the ADDR_MOD_7 = {0,0,0}
        // invariant, and clears LaneConfig (precondition for the backdoor
        // template writes). Do not reorder.
        _llk_math_eltwise_unary_sfpu_init_once_();

        // Hardware Dst advance for the replayed bodies (recorded instruction
        // words are immutable; the walk cannot use sfpi::dst_reg++).
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = DST_ADDR_PER_LOAD},
        }
            .set(ADDR_MOD_WALK);

        // Clear stale lane predication: SFPGT's SET_VD write is gated on
        // LaneEnabled; a stale mask is a silent per-lane undercount.
        TTI_SFPENCC(0, 0, 0, SFPENCC_MOD1_EI);

        ckernel::sfpu::_sfpu_load_imm32_(L_THR, THR_BITS);
        ckernel::sfpu::_sfpu_load_imm32_(L_ACC1, 0);

        if constexpr (CGTCEQ_ARM == ARM_STREAM_SINGLE || CGTCEQ_ARM == ARM_RATE || CGTCEQ_ARM == ARM_RENDEZVOUS)
        {
            configure_macro0_gt();
            record_single_body();
            program_mop_single();
        }
        else if constexpr (CGTCEQ_ARM == ARM_STREAM_DUAL)
        {
            configure_macro0_gt();
            configure_macro1_gt2();
            ckernel::sfpu::_sfpu_load_imm32_(L_THR2, THR2_BITS);
            ckernel::sfpu::_sfpu_load_imm32_(L_ACC2, 0);
            record_dual_body();
            program_mop_dual();
        }
        else if constexpr (CGTCEQ_ARM == ARM_BISECT)
        {
            configure_macro0_gt();
            configure_macro1_gt2();
            record_single_body();
            record_dual_body();
            program_mop_single(); // interior probes; certification reprograms
        }
        else if constexpr (CGTCEQ_ARM == ARM_CTRL_LOAD)
        {
            // CONTROL -- frontend floor. Plain loads, deliberately NOT
            // SFPLOADMACRO (a macro issue against stale LoadMacroConfig from a
            // previously-run kernel is undefined; sfpu_count_above_perf.cpp
            // documents the resulting non-deterministic hang).
            load_replay_buf<NoExec>(
                RB_SINGLE,
                2,
                []
                {
                    TTI_SFPLOAD(L_A, ckernel::InstrModLoadStore::INT32, ADDR_MOD_WALK, 0);
                    TTI_SFPLOAD(L_B, ckernel::InstrModLoadStore::INT32, ADDR_MOD_WALK, 0);
                });
            ckernel_unpack_template::lA(lltt::replay_insn(RB_SINGLE, 2), TT_OP_NOP).program();
        }
        else if constexpr (CGTCEQ_ARM == ARM_CTRL_SWAP)
        {
            // CONTROL / tripwire: SFPSWAP is documented 2 backend cycles with
            // a non-fillable bubble; must read ~2.0x ctrl_load or the run is
            // invalid.
            load_replay_buf<NoExec>(
                RB_SINGLE,
                2,
                []
                {
                    TTI_SFPSWAP(0, L_A, L_B, p_sfpswap::ALL_ROWS_MAX);
                    TTI_SFPSWAP(0, L_B, L_A, p_sfpswap::ALL_ROWS_MAX);
                });
            ckernel_unpack_template::lA(lltt::replay_insn(RB_SINGLE, 2), TT_OP_NOP).program();
        }

        if constexpr (IS_FILL_ARM)
        {
            // Pull FILL_TILES tiles of KNOWN stimulus into Dst (unpack did the
            // matching _llk_unpack_A_ calls in its own INIT; with
            // unpack_to_dest the datacopy is pure synchronization).
            _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();
            for (std::uint32_t t = 0; t < FILL_TILES; ++t)
            {
                _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC_MODE, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                    t, formats.math, formats.math);
            }

            // Program this thread's RISC_DEST_ACCESS_CTRL section for MMIO Dst
            // reads (fmt INT32: Dst32b[512][16], sign-mag -> 2's-comp on load;
            // identity on the non-negative counts stored here). Recipe from
            // tests/helpers/include/dprint_tensix.h:54-62.
            configure_dest_access<MathThreadId>(DataFormat::Int32);
            tensix_sync();
            wait(1000);

            // Zero the scratch window so partial-width stores leave summable
            // rows (both column parities of the 4-row group(s)).
            ckernel::sfpu::_sfpu_load_imm32_(L_B, 0);
            if constexpr (CGTCEQ_ARM == ARM_RENDEZVOUS)
            {
                // Point the math-side Dst offset at tile 0 ONCE (the INIT
                // datacopy leaves the offset wherever it last wrote; without
                // this the walk would not start at physical row 0 and the
                // MMIO window model breaks). It stays put for the whole arm.
                _llk_math_eltwise_sfpu_start_(0);
                math::clear_dst_reg_addr();
                TTI_SFPSTORE(L_B, InstrModLoadStore::INT32, ADDR_MOD_7, RDV_SCRATCH_ADDR);
                TTI_SFPSTORE(L_B, InstrModLoadStore::INT32, ADDR_MOD_7, RDV_SCRATCH_ADDR + 2);
            }
            else
            {
                _llk_math_eltwise_sfpu_start_(BIS_SCRATCH_TILE);
                math::clear_dst_reg_addr();
                TTI_SFPSTORE(L_B, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
                TTI_SFPSTORE(L_B, InstrModLoadStore::INT32, ADDR_MOD_7, 2);
                TTI_SFPSTORE(L_B, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
                TTI_SFPSTORE(L_B, InstrModLoadStore::INT32, ADDR_MOD_7, 6);
            }
            tensix_sync();

            // Bring-up probe (outside every timed zone). [0]/[1]: raw MMIO
            // window words (established: full-width data words read as two
            // packed 16-bit halves -- the window's half-plane pairing differs
            // from the engine layout; small counts live in the lo half and
            // read fine). [2..5]: SFPU-side per-tile counts, twin-style
            // (sfpu_start(t) + 16-pass walk), which probe what the WALK sees:
            //   [2] count(tile 0, thr = -0.0)  -- expect 1024 (all positives)
            //   [3] count(tile 1, thr = -0.0)  -- expect 0 (all negatives);
            //                                     1024 would mean tile 1 reads
            //                                     as zeros to the SFPU
            //   [4] count(tiles 0..1 pinned-offset 32-pass walk, thr = -0.0)
            //                                  -- expect 1024
            //   [5] count(tile 1, thr = THR_BITS) -- expect 1023 (all but min)
            //   [6] scratch word after zeroing -- expect 0
            if constexpr (CGTCEQ_ARM == ARM_RENDEZVOUS)
            {
                {
                    volatile std::uint32_t* const p = dst_mmio_ptr();
                    diag_probe[0]                   = p[0 * 16 + 0];
                    diag_probe[1]                   = p[64 * 16 + 0];
                    diag_probe[6]                   = p[RDV_MMIO_BASE + R0_WORD];
                }
                auto probe_count = [&](const std::uint32_t tile, const std::uint32_t thr_bits, const bool two_tiles) -> std::uint32_t
                {
                    _llk_math_eltwise_sfpu_start_(tile);
                    ckernel::sfpu::_sfpu_load_imm32_(L_THR, thr_bits);
                    ckernel::sfpu::_sfpu_load_imm32_(L_ACC1, 0);
                    ckernel::sfpu::_sfpu_load_imm32_(L_B, 0);
                    math::clear_dst_reg_addr();
                    ckernel_unpack_template::run(two_tiles ? PASSES_PER_SEGMENT : SINGLE_PASSES_PER_TILE);
                    TTI_SFPNOP;
                    TTI_SFPNOP;
                    TTI_SFPNOP;
                    TTI_SFPIADD(0, L_B, L_ACC1, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
                    fold_single<0>();
                    _llk_math_eltwise_sfpu_start_(0);
                    math::clear_dst_reg_addr();
                    TTI_SFPSTORE(L_ACC1, InstrModLoadStore::INT32, ADDR_MOD_7, RDV_SCRATCH_ADDR);
                    tensix_sync();
                    return dst_mmio_ptr()[RDV_MMIO_BASE + R0_WORD];
                };
                diag_probe[2] = probe_count(0, 0x80000000u, false);
                diag_probe[3] = probe_count(1, 0x80000000u, false);
                diag_probe[4] = probe_count(0, 0x80000000u, true);
                diag_probe[5] = probe_count(1, THR_BITS, false);
                // Re-zero the scratch word the probes dirtied.
                ckernel::sfpu::_sfpu_load_imm32_(L_B, 0);
                math::clear_dst_reg_addr();
                TTI_SFPSTORE(L_B, InstrModLoadStore::INT32, ADDR_MOD_7, RDV_SCRATCH_ADDR);
                TTI_SFPSTORE(L_B, InstrModLoadStore::INT32, ADDR_MOD_7, RDV_SCRATCH_ADDR + 2);
                tensix_sync();
            }

            if constexpr (SYNC_PRIM == 1)
            {
                // Drain any residual tokens a previous kernel left behind.
                while (semaphore_read(RDV_SEM) > 0)
                {
                    semaphore_get(RDV_SEM);
                }
            }
        }

        PROFILER_SYNC();
    }

    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (IS_STREAM_ARM)
        {
            // ------------------------------------------------- additivity ---
            // topk_negfilter_perf.cpp pipeline shape: TILE_CNT tiles streamed
            // by unpack (except under MATH_ISOLATE), one count body per tile.
            auto run_math_body = [&](const std::uint32_t block_tile)
            {
                if constexpr (CGTCEQ_ARM == ARM_STREAM_NONE)
                {
                    // Floor: no SFPU work at all.
                }
                else if constexpr (CGTCEQ_ARM == ARM_STREAM_SINGLE)
                {
                    _llk_math_eltwise_sfpu_start_(block_tile);
                    ckernel::sfpu::_sfpu_load_imm32_(L_B, 0);
                    math::clear_dst_reg_addr();
                    ckernel_unpack_template::run(SINGLE_PASSES_PER_TILE);
                    TTI_SFPNOP;
                    TTI_SFPNOP;
                    TTI_SFPNOP;
                    TTI_SFPIADD(0, L_B, L_ACC1, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
                    _llk_math_eltwise_sfpu_done_();
                }
                else if constexpr (CGTCEQ_ARM == ARM_STREAM_DUAL)
                {
                    _llk_math_eltwise_sfpu_start_(block_tile);
                    ckernel::sfpu::_sfpu_load_imm32_(L_B, 0);
                    ckernel::sfpu::_sfpu_load_imm32_(L_B2, 0);
                    math::clear_dst_reg_addr();
                    ckernel_unpack_template::run(DUAL_PASSES_PER_TILE);
                    TTI_SFPNOP;
                    TTI_SFPNOP;
                    TTI_SFPNOP;
                    TTI_SFPIADD(0, L_B, L_ACC1, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
                    TTI_SFPIADD(0, L_B2, L_ACC2, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
                    _llk_math_eltwise_sfpu_done_();
                }
                else
                {
                    // ctrl_load / ctrl_swap: 16 replay passes of the 2-deep
                    // control body per tile.
                    _llk_math_eltwise_sfpu_start_(block_tile);
                    math::clear_dst_reg_addr();
                    ckernel_unpack_template::run(SINGLE_PASSES_PER_TILE);
                    TTI_SFPNOP;
                    TTI_SFPNOP;
                    TTI_SFPNOP;
                    _llk_math_eltwise_sfpu_done_();
                }
            };

            if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
            {
                // Unpack does nothing here, so math must not call the
                // datacopy handshake. The stimulus is whatever is in Dest --
                // the right trade for an issue-rate number.
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        run_math_body(block_tile);
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
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC_MODE, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
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
        }
        else if constexpr (CGTCEQ_ARM == ARM_RATE)
        {
            // ------------------------------------- plain-rate slope partner --
            // The exact sfpu_count_above ARM_COUNT_D1 shape (throughput only:
            // the stated 1-vector pipeline bias applies, and the walk wraps
            // Dst freely). Everything the rendezvous arm adds is measured as
            // a slope DELTA against this arm.
            constexpr std::uint32_t PASSES    = ITER_COUNT / 2;
            constexpr std::uint32_t FULL_RUNS = PASSES / MOP_MAX_ITERS;
            constexpr std::uint32_t REM       = PASSES % MOP_MAX_ITERS;
            for (std::uint32_t i = 0; i < FULL_RUNS; ++i)
            {
                ckernel_unpack_template::run(MOP_MAX_ITERS);
            }
            if constexpr (REM > 0)
            {
                ckernel_unpack_template::run(REM);
            }
            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;
        }
        else if constexpr (CGTCEQ_ARM == ARM_RENDEZVOUS)
        {
            // ------------------------------------------ fold + readback -----
            // SEGMENTS data-dependent decisions. Every segment counts tiles
            // 0..1 (the walk is rewound each restart), so the exact expected
            // count sequence is host-computable: the python driver simulates
            // this same automaton and any deviation fails the test.
            std::uint32_t thr = THR_BITS;
            for (std::uint32_t seg = 0; seg < SEGMENTS; ++seg)
            {
                arm_sentinel(RDV_MMIO_BASE + R0_WORD);

                // Restart: threshold reload, accumulator zero, pipeline
                // prologue, Dst rewind, MOP re-issue.
                ckernel::sfpu::_sfpu_load_imm32_(L_THR, thr);
                ckernel::sfpu::_sfpu_load_imm32_(L_ACC1, 0);
                ckernel::sfpu::_sfpu_load_imm32_(L_B, 0);
                math::clear_dst_reg_addr();
                ckernel_unpack_template::run(PASSES_PER_SEGMENT);
                TTI_SFPNOP;
                TTI_SFPNOP;
                TTI_SFPNOP;
                TTI_SFPIADD(0, L_B, L_ACC1, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);

                fold_single<FOLD_DEPTH>();

                // Store to the scratch 4-row group (never the counted data).
                math::clear_dst_reg_addr();
                TTI_SFPSTORE(L_ACC1, InstrModLoadStore::INT32, ADDR_MOD_7, RDV_SCRATCH_ADDR);

                rendezvous_order(RDV_MMIO_BASE + R0_WORD, diag_flags);

                const std::uint32_t c = read_count(RDV_MMIO_BASE, diag_flags);

                // The branch a real search makes: the next threshold is a
                // function of the count just read.
                diag_checksum ^= c + seg;
                diag_last = c;
                thr       = (c & 1u) ? THR2_BITS : THR_BITS;
            }
        }
        else if constexpr (CGTCEQ_ARM == ARM_BISECT)
        {
            // --------------------------------------------- bisection --------
            for (std::uint32_t row = 0; row < BISECT_ROWS; ++row)
            {
                const std::uint64_t t_start = read_wall_clock();

                std::uint32_t lo = 0, hi = 0xFFFFu;
                std::uint32_t decisions = 0;
                std::uint32_t exit_mode = 0; // 1 = CERT, 2 = VALIDSET
                std::uint32_t found_thr = 0, cgt = 0, ceq = 0;

                auto probe_single = [&](const std::uint32_t thr_bits) -> std::uint32_t
                {
                    arm_sentinel(BIS_MMIO_BASE + R0_WORD);
                    _llk_math_eltwise_sfpu_start_(row);
                    count_tile_single(thr_bits);
                    fold_single<FOLD_DEPTH>();
                    _llk_math_eltwise_sfpu_start_(BIS_SCRATCH_TILE);
                    math::clear_dst_reg_addr();
                    TTI_SFPSTORE(L_ACC1, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
                    rendezvous_order(BIS_MMIO_BASE + R0_WORD, diag_flags);
                    ++decisions;
                    return read_count(BIS_MMIO_BASE, diag_flags);
                };

                while (lo < hi)
                {
                    const std::uint32_t mid      = lo + ((hi - lo) >> 1);
                    const std::uint32_t thr_bits = ord_to_key(mid) << 16; // bf16 key -> fp32 pattern
                    const std::uint32_t c        = probe_single(thr_bits);
                    if (c == BISECT_K)
                    {
                        // Valid top-K set: exactly K elements strictly above.
                        exit_mode = 2;
                        found_thr = thr_bits;
                        cgt       = c;
                        ceq       = 0;
                        break;
                    }
                    if (c < BISECT_K)
                    {
                        hi = mid;
                    }
                    else
                    {
                        lo = mid + 1;
                    }
                }

                if (exit_mode == 0)
                {
                    // Certification: dual (Cgt, Cge) at the found key and its
                    // predecessor in the sign-magnitude order. m == 0 has no
                    // predecessor: Cge is then the whole row by definition.
                    const std::uint32_t m         = lo;
                    const std::uint32_t key       = ord_to_key(m);
                    found_thr                     = key << 16;
                    const std::uint32_t pred_bits = (m > 0) ? (ord_to_key(m - 1) << 16) : 0;

                    arm_sentinel(BIS_MMIO_BASE + R0_WORD);
                    program_mop_dual();
                    _llk_math_eltwise_sfpu_start_(row);
                    if (m > 0)
                    {
                        count_tile_dual(found_thr, pred_bits);
                    }
                    else
                    {
                        count_tile_dual(found_thr, found_thr); // Cge overwritten below
                    }
                    fold_dual_full(); // certification always full-folds
                    _llk_math_eltwise_sfpu_start_(BIS_SCRATCH_TILE);
                    math::clear_dst_reg_addr();
                    TTI_SFPSTORE(L_ACC1, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
                    TTI_SFPSTORE(L_ACC2, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
                    rendezvous_order(BIS_MMIO_BASE + R0_WORD, diag_flags);
                    ++decisions;
                    program_mop_single(); // restore for the next row

                    cgt               = dst_mmio_ptr()[BIS_MMIO_BASE + R0_WORD];
                    std::uint32_t cge = dst_mmio_ptr()[BIS_MMIO_BASE + 4 * 16 + R0_WORD];
                    if (m == 0)
                    {
                        cge = VECTORS_PER_TILE * 32; // no predecessor: everything counts
                    }
                    ceq       = cge - cgt;
                    exit_mode = 1;
                }

                const std::uint32_t cycles = static_cast<std::uint32_t>(read_wall_clock() - t_start);

                const bool invariant_ok = (exit_mode == 2) ? (cgt == BISECT_K) : (cgt < BISECT_K && BISECT_K <= cgt + ceq);

                row_rec[row][0] = found_thr;
                row_rec[row][1] = lo;
                row_rec[row][2] = cgt;
                row_rec[row][3] = ceq;
                row_rec[row][4] = decisions;
                row_rec[row][5] = exit_mode;
                row_rec[row][6] = cycles;
                row_rec[row][7] = invariant_ok ? 1u : 0u;
            }
        }

        PROFILER_SYNC();
    }

    if constexpr (IS_FILL_ARM)
    {
        // Both fill arms opened an SFPU section with _llk_math_eltwise_sfpu_start_.
        _llk_math_eltwise_sfpu_done_();
        // Deliberately NO _llk_math_dest_section_done_ here (bring-up root cause,
        // 2026-08-16): it posts semaphore::MATH_PACK (set_math_semaphores), and in
        // the fill arms the pack thread is idle -- nothing ever consumes the
        // token. The harness re-runs the same kernel run_count times WITHOUT any
        // semaphore reset, and the next run's _llk_math_pack_sync_init_ spins
        // `while (semaphore_read(MATH_PACK) > 0)` on the leaked token: math wedges
        // in INIT, unpack wedges at the fill's mailbox_read, and the leaked token
        // then poisons every subsequent test on the un-reset device (the observed
        // "waited 2 seconds for Math, Unpacker" cascade). The dest-half flip the
        // call would also do is re-established by the next run's
        // _llk_math_pack_sync_init_ (reset_dest_offset_id + StartZero), so
        // skipping it here is state-clean.

        // Diagnostics dump, outside every timed zone, straight into L1.
        volatile std::uint32_t* diag = reinterpret_cast<volatile std::uint32_t*>(buffer_Res[0]);
        diag[0]                      = DIAG_MAGIC;
        diag[1]                      = CGTCEQ_ARM;
        diag[2]                      = FOLD_DEPTH;
        diag[3]                      = SYNC_PRIM;
        diag[4]                      = (CGTCEQ_ARM == ARM_BISECT) ? BISECT_ROWS : SEGMENTS;
        diag[5]                      = BISECT_K;
        diag[6]                      = diag_checksum;
        diag[7]                      = diag_flags;
        diag[8]                      = diag_last;
        for (std::uint32_t i = 0; i < 7; ++i)
        {
            diag[9 + i] = diag_probe[i];
        }
        if constexpr (CGTCEQ_ARM == ARM_BISECT)
        {
            for (std::uint32_t r = 0; r < BISECT_ROWS; ++r)
            {
                for (std::uint32_t w = 0; w < 8; ++w)
                {
                    diag[16 + 8 * r + w] = row_rec[r][w];
                }
            }
        }
    }
}

#endif // LLK_TRISC_MATH

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
        _llk_unpack_A_init_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */,
            0 /* within_face_16x16_transpose */,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
            formats.unpack_A_src,
            formats.unpack_A_dst);

        if constexpr (IS_FILL_ARM)
        {
            // The known stimulus the self-checking arms count. Outside the
            // timed zone in every run type (pack_exp_histogram_perf idiom).
            for (std::uint32_t t = 0; t < FILL_TILES; ++t)
            {
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    L1_ADDRESS(buffer_A[t]), formats.unpack_A_src, formats.unpack_A_dst);
            }
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (IS_STREAM_ARM)
        {
            if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE && PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
            {
                constexpr std::uint32_t SRC_SLOTS = 16; // ring, must match the driver
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_unpack_A_<BroadcastType::NONE, false, reuse_dest_type, unpack_to_dest>(
                        L1_ADDRESS(buffer_A[i & (SRC_SLOTS - 1)]), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_UNPACK

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
    constexpr DstSync DST_SYNC_MODE        = DstSync::SyncHalf;
    constexpr std::uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;

    {
        START_PERF_MEASURE("INIT")
        if constexpr (IS_STREAM_ARM)
        {
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
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (IS_STREAM_ARM)
        {
            if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION || PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
            {
                constexpr std::uint32_t RES_SLOTS = 16; // ring, must match the driver
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
                // The RISC runs far ahead of the packer.
                TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK);
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_PACK
