// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"

using namespace ckernel;
using namespace ckernel::math;

// #43563: 1 = real-zero mm1 via ZEROACC CLR_SPECIFIC (writes data) on the non-mask path. VALIDATED
// PARTIAL FIX: improves per-shard PCC 0.990->0.9915 (confirms CLR_SPECIFIC writes real data and the
// SFPU was reading corrupting leftover), no hang — but does NOT fully close the iter-parity bug on its
// own (other FPU exact-0 writes, e.g. bcast_sub mm1-max=0, are still flag-optimized and read by the
// SFPU exp/reduce_sum). Part 1 of the #43563 fix; combine with NARROW_DST_ZEROFLAG_43563 (sdpa.h) which
// real-zeroes the dynamic bcast_sub exact-0. 0 = original ZEROACC CLR_16 (flag-only).
#define REALZERO_MM1_43563 0

// #43563 iter-parity fix: STALLWAIT(STALL_MATH, TRISC_CFG) after the DEST-offset SETC16 so it commits
// before the matmul's DEST writes (non-mask path had no such ordering). 1 = on.
#define FIX_DEST_OFFSET_ORDER_43563 0

// #43563 iter-parity experiment: a real STALLWAIT (not NOPs) at the VERY START of
// _llk_math_sdpa_custom_mm_mask_dest_, before the DEST-offset SETC16. 21+ MATH NOPs placed here fix the
// bug; this tests whether a STALLWAIT does the same. 0 = off.
//   1 = TTI_STALLWAIT(STALL_MATH, TRISC_CFG)  (drain pending CFG before the SETC16)
//   2 = TTI_STALLWAIT(STALL_MATH, MATH)       (drain the MATH pipe)
//   3 = TTI_STALLWAIT(STALL_MATH, SRCA_VLD|SRCB_VLD) (may deadlock)
#define FIX_MASKDEST_STALL_43563 0

// #43562/3 EXPERIMENT: instead of the SRC_VLD stallwait, run a pure-delay loop of N MATH NOPs at the
// SAME location (start of _llk_math_sdpa_custom_mm_mask_dest_, before the DEST-offset SETC16). Tests
// whether the shallow-position fix is just TIMING (a big enough delay fixes it) vs the stallwait itself.
// Set FIX_MASKDEST_STALL_43563=0 and FIX_FULL_VIA_MASK_43563=0 (scratch all fixes) to test in isolation.
// N = number of NOPs (0 = off).
#define NOPLOOP_MASKDEST_43563 0

// #43562/3 EXPERIMENT: in the mask branch (FIX_FULL_VIA_MASK=1), replace the coherent MOVB2D write with
// a pure NOP delay (N NOPs). NOTE: no DEST clear -> mm1 base stays uncleared (matmul accumulates onto
// stale), so this tests "delay in the mask-path CB-management context" but is confounded by the stale
// base. N = NOP count (0 = off). Takes precedence over FIX_MASK_MOVB2A / FIX_MASK_ZEROACC.
#define FIX_MASK_NOPLOOP_43563 0

// #43562/3 EXPERIMENT: in the mask branch, use flag-only ZEROACC instead of the coherent MOVB2D, keeping
// the mask SrcB unpack + STALLWAIT + terminal SETRWC(CLR_B) scaffolding. Isolates bank-switch vs MOVB2D.
// Requires FIX_FULL_VIA_MASK_43563=1 (route full chunks here). 1 = ZEROACC, 0 = original MOVB2D.
#define FIX_MASK_ZEROACC_43563 1

// #43562/3 EXPERIMENT: ZEROACC (clean DEST) + MOVB2A (SrcB->SrcA, the SrcB-consuming FPU op that does NOT
// write DEST). Discriminates SrcB-read-via-FPU vs MOVB2D's DEST-write. Requires FIX_FULL_VIA_MASK=1.
// 1 = ZEROACC+MOVB2A; 0 = off. (Takes precedence over FIX_MASK_ZEROACC.)
#define FIX_MASK_MOVB2A_43563 0

// #43563 SrcB read-bank reset (rmillerTT opt1, #842 family).
//
// Root cause (established at pos8190): chunk-1's QK^T mm1 produces bank-dependent output from
// provably-identical Q,K. The only intervening op is the PV matmul
// (_llk_math_sdpa_custom_mm_reuse_dest_srcb_), which manipulates SrcB via MOVD2B and exits with a
// single SETRWC CLR_B. That leaves the MATH-side SrcB read-bank pointer on a drifted/iteration-
// dependent bank, so the next QK reads SrcB(K) from the wrong bank -> bank-dependent mm1.
// The SRC_VLD stall (FIX_MASKDEST_STALL_43563=3) waits for *valid* but not for the *right bank*,
// so it does not fix this.
//
// Fix: at the QK matmul entry, reset the SrcB read bank to a known absolute state (bank 0 / cleared)
// via the SETRWC CLR_B form the LLK uses everywhere (== clear_bank_valid<SrcB>): touches only SrcB,
// leaves SrcA + DEST untouched.
//   1 = TTI_SETRWC(CLR_B, ..., SET_B)  reset SrcB read bank + valid to known state.
//       NOTE: form 1 DEADLOCKS at pos8190 (re-arms SET_B, desyncs vs the unpacker that produces
//       SrcB(K) -> MATH stalls forever -> host busy-spin). Do not use.
//   2 = TTI_SETRWC(CLR_B, ..., 0)      clear/advance SrcB read bank only, no SET bitmask
//       (the form generic matmuls use between ops; does not re-arm SrcB-valid).
//       RESULT (pos8190): form 2 also DEADLOCKS (host busy-spin, MATH stalls on SrcB never-valid).
// VERDICT: an extra CLR_B at QK entry flips the SrcB bank into a state where the QK MVMUL's
// SrcB(K) read never goes valid -> deadlock. Neither form fixes pos8190; both hang. Gated OFF.
// 0 = original (default).
#define FIX_QK_SRCB_BANK_RESET_43563 0

// #43562/3 DISCRIMINATOR experiment (default 0). Replace the QK^T MVMUL with an FPU ELWADD (Q+K) at the
// SAME place: SAME SrcA(Q)/SrcB(K) from the SAME unpack, SAME mm1 DEST region. UNPACK is unchanged,
// FIX_MASKDEST_STALL_43563=3 stays, the DEST-offset SETC16 + ZEROACC stay. Only the MATH consuming op
// changes from MVMUL -> ELWADD. Correctness of Q+K is IRRELEVANT; we only test whether the per-core op
// output still diverges iter1-vs-iter2 at pos8190. If ELWADD diverges too => bank-dependence is
// OP-INDEPENDENT (operands/setup); if ELWADD is bank-invariant => MVMUL-SPECIFIC. 0 = original MVMUL.
#define QK_ELWADD_43563 0

// #43562/3 SRC DISCRIMINATOR (default 0). Replace the QK^T MVMUL MATH op with a straight register->DEST
// copy of ONE Src register, at the SAME place, SAME unpack (Q->SrcA, K->SrcB unchanged), SAME mm1 DEST
// region. The op output then == the chosen Src register's contents (not Q@K, not Q+K). We test whether
// the DUMPED register diverges iter1-vs-iter2 at pos8190 to localize WHICH Src carries the bank-
// dependence. FIX_MASKDEST_STALL_43563=3 stays, the DEST-offset SETC16 + ZEROACC stay.
//   1 = dump SrcA (Q) via TTI_MOVA2D  (SrcA -> DEST) -> op output = Q as loaded into SrcA.
//   2 = dump SrcB (K) via TTI_MOVB2D  (SrcB -> DEST) -> op output = K as loaded into SrcB.
// MOVA2D MOV_8_ROWS copies one 8-row face; MOVB2D MOV_4_ROWS copies 4 rows. We walk the score-tile rows
// with DEST-only addr_mod increments (Src read offset is the `src` field; the unpack-produced register
// is read in place). 0 = original MVMUL.
//
// RESULT (pos8190, both clean builds): BOTH =1 (MOVA2D/Q) and =2 (MOVB2D/K) DEADLOCK on device (host
// blocks, no EXP_IN written, tt-smi -r to recover). Root cause: the QK MVMUL reads BOTH SrcA and SrcB
// every op, consuming both unpacker-produced data-valids; a single-Src MOV consumes only one, leaving
// the OTHER Src's dvalid unconsumed -> unpack/MATH handshake desyncs -> MATH stalls forever. Same
// failure family as FIX_QK_SRCB_BANK_RESET_43563 (SrcB never-consumed -> hang). So this in-place
// single-MOV swap cannot read out either Src without also consuming the other's dvalid; the dual-Src
// dvalid handshake would have to be preserved (e.g. issue a balancing consume of the non-dumped Src)
// for the dump to be viable. Gated OFF. 0 = original MVMUL.
#define QK_SRCDUMP_43563 0

// #43562/3 BALANCED DUAL-MOV SRC DISCRIMINATOR (default 0). The single-Src QK_SRCDUMP DEADLOCKS because
// the QK MVMUL reads BOTH SrcA+SrcB every op (consumes both unpacker dvalids); a lone MOV consumes only
// one, the other dvalid never clears -> MATH stalls forever. Fix: per face do BOTH a MOVA2D (SrcA=Q ->
// DEST tile0, the mm1 region) AND a MOVB2D (SrcB=K -> DEST tile1, mm1 region + 16 rows), so BOTH SrcA and
// SrcB dvalids are consumed each op slot (balanced handshake -> no deadlock). End-tile CLR_A clears SrcA,
// end-block CLR_AB clears both, mirroring the MVMUL Src lifetime exactly. UNPACK unchanged (Q->SrcA,
// K->SrcB), FIX_MASKDEST_STALL_43563=3 stays, the DEST-offset SETC16 + ZEROACC stay. op output: tile0 ==
// Q as loaded into SrcA, tile1 == K as loaded into SrcB (NOT Q@K). Host taps tile0 (per_core_QDUMP) and
// tile1 (per_core_KDUMP) and diffs iter1-vs-iter2 to localize WHICH Src carries the bank-dependence.
// 1 = enabled. 0 = original MVMUL.
//
// RESULT (pos8190, 3 clean builds, 2026-06-08): the {MOVA2D,MOVB2D}-pair-per-MVMUL substitution DEADLOCKS
// on device (MATH stalls forever at op launch, host busy-spin, tt-smi -r to recover) across THREE balanced
// variants: (a) even A/B pairing with MOV_8_ROW_BRCST, (b) 1:1 pair-per-MVMUL with MOV_8_ROW_BRCST, and
// (c) 1:1 pairs with MOV_4_ROWS + a matmul-faithful clear cadence (SrcA cleared per tile via the MOVA2D
// addr_mod, SrcB reused across width and cleared only at block end via the MOVB2D addr_mod). The
// MOVA2D/MOVB2D `src_mask` (0x1 / 0x2) does gate the right Src each, but splitting one MVMUL (src_mask
// 0x3, single clear_dvalid field) into two MOV ops desyncs the custom-matmul UNPACKER MOP
// (ckernel_unpack_template with UNPACR_A1/2/3 SrcB-reuse), whose UNPACR cadence is tightly coupled to the
// ONE-MVMUL-per-tile structure and to MVMUL's dedicated clear_dvalid bank advance -- the doubled MATH op
// count / addr_mod-only clears cannot be made to match. Same deadlock family as QK_SRCDUMP and
// FIX_QK_SRCB_BANK_RESET. So an in-MOP MOV-based dual Src readout is NOT viable here; reading SrcA/SrcB
// out separately would need a non-MOP path (e.g. drain the unpacker first, then MOV out of the settled
// Src banks). Left gated OFF. 0 = original MVMUL.
#define QK_DUALMOV_43563 0

// #43562/3 NON-MASK -> MASK MIMIC (default 0). The MASK branch is bank-deterministic, the NON-MASK
// (full-chunk) branch is bank-dependent. Static diff of the two branches in
// _llk_math_sdpa_custom_mm_mask_dest_ shows the MASK branch, after writing DEST, ends with
//   TTI_SETRWC(CLR_B, 0,0,0,0, SET_ABD)
// which clears SrcB-valid AND resets ALL RWC counters (SrcA/SrcB/DEST read+write pointers) to a known
// absolute state right before the QK matmul runs. The NON-MASK branch ends with only a flag-only
// ZEROACC and leaves the RWC counters wherever the prior PV block-end CLR_AB left them -> the QK MVMUL
// reads SrcB(K) from a drifted/iteration-dependent read bank (the established #842 SrcB-bank-drift root
// cause). FIX_QK_SRCB_BANK_RESET (a CLR_B BEFORE the SETC16/ZEROACC) deadlocked; this instead replicates
// the mask path EXACTLY: the CLR_B + SET_ABD comes AFTER the DEST zero-write, mirroring the mask branch
// line-for-line, so the upcoming matmul unpack re-arms SrcB-valid as it already does on the mask path.
// RESULTS (pos4351, clean builds, 2026-06-08; baseline iter1-vs-iter2 = 6079/7168, max|d|=0.0547):
//   1 = full mask-path SETRWC (CLR_B + SET_ABD) -> HANGS (SrcB-clear desyncs the non-mask unpack/MATH
//       handshake; same deadlock family as FIX_QK_SRCB_BANK_RESET / the MOVB2D-in-non-mask attempts).
//   2 = RWC counter/pointer reset only (SET_ABD, no CLR_B) -> NO-CHANGE: 6079/7168, max|d|=0.0547,
//       PCC-vs-golden 0.9936 (sane). The counter/pointer reset is NOT the determinism carrier.
// VERDICT: the mask path's terminal SETRWC is NOT the carrier. Its safe half (SET_ABD) is a no-op and its
// SrcB-clear half can't be applied to the non-mask path without hanging. The remaining (unreplicable)
// difference is the mask branch's COHERENT MOVB2D real-data DEST write through the FPU datapath, which
// requires a valid SrcB the non-mask path does not have at that point (every faithful MOVB2D attempt hangs).
// 1 = full mask SETRWC; 2 = SET_ABD-only counter reset; 0 = original (default).
#define FIX_NONMASK_LIKE_MASK_43563 0

#if (defined(QK_SRCDUMP_43563) && QK_SRCDUMP_43563) + (defined(QK_ELWADD_43563) && QK_ELWADD_43563) + \
        (defined(QK_DUALMOV_43563) && QK_DUALMOV_43563) >                                             \
    1
#error "QK_SRCDUMP_43563 / QK_ELWADD_43563 / QK_DUALMOV_43563 are mutually exclusive (all override the QK MATH op)"
#endif

template <bool transpose>
inline void sdpa_custom_mm_configure_addrmod() {
    constexpr std::uint32_t face_r_dim = 8;
    constexpr std::uint8_t ADDR_MOD_0_SRCA_INCR = transpose ? 32 : 16;
    constexpr std::uint8_t ADDR_MOD_1_SRCA_INCR = transpose ? (64 - 16) : 16;
    constexpr std::uint16_t ADDR_MOD_1_DEST_INCR = (1024 - face_r_dim);
    constexpr std::uint8_t ADDR_MOD_2_DEST_INCR = 2 * face_r_dim;

    addr_mod_t{
        .srca = {.incr = ADDR_MOD_0_SRCA_INCR, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = face_r_dim, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    addr_mod_t{
        .srca = {.incr = ADDR_MOD_1_SRCA_INCR, .clr = 0, .cr = 0},
        .srcb = {.incr = 16, .clr = 0, .cr = 0},
        .dest = {.incr = ADDR_MOD_1_DEST_INCR, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1);

    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = ADDR_MOD_2_DEST_INCR, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_2);

    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 1, .cr = 0},
    }
        .set(ADDR_MOD_3);

    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 1, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_4);

    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_7);

#if defined(QK_ELWADD_43563) && QK_ELWADD_43563
    // #43562/3 DISCRIMINATOR: override ADDR_MOD_0/1/2/3 for the ELWADD variant. ELWADD is element-wise:
    // SrcA, SrcB and DEST must advance TOGETHER (no matmul-style SrcA-only width walk). We walk one full
    // face (16 rows) per ELWADD across all three regs, then clear Src + step/reset DEST at the tile/block
    // boundaries exactly like the MVMUL modes do (so the loop structure + DEST coverage is unchanged).
    addr_mod_t{
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 16, .clr = 0, .cr = 0},
        .dest = {.incr = 16, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);  // next face, all three regs together
    addr_mod_t{
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 16, .clr = 0, .cr = 0},
        .dest = {.incr = 16, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1);  // next face, all three regs together
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = ADDR_MOD_2_DEST_INCR, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_2);  // end-of-tile: clear SrcA/SrcB, step DEST to next tile (cr commits)
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 1, .cr = 0},
    }
        .set(ADDR_MOD_3);  // end-of-block: clear everything
#endif

#if defined(QK_SRCDUMP_43563) && QK_SRCDUMP_43563
    // #43562/3 SRC DISCRIMINATOR: override ADDR_MOD_0/1/2/3 for the MOVA2D/MOVB2D register-dump variant.
    // A MOV reads the chosen Src register (read offset = the instr `src` field) and writes 8 (MOVA2D) or
    // 4 (MOVB2D) rows to DEST. Only DEST needs to walk so the 3 dump ops land in successive faces of the
    // score tile; Src + fidelity stay put (the unpack-produced register is read in place). At end-tile we
    // step DEST to the next tile and clear Src (mirroring the MVMUL CLR_A / CLR_AB so the loop structure +
    // Src-valid lifetime are identical to the matmul path), at end-block we clear everything.
    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);  // advance DEST by 8 rows (one MOVA2D face / two MOVB2D MOV_4_ROWS)
    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1);  // advance DEST by 8 rows
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = ADDR_MOD_2_DEST_INCR, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_2);  // end-of-tile: clear SrcA/SrcB, step DEST to next tile (cr commits)
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 1, .cr = 0},
    }
        .set(ADDR_MOD_3);  // end-of-block: clear everything
#endif

#if defined(QK_DUALMOV_43563) && QK_DUALMOV_43563
    // #43562/3 BALANCED DUAL-MOV addr-mods. CRITICAL: MVMUL's per-op dvalid clear is its dedicated
    // clear_dvalid field (CLR_A/CLR_AB), and the matmul clears SrcA per TILE (CLR_A) but REUSES SrcB
    // across the ct_dim width, clearing SrcB only at BLOCK end (CLR_AB). For MOVA2D/MOVB2D there is no
    // clear_dvalid field -> the ONLY clear is the addr_mod's SETRWC-style .srca.clr/.srcb.clr. So we must
    // mirror the matmul's clear cadence EXACTLY: clear SrcA at each tile boundary, clear SrcB ONLY at the
    // block boundary (a premature SrcB clear made the next tile's MOVB2D wait on a SrcB-valid that the
    // unpacker -- which reuses SrcB across width -- never re-produces -> MATH deadlock, the earlier bug).
    //   ADDR_MOD_0/1: in-loop MOVB2D -- no Src clear, DEST += 8 (face step).
    //   ADDR_MOD_7   : in-loop MOVA2D -- no Src clear, no DEST step (set all-zero in the base config).
    //   ADDR_MOD_2   : end-tile MOVA2D -- clear SrcA ONLY (== MVMUL CLR_A), no DEST change.
    //   ADDR_MOD_3   : end-tile MOVB2D -- NO Src clear (SrcB reused), step DEST to the next tile (cr).
    //   ADDR_MOD_5   : end-block MOVA2D -- clear SrcA.
    //   ADDR_MOD_6   : end-block MOVB2D -- clear SrcB (== the SrcB half of MVMUL CLR_AB), clear DEST.
    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);  // in-loop MOVB2D: DEST += 8 (next face), no Src clear
    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1);  // in-loop MOVB2D: DEST += 8 (next face), no Src clear
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_2);  // end-tile MOVA2D: clear SrcA only (MVMUL CLR_A), no DEST step
    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = ADDR_MOD_2_DEST_INCR, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_3);  // end-tile MOVB2D: NO SrcB clear (reuse), step DEST to next tile (cr commits)
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_5);  // end-block MOVA2D: clear SrcA
    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 1, .cr = 0},
    }
        .set(ADDR_MOD_6);  // end-block MOVB2D: clear SrcB (SrcB half of CLR_AB), clear DEST
#endif
}

inline void sdpa_custom_mm_configure_mop(const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim) {
#if defined(QK_DUALMOV_43563) && QK_DUALMOV_43563
    // BALANCED DUAL-MOV. The original MVMUL path issues exactly ONE MVMUL per op slot (replay buffer = 3
    // MVMULs, end-tile = 1 MVMUL CLR_A, end-block = 1 MVMUL CLR_AB), and each MVMUL reads BOTH SrcA(Q)
    // and SrcB(K), consuming one SrcA-valid + one SrcB-valid per op (the working ELWADD variant mirrors
    // this 1:1 and never hangs). To read the two Src out SEPARATELY we replace EACH single MVMUL with a
    // {MOVA2D, MOVB2D} PAIR: MOVA2D consumes the SrcA-valid (-> Q in DEST tile0), MOVB2D consumes the
    // SrcB-valid (-> K in DEST tile1 via a +16-row absolute `dst`). Keeping the pairing 1:1 with the
    // MVMULs preserves the exact SrcA/SrcB dvalid consumption -> no desync, no deadlock. The non-clearing
    // in-loop pairs use ADDR_MOD_7 (MOVA2D, no DEST step) + ADDR_MOD_0/1 (MOVB2D, +8 rows/face) so the two
    // faces of tile0/tile1 are covered. The clear cadence MIRRORS the matmul (see the addr-mod block):
    // SrcA is cleared per TILE (end-tile MOVA2D, ADDR_MOD_2) and at block end (ADDR_MOD_5); SrcB is REUSED
    // across the width and cleared ONLY at BLOCK end (end-block MOVB2D, ADDR_MOD_6) -- the end-tile MOVB2D
    // (ADDR_MOD_3) must NOT clear SrcB (else the next tile's MOVB2D waits on a SrcB-valid the unpacker
    // never re-produces -> deadlock). Three small replay buffers:
    //   [16..21] body pair x3 (mirrors the 3-MVMUL replay), [22..23] end-tile pair, [24..25] end-block pair.
    constexpr std::uint32_t DUALMOV_K_DST_OFFSET = 16;  // K -> tile1 (mm1 region + 1 tile = +16 rows)
    constexpr std::uint32_t DUALMOV_BODY_OFF = ckernel::math::replay_buf_offset;  // 16
    constexpr std::uint32_t DUALMOV_END_TILE_OFF = DUALMOV_BODY_OFF + 6;          // 22
    constexpr std::uint32_t DUALMOV_END_BLOCK_OFF = DUALMOV_END_TILE_OFF + 2;     // 24
    load_replay_buf(DUALMOV_BODY_OFF, 6, [] {
        TTI_MOVA2D(0, p_mova2d::MATH_HALO_ROWS, ADDR_MOD_7, p_mova2d::MOV_8_ROWS, 0);                      // Q face 0
        TTI_MOVB2D(0, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, DUALMOV_K_DST_OFFSET);  // K face 0
        TTI_MOVA2D(0, p_mova2d::MATH_HALO_ROWS, ADDR_MOD_7, p_mova2d::MOV_8_ROWS, 0);                      // Q face 1
        TTI_MOVB2D(0, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, DUALMOV_K_DST_OFFSET);  // K face 1
        TTI_MOVA2D(0, p_mova2d::MATH_HALO_ROWS, ADDR_MOD_7, p_mova2d::MOV_8_ROWS, 0);  // Q (3rd accumulate)
        TTI_MOVB2D(0, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, DUALMOV_K_DST_OFFSET);  // K (3rd)
    });
    load_replay_buf(DUALMOV_END_TILE_OFF, 2, [] {
        TTI_MOVA2D(0, p_mova2d::MATH_HALO_ROWS, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0);  // Q end-tile: clear SrcA only
        TTI_MOVB2D(
            0,
            p_movb2d::SRC_ZERO_OFFSET,
            ADDR_MOD_3,
            p_movb2d::MOV_4_ROWS,
            DUALMOV_K_DST_OFFSET);  // K: no SrcB clear (reuse), step DEST tile
    });
    load_replay_buf(DUALMOV_END_BLOCK_OFF, 2, [] {
        TTI_MOVA2D(0, p_mova2d::MATH_HALO_ROWS, ADDR_MOD_5, p_mova2d::MOV_8_ROWS, 0);  // Q end-block: clear SrcA
        TTI_MOVB2D(
            0,
            p_movb2d::SRC_ZERO_OFFSET,
            ADDR_MOD_6,
            p_movb2d::MOV_4_ROWS,
            DUALMOV_K_DST_OFFSET);  // K end-block: clear SrcB + DEST
    });
    const std::uint32_t mvmul_base = lltt::replay_insn(DUALMOV_BODY_OFF, 6);
    const std::uint32_t mvmul_end_tile = lltt::replay_insn(DUALMOV_END_TILE_OFF, 2);
    const std::uint32_t mvmul_end_block = lltt::replay_insn(DUALMOV_END_BLOCK_OFF, 2);

    ckernel_template tmp = ckernel_template(1, ct_dim, mvmul_base, mvmul_end_tile);
    tmp.set_last_inner_loop_instr(mvmul_end_block);
    tmp.set_last_outer_loop_instr(mvmul_end_block);
    tmp.program();
#else
    constexpr std::uint32_t replay_buf_len = 3;
#if defined(QK_SRCDUMP_43563) && QK_SRCDUMP_43563
    // #43562/3 SRC DISCRIMINATOR: 3x register->DEST copy replacing the 3x MVMUL, SAME unpack-produced
    // Src register, SAME mm1 DEST region. op output == the dumped Src (Q for =1 / K for =2), NOT Q@K.
    // Each op walks DEST by 8 rows (ADDR_MOD_0/1) so the 3 copies land in successive faces; end-tile
    // ADDR_MOD_2 steps DEST to the next tile + clears Src, end-block ADDR_MOD_3 clears everything
    // (mirroring the MVMUL CLR_A / CLR_AB Src lifetime).
#if QK_SRCDUMP_43563 == 1
    // SrcA (Q) -> DEST. MOVA2D MOV_8_ROWS copies one 8-row face from SrcA read offset MATH_HALO_ROWS(=0).
    load_replay_buf(ckernel::math::replay_buf_offset, replay_buf_len, [] {
        TTI_MOVA2D(0, p_mova2d::MATH_HALO_ROWS, ADDR_MOD_0, p_mova2d::MOV_8_ROWS, 0);  // face 0
        TTI_MOVA2D(0, p_mova2d::MATH_HALO_ROWS, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 0);  // face 1
        TTI_MOVA2D(0, p_mova2d::MATH_HALO_ROWS, ADDR_MOD_0, p_mova2d::MOV_8_ROWS, 0);  // face 2
    });
    const std::uint32_t mvmul_base = lltt::replay_insn(ckernel::math::replay_buf_offset + 0, replay_buf_len);
    const std::uint32_t mvmul_end_tile = TT_OP_MOVA2D(0, p_mova2d::MATH_HALO_ROWS, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0);
    const std::uint32_t mvmul_end_block =
        TT_OP_MOVA2D(0, p_mova2d::MATH_HALO_ROWS, ADDR_MOD_3, p_mova2d::MOV_8_ROWS, 0);
#elif QK_SRCDUMP_43563 == 2
    // SrcB (K) -> DEST. MOVB2D MOV_4_ROWS copies 4 rows from SrcB read offset SRC_ZERO_OFFSET(=0).
    load_replay_buf(ckernel::math::replay_buf_offset, replay_buf_len, [] {
        TTI_MOVB2D(0, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 0);  // face 0
        TTI_MOVB2D(0, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 0);  // face 1
        TTI_MOVB2D(0, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 0);  // face 2
    });
    const std::uint32_t mvmul_base = lltt::replay_insn(ckernel::math::replay_buf_offset + 0, replay_buf_len);
    const std::uint32_t mvmul_end_tile =
        TT_OP_MOVB2D(0, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
    const std::uint32_t mvmul_end_block =
        TT_OP_MOVB2D(0, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_4_ROWS, 0);
#endif
#elif defined(QK_ELWADD_43563) && QK_ELWADD_43563
    // #43562/3 DISCRIMINATOR: 3x ELWADD (Q+K) replacing the 3x MVMUL, SAME SrcA(Q)/SrcB(K)/DEST.
    // dest_accum_en=0 (overwrite DEST, not accumulate), SRCB_NO_BCAST (plain element-wise A+B),
    // clear_dvalid=0 in the replay (Src cleared by the end-tile/end-block ops, mirroring the MVMUL path).
    load_replay_buf(ckernel::math::replay_buf_offset, replay_buf_len, [operandB_face_r_dim] {
        TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_0, 0);  // face 0
        TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_1, 0);  // face 1
        TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_0, 0);  // face 2
    });

    const std::uint32_t mvmul_base = lltt::replay_insn(ckernel::math::replay_buf_offset + 0, replay_buf_len);
    const std::uint32_t mvmul_end_tile = TT_OP_ELWADD(p_setrwc::CLR_A, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
    const std::uint32_t mvmul_end_block = TT_OP_ELWADD(p_setrwc::CLR_AB, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0);
#else
    load_replay_buf(ckernel::math::replay_buf_offset, replay_buf_len, [operandB_face_r_dim] {
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // 0
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // 16
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // 0
    });

    const std::uint32_t mvmul_base = lltt::replay_insn(ckernel::math::replay_buf_offset + 0, replay_buf_len);
    const std::uint32_t mvmul_end_tile = TT_OP_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_2, 0);
    const std::uint32_t mvmul_end_block = TT_OP_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);
#endif

    ckernel_template tmp = ckernel_template(1, ct_dim, mvmul_base, mvmul_end_tile);
    tmp.set_last_inner_loop_instr(mvmul_end_block);
    tmp.set_last_outer_loop_instr(mvmul_end_block);

    tmp.program();
#endif  // QK_DUALMOV_43563 (else branch: original / SRCDUMP / ELWADD MOP)
}

template <bool transpose = false>
inline void _llk_math_sdpa_custom_mm_init_(const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim = 1) {
    sdpa_custom_mm_configure_addrmod<transpose>();
    sdpa_custom_mm_configure_mop(operandB_face_r_dim, ct_dim);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void _llk_math_sdpa_custom_mm_mask_dest_(
    const std::uint32_t dst_index, const std::uint32_t ct_dim, bool mask_chunk = false) {
#if defined(FIX_MASKDEST_STALL_43563) && FIX_MASKDEST_STALL_43563
    // #43563 experiment: real STALLWAIT at function entry (replaces the 21+ MATH NOPs that fix the bug).
#if FIX_MASKDEST_STALL_43563 == 1
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::TRISC_CFG);
#elif FIX_MASKDEST_STALL_43563 == 2
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH);
#elif FIX_MASKDEST_STALL_43563 == 3
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
#endif
#endif
#if defined(NOPLOOP_MASKDEST_43563) && NOPLOOP_MASKDEST_43563
    // #43562/3 experiment: pure-delay NOP loop in place of the SRC_VLD stallwait (same location). volatile
    // counter so the compiler neither unrolls nor elides the loop; TTI_NOP is a real MATH-pipe instruction.
    for (volatile uint32_t _noploop = 0; _noploop < (NOPLOOP_MASKDEST_43563); _noploop++) {
        TTI_NOP;
    }
#endif
#if defined(FIX_QK_SRCB_BANK_RESET_43563) && FIX_QK_SRCB_BANK_RESET_43563
    // #43563 (rmillerTT opt1): reset the SrcB read bank to a known absolute state before QK reads
    // SrcB(K). CLR_B touches only SrcB; SrcA + DEST untouched.
#if FIX_QK_SRCB_BANK_RESET_43563 == 1
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_B);
#elif FIX_QK_SRCB_BANK_RESET_43563 == 2
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
#endif
#endif
    // Zero Dest
    uint32_t dst_offset = dst_index + get_dest_buffer_base();
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_offset);
#if defined(FIX_DEST_OFFSET_ORDER_43563) && FIX_DEST_OFFSET_ORDER_43563
    // #43563 iter-parity fix: ensure the DEST-offset SETC16 above COMMITS before the matmul's
    // ZEROACC/MVMUL DEST writes below. The mask path is incidentally covered by its STALLWAIT(SRCB_VLD);
    // the non-mask path had nothing, so the matmul could write at the STALE (prev-iteration, wrong-bank)
    // DEST offset -> iteration-dependent mm1. MATH-side analog of the UNPACK reset_config_context fix.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::TRISC_CFG);
#endif
    if (mask_chunk) {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCB_VLD);
        // Zero Dest
        uint32_t dst_face = dst_offset / 16;
#if defined(FIX_MASK_NOPLOOP_43563) && FIX_MASK_NOPLOOP_43563
        // #43562/3 EXPERIMENT: pure NOP delay instead of the MOVB2D coherent write (no DEST clear).
        for (volatile uint32_t _mnop = 0; _mnop < (FIX_MASK_NOPLOOP_43563); _mnop++) {
            TTI_NOP;
        }
#elif defined(FIX_MASK_MOVB2A_43563) && FIX_MASK_MOVB2A_43563
        // #43562/3 EXPERIMENT: ZEROACC gives mm1 a clean DEST base; then MOVB2A is the SrcB-CONSUMING FPU
        // op — but it writes SrcA (NOT DEST). Discriminates: SrcB-read-via-FPU (any such op => 0/7168) vs
        // specifically MOVB2D's mm1-DEST write (=> 6371, == ZEROACC-only). ttsim(v<=1): MOVB2A waits for
        // SrcB-valid, reads SrcB, writes SrcA data, touches no valids (CLR_B below still clears SrcB).
        for (uint32_t i = 0; i < ct_dim; i++) {
            TT_ZEROACC(p_zeroacc::CLR_16, 0, 0, ADDR_MOD_7, dst_face);
            dst_face++;
        }
        for (uint32_t i = 0; i < ct_dim * 2; i++) {
            TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ZERO_OFFSET);
        }
#elif defined(FIX_MASK_ZEROACC_43563) && FIX_MASK_ZEROACC_43563
        // #43562/3 EXPERIMENT: replace the coherent MOVB2D (SrcB->DEST) write with a flag-only ZEROACC
        // (CLR_16), but KEEP the rest of the mask-path scaffolding. Tests bank-switch/scaffolding (=>0/7168)
        // vs the MOVB2D coherent write (=>6371). RESULT: 6371 (broken) -> the MOVB2D op is the fixer.
        for (uint32_t i = 0; i < ct_dim; i++) {
            TT_ZEROACC(p_zeroacc::CLR_16, 0, 0, ADDR_MOD_7, dst_face);
            dst_face++;
        }
#else
        for (uint32_t i = 0; i < ct_dim * 2; i++) {
            TTI_MOVB2D(0, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_4, p_movb2d::MOV_8_ROW_BRCST, 0);
        }
#endif
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
    } else {
#if defined(REALZERO_MM1_43563) && REALZERO_MM1_43563
        // #43563 FIX (dormant unless REALZERO_MM1_43563 != 0).
        //
        // Root cause: the Blackhole SFPU does NOT honour DEST zero-flags (only FPU/Packer/RISC do).
        // The original no-mask path zeroes mm1 with ZEROACC CLR_16, which (per ISA) only SETS the
        // zero-flag bits and does NOT write data ("zero-flags are set to emulate clearing"). The
        // downstream SFPU reduce_max/reduce_sum then read mm1 and see stale, bank-dependent leftover
        // -> bank-dependent MS -> iter-parity / ND-PCC bug.
        //
        // Fix: write REAL zero DATA into every datum-row of the mm1 region using ZEROACC in
        // CLR_SPECIFIC (single-row) mode, which physically zeroes one DEST register (ISA: "Single
        // (00) mode clears one register"), and with clear_zero_flags=1 so the flag is CLEARED (row
        // treated as real, initialized data by every consumer, incl. the SFPU which ignores flags).
        //
        // Why this is hang-free (unlike the prior MOVB2D attempt):
        //   * ZEROACC touches ONLY DEST. It reads/writes NO Src register, so SrcB data, SrcB dvalid,
        //     and the SrcB read pointer are all completely untouched. The MVMUL that runs right after
        //     gets its SrcB (K/Q) + dvalid from the matmul unpacker exactly as before -> no desync.
        //   * No STALLWAIT(SRCB_VLD): on the no-mask path SrcB is produced by the LATER matmul unpack,
        //     so we must NOT wait on it here (that would deadlock). ZEROACC needs no Src, so we don't.
        //   * ZEROACC's opcode (TT_OP 0x10) keeps bits 23:22 = 0, so it does NOT clear SrcA/SrcB
        //     data-valid (the ISA note warns only nonzero 23:22 would).
        //   * ADDR_MOD_7 has all increments = 0, so the DEST RWC counter does NOT advance. We address
        //     each row purely via the absolute `where` field (DEST_Offset already = dst_offset, RWC
        //     counter already at the region base from init/prior block-end CLR_AB). The counter is
        //     left exactly where the following MVMUL requires it -> no DEST read-pointer drift, and
        //     no SETRWC/CLR_B needed (so SrcB is never cleared, unlike the prior attempt).
        //
        // A tile is 16 rows (2 faces of 8); clear ct_dim*16 rows == ct_dim tiles == the full mm1
        // region the reduce reads. CLR_SPECIFIC's `where` is a 14-bit row offset; ct_dim<=16 ->
        // max offset 255, well in range. This is the same ZEROACC instruction family already on this
        // path, just the data-writing mode instead of the flag-only mode.
        for (uint32_t row = 0; row < ct_dim * 16; row++) {
            // TT_ (runtime) form: `row` is a runtime loop var, so the immediate TTI_ form fails to compile.
            TT_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 1, ADDR_MOD_7, row);
        }
#else
        // Zero Dest
        uint32_t dst_face = dst_offset / 16;
        // A tile is 2 faces of 8x16, so clearing 16 rows per tile is equivalent to clearing the tile
        for (uint32_t i = 0; i < ct_dim; i++) {
            TT_ZEROACC(p_zeroacc::CLR_16, 0, 0, ADDR_MOD_7, dst_face);
            dst_face++;
        }
#endif
#if defined(FIX_NONMASK_LIKE_MASK_43563) && FIX_NONMASK_LIKE_MASK_43563
        // #43563: mimic the MASK branch's terminal counter/bank reset. The mask path ends with this exact
        // SETRWC (CLR_B + SET_ABD) after its DEST write; the non-mask path lacked it, leaving the SrcB read
        // bank + RWC counters in a drifted/iteration-dependent state for the QK matmul -> bank-dependent
        // mm1.
        //   1 = full mask-path SETRWC (CLR_B + SET_ABD). RESULT (pos4351, clean build): HANGS — clearing
        //       SrcB-valid desyncs the non-mask unpack/MATH handshake (the non-mask matmul's SrcB lifetime
        //       differs from the mask path, which re-unpacks SrcB fresh after its SETRWC) -> MATH stalls
        //       forever -> host busy-spin (tt-smi -r). Same deadlock family as FIX_QK_SRCB_BANK_RESET /
        //       the MOVB2D-in-non-mask attempts.
        //   2 = candidate (b): RWC counter/pointer reset ONLY (SET_ABD), NO CLR_B. Resets SrcA/SrcB/DEST
        //       read+write pointers to a known absolute state (targets the pointer/bank-drift root cause)
        //       WITHOUT clearing SrcB-valid -> no handshake desync.
#if FIX_NONMASK_LIKE_MASK_43563 == 2
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
#else
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
#endif
#endif
    }
}

inline void _llk_math_sdpa_custom_mm_(
    const std::uint32_t operandB_face_r_dim,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1,
    const bool mask_chunk = false) {
    // dst offset initialized by _llk_math_sdpa_custom_mm_mask_dest_
    _llk_math_sdpa_custom_mm_mask_dest_(dst_index, ct_dim, mask_chunk);

    for (std::uint32_t i = 0; i < kt_dim - 1; i++) {
        TTI_MOP(1, 0, 0);
    }
    for (uint32_t i = 0; i < ct_dim - 1; i++) {
#if defined(QK_DUALMOV_43563) && QK_DUALMOV_43563
        // BALANCED DUAL-MOV: body pair-x3 replay [16..21] + end-tile pair replay [22..23], mirroring the
        // MVMUL path's replay(3) + 1 MVMUL(CLR_A). Each {MOVA2D,MOVB2D} pair consumes one SrcA + one SrcB,
        // exactly like one MVMUL -> balanced handshake.
        lltt::replay(ckernel::math::replay_buf_offset, 6);
        lltt::replay(ckernel::math::replay_buf_offset + 6, 2);  // end-tile pair (CLR_A on the MOVB2D)
#else
        lltt::replay(ckernel::math::replay_buf_offset, 3);
#if defined(QK_SRCDUMP_43563) && QK_SRCDUMP_43563
#if QK_SRCDUMP_43563 == 1
        TTI_MOVA2D(0, p_mova2d::MATH_HALO_ROWS, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0);
#elif QK_SRCDUMP_43563 == 2
        TTI_MOVB2D(0, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
#endif
#elif defined(QK_ELWADD_43563) && QK_ELWADD_43563
        TTI_ELWADD(p_setrwc::CLR_A, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
#else
        TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_2, 0);
#endif
#endif
        t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU);
    }
#if defined(QK_DUALMOV_43563) && QK_DUALMOV_43563
    lltt::replay(ckernel::math::replay_buf_offset, 6);
    lltt::replay(ckernel::math::replay_buf_offset + 8, 2);  // end-block pair [24..25] (CLR_AB on the MOVB2D)
#else
    lltt::replay(ckernel::math::replay_buf_offset, 3);
#if defined(QK_SRCDUMP_43563) && QK_SRCDUMP_43563
#if QK_SRCDUMP_43563 == 1
    TTI_MOVA2D(0, p_mova2d::MATH_HALO_ROWS, ADDR_MOD_3, p_mova2d::MOV_8_ROWS, 0);
#elif QK_SRCDUMP_43563 == 2
    TTI_MOVB2D(0, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_4_ROWS, 0);
#endif
#elif defined(QK_ELWADD_43563) && QK_ELWADD_43563
    TTI_ELWADD(p_setrwc::CLR_AB, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0);
#else
    TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);
#endif
#endif
    t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU);
}
