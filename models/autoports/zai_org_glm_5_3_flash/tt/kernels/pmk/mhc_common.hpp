// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prefill megakernel (PMK) mHC phases: the mHC "pre" (P1 / the pre half of P10) and "post" (the post half of P10 / P16)
// of ONE persistent layer program at 256 SP rows per chip (doc/ttnn/16-prefill-production/prefill_megakernel.md §2,
// §4b; tt/pmk_mhc.py is the host side and documents the design).
//
// Geometry (contract §2): 64 members = 8 SP row blocks (32 rows) x 8 column slabs; member (b, j) owns the column tiles
// [16 j, 16 j + 16) of all 4 streams of row block b = 64 fp32 tiles = 256 KB, RESIDENT in its L1 across the layer
// (CB_X, pushed once, never popped). 8 roots, root b serves row block b. This is Stage 7a's MhcPrePf geometry
// (G = 8 groups x P = 8 members, CPC = 16, PF_FPU arithmetic) with one row block per group and the residual in place.
//
// Roles (compile-time defines): PMK_MEMBER, PMK_ROOT. Three sources per role: mhc_stream.cpp (BRISC, NoC0: DRAM
// reads / writes at program boundaries, the post's in-place copy-back), mhc_sync.cpp (NCRISC, NoC1: partials,
// messages, coefficient tiles, hand-off counters), mhc_compute.cpp (TRISCs).
//
// CB indices follow the contract's owner map (§4b): mHC pre 0..20 (7a's PF set 0..19 + 20), mHC post 21..28.
// Semaphores: 0..3 the pre's intra-phase rounds (monotone per layer), the hand-off counters below (ids are host
// parameters; defaults in tt/pmk_mhc.py). PARTS (Stage 16 blocker): PMK_MHC_PART = 1 (P1 = pre of sublayer 0), 10 (P10
// = post of sublayer 0 + pre of sublayer 1), 16 (P16 = post of sublayer 1), 0 = all three in one kernel_main (the
// standalone unit / mock). The members are also the stationary MM cores, so every part RETURNS at its phase end; the
// state that stays live between parts is in CBs (README §11). Every wait is `noc_semaphore_wait_min` on a PROGRAM-WIDE
// cumulative count (round0 = the earlier layers' rounds folded into every expected value; the only reset is the
// launch's initial 0 — integrator 10:02Z, contract §4b counter rule). The x-hat holder's word is FS_ARRIVE (15), raised
// once per member block and polled by coll with cumulative expected values.
#pragma once

#include <cstdint>

#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/mhc_pre_common.hpp"

// phase argument bases (Stage 14's kernels/mk/phase_args.hpp convention): a standalone program uses 0 / 0; a combined
// megakernel source defines MK_PHASED and, per phase namespace, PH_CT0 / ph_rt0 before including the phase kernel.
#if !defined(MK_PHASED)
constexpr uint32_t PH_CT0 = 0;
constexpr uint32_t ph_rt0 = 0;
#endif

namespace pmk_mhc {

using mhc::BF16_TILE_BYTES;
using mhc::COL_TILES;
using mhc::FP32_TILE_BYTES;
using mhc::MSG_FLOATS;
using mhc::N_MIX;
using mhc::N_STREAMS;
using mhc::plane_off;
using mhc::TILE;
using mhc::tile_off;

constexpr uint32_t P = 8;                      // members per row block (column slabs)
constexpr uint32_t CPC = COL_TILES / P;        // 16 column tiles per member and stream
constexpr uint32_t X_TILES = N_STREAMS * CPC;  // 64 residual tiles per member (the resident block)
constexpr uint32_t N_GROUPS = 8;  // row-block groups (roots) per chip: group g handles blocks g, g + 8, ...
// Geometry defines (host): PMK_NRB = row blocks per group (SP rows / 32 / 8: 1 at TP8's 256 rows, 4 at TP2's 1,024),
// PMK_RESIDENT = 1: the group's single block lives in CB_X for the whole layer (TP8); 0: blocks stream from / to the
// DRAM residual per sublayer and phase (TP2: the 64 MB residual cannot sit in L1). PMK_NSUB = sublayers per program.

// ---- circular buffers: pre (0..20 = 7a's mhc_pre_pf set, same meaning)
constexpr uint32_t CB_GATHER =
    0;  // fp32 2P pages, root: round 1 = P x (M, S), round 2 = P x S2 (uniform address: all cores)
constexpr uint32_t CB_MSG = 1;  // 512 B mailbox root -> members (uniform address: all cores)
constexpr uint32_t CB_X = 2;    // fp32 X_TILES pages: THE RESIDENT RESIDUAL BLOCK (matmul operand), index s * CPC + j
constexpr uint32_t CB_X32 = 3;  // alias of CB_X: exact fp32 copy_tile operand
constexpr uint32_t CB_FN = 4;   // bf16 X_TILES pages: fn^T slab of the sublayer (re-streamed per sublayer)
constexpr uint32_t CB_W = 5;  // bf16 CPC pages: the sublayer's norm weight tiles (row 0; the checkpoint's bf16 values)
constexpr uint32_t CB_PART = 6;    // fp32 2 pages: (M, S) partial
constexpr uint32_t CB_PRE = 7;     // fp32 4 pages: diag(pre_s) tiles (PF_FPU collapse operand)
constexpr uint32_t CB_H = 8;       // fp32 CPC pages: collapsed h tiles
constexpr uint32_t CB_S2 = 9;      // fp32 1 page: sum h^2 partial
constexpr uint32_t CB_INV = 10;    // fp32 1 page: 1/rms(h) in column 0
constexpr uint32_t CB_OUT = 11;    // bf16 CPC pages: hn tiles = H1 / H7 x-hat (contract §4b: "member OUT 11")
constexpr uint32_t CB_H16 = 12;    // alias of CB_H: the S2 matmul operand
constexpr uint32_t CB_RED = 14;    // fp32 1 page, root: z [tokens x 24]
constexpr uint32_t CB_PLANE = 15;  // fp32 1 page, root: z in the plane layout
constexpr uint32_t CB_COEF = 16;   // fp32 1 page, root: pre | post | comb planes
constexpr uint32_t CB_SB = 17;  // fp32 3 pages, root: scale row, bias row, identity OF THE SUBLAYER (rev 6: pushed per
                                // sublayer by the stream part / prelude, popped by the compute after its rounds)
constexpr uint32_t CB_INVR = 18;   // fp32 1 page, root: 1/rms(h) column
constexpr uint32_t CB_SCR = 19;    // fp32 2 pages, root: post / comb output tile assembly (DRAM mode only)
constexpr uint32_t CB_TOKEN = 20;  // 64 B 1 page, member: "residual copy-back done" (post -> the following pre)
// ---- post (21..28)
constexpr uint32_t CB_Y =
    21;  // fp32 (or bf16) CPC pages per block (NRB blocks lane-landed, 1 block DRAM-read): y = H6 / H10
constexpr uint32_t CB_PCOEF = 22;   // fp32 NRB x 8 pages: K_s | L_s packed coefficient tiles per row block (root ->
                                    // members; uniform address: all cores)
constexpr uint32_t CB_PSTAGE = 23;  // fp32 8 pages: 2 units of post output tiles, copied back in place by the BRISC
constexpr uint32_t CB_DBG = 24;     // fp32 N_TAPS pages, root (PMK_PROBE): the coefficient-pass taps of a round
constexpr uint32_t N_TAPS = 4;  // 0 z planes as loaded, 1 the pass (ITERS), 2 the pass with ITERS = 1, 3 with ITERS = 2
// 25..28 free (reserved for the integrator's post-side needs)

// ---- the root's verified coefficient pack (rev 5; README §9e). Planes 24..31 of the plane-layout tile are free
// (24 = the pass's row-max scratch, DST only). The pass ends by storing MARKER_BITS(round) in plane PLANE_MARKER (its
// LAST SFPU store); the pack thread packs DST tile 0 into CB_COEF page 0 (out-of-order pack: the CB is ONE page, the
// tile pointer must not advance -- rev 4's three bumping packs wrote pages 1 / 2 = CB_SB's scale / bias tiles) until
// the packed page carries the marker, then once more (that pack starts after the SFPU's last store). Diagnostics:
// the pack thread leaves the attempt counts in the (consumed) CB_RED page, plane PLANE_DIAG lanes 0 / 1; the sync
// RISC checks the marker and, with PMK_PROBE, copies attempts / ok into tap 0's plane PLANE_DIAG lanes 0..2.
constexpr uint32_t PLANE_MARKER = 25;
constexpr uint32_t PLANE_DIAG = 26;
constexpr uint32_t PACK_MAX_ATTEMPTS = 2;  // rev 6: small bound (silicon 23:20Z: 64 attempts x 16 = +2,942 us per
constexpr uint32_t PACK_POLL_READS = 64;   // layer with the marker never seen); the final pack always follows
// rev 6 diagnostics: tap 3 (the former ITERS = 2 tap) is a record the SYNC RISC builds in its own mailbox scratch
// (never a packer-written page): words DIAG_MAGIC, r, sub, k, marker expected, plane-25 lane 0 / lane 31 as read from
// the pushed COEF page, marker_ok (all 32 lanes), attempts1 / attempts2 (the pack thread's words in the CB_RED page:
// an unverified channel), rev, idx.
constexpr uint32_t TAP_DIAG = 3;
constexpr uint32_t DIAG_MAGIC = 0xD1A60006u;
constexpr uint32_t DIAG_WORDS = 12;
constexpr uint32_t marker_bits(uint32_t r) { return 0x44000000u + (r + 1u) * 0x10000u; }  // fp32 512 + 4 (r + 1)
constexpr uint32_t marker_final(uint32_t r) { return marker_bits(r) | 0x80000000u; }  // cycle 3's: -(512 + 4 (r + 1))

// ---- post DST slots (7a's mhc_post_pf7a/common.hpp, full-sync fp32 DST: 8 tiles)
constexpr uint32_t D_Y = 0;
constexpr uint32_t D_X0 = 1;  // .. D_X0 + 3
constexpr uint32_t D_K = 5;
constexpr uint32_t D_L = 6;
constexpr uint32_t D_OUT = 7;

// ---- semaphores: the pre's rounds (contract: owners + root 0..3 = mhc_pre)
constexpr uint32_t SEM_G1 = 0;  // root: round-1 partials (+P per sublayer)
constexpr uint32_t SEM_G2 = 1;  // root: round-2 partials (+P per sublayer)
constexpr uint32_t SEM_M1 = 2;  // member: pre message (+1 per sublayer)
constexpr uint32_t SEM_M2 = 3;  // member: 1/rms(h) message (+1 per sublayer)
// the hand-off counters are compile-time defines (PMK_SEM_Y = FS_REDUCED, PMK_SEM_XH = FS_ARRIVE), set by the host.
// There is NO coefficient semaphore (7a review: id 12 is the family's XS_FREE): the root's packed-coefficient unicast
// is write-barriered before its round-2 message, so a member's M2 >= idx implies the tiles landed.
#if !defined(PMK_MHC_PART)
#define PMK_MHC_PART 0
#endif

// ---- member flags (runtime arg)
constexpr uint32_t F_X_LOAD = 1;    // resident: load the residual block from the DRAM residual at program start
constexpr uint32_t F_X_STORE = 2;   // resident: write the residual block back to the DRAM residual at program end
constexpr uint32_t F_XH_DRAM = 4;   // hn tiles -> the DRAM hn tensor (else: L1 push to the lane + SEM_XH)
constexpr uint32_t F_Y_DRAM = 8;    // y tiles read from the DRAM y tensor (else: landed by the RS lanes + SEM_Y)
constexpr uint32_t F_PC_DRAM = 16;  // root: also write the post / comb tiles of the row block to DRAM (bench)
constexpr uint32_t F_INIT = 32;     // once-per-program initialisations (the first layer's P1): SB load, page zeroing
constexpr uint32_t F_FN_PREFETCH =
    64;  // P1 ends by reading sublayer 1's fn^T / w (after the IN0-free gate); else P10 reads them
constexpr uint32_t F_PROBE = 128;       // root: write the PMK_PROBE taps of every round to the DRAM taps tensor
constexpr uint32_t F_SB_PRELUDE = 256;  // root (P10): the integrator's prelude pushes sublayer 1's SB tiles (3 pages)
                                        // into CB_SB; else the stream part loads them (rev 6: CB_SB is PER SUBLAYER --
                                        // the scale / bias rows differ between hc_attn and hc_ffn; the compute pops
                                        // them after each sublayer's rounds. Root stream runtime: sb(0) address, flags,
                                        // sb(1) address)

}  // namespace pmk_mhc
