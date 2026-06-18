// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "../micro_ops/flash_mla/kernels/rt_args_common.hpp"

// #43563 STEP0 GROUND-TRUTH: 1 = on the BRISC reader, after get_runtime_args, DPRINT one line per
// core: (x,y), core_num, local_cur_pos, num_chunks, k_chunk_start, k_chunk_end, and the implied
// per-core chunk count = (k_chunk_end - k_chunk_start + stride - 1)/stride. Settles whether the SP
// path feeds get_runtime_args the GLOBAL or LOCAL cur_pos and the actual per-core chunk distribution.
// 0 = off. Defaulted OFF.
#define DUMP_CHUNKRANGE_43563 0

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#include "mcast.hpp"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#include "mcast.hpp"
#include "api/debug/assert.h"
#elif defined(COMPILE_FOR_TRISC)
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h"
#include "api/compute/eltwise_unary/exp.h"
#endif

// #43563 STEP0: pull in DPRINT for the chunk-range ground-truth dump (BRISC only).
#if defined(DUMP_CHUNKRANGE_43563) && DUMP_CHUNKRANGE_43563 && defined(COMPILE_FOR_BRISC)
#include "api/debug/dprint.h"
#endif

// #43562/3 CANDIDATE REAL FIX: route FULL (non-mask) chunks through the proven-clean MASK path with an
// all-ZERO (all-pass) additive mask. The MASK branch of _llk_math_sdpa_custom_mm_mask_dest_ is bank-
// DETERMINISTIC: it does STALLWAIT(SRCB_VLD) + a COHERENT MOVB2D (MOV_8_ROW_BRCST) real-data FPU write
// of the resident SrcB=mask into every mm1 DEST row, then accumulates Q@K -> mm1 = mask + Q@K. The
// NON-MASK (full-chunk) branch is bank-DEPENDENT: ZEROACC CLR_16 is FLAG-ONLY (no real DEST write),
// so the QK MVMUL reads/accumulates onto bank-dependent leftover -> the #43562/3 iter1-vs-iter2
// alternation. The mask is ADDITIVE (mm1 = mask + Q@K; mask_last_chunk writes 0x00000000 for unmasked
// positions, 0xFF80 = bf16 -inf for masked), so an all-ZERO mask is a CORRECT no-op for a full chunk:
// mm1 = 0 + Q@K = correct Q@K, computed via the clean coherent MOVB2D write -> deterministic.
//
// SCOPE: only cores that have NO real partial-mask last chunk (mask_last_chunk == false). The cb_mask
// CB holds ONE tile per invocation; we cannot have BOTH a zero tile (for earlier full chunks) and the
// real partial tile (for the masked last chunk) in it simultaneously, so we leave the real-mask cores
// untouched and only re-route the all-full cores (the documented pos4351 worker = 2 full chunks, no
// mask). When active on such a core: the reader pushes ONE all-zero mask tile (cb_mask), the compute
// wait_front/pop's it, and forces mask_chunk=true for EVERY chunk so each QK matmul takes the MASK path.
// RESULT (pos4351, clean build, 2026-06-08; baseline iter1-vs-iter2 = 6079/7168, max|d|=0.0547):
//   FIX WORKS -> differ=0/7168, max|d|=0 (bit-identical iter1==iter2), PCC-vs-golden 0.9936 (UNCHANGED,
//   per-shard 0.9930..0.9944, bit-identical across iters). No hang. CONFIRMS the MASK branch's COHERENT
//   MOVB2D real-data DEST write IS the determinism source: routing full chunks through it (with an
//   all-zero additive mask) closes the #43562/3 iter-parity bug while preserving correctness. This is a
//   viable REAL fix (scoped to all-full cores; cores with a real partial mask are untouched).
// 1 = on. 0 = off (default; original non-mask path for full chunks).
#define FIX_FULL_VIA_MASK_43563 1

// #43562/3 EXPERIMENT: zero the Q input (cb_q) on active worker cores so the QK matmul computes
// mm1 = 0 @ K = 0 (exactly). Any non-zero, iter-dependent mm1 then == the pure spurious bias. Captured
// via SDPA_MM1DIRECT_TAP (raw mm1). 1 = zero Q; 0 = off.
#define ZERO_Q_INPUT_43563 1

// #43563 ISOLATION: 1 = let compute_sdpa_chunk run normally (semaphores intact, parity preserved),
// then overwrite its L output (mm2 DEST tiles) with cb_q before packing => SDPA output becomes
// deterministic/iteration-independent. If the iters=1-vs-2 bug PERSISTS, compute_sdpa_chunk's
// output content is ruled out (the bug lives in the downstream op interaction).
#define STUB_SDPA_43563 0
// #43563 zero-flag test: 1 = fill the active DEST half with a deterministic NON-ZERO constant (cb_q)
// right after tile_regs_acquire, before the SDPA compute. Tests the "SFPU reads bank-dependent
// zero-flagged leftover" hypothesis: making all DEST data deterministic (bank-independent) before the
// matmul should kill the iter-parity bug if leftover-read is the cause. (Non-zero so the FPU write
// isn't optimized into a flag-only set; copy_tile from cb_q writes real data + clears zero-flags.)
#define FILL_DEST_43563 0
// #43563 LOCALIZE: 1 = reset the DEST bank to 0 at the SDPA-compute -> tail-reduce boundary
// (after the compute tile_regs_release, before do_tail_reduce). Pins the tail + everything after
// to bank 0 every iter while the per-core SDPA compute stays on the alternating bank. If pos8190
// goes to ~0 differ, the bank-dependent carrier is in the tail/downstream; if it stays ~6337, the
// carrier is in the per-core SDPA compute. Default 0 (off).
// RESULT: pos8190 -> 6353/7168 differ, max 0.0625 (baseline 6337/7168, max 0.078) => UNCHANGED.
// Carrier is in the per-core SDPA compute, NOT the tail/downstream.
#define BANK_RESET_AT_TAIL_43563 0
// #43563 MS-TILE DEST DUMP: 1 = at the post-chunk-loop location (after the reduce has written the
// real max/sum into the MS tile, before the MS pack), dump all 16 rows of the MS DEST tile so we can
// read every datum. The MS tile is based at ABSOLUTE DEST row max_dst_offset (= max_dst_tile_offset *
// packed_tile_size, packed_tile_size=16). The reduce writes max into rows {0,4} and sum into {2,6}
// (rel max_dst_offset); the OTHER rows are the suspected dirty carry-over. We MATH-stall first so the
// reduce is settled, then dump absolute DEST rows max_dst_offset+0 .. +15 via the low-level
// dprint_tensix_dest_reg_row_float32(row) helper (on Blackhole this directly reads DEST at
// 0xFFBD8000 + (row<<4), so it matches the SDPA packed-tiny-tile layout exactly — we do NOT use
// dprint_tensix_dest_reg(tile_id) because that assumes 32 rows/tile and would mis-address). Run with
// FILL_DEST_43563=0 for baseline, then =1 to test carry-over (rows that change with FILL hold pre-
// compute DEST state = dirty leftover). 0 = off (kernel unchanged). Defaulted OFF.
#define DUMP_MS_43563 0
#if defined(COMPILE_FOR_TRISC) && defined(DUMP_MS_43563) && DUMP_MS_43563
#include "api/debug/dprint.h"
#include "api/debug/dprint_tensix.h"
#endif
// #43563 fix part 2: disable the FPU dst zero-flag WRITE-optimization in the SDPA region so FPU
// exact-0 results (e.g. bcast_sub mm1-max=0) are written as REAL zero data (not flag-only). With the
// CLR_SPECIFIC real-zero init (REALZERO_MM1_43563), every DEST zero becomes real data -> the SFPU
// (which ignores zero-flags) always reads correct values -> closes the iter-parity divergence.
// NOTE: =1 GARBAGES output — it also disables FPU *honouring* of zero-flags, breaking the other SDPA
// regions (mm2/max/sum) that legitimately rely on flag-honouring (only mm1 init was real-zeroed). Keep 0.
#define DISABLE_DST_ZEROFLAG_43563 0
// #43563: Lazar's NOP-drain experiment (tt-blaze#842): 50 MATH NOPs between compute_sdpa_chunk
// iterations, to test whether the next chunk's Q@K corrupting the prev chunk's sum partial is the
// (multi-chunk) bug. Only meaningful when num_chunks>1 (>=2 chunks/core, e.g. pos>=2047).
// #43563 race-vs-state test: bumped to 1000 NOPs (fires once after the single chunk at pos255).
#define NOP_DRAIN_43563 0
// #43563 STAGE CUT harness: sever compute_sdpa_chunk at a configurable stage and pack the
// chosen intermediate (still in DEST after the chunk loop) to the OUTPUT core's final output CB
// so the host can read it back per-core and compare against a torch golden. 0 = normal full
// chunk (kernel unchanged). 1=Stage A (mm1 = QK^T*scale of the LAST chunk), 2=Stage B (running
// row-max over the core's chunks). C/D/E reserved (3/4/5). When != 0 we (a) force the output
// core to pack into cb_out_final directly and (b) SKIP the cross-core tree reduction, so each
// output core's own per-chunk intermediate is observable in isolation (no cross-core merge).
// Only meaningful in the 2-chunks/core regime (pos>=2047, no mask). Defaulted OFF.
#define SDPA_STAGE_CUT_43563 0
// #43563 NON-INVASIVE SUM TAP: unlike SDPA_STAGE_CUT_43563 this leaves the FULL pipeline control
// flow UNCHANGED (does NOT force is_final, does NOT set last_chunk=false, does NOT disable the tree
// reduction). The buggy path runs exactly as in production. The ONLY addition is, on the OUTPUT
// core, an EXTRA packer copy of the max/sum DEST tile (max_dst_tile_offset: max in col0, running
// sum in col1) into a dedicated side CB (cb_sum_tap), done right after the per-core chunk loop and
// adjacent to the existing cb_interm_ms pack -- i.e. AFTER all of the output core's chunks have been
// accumulated but BEFORE the cross-core tree reduction consumes/combines anything. The extra pack
// reads the same already-settled DEST tile the kernel itself packs to cb_interm_ms, so it does not
// alter the DEST lifetime, the mm2 packer handshake, or any semaphore. Host reads cb_sum_tap back
// per output core and compares col1 (the running sum) to the torch golden per-core running sum.
// 0 = no tap (kernel unchanged). 1 = tap enabled. Defaulted OFF.
#define SDPA_SUM_TAP_43563 0
// #43563 L+MS TAP (this investigation): same NON-INVASIVE idea as SDPA_SUM_TAP, but capture BOTH the
// per-core shipped L (mm2 / PV attention output, vDHt tiles) AND the full MS tile (max in col0, sum in
// col1) into the side CB cb_iter1_dump, on the OUTPUT core, right after the per-core chunk loop and
// BEFORE the cross-S-block tree reduction consumes anything. Layout in cb_iter1_dump:
//   tiles 0..vDHt-1 = L (mm2_dst_tile_offset + i)   <- the real shipped per-core attention output
//   tile  vDHt      = MS (max_dst_tile_offset)      <- the real shipped per-core max/sum stats tile
// The extra packs read the SAME settled DEST tiles the kernel itself packs to sdpa_output_cb /
// sdpa_ms_cb below, so DEST lifetime, the mm2 packer handshake and all semaphores are unchanged. The
// side CB cb_iter1_dump must be sized (vDHt + 1) tiles by the host (attention_block/op.py). Host reads
// the side shard back per output core and diffs iter1-vs-iter2. 0 = off. Defaulted OFF.
#define SDPA_LMS_TAP_43563 0
// #43563 FIX (the candidate fix): before the QK^T matmul writes mm1, physically zero the entire
// DEST region the SDPA addresses using SFPU stores of literal 0 (sfpi dst_reg = 0 -> SFPSTORE).
// See sdpa.h::sfpu_zero_dest_43563. This writes REAL zero data (NOT a zero-flag), so when the SFPU
// reduce later reads any unwritten leftover row it gets a genuine bank-independent 0, killing the
// bank-dependent max/sum and the iter-parity alternation. The matmul overwrites valid mm1 on top.
// 0 = off (kernel unchanged). 1 = fix enabled (one-shot pre-loop whole-DEST zero).
// 2 = additionally re-zero the mm1 region per-chunk right before each QK matmul (see sdpa.h).
// Defaulted OFF.
// VALIDATION RESULT (2026-06-04, test_flash_mla_sum_tap, pos=2047, 2 chunks/core): this fix does
// NOT close the bank0-vs-bank1 MS gap (both variants leave the same 24/64 sum + 3/64 max diff as
// fix-OFF). It DOES neutralize injected FILL poison in leftover rows (proving the SFPU zero reaches
// them) and does not regress output PCC (~0.999). The residual native bank dependence therefore
// originates in the matmul-written mm1 / reduce arithmetic, not in unwritten leftover-row reads.
#define SFPU_ZERO_DEST_43563 0
// #43563 MAX/SUM SPLIT BISECTION: stub ONLY the per-core SDPA max (STUB_MAX) or ONLY the per-core
// sum (STUB_SUM) to a deterministic finite constant via a REAL SFPU datum store, at the SAME proven-
// safe post-chunk-loop location as STUB_SDPA_43563 (after the chunk loop, before the MS pack/recip).
// The reduce writes max into DEST rows {0,4} rel max_dst_offset and sum into {0,4} rel sum_dst_offset
// (col0/col1 of the same MS tile). We overwrite exactly those datums. Goal: see which of max/sum,
// when forced iteration-independent, makes iters=1 == iters=2.
// COUPLING CAVEAT: sum = Σ exp(scores - max) uses the REAL max, so stubbing only max can still differ
// (sum inherited max's bank dependence). The informative signal is the asymmetry. 0 = off.
#define STUB_MAX_43563 0
#define STUB_SUM_43563 0
// Constants written by the split stub (benign finite; output may be garbage, only determinism matters).
#define STUB_MAX_C_43563 1.0f
#define STUB_SUM_C_43563 1.0f
// #43563 CANDIDATE FIX: SFPU-zero the MS-tile rows the per-core reduce never writes. The reduce stores
// max into rows {0,4} and sum into rows {2,6} of the 16-row MS tile (base=max_dst_offset); rows
// {1,3,5,7,8..15} are NEVER written and hold bank-dependent leftover, yet the WHOLE tile is packed to
// sdpa_ms_cb and consumed by the cross-core tail reduce -> iter-parity alternation. At the SAME proven-
// safe post-chunk-loop location as STUB_MAX/SUM (after the chunk loop, MATH-stalled on max so the
// reduce is settled, on the PACK thread), we SFPU-store literal 0 (sfpi dst_reg, REAL data) into:
//   1 = ONLY the candidate junk rows {1,3,5,7,8..15} (assumed real datums {0,2,4,6} preserved).
//   2 = ALL 16 rows (control).
//   3 = ONLY the upper half {8..15} (preserve lower half {0..7}) -- diagnostic.
// 0 = off (kernel unchanged). Defaulted OFF.
//
// EMPIRICAL RESULT (2026-06-05, test_decoder_mlp pos=255, iters=1 vs 2):
//   =2 (all-16 SFPU zero): output GARBLED + NaN (PCC ~0.087), iters NOT identifiable (NaN!=NaN
//       makes diff/equal meaningless). Expected garble (max/sum zeroed) BUT it ALSO did not
//       cleanly reproduce the copy_tile fix's determinism -- the SFPU full-tile zero perturbs
//       more than the copy_tile path did.
//   =1 (candidate junk-only): output ALSO GARBLED + NaN (PCC ~0.10), iters still differ. This
//       DISPROVES the {0,2,4,6}=real / rest=junk row-model: zeroing {1,3,5,7} destroyed real
//       data the tail/recip consume. The reduce's raw TTI_SFPSTORE ZERO_ADDR_MOD offsets {0,4}
//       are NOT the same physical datums as sfpi dst_reg[{0,2,4,6}], so the assumed junk set is
//       wrong.
//   =3 (upper half {8..15}): NO-OP -- iters still differ EXACTLY as baseline (6543/7168, max
//       delta 0.109) AND PCC stays ~0.99. Proves rows {8..15} are NOT the leftover carrier and
//       are already benign; the consumed leftover lives in the lower 8 rows, interleaved with
//       the real max/sum. A simple per-row SFPU zero cannot separate junk from real here; the
//       working copy_tile fix succeeds because it rewrites a coherent full tile.
#define STUB_MS_JUNK_43563 0

// #43563 CANDIDATE FIX (FIX_MS_JUNK_43563). At the post-chunk-loop location (MATH-stalled on max so
// the reduce is settled, on the PACK thread), SFPU-store literal 0 (sfpi dst_reg, REAL data) into the
// bank-dependent "junk" the per-core SDPA leaves in DEST, while PRESERVING the real max/sum.
//
// DPRINT GROUND TRUTH (2026-06-05, fast venue test_flash_mla pos127 single-core AND pos255 worker
// cores 0..2; FORCE_ODD_BANK 0 vs 1, per-row Float16_b reader):
//   * The 16-row MS tile that is actually packed (pack_block_contiguous(max_dst_tile_offset,
//     sdpa_ms_cb, 1) = abs DEST rows 256..271) is BIT-IDENTICAL bank0==bank1 on EVERY core. Rows 0..7
//     hold the real interleaved max/sum (per query head), rows 8..15 are real zeros. There is NO
//     bank-dependent junk inside the packed MS tile.
//   * The ONLY bank-dependent datums are in the corr_exp tile at corr_exp_dst_offset (abs rows 272..
//     273), ODD lanes (col1). That tile is internal scratch: it is NOT packed/shipped, and at single
//     chunk (first_chunk==last_chunk, pos127/255) the !first_chunk correction path that would consume
//     it never runs -> it is inert.
// So this "zero the MS junk" fix has no bank-dependent datum to remove inside the shipped tile; we
// still defensively zero the corr_exp tile (the only observed bank-divergent region). 0 = off.
//
// VERDICT (2026-06-05): this fix is a NO-OP. (a) Fast-venue DPRINT (FORCE_ODD_BANK 0 vs 1, pos255
// worker cores 0..2 and pos127) proved the packed MS tile (rows 256..271) is ALREADY bit-identical
// bank0==bank1; the only bank-divergent region is the corr_exp scratch (rows 272..273 odd lanes),
// which is not packed and is inert at single chunk. (b) With FIX=1 the corr_exp tile reads 0 on BOTH
// banks and the MS max/sum is preserved (verified), yet the decoder pos255 result is BIT-FOR-BIT the
// baseline: iters=1 vs 2 differ 6543/7168, max delta 0.109375, shard0 0.98972 vs 0.98995. So the
// #43563 carrier is NOT in compute_sdpa_chunk's DEST (MS or corr_exp) -- it is downstream (sdpa_tail
// / MoE). Left OFF.
#define FIX_MS_JUNK_43563 0
// Verification dump for the fix (DPRINT the MS + corr_exp tiles AFTER the fix). 0 = off.
#define FIX_VERIFY_43563 0
#if defined(COMPILE_FOR_TRISC) && defined(FIX_VERIFY_43563) && FIX_VERIFY_43563
#include "api/debug/dprint.h"
#include "api/debug/dprint_tensix.h"
#endif

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
static_assert(noc_mode == DM_DYNAMIC_NOC, "Flash MLA Decode kernel only supports DM_DYNAMIC_NOC");
#endif

// ============================================================================
// BRISC helpers (Reader)
// ============================================================================
#if defined(COMPILE_FOR_BRISC)
template <typename Accessor>
FORCE_INLINE uint64_t get_shard_noc_addr_helper(const Accessor& reader, uint32_t shard_id, uint8_t noc = noc_index) {
    return reader.get_shard_noc_addr(shard_id, 0, noc);
}

constexpr uint32_t MCAST_INVALID = 0;
constexpr uint32_t MCAST_VALID = 1;
#endif

// ============================================================================
// NCRISC helpers (Writer)
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
template <uint32_t bits_per_step>
FORCE_INLINE constexpr uint32_t step_semaphore_inc(uint32_t step, uint32_t sub_bit = 0) {
    return 1U << (step * bits_per_step + sub_bit);
}
template <uint32_t bits_per_step>
FORCE_INLINE constexpr uint32_t step_semaphore_shift(uint32_t step, uint32_t sub_bit = 0) {
    return step * bits_per_step + sub_bit;
}

FORCE_INLINE void mask_last_chunk(
    uint32_t cb_mask, uint32_t k_chunk_size, uint32_t cur_pos, uint32_t k_chunk_end, uint32_t k_num_chunks) {
    bool mask_last_chunk = k_chunk_end == k_num_chunks && (cur_pos + 1) % k_chunk_size != 0;
    if (mask_last_chunk) {
        DeviceZoneScopedN("mask-last-chunk");
        cb_reserve_back(cb_mask, 1);
        volatile tt_l1_ptr uint32_t* mask_write_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_mask));
        uint32_t num_unmasked = cur_pos % k_chunk_size + 1;
        uint32_t i = 0;
        for (; i < num_unmasked / 2; i++) {
            *mask_write_ptr++ = 0x00000000;
        }
        if (num_unmasked % 2 == 1) {
            *mask_write_ptr++ = 0xFF800000;
            i++;
        }
        for (; i < k_chunk_size / 2; i++) {
            *mask_write_ptr++ = 0xFF80FF80;
        }
        cb_push_back(cb_mask, 1);
    }
#if defined(FIX_FULL_VIA_MASK_43563) && FIX_FULL_VIA_MASK_43563
    else {
        // #43562/3 FIX: this core has no real partial-mask chunk. Push ONE all-ZERO (all-pass) mask tile
        // so the compute can route its FULL chunks through the bank-deterministic MASK path (mm1 = 0 +
        // Q@K). All k_chunk_size positions are unmasked -> additive 0x00000000 (bf16 +0 in both halves).
        cb_reserve_back(cb_mask, 1);
        volatile tt_l1_ptr uint32_t* zero_mask_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_mask));
        for (uint32_t i = 0; i < k_chunk_size / 2; i++) {
            *zero_mask_ptr++ = 0x00000000;
        }
        cb_push_back(cb_mask, 1);
    }
#endif
}
#endif

namespace deepseek_b1_ops {

// ============================================================================
// Flash MLA Decode micro-op
//
// Implements flash attention decode for MLA where K and V share the same tensor.
//   BRISC (Reader): Read Q from sharded memory, read K from ND-sharded DRAM
//   NCRISC (Writer):  Multicast K data to S block receivers, tree reduction
//   TRISC (Compute): SDPA compute with flash attention chunking and tree reduction
// ============================================================================
struct FlashMLADecode {
    // ========================================================================
    // Args structs - different layout per RISC.
    // Includes both per-core runtime values and compile-time constants.
    // ========================================================================

    template <uint32_t k_page_size_, uint32_t vDHt_, uint32_t cb_out_o_, bool use_alt_mcast_vc_ = false>
    struct WriterCTArgs {
        static constexpr uint32_t k_page_size = k_page_size_;
        static constexpr uint32_t vDHt = vDHt_;
        static constexpr uint32_t cb_out_o = cb_out_o_;
        static constexpr bool use_alt_mcast_vc = use_alt_mcast_vc_;
    };

    struct ReaderCTArgs {};

    template <
        uint32_t cb_q_in_,
        uint32_t cb_k_in_,
        uint32_t cb_mask_,
        uint32_t cb_interm_out_,
        uint32_t cb_interm_ms_,
        uint32_t cb_out_in_,
        uint32_t cb_ms_in_,
        uint32_t cb_out_o_,
        uint32_t cb_out_ms_,
        uint32_t cb_out_final_,
        uint32_t cb_iter1_dump_>  // DEBUG #43563
    struct ComputeCTArgs {
        static constexpr uint32_t cb_q_in = cb_q_in_;
        static constexpr uint32_t cb_k_in = cb_k_in_;
        static constexpr uint32_t cb_mask = cb_mask_;
        static constexpr uint32_t cb_interm_out = cb_interm_out_;
        static constexpr uint32_t cb_interm_ms = cb_interm_ms_;
        static constexpr uint32_t cb_out_in = cb_out_in_;
        static constexpr uint32_t cb_ms_in = cb_ms_in_;
        static constexpr uint32_t cb_out_o = cb_out_o_;
        static constexpr uint32_t cb_out_ms = cb_out_ms_;
        static constexpr uint32_t cb_out_final = cb_out_final_;
        static constexpr uint32_t cb_iter1_dump = cb_iter1_dump_;  // DEBUG #43563
    };

    struct ReaderArgs {
        uint32_t k_addr;
        uint32_t local_cur_pos;
        uint32_t slot_id;
        uint32_t cur_batch;
        uint32_t core_num_in_reduce;
        uint32_t is_mcast_sender;
        uint32_t mcast_start_x;
        uint32_t mcast_start_y;
        uint32_t mcast_end_x;
        uint32_t mcast_end_y;
        uint32_t num_mcast_dests;
        uint32_t vc;
        uint32_t St;
        uint32_t DHt;
        uint32_t Sk_chunk_t;
        uint32_t num_cores_per_head;
        uint32_t k_chunk_size;
        uint32_t mcast_semaphore_addr;
        uint32_t k_page_size;
        uint32_t k_num_pages;
        uint32_t ncrisc_brisc_sync_semaphore_addr;
        uint32_t receiver_ready_semaphore_addr;
        uint32_t kv_cache_cur_pos_ready_semaphore_addr;
        uint32_t kv_cache_cur_pos_ready_value;
        uint32_t cb_k_in;
    };

    struct WriterArgs {
        uint32_t local_cur_pos;
        uint32_t slot_id;
        uint32_t cur_batch;
        uint32_t core_num_in_reduce;
        uint32_t is_output_core;
        uint32_t is_mcast_sender;
        uint32_t output_core_noc_x;
        uint32_t output_core_noc_y;
        uint32_t mcast_start_x;
        uint32_t mcast_start_y;
        uint32_t mcast_end_x;
        uint32_t mcast_end_y;
        tt_l1_ptr uint32_t* tree_reduction_info;
        uint32_t Sk_chunk_t;
        uint32_t num_cores_per_head;
        uint32_t reducer_semaphore_addr;
        uint32_t k_chunk_size;
        uint32_t q_chunk_size_bytes;
        uint32_t DHt;
        uint32_t num_mcast_dests;
        uint32_t full_grid_mcast_start_x;
        uint32_t full_grid_mcast_start_y;
        uint32_t full_grid_mcast_end_x;
        uint32_t full_grid_mcast_end_y;
        uint32_t full_grid_mcast_num_dests;
        uint32_t q_input_mcast_semaphore_addr;
        uint32_t mcast_semaphore_addr;
        uint32_t ncrisc_brisc_sync_semaphore_addr;
        uint32_t k_num_pages;
        uint32_t num_tree_reduction_steps;
        uint32_t receiver_ready_semaphore_addr;
        uint32_t cb_k_in;
        uint32_t cb_q_in;
        uint32_t cb_mask;
        uint32_t cb_out_in;
        uint32_t cb_ms_in;
        uint32_t cb_out_ms;
    };

    struct ComputeArgs {
        uint32_t local_cur_pos;
        uint32_t do_reduce;
        uint32_t do_output;
        uint32_t slot_id;
        uint32_t cur_batch;
        uint32_t core_num_in_reduce;
        uint32_t is_sender_after_reduce;
        tt_l1_ptr uint32_t* tree_reduction_info;
        uint32_t k_chunk_size;
        uint32_t num_cores_per_head;
        uint32_t num_tree_reduction_steps;
    };

    using RTArgs = unified_kernels::SelectByRISCV<WriterArgs, ReaderArgs, ComputeArgs>;

    // ========================================================================
    // Op - templated on CTArgs (compile-time args) and IsActiveCore
    // ========================================================================
    template <typename CTArgs, bool IsActiveCore>
    class Op {
    public:
        void operator()(const RTArgs& args) {
            if constexpr (IsActiveCore) {
                impl(args);
            }
        }

        void set_pos_and_slot(RTArgs& args, uint32_t local_cur_pos, uint32_t slot_id) {
            args.local_cur_pos = local_cur_pos;
            args.slot_id = slot_id;
        }

        /**
         * Push dummy tiles into the hand-off CBs (cb_out_o, cb_out_ms) so that
         * downstream SDPA reduce does not hang when Flash MLA is skipped on this
         * device (e.g. SP2/SP3 with no sequence data). Call on S1 cores only.
         *
         * TODO: Fuse the final SP reduce into Flash MLA and handle this internally,
         * eliminating the need for callers to manage the dummy push.
         */
        static void push_dummy_sdpa_inputs() {
#if defined(COMPILE_FOR_TRISC)
            constexpr uint32_t cb_out_o = CTArgs::cb_out_o;
            constexpr uint32_t cb_out_ms = CTArgs::cb_out_ms;
            constexpr uint32_t Sq_chunk_t = get_named_compile_time_arg_val("PNHt");
            constexpr uint32_t vDHt = get_named_compile_time_arg_val("vDHt");
            constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

            cb_reserve_back(cb_out_ms, 1);
            cb_push_back(cb_out_ms, 1);
            cb_reserve_back(cb_out_o, out_chunk_tiles);
            cb_push_back(cb_out_o, out_chunk_tiles);
#endif
        }

    private:
        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
            constexpr uint8_t MCAST_NOC_INDEX = 0;
            constexpr uint8_t ATOMIC_NOC_INDEX = 1;
            constexpr uint32_t BRISC_MCAST_LOOPS = 2;
            noc_async_write_set_trid(0, MCAST_NOC_INDEX);
#endif
// ====================================================================
// BRISC (Reader)
// ====================================================================
#if defined(COMPILE_FOR_BRISC)
            constexpr uint8_t READ_NOC_INDEX = 0;
            constexpr auto k_tensor_args = TensorAccessorArgs<0>();

            const bool is_mcast_sender = args.is_mcast_sender == 1;

            uint32_t cur_pos = args.local_cur_pos;

            auto [k_num_chunks, k_chunk_start, k_chunk_end] = get_runtime_args(
                cur_pos, args.cur_batch, args.core_num_in_reduce, args.num_cores_per_head, args.k_chunk_size);

#if defined(DUMP_CHUNKRANGE_43563) && DUMP_CHUNKRANGE_43563
            // #43563 STEP0 ground-truth: one line per core. per_core = ceil((end-start)/stride).
            {
                uint32_t _stride = args.num_cores_per_head;
                uint32_t _per_core =
                    (k_chunk_end > k_chunk_start) ? ((k_chunk_end - k_chunk_start + _stride - 1) / _stride) : 0;
                DPRINT << "CHUNKRANGE_43563 batch=" << args.cur_batch << " core=" << args.core_num_in_reduce
                       << " lpos=" << cur_pos << " nchunks=" << k_num_chunks << " kstart=" << k_chunk_start
                       << " kend=" << k_chunk_end << " stride=" << _stride << " per_core=" << _per_core << ENDL();
            }
#endif

            volatile tt_l1_ptr uint32_t* kv_cache_cur_pos_ready_semaphore_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.kv_cache_cur_pos_ready_semaphore_addr);

            if (k_chunk_start == k_chunk_end) {
                return;
            }

            const uint32_t k_chunk_tiles = args.Sk_chunk_t * args.DHt;

            const auto k_reader = TensorAccessor(k_tensor_args, args.k_addr);

            const uint32_t num_chunks_per_batch = args.St / args.Sk_chunk_t;

            volatile tt_l1_ptr uint32_t* mcast_semaphore_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.mcast_semaphore_addr);

            volatile tt_l1_ptr uint32_t* ncrisc_brisc_sync_curr_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.ncrisc_brisc_sync_semaphore_addr);
            volatile tt_l1_ptr uint32_t* ncrisc_brisc_sync_next_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.ncrisc_brisc_sync_semaphore_addr + 4);
            volatile tt_l1_ptr uint32_t* k_write_curr_ptr_shared =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.ncrisc_brisc_sync_semaphore_addr + 8);
            volatile tt_l1_ptr uint32_t* k_write_next_ptr_shared =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.ncrisc_brisc_sync_semaphore_addr + 12);
            // Reset the other semaphores outside the base offset to 0
            noc_semaphore_set(ncrisc_brisc_sync_next_ptr, 0);

            volatile tt_l1_ptr uint32_t* receiver_ready_semaphore_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.receiver_ready_semaphore_addr);
            const uint64_t sender_receiver_ready_noc_addr = get_noc_addr(
                args.mcast_start_x, args.mcast_start_y, args.receiver_ready_semaphore_addr, ATOMIC_NOC_INDEX);

            const uint64_t brisc_mcast_noc_addr = get_noc_multicast_addr<MCAST_NOC_INDEX>(
                args.mcast_start_x, args.mcast_start_y, args.mcast_end_x, args.mcast_end_y, 0);
            const uint64_t brisc_mcast_sem_addr = brisc_mcast_noc_addr | args.mcast_semaphore_addr;
            const uint32_t k_chunk_total_size = args.k_num_pages * args.k_page_size;

            // Only the core handling the last chunk needs to wait for the KV cache cur pos ready
            bool wait_for_kv_cache_ready = k_chunk_end == k_num_chunks;
            uint32_t loop_iter = 0;
            for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; k_chunk += args.num_cores_per_head) {
                {
                    DeviceZoneScopedN("reader-k-read");

                    cb_reserve_back(args.cb_k_in, k_chunk_tiles);
                    uint32_t k_write_ptr = get_write_ptr(args.cb_k_in);

                    if (is_mcast_sender && loop_iter < BRISC_MCAST_LOOPS) {
                        DeviceZoneScopedN("mcast-sender-serialized-read-and-mcast");
                        const uint32_t shard_id = args.slot_id * num_chunks_per_batch + k_chunk;
                        uint64_t k_src_noc_addr = get_shard_noc_addr_helper(k_reader, shard_id, READ_NOC_INDEX);

                        if (wait_for_kv_cache_ready && (k_chunk + args.num_cores_per_head) >= k_chunk_end) {
                            DeviceZoneScopedN("wait-for-kv-cache-ready");
                            noc_semaphore_wait(kv_cache_cur_pos_ready_semaphore_ptr, args.kv_cache_cur_pos_ready_value);
                            noc_semaphore_set(kv_cache_cur_pos_ready_semaphore_ptr, 0);
                        }

                        {
                            DeviceZoneScopedN("noc-read");
                            noc_async_read(k_src_noc_addr, k_write_ptr, k_chunk_total_size, READ_NOC_INDEX);
                            noc_async_read_barrier(READ_NOC_INDEX);
                        }

                        {
                            DeviceZoneScopedN("noc-multicast");
                            noc_semaphore_wait(receiver_ready_semaphore_ptr, args.num_mcast_dests);
                            noc_semaphore_set(receiver_ready_semaphore_ptr, 0);

                            uint64_t mcast_dest_addr = brisc_mcast_noc_addr | k_write_ptr;
                            noc_async_write_multicast(
                                k_write_ptr,
                                mcast_dest_addr,
                                k_chunk_total_size,
                                args.num_mcast_dests,
                                false,
                                MCAST_NOC_INDEX);

                            noc_semaphore_set(mcast_semaphore_ptr, MCAST_VALID);
                            noc_semaphore_set_multicast(
                                args.mcast_semaphore_addr,
                                brisc_mcast_sem_addr,
                                args.num_mcast_dests,
                                false,
                                MCAST_NOC_INDEX);
                            noc_async_writes_flushed(MCAST_NOC_INDEX);
                        }
                    } else if (is_mcast_sender) {
                        DeviceZoneScopedN("mcast-sender-sharded-read");
                        const uint32_t shard_id = args.slot_id * num_chunks_per_batch + k_chunk;
                        uint64_t k_src_noc_addr = get_shard_noc_addr_helper(k_reader, shard_id, READ_NOC_INDEX);

                        if (loop_iter == BRISC_MCAST_LOOPS) {
                            noc_async_read_one_packet_set_state<true>(
                                k_src_noc_addr, args.k_page_size, args.vc, READ_NOC_INDEX);
                            reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, READ_NOC_INDEX);
                        }

                        constexpr uint32_t NUM_TRIDS = NOC_MAX_TRANSACTION_ID - 1;
                        uint32_t src_base_addr = (uint32_t)(k_src_noc_addr & 0xFFFFFFFF);
                        uint32_t src_offset = 0;
                        uint32_t dst_addr = k_write_ptr;

                        uint32_t curr_trid = 1;
                        uint32_t wait_trid = 1;
                        uint32_t pages_issued = 0;
                        uint32_t pages_completed = 0;
                        if (wait_for_kv_cache_ready && (k_chunk + args.num_cores_per_head) >= k_chunk_end) {
                            noc_semaphore_wait(kv_cache_cur_pos_ready_semaphore_ptr, args.kv_cache_cur_pos_ready_value);
                            noc_semaphore_set(kv_cache_cur_pos_ready_semaphore_ptr, 0);
                        }

                        noc_semaphore_wait(ncrisc_brisc_sync_curr_ptr, 0);
                        *k_write_curr_ptr_shared = k_write_ptr;
                        for (uint32_t i = 0; i < NUM_TRIDS && pages_issued < args.k_num_pages; ++i) {
                            noc_async_read_set_trid(curr_trid, READ_NOC_INDEX);
                            noc_async_read_one_packet_with_state_with_trid(
                                src_base_addr, src_offset, dst_addr, curr_trid, READ_NOC_INDEX);
                            src_offset += args.k_page_size;
                            dst_addr += args.k_page_size;
                            curr_trid = (curr_trid % NUM_TRIDS) + 1;
                            pages_issued++;
                        }

                        while (pages_completed < args.k_num_pages) {
                            noc_async_read_barrier_with_trid(wait_trid, READ_NOC_INDEX);
                            *ncrisc_brisc_sync_curr_ptr += 1;
                            pages_completed++;

                            if (pages_issued < args.k_num_pages) {
                                noc_async_read_set_trid(curr_trid, READ_NOC_INDEX);
                                noc_async_read_one_packet_with_state_with_trid(
                                    src_base_addr, src_offset, dst_addr, curr_trid, READ_NOC_INDEX);
                                src_offset += args.k_page_size;
                                dst_addr += args.k_page_size;
                                curr_trid = (curr_trid % NUM_TRIDS) + 1;
                                pages_issued++;
                            }

                            wait_trid = (wait_trid % NUM_TRIDS) + 1;
                        }

                        std::swap(ncrisc_brisc_sync_curr_ptr, ncrisc_brisc_sync_next_ptr);
                        std::swap(k_write_curr_ptr_shared, k_write_next_ptr_shared);
                    } else {
                        DeviceZoneScopedN("mcast-receiver-signal-ready");
                        noc_semaphore_inc(sender_receiver_ready_noc_addr, 1, ATOMIC_NOC_INDEX);

                        noc_semaphore_wait(mcast_semaphore_ptr, MCAST_VALID);
                        noc_semaphore_set(mcast_semaphore_ptr, MCAST_INVALID);
                    }

                    cb_push_back(args.cb_k_in, k_chunk_tiles);
                }
                loop_iter++;
            }
            noc_async_write_barrier(MCAST_NOC_INDEX);
// ====================================================================
// NCRISC (Writer)
// ====================================================================
#elif defined(COMPILE_FOR_NCRISC)
            constexpr uint8_t READ_NOC_INDEX = 1;
            constexpr uint8_t WRITE_NOC_INDEX = 1;

            constexpr uint32_t k_page_size = CTArgs::k_page_size;
            constexpr uint32_t vDHt = CTArgs::vDHt;
            constexpr uint32_t cb_out_o = CTArgs::cb_out_o;

            constexpr uint32_t out_chunk_tiles = vDHt;
            constexpr uint32_t tile_bytes_intermed = get_tile_size(cb_out_o);
            constexpr uint32_t o_write_size = out_chunk_tiles * tile_bytes_intermed;
            constexpr uint32_t ms_write_size = tile_bytes_intermed;
            constexpr uint32_t q_mcast_vc =
                CTArgs::use_alt_mcast_vc ? NOC_DISPATCH_MULTICAST_WRITE_VC : NOC_MULTICAST_WRITE_VC;

            const uint32_t q_chunk_tiles = args.DHt;

            volatile tt_l1_ptr uint32_t* q_input_mcast_semaphore_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.q_input_mcast_semaphore_addr);

            const bool is_mcast_sender = args.is_mcast_sender == 1;
            const bool is_output_core = args.is_output_core == 1;

            uint32_t cur_pos = args.local_cur_pos;

            auto [k_num_chunks, k_chunk_start, k_chunk_end] = get_runtime_args(
                cur_pos, args.cur_batch, args.core_num_in_reduce, args.num_cores_per_head, args.k_chunk_size);

            {
                DeviceZoneScopedN("reader-q-read");
                if (is_output_core) {
                    cb_wait_front(args.cb_q_in, q_chunk_tiles);
                    if (is_mcast_sender) {
                        uint64_t q_input_mcast_sem_noc_addr = get_noc_multicast_addr<MCAST_NOC_INDEX>(
                            args.full_grid_mcast_start_x,
                            args.full_grid_mcast_start_y,
                            args.full_grid_mcast_end_x,
                            args.full_grid_mcast_end_y,
                            args.q_input_mcast_semaphore_addr);
                        noc_semaphore_wait(q_input_mcast_semaphore_ptr, args.num_mcast_dests);
                        noc_semaphore_inc_multicast(
                            q_input_mcast_sem_noc_addr, 1, args.full_grid_mcast_num_dests, MCAST_NOC_INDEX, q_mcast_vc);
                        mask_last_chunk(args.cb_mask, args.k_chunk_size, cur_pos, k_chunk_end, k_num_chunks);
                        // This is needed because we need to wait for all transactions before resetting the trids
                        // Could move it later but don't think it makes much difference
                        noc_async_atomic_barrier(MCAST_NOC_INDEX);
                    } else {
                        const uint64_t sender_receiver_ready_noc_addr = get_noc_addr(
                            args.mcast_start_x,
                            args.mcast_start_y,
                            args.q_input_mcast_semaphore_addr,
                            ATOMIC_NOC_INDEX);
                        noc_semaphore_inc(sender_receiver_ready_noc_addr, 1, ATOMIC_NOC_INDEX);
                        mask_last_chunk(args.cb_mask, args.k_chunk_size, cur_pos, k_chunk_end, k_num_chunks);
                        noc_async_atomic_barrier(ATOMIC_NOC_INDEX);
                        noc_semaphore_wait(q_input_mcast_semaphore_ptr, 1);
                    }
                } else if (k_chunk_start == k_chunk_end) {
                    noc_semaphore_wait(q_input_mcast_semaphore_ptr, 1);
                } else {
                    // wait for 8 q heads
                    uint64_t q_noc_addr = get_noc_addr(
                        args.output_core_noc_x, args.output_core_noc_y, get_read_ptr(args.cb_q_in), READ_NOC_INDEX);
                    cb_reserve_back(args.cb_q_in, q_chunk_tiles);
                    noc_semaphore_wait(q_input_mcast_semaphore_ptr, 1);
                    noc_async_read(q_noc_addr, get_write_ptr(args.cb_q_in), args.q_chunk_size_bytes, READ_NOC_INDEX);
                    mask_last_chunk(args.cb_mask, args.k_chunk_size, cur_pos, k_chunk_end, k_num_chunks);
                    noc_async_read_barrier(READ_NOC_INDEX);
#if defined(ZERO_Q_INPUT_43563) && ZERO_Q_INPUT_43563
                    // #43562/3 EXPERIMENT: overwrite the just-read Q with zeros so mm1 = 0 @ K = 0.
                    {
                        volatile tt_l1_ptr uint32_t* _qz =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(args.cb_q_in));
                        for (uint32_t _i = 0; _i < args.q_chunk_size_bytes / 4; _i++) {
                            _qz[_i] = 0;
                        }
                    }
#endif
                    cb_push_back(args.cb_q_in, q_chunk_tiles);
                }
                noc_semaphore_set(q_input_mcast_semaphore_ptr, 0);
            }

            if (k_chunk_start == k_chunk_end) {
                return;
            }

            // =================================================================
            // KV Cache Multicast (page-level pipelining)
            // Skip first BRISC_MCAST_LOOPS iterations — handled by BRISC
            // =================================================================
            const uint32_t num_k_chunks = k_chunk_end - k_chunk_start;
            const uint32_t num_loop_iters = (num_k_chunks + args.num_cores_per_head - 1) / args.num_cores_per_head;
            if (is_mcast_sender && num_loop_iters > BRISC_MCAST_LOOPS) {
                volatile tt_l1_ptr uint32_t* mcast_semaphore_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.mcast_semaphore_addr);

                volatile tt_l1_ptr uint32_t* ncrisc_brisc_sync_curr_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.ncrisc_brisc_sync_semaphore_addr);
                volatile tt_l1_ptr uint32_t* ncrisc_brisc_sync_next_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.ncrisc_brisc_sync_semaphore_addr + 4);
                volatile tt_l1_ptr uint32_t* k_write_curr_ptr_shared =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.ncrisc_brisc_sync_semaphore_addr + 8);
                volatile tt_l1_ptr uint32_t* k_write_next_ptr_shared =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.ncrisc_brisc_sync_semaphore_addr + 12);

                volatile tt_l1_ptr uint32_t* receiver_ready_semaphore_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.receiver_ready_semaphore_addr);

                const uint64_t mcast_noc_addr = get_noc_multicast_addr<MCAST_NOC_INDEX>(
                    args.mcast_start_x, args.mcast_start_y, args.mcast_end_x, args.mcast_end_y, 0);
                const uint64_t mcast_sem_addr = mcast_noc_addr | args.mcast_semaphore_addr;

                noc_semaphore_set(mcast_semaphore_ptr, 1);

                for (uint32_t k_chunk = k_chunk_start + BRISC_MCAST_LOOPS * args.num_cores_per_head;
                     k_chunk < k_chunk_end;
                     k_chunk += args.num_cores_per_head) {
                    DeviceZoneScopedN("mcast-sender-multicast");

                    noc_semaphore_wait_min(ncrisc_brisc_sync_curr_ptr, 1);
                    invalidate_l1_cache();
                    uint32_t page_addr = *k_write_curr_ptr_shared;

                    uint64_t mcast_dest_addr = mcast_noc_addr | page_addr;

                    noc_semaphore_wait(receiver_ready_semaphore_ptr, args.num_mcast_dests);
                    noc_semaphore_set(receiver_ready_semaphore_ptr, 0);

                    noc_async_write_multicast<k_page_size>(
                        page_addr, mcast_dest_addr, k_page_size, args.num_mcast_dests, false, MCAST_NOC_INDEX);

                    for (uint32_t page = 1; page < args.k_num_pages; ++page) {
                        page_addr += k_page_size;
                        mcast_dest_addr = mcast_noc_addr | page_addr;
                        noc_semaphore_wait_min(ncrisc_brisc_sync_curr_ptr, page + 1);
                        noc_async_write_multicast<k_page_size>(
                            page_addr, mcast_dest_addr, k_page_size, args.num_mcast_dests, false, MCAST_NOC_INDEX);
                    }

                    noc_semaphore_set_multicast(
                        args.mcast_semaphore_addr, mcast_sem_addr, args.num_mcast_dests, false, MCAST_NOC_INDEX);
                    noc_async_writes_flushed(MCAST_NOC_INDEX);
                    *ncrisc_brisc_sync_curr_ptr = 0;
                    std::swap(ncrisc_brisc_sync_curr_ptr, ncrisc_brisc_sync_next_ptr);
                    std::swap(k_write_curr_ptr_shared, k_write_next_ptr_shared);
                }
                noc_async_write_barrier(MCAST_NOC_INDEX);
            }

            // =================================================================
            // Tree Reduction
            // =================================================================
            constexpr uint32_t bits_per_step = 2;
            constexpr uint32_t ms_sub_bit = 0;
            constexpr uint32_t o_sub_bit = 1;

            volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.reducer_semaphore_addr);

            uint32_t num_active_s_blocks =
                (k_num_chunks < args.num_cores_per_head) ? k_num_chunks : args.num_cores_per_head;
            bool needs_reduction = (num_active_s_blocks > 1);
            uint32_t cb_ms_in_base_addr = get_write_ptr(args.cb_ms_in);
            uint32_t cb_out_in_base_addr = get_write_ptr(args.cb_out_in);

            if (needs_reduction) {
                for (uint32_t step = 0; step < args.num_tree_reduction_steps; ++step) {
                    DeviceZoneScopedN("tree-reduction-step");
                    uint32_t role_code = args.tree_reduction_info[step * 4 + 0];
                    uint32_t partner_s_block_idx = args.tree_reduction_info[step * 4 + 1];
                    uint32_t partner_x = args.tree_reduction_info[step * 4 + 2];
                    uint32_t partner_y = args.tree_reduction_info[step * 4 + 3];

                    if (role_code != 0 && partner_s_block_idx >= num_active_s_blocks) {
                        continue;
                    }

                    if (role_code == 1) {
                        DeviceZoneScopedN("tree-reduction-sender");
                        uint32_t inc_value = step_semaphore_inc<bits_per_step>(step, ms_sub_bit);
                        uint64_t output_write_coord = get_noc_addr(partner_x, partner_y, 0, WRITE_NOC_INDEX);
                        uint64_t partner_semaphore_addr = output_write_coord | args.reducer_semaphore_addr;
                        uint64_t output_write_addr = output_write_coord | (cb_ms_in_base_addr + step * ms_write_size);
                        cb_wait_front(args.cb_out_ms, 1);
                        noc_async_write<ms_write_size, false, /*posted=*/true>(
                            get_read_ptr(args.cb_out_ms), output_write_addr, ms_write_size, WRITE_NOC_INDEX);
                        noc_semaphore_inc(partner_semaphore_addr, inc_value, WRITE_NOC_INDEX);
                        inc_value = step_semaphore_inc<bits_per_step>(step, o_sub_bit);
                        output_write_addr = output_write_coord | (cb_out_in_base_addr + step * o_write_size);
                        cb_wait_front(cb_out_o, out_chunk_tiles);
                        noc_async_write<o_write_size, false, /*posted=*/true>(
                            get_read_ptr(cb_out_o), output_write_addr, o_write_size, WRITE_NOC_INDEX);
                        noc_semaphore_inc(partner_semaphore_addr, inc_value, WRITE_NOC_INDEX);

                        noc_async_posted_writes_flushed(WRITE_NOC_INDEX);
                        cb_pop_front(args.cb_out_ms, 1);
                        cb_pop_front(cb_out_o, out_chunk_tiles);
                        noc_async_atomic_barrier(WRITE_NOC_INDEX);
                        break;

                    } else if (role_code == 2) {
                        DeviceZoneScopedN("tree-reduction-receiver");
                        uint32_t shift_value = step_semaphore_shift<bits_per_step>(step, ms_sub_bit);
                        cb_reserve_back(args.cb_ms_in, 1);
                        uint32_t sem_val;
                        do {
                            invalidate_l1_cache();
                            sem_val = *in0_receiver_semaphore_addr_ptr;
                        } while (((sem_val >> shift_value) & 1U) == 0);
                        cb_push_back(args.cb_ms_in, 1);

                        shift_value = step_semaphore_shift<bits_per_step>(step, o_sub_bit);
                        cb_reserve_back(args.cb_out_in, out_chunk_tiles);
                        do {
                            invalidate_l1_cache();
                            sem_val = *in0_receiver_semaphore_addr_ptr;
                        } while (((sem_val >> shift_value) & 1U) == 0);
                        cb_push_back(args.cb_out_in, out_chunk_tiles);
                    }
                }
                noc_semaphore_set(in0_receiver_semaphore_addr_ptr, 0);
            }

// ====================================================================
// TRISC (Compute)
// ====================================================================
#elif defined(COMPILE_FOR_TRISC)
            constexpr uint32_t DHt = get_named_compile_time_arg_val("DHt");
            constexpr uint32_t vDHt = get_named_compile_time_arg_val("vDHt");
            constexpr uint32_t Sq_chunk_t = get_named_compile_time_arg_val("PNHt");
            constexpr uint32_t Sk_chunk_t = get_named_compile_time_arg_val("Sk_chunk_t");
            constexpr uint32_t scale_fp32 = get_named_compile_time_arg_val("scale_fp32");
            constexpr uint32_t dst_size = get_named_compile_time_arg_val("dst_size");
            constexpr uint32_t cb_q_in = CTArgs::cb_q_in;
            constexpr uint32_t cb_k_in = CTArgs::cb_k_in;
            constexpr uint32_t cb_mask = CTArgs::cb_mask;
            constexpr uint32_t cb_interm_out = CTArgs::cb_interm_out;
            constexpr uint32_t cb_interm_ms = CTArgs::cb_interm_ms;
            constexpr uint32_t cb_out_in = CTArgs::cb_out_in;
            constexpr uint32_t cb_ms_in = CTArgs::cb_ms_in;
            constexpr uint32_t cb_out_o = CTArgs::cb_out_o;
            constexpr uint32_t cb_out_ms = CTArgs::cb_out_ms;
            constexpr uint32_t cb_out_final = CTArgs::cb_out_final;
            constexpr uint32_t cb_iter1_dump = CTArgs::cb_iter1_dump;  // DEBUG #43563

            constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
            constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

            static_assert(out_chunk_tiles % 2 == 0, "out_chunk_tiles must be even");

            const bool do_reduce = args.do_reduce == 1;
            const bool do_output = args.do_output == 1;                            // set to 0 in fused
            const bool is_sender_after_reduce = args.is_sender_after_reduce == 1;  // set to 1 in fused

            constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

            constexpr bool transpose_k = true;
            constexpr bool transpose_v = false;

            reconfig_data_format<false, true>(cb_k_in, cb_q_in);
            pack_reconfig_data_format<true>(cb_out_o);
            PACK((llk_math_sfpu_sdpa_reduce_row_init<false, DST_ACCUM_MODE, DataFormat::Float16_b>()));
            PACK(SFPU_TEMPLATE_INIT_KERNEL(exponential, sfpu::exp_init, true, scale_fp32, true));
            sdpa_custom_mm_block_init_pack_short();
#if defined(DISABLE_DST_ZEROFLAG_43563) && DISABLE_DST_ZEROFLAG_43563
            // #43563 fix part 2 (see macro def): FPU exact-0 writes -> real zero data, not flag-only.
            MATH((cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_dst_RMW>(1)));
#endif

            uint32_t cur_pos = args.local_cur_pos;
            auto [k_num_chunks, k_chunk_start, k_chunk_end] = get_runtime_args(
                cur_pos, args.cur_batch, args.core_num_in_reduce, args.num_cores_per_head, args.k_chunk_size);
            if (k_chunk_start == k_chunk_end) {
                return;
            }

            uint32_t num_active_s_blocks =
                (k_num_chunks < args.num_cores_per_head) ? k_num_chunks : args.num_cores_per_head;

            uint32_t num_cores_to_wait = 0;
            for (uint32_t step = 0; step < args.num_tree_reduction_steps; ++step) {
                uint32_t role_code = args.tree_reduction_info[step * 2 + 0];
                uint32_t partner_s_block_idx = args.tree_reduction_info[step * 2 + 1];
                if (role_code == 2 && partner_s_block_idx < num_active_s_blocks) {
                    num_cores_to_wait++;
                }
            }

            constexpr uint32_t packed_tile_size = 8 * 2;
            constexpr uint32_t mm2_dst_offset = 0;
            constexpr uint32_t mm2_dst_tile_offset = mm2_dst_offset / packed_tile_size;
            constexpr uint32_t max_dst_offset = mm2_dst_offset + packed_tile_size * vDHt;
            constexpr uint32_t max_dst_tile_offset = max_dst_offset / packed_tile_size;
            constexpr uint32_t sum_dst_offset = max_dst_offset + 2;
            constexpr uint32_t corr_exp_dst_offset = max_dst_offset + packed_tile_size;
            constexpr uint32_t mm1_dst_offset = corr_exp_dst_offset + packed_tile_size;
            constexpr uint32_t mm1_dst_tile_offset = mm1_dst_offset / packed_tile_size;
            constexpr uint32_t sum_dst_tile_offset = sum_dst_offset / packed_tile_size;

            constexpr bool exp_approx_mode = false;

            constexpr uint32_t output_granularity = out_chunk_tiles;
            static_assert(
                out_chunk_tiles % output_granularity == 0, "out_chunk_tiles must be divisible by output_granularity");

            bool sdpa_output_is_final = do_output && (!do_reduce || num_cores_to_wait == 0);
#if defined(SDPA_STAGE_CUT_43563) && (SDPA_STAGE_CUT_43563 != 0)
            // #43563 STAGE CUT: on output cores, route the per-core intermediate straight to the
            // final output CB (no cross-core reduction). Forcing is_final=true makes the code below
            // take the "final" structural path; we then override the packed payload with the chosen
            // intermediate and the end-of-fn reduction block is gated off (see SDPA_STAGE_CUT below).
            if (do_output) {
                sdpa_output_is_final = true;
            }
#endif
            uint32_t sdpa_output_cb = 0;
            uint32_t sdpa_ms_cb = 0;
            if (sdpa_output_is_final) {
                sdpa_output_cb = cb_out_final;
                sdpa_ms_cb = cb_out_ms;
            } else if (num_cores_to_wait > 0) {
                sdpa_output_cb = cb_interm_out;
                sdpa_ms_cb = cb_interm_ms;
            } else {
                // Fused with sdpa reduce worker
                sdpa_output_cb = cb_out_o;
                sdpa_ms_cb = cb_out_ms;
            }
            pack_block_contiguous_init(sdpa_output_cb);
            uint32_t num_chunks = (k_chunk_end - k_chunk_start + args.num_cores_per_head - 1) / args.num_cores_per_head;
            bool mask_last_chunk = k_chunk_end == k_num_chunks && (cur_pos + 1) % args.k_chunk_size != 0;
#if defined(FIX_FULL_VIA_MASK_43563) && FIX_FULL_VIA_MASK_43563
            // #43562/3 FIX: on a core with NO real partial-mask chunk, the reader pushed ONE all-zero
            // mask tile. Wait on it so every FULL chunk can route its QK matmul through the clean MASK
            // path (mm1 = 0 + Q@K via the coherent MOVB2D write). use_zero_mask gates the per-chunk
            // mask_chunk=true override below and the matching pop after the loop.
            bool use_zero_mask = !mask_last_chunk;
            if (use_zero_mask) {
                cb_wait_front(cb_mask, 1);
            }
#endif
            if (mask_last_chunk) {
                cb_wait_front(cb_mask, 1);
            }
            cb_wait_front(cb_q_in, q_chunk_tiles);
            cb_reserve_back(sdpa_output_cb, vDHt);
            cb_reserve_back(sdpa_ms_cb, Sq_chunk_t);
            tile_regs_acquire();
#if defined(FILL_DEST_43563) && FILL_DEST_43563
            // #43563: pre-fill the active DEST half with a deterministic non-zero constant (cb_q) so
            // any DEST row the SFPU reduce later reads but the matmul doesn't overwrite holds
            // bank-INDEPENDENT data instead of bank-dependent leftover. Covers the whole half (32 tiles
            // of 16 rows = 512 rows). The matmul then overwrites the valid mm1; reduce-read padding =
            // cb_q (deterministic). If the iter-parity bug vanishes -> SFPU-reads-zero-flagged-leftover
            // confirmed as the root cause.
            copy_tile_to_dst_init_short(cb_q_in);
            for (uint32_t _t = 0; _t < 32; _t++) {
                copy_tile(cb_q_in, _t % q_chunk_tiles, _t);
            }
#endif
#if defined(SFPU_ZERO_DEST_43563) && SFPU_ZERO_DEST_43563
            // #43563 FIX: SFPU literal-zero of the whole DEST region the SDPA addresses, BEFORE the
            // QK^T matmul. Covers mm2/max/sum/corr_exp + all chunk mm1 tiles (rows 0 ..
            // mm1_dst_offset + Sk_chunk_t*packed_tile_size). Placed AFTER the optional FILL block so
            // the zero erases the FILL poison in the leftover rows the reduce reads. Uses sfpi
            // dst_reg = 0 stores (SFPSTORE), NOT a zero-flag clear. The matmul overwrites mm1 on top.
            {
                constexpr uint32_t sdpa_dest_rows = mm1_dst_offset + Sk_chunk_t * packed_tile_size;
                PACK((ckernel::sfpu_zero_dest_43563(sdpa_dest_rows)));
            }
#endif
            for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
                bool last_chunk = chunk == (num_chunks - 1);
                compute_sdpa_chunk<
                    Sk_chunk_t,
                    q_chunk_tiles,
                    out_chunk_tiles,
                    scale_fp32,
                    scale_bf16,
                    transpose_k,
                    transpose_v,
                    packed_tile_size,
                    exp_approx_mode,
                    output_granularity,
                    false>(
                    cb_q_in,
                    cb_k_in,
                    cb_mask,
                    sdpa_output_cb,
                    mm1_dst_offset,
                    mm2_dst_offset,
                    max_dst_offset,
                    sum_dst_offset,
                    corr_exp_dst_offset,
                    chunk == 0,
                    !sdpa_output_is_final && last_chunk,
#if defined(FIX_FULL_VIA_MASK_43563) && FIX_FULL_VIA_MASK_43563
                    // #43562/3 FIX: route EVERY full chunk through the MASK path (all-zero mask) when
                    // this core has no real partial mask; otherwise keep the original last-chunk mask.
                    use_zero_mask ? true : (mask_last_chunk && last_chunk),
#else
                    mask_last_chunk && last_chunk,
#endif
                    cb_iter1_dump);  // #43563 EXP I/O TAP side CB (used only if SDPA_EXPIO_TAP_43563)
#if defined(NOP_DRAIN_43563) && NOP_DRAIN_43563
                // #43563: Lazar's experiment (tt-blaze#842) — drain the MATH pipe BETWEEN chunks. The
                // next chunk's Q@K matmul was found to corrupt the previous chunk's SFPU sum partial
                // (a cross-chunk DEST/pipeline hazard). 50 NOPs between compute_sdpa_chunk iterations
                // fixed both 2-chunks/core cases (mask + no-mask). Only fires when num_chunks>1.
                for (uint32_t _nop = 0; _nop < 1000; _nop++) {
                    MATH(TTI_NOP);
                }
#endif
            }
#if defined(DUMP_MS_43563) && DUMP_MS_43563
            // #43563 MS-TILE DEST DUMP. Same proven-safe location/threading as the STUB blocks below:
            // MATH-stall on the max semaphore so the last chunk's SFPU reduce (which writes the final
            // max/sum into the MS tile) is fully settled in DEST, then read & print all 16 absolute
            // DEST rows of the MS tile (base = max_dst_offset). The reduce writes max into rows {0,4}
            // and sum into rows {2,6} (rel max_dst_offset); the rest are the suspected dirty leftover.
            // We use the low-level per-row float32 reader inside a MATH()/halt window because on
            // Blackhole it reads DEST directly at 0xFFBD8000 + (row<<4), matching the SDPA packed tiny-
            // tile layout exactly (dprint_tensix_dest_reg(tile_id) would assume 32 rows/tile and
            // mis-address this 16-row packed tile).
            MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
            DPRINT << "MSDUMP core begin max_dst_offset=" << (uint32_t)max_dst_offset
                   << " sum_dst_offset=" << (uint32_t)sum_dst_offset
                   << " max_dst_tile_offset=" << (uint32_t)max_dst_tile_offset
                   << " mm2_dst_tile_offset=" << (uint32_t)mm2_dst_tile_offset << ENDL();
            // Report what DEST format the SFPU/FPU left the accumulator in (drives which reader is
            // correct). Then dump 16 absolute rows of the MS tile (base=max_dst_offset) with BOTH
            // the float32 direct reader AND the float16_b reader so we can tell which layout holds
            // the data. Also dump mm2 tile 0 (a known-non-zero L tile) as a sanity reference.
            dbg_halt();
            MATH({
                uint32_t fmt = READ_HW_CFG_0_REG_FIELD(ALU_FORMAT_SPEC_REG2_Dstacc);
                uint32_t fp32en = READ_HW_CFG_0_REG_FIELD(ALU_ACC_CTRL_Fp32_enabled);
                uint32_t remap = READ_HW_CFG_0_REG_FIELD(DEST_ACCESS_CFG_remap_addrs);
                uint32_t swiz = READ_HW_CFG_0_REG_FIELD(DEST_ACCESS_CFG_swizzle_32b);
                DPRINT << "MSDUMP destfmt=" << fmt << " fp32en=" << fp32en << " remap=" << remap << " swiz=" << swiz
                       << ENDL();
                // RAW (no remap/swizzle): read 16 dwords straight from the DEST fp16 region so we can
                // tell whether the remap in dprint_tensix_dest_reg_row_float16 is mis-landing.
                {
                    const uint32_t* a = reinterpret_cast<const uint32_t*>(0xFFBD8000);
                    for (uint16_t r = 0; r < 8; r++) {
                        uint16_t ar = (uint16_t)(max_dst_offset + r);
                        DPRINT << "MSDUMP raw row " << (uint32_t)r << " :";
                        for (int i = 0; i < 8; i++) {
                            DPRINT << " " << HEX() << a[i + (ar << 3)] << DEC();
                        }
                        DPRINT << ENDL();
                    }
                }
                for (uint16_t r = 0; r < 16; r++) {
                    DPRINT << "MSDUMP f32 row " << (uint32_t)r << " : ";
                    dprint_tensix_dest_reg_row_float32((uint16_t)(max_dst_offset + r));
                    DPRINT << "MSDUMP f16 row " << (uint32_t)r << " : ";
                    dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(max_dst_offset + r));
                }
                DPRINT << "MSDUMP mm2t0 f32 row0 : ";
                dprint_tensix_dest_reg_row_float32((uint16_t)(mm2_dst_offset + 0));
                DPRINT << "MSDUMP mm2t0 f16 row0 : ";
                dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(mm2_dst_offset + 0));
            })
            dbg_unhalt();
            DPRINT << "MSDUMP core end" << ENDL();
#endif
#if defined(STUB_SDPA_43563) && STUB_SDPA_43563
            // #43563: FULL SDPA-output stub. Overwrite L (mm2 tiles 0..out_chunk_tiles-1) AND
            // MS (max/sum tile at max_dst_tile_offset) in DEST with cb_q -> deterministic,
            // iteration-independent. MATH stall first so the last chunk's SFPU (reduce_sum writing
            // sum into the max tile) is finished -> no race on the MS tile. Parity preserved (still
            // inside the one tile_regs region). If iters=1-vs-2 STILL differ, compute_sdpa_chunk is
            // FULLY ruled out and the bug is in sdpa_tail-or-later.
            MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
            copy_tile_to_dst_init_short(cb_q_in);
            for (uint32_t i = 0; i <= max_dst_tile_offset; i++) {
                copy_tile(cb_q_in, i, i);
            }
#endif
#if (defined(STUB_MAX_43563) && STUB_MAX_43563) || (defined(STUB_SUM_43563) && STUB_SUM_43563)
            // #43563 MAX/SUM SPLIT STUB. Same proven-safe location/threading as STUB_SDPA_43563:
            // MATH stall so the last chunk's SFPU reduce (which writes the final max/sum into the MS
            // tile) is fully done, then do the constant SFPU store on the PACK thread, sequenced after
            // the reduce. We overwrite ONLY the requested datums; the other (max or sum) stays real.
            MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
#if defined(STUB_MAX_43563) && STUB_MAX_43563
            PACK((ckernel::sfpu_const_subtile_43563(max_dst_offset, STUB_MAX_C_43563)));
#endif
#if defined(STUB_SUM_43563) && STUB_SUM_43563
            PACK((ckernel::sfpu_const_subtile_43563(sum_dst_offset, STUB_SUM_C_43563)));
#endif
#endif
#if defined(STUB_MS_JUNK_43563) && STUB_MS_JUNK_43563
            // #43563 CANDIDATE FIX. Same proven-safe location/threading as the STUB_MAX/SUM split:
            // MATH-stall on max so the last chunk's SFPU reduce has finished writing the MS tile,
            // then on the PACK thread SFPU-zero the MS tile's leftover rows. The MS tile is based at
            // DEST row max_dst_offset (= max_dst_tile_offset*packed_tile_size). =1 zeros only the junk
            // rows {1,3,5,7,8..15} (preserves max@{0,4}, sum@{2,6}); =2 zeros all 16 rows (control).
            MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
            PACK((ckernel::sfpu_zero_ms_junk_43563(max_dst_offset, STUB_MS_JUNK_43563)));
#endif
#if defined(FIX_MS_JUNK_43563) && FIX_MS_JUNK_43563
            // #43563 CANDIDATE FIX. MATH-stall on max so the last chunk's SFPU reduce has fully
            // written the MS tile, then on the PACK thread SFPU-zero (sfpi dst_reg = REAL 0) the ONLY
            // bank-dependent datums the DPRINT ground truth found: the corr_exp tile (base
            // corr_exp_dst_offset; its odd lanes are bank-dependent). The packed MS tile (base
            // max_dst_offset) is already bank-independent (verified), so we leave its real max/sum
            // (rows 0..7) untouched and only re-zero its unwritten upper rows {8..15} defensively.
            MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
            // corr_exp tile: holds only bank-dependent scratch at single chunk -> zero the whole tile.
            PACK((ckernel::sfpu_zero_ms_junk_43563(corr_exp_dst_offset, 2 /* zero all 16 rows */)));
            // MS tile upper half {8..15}: real zeros already, re-assert bank-independent 0.
            PACK((ckernel::sfpu_zero_ms_junk_43563(max_dst_offset, 3 /* zero rows 8..15 only */)));
#if defined(FIX_VERIFY_43563) && FIX_VERIFY_43563
            // Post-fix verification dump: read the MS tile (rows 0..15) AND the corr_exp tile
            // (rows 0..1) so we can confirm bank0==bank1 with real max/sum preserved + junk -> 0.
            MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
            DPRINT << "FIXDUMP begin" << ENDL();
            dbg_halt();
            MATH({
                for (uint16_t r = 0; r < 16; r++) {
                    DPRINT << "FIXDUMP ms f16 row " << (uint32_t)r << " : ";
                    dprint_tensix_dest_reg_row_float16((uint32_t)DataFormat::Float16_b, (uint16_t)(max_dst_offset + r));
                }
                for (uint16_t r = 0; r < 2; r++) {
                    DPRINT << "FIXDUMP corr f16 row " << (uint32_t)r << " : ";
                    dprint_tensix_dest_reg_row_float16(
                        (uint32_t)DataFormat::Float16_b, (uint16_t)(corr_exp_dst_offset + r));
                }
            })
            dbg_unhalt();
            DPRINT << "FIXDUMP end" << ENDL();
#endif
#endif
#if defined(SDPA_STAGE_CUT_43563) && (SDPA_STAGE_CUT_43563 != 0)
            // #43563 STAGE CUT: pack the chosen intermediate (still resident in DEST after the
            // chunk loop) into the output CB so the host can read it back per-core. We pad the
            // remaining out_chunk_tiles with mm2 leftover so the CB shard is fully populated; only
            // the meaningful low tiles are compared host-side. On non-output cores keep the
            // ORIGINAL behavior so the (now-unread) handoff CBs are still pushed -> no hang.
            if (do_output) {
                // Drain MATH so the last chunk's SFPU writes (mm1/max/sum) are settled in DEST.
                MATH(t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU));
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                // All stages below read DEST tiles the SDPA kernel itself wrote (so the packer's
                // addressing matches). The chosen tile(s) are packed into the low tiles of the
                // output CB; host compares the meaningful columns.
                //   Stage B (2): running row-max  -> tile @max_dst_tile_offset, host reads col 0.
                //   Stage C (3): exp(scores-max) of the LAST chunk (post-exp probabilities, before
                //                they are consumed by P@V) -> Sk_chunk_t mm1 tiles, host reads :128.
                //   Stage D (4): P@V partial accumulator over the core's chunks -> vDHt mm2 tiles.
                //   Stage E (5): running row-sum -> SAME tile as max (sum lives in col 1), host
                //                reads col 1.  *** prime suspect for the multichunk bug ***
                // Stage A (1): raw UNSCALED QK^T is overwritten in-place by exp within the same
                //   chunk, so it is NOT observable post-loop with the kernel's native addressing.
                //   We pack the post-exp mm1 here too; the host treats stage 1 like stage 3 (this
                //   slot is a placeholder — true raw-score capture needs in-chunk SDPA-addressed
                //   surgery and is intentionally left unimplemented to keep the kernel low-risk).
#if SDPA_STAGE_CUT_43563 == 2
                constexpr uint32_t cut_src_tile = max_dst_tile_offset;
                constexpr uint32_t cut_num_tiles = 1;
#elif (SDPA_STAGE_CUT_43563 == 1) || (SDPA_STAGE_CUT_43563 == 3)
                constexpr uint32_t cut_src_tile = mm1_dst_tile_offset;
                constexpr uint32_t cut_num_tiles = Sk_chunk_t;
#elif SDPA_STAGE_CUT_43563 == 4
                constexpr uint32_t cut_src_tile = mm2_dst_tile_offset;
                constexpr uint32_t cut_num_tiles = vDHt;
#elif SDPA_STAGE_CUT_43563 == 5
                constexpr uint32_t cut_src_tile = sum_dst_tile_offset;  // == max tile; sum in col 1
                constexpr uint32_t cut_num_tiles = 1;
#else
#error "SDPA_STAGE_CUT_43563: valid stages are 1..5"
#endif
                pack_block_contiguous(cut_src_tile, sdpa_output_cb, cut_num_tiles);
                // Pad the rest of the shard from mm2 leftover so the CB shard is complete.
                for (uint32_t i = cut_num_tiles; i < out_chunk_tiles; i++) {
                    pack_block_contiguous(mm2_dst_tile_offset + i, sdpa_output_cb, 1);
                }
                cb_push_back(sdpa_output_cb, out_chunk_tiles);
            } else {
                // Non-output cores: unchanged original handoff so downstream does not hang.
                if (!sdpa_output_is_final) {
                    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                    pack_block_contiguous(max_dst_tile_offset, sdpa_ms_cb, 1);
                    cb_push_back(sdpa_ms_cb, Sq_chunk_t);
                } else {
                    compute_sdpa_recip<out_chunk_tiles, exp_approx_mode, scale_bf16>(
                        cb_q_in, sum_dst_offset, corr_exp_dst_offset, mm2_dst_offset);
                }
                for (uint32_t i = 0; i < out_chunk_tiles; i += output_granularity) {
                    PACK(t6_semaphore_wait_on_zero<p_stall::STALL_PACK>(semaphore::FPU_SFPU));
                    pack_block_contiguous(mm2_dst_tile_offset + i, sdpa_output_cb, output_granularity);
                    PACK(t6_semaphore_get<p_stall::PACK>(semaphore::FPU_SFPU));
                }
                cb_push_back(sdpa_output_cb, out_chunk_tiles);
            }
#else
#if defined(SDPA_SUM_TAP_43563) && SDPA_SUM_TAP_43563
            // #43563 NON-INVASIVE SUM TAP. Pipeline control flow is UNCHANGED (we did NOT touch
            // sdpa_output_is_final, last_chunk, or do_tail_reduce). The per-core chunk loop above has
            // fully accumulated this output core's running max (col0) and running sum (col1) into the
            // max/sum DEST tile (max_dst_tile_offset). Here, BEFORE the cross-core tree reduction
            // consumes anything, we make ONE extra packer copy of that already-settled DEST tile into
            // the side CB so the host can read this core's pre-merge running sum. The extra pack reads
            // the same DEST tile the kernel itself packs to cb_interm_ms / recip-reads below, so it
            // does not change DEST lifetime, the mm2 packer handshake, or any semaphore.
            if (do_output) {
                // Drain SFPU so the last chunk's reduce_sum write to the sum tile is settled, then
                // gate the packer on SFPU (same handshake the production max-tile pack uses).
                MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                pack_block_contiguous_init(cb_iter1_dump);  // cb_iter1_dump == cb_sum_tap side CB
                cb_reserve_back(cb_iter1_dump, vDHt);
                // Pack the max/sum tile into low tile of the side shard; pad the rest with mm2
                // leftover so the side CB shard is fully populated (host only reads col1 of tile 0).
                pack_block_contiguous(max_dst_tile_offset, cb_iter1_dump, 1);
                for (uint32_t i = 1; i < vDHt; i++) {
                    pack_block_contiguous(mm2_dst_tile_offset + i, cb_iter1_dump, 1);
                }
                cb_push_back(cb_iter1_dump, vDHt);
                // Re-arm the production packer init for the real output pack that follows.
                pack_block_contiguous_init(sdpa_output_cb);
            }
#endif
#if defined(SDPA_LMS_TAP_43563) && SDPA_LMS_TAP_43563
            // #43563 L+MS TAP. Capture the per-core shipped L (mm2/PV output, vDHt tiles) and the full
            // MS tile (max@col0/sum@col1) into the side CB cb_iter1_dump BEFORE the cross-core tree
            // reduce. Reads the same already-settled DEST tiles the production packs below read, so the
            // pipeline (DEST lifetime / mm2 handshake / semaphores) is unchanged. Side CB layout:
            //   tiles 0..vDHt-1 = L (mm2),  tile vDHt = MS (max/sum). Host reads this back per core.
            // NOTE: in the FUSED decoder, do_output is hardcoded 0 and is_sender_after_reduce 1, so the
            // per-core SDPA output is shipped via cb_out_o/cb_out_ms on EVERY active SDPA core. We
            // therefore fire the tap UNCONDITIONALLY here (this block only runs on active SDPA cores),
            // capturing each core's pre-tree-reduce shipped L (mm2) and MS (max/sum).
            {
                MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                pack_block_contiguous_init(cb_iter1_dump);
                cb_reserve_back(cb_iter1_dump, vDHt + 1);
                for (uint32_t i = 0; i < vDHt; i++) {
                    pack_block_contiguous(mm2_dst_tile_offset + i, cb_iter1_dump, 1);  // L (mm2)
                }
                pack_block_contiguous(max_dst_tile_offset, cb_iter1_dump, 1);  // MS (max/sum)
                cb_push_back(cb_iter1_dump, vDHt + 1);
                // Re-arm the production packer init for the real output pack that follows.
                pack_block_contiguous_init(sdpa_output_cb);
            }
#endif
            if (!sdpa_output_is_final) {
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                pack_block_contiguous(max_dst_tile_offset, sdpa_ms_cb, 1);
                cb_push_back(sdpa_ms_cb, Sq_chunk_t);
            } else {
                compute_sdpa_recip<out_chunk_tiles, exp_approx_mode, scale_bf16>(
                    cb_q_in, sum_dst_offset, corr_exp_dst_offset, mm2_dst_offset);
            }
            for (uint32_t i = 0; i < out_chunk_tiles; i += output_granularity) {
                PACK(t6_semaphore_wait_on_zero<p_stall::STALL_PACK>(semaphore::FPU_SFPU));
                pack_block_contiguous(mm2_dst_tile_offset + i, sdpa_output_cb, output_granularity);
                PACK(t6_semaphore_get<p_stall::PACK>(semaphore::FPU_SFPU));
            }
            cb_push_back(sdpa_output_cb, out_chunk_tiles);
#endif
            tile_regs_commit();
            tile_regs_wait();
            tile_regs_release();
#if defined(BANK_RESET_AT_TAIL_43563) && (BANK_RESET_AT_TAIL_43563 != 0)
            // #43563 LOCALIZE: at the SDPA-compute -> tail-reduce boundary (after the compute
            // tile_regs_release, before do_tail_reduce's acquire), DEST carries no live state.
            // Pin the DEST bank to 0 every iteration (Austin's iter-top pin form) so the tail
            // reduce and everything after run on a consistent bank, while the per-core SDPA
            // compute above stays on the alternating bank.
            MATH((llk_math_pack_sync_init<false>()));
            PACK((llk_pack_dest_init<false, false>(0)));
#endif
            sdpa_custom_mm_block_uninit();
            MATH(t6_semaphore_wait_on_max<p_stall::STALL_SFPU>(semaphore::FPU_SFPU));

            static_assert(vDHt % dst_size == 0, "vDHt must be divisible by dst_size");
            constexpr uint32_t num_blocks = vDHt / dst_size;
            constexpr uint32_t block_size = vDHt / num_blocks;

            bool do_tail_reduce = do_reduce && num_cores_to_wait > 0;
#if defined(SDPA_STAGE_CUT_43563) && (SDPA_STAGE_CUT_43563 != 0)
            // #43563 STAGE CUT: on output cores we already packed the per-core intermediate into
            // cb_out_final and bypassed the cross-core merge, so skip the TRISC tail reduction
            // (it would re-pack cb_out_final and consume sender data we deliberately ignore).
            if (do_output) {
                do_tail_reduce = false;
            }
#endif
            if (do_tail_reduce) {
                reconfig_data_format_srca<false, true>(cb_ms_in);
                exp_tile_init<exp_approx_mode, scale_fp32>();
                for (uint32_t i = 0; i < num_cores_to_wait - 1; i++) {
                    sdpa_tail<exp_approx_mode, false, block_size, num_blocks, scale_fp32, VectorMode::C>(
                        cb_ms_in, cb_interm_ms, cb_interm_ms, cb_out_in, cb_interm_out, cb_interm_out);
                }
                if (is_sender_after_reduce) {
                    sdpa_tail<exp_approx_mode, false, block_size, num_blocks, scale_fp32, VectorMode::C>(
                        cb_ms_in, cb_interm_ms, cb_out_ms, cb_out_in, cb_interm_out, cb_out_o);
                } else {
                    sdpa_tail<exp_approx_mode, true, block_size, num_blocks, scale_fp32, VectorMode::C>(
                        cb_ms_in, cb_interm_ms, cb_out_ms, cb_out_in, cb_interm_out, cb_out_final);
                }
            }

            cb_pop_front(cb_q_in, q_chunk_tiles);
            if (mask_last_chunk) {
                cb_pop_front(cb_mask, 1);
            }
#if defined(FIX_FULL_VIA_MASK_43563) && FIX_FULL_VIA_MASK_43563
            // #43562/3 FIX: release the all-zero mask tile (paired with the wait_front above).
            if (use_zero_mask) {
                cb_pop_front(cb_mask, 1);
            }
#endif
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
