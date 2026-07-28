// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NOT COMPILED. Copy-paste source for graduating the measured Perf-2 phase-4 follow-ups
// (sub-lever a: bf16 stat CBs; sub-lever b: packed rsqrt) into rms_norm.
// Measured by tests/.../rms_norm/perf_experiments/phase4_packed_rsqrt_and_bf16_stats,
// focus geometry (1,1,8192,1024) BLOCK_SHARDED grid(8,8), HT_BLOCK=8:
//
//   option              ns    ns/tile  vs baseline   col-0 PCC     precision
//   baseline           8679     271.2       1.000x    0.9999898    fp32 stat CBs (today)
//   baseline_bf16      8461     264.4       1.026x    0.9999898    BIT-EXACT vs baseline
//   pack_here_c        5350     167.2       1.622x    0.9999797    same contract
//   pack_here_c_bf16   5240     163.8       1.656x    0.9999797    same contract
//   pack_here_cskip    4610     144.1       1.883x    0.9999797    same contract, ht<=8
//   pack_given_c       4497     140.5       1.930x    0.9999862    same contract (needs sibling)
//   pack_given_c_bf16  4305     134.5       2.016x    0.9999862    same contract (needs sibling)
//   pack_given_cskip   3692     115.4       2.351x    0.9999862    same contract, ht<=8, needs sibling+co-design
//
// All eight meet the precision contract identically (same rsqrt body, same
// fp32_dest_acc_en/math_fidelity/math_approx_mode as every other option); none trade
// precision for speed. bf16 stat CBs are measured BIT-EXACT vs fp32 stat CBs at
// fp32_dest_acc_en=False (max|diff|=0.0, test_bf16_stat_bit_exactness_and_guard) and
// measurably DIFFERENT at fp32_dest_acc_en=True (max|diff|=5.1e-2) -- confirming sub-lever
// (a)'s own predicate: guard it on `!fp32_dest_acc_en`.
//
// PREDICATE (ht_block sweep, focus grid, contiguous "_c" scope; cskip tracks the same
// shape but only exists for ht_block<=8):
//   ht_block   1        2        4        8        16
//   pack_here  0.422x   0.682x   1.090x   1.622x   2.150x
//   pack_given 0.443x   0.734x   1.238x   1.930x   2.754x
// Monotone increasing in ht_block. REGRESSES below ht_block~=3-4 (severely at ht_block=1,
// where there is exactly one tile per row-block and nothing to pack -- pure reduce_init /
// reduce_uninit / scaler-wait overhead with zero saving). Graduation guard:
// `HT_BLOCK >= 4` (a comfortable margin inside the measured >=1.09x crossover at
// HT_BLOCK==4; HT_BLOCK in {1,2} stay on the byte-identical fallback).
//
// ---------------------------------------------------------------------------------------
// PART 0 -- WHICH OPTION TO GRADUATE. Two independent choices, NOT mutually exclusive:
//
//   (A) STANDALONE, no dependency on any sibling: `pack_here_c` (1.622x) or, if the even-
//       column packing is judged worth its extra host-side complexity, `pack_here_cskip`
//       (1.883x). Either works TODAY regardless of what the gather_payload_shrink sibling
//       does.
//   (B) COMPOSED with gather_payload_shrink's column-packed cross-core gather: IF that
//       sibling graduates a variant that multicasts the packed statistic tile directly
//       (rather than colsel-extracting on the root before the broadcast, which is what its
//       CURRENT `colpack` implementation does), then `pack_given_c` (1.930x, works with the
//       sibling's ACTUAL contiguous-column packing, unchanged) or `pack_given_cskip`
//       (2.351x, the best number measured here, but requires the sibling to ALSO switch its
//       own packsel scaler to place values at EVEN columns -- a joint refinement, not a
//       drop-in). `pack_given` is strictly cheaper than `pack_here` at every HT_BLOCK
//       because it skips the pack step entirely; it graduates only as a REPLACEMENT for
//       the sibling's current post-fold colsel-extract-then-broadcast-ht-tiles step, not
//       as an independent change.
//
// The safe, no-coordination recommendation is (A) `pack_here_c`. (B) is a strictly better
// number IF AND ONLY IF gather_payload_shrink also graduates and is willing to change its
// broadcast payload shape (1 tile instead of `ht`) and drop its own colsel step (which this
// change absorbs).
// ---------------------------------------------------------------------------------------
//
// PART 1 -- the PACK / EXTRACT mechanism, raw compute API (adapted from the
// gather_payload_shrink sibling's device-verified column-pack; ITS probe_mechanism.py
// established the (scaler position) -> (dest position) map on real silicon, reused here by
// derivation, not re-probed). Natural home: a new pair of helpers next to
// ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp (e.g. `reduce_helpers_compute::
// column_pack` / `column_select`), since no existing reduce helper takes a bank of
// non-canonical per-output-column scalers -- ckl::reduce / reduce_mean hard-code exactly
// ONE canonical "row 0 of every face" scaler tile.
//
// MECHANISM. BH's REDUCE_ROW SUM is an MVMUL with the scaler tile as the WEIGHT matrix
// (SrcA) and the data as the moving operand (SrcB): dest[i, j] = sum_k W[j, k] * data[i, k],
// where the scaler's own tile-relative (face, row-in-face, col-in-face) address IS (j, k)
// via the standard 4-face decomposition (row-in-face + 16*bottom_face = j, col-in-face +
// 16*right_face = k) -- W is addressed as an ordinary 32x32 tile, built on the HOST as a
// plain (row, col) torch tensor and handed to `ttnn.from_torch(..., layout=TILE_LAYOUT)`,
// which tilizes it into the same physical layout the hardware expects (no raw L1 fill
// needed, unlike the sibling's on-device writer-side construction -- this bench builds the
// banks once, host-side, since they are true kernel-invocation-wide constants here).
//
//   PICK0[h]:  W[h,0] = W[h+16,0] = 1, else 0
//              -> dest[i,h] = data[i,0]: pack step -- "take column 0 of source tile h,
//                 place it at destination column h (2h under the cskip/even-column form)".
//                 `ht` reduce_tile calls (one source tile per h, all writing idst=0)
//                 accumulate into ONE dest tile because each call's nonzero output column
//                 is disjoint from every other call's.
//   COLSEL[h]: W[0,h] = W[16,h] = 1, else 0
//              -> dest[i,0] = data[i,h]: extract step -- the inverse selector, identical in
//                 spirit to gather_payload_shrink's own `cb_colsel` (that one carries 1/N
//                 for its finalize; this one carries plain 1.0 since the packed tile has
//                 already been fully rsqrt'd).
//
// PACK needs `reduce_uninit()` between `tile_regs_commit` and its pack -- reduce_init's
// default packer edge mask force-zeros every output column but 0, which would erase PICK0's
// whole point (a non-zero column h). EXTRACT wants column-0-only output, which the mask
// already gives for free, so it is left ON through the whole per-h loop and cleared once
// afterward (mirrors the sibling's own step-2 exactly).
//
// THE FUSION (pack_here only): PICK0's `ht` reduce_tile calls and the rsqrt SFPU pass run
// in the SAME tile_regs_acquire/commit window -- the fused-rsqrt body is called directly on
// DEST slot 0 right after the last reduce_tile, before tile_regs_commit. This is legal
// because DEST-sync windows may freely mix an FPU op (reduce_tile) and an SFPU op (rsqrt)
// in program order; it saves an entire extra pack-out/copy-in round trip for the raw packed
// sum, at the cost of re-running `reduce_init` immediately before (SFPU work needs no
// re-init of its own beyond `rsqrt_tile_init()`, called fresh every row-block group exactly
// as the shipped `RsqrtAddUnaryColZero::init()` already is via `eltwise_chain`'s CRTP hook).
//
// SFPU SCOPE for the packed tile: `_c` uses `VectorMode::C` (16 vectors -- covers tile
// columns 0..15, i.e. any HT_BLOCK<=16 placed contiguously); `_cskip` places each h's value
// at column 2h (host-side scaler choice only -- the kernel loop is IDENTICAL, only the
// scaler CONTENT and the SFPU stride differ) so the SAME even-parity-stride body the
// shipped `RsqrtAddUnaryColZero` already uses (8 vectors) covers the whole packed tile in
// one pass, at the cost of `HT_BLOCK<=8` (a full 32-column tile only has 16 even slots split
// across both row-halves, i.e. 8 usable positions <=15 without also touching faces 1/3).
//
// RAW-LLK JUSTIFICATION (helpers bypassed and why -- required so a later helper-usage pass
// does not "fix" this back and undo the win):
//   ckl::reduce / ckl::reduce_mean -- their scaler is ONE canonical tile (row 0 of every
//     face); neither can express a per-output-column, non-canonical scaler bank at all.
//   reduce_init's packer edge mask -- invisible at the helper level; PICK0 needs it
//     defeated (`reduce_uninit()` before the pack), COLSEL needs it left alone.
//   `copy_tile_to_dst_init_short` -- documented as NOT reconfiguring the unpacker data
//     type; a bf16 `cb_packed_in` read back through a stale fp32-configured unpacker
//     silently misinterprets bytes (measured: PCC ~0.59, no crash) unless
//     `reconfig_data_format_srca(cb_packed_in)` precedes it explicitly.
//   `compute_kernel_hw_startup`'s three operands must all be CBs the PROGRAM actually
//     declares -- referencing an undeclared CB (measured: `cb_in` when only
//     `cb_packed_in` is attached, i.e. the `pack_given` mode) silently corrupts operand
//     format tracking for every OTHER CB touched later in the kernel (measured: PCC 0.889,
//     not a crash or an assert -- a believable-looking but wrong number).
//
// ---------------------------------------------------------------------------------------
// PART 2 -- sub-lever (a), narrowing the two intermediate CBs (independent of part 1,
// composes with either pack mode). rms_norm_program_descriptor.py `cb_plan()`:
//     ("cb_rms_sum",   CB_RMS_SUM,   self.fp32_tile_bytes, H)   ->  self.tile_bytes
//     ("cb_rms_recip", CB_RMS_RECIP, self.fp32_tile_bytes, H)   ->  self.tile_bytes
// GUARD: only when `!fp32_dest_acc_en` (measured bit-exact there; measurably different,
// hence load-bearing, at `fp32_dest_acc_en=True` -- test_bf16_stat_bit_exactness_and_guard,
// max|diff| 0.0 vs 5.1e-2). The freed L1 (2 * H * (4-2) KB per core) is real headroom for
// the L1-bound prefill geometries; round 1 measured spending it on a bigger block worth a
// further ~1.05x there.
