# Rank-stamp extension for classic stable topk — running log

Branch: nkapre/rankstamp-53557 (from nkapre/repair-53557-32bit tip 6d49857e9e2).
Goal: stable topk at u32-index widths (W>65535 or explicit u32/i32 indices) runs the FAST
unstable network with sign-conditioned LOCAL RANK tags in the value words' lo16, true u32
index riding via index-tracking mode. FLOAT32 keeps comparator-stable.

## Derived design decisions (session log, newest last)

### D1. Classic Dst layout (from LLK addressing + XL lane formula)
- 32-bit Dst: 1 tile = 64 SFPLOAD address units (rows of 16 words). SFPLOAD INT32 at Addr
  covers 4 consecutive Dst rows x 8 columns (even/odd parity = Addr bit 1); lane j ->
  word ((Addr&~3)+j/8)*16 + (j&7)*2 + (Addr&2?1:0).
- After transpose_tile, width index w runs along Dst rows; logical seq offset o (0..63,
  across 2 tiles) maps to address a = (o & 0xF) + (o >> 4)*32 (faces interleave), i.e.
  rows 0-15,32-47 (tile0) then 64-79,96-111 (tile1).
- Lane j at address a holds seq position o(a) + (j>>3); (j&7)*2+parity = batch column b
  (independent sort problem per b).

### D2. Direction bookkeeping at classic merge time — STATIC
Traced kernel alternation (process_and_sort_tiles / process_iteration / rebuild alternation):
at EVERY topk_merge call in the classic tree, the LEFT partner run (lower Dst offset AND
lower global width range) is sorted in the GLOBAL direction (ascending=!largest), the RIGHT
partner in the mirror direction. Same static pattern the merge's compile-time top_min already
relies on. So the dual stamp needs NO direction argument for topk (XL's fixed
slot0-lower/slot1-flipped applies verbatim). ttnn.sort would need the 4-combo generalization
(deferred, per design).

### D3. Stamp values (run storage position p in [0,k), k = topk K)
- left run: rank r = p, range [0,k)
- right run: rank r = (2k-1) - p = p XOR (2k-1), range [k,2k)
- stored tag = r XOR (0xFFFF iff value_positive XNOR largest)  (same conditioning as fuse)
- Verified strict monotonicity per run and tie->ascending-global-index in both directions,
  both largest polarities (see derivation in session).
- Local-position stamp before local sort: tag = seq position o in [0,64), same conditioning
  (this IS the fuse with o instead of the true index).

### D4. Mode: RANK_STAMPED template bool, unfused skeleton
- index tracking ON, index tiles at Dst+128, instr_mod_index = INT32 (fp32_dest_acc_en forced)
- instr_mod_value = INT32 (tags are denormal-fragile like fused keys)
- ldst_count stays 8; network bodies = plain UNSTABLE replay path (STABLE_SORT=false-like)
- static_asserts: RANK_STAMPED requires is_fp32_dest_acc_en, !FUSED, !STABLE_SORT,
  !TOPK_UINT16_IN_FP32_DEST
- TEN-2932: stamp keeps state in LREG0..3 + programmable const LREGs only; loads' captures
  into LREG4..7 are dead (merge reloads indices after stamping).

## Open questions
- OQ1: single-core insertion path (topk.cpp) structure — read next.
- OQ2: where the stamp sweeps run (standalone LLK entries called from kernel vs folded into
  merge). Leaning: standalone sweeps (option A) first for correctness, option B fold later
  only if perf demands — A is simpler to validate on silicon.
- OQ3: LLK test harness shape for a rank_stamped mode.

### D5. Single-core insertion path uses ONLY topk_local_sort (no merge/rebuild!)
topk.cpp PHASE 2: per insertion, Dst0=accumulator (32 sorted, global dir, stable), Dst1=incoming
fresh chunk; full 64-sort (end_phase=5, idir=!largest); keep Dst0. So single-core rank-stamped
needs ONE primitive: stamp local positions o in [0,64) before EVERY topk_local_sort. Accumulator
tie order along storage = ascending index (invariant) -> rank=o correct; incoming chunk position
= index order; ranges [0,32) < [32,64) = index ranges ✓. Tags NEVER need to survive CB transport
(re-stamped fresh each call) but values must move raw -> Float32 value CBs (as fused does;
UInt32-format CBs corrupt under pack_tile<true>, #53466).

### D6. Stamp/strip/canonicalize plan
- `_topk_stamp_local_positions_<largest>(rank_base?)`: inline sweep, 64 vectors x ~9 instrs
  over Dst tiles 0,1. Per vector at addr a: rank = o_base(g) + (j>>3), o_base(g)=(g&3)*4+(g>>3)*16,
  g = a>>2, j>>3 = LTILEID>>4. Clears lo16 (SFPLOADI LOWER 0), OR rank, sign-conditioned XOR 0xFFFF
  (LREG12), all-lanes bracket. FOLD -0 canonicalization in: after lo16 clear, w<<1==0 lanes get
  SFPLOADI UPPER 0 (+5 ops). TEN-2932 ok: writes only LREG0..3; load captures to LREG4.. dead.
- Merge dual stamp: OPTION B (inline in _bitonic_topk_merge<RANK_STAMPED>): LREG2=rank iota
  (LTILEID>>4 + rank_base at pair start, +4/iter), LREG3 = LREG2 XOR LREG13(=2k-1 runtime const
  programmed at merge entry); stamp LREG0 with LREG2, LREG1 with LREG3, ~13 instrs/iter between
  load8 and SFPSWAP. rank_base runtime arg (t*32) for K=64 split calls. Left run always global
  dir/lower range (D2) -> fixed assignment.
- Rebuild: NO stamp needed (inherits distinct tags from merge; multicore rebuild pairs two
  merge-winner runs, each internally distinct, rebuild sorts runs independently).
- `_topk_strip_rank_tags_`: lo16 clear sweep (load INT32, SFPLOADI LOWER 0, store INT32) on the
  final value tiles after last transpose, before bf16 pack (RNE would round junk up).
- 32-bit transpose of [bf16|tag] words: rely on same machinery as u32 index-tile transpose
  (proven by sort/fp32 paths); strip AFTER final transpose, in Dst, before pack.

### D7. Transport formats (C3): value transposed/intermed/result_prep CBs -> Float32 raw
(like fused packed CBs); index CBs UInt32 as today. fp32_dest_acc_en forced true.

### D8. TEN-2932 audit of stamp instruction selection (ttsim 5734-5743 + XL 597-601)
- ttsim models: ALU writes (opcodes 0x73..0x99 except SFPLUT-family exclusions 0x91/92/93) to
  LREG4..7 corrupted under ENABLE_DEST_INDEX; loads/stores/SFPCONFIG/SFPSWAP/SFPTRANSP exempt.
  Capture side (SFPLOAD to L0..3 -> junk into L4+n) NOT modeled in ttsim; XL comment is the
  silicon authority; SFPLOADI capture status unspecified -> treat as capturing.
- MERGE inline stamp runs AFTER true indices are in LREG4/5 => NO SFPLOAD/SFPLOADI to L0..3
  allowed there. lo16 clear via SFPAND with LREG11 = 0xFFFF0000 (new programmable const).
  All merge-stamp ops are ALU writes to LREG0..3 only: MOV/XOR/AND/OR/SETCC/ENCC/IADD/SHFT ✓.
- Standalone sweeps (local-position stamp, strip) run while LREG4..7 dead => SFPLOAD/SFPLOADI OK.
- Constants: LREG12=0x0000FFFF (complement), LREG11=0xFFFF0000 (clear mask), LREG13=2k-1
  (runtime, merge entry), LREG14=rank_base (runtime, merge entry). Config writes under
  all-lanes-on, before loads (SFPCONFIG lane-predicated + transient LREG0 clobber).
- vConstIntPrgm0 = LREG12 (classic precedent); use _sfpu_load_config32_ for 11/13/14.

### D9. API shape decisions
- Standalone `_topk_stamp_local_positions_<largest>()` (fuse-like, kernel calls before every
  topk_local_sort in rank-stamped mode). +-0 canonicalization folded into this sweep
  (w<<1==0 -> predicated SFPLOADI UPPER 0), since every datum passes local sort exactly once.
- `_bitonic_topk_merge<..., RANK_STAMPED>` stamps internally (option B), new runtime arg
  rank_base (t*32 for K=64 split calls), largest = !top_min (compile-time).
- Rebuild: only RANK_STAMPED threading for INT32 value ld/st; no stamping (inherits distinct
  tags from merge via raw Float32 CB transport, proven by fused engine).
- `_topk_strip_rank_tags_(num_tiles)` for final extraction after transpose, before bf16 pack.
- LLK test (C1 validation): mechanism test with u16-index harness at W=128, 1 iteration
  (same restriction as fused: bf16 L1 round-trip loses tags; harness can't do u32 index tiles
  or W>=256 anyway, tt-llk#1344). True-u32 + wide widths validated at ttnn level (C3/C4).

## C1 run log (BH silicon p150a)
- First run: 18 failed at JIT — "too few arguments": SFPU wrapper forwards through a callable,
  DEFAULT ARGUMENTS DO NOT APPLY. rank_base must be passed explicitly at every SFPU_UNARY_CALL
  site (test kernel + compute_kernel_api.h topk_merge patched with explicit 0u).
- After fix: LLK test_topk.py -k rank_stamped: 18 passed, 2 skipped (all adversarial tie classes
  incl. signed_zero/nan_payloads/mixed_sign_ties, both directions, STRICT canon golden).
- Full LLK test_topk.py: 60 passed, 8 skipped (baseline was 42p/6s; +18 new, no regressions).
- ttnn blast-radius smoke (fresh JIT after cache clear): test_topk.py -k stable_index_parity
  24 passed (comparator single/multi, fused, fp32, wide u32, prealloc all still green with the
  reshaped merge signature).
- WH: mirror applied; knob-normalized diff of all new sections = IDENTICAL except intended
  WH-only SETC16/STALLWAIT(STALL_CFG) brackets in the strip (mirrors move_dest_tile helper).
  ttsim WH build kicked off for functional sim check.

## C1 WH verification (final)
- ttsim WH (libttsim_wh built from ttsim-private): rank_stamped cells 18p/2s; full suite minus
  comparator-stable 42p/6s. PRE-EXISTING sim gap: comparator-stable trips ttsim
  "UnsupportedFunctionality: tensix_sfpswap: ENABLE_DEST_INDEX: lreg_c=5" (explicit SFPSWAP on
  index regs L4-7 unmodeled; silicon supports it). ttsim also enforces the TEN-2932 ALU-write
  verify, so the stamp's register discipline is machine-checked on WH.

## C2
- compute_kernel_api.h: rank_stamped template on topk_local_sort/topk_merge/topk_rebuild,
  topk_tile_init<fused, rank_stamped>, new topk_stamp_local_positions<largest>(idst),
  topk_strip_rank_tags(idst). topk_merge gains runtime rank_base (explicitly forwarded).
- quasar: _topk_strip_rank_tags_ no-op stub (unreachable; mode static_asserted off).
- Fresh-JIT ttnn smoke post-change: 24 passed.
