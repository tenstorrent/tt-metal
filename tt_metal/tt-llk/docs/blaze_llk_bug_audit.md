# Experimental LLK bug audit — 15 blaze-promotion PRs

What this is: a scan of 15 PRs against the bug patterns in the colleague's
`REMAINING_WORK.md` (items C1–C6, A5, F). Same four buckets the ask named:
redundant open PRs, functional bugs, missing tests, infra problems.

- **MEASURED** = proven by a report result or a test that fails; **INFERRED** = read
  from the merged source, not re-run on silicon.
- File:line point at `origin/main` (the merged experimental headers), unless a PR is named.
- These are experimental LLKs — they break the normal programming model on purpose, so
  not every deviation is a bug. The ones below are ordinary programming errors.

PRs by state: **open** — 53130, 53361, 52720, 52646 (draft). **merged** — everything else.

---

## 1. Redundant open PRs

- **53130 and 53361 duplicate the sampling test.** Both add/edit `test_sfpu_sampling.py`
  **and** `sfpu_sampling_test.cpp`, and both add a byte-identical `SAMPLING_PRGM0_HAZARD`
  class to `test_variant_parameters.py`. 52163 (merged) already owns the base sampling test.
  → one open PR should own the Prgm0-hazard case; drop it from the other.
- **52720 and 53130 both drive `compressed_custom_mm`.** 52720 adds a fresh
  `compressed_custom_mm_test.cpp`; 53130 adds `test_matmul_custom_compressed.py` over the
  already-merged `matmul_custom_compressed_test.cpp`. Both bf16-dest only; only 53130 hits the
  `kt*ct % 10 == 0` metadata boundary. → keep 53130's; fold or drop 52720's compressed cpp.
  52720's `sdpa_bcast` scaffolds are **not** redundant — keep them.
- **52646 (draft) contradicts merged 52699.** 52699 keeps `generalized_moe_gate` GATE in the
  bit-exact suite by scrubbing scratch (live on main); 52646 opts GATE out of it. Two
  mutually-exclusive fixes for one problem. Since 52699 merged, **52646 is droppable** (or
  revert 52699 — not both). Neither hides a wrong answer; GATE's non-answer lanes are
  contractual padding.
- **51777 vs 53663 (topk_xl) — NOT redundant.** Share 6 files, but 53663 is purely additive
  (new `early_exit_K64`, row-major splits). No action.
- **53361's `Top32RmGolden` is an orphan.** Added to `golden_generators.py` but 53361 has no
  top32_rm test; the owner (53130) computes that golden inline. → dead code or a bad split.

---

## 2. Functional bugs in the LLKs under test

### Confirmed analogues of the report's defects

- **C1 — packer W-stride hardcodes `* 2`, not format-aware, in FIVE headers.**
  `cpack_common.h` derives the stride from `datum_size_in_bytes(pack_src_format)`; these
  hardcode bf16's 2 bytes, so a **Float32 pack source is 2× off**:
  - `custom_mm.h:71,115,261` — MEASURED 0.25 match (report C1)
  - `compressed_custom_mm.h:71,113,262` — latent (bf16-only tests)
  - `sdpa_custom_mm.h:19-20,74-75` — INFERRED, latent
  - `custom_mm_reuse_dest_srcb.h:43-44,58-59` — INFERRED, latent
  `*_block_uninit` takes no `out_cb_id`, so it cannot restore the stride format-aware either.
- **C5 — out-of-bounds metadata read, live on main.** `llk_unpack_AB_compressed_custom_mm.h:243`
  reads `meta_ptr[full_iters]` unconditionally; when `rem_iters == 0` (smallest case
  `kt=10,ct=1`) the word is past the buffer. Math side same at
  `llk_math_compressed_custom_mm.h:192,215`. MEASURED (report C5). The guard sits on a separate
  unmerged branch; 53130 adds the boundary test but not the fix.
- **C6 — `top32_rm` 32-bit unpack branch sorts against stale Dest.** The 16-bit branch clears
  SrcA to −inf (`CLR_SRC_NEGINF`); the 32-bit branch
  (`llk_unpack_A_top32_rm.h:73-93`) clears **nothing**, and the ZEROACC loop
  (`llk_math_top32_rm.h:99-102`) is bounded by `num_faces`, so a partial chunk keeps whatever
  Dest held. MEASURED (report: a 160-element row returned 11026/10041/9058, not in the input).
  Pinned by non-strict xfail in 53130. The single load-bearing difference is the missing
  `CLR_SRC_NEGINF`.
- **C4 — `mul_reduce_scalar` re-entry inside one DEST section gives wrong answers.**
  `_llk_math_mul_reduce_scalar_init_` (`llk_math_mul_reduce_scalar.h:178-189`) re-arms less than
  the `dest_section_done`/`wait_for_dest_available` handshake does. The chunked op
  `mul_reduce_scalar_chunked_tile` (`rmsnorm.h:120-166`) re-enters every batch inside one
  acquired section — exactly the broken pattern. MEASURED 9.3–9.9× golden (report C4).
- **C2 — `eltwise_mul_scalar` HiFi init hardcodes the tile shape.** The HiFi branch of the init
  passes `DEFAULT_TENSOR_SHAPE` (`eltwise_mul_scalar.h:74-88`) while the paired execute derives
  the shape from the CB. Inert on a 4-face CB (identical to the LoFi shorthand), **deadlocks the
  MATH_PACK handshake on a 2-face tile**. MEASURED (report C2). `eltwise_add_scalar` and
  `hadamard` were checked and are clean (no HiFi fork / fixed 1-face).

### New bugs this scan turned up

- **`sum_reduce_scalar`: a 1-face (16×16) multi-tile reduce only sums the first tile.** The
  face-fold in `_llk_math_mul_reduce_column_` (`llk_math_mul_reduce_scalar.h:207-235`) is guarded
  on `!is_narrow_tile && total_num_faces() > 1`, so 1-face tiles skip tiles 1..N-1. Currently
  hidden behind `pytest.skip` in 53361, not xfail. INFERRED (root cause) / MEASURED (skip note).
- **`sdpa_reduce_row`: the SUM pool returns a wrong answer** (Max passes). Hidden behind
  `pytest.skip` in 53361 after #53295. MEASURED (skip note).
- **`softmax_k`: k<16 on a 32-bit DEST softmaxes all 16 lanes, not k.** The odd/even column mask
  and `_zero_paired_odd_tail_lane_` (`ckernel_sfpu_softmax_k.h:40-67,145-150`) only work while two
  bf16 datums share one 32-bit word; on fp32 DEST the padding lanes are never zeroed and enter
  the row sum. Same shape as C2 — the kernel compiles for 32-bit but its masking assumes bf16.
  MEASURED (test docstring). 52163.
- **`recip_init` sets `vConstFloatPrgm0` for a consumer it never calls**
  (`ckernel_sfpu_recip.h:398`, 52710). Dead init — the fast paths don't read it — and it makes
  the sampling comment stale. Minor.

### Looked, not a bug

- moe_gate stale-scratch (52699/52949): the residue lanes are declared padding by the op
  contract, so it's a determinism/bit-exact-rerun symptom, not corruption. But 52949's own
  comment records that zeroing LREG0-7 changed nothing — so the **mechanism is still unproven**.
- `sdpa_exp_unclamped` overflow for val>0 is a documented by-design gap (SDPA contract is val≤0).
- 53658 (retire INT32_MIN reduce xfail) is backed by a real fix in `ckernel_sfpu_reduce.h`;
  only the comment's cited symbol name has drifted.

---

## 3. Missing tests

- **A5 — the HiFi-init test never runs.** `test_eltwise_mul_scalar_hifi.py` (53361) is
  `@pytest.mark.skip`, not xfail, so the C2 sequence above is compile-only. The one thing A5
  wanted proven is still unproven at runtime.
- **C3 — no topk_xl → foreign-op transition test.** `test_topk_xl_reinit_after_copy` (53663)
  restores topk_xl's *own* state only; nothing runs topk_xl then `eltwise_binary`, which is the
  reconfig escape the report flags. topk_xl also ships no uninit (`topk_xl.h`).
- **`top32_rm` 32-element tail (row=3232) is excluded** — blocked by C6; becomes one line once
  the 32-bit branch clears its tile.
- **`sdpa_weighted_reduce` test doesn't run the shipped op** — it substitutes a standard matmul
  unpack + full-tile pack; the raw UNPACR/PACR path (`sdpa_weighted_reduce.h:109-207`) has zero
  callers and is unexercised. 53361.
- **No re-entry / passes=2 test for `sum_reduce_scalar`** — it shares the C4-weak reduce tail.
- **Orphan LLKs with real ttnn callers but no LLK test:** `indexer_mul_custom.h`
  (ttnn `compute_indexer_score.cpp`), `sdpa_sub_custom.h` (ttnn `compute_streaming.hpp`),
  `deepseek_compute_kernel_hw_startup.h`, and `compute_kernel_hw_cleanup.h` (only a fully-skipped
  test).
- Smaller: `softmax_k` fp32/k<16 broken cell is excluded from the sweep instead of xfail-pinned;
  `moe_gate_topk` has no tie/duplicate-key case; `set_dst_write_addr_offset` never hits its
  `LLK_ASSERT(addr < DEST_REGISTER_HALF_SIZE)` boundary.

---

## 4. Infra problems and inconsistencies

- **`test_variant_parameters.py` will collide on merge.** 53130 and 53361 both define
  `SFPU_FAST_APPROX`, `CUSTOM_MM_UNINIT`, `SAMPLING_PRGM0_HAZARD`, `PACK_NUM_TILES`. Three are
  identical; **`CUSTOM_MM_UNINIT` conflicts** — 53361 adds a `restore_mop` field + an
  `UNINIT_RESTORE_MOP` define 53130 lacks. De-dup before either merges. (52720 and 52646 both
  edit `test_config.py` in non-overlapping regions — rebase, no conflict.)
- **`pytest.skip` used where a non-strict xfail belongs — it hides live wrong answers forever.**
  `sum_reduce_scalar` (1-face) and `sdpa_reduce_row` (Sum pool) are both wrong-answer skips in
  53361; they should be `xfail(strict=False)` so they run and flip to XPASS on fix. The *other*
  53361 skips (eltwise_mul_scalar_hifi, hw_cleanup) are legitimate — a device wedge can't be
  xfailed.
- **Tests that replicate the compute-API body instead of calling it (report B1).** custom_mm and
  eltwise_mul_scalar_hifi tests expand the LLK sequence rather than calling
  `ckernel::custom_mm_block*` / the HiFi init, so a header fix isn't caught. Worse:
  `custom_mm_test.cpp` (53361) writes a **format-aware** stride the header doesn't
  (`* (is_fp32_dest_acc_en ? 4 : 2)`), so its fp32 case passes while `custom_mm.h:71` stays
  broken — it actively masks C1.
- **Compile-only-on-HW tests:** `test_eltwise_mul_scalar_hifi.py` and `test_hw_cleanup.py` are
  fully skipped; `test_custom_mm_uninit_parity.py` is a device-free text match by design (B1
  guard — it can only say the two uninit bodies still *read* the same, not that they *work*).
- **Missed reuse:** 53361 builds custom_mm grids/strides inline instead of using the
  `custom_mm_utils.py` that 52720 adds.
- **Vacuous tests** (constant-fill or single-lane, would pass even if the kernel dropped
  lanes/faces/order): `sdpa_reduce_row`, `sdpa_weighted_reduce` (checks 16 of 32 lanes),
  `zero_pad` / `sparse_k_filter` (per-slot-constant). Per the report, budget a mutation before
  trusting a first-try pass.

---

## Shortlist

1. Land the C5 guard on main (it's a live memory-safety defect, fix already written).
2. Route C1/C4/C6/C2 to header owners — all four have a reproducer or an xfail attached.
3. Fix the two `skip`→`xfail` reduce tests and the `test_variant_parameters.py` collision before
   53130/53361 merge.
4. Drop 52646; de-dup the 53130/53361 sampling test; fold 52720's compressed cpp into 53130.
5. Un-skip A5's HiFi test (behind C2) and add the topk_xl→eltwise_binary transition test (C3).
