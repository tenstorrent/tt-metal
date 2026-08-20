# tt-llk blaze promotions — everything still to do

Single actionable index of what is left, with a plan per item. **Updated 2026-08-20** after
#52713 merged and A2 landed — see [§ Closed on 2026-08-20](#closed-on-2026-08-20).
Previously updated **2026-08-19** after #52727 merged and the branch was rebased onto it — see
[§ Closed on 2026-08-19](#closed-on-2026-08-19). Previously updated **2026-08-18** after a
working session that closed A3, A6 and D2, added coverage for a boundary nothing reached, and
**located the `mul_reduce_scalar_chunked_tile` defect** (A4). See
[§ Closed on 2026-08-18](#closed-on-2026-08-18) for what changed and
[§ Corrections](#corrections-to-the-2026-08-17-version) for two claims in the previous
version that were wrong.

- **What is already done** and the findings that came out of it:
  [`BLAZE_PROMOTION_TESTS_DONE.md`](BLAZE_PROMOTION_TESTS_DONE.md).
- **Forensic detail** for A5 stays in
  [`blaze_llk_promotion_test_strategy.md`](blaze_llk_promotion_test_strategy.md) §9. A4's §3 is
  now resolved — read it for the two dead ends, not for the plan.

Everything below was verified against the tree on 2026-08-18. Where an item says "uncalled",
that means no driver under `tt_metal/tt-llk/tests/sources/` references the symbol.

---

## At a glance

| # | Item | Type | Size | Blocked on |
|---|------|------|------|-----------|
| **A1** | `custom_mm` (plain) — **now under test**; transpose + split_acc/finalize left | test | ~0.5 d left | — |
| ~~A2~~ | ~~`top32_rm` — entire family untested~~ | — | — | **DONE 2026-08-20** (two combinations left, below) |
| **A2'** | `top32_rm` — mixed 1024+tail shape, and the metal wrapper layer | test | ~0.5 d + B1-shaped | — / needs an owner in the metal tree |
| ~~A3~~ | ~~`set_dst_write_addr_offset` behaviour~~ | — | — | **DONE 2026-08-18** |
| ~~A4~~ | ~~`mul_reduce_scalar_chunked_tile` untested~~ | — | — | **Defect located → C4** |
| **A5** | `eltwise_mul_scalar` HiFi init — untested, rationale disproved | test | unknown | C2 |
| ~~A6~~ | ~~Thin spots inside landed tests~~ | — | — | **DONE 2026-08-18** |
| **B1** | `custom_mm` vs `compressed_custom_mm` divergence guard | test, **outside tt-llk** | ~1 d | needs an owner (interim static guard landed) |
| **C1** | `dense_packing` W-stride not format-aware | **defect** | ~0.5 d once decided | owner decision |
| **C2** | `eltwise_mul_scalar` HiFi workaround rationale does not hold | **question** | — | #52709 author |
| **C3** | `topk_xl` → `eltwise_binary` reconfig escape | **defect**, pre-existing | unknown | needs an owner |
| **C4** | `mul_reduce_scalar` re-entry needs a DEST-section boundary | **defect** | unknown | needs an owner |
| **C5** | OOB metadata read shipped to main with #52727 | **defect**, NEW | minutes | needs the fix cherry-picked |
| **D1** | `mul_reduce_scalar_chunked_tile` ships with no caller | cleanup | — | C4 decides it |
| ~~D2~~ | ~~bare `false, false` in `rmsnorm.h:27`~~ | — | — | **DONE 2026-08-18** |
| ~~D3~~ | ~~`restore_tile_pack_mop` has no consumer~~ | — | — | **Resolved upstream: flag deleted** |
| **E** | PR mechanics | chore | minutes | — |
| **F** | `test_matmul_custom_compressed` hangs — host/BRISC desync | **defect**, nightly-only | unknown | needs an owner |

**A1 is no longer a hole.** #52727 merged on 2026-08-18 at 23:37 UTC, and the plain family now
has coverage (see below). **A2 is no longer one either**: #52713 merged, and `top32_rm` went
from zero coverage to 10 passing variants across both of its modes on 2026-08-20. What is left
of it is two combinations, tracked as A2' — the mixed 1024+tail shape, and the metal wrapper
layer, which a tt-llk test cannot reach at all.

---

## Closed on 2026-08-20

#52713 **merged**, which was A2's only blocker. `top32_rm` now has coverage; two commits on
`ldjurovic/llk-tests-blaze-promotions`.

### A2 — the top32_rm family is under test, `5b768385ee1` + `edcdc8f4157`

`tests/sources/top32_rm_test.cpp` + `tests/python_tests/test_top32_rm.py`.
**BH p100a: 10 passed** — 8 plain (row lengths 64/128/160/256 x `dest_acc` No/Yes) and 2
pre-sorted (1024, 2048). All nine LLK/SFPU entry points the family exposes to a kernel are now
called; the previous state was that **none** of them were, and `_top32_rm_init_` looked covered
on a grep only because the sole occurrence in the test tree was inside a comment.

Both of the consumer's modes are driven, statement for statement:

| Mode | Entry points | What it is |
|---|---|---|
| plain (`< 1024`) | `_llk_unpack_A_top32_rm_init_`/`_`, `_llk_math_top32_rm_init_`/`_`, `_top32_rm_init_`, `_bitonic_top32_phases_steps_`, `_bitonic_top32_merge_`, `_bitonic_top32_rebuild_` | `top32_rm_dev_compute.cpp`: 64 row-major elements at a time into a Dest **column**, sort, merge the running top32 across tiles |
| pre-sorted (`>= 1024`) | `_bitonic_top32_of_1024_rm_pre_sorted_{prep,combine,final}_` | `top32_rm_dev_compute_v2.cpp`: whole 32x32 tiles transposed into Dest, 16 columns reduced at once |

The Dest layout is what made this non-obvious. `_llk_unpack_A_top32_rm_` takes 64 consecutive
**row-major** elements — not a tilized tile — and lands them in the first COLUMN of 64 Dest
rows, 16 per face, transposed within the face. That column is what the bitonic sort addresses
(its distances are Dest rows: 8/16/32/64), the index operand has to sit exactly +2 tiles away
because `load16`/`store16` hardcode `dst_indices_offset = 128`, and the pack side narrows the
packer to one datum per row (`SETADCXX PAC`) to turn the surviving column back into 32
contiguous L1 elements.

Two deliberate deviations from the consumer, both recorded in the files:

- **One format for both operands.** The consumer runs bf16 values + uint32 indices, which needs
  a srcA reconfig between the two unpacks and a pack reconfig on the way out. The test carries
  indices as floats holding the integer itself, so it measures the sort rather than the
  reconfig sequence — C3 already owns the reconfig question. In the plain mode that caps the
  row at 256 (bf16 has 8 mantissa bits); the pre-sorted mode is Float32, where it does not bind.
- **Distinct values straddling zero.** Distinct means "the top 32" determines the indices, so
  indices are asserted **exactly** rather than tolerating hardware tie order the way
  `test_topk.py` has to. Straddling zero is what makes the -inf padding load-bearing: a
  32-element tail chunk fills two faces and the other two arrive as -inf from
  `CLR_SRC_NEGINF`, so with non-negative stimuli the `row=160` case would pass on padding alone.

**`dest_acc` turned out to be a real axis, not a formality.** It selects the index word width
inside the sort (`InstrModLoadStore::INT32` vs `LO16`) *and* the Dest-move opcode in
`llk_math_top32_rm_configure_mop` (ELWADD against a zeroed SrcB at fp32 Dest, MOVA2D
otherwise). The consumer only ever builds fp32 Dest, so the `dest_acc=No` cells are the first
exercise the 16-bit half of this family has had — and they pass.

Discrimination, per Finding 13 (a test that passes first try has not been shown to test
anything):

- plain mode: rebuilding the driver with a 4-byte chunk stride instead of 2 makes it re-read
  overlapping chunks, and the answer comes back `[63, 62, 61, 59, 57, ...]` against a golden of
  `[63, 62, 61, 60, 59, ...]`. Both assertions fire.
- pre-sorted mode: discriminates by construction at 2048, which is why that row length is in
  the sweep — the tiebreak is a permutation over all 64 runs, so **17 of the 32 winning indices
  land in the second 1024-chunk**. A driver that dropped the second chunk, or a `combine` that
  did not merge across tiles, cannot produce that answer.

**New finding, fixed in the same commit: `llk_math_top32_rm.h` did not compile under the tt-llk
build** (Finding 17 in the DONE document). `_llk_math_top32_rm_init_`'s `num_faces` and
`llk_math_top32_rm_configure_mop`'s `total_rows` are never read, and the tt-llk build compiles
with `-Werror=unused-parameter`, so the first test to include the header failed to build.
Dropped both and updated `llk_math_top32_rm_api.h`, its only caller — the same call the #53130
reviewers made on the two unpack headers. That is now **three** promoted headers in this state,
and the common cause is worth naming: a promoted header no in-tree test compiles has only ever
been compiled by the JIT path, under a weaker warning set.

**A2's second half is resolved rather than done:** the 7 `llk_math_deepseek_top32_rm_*` metal
wrappers dropped during #53130's review never needed restoring — they are on main, having
arrived with #52713 — and the recheck confirms they still have no caller anywhere. What is left
is covering the wrapper layer, which needs a metal-side test (Finding 18), same blocker as B1.

### Regression run for the 2026-08-20 change set

Everything the change set could touch, re-run serially through
`.claude/scripts/run_test.sh` on BH p100a. All PASS:

| Suite | Why it is in the set | Result |
|---|---|---|
| `test_top32_rm.py` | new | **10 passed** |
| `test_rmsnorm_bcast_scalar_dest_reuse.py` | `llk_unpack_A_rmsnorm.h` selector change | 114 passed, 114 skipped |
| `test_sfpu_sampling.py` | `pollute` axis pinned into the driver | 63 passed, 97 skipped |
| `test_custom_mm_uninit_restore.py` | driver header rescoped | 7 passed, 8 skipped, **1 xfailed** (C1, expected) |
| `test_custom_mm_uninit_parity.py` | dangling-reference edits | 2 passed |
| `test_matmul_custom_mm.py` | dangling-reference edits | 32 passed |
| `test_sort_headers_coexist.py` | same sort headers | 2 passed |
| `test_set_dst_write_addr_offset.py` | helper both sort families use | 14 passed, 14 skipped |
| `test_topk_xl.py` | the other sort family in this area (C3) | 100 passed |
| `test_matmul_custom_compressed.py` | branch's other half | 588 passed |

**Two things worth recording from doing it, both about the harness rather than the code.**

*Do not run plain `pytest` while `run_test.sh` holds the device.* Doing exactly that —
a background `pytest test_matmul_custom_compressed.py test_topk_xl.py` overlapping a
`run_test.sh` run — produced **137 failures out of 688 and a `TENSIX TIMED OUT`** (BRISC
command poll: python counter 1239 vs brisc 157). Serially afterwards, the same two suites are
100/100 and 588/588. So the "use `run_test.sh`, never `pytest`" rule is not only about
tidiness: the `flock` is what makes concurrent agents' results mean anything, and without it
the failure mode is a large, entirely spurious failure count that looks like a real regression.

*One intermittent value failure was seen and did not reproduce.* In the first serial run after
that incident, `test_matmul_custom_compressed[shape=(1, 64, 32), formats=('bfp4',)]` failed with
**PCC -0.033**, 587/588. Re-running that variant in isolation passes (17/17 for `single and
bfp4`), and the full suite then passes 588/588. Nothing in the 2026-08-20 change set is on the
compressed path. Recorded as a data point for **Finding 12 / item F** — the known
intermittency in this suite — and specifically as one that presents as a *wrong answer* rather
than a hang, which is new for that finding.

### What is left of A2 → tracked as A2'

1. **The mixed shape** — whole 1024-element chunks *plus* a 64-element tail, i.e. the Metal dev
   test's `row=3232`, which runs the pre-sorted mode and then finishes in the plain one. Both
   halves are covered; their composition is not. ~0.5 d: the driver already has both paths, so
   this is a tail loop plus the stimuli question (indices past 256 force Float32, which forces
   the plain mode's unpack down its 32-bit branch — the one that pads with **zeros** instead of
   -inf, so the tail chunk's padding stops being safe and that needs checking before it is
   claimed).
2. **The metal wrapper layer** — the 7 uncalled wrappers above. B1-shaped: needs an owner in
   the metal test tree.
3. The 8-datum `bitonic_top32_load8`/`store8` helpers stay uncovered on purpose; the header
   itself records that no kernel references them.

---

## Closed on 2026-08-19

#52727 **merged** (squash, `a85c79a9829`, 2026-08-18 23:37 UTC). #52713 re-checked and still
open. Branch rebased onto main and now sits at 44 commits.

### The rebase, and what it revealed

Five commits were **dropped**: the branch's own copy of #52727's promotion payload
(`a63a9fd2563` and the four that followed it — promote-branch SHAs, so they resolve on
`pmilenkovic/promote-custom-mm`, not here). Main has that content, but as a *squash* merge,
so git could not see them as duplicates and replaying them conflicted three times over.
Dropping them explicitly let the other 40 commits replay with a single conflict.

That one conflict is the important part. **Main's merged `*_block_uninit` has no
`restore_tile_pack_mop` and no MOP restore at all** — its body is the `dense_packing` W-stride
write and nothing else. The branch had gone one revision further than what shipped: an earlier
promote commit made the MOP restore unconditional, the next made it an opt-in flag, and
neither survived review. So the commit documenting that flag was dropped too, and D3 is
resolved upstream by deletion rather than by a decision anyone still owes.

- **`test_custom_mm_uninit_restore.py` was silently testing a ghost.** It replicates the uninit
  body rather than calling the API, so after the merge it kept passing while asserting a MOP
  restore no compute-API function performs. Narrowed to what shipped in `1d06517c59f`:
  `restores_dense_wstride` (the whole of the uninit, keeping the C1 fp32 xfail, which still
  reproduces), `is_load_bearing`, and a new `leaves_the_caller_mop_installed` that pins the
  **absence** of a restore via a 2-face block MOP. 7 passed / 8 skipped / 1 xfailed, from
  15/16/1 covering a flag that no longer exists.
- Note this is exactly the staleness class `test_custom_mm_uninit_parity.py` exists for, and it
  **did not catch it**: that guard compares the two bodies to each other and the W-stride
  expressions to the headers, neither of which changes when a knob the test drives disappears
  upstream. Worth knowing about the guard's reach.

### A1 — plain `custom_mm` now under test, `ed43f6f7b8f`

`tests/sources/matmul_custom_mm_test.cpp` + `tests/python_tests/test_matmul_custom_mm.py`.
**32 passed on BH p100a, PCC >= 0.99999.**

The shape is why this needed real work rather than a copy of `matmul_test.cpp`: operand a is a
full 32x32 4-face tile, operand b a narrow `[{1,2,4,8}, 32]` tile using only its top two faces,
one face per unpack instruction. So it computes `(M x K) @ (K x N)` for M in {1,2,4,8} from ONE
call per thread, with the operands **swapped** (full tiles in `buffer_B`, passed as
`base_address_a`). Stimuli therefore need a raw-bytes config like the compressed sibling's,
because the harness's tensor path assumes one tile layout for both operands; golden, result
reorder and the kt-scaled atol are reused from `helpers/compressed_utils.py`.

Two things the sweep pins that the doc tables only assert:

| Constraint | Where it comes from |
|---|---|
| `kt_dim` even | `_llk_unpack_AB_custom_mm_run_` issues `TT_MOP(0, (kt_dim / 2) - 1, 0)`, so an odd value runs the wrong iteration count. This is the origin of the tables' "even number from 2 to 256". |
| `ct_dim <= 8` | the ct output tiles are all live in DEST and half-sync holds 8 bf16 tiles, so **ct_dim > 8 is unreachable from a single call** in this configuration. The tables claim any integer 1..16; the upper half needs `DstSync::SyncFull` or a caller that splits the block. |

That second row **answers A1's open documentation question** (`ct ∈ {7, 9, 11}`): 7 works and is
swept along with 3, the other odd middle value; 9 and 11 are not reachable at all here, which is
a stronger statement than "unverified".

Verified to discriminate: offsetting operand a by one tile drops PCC from 0.999993 to **-0.092**.

**Left for the next increment** (~0.5 d, no longer blocked): `transpose`, and `split_acc` /
`finalize` — both of which ARE forwarded on this family, unlike the compressed one. Recorded in
the test files as well, not just here.

---

## Closed on 2026-08-18

All on `ldjurovic/llk-tests-blaze-promotions`, all verified on BH p100a. **Thirteen commits,
`54e218ebbce..096ff04e219`.**

> **SHAs were rewritten on 2026-08-18** when the branch was rebased onto `main`
> (`b62ff4a6af1`). Every SHA in this document is post-rebase and reachable from the branch
> tip; the pre-rebase tip is kept locally as `backup/pre-rebase-2026-08-18` on the machine
> that did it, and nowhere else. If a SHA here does not resolve, you are looking at a
> checkout from before that rebase — fetch again.

### PR #53130 review comments — 6 fixed, 3 already fixed, 2 reply-only

The eleven open threads were triaged against the tree. Fixed:

| Commit | Comment |
|---|---|
| `54e218ebbce` | Copilot 🔴 — out-of-bounds remainder metadata read in `llk_unpack_AB_compressed_custom_mm.h`, reachable at `kt_dim=10, ct_dim=1` |
| `0f07f6b3bd8` | dead `unpack_{src,dst}_format` deleted end to end — **3** files, not the 2 the bot predicted (the new tt-llk driver also passed them) |
| `e94b5dd0fbe` | imperative `pytest.xfail()` → marker (see Correction 1) |
| `1ebef7cd72c` | `add_rsqrt` asserted the sign of a value the implementation leaves undefined |
| ~~dropped~~ | ~~`restore_tile_pack_mop` documented as an *install*, not a restore~~ — **dropped in the 2026-08-19 rebase**, because main's merged #52727 has no such flag |
| `ec1fef679a0` | bare `0` → `0 /*addr*/` |

Already fixed by earlier commits, reply-and-resolve only: the sampling import order
(`f628a3de0be`), the `split_acc`/`finalize` doc tables (`795ff816b1f`), and the sort-header
coverage claims (`c6c52b16063`). Two need a written answer rather than a change: Copilot's
"exercise the real uninit API" (that is B1, and the limitation is already documented in the
test) and "PR metadata is still the template" (that is E).

### A3 — `set_dst_write_addr_offset` behaviour, `ecb5f96f8ab`

`tests/sources/set_dst_write_addr_offset_test.cpp` +
`tests/python_tests/test_set_dst_write_addr_offset.py` + a `DST_WRITE_ADDR_OFFSET` template
param. **14 passed, 14 skipped.**

Two facts made this simpler than the 2026-08-17 plan assumed:

- The offset register counts Dst **rows**, 64 per 32x32 tile, because
  `math::set_dst_write_addr<Tile32x32>(tile_index)` computes `tile_index << 6`. So at a
  multiple of 64 the helper computes the same address as the LLK's own function, which gives
  an exact assertion needing no model of DEST layout:
  `helper(N * 64) at dst_index 0 == no helper at dst_index N`. That is better than the
  "write a sentinel and find it" plan, which would have needed the face layout.
- `VectorMode::None` is **required**, not incidental. `RC` invokes the body once per face with
  `_llk_math_eltwise_sfpu_inc_dst_face_addr_()` in between, and the helper writes an absolute
  address — so under RC every face is redirected to the same rows. The real consumers do their
  own addressing for the same reason.

Both real call patterns are covered: `tile_offset` (`dst_index << 6`) and `tile_offset + 2`.
Mutation-verified — a helper that discards the offset fails 10 of 14, and one that reads rows
as datums trips the sub-tile spill check.

**Caveat recorded in the file:** the `tile=0` variants cannot detect a discarded offset, since
offset 0 is a no-op even when the helper is broken. They pin that offset 0 does *not* move the
write, which is a different claim. A `tile=0`-only pass is not evidence about the helper.

**Not done, deliberately:** the `LLK_ASSERT(addr < DEST_REGISTER_HALF_SIZE)` negative test that
the old plan listed as step 4. Nothing in the suite expects an LLK assert — conftest reports
`LLKAssertException` as a failure — and tripping one mid-kernel risks wedging the device for
whatever runs next. If that is wanted, the harness needs an expected-assert mechanism first.

### A6 — thin spots, `18f07722c90`

`test_rmsnorm_bcast_scalar_dest_reuse.py`: **114 passed, up from 66.** Three of the five rows
in the old A6 table; the other two were "leave it" and "fine as-is" and still are.

- **HiFi3 added.** It was missing from every sweep in the suite while `_FIDELITY_PHASES`
  already defined it as 3. It is the one fidelity whose phase count is not a power of two, so
  an implementation deriving the bound by shifting would pass LoFi/HiFi2/HiFi4 and fail only
  there. The accumulate test now shares `_fidelities()` instead of repeating the list.
- **Transpose-fold swept over fidelity.** For ELWMUL the replay buffer decides which face
  lands in which SrcA bank while each phase masks a different mantissa slice; nothing ran the
  two together.
- **Partial faces widened to `num_tiles ∈ {1,2}` and all fidelities.** This is the one with
  substance. The addr_mod increments DEST by `8 + (4 - num_faces) * 16`, i.e. after a tile's
  covered faces it **skips the uncovered ones onto the next tile's base**, while the unpack
  reads `num_tiles * num_faces` faces *contiguously* out of L1. The golden therefore slices
  the input contiguously and indexes the output per tile — assuming one contiguous run would
  have produced a wrong golden. `num_tiles` is the load-bearing axis: halving the skip term
  fails every `num_tiles=2` variant and **no** `num_tiles=1` variant, so that term was
  previously unpinned.

### Boundary coverage nothing reached, `ae095985110`

`test_matmul_custom_compressed_metadata_word_boundary`, 3 shapes with `kt*ct` divisible by 10
(`kt=10/ct=1`, `kt=2/ct=5`, `kt=10/ct=2`). **588 passed**, was 582.

The 16 shapes in `SHAPES` have products 2, 4, 8, 16, 28, 32, 56, 64, 112, 128, 192, 224, 448,
512 — **none divisible by 10**, so nothing exercised the path `54e218ebbce` fixed.

**Scope, stated in the test:** this does **not** detect the out-of-bounds read, and cannot. At
`rem_iters == 0` the remainder loop never runs, so the word read past the buffer is never used
and the golden is unaffected — confirmed by running the six variants against the unguarded
kernel, where they also pass. Catching that read needs an L1 memory-safety check. What the
test buys is the exact-multiple shape class, so a later change that mishandles it (stale word,
or dropping the final group of 10 tiles) fails on the golden.

### D2 — `729f712aaf8`

`false, false, icb0` → `false /*transpose_of_faces*/, false /*within_face_16x16_transpose*/,
icb0` in `rmsnorm_bcast_scalar_reuse_tiles_init`, matching the `_fidelity` variant below it.
Comment-only. No tt-llk test compiles the compute API — that is B1 — so there is nothing to
run for it.

---

## Corrections to the 2026-08-17 version

**Correction 1 — C1's tripwire was inert.** The old C1 said "the `xfail` flips to XPASS the
moment either lands, so the test tells you when it is fixed." That was **not true when
written**: `test_custom_mm_uninit_restore.py` used imperative `pytest.xfail()`, which raises
immediately and aborts the body, so the variant never built, never ran, and could never report
XPASS. C1's owner decision was resting on a detector that was not armed. Fixed in
`e94b5dd0fbe`; the sentence is true now, and the fix is visible in the run as a real
golden-vs-device comparison under XFAIL rather than a bare skip.

**Correction 2 — A4's hypothesis was too coarse.** The old A4 said the prime suspect was "that
`_llk_math_mul_reduce_scalar_init_` is not re-enterable". It is re-enterable; what breaks is
re-entry *without an intervening DEST-section boundary*. See C4.

---

## A. Functional test gaps

### A1. `custom_mm` (plain) — **under test since 2026-08-19**, two axes left

**Unblocked and largely done** — see [§ Closed on 2026-08-19](#closed-on-2026-08-19) for what
landed and what it found. What remains is `transpose` and `split_acc` / `finalize`, roughly half
a day, and nothing blocks it.

Historical note, since the framing below was written when this was a hole. Every top-level entry
point *was* uncalled: `_llk_math_custom_mm_init_`, `_llk_math_custom_mm_`,
`_llk_unpack_AB_custom_mm_init_`, `_llk_unpack_AB_custom_mm_`.

Note the asymmetry that makes this easy to miss: **`compressed_custom_mm` is covered** by
`test_matmul_custom_compressed.py`, so the compressed variant is exercised and the plain,
simpler one is not. `test_matmul_custom.py` does *not* cover it either — that drives
`llk_math_matmul_custom_no_mop.h`, an unrelated family. The only `custom_mm` thing tested today
is `block_uninit`, and only via a replicated body (see B1).

**Plan.**

1. Start from `tests/sources/matmul_custom_compressed_test.cpp` and strip the compression path
   rather than writing from scratch.
2. Drive the LLK pair directly: `_llk_unpack_AB_custom_mm_init_<transpose>` +
   `_llk_unpack_AB_custom_mm_`, `_llk_math_custom_mm_init_<transpose, split_acc, dense_packing>`
   + `_llk_math_custom_mm_<finalize>`.
3. Sweep what the doc tables constrain: `kt_dim` even 2..256; `ct_dim` 1..16; `rt_dim` 1; LoFi
   only; in0 tile shape `[{1,2,4,8}, 32]`. **`ct ∈ {7, 9, 11}` is the open documentation
   question** this item should settle.
4. `split_acc` and `finalize` *are* forwarded on this family, unlike the compressed one. Sweep
   both; that asymmetry is worth pinning.
5. Reuse the existing matmul golden and `helpers/matmul_sweep.py`. Do **not** write a new
   golden generator.
6. **New:** include shapes with `kt*ct` divisible by 10 if the plain family has an equivalent
   metadata walk. On the compressed side nothing reached that boundary until `ae095985110`.

**Watch for.** The `-Werror` prerequisite (Finding 5) — budget for a build fix before any test
compiles.

### A2'. `top32_rm` — **under test since 2026-08-20**, two combinations left

The family went from zero coverage to 10 passing variants across both modes; see
[§ Closed on 2026-08-20](#closed-on-2026-08-20) for what landed, how it discriminates, and the
header defect it turned up. What remains:

**1. The mixed shape (~0.5 d, unblocked).** `row = 3232` in the Metal dev test: whole
1024-element chunks through the pre-sorted path, then a 64-element tail through the plain one.
Both halves pass in isolation; their composition is untested. The driver already contains both
paths, so this is a tail loop plus one open question — indices past 256 force Float32, and
Float32 sends the plain mode's unpack down its **32-bit** branch, which pads with zeros rather
than the `CLR_SRC_NEGINF` the 16-bit branch uses. A tail chunk that pads with zeros is only
safe for non-negative inputs, so that needs establishing rather than assuming.

**2. The metal wrapper layer (B1-shaped, needs an owner).** The 7
`llk_math_deepseek_top32_rm_*` wrappers are on main with no caller — they arrived with #52713,
not with this branch, so the #53130 removal is moot. A tt-llk test cannot reach the metal API
layer; covering them needs a metal-side test, exactly like B1.

**Not planned:** the 8-datum `bitonic_top32_load8`/`store8` helpers, which the header records as
referenced by no kernel.

**Still true, and still worth knowing before touching this area:** `_top32_rm_init_()` and
`_topk_xl_init_<K, fused>()` **cannot both be called in one kernel** — they overlap in the
ADDR_MODs, the MOP and the REPLAY buffer, and the math thread hangs (Finding 3). And see C3: a
pre-existing reconfig escape lives in this area, so bisect single-file-then-target before
blaming your own driver.

### A5. `eltwise_mul_scalar` HiFi init — untested, and its rationale does not hold

**Unchanged, still blocked on C2.** Smaller than it looks: the underlying shapes **are** covered
generically — `test_eltwise_binary.py` sweeps `DEST_TO_SRCA`/`DEST_TO_SRCB` dest-reuse and
`BroadcastType.Scalar`. What has no test is the **HiFi init sequence** specifically.

Resolve C2 first. If the workaround's real mechanism is something else the test to write
changes completely, and if it is inert the honest outcome may be deleting it rather than
testing it. Read §9 for the earlier attempt, which hung the device as first written.

---

## B. Gap that cannot live in tt-llk

### B1. Nothing can catch `custom_mm` vs `compressed_custom_mm` divergence

**Unchanged.** `custom_mm_uninit_restore_test.cpp` **replicates** the uninit body rather than
calling `custom_mm_block_uninit` / `compressed_custom_mm_block_uninit`, because a tt-llk test
cannot include `tt_metal/hw/inc/api/compute`. The two bodies are currently identical; if they
diverge, every existing test keeps passing. Copilot raised the same point independently on
#53130.

**Plan.** A compute-kernel test under `tests/tt_metal/` that calls the real entry points. ~1 d,
but it needs an owner who works in that tree.

**Cheaper interim option — DONE 2026-08-18.** `tests/python_tests/test_custom_mm_uninit_parity.py`
(commit `096ff04e219`), a device-free static gate rather than a pre-commit hook, so it runs in
the smoke job that already collects the whole `python_tests` directory. It asserts two things:

- the two compute-API uninit bodies are still byte-identical modulo comments (divergence); and
- the driver's replicated `DENSE_WSTRIDE` / `DEFAULT_WSTRIDE` expressions are still the ones
  the headers use (staleness) — the driver hardcodes them, so if a header changes the driver
  keeps asserting the old behaviour *and passes*, because it programs that stride itself.

Both mutation-checked: dropping the `restore_tile_pack_mop` branch from
`compressed_custom_mm.h` fails the first with a diff naming the missing branch; changing the
driver's `DENSE_WSTRIDE` to `* 4` fails the second.

**This does not close B1.** A text match cannot say the functions *work*, only that they still
say the same thing. The metal-side test calling the real entry points is still wanted, and is
still the item that needs an owner. What the guard buys is that divergence now fails loudly
instead of silently, which was the specific risk.

**Note.** The commit that documented `restore_tile_pack_mop` was dropped in the 2026-08-19
rebase — main's merged #52727 has no such flag at all, so there was nothing left to document.

---

## C. Product issues needing a decision, not a test

### C1. `dense_packing` W-stride is not format-aware — **defect**

`set_packer_strides` (`cpack_common.h:301-305`) derives the field as
`TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * datum_size_in_bytes(pack_src_format)`, while
`custom_mm.h:69` / `:261` and `compressed_custom_mm.h:69` / `:262` hardcode `* 2`. On a Float32
pack source both halves are 2x off: init programs 1024 where 2048 is correct, and the uninit
restores 2048 where 4096 is correct. Measured at 0.25 match.

**Owner:** whoever owns `custom_mm.h`. **Decision needed**, then ~half a day:

- **Option 1 — guard.** `LLK_ASSERT` in `*_block_init` that
  `datum_size_in_bytes(pack_src_format[out_cb]) == 2` when `dense_packing` is set. No API
  change; turns silent corruption into a loud failure; leaves 32-bit unsupported.
- **Option 2 — full fix.** Derive the datum size in init from `out_cb_id` and **add an
  `out_cb_id` parameter to `*_block_uninit`**, which currently takes none. Correct on 32-bit,
  but changes a signature that `matmul.hpp`, `flash_mla.hpp`, `dram_streaming_matmul*.hpp` and
  `matmul_custom_compressed_kernel.cpp` all call.

The `xfail` in `test_custom_mm_uninit_restore.py` flips to XPASS when either lands — **and as of
`e94b5dd0fbe` that is actually true**; see Correction 1.

### C2. The `eltwise_mul_scalar` HiFi workaround's mechanism does not survive review

**Unchanged.** `deepseek_binary_dest_reuse_tiles_init`'s HiFi branch hardcodes
`DEFAULT_TENSOR_SHAPE` and attributes a HiFi4 fix to the shorthand init "mis-specialising the
tile shape". But `get_effective_math_fidelity<ELWMUL, f>()` is the identity for ELWMUL,
`acc_to_dest` is 0 in both arms, and the shorthand resolves the shape from the CB regardless of
fidelity — so tensor shape is the only difference, and on a standard 4-face CB
`get_operand_tensor_shape` returns exactly `DEFAULT_TENSOR_SHAPE`, making the HiFi arm
bit-identical to the shorthand it replaces. Meanwhile the paired execute *does* derive the shape
from the CB, so on non-default geometry init and execute disagree.

Combined with the measured fact that forcing `DEFAULT_TENSOR_SHAPE` on a 2-face tile
**deadlocks the MATH_PACK handshake**, the workaround is either inert (4-face CB) or hangs
(2-face CB). The cited failing config is `gated_local_reduce` at HiFi4 (0.70 → 0.9996), and that
measurement is not explained by the stated mechanism.

**Owner:** the #52709 author. **Needed:** either the real mechanism, or a corrected comment. A5
is blocked on this.

### C3. Pre-existing `topk_xl` → `eltwise_binary` reconfig escape — **defect**

**Unchanged.** Reproduces on clean `main` with every promotion change stashed, so it is
unrelated to the promotions. Recorded because A2 shares the sort headers, so whoever picks that
up will see a failure in this area and assume it is theirs. **Bisect single-file-then-target
before blaming your own driver.** `tt-smi -r` must **not** be used to paper over it. Needs an
owner. See also F, which may or may not be the same thing.

### C4. `mul_reduce_scalar` re-entry needs a DEST-section boundary — **defect, NEW**

This is what A4 turned into. `b59c5df50aa` adds a ~40-line driver
(`tests/sources/mul_reduce_scalar_reenter_test.cpp` +
`tests/python_tests/test_mul_reduce_scalar_reenter.py`) that runs the known-good non-chunked
sequence twice over the same input. On BH p100a:

| Configuration | Result |
|---|---|
| `passes=1`, either mode (control) | correct |
| `passes=2`, DEST-section boundary between passes | correct, and **bit-identical** across passes |
| `passes=2`, one shared DEST section | **wrong — all 12 variants, 9.27x to 9.93x golden** |

So the family **is** re-enterable. What is broken is re-entry with no
`dest_section_done` / `wait_for_dest_available` pair in between: that handshake restores whatever
the second `_llk_math_mul_reduce_scalar_init_` does not.

**And that is exactly how the chunked op is built.** `mul_reduce_scalar_chunked_tile`
(`rmsnorm.h:105`) documents that the caller "must ... acquire DST before calling", then
re-enters every batch inside that one section, with `if (batch > 0) mul_reduce_scalar_init(...)`
as its only restoration attempt. The reverted chunked driver reported 5-30x golden and "not a
clean multiple of anything"; this reproduces 9.3-9.9x, also non-integer. Same signature, so
very likely the same defect — now with a minimal reproducer instead of a full chunked
implementation.

**Owner needed.** Two shapes the fix could take, and it is not this document's call which:

- **In the LLK** — make `_llk_math_mul_reduce_scalar_init_` (or `switch_to_reduce`) restore
  whatever the section boundary restores. Right if re-entry inside a section is meant to work.
- **In the compute API** — have `mul_reduce_scalar_chunked_tile` close and reacquire the DEST
  section per batch, or document that it cannot be used as written. Right if the per-batch
  handshake is considered the caller's job.

**Whoever takes it:** the reproducer is `--test test_mul_reduce_scalar_reenter.py`, and the
`single_dest_section` axis is the whole experiment. The 12 failing variants are xfail (marker
form, so the body runs), and flip to XPASS the moment re-entry inside one section restores
state. Do **not** re-investigate the accumulator fill or a missing UNPACK/MATH barrier —
§3 records both as tried on silicon and disproved, and this result explains why neither moved
the number.

### C5. The out-of-bounds metadata read shipped to main — **defect, NEW**

`#52727` merged **without** the fix for the out-of-bounds remainder read Copilot found on
#53130. Verified on 2026-08-19: `grep 'rem_iters != 0'` on main's
`llk_unpack_AB_compressed_custom_mm.h` returns nothing, so the unguarded
`meta_ptr[full_iters]` is live on main.

The guard exists only on this branch (`54e218ebbce`). Reachable inside the documented ranges
whenever `kt_dim * ct_dim` is a multiple of 10 — `kt_dim=10, ct_dim=1` is the smallest case.

**What it costs, stated precisely so nobody over- or under-reacts.** At `rem_iters == 0` the
remainder loop never runs, so the word read past the buffer is *never used* and no golden can
see it — confirmed by running the boundary test against the unguarded kernel, where it passes.
It is a memory-safety defect, not a wrong-answer defect: an L1 read of whatever follows the
metadata buffer.

**Fix:** cherry-pick `54e218ebbce` onto main, or re-apply the three-line guard. Minutes of work.
It will otherwise ride in on this branch whenever #53130 merges, which is fine but leaves main
carrying it in the meantime.

---

## D. Review comments resolved but not fixed

- **D1 — `mul_reduce_scalar_chunked_tile` ships with no caller and no test.** C4 now says the
  op is broken as written, not merely untested. Removal is a legitimate outcome, and is now the
  cheaper one unless someone wants the chunked form to work.
- ~~**D2**~~ — done, `729f712aaf8`.
- ~~**D3 — `restore_tile_pack_mop` is end-of-call-cleanup with no consumer.**~~ **Resolved
  upstream on 2026-08-18: the flag was deleted.** Main's merged `*_block_uninit` has no MOP
  restore at all — an earlier revision of #52727 made it unconditional, the next made it this
  opt-in flag, and neither survived review. The reviewer's suggestion on #53130 (pair the fused
  caller with `pack_block_contiguous_uninit` instead of adding a flag to the op uninit)
  effectively won. Nothing left to decide, and no `custom_mm.h` owner needed. Historical
  reasoning, no longer actionable: the branch had corrected its
  documentation — it *installs* fixed 32x32/4-face geometry rather than restoring anything, it
  restores nothing at all on the `_init_short` path, it leaves `set_packer_strides`/`SETADCXX`
  untouched, and its body is byte-identical to the pre-existing
  `pack_block_contiguous_uninit()` — which, note, is **also gone from main**: neither
  `pack_block_uninit.h` nor that function exists in the compute API any more, so that whole area
  was reshaped by the merge, not just the flag.

---

## E. PR mechanics

- **The title still reads `[do not review]`.** Re-checked 2026-08-18: still there, and the PR
  shows `REVIEW_REQUIRED`. This is the single thing blocking anyone from looking at it.
- **CI was red for a reason that predates this session's work, and is now fixed.**
  `llk_smoke_blackhole group 2/2` failed on
  `test_perf_header_gate.py::test_parameter_field_names_are_globally_unique` — `RMSNORM_DEST_REUSE`
  declared bare `num_tiles` / `num_faces`, names already owned by `PACK_NUM_TILES` and
  `NUM_FACES`. The identical failure was present on the pre-session tip, so it arrived with the
  commit that added that class, not with any later work. Fixed in `55b57d28045` by renaming to
  `rmsnorm_num_tiles` / `rmsnorm_num_faces`, which also matches the constants they emit.
  **Note that job runs `pytest -x`**, so it aborted at the gate: anything after it in the split
  was never reached, and may surface now that it passes. That would be newly *visible*, not
  newly broken.
- **The body is still the untouched template**, with an empty Summary, which Copilot also
  raised. Drafted title, Summary and Notes-for-reviewers are committed alongside this file as
  [`pr-53130-replies.md`](pr-53130-replies.md), together with a reply for each of the eleven
  review threads. Copy-paste ready; nothing in it has been posted, because the session that
  wrote it had no GitHub write access.
- **Rebased again on 2026-08-19**, onto main with #52727 merged. Five commits dropped (the
  branch's copy of that promotion, which main has as a squash) and one skipped (the
  `restore_tile_pack_mop` documentation, whose subject no longer exists). 44 commits now.
- **Rebased onto `main` (`b62ff4a6af1`) on 2026-08-18** — 49 commits replayed with **zero
  conflicts**. Two predictions in the earlier version of this document were wrong and are worth
  correcting for next time: the expected `tt_metal/hw/sources.cmake` conflict did **not**
  happen (main has not touched that file since the merge-base), and the only file both sides
  touched was `helpers/test_variant_parameters.py`, where main's four new parameter classes and
  this branch's seven merged without conflict (91 base + 7 + 4 = 102, verified by count).
- **#52727 is in; #52713 is not.** Re-checked 2026-08-19. The branch no longer carries the
  custom_mm payload (dropped in the rebase) but still carries `top32_rm`'s, so rebase again when
  #52713 lands and expect those five commits to drop the same way.
- `backup/llk-tests-pre-rebase` is a local-only safety ref from the first rebase; delete it
  once you are satisfied.

---

## F. Intermittent `test_matmul_custom_compressed` hangs — **diagnosed 2026-08-18**

Was an unidentified single failure; now characterised. Six back-to-back runs of the suite on
BH p100a:

| run | outcome |
|---|---|
| 1 | 588 passed |
| 2 | **hang** (exit 5) in `test_matmul_custom_compressed_clustered` |
| 3 | 2 failed — build-tree race, see below |
| 4 | 3 failed (`TTException`) in `test_matmul_custom_compressed_single` |
| 5 | **hang** (exit 5) in `test_matmul_custom_compressed_interleaved` |
| 6 | 588 passed |

**It is a hang, not a golden mismatch.** `run_test.sh`'s triage on run 2 caught the state:

```
Unpacker/Math/Packer mailboxes = 0x0 (KERNEL_STARTED)
TRISC0/1/2  in_reset=True
BRISC       pc=0x368, unchanged  (spinning)
BriscCounter=0x118 (280)   host Python counter: 281
```

All three TRISCs sit in soft reset while BRISC spins one command behind the host — a host↔BRISC
command desync, not an LLK compute bug. `get_tensix_state` then failed to halt BRISC, so the
device was already unresponsive.

**It does not affect the PR gate.** Every failing variant reproduced —
`clustered`, `interleaved`, `single` — is `@pytest.mark.nightly`, and the gate filters
`not nightly`. It would affect a nightly run.

**Two caveats on this reproduction, both important:**

1. **Back-to-back runs are not how CI runs it,** and may be the aggravating factor rather than
   an independent trigger. Runs 1 and 6 were clean; the failures cluster in the middle.
2. **Run 3 is not a real failure.** It hit
   `test_matmul_custom_compressed_metadata_word_boundary` with
   `ld: cannot open output file .../elf/pack.elf: No such file or directory` — a `/tmp/tt-llk-build`
   tree race left behind when run 2's hang handler killed the process tree mid-compile. An
   artifact of looping, not a defect, but worth knowing since it is the one failure that landed
   on a gate-visible (non-nightly) test.

**Six runs also wedged the device** (`PcieHangError`, all devices unhealthy), needing
`tt-smi -r`. That is the sanctioned remedy here per the tt-llk notes — a runtime timeout, not a
reconfig escape — but it means this reproduction is not free, and whoever repeats it should
expect to reset.

**Still needs an owner**, and it is now a much better-specified ask than before: a host/BRISC
command-protocol desync under repeated kernel launches, reproducible ~2 in 6 on p100a, with
triage output above. Whether it is the same phenomenon as C3's reconfig escape is still
unproven — C3 is a golden mismatch under a specific test ordering, this is a hang, so they are
probably different.

## Environment setup, for whoever picks this up next

The tt-llk suite would not run at all on a fresh dev box on 2026-08-18. Three separate
blockers, all fixed, none of them documented in `tests/README.md`:

**The short version: run `source tests/setup_external_testing_env.sh`.** §8 of the DONE
document already says so, and it does the whole job — creates `tests/.venv`, installs
`requirements.txt` with the `--index-strategy unsafe-best-match` that the multi-index
requirements file needs, and fetches SFPI. The three blockers below are what you hit if you
*do not* find that script first, recorded because each one presents as something worse than it
is:

1. **No `tests/.venv`.** `.claude/scripts/run_test.sh` requires it (exits 3, `ENV_ERROR`) and
   activates it for `ttexalens`. The ambient `/opt/venv` had `tt-exalens 0.3.11` where
   `requirements.txt` pins **0.3.29**, and `CallstackEntry` moved modules between those
   versions. That ImportError blocks *collection of the entire suite* from `conftest.py`, so it
   reads like a broken repo rather than one stale dependency.
   **Confirmed again on 2026-08-20:** `tests/.venv` (0.3.29) is present and `run_test.sh` works,
   while `/opt/venv` is still 0.3.11 — so `python -m pytest` with the ambient interpreter still
   fails at collection. If you are going to reach for pytest anyway, the pinned versions are
   `tt-exalens 0.3.31` **plus `tt-umd 0.9.9`**; 0.3.31 against the ambient `tt_umd 0.9.3` fails
   later and more confusingly, on
   `TopologyDiscoveryOptions.device_init_failure_action`. Use `run_test.sh`.
2. **No SFPI toolchain** (`tests/sfpi/`) — every compile fails with
   `riscv-tt-elf-g++: not found`, which looks like a broken driver rather than a missing
   toolchain. `tests/setup_testing_env.sh` fetches and sha256-verifies it (7.69.0), but note it
   *also* runs `pre-commit install`; run only the download half if you do not want git hooks
   added to your checkout.
3. **`uv` cache** — if `~/.cache` is a dangling symlink (it was, to a non-existent
   `/localdev/$USER/.cache`), `uv venv` fails on cache init. `mkdir` the target or set
   `UV_CACHE_DIR`.

Both `tests/sfpi/` and `tests/.venv` are gitignored, so none of this shows up in `git status`.

**Use `run_test.sh`, never `pytest` directly** — the tt-llk `CLAUDE.md` requires it, and it
serialises silicon access with `flock`, kills stale processes, and triages hangs. Its default
`--maxfail 10` will stop a run early; pass `--maxfail` higher when you want a full count.

**On method.** Every test added on 2026-08-18 passed on the first hardware run, and in every
case a mutation was what established it was not vacuous — a deliberately broken helper, a
halved addr_mod stride, an unguarded load. Two of those mutations changed what the test
claimed: one exposed that the `tile=0` variants prove nothing about the helper, another that
the OOB read cannot be detected by any golden. A test that passes first try has not yet been
shown to test anything.

---

## Explicitly out of scope

**Perf tests.** There is no perf coverage for any promoted family (nor for `topk_xl` or
`sampling`), and 56 functional test modules have no perf counterpart. This was reviewed and
**deliberately ruled out** — recorded here so it is not re-raised as an oversight. The perf
infrastructure itself is ready if that changes: discovery is marker-driven with pytest-split
sharding, `PerfRunType` already provides the isolation modes, and no registry edit is needed to
onboard a new op. The two things that *would* need doing first are wiring
`compare_test_and_perf.py` into CI (it exists, runs nowhere) and fixing its filename-based
pairing, which reports real pairs as unmatched.

---

## Suggested order

1. **E** — retitle the PR. Minutes, and nothing else gets reviewed until it is done.
2. **C5** — cherry-pick the OOB guard onto main. Minutes, and main is carrying the defect until
   someone does.
3. **C4** — route to an owner. It is a located defect in a shipping op with a minimal
   reproducer, which makes it the cheapest real fix on the list, and it decides D1.
4. **C1 / C2 / C3** — route to owners too; they are decisions, and C2 gates A5.
5. **A1's remainder** — `transpose` and `split_acc` / `finalize`, ~0.5 d, unblocked. Then
   **A2'** — the mixed 1024+tail `top32_rm` shape, also ~0.5 d; A2 itself closed on 2026-08-20.
6. **B1** once an owner in the metal tree exists.
7. **F** — now diagnosed (host/BRISC command desync, nightly-only). Route to an owner with the
   triage output; no further reproduction needed, and repeating it costs a device reset.
