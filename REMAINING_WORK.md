# tt-llk blaze promotions — everything still to do

Single actionable index of what is left, with a plan per item. Compiled 2026-08-17 after the
rmsnorm dest-reuse item landed and PR
[#53130](https://github.com/tenstorrent/tt-metal/pull/53130) went through review.

- **What is already done** and the eight findings that came out of it:
  [`BLAZE_PROMOTION_TESTS_DONE.md`](BLAZE_PROMOTION_TESTS_DONE.md).
- **Forensic detail** for the two items that were attempted and reverted (A4, A5) stays in
  [`blaze_llk_promotion_test_strategy.md`](blaze_llk_promotion_test_strategy.md) §3 and §9 —
  it is long, and this file deliberately does not repeat it. Read those sections before
  restarting either item.

Everything below was verified against the tree at `origin/main` on 2026-08-17, not carried
forward from an older plan. Where an item says "uncalled", that means no driver under
`tt_metal/tt-llk/tests/sources/` references the symbol.

---

## At a glance

| # | Item | Type | Size | Blocked on |
|---|------|------|------|-----------|
| **A1** | `custom_mm` (plain) — entire family untested | test | ~2 d | #52727 merging |
| **A2** | `top32_rm` — entire family untested | test | ~3–4 d | #52713 merging |
| **A3** | `set_dst_write_addr_offset` — behaviour never asserted | test | ~0.5 d | — |
| **A4** | `mul_reduce_scalar_chunked_tile` — untested, ships publicly | test | ≥1 d spent, unfinished | — |
| **A5** | `eltwise_mul_scalar` HiFi init — untested, rationale disproved | test | unknown | C2 |
| **A6** | Thin spots inside landed tests | test | ~1 d total | — |
| **B1** | `custom_mm` vs `compressed_custom_mm` divergence guard | test, **outside tt-llk** | ~1 d | needs an owner |
| **C1** | `dense_packing` W-stride not format-aware | **defect** | ~0.5 d once decided | owner decision |
| **C2** | `eltwise_mul_scalar` HiFi workaround rationale does not hold | **question** | — | #52709 author |
| **C3** | `topk_xl` → `eltwise_binary` reconfig escape | **defect**, pre-existing | unknown | needs an owner |
| **D1–D3** | Review comments resolved but not fixed | cleanup | ~0.5 d total | against `main` now |
| **E** | PR mechanics | chore | minutes | — |

Two families — A1 and A2 — are the real holes: both are promoted, both ship, and neither has
a single line of coverage.

---

## A. Functional test gaps

### A1. `custom_mm` (plain) — the entire family is untested

**Evidence.** Every top-level entry point is uncalled:

```
_llk_math_custom_mm_init_        _llk_math_custom_mm_
_llk_unpack_AB_custom_mm_init_   _llk_unpack_AB_custom_mm_
```

Note the asymmetry that makes this easy to miss: **`compressed_custom_mm` is covered** by
`test_matmul_custom_compressed.py`, so the compressed variant is exercised and the plain,
simpler one is not. `test_matmul_custom.py` does *not* cover it either — that drives
`llk_math_matmul_custom_no_mop.h`, an unrelated family. The only `custom_mm` thing tested
today is `block_uninit`, and only via a replicated body (see B1).

**Plan.**

1. Start from `tests/sources/matmul_custom_compressed_test.cpp` — the compressed sibling is
   the closest working driver and the two families share their block structure. Strip the
   compression path rather than writing from scratch.
2. Drive the LLK pair directly:
   `_llk_unpack_AB_custom_mm_init_<transpose>` + `_llk_unpack_AB_custom_mm_` on unpack,
   `_llk_math_custom_mm_init_<transpose, split_acc, dense_packing>` +
   `_llk_math_custom_mm_<finalize>` on math.
3. Sweep the axes the compute API exposes and the doc tables constrain: `kt_dim` even,
   2..256; `ct_dim` 1..16; `rt_dim` 1; LoFi only; in0 tile shape `[{1,2,4,8}, 32]`.
   **`ct ∈ {7, 9, 11}` is the open documentation question** this item is expected to settle —
   the tables claim 1..16 but nothing verifies the odd middle values.
4. `split_acc` and `finalize` are forwarded on this family (unlike the compressed one, where
   both are hardcoded off — see the doc-table caveats added by review). Sweep both here;
   that asymmetry is itself worth pinning.
5. Reuse the existing matmul golden and `helpers/matmul_sweep.py`. Do **not** write a new
   golden generator.

**Watch for.** The `-Werror` prerequisite (Finding 5) — budget for a build fix before any
test compiles. Check both headers for dead locals first; it costs an hour and avoids
mistaking a build failure for a driver bug.

---

### A2. `top32_rm` — the entire family is untested

**Evidence.** All 19 SFPU entry points are uncalled — `_bitonic_top32_merge_`,
`_bitonic_top32_rebuild_`, `_bitonic_top32_phases_steps_`, the four `ph*_st*_to_1` stages,
`load16`/`store16`, `inc_x4_dest`/`inc_x8_dest`, `step_N`, `_top32_rm_configure_addrmod_`,
and the three `_of_1024_rm_pre_sorted_*` functions — plus all four LLK wrappers
(`_llk_math_top32_rm_init_`/`_`, `_llk_unpack_A_top32_rm_init_`/`_`).

`_top32_rm_init_` looks covered on a naive grep. **It is not**: the only occurrence anywhere
in the test tree is inside a comment in `sort_headers_coexist_test.cpp`. Verify with
`grep -n` before believing any coverage claim here.

**This item also gates a removal.** The 7 `llk_math_deepseek_top32_rm` metal wrappers were
dropped from the promotion during review of #53130 — no in-tree caller, no test. They come
back when this test exists to justify them, and not before.

**Plan.**

1. Two modes, per the family's own shape: the plain `top32_rm` sort and the
   `top32_of_1024_rm_pre_sorted_*` three-stage path (`prep` → `combine` → `final`). Treat
   them as two test functions in one file, not two files.
2. Model the driver on `tests/sources/topk_xl_test.cpp` — same sort domain, same SFPU
   idioms, and it already solves the stimuli/golden shape for bitonic results.
3. Golden: extend `TopKGolden`/`TopKXLGolden` rather than writing a third sort golden.
   Expect tie-handling to be the hard part — `test_topk_xl.py` already distinguishes
   "signed" (deterministic) from "random" (ties likely, compare the value multiset and
   K distinct indices only). Copy that split.
4. Sweep `top_min` (both polarities), sort direction, and the fidelity/format axes the
   wrappers template on.
5. Once green, restore
   `tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/llk_math_deepseek_top32_rm.h`
   in the same PR, so the wrappers land with their caller.

**Watch for.** `_top32_rm_init_()` and `_topk_xl_init_<K, fused>()` **cannot both be called
in one kernel** — they hang the math thread on overlapping ADDR_MOD slots, the MOP and the
REPLAY buffer (Finding 3). Keep the two families in separate kernels. And see C3: a
pre-existing reconfig escape already exists in this area, so a failure in a combined run may
not be yours.

---

### A3. `set_dst_write_addr_offset` — behaviour never asserted

**Evidence.** The helper that the whole #52713 extraction exists for is referenced by exactly
one driver, `sort_headers_coexist_test.cpp`, and review established that its offset is
**unobservable there**: `_llk_math_eltwise_unary_datacopy_` calls
`math::set_dst_write_addr<Tile32x32, SrcRegs>(dst_index)` — the same
`DEST_TARGET_REG_CFG_MATH_Offset_ADDR32` the helper writes — before anything touches DEST.
The offset sweep that used to imply coverage was removed for exactly this reason.

So today we prove the two headers *coexist in one translation unit*, and prove nothing about
what the helper does. This gap was **created by the review** (an honest correction of a test
that overclaimed) and is not yet tracked anywhere else.

**Plan.**

1. Reach DEST through a consumer that does **not** reprogram the offset register first. The
   natural candidate is an SFPU store: `_llk_math_eltwise_unary_sfpu_params_` calls
   `_llk_math_eltwise_sfpu_start_` → `set_dst_write_addr(dst_index)`, so it has the same
   problem — the offset must be applied **inside** the sfpu lambda, after `start_` has run,
   and then the SFPSTORE lands at the rebased address.
2. Assert positionally, not numerically: write a known pattern at offset `N`, then pack and
   check the data appears `N` Dst rows from the tile base. A value-only check cannot
   distinguish a correct rebase from no rebase at all — that is precisely the trap the
   original version fell into.
3. Sweep the offsets the two real callers use: `2` (topk_xl's column-group flip) and `64`
   (deepseek_top32_rm's whole-tile rebase), plus `0` as the no-op baseline.
4. Add a negative variant with `addr >= DEST_REGISTER_HALF_SIZE` to exercise the
   `LLK_ASSERT` in the helper, under `ENABLE_LLK_ASSERT`, if the harness supports
   expected-assert tests.

**Size.** Small — half a day. This is the best value-per-hour item on the list, and it is
not blocked on any PR.

---

### A4. `mul_reduce_scalar_chunked_tile` — untested and shipping

Published in `HW_JIT_API_HEADERS`, no in-tree caller, no test. Attempted twice and reverted;
results came back 5–30x golden. **The driver was deleted before committing, so it is not in
git history and must be rewritten** from the notes.

**Do not restart this from scratch.** Read `blaze_llk_promotion_test_strategy.md` §3 first —
it records the scaffolding worth recovering (the `CHUNKED_REDUCE` parameter, the file-scope
`params.h` requirement, the SFPU accumulator fold, the batch-boundary tile sweep), what has
already been ruled out, and the localisation work that narrowed the bug to **inside a single
batch's reduce** rather than the cross-batch accumulation. The prime remaining suspect is
that `_llk_math_mul_reduce_scalar_init_` is not re-enterable — which is exactly the property
the per-batch re-init depends on.

**Plan.** Rebuild the driver from §3, then test the re-enterability hypothesis directly and
in isolation: call `_llk_math_mul_reduce_scalar_init_` twice in one kernel with a single
batch and check whether the second reduce matches the first. That is a much smaller
experiment than the full chunked driver and it either confirms or kills the leading theory
before any more effort goes into the sweep.

**Note.** This is also review comment D1 below — the reviewer independently reached
"don't ship this function until it has a test". If the test confirms a defect, removing the
function from `main` is a legitimate outcome and probably the faster one.

---

### A5. `eltwise_mul_scalar` HiFi init — untested, and its rationale does not hold

Smaller than it looks: the underlying shapes **are** covered generically —
`test_eltwise_binary.py` sweeps `DEST_TO_SRCA`/`DEST_TO_SRCB` dest-reuse and
`BroadcastType.Scalar`. What has no test is the **HiFi init sequence** specifically.

It is also the one item sitting on a disproved rationale — see C2. Resolve C2 first: if the
workaround's real mechanism turns out to be something else, the test to write changes
completely, and if the workaround is inert the honest outcome may be deleting it rather than
testing it.

**Plan.** Blocked on C2. Once the actual failing configuration is known, reproduce *that*
configuration at the LLK level rather than testing the workaround's stated intent. Read §9
for the earlier attempt, which hung the device as first written.

---

### A6. Thin spots inside tests that already pass

Not holes, but places where a passing suite implies more than it covers.

| Test | Gap | Fix |
|---|---|---|
| `test_rmsnorm_bcast_scalar_dest_reuse.py` | `num_faces` 1 and 2 only at `num_tiles=1`, LoFi, `clear_dest=True` | widen the partial-faces test to `num_tiles ∈ {1,2}` and both ops at HiFi |
| same | transpose-fold path only at LoFi | add HiFi2/HiFi4 — the transpose interacts with the fidelity phase loop and nothing checks that |
| same | MOP's ELWSUB branch untested | **leave it.** Deliberate: no compute-API function instantiates ELWSUB on this family, and speculative coverage of an uninstantiated branch is how tests rot |
| all families | **HiFi3 is never swept anywhere** | add it to one representative ELWMUL sweep; the fidelity phase count is 3 there and no test exercises it |
| `test_custom_mm_uninit_restore.py` | 15 variants after the duplicate axis came out | fine as-is — the drop was removing a no-op, not losing coverage |

**Size.** About a day for the whole row set. Low risk, no new drivers.

---

## B. Gap that cannot live in tt-llk

### B1. Nothing can catch `custom_mm` vs `compressed_custom_mm` divergence

`custom_mm_uninit_restore_test.cpp` **replicates** the uninit body rather than calling
`custom_mm_block_uninit` / `compressed_custom_mm_block_uninit`, because a tt-llk test cannot
include `tt_metal/hw/inc/api/compute`. The two headers currently have identical uninit
bodies; if they diverge, every existing test keeps passing.

**Plan.** This needs a test on the **metal** side that calls the real compute-API entry
points — a small compute-kernel test under `tests/tt_metal/`, not a tt-llk driver. Scope is
about a day, but it needs an owner who works in that tree; it is outside the tt-llk test
remit entirely.

**Cheaper interim option.** A `static_assert`-style guard, or a pre-commit check, that fails
if the two uninit bodies stop matching textually. Ugly, but it closes the specific risk in an
hour and does not need a new test tree.

---

## C. Product issues needing a decision, not a test

### C1. `dense_packing` W-stride is not format-aware — **defect**

`set_packer_strides` (`cpack_common.h:301-305`) derives the field as
`TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * datum_size_in_bytes(pack_src_format)`, while
`custom_mm.h:69` / `:261` and `compressed_custom_mm.h:69` / `:262` hardcode `* 2`. On a
Float32 pack source both halves are 2x off: init programs 1024 where 2048 is correct, and the
uninit restores 2048 where 4096 is correct. Measured at 0.25 match, recorded as an `xfail` in
`test_custom_mm_uninit_restore.py`, and flagged on the PR.

**Owner:** whoever owns `custom_mm.h`. **Decision needed**, then ~half a day:

- **Option 1 — guard.** `LLK_ASSERT` in `*_block_init` that
  `datum_size_in_bytes(pack_src_format[out_cb]) == 2` when `dense_packing` is set. No API
  change; turns silent corruption into a loud failure; leaves 32-bit unsupported.
- **Option 2 — full fix.** Derive the datum size in init from `out_cb_id` and **add an
  `out_cb_id` parameter to `*_block_uninit`**, which currently takes none. Correct on 32-bit,
  but changes a signature that `matmul.hpp`, `flash_mla.hpp`, `dram_streaming_matmul*.hpp`
  and `matmul_custom_compressed_kernel.cpp` all call.

The `xfail` flips to XPASS the moment either lands, so the test tells you when it is fixed.

### C2. The `eltwise_mul_scalar` HiFi workaround's mechanism does not survive review

`deepseek_binary_dest_reuse_tiles_init`'s HiFi branch hardcodes `DEFAULT_TENSOR_SHAPE` and
attributes a HiFi4 fix to the shorthand init "mis-specialising the tile shape". Reading the
code it calls: `get_effective_math_fidelity<ELWMUL, f>()` is the **identity** for ELWMUL,
`acc_to_dest` is 0 in both arms, and the shorthand resolves the shape from the CB regardless
of fidelity. So tensor shape is the only difference — and on a standard 4-face CB
`get_operand_tensor_shape` returns exactly `DEFAULT_TENSOR_SHAPE`, making the HiFi arm
bit-identical to the shorthand it replaces. Meanwhile the paired execute *does* derive the
shape from the CB, so on non-default geometry init and execute disagree.

Combined with the measured fact that forcing `DEFAULT_TENSOR_SHAPE` on a 2-face tile
**deadlocks the MATH_PACK handshake**, the workaround is either inert (4-face CB) or hangs
(2-face CB). There is no configuration where it does what its comment says.

The cited failing config is `gated_local_reduce` at HiFi4 (0.70 → 0.9996), and that
measurement is not explained by the stated mechanism.

**Owner:** the #52709 author. **Needed:** either the real mechanism, or a corrected comment.
A5 is blocked on this.

### C3. Pre-existing `topk_xl` → `eltwise_binary` reconfig escape — **defect**

```
pytest test_eltwise_binary.py   (alone)  -> 4388 passed, 72 skipped
pytest test_topk_xl.py          (alone)  ->   71 passed
pytest test_topk_xl.py <target>          ->    1 failed
```

`<target>` being
`test_eltwise_binary[dest_acc:No-formats:Bfp4_b->Bfp4_b-broadcast_type:None_-math_op:Elwmul-math_fidelity:LoFi-transpose_srca:Yes-input_dimensions:[256, 32]-tile_dimensions:[32, 32]]`.

**Unrelated to the promotions** — reproduces on clean `main` with every promotion change
stashed. Recorded here because A2 (`top32_rm`) shares the sort headers, so whoever picks that
up will see a failure in this area and assume it is theirs. **Bisect single-file-then-target
before blaming your own driver.**

Per the tt-llk notes a reconfig escape is a real bug rather than a test-ordering nuisance, and
`tt-smi -r` must **not** be used to paper over it. Needs an owner.

---

## D. Review comments resolved but not fixed

All three landed in `main` via #52709 before they could be acted on, so the PR threads were
answered and resolved rather than reverting merged API. They are open work against `main`.

- **D1 — `mul_reduce_scalar_chunked_tile` ships with no caller and no test.** Same subject as
  A4. If A4 confirms a defect, removal is a legitimate outcome.
- **D2 — bare `false, false` in `rmsnorm.h:27`.** Should read
  `false /*transpose_of_faces*/, false /*within_face_16x16_transpose*/, icb0`, matching both
  the deleted vendored source and `rmsnorm_bcast_scalar_reuse_tiles_init_fidelity` a few
  lines below in the same file. One line.
- **D3 — `restore_tile_pack_mop` is end-of-call-cleanup with no consumer.** Kept deliberately:
  it defaults to `false`, nothing in the tree opts in, and moving the family to a
  clean-state-on-entry contract is an API change that did not belong in a test PR. Both
  polarities and the inert-at-matching-geometry case are pinned, which is what a contract
  change will need. Revisit when someone owns `custom_mm.h`.

---

## E. PR mechanics

- **The title still reads `[do not review]`.** The PR is no longer a draft and shows
  `REVIEW_REQUIRED`, so this is the single thing blocking anyone from looking at it.
- **Rebase again once #52713 and #52727 merge.** The branch still carries their promotion
  payload — 32 files, byte-identical to `pmilenkovic/promote-top32-rm` and
  `pmilenkovic/promote-custom-mm`. They drop out cleanly, as #52709's did. Expect one
  conflict in `tt_metal/hw/sources.cmake`, which is where all three promotions add entries.
- `backup/llk-tests-pre-rebase` is a local-only safety ref from the first rebase; delete it
  once you are satisfied.

---

## Explicitly out of scope

**Perf tests.** There is no perf coverage for any promoted family (nor for `topk_xl` or
`sampling`), and 56 functional test modules have no perf counterpart. This was reviewed and
**deliberately ruled out** — recorded here so it is not re-raised as an oversight. The perf
infrastructure itself is ready if that ever changes: discovery is marker-driven with
pytest-split sharding, `PerfRunType` already provides the isolation modes, and no registry
edit is needed to onboard a new op. The two things that *would* need doing first are wiring
`compare_test_and_perf.py` into CI (it exists, runs nowhere) and fixing its filename-based
pairing, which reports real pairs as unmatched.

---

## Suggested order

1. **E** — retitle the PR. Minutes, and nothing else gets reviewed until it is done.
2. **A3** — half a day, unblocked, and closes a gap the review itself opened.
3. **C1 / C2 / C3** — route to owners now; they are decisions, and they gate A5 and colour A2.
4. **A6** — a day, low risk, tightens what already passes.
5. **A1** when #52727 merges, **A2** when #52713 merges. A2 is the bigger job and also
   restores the dropped wrappers.
6. **A4** via the small re-enterability experiment first, not the full driver.
7. **B1** once an owner in the metal tree exists.
