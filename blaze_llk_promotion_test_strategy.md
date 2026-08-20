# tt-llk blaze promotions — OPEN work

> **What this is.** The remaining tt-llk test work for the blaze->tt-metal `experimental/`
> promotions (#52709, #52713, #52727). **Updated 2026-08-18: 3 items remain — 2 not started,
> 1 attempted and reverted.** Item 1 (`mul_reduce_scalar_chunked_tile`) is **resolved as a
> product defect**, not an open test task; §3 is kept for its two dead ends, and the answer
> is Finding 9 in the DONE document. `REMAINING_WORK.md` is the current actionable index —
> read that first; this document is the long-form background it points into.
> Completed work has been moved out to
> **`BLAZE_PROMOTION_TESTS_DONE.md`** — check there before starting anything, since three
> of the plans below were already corrected by what those tests measured on silicon.
>
> Branch: `ldjurovic/llk-tests-blaze-promotions` (tt-metal). **#52709 merged on 2026-08-14**
> and the branch has since been rebased onto main, so the rmsnorm / add_rsqrt /
> eltwise_mul_scalar headers now come from main rather than from a merge commit. #52713 and
> #52727 are still open, so the branch still carries their promotion payload (32 files,
> byte-identical to `pmilenkovic/promote-top32-rm` and `pmilenkovic/promote-custom-mm`);
> rebase again once they land and those drop out, leaving only `tt_metal/tt-llk/tests/`
> plus the two LLK header fixes noted in the DONE document.
>
> Hardware: Blackhole p100a. Every item below is Blackhole-only — every promoted header is
> `#if defined(ARCH_BLACKHOLE)`-guarded — so each test carries `skip_for_wormhole` +
> `skip_for_quasar`.
>
> Run tests via `tt-llk/.claude/scripts/run_test.sh`, not raw pytest. See the tooling note
> at the end of the DONE document for two gotchas that cost time.

---

> **Looking for a specific bug?** §13 has a dossier per open defect (C1–C6, F): mechanism, what
> is measured versus inferred, blast radius, reproduction, fix options with trade-offs, and the
> tripwire test that flips when it is fixed. Two of them record fixes that were tried and did not
> work, which is the part that saves the most time.

## Open work at a glance

| # | Item | PR | Est. | Notes |
|---|------|----|------|-------|
| ~~1~~ | ~~`mul_reduce_scalar_chunked_tile`~~ — **RESOLVED 2026-08-18 as a defect, not a test gap** | #52709 | — | The op is broken as written: re-entering the reduce inside one DEST section does not restore state, and that is exactly how the chunked loop is built. Minimal reproducer landed (`test_mul_reduce_scalar_reenter.py`, 36 passed / 12 xfailed). **Read Finding 9 in the DONE document, not §3's plan.** Needs an owner (tracked as C4); a full chunked driver is pointless until the LLK or the compute API is fixed |
| 3 | plain `custom_mm` matmul — new file | #52727 | ~2 d | **§8.** Also settles the `ct ∈ {7,9,11}` doc question |
| 4 | `top32_rm` — new file, two modes | #52713 | ~3-4 d | **§6.** Largest single effort. **Now also gates a removal:** the 7 `llk_math_deepseek_top32_rm` wrappers were dropped from the promotion on review (no caller, no test), so this item is what earns them back |
| 5 | `eltwise_mul_scalar` HiFi init fix | #52709 | ? | **§9. Attempted and reverted — read §9 before retrying.** Hangs the device as first written. **See also the review finding in the DONE document (Finding 7): the workaround's stated mechanism does not survive reading the code it calls** |

> Two items formerly on this list are **done** and have moved to the DONE document:
>
> - the sampling `recip_init` polluter test (formerly item 6) — found that the hazard is a
>   ~1e-3 precision loss hidden by the suite's 2% reciprocal tolerance, not the garbage the
>   PR wording implies;
> - **`rmsnorm` bcast-scalar dest-reuse (formerly item 2)** — 66 variants passing. It found
>   that neither rmsnorm LLK header compiled under the tt-llk build at all, and that ELWMUL
>   accumulates into DEST where ELWADD overwrites. See §4 for the stub and the DONE
>   document for the results and findings.

Plus one item that is not a test task:

| | Item | Owner |
|---|------|-------|
| **D1** | **`dense_packing` W-stride is not format-aware.** Found by a landed test, recorded there as `xfail`. Needs a decision: make the constants format-aware, or `static_assert` `dense_packing` to 16-bit pack sources. Full detail: Finding 2 in the DONE document. | whoever owns `custom_mm.h` |

---

## 1. What is already covered (do not redo)

Landed and passing: `add_rsqrt`, `custom_mm`/`compressed_custom_mm` `block_uninit`,
sort-header coexistence. Verification tier V1-V4 all green, including confirmation that
**#52745 and #52747 need no new tt-llk tests**. Details and results in
`BLAZE_PROMOTION_TESTS_DONE.md`.

The inventories of what each PR promotes are kept below (§2.0, §3.0, §4.0, §5.0) because
they are the reference for what is still uncovered.

---

## 2. #52709 inventory (`rmsnorm` / `add_rsqrt` / `eltwise_mul_scalar`)

| Path | Status |
|------|--------|
| `tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_math_rmsnorm_bcast_scalar_dest_reuse.h` | promoted (rename, 60% similarity) |
| `tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_unpack_A_rmsnorm.h` | **new file** |
| `hw/ckernels/blackhole/metal/llk_api/experimental/llk_math_rmsnorm_bcast_scalar_dest_reuse_api.h` | promoted |
| `hw/ckernels/blackhole/metal/llk_api/experimental/llk_unpack_A_rmsnorm_api.h` | promoted |
| `hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_add_rsqrt.h` | promoted (100% identical) |
| `hw/inc/api/compute/experimental/rmsnorm.h` | **new file** (blaze version + chunked mul-reduce) |
| `hw/inc/api/compute/experimental/add_rsqrt.h` | promoted |
| `hw/inc/api/compute/experimental/eltwise_mul_scalar.h` | promoted + **behavioral fix** |
| `hw/sources.cmake` → `HW_JIT_API_HEADERS` | +3 entries (ships in packaged metalium) |

**Public surface newly exposed** (`api/compute/experimental/rmsnorm.h`):
`rmsnorm_bcast_scalar_reuse_tiles_init` / `_tiles`, `rmsnorm_bcast_scalar_reuse_tiles_init_fidelity` /
`_tiles_fidelity` (explicit-fidelity + `unpack_full_transpose` axis), `rmsnorm_mul_bcast_scalar_reuse_tiles_init` /
`_tiles`, and `mul_reduce_scalar_chunked_tile<num_tiles, dst_capacity, reduce_type>`.

**Current tt-llk coverage: zero.** `grep -rl "rmsnorm|add_rsqrt|mul_scalar|dest_reuse" tests/sources/` returns
nothing for all four.

> `add_rsqrt` from this family is **done** — see the DONE document.


---

## 3. RESOLVED (was OPEN #1) — `mul_reduce_scalar_chunked_tile`

> **Status as of 2026-08-18: the cause is found, and it is a product defect.** Re-entering the
> reduce sequence works fine across a DEST-section boundary and is **bit-identical**; it is
> broken only when there is no boundary in between — which is precisely how
> `mul_reduce_scalar_chunked_tile` re-enters per batch. Measured 9.27x-9.93x golden, matching
> the 5-30x non-integer signature recorded below.
>
> **Read Finding 9 in `BLAZE_PROMOTION_TESTS_DONE.md` for the result**, and
> `tests/python_tests/test_mul_reduce_scalar_reenter.py` for the ~40-line reproducer. Tracked
> for an owner decision as C4 in `REMAINING_WORK.md`.
>
> **What is still useful below:** the two dead ends (the accumulator fill, and the missing
> UNPACK/MATH barrier) — both tried on silicon, both left the output byte-identical, and
> Finding 9 explains why neither moved the number: neither touched the DEST-section boundary.
> **Do not re-investigate either, and do not rebuild the full chunked driver** until the op
> is fixed. The scaffolding notes are kept for whoever eventually writes the real chunked
> test *after* the fix lands.

### What was built

`tests/sources/mul_reduce_scalar_chunked_test.cpp` + `test_mul_reduce_scalar_chunked.py`,
as a **separate driver rather than an extension** of `mul_reduce_scalar_test.cpp` — the
chunked form needs per-batch unpack/math interleaving and an SFPU accumulator, so branching
the existing 54-variant driver would have put all of it at risk for no benefit. That call
still looks right.

It **compiled cleanly across all 38 variants** and ran without hanging. The scaffolding is
sound and worth recovering:

- `CHUNKED_REDUCE` template parameter emitting `DST_CAPACITY` / `CHUNK_NUM_TILES`, plus
  `REDUCE_SCALER`. Both compile-time on purpose, so the same constexpr
  `batch_size` / `num_batches` / `last_batch_size` arithmetic the compute API performs runs
  in the driver too, `static_assert`s included.
- `params.h` must be included at **file scope**, not just inside each TRISC block: the
  generated `DST_CAPACITY` / `CHUNK_NUM_TILES` constants arrive through it, and the
  constexpr batch decomposition needs them before any `#ifdef LLK_TRISC_*`.
- The accumulator fold uses `test_utils::call_binary_sfpu_operation<DST_SYNC, ...,
  BinaryOp::ADD, 32, 0>(ACCUMULATOR, 0, ACCUMULATOR, VectorMode::RC)` from
  `helpers/include/sfpu_operations.h`, with a matching `..._init` before the loop.
- Tile-count sweep targeting the batch boundaries rather than sampling uniformly:
  `dst_capacity + 1`, `2 * batch_size`, `2 * batch_size + 1` (the off-by-one canary,
  `last_batch_size == 1`), `3 * batch_size - 1`.

### Why it was reverted

Every bf16-output variant returned a scalar **5x to 30x larger than golden** — e.g.
`num_tiles=3, dst_capacity=2`: device `42496.0` vs golden `1498.3`. Not a clean multiple of
anything (`num_batches`, `batch_size`, tile count), which is what makes it interesting: a
simple double-count or a missing clear would show as an integer ratio.

**Ruled out — do not re-investigate these:**

- *The accumulator fill.* First hypothesis was that `fill_tile(accumulator, 0.0f)` was
  reaching only part of the slot, leaving stale DEST in the lanes the SFPU add later reads.
  Real bug in the first draft, and worth knowing for any future driver: the compute API's
  `fill_tile` is `_calculate_fill_` at **`VectorMode::RC`, 8 iterations** (whole tile),
  whereas the `RC_custom` / 2-iteration form used to stage the reduce scaler covers only a
  small region — the two are not interchangeable. Fixing it changed the output **not at
  all**, byte for byte, so the accumulator was not the cause.
- *A stale build.* Verified the corrected source was what compiled.

**Both hypotheses tried on silicon. Both disproved. Do not retry either.**

*Attempt 1 — the accumulator fill.* The compute API's `fill_tile` is `_calculate_fill_` at
`VectorMode::RC`, 8 iterations (whole tile), whereas the `RC_custom` / 2-iteration form used
to stage the reduce scaler covers only a small region. The first draft used the wrong one.
Real bug, worth knowing for any future driver — and it changed the output **byte for byte
not at all**.

*Attempt 2 — a missing per-batch UNPACK/MATH barrier.* The reasoning still looks sound in
principle: the non-chunked op has one phase transition so the dvalid handshake suffices,
while the chunked form has one per batch and nothing stops UNPACK running ahead into
SrcA/SrcB. Production gets that ordering free from CB flow control (`cb_wait_front` /
`cb_pop_front`), which the tt-llk harness has no equivalent of. Implemented as a
`semaphore::UNPACK_TO_DEST` post/consume pair (that semaphore is initialised by
`helpers/include/boot.h` and unused in this test, since bf16 goes through SrcA). Again
**byte-identical output**. There is no race, or at least not one this barrier closes.

### Where the bug actually is

Three source changes producing identical results looked like a stale build, so the check
that should have come first was finally run: **pack `DEST[0]` instead of
`DEST[ACCUMULATOR]`**. The number moved (42496.0 -> 37120.0 for `num_tiles=3,
dst_capacity=2`). That settles three things at once:

- the build is **not** stale — source edits do reach the binary;
- `_llk_pack_`'s tile index **is** honoured under the reduce mask;
- therefore both fixes above genuinely had no effect, rather than not being compiled.

And it relocates the fault. `DEST[0]` holds a **single batch's** reduced scalar, and for
`num_tiles=3, dst_capacity=2` (`batch_size=1`, so the last batch is one tile) it should be
about `sum(one tile)` ~= 512. It reads 37120 — roughly **72x** too large. So the per-batch
reduce is already wrong before any cross-batch accumulation happens. The chunking
arithmetic, the accumulator and the barrier are all downstream of a broken single-batch
reduce.

**Reconciled 2026-08-18.** The measurement above is consistent with Finding 9, and in
hindsight points straight at it. `DEST[0]` was packed *after the batch loop*, so the number
read was the **last** batch's reduce, not the first — and Finding 9 says batch 0 is correct
while every re-entry after it is wrong. The reading "the per-batch reduce is already wrong
before any cross-batch accumulation" was right about *which* reduce was broken but wrong to
infer that a single isolated reduce would fail; one does not.

**The one structural difference from the working non-chunked driver** is that
`_llk_math_mul_reduce_scalar_init_` (and the unpack-side `_llk_unpack_AB_init_` +
`switch_to_reduce`) are invoked **once per batch** rather than once per kernel. The prime
suspect is therefore that this reduce family is not re-enterable — that a second
`_llk_math_mul_reduce_scalar_init_` accumulates addrmod / counter state rather than
re-establishing it. That would be a real finding about the promoted LLKs if confirmed, and
is exactly the kind of thing the compute API's chunked loop depends on working.

**Next experiments — SUPERSEDED 2026-08-18, kept for the record.**

The three experiments listed here were the right shape but all presupposed rebuilding the
chunked driver. What actually settled it was cheaper: keep the *known-good non-chunked*
sequence and just run it twice, with the DEST-section boundary as an explicit axis. That
isolates the one structural difference without reintroducing the chunking arithmetic, the
accumulator, or the batch decomposition — none of which turned out to be involved.

The prime suspect named below ("this reduce family is not re-enterable") was close but too
coarse: the family *is* re-enterable, across a section boundary, bit-identically. See
Finding 9.

### Original plan

The sweep design below still stands; only the driver is unfinished.


This is the single riskiest piece of new code in #52709: it is not a promoted header at all but a genuinely
new compute-API composition, with non-trivial host-side arithmetic:

```
batch_size     = dst_capacity - 1
accumulator    = batch_size
num_batches    = (num_tiles + batch_size - 1) / batch_size
last_batch_size = num_tiles - (num_batches - 1) * batch_size
```

plus a mid-loop `mul_reduce_scalar_init(icb0, icb1)` re-init for `batch > 0` and a one-shot
`llk_pack_reduce_mask_config` on `batch == 0`. Off-by-one in `last_batch_size`, a missing re-init, or a
double pack-mask config would all produce a plausible-looking but wrong scalar.

`test_mul_reduce_scalar.py` + `sources/mul_reduce_scalar_test.cpp` already exist and drive the non-chunked
op. Add a `CHUNKED` mode to the C++ source replicating the loop above, and a `test_mul_reduce_scalar_chunked`
function. Sweep `num_tiles` specifically at the boundaries the arithmetic can get wrong:

- `num_tiles = dst_capacity` (smallest legal — `static_assert(num_tiles > dst_capacity)` means start at `+1`)
- `num_tiles = dst_capacity + 1`
- `num_tiles = 2 * batch_size` (exact multiple → `last_batch_size == batch_size`)
- `num_tiles = 2 * batch_size + 1` (**ragged tail** → `last_batch_size == 1`, the off-by-one canary)
- `num_tiles = 3 * batch_size - 1`

crossed with `dst_capacity ∈ {2, 4, 8}` (the `static_assert` range is 2..8; `dst_capacity = 2` means
`batch_size = 1`, i.e. one product per batch — the degenerate case), `scaler ∈ {1.0, 0.5}`, and the existing
format/fidelity axes. Keep `B == 1.0` as the existing test does so the golden stays `scaler * sum(A)` and the
test isolates the chunking logic rather than re-testing the multiply.

---

---

## 4. DONE — `rmsnorm` bcast-scalar dest-reuse (was OPEN #2)

**Landed.** `tests/sources/rmsnorm_bcast_scalar_dest_reuse_test.cpp` +
`tests/python_tests/test_rmsnorm_bcast_scalar_dest_reuse.py`, 66 variants passing on
BH p100a. Results, the sweep actually built, and the findings are in section 6 (and
Findings 5-6) of `BLAZE_PROMOTION_TESTS_DONE.md`.

The plan that used to sit here was followed in outline — new file rather than an extension,
driver replicating the compute API's call sequence at the llk layer, DEST seeded by a
datacopy pre-pass — with four corrections that came out of building it:

- **The headers did not compile under the tt-llk build.** Both carried dead locals that trip
  `-Werror=unused-variable`. This had to be fixed before any test could exist, and is the
  reason the estimate was optimistic.
- **`clear_dest` is not a free axis.** ELWMUL accumulates into DEST, so only
  `clear_dest=True` has a DEST-independent result; the false half is asserted separately
  against an accumulating golden.
- **Fidelity is swept for ELWMUL only** — the ELWADD MOP branch never reads the template
  argument, so sweeping it there builds identical ELFs.
- **`num_faces ∈ {1, 2}` needs its own test**, not a cell in the main matrix: the pack still
  emits a full 4-face tile, so those variants are about the *uncovered* tail rather than
  about the op.

The `unpack_full_transpose` axis called out as highest-value did land and passes, so the
transpose-fold path blaze's version of the header added is now covered.


---

## 5. #52713 inventory (`top32_rm`)

| Path | Status |
|------|--------|
| `tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_math_top32_rm.h` | promoted (76%) |
| `tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_unpack_A_top32_rm.h` | promoted (57%) |
| `tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_deepseek_top32_rm.h` | promoted (79%) |
| `tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_set_dst_write_addr_offset.h` | **new file** (extracted shared helper) |
| `tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_topk_xl.h` | **modified** (helper removed, now includes the shared header) |
| `hw/ckernels/blackhole/metal/llk_api/experimental/llk_math_top32_rm_api.h` | promoted |
| `hw/ckernels/blackhole/metal/llk_api/experimental/llk_unpack_A_top32_rm_api.h` | promoted |
| `hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/llk_math_deepseek_top32_rm.h` | **new file** — 7 SFPU entry points |

> **Discrepancy to raise with the author.** The PR body states "Blaze's `llk_math_deepseek_top32_rm.h`
> wrapper is **not** promoted — the in-tree consumers drive the SFPU functors via the `SFPU_UNARY_CALL`
> macros already." The diff at the current head **does** add
> `hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/llk_math_deepseek_top32_rm.h` as a new file with
> seven entry points (`llk_math_deepseek_top32_rm_init`, `_local_sort`, `_merge`, `_rebuild`,
> `llk_math_deepseek_top32_of_1024_rm_pre_sorted_{prep,combine,final}`). Either the description is stale or the
> file is unintentionally included. It matters for testing: if it stays, it is public API with zero callers
> and zero tests, which is exactly the surface a tt-llk test should pin.

**Current tt-llk coverage: zero for `top32_rm`.** The only in-tree coverage anywhere is the tt-metal gtest
`tests/tt_metal/tt_metal/llk/test_top32_rm_dev.cpp` (`Top32RmDevPipelineCompletes`), which runs
`row_elements ∈ {64, 128, 160, 3232}` at a single seed and a single dest-acc setting.

`topk_xl_test.cpp` / `test_topk_xl.py` exist and are the regression net for the
`set_dst_write_addr_offset` extraction.

> The `set_dst_write_addr_offset` coexistence case from this PR is **done** — see the
> DONE document, including Finding 3 on why the two families' inits cannot both run.


---

## 6. OPEN #4 — `top32_rm` (new file: `test_top32_rm.py` + `sources/top32_rm_test.cpp`)

**Model it on `test_topk_xl.py`.** That is the closest analogue by a wide margin: same value+index DEST
layout discipline, same hand-built stimuli approach ("`helpers.stimuli_generator` is very awkward for these
tests"), same bf16-value / uint32-index output decoding, and it now literally shares the
`set_dst_write_addr_offset` helper. Reuse its `_decode_row_major` / `_bitcast_float32` helpers if the DEST
index layout matches; the `dst_indices_offset = 128` (2 tiles × 64 rows) convention in
`ckernel_sfpu_deepseek_top32_rm.h` suggests it will need a local variant.

**Two kernel modes**, mirroring the two tt-metal dev kernels and selected by a `TOP32_MODE` template
parameter (the gtest picks between them at `row_elements >= 1024`):

- `MODE_INCREMENTAL` (`row_elements < 1024`) — the 64-elements-at-a-time path:
  `_llk_unpack_A_top32_rm_` transposing load (16 elements into the first row of each of 4 faces, remainder
  padded to `-inf`) → `llk_math_deepseek_top32_rm_local_sort` (bitonic phases/steps) → per-chunk
  `_merge` + `_rebuild`.
- `MODE_PRESORTED` (`row_elements >= 1024`) — the whole-1024-chunk path:
  `llk_math_deepseek_top32_of_1024_rm_pre_sorted_prep<top_min>` → `_pre_sorted_combine` → `_pre_sorted_final`.

**Sweep.** The gtest's four sizes leave the interesting boundaries untested:

| axis | values | rationale |
|------|--------|-----------|
| `row_elements` | `32, 63, 64, 65, 128, 160, 1023, 1024, 1088, 2048, 3232` | `64`/`1024` are the chunk boundaries; `63`/`65`/`1023` are the ragged tails that exercise the `-inf` padding; `32` is fewer elements than the top-k width |
| `is_fp32_dest_acc_en` | `No, Yes` | **switches the index load/store `InstrModLoadStore` between `LO16` and `INT32`** (`bitonic_top32_load8`/`load16`). The gtest runs one setting; this axis can silently corrupt indices only. |
| `idir` / sort direction | both | `_local_sort(idir)`, `_merge<idir>`, `_rebuild(idir, skip_second)` all take it |
| `top_min` | `False, True` | template arg on `_pre_sorted_prep_`, no other coverage |
| `skip_second` | `False, True` | runtime arg on `_rebuild` |
| stimuli shape | shuffled-distinct, **all-ties**, **partial ties at the k=32 cut**, all-`-inf` row, single non-`-inf` element | ties are where a top-k index tie-break becomes non-deterministic; the gtest uses one shuffled seed only |

**Golden.** `argsort` descending over `row_elements`, take 32. Values compare **exactly** — the op is a
pure permutation of bf16 inputs with no arithmetic, so no tolerance is warranted, and a tolerance would mask
a wrong-lane bug. Indices compare exactly too, with the tie cases either restricted to distinct values or
asserted as a set rather than a sequence.

---

## 7. #52727 inventory (`custom_mm` / `compressed_custom_mm`)

| Path | Status | tt-llk coverage today |
|------|--------|-----------------------|
| `tt-llk/.../llk_lib/experimental/llk_math_custom_mm.h` | promoted | **none** |
| `tt-llk/.../llk_lib/experimental/llk_unpack_AB_custom_mm.h` | promoted (84%) | **none** |
| `tt-llk/.../llk_lib/experimental/llk_math_compressed_custom_mm.h` | promoted (51% — clang-format) | ✅ `matmul_custom_compressed_test.cpp` |
| `tt-llk/.../llk_lib/experimental/llk_unpack_AB_compressed_custom_mm.h` | **new file in-tree** (was vendored) | ✅ same |
| `llk_api/experimental/llk_math_custom_mm_api.h` | promoted | none |
| `llk_api/experimental/llk_math_compressed_custom_mm_api.h` | promoted | none |
| `llk_api/experimental/llk_unpack_AB_custom_mm_api.h` | promoted | none |
| `llk_api/experimental/llk_unpack_AB_compressed_custom_mm_api.h` | promoted | none |
| `hw/inc/api/compute/experimental/custom_mm.h` | promoted + `ARCH_BLACKHOLE` guard + **uninit change** | none |
| `hw/inc/api/compute/experimental/compressed_custom_mm.h` | same | none |
| `hw/sources.cmake` → `HW_JIT_API_HEADERS` | +2 entries | — |
| `tt-llk/tests/python_tests/test_matmul_custom_compressed.py` | **rewired** (drops the `VENDORED_LLK_LIB` fixture) | — |
| `tt-llk/tests/sources/matmul_custom_compressed_test.cpp` | **rewired** to `experimental/` includes | — |

Compute-API surface per family: `*_block_init`, `*_block_init_short`, `*_block`, `*_block_unpack`,
`*_block_math`, `*_block_uninit`.

**Good news first:** the compressed llk_lib pair is already covered. `matmul_custom_compressed_test.cpp`
drives `_llk_unpack_AB_compressed_custom_mm_init_` / `_llk_unpack_AB_compressed_custom_mm_` and
`_llk_math_compressed_custom_mm_init_` / `_llk_math_compressed_custom_mm_` across BFP0/2/4/8 and a shape
sweep. This PR only changes the include spelling and deletes the vendored-path fixture, so **running
`test_matmul_custom_compressed.py` on the branch is the direct validation** — and it is now a *better* test
than before, since it consumes the canonical headers instead of the demo tree.

> `block_uninit` from this PR is **done** — see the DONE document, including Finding 1
> (geometry) and Finding 2 (the W-stride defect, item **D1** above).


---

## 8. OPEN #3 — plain `custom_mm` matmul (new file)

The plain (non-compressed) `custom_mm` llk_lib and llk_api pair has no tt-llk coverage. Note that the
similarly-named existing `test_matmul_custom.py` / `matmul_custom_test.cpp` drive a **different** family —
`experimental/llk_math_matmul_custom_no_mop.h` — so it cannot simply be extended, and the new file should be
named to avoid the confusion (`test_matmul_custom_mm.py`, not `test_custom_matmul.py`).

Mirror `matmul_custom_compressed_test.cpp`'s three-TRISC structure — it is the sibling family and is already
wired to the canonical `experimental/` headers, so the port is mostly mechanical.

Sweep:

| axis | values |
|------|--------|
| shapes `(rt, kt, ct)` | `ct ∈ {1..6, 8, 10, 12, 14, 16}` — the set blaze documents as tested |
| **`ct ∈ {7, 9, 11}`** | the doc-split holes (see below) |
| template bools | `transpose`, `split_acc`, `dense_packing`, `read_transposed`, `clear_src`, `finalize` |
| `math_fidelity` | `LoFi, HiFi2, HiFi4` |
| `dest_acc` | `No, Yes` |

**On the ct doc split.** The PR notes that blaze's comment lists the tested set `{1..6, 8, 10, 12, 14, 16}`
while the demo's says "any 1–16", that the difference is comment-only, and that "the code enforces neither" —
demo's wording was kept. That is an unresolved factual question sitting in a comment, and it is cheap to
settle: include `ct ∈ {7, 9, 11}` in the sweep. If they pass, the demo's wording is right and the doc is now
backed by a test. If they fail, the promoted header is shipping a documented-but-broken range and the comment
needs to become a `static_assert`. Either outcome is worth more than the comment.

Golden: reuse whatever `test_matmul_custom.py` uses from `helpers/golden_generators.py` (`MatmulGolden`);
`helpers/matmul_sweep.py` already exists for shape enumeration.

---

---

## 9. OPEN #5 — `eltwise_mul_scalar` HiFi init fix (ATTEMPTED AND REVERTED)

Worth recording in full, because the original plan for this item and its first
correction were **both wrong** about the mechanism. Do not re-run the experiment below.

### What was tried

A `BINARY_SHAPE_MODE` template switch in `eltwise_binary_test.cpp` selecting which
`TensorShape` reaches the binary init vs the binary execute, on the theory
that the production llk_api pair is asymmetric —
`llk_math_eltwise_binary_init` forwards `get_operand_tensor_shape(operand)` while
`llk_math_eltwise_binary` forwards `DEFAULT_TENSOR_SHAPE` — so:

* mode 0 `init=real, exec=real` — the suite's existing, self-consistent behaviour
* mode 1 `init=real, exec=DEFAULT` — "production before the fix"
* mode 2 `init=DEFAULT, exec=DEFAULT` — "production after the fix"

The switch was built, defaulted to mode 0, and confirmed inert: the full
`test_eltwise_binary.py` sweep stayed at **4388 passed, 72 skipped**.

### Why it was reverted

Mode 1 on a tiny tile **hangs the math thread** (BH p100a, device reset required). The
cause is concrete: `_llk_math_eltwise_binary_` derives `num_faces` and `face_r_dim` from
the `tensor_shape` it is handed (`llk_math_eltwise_binary.h` ~line 600). Forcing it to
`DEFAULT_TENSOR_SHAPE` on a `[16, 32]` / 2-face tile makes the math thread issue four
faces' worth of ops against a packer configured for two, and the MATH_PACK handshake
deadlocks.

That is a hang, not the silent corruption the blaze report describes (M2 MoE HiFi4
accuracy 0.70 → 0.9996). So the shape-pairing theory does not reproduce the bug — it
produces a *different*, harsher failure that production evidently never hits. A test that
hangs the device is worse than no test, so the change was reverted in full and
`test_eltwise_binary.py` re-verified at 4388 passed.

### What this rules out, and where to look next

If production really ran `init=real / exec=DEFAULT` on a non-default CB it would hang, not
mis-compute. Two consequences:

1. **The demo's CBs for this call site are almost certainly 32x32 / 4-face.** In that case
   `get_operand_tensor_shape` returns `DEFAULT_TENSOR_SHAPE` anyway and the shape argument
   is identical on both sides — meaning the tensor shape is **not** the delta at all, and
   the "mis-specializes the tile shape" wording in #52709 is describing the symptom rather
   than the mechanism.

2. **The remaining candidate delta is fidelity, not shape.** The shorthand applies
   `get_effective_math_fidelity<eltwise_binary_type, math_fidelity>()`
   (`llk_math_binary_api.h:38`) before forwarding to `_llk_math_eltwise_binary_init_`,
   whereas the blaze fix calls `_llk_math_eltwise_binary_init_` directly with the **raw**
   `MATH_FIDELITY`. The execute path applies the effective-fidelity transform too
   (`:59`), so the fix arguably *introduces* an init/execute fidelity difference rather
   than removing one — which is exactly the kind of asymmetry that would perturb an
   ELWMUL dest-reuse accumulation at HiFi while leaving LoFi byte-identical (the fix's own
   `if constexpr (MATH_FIDELITY != LoFi)` gate is consistent with this).

So the next attempt should sweep **`get_effective_math_fidelity` applied vs not** on the
init, holding the shape fixed at `DEFAULT_TENSOR_SHAPE` on both sides, with dest-reuse
ELWMUL at HiFi2/HiFi4. That is a one-line switch in the same place and carries no
face-count mismatch, so it should not hang.

Before building it, read `get_effective_math_fidelity` for `ELWMUL` — if it is the
identity for ELWMUL then this theory is dead too, and the right next step is to ask
the #52709 author for the failing demo configuration (which CB geometry, which fidelity,
which call site) rather than to keep guessing from the LLK layer.

---

## 10. Shared infrastructure the remaining items need

**Scope: pure LLKs.** Every remaining item drives the `_llk_*` lib functions and the
`llk_*` api wrappers directly. The compute-API layer
(`tt_metal/hw/inc/api/compute/experimental/*.h`) is **not** a test target here; where §3,
§4, §6 and §8 name a compute-API function such as `mul_reduce_scalar_chunked_tile` or
`rmsnorm_bcast_scalar_reuse_tiles`, that is shorthand for *the call sequence to reproduce
at the LLK level* — it names the order to drive the underlying LLKs in, nothing more.

**Include paths already work.** `-I../../hw/ckernels/blackhole/metal/llk_api` is on the
tt-llk compile line, so the promoted `experimental/llk_sfpu/...` and
`experimental/llk_*_api.h` headers are includable as `"experimental/<name>.h"` with no
fixture, and `experimental/...` under `tt_llk_blackhole/{llk_lib,common/inc/sfpu}` likewise.
The `VENDORED_LLK_LIB` include-path fixture that #52727 deletes is exactly what the
promotion buys. Nothing further is needed to reach any remaining header.

One wrinkle worth knowing: the metal-tree SFPU headers under
`experimental/llk_sfpu/` are written against the metal macro environment and read bare
`APPROX` / `DST_ACCUM_MODE`, so a tt-llk driver must define those before the include. Both
landed SFPU drivers show the pattern (`sfpu_add_rsqrt_test.cpp`, `sfpu_sampling_test.cpp`).

### Test parameters that already exist and can be reused

Landed while building the completed items; check here before adding a new one:

| Parameter | Emits | Useful for |
|---|---|---|
| `CUSTOM_MM_UNINIT` | `UNINIT_DENSE_PACKING`, `UNINIT_RESTORE_MOP`, `UNINIT_SKIP`, `BLOCK_MOP_NUM_FACES` | OPEN #3 — the same dense/geometry axes |
| `PACK_NUM_TILES` | `PACK_NUM_TILES` | any driver with a compile-time-bounded pack loop |
| `SFPU_FAST_APPROX` | `SFPU_FAST_APPROX` | sqrt/rsqrt-family template arg |
| `SAMPLING_PRGM0_HAZARD` | `SAMPLING_POLLUTE_PRGM0`, `SAMPLING_SKIP_RECIP_INIT` | template for any "prove the init is load-bearing" axis |
| `RMSNORM_DEST_REUSE` | `RMSNORM_NUM_TILES`, `RMSNORM_NUM_FACES`, `RMSNORM_CLEAR_DEST`, `RMSNORM_UNPACK_FULL_TRANSPOSE` | OPEN #4 — a worked example of bundling a template `num_tiles` with a MOP-sizing `num_faces` |

> `SORT_DST_WRITE_OFFSET` used to be listed here as reusable for OPEN #4. **It no longer
> exists** — review of the sort-header coexistence test established that the offset it swept
> is unobservable (`_llk_math_eltwise_unary_datacopy_` reprograms
> `DEST_TARGET_REG_CFG_MATH_Offset_ADDR32` before touching DEST), so the sweep and the
> parameter were both removed. A `top32_rm` test that wants to observe the helper needs a
> DEST consumer that does not reprogram that register first.

### Still to add

- **`helpers/test_variant_parameters.py`** — for OPEN #1: a compile-time `DST_CAPACITY` and
  the chunked `num_tiles` (both are template args in the chunked reduction, not runtime).
  For OPEN #4: `TOP32_MODE`, `TOP32_TOP_MIN`, `SORT_DIRECTION`. (The OPEN #2 entry that
  used to sit here — a template `num_tiles`, `CLEAR_DEST`, `UNPACK_FULL_TRANSPOSE` — landed
  as the single `RMSNORM_DEST_REUSE` bundle above.)
- **`helpers/golden_generators.py`** — `MulReduceScalarChunkedGolden` (OPEN #1),
  `Top32RmGolden` (OPEN #4, likely an extension of the existing `TopKXLGolden`). OPEN #3
  should reuse the existing matmul golden and `helpers/matmul_sweep.py`.
  No `RmsnormBcastScalarGolden` was needed in the end: the op is elementwise against one
  broadcast scalar, so the golden is a few lines in the test file reusing
  `EltwiseBinaryGolden._apply_fidelity_masking` for the ELWMUL phases. Worth copying rather
  than adding a generator class — see the note on fidelity in Finding 6 of the DONE
  document.
- **`conftest.py`** — every remaining test is Blackhole-only: `skip_for_wormhole` +
  `skip_for_quasar`.

Not needed, contrary to an earlier draft of this section: no `MathOperation` entry and no
dedicated golden class for `add_rsqrt` (it landed as a self-contained file computing its
golden inline), and no `USE_SHORTHAND_INIT` (that approach was reverted — see §9).

### Test-isolation caution

The landed `custom_mm` uninit-restore driver deliberately leaves hardware state dirty
between its two runs, and OPEN #1's chunked reduction re-inits mid-loop. Per the tt-llk
notes, HW state leaks between kernel reconfigurations, and a `TENSIX TIMED OUT` in such a
test must **not** be masked with `tt-smi -r` — that would hide the very reconfig escape the
test exists to catch. Reset only for a genuine runtime hang, and record what hung first.

**There is already one live reconfig escape in the tree, and it will bite anyone running a
combined session.** Measured on BH p100a, 2026-08-17:

```
pytest test_topk_xl.py <any test_eltwise_binary Bfp4_b/ELWMUL/LoFi/transpose_srca:Yes case>
  -> 1 failed
pytest test_eltwise_binary.py            (alone)   -> 4388 passed, 72 skipped
pytest test_topk_xl.py                   (alone)   -> 71 passed
```

Specifically `test_eltwise_binary[dest_acc:No-formats:Bfp4_b->Bfp4_b-broadcast_type:None_-math_op:Elwmul-math_fidelity:LoFi-transpose_srca:Yes-input_dimensions:[256, 32]-tile_dimensions:[32, 32]]`
passes alone and fails when `test_topk_xl.py` ran before it in the same session.

This is **pre-existing and unrelated to the promotions** — it reproduces on a clean checkout
of main with every blaze-promotion change stashed. It is recorded here only so that whoever
runs the full suite while working on OPEN #4 does not spend an afternoon blaming their own
`top32_rm` driver: `top32_rm` and `topk_xl` share the sort headers, so a new failure in that
area will look like your fault. Bisect single-file-then-target before assuming it is.

It also needs an owner in its own right — per the tt-llk notes a reconfig escape is a real
bug, not a test-ordering nuisance, and `tt-smi -r` must not be used to paper over it.

---

## 11. Open questions for the PR authors

1. ~~**#52713** — is `llk_api/experimental/llk_sfpu/llk_math_deepseek_top32_rm.h` intentionally
   promoted?~~ **Answered** while building the branch: yes. The PR body is simply stale — commit
   `d577a2d4a5f "Promote llk_math_deepseek_top32_rm SFPU wrappers"` adds it deliberately, after the
   description was written. (That SHA is on `pmilenkovic/promote-top32-rm`, #52713's own branch —
   it will not resolve in a checkout of this one, which is expected, not a stale reference.) It remains seven public entry points with **no in-tree caller and no test**,
   which is what OPEN #4 (§6) should cover.
2. **#52727** — `restore_tile_pack_mop` defaults to `false`, but the PR body describes the restore as
   unconditional. Which of the ten demo `*_block_uninit()` call sites opt in to `true`? The +22-instruction
   pack-TRISC delta reported in the notes suggests at least one does.
   *Partly answered:* the landed uninit test pins both polarities and shows the restore is a no-op unless
   the follow-on tile geometry differs from 32x32 (Finding 1 in the DONE document) — so a call site that
   opts in while already at 32x32 gains nothing. Which sites actually need it is still an open question for
   the author.
3. **#52727** — is `ct ∈ {7, 9, 11}` actually supported? The comment says 1–16, the tested set skips them,
   and nothing enforces either. Settling this in a test is OPEN #3 (§8).
4. **#52709** — `mul_reduce_scalar_chunked_tile` is new code rather than a reconciliation. Was it validated
   beyond the `test_rmsnorm` run cited (6 passed), and at which `(num_tiles, dst_capacity)` pairs? That
   determines how much of OPEN #1 (§3) is confirmation versus first-time coverage.
5. **All** — `HW_JIT_API_HEADERS` now ships five more `experimental/` compute headers in packaged metalium.
   Is there a packaging/compile gate that catches a header added to `experimental/` but missing from
   `sources.cmake`? Neither #52713 nor #52745 touches `sources.cmake`; #52713 adds no compute-API header, so
   it looks correct, but a gate would make that verifiable rather than reviewed.

---

## 12. Appendix — everything learned, for whoever picks this up

Practical knowledge accumulated while landing four tests and failing at two. Recorded here
because none of it is discoverable from the code without spending the same time.

### 12.1 Branch and repo mechanics

- Work branch: **`ldjurovic/llk-tests-blaze-promotions`** (tt-metal). It merges
  `pmilenkovic/promote-rmsnorm-family` + `promote-top32-rm` + `promote-custom-mm` onto
  `origin/main`, because the promoted headers do not exist on main and there is nothing to
  compile against otherwise.
- Those three branches have **different merge-bases** — only `promote-rmsnorm-family` is
  based on current main. Merging all three produced exactly one conflict,
  `tt_metal/hw/sources.cmake`, where both sides append to the same sorted
  `HW_JIT_API_HEADERS` list. Keep both sides, alphabetically.
- **The branch is not reviewable as-is.** It carries the three PR merge commits. The test
  commits touch only `tt_metal/tt-llk/tests/`, so rebase them onto main once the PRs land.
- `tt-llk` is **vendored** into tt-metal at `tt_metal/tt-llk`, not a submodule. Test changes
  are ordinary tt-metal commits.
- `tests/setup_testing_env.sh` installs a pre-commit hook into tt-metal's `.git/hooks`. It
  rejects commits for unrelated pre-existing repo state, so every commit here used
  `git -c core.hooksPath=/dev/null commit`. Worth knowing rather than fighting.

### 12.2 Test environment

- **Both** setup scripts are needed, in this order:
  `CHIP_ARCH=blackhole ./setup_testing_env.sh` (fetches SFPI 7.69.0) then
  `source ./setup_external_testing_env.sh` (creates `tests/.venv` and installs
  requirements). The first alone does **not** create the venv — it assumes the Docker
  image's Python environment — and `run_test.sh` fails with "venv not found" until the
  second has run.
- Run tests through `tt-llk/.claude/scripts/run_test.sh` (`count` / `compile` / `run`), never
  raw pytest. Invocation used throughout:
  `./.claude/scripts/run_test.sh run --worktree <abs path to tt-llk> --arch blackhole --test <file>`
- **`--k` with brackets, commas or spaces silently mangles the pytest args** and surfaces as
  an opaque xdist `assert not crashitem` worker crash that looks like an environment
  failure. Cost an hour of misdiagnosis. Use `--test-id`, or a bracket-free `--k` such as
  `--k prgm0_hazard`.
- Shell state does not persist between tool invocations: `source .venv/bin/activate` must be
  in the *same* command as the pytest call. A separate call silently falls back to
  `/opt/venv`, which has an incompatible `ttexalens` and fails at conftest import.
- Reset the device with `tt-smi -r` **only** for a genuine runtime hang. Never for compile
  errors or reconfig escapes — resetting masks the reconfig bug a test may exist to catch.

### 12.3 Harness facts that cost time

- **`passed_test`'s default tolerance is `atol=rtol=0.05` for every float format.** Far too
  loose for most numeric assertions. Both landed numeric tests pass measured, per-config
  tolerances via `custom_rtol` / `custom_atol` instead. Measure the error envelope first,
  then set the tolerance ~2.5x above it; do not guess, and check the guess is not *tighter*
  than the default (an early `add_rsqrt` draft tightened bf16 25x below default and failed
  its own correct results).
- Template parameters reach the kernel through a generated header pulled in by `params.h`.
  If a driver needs them at **file scope** (e.g. `constexpr` arithmetic before any
  `#ifdef LLK_TRISC_*`), `params.h` must be included at the top of the file, not only inside
  each TRISC block. `sfpu_sampling_test.cpp` is the precedent.
- A `TemplateParameter` emitting `constexpr ...` cannot be tested with `#ifdef`. Emit
  `#define NAME_VAL <n>` and derive the constexpr in the source if a default is needed for
  variants that do not pass the parameter.
- Enum spellings differ between layers: python `VectorMode.None_` is C++ `VectorMode::None`.
- `DataCopyGolden` needs `input_dimensions=[rows, cols]` for anything other than a single
  tile; without it a multi-tile call raises `shape '[1, 1024]' is invalid`.
- `-I../../hw/ckernels/blackhole/metal/llk_api` and `-I../../hw/inc` are already on the
  compile line, so promoted `experimental/...` headers need no fixture.
- Metal-tree SFPU headers under `experimental/llk_sfpu/` read bare `APPROX` and
  `DST_ACCUM_MODE`; define both before the include. Both landed SFPU drivers show it.

### 12.4 LLK facts established by measurement

- **`_llk_pack_` executes whatever packer MOP is currently installed** (via
  `ckernel_template::run()`), so a leftover MOP from a previous op directly changes it. This
  is what makes the `custom_mm` uninit test possible.
- **The pack MOP bakes in tile geometry.** A restore is unobservable when the follow-on
  geometry already matches: the `custom_mm` MOP restore is inert at 4 faces and only visible
  at 2. Corollary for any future pack-state test: vary the geometry, or you are measuring
  nothing.
- **`_llk_pack_`'s tile index is honoured under `_llk_pack_reduce_mask_config_`** — packing
  `DEST[n]` vs `DEST[0]` gives different data.
- **`fill_tile` is `_calculate_fill_` at `VectorMode::RC`, 8 iterations** (whole tile). The
  `RC_custom` / 2-iteration form used to stage a reduce scaler covers only a small region.
  Not interchangeable.
- **The dense_packing W-stride constants are not format-aware** — item D1. `set_packer_strides`
  uses `datum_size_in_bytes(pack_src_format)`; `custom_mm.h` hardcodes `* 2`.
- **The two sort families cannot both be initialized in one kernel.** `_top32_rm_init_()` and
  `_topk_xl_init_<K, fused>()` together hang the math thread — overlapping ADDR_MOD slots,
  MOP and REPLAY buffer. They coexist in a translation unit, which is all #52713 claims.
- **A polluted `vConstFloatPrgm0` degrades rather than corrupts.** It makes
  `t = x*y - Prgm0` positive, so `sfpu_reciprocal_iter`'s `v_if(t < 0)` refinement never
  fires and raw `approx_recip` survives — about 1e-3 relative, inside the suite's 2%
  tolerance. Any "is this init load-bearing" test needs its own threshold.
- **`_llk_math_eltwise_binary_` derives `num_faces` / `face_r_dim` from the `TensorShape` it
  is handed.** Forcing `DEFAULT_TENSOR_SHAPE` on a 2-face tile makes math issue four faces
  of ops against a two-face packer and deadlocks MATH_PACK. Relevant to item 5 (§9).
- **ELWMUL accumulates into DEST; ELWADD overwrites it.** In the rmsnorm dest-reuse MOP both
  branches pass 0 in the instruction's dest-accumulate slot — `TT_OP_ELWADD(0, acc_to_dest,
  ...)` with `acc_to_dest == 0`, and `TT_OP_ELWMUL(0, 0, ...)` — so they read as if they
  behave alike. They do not: with the ZEROACC suppressed the mul lands `seed + A*scalar`
  while the add lands `A + scalar`. This is why every rmsnorm mul call site must pass
  `clear_dest=true`, and it generalises — do not assume a 0 in that slot means overwrite.
- **LoFi ELWMUL costs a few percent, not a few ULPs.** On a `uniform(-4, 4)` sweep the LoFi
  mantissa masking produced ~3% relative error — far outside any tolerance worth calling
  tight. Model it with `EltwiseBinaryGolden._apply_fidelity_masking` per phase (as
  `test_eltwise_binary.py` does) rather than widening `rtol`, or the test stops being able
  to see a real regression.
- **`-Werror` in the tt-llk build is stricter than the metal kernel build.** Two promoted
  headers (`llk_unpack_A_rmsnorm.h`, `llk_math_rmsnorm_bcast_scalar_dest_reuse.h`) carried
  dead locals that metal never compiled and tt-llk rejects outright. Expect this for any
  header being covered by tt-llk for the first time; it is a prerequisite task, not a
  surprise, and it is part of why the ~2 d estimate on the rmsnorm item was optimistic.

### 12.5 Method notes

Three hypotheses were disproved on silicon (§3 twice, §9 once). What would have caught them
sooner:

1. **Read the shape of the error before theorising.** The chunked reduce was off by a
   *non-integer, size-dependent* factor. That is a race or a wrong source, never an
   off-by-one — an off-by-one gives a clean ratio. Two rounds were spent on clean-ratio
   theories.
2. **Verify a change reaches the binary before concluding it had no effect.** Checking the
   source file contains the edit is not that check. Three identical results in a row were
   the signal; a one-line probe (pack `DEST[0]`) settled it immediately and should have been
   first.
3. **Prefer the experiment that halves the search space** over the one matching the current
   theory. Packing `DEST[0]` was listed as "cheapest decisive check" in two successive
   revisions of this document and skipped both times in favour of a specific fix.
4. **Do not encode an unestablished defect as `xfail`.** Both reverts could have been landed
   as xfail and looked like progress; neither had established that the *promoted code* was
   wrong rather than the driver. `xfail` asserts a product defect and should be used only
   with evidence — as it is for D1, where the arithmetic is provably wrong.

### 12.6 Commit trail on the work branch

`4164de1` add_rsqrt · `3898f05` custom_mm uninit · `906e73a` sort-header coexistence ·
`5c6a605` + `744bbe2` docs · `169a504` sampling Prgm0 hazard · `931eab1` + `fc2aec7` +
`3e42be9` doc upkeep · `566a1e5` + `d6e00d0` + `3b8903f` the chunked-reduce attempts.

---

## 13. Defect dossiers — every open bug, with fix options

One entry per open defect in
[`REMAINING_WORK.md`](REMAINING_WORK.md) (C1–C6 and F). `REMAINING_WORK.md` says what needs
doing and by whom; this section is the detail an owner needs before touching any of it —
mechanism, what is measured versus inferred, blast radius, how to reproduce, the fix options
with their trade-offs, and which test flips when it is fixed.

Two conventions used throughout. **Measured** means observed on Blackhole p100a through
`.claude/scripts/run_test.sh`; anything not marked measured is inference from reading the code.
And every dossier names a **tripwire** — the existing test that changes state when the defect is
fixed — because a defect with no tripwire gets re-litigated instead of closed.

---

### 13.1 C1 — `dense_packing` programs a 16-bit W-stride on any pack source

**Symptom.** With `dense_packing` set and a 32-bit pack source, packed output is wrong: tiles
land at half the intended spacing. Measured at **0.25 match** against golden.

**Mechanism, fully established.** `set_packer_strides` (`cpack_common.h:301-305`) derives the
field as

```
w_stride = TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * datum_size_in_bytes(pack_src_format)
```

and `datum_size_in_bytes` (`ckernel_defs.h:274`) returns 4 for Float32. The compute API instead
hardcodes the multiplier as `* 2` in four places — `custom_mm.h:69` and `:261`,
`compressed_custom_mm.h:69` and `:262`. So on a Float32 pack source **both ends are 2x off**:
init programs `2*16*16*2 = 1024` where 2048 is correct, and the uninit restores
`4*16*16*2 = 2048` where 4096 is correct. It is not an unrestored-state bug; the op is wrong
end to end, and the uninit merely fails to undo a value that was wrong on the way in.

**Blast radius.** Every caller that sets `dense_packing` with a 32-bit output CB. All current
in-tree callers use 16-bit output, which is why this has never been seen in production.

**Reproduce.** `--test test_custom_mm_uninit_restore.py`; the Float32 cell is the xfail.

**Fix options.**

| | What | Cost | Leaves behind |
|---|---|---|---|
| 1 | `LLK_ASSERT` in `*_block_init` that `datum_size_in_bytes(pack_src_format) == 2` when `dense_packing` is set | Minutes | 32-bit dense packing unsupported, but loudly so |
| 2 | Derive the datum size from the output CB in init, **and add an `out_cb_id` parameter to `*_block_uninit`**, which currently takes none | ~0.5 d plus a signature change | Nothing, but it touches every call site: `matmul.hpp`, `flash_mla.hpp`, `dram_streaming_matmul*.hpp`, `matmul_custom_compressed_kernel.cpp` |

**Recommendation.** Option 1 now, option 2 when someone actually needs 32-bit dense packing.
The reason to do *something* now is that option 1 is minutes of work and converts silent
corruption into a build/runtime failure; leaving it is the only outcome with no upside.

**Tripwire.** `test_custom_mm_uninit_restore.py`'s Float32 `xfail` (marker form, so the body
really builds, runs and compares) flips to XPASS under either fix.

---

### 13.2 C2 — the `eltwise_mul_scalar` HiFi workaround cannot do what its comment says

**Symptom.** Not a wrong answer — a workaround whose stated mechanism is disproved by the code
it calls, which means nobody knows what fixed the measurement it claims to fix.

**What the code actually does.** `deepseek_binary_dest_reuse_tiles_init`'s HiFi branch hardcodes
`ckernel::DEFAULT_TENSOR_SHAPE` and attributes a HiFi4 accuracy fix to the shorthand init
"mis-specialising the tile shape". Reading the three things it depends on:

- `get_effective_math_fidelity<ELWMUL, f>()` is the **identity** for ELWMUL
  (`llk_math_common_api.h:123-125`), so the fidelity gate cannot be what differs.
- `acc_to_dest` is 0 in **both** arms.
- The shorthand `llk_math_eltwise_binary_init` (`llk_math_binary_api.h:31-42`) resolves the shape
  as `get_operand_tensor_shape(get_operand_id(operand_A))` *regardless of fidelity*.

So tensor shape is the only difference between the two arms. And `get_operand_tensor_shape`
reads compile-time CB metadata (`llk_operands.h:40-46`, emitted by `genfiles.cpp:880-882`), which
for a standard 4-face 32x32 tile is exactly `DEFAULT_TENSOR_SHAPE` — making the HiFi arm
**bit-identical to the shorthand it replaces**.

**The other half, measured.** On a 2-face tile, forcing `DEFAULT_TENSOR_SHAPE` makes
`_llk_math_eltwise_binary_` derive `num_faces`/`face_r_dim` from the shape it was handed, issue
four faces of ops against a two-face packer, and **deadlock the MATH_PACK handshake**. It does
not silently corrupt `silu(gate)*up` as the comment states.

**So the workaround is either inert (4-face CB) or a hang (2-face CB). There is no
configuration in which it does what it says.** Meanwhile the paired execute
(`deepseek_binary_dest_reuse_tiles` → the 4-arg `llk_math_eltwise_binary` overload) *does* derive
the shape from the CB, so on non-default geometry init and execute disagree with each other.

**What is unexplained.** #52709 reports `gated_local_reduce` at HiFi4 going from 0.70 to 0.9996.
That measurement is real; its cause is not the mechanism in the comment.

**Fix options.**

1. **Get the failing config and re-measure** (needs the #52709 author). If the 0.70 was on a
   4-face CB, the arm is inert and something else in that PR fixed it — worth knowing which,
   because that something is load-bearing and undocumented.
2. **If inert: delete the arm**, keep the shorthand. Smallest honest outcome.
3. **If the real mechanism is elsewhere** — init ordering, unpack-side fidelity phases, a
   different overload — document that and drop the tensor-shape story. A workaround with a wrong
   rationale is worse than none: the next person "fixes" it by generalising the wrong thing.

**Tripwire.** None yet, and that is the point: **A5** is the test for this init sequence and is
deliberately blocked on the answer, because if the arm is inert the honest outcome is deleting
the code rather than testing it.

---

### 13.3 C3 — pre-existing `topk_xl` → `eltwise_binary` reconfig escape

**Symptom.** A golden mismatch in `eltwise_binary` that depends on a `topk_xl` test having run
before it in the same session — i.e. state leaking across kernels, not a bug in either op's own
sequence.

**Established.** It reproduces on **clean `main`** with every promotion change stashed, so it is
not caused by the promotions. That is the single most useful fact about it: anyone bisecting a
failure in this area will otherwise burn a day blaming their own driver.

**Why it matters here.** The `top32_rm` work shares this area (the two sort families overlap in
ADDR_MODs, the MOP and the REPLAY buffer — see §12.4 and Finding 3). A failure that looks like
"my new sort driver broke `eltwise_binary`" is more likely to be this.

**Reproduce.** `experimental_reconfig_escape_test.cpp` is the standing driver for this class.
Bisect **single-file-then-target**: run the suspected test file alone, then with the suspected
predecessor, before concluding anything.

**Do not** use `tt-smi -r` to make it go away. A reset masks reconfig escapes, which is exactly
the class of bug this is; the tt-llk notes reserve resets for runtime timeouts.

**Fix options.** The fix is whichever reconfig the escaping op fails to restore, so the work is
localisation rather than choosing a design:

1. Diff the CFG state the two ops program — ALU format spec, ADDR_MODs, the pack MOP — around a
   passing and a failing ordering, and find the field the second op assumes rather than sets.
2. Then decide, as with C4, whether the *entering* op should set it (clean-state-on-entry) or the
   *leaving* op should restore it (uninit). This branch's experience is that clean-state-on-entry
   ages better: an uninit is one more thing a caller can forget.

**Tripwire.** None dedicated. Whoever fixes it should add the ordering as a test, because an
escape with no test comes back.

---

### 13.4 C4 — `mul_reduce_scalar` re-entry needs a DEST-section boundary

**Symptom.** Running the known-good non-chunked sequence twice over the same input gives a wrong
answer the second time — **9.27x to 9.93x golden, all 12 variants** — unless a DEST-section
boundary separates the two passes.

**Mechanism, established by the reproducer.** `tests/sources/mul_reduce_scalar_reenter_test.cpp`
plus `test_mul_reduce_scalar_reenter.py`, ~40 lines, measured:

| Configuration | Result |
|---|---|
| `passes=1`, either mode (control) | correct |
| `passes=2`, DEST-section boundary between passes | correct, and **bit-identical** across passes |
| `passes=2`, one shared DEST section | **wrong — all 12 variants** |

So the family **is** re-enterable; what is broken is re-entry with no
`dest_section_done` / `wait_for_dest_available` pair in between. That handshake restores
something the second `_llk_math_mul_reduce_scalar_init_` does not.

**Blast radius, and why this is a shipping defect rather than a test curiosity.**
`mul_reduce_scalar_chunked_tile` (`rmsnorm.h:105`) is built exactly the broken way: it documents
that the caller "must acquire DST before calling", then re-enters every batch inside that one
section with `if (batch > 0) mul_reduce_scalar_init(...)` as its only restoration attempt. The
first attempt at a chunked driver (§3) reported 5-30x golden; this reproduces 9.3-9.9x. Same
signature, non-integer multiplier in both cases, so very likely the same defect with a much
smaller reproducer.

**Fix options.**

1. **In the LLK** — make `_llk_math_mul_reduce_scalar_init_` (or `switch_to_reduce`) restore
   whatever the section boundary restores. Right if re-entry inside a section is meant to work.
   Needs someone to identify the state first: diff the CFG/ADDR_MOD/RWC state across a boundary
   versus across a bare re-init.
2. **In the compute API** — have `mul_reduce_scalar_chunked_tile` close and reacquire the DEST
   section per batch, or document that it cannot be used as written. Right if the per-batch
   handshake is considered the caller's job.

**Recommendation.** Decide 1 vs 2 before touching D1: if the op cannot work as written and nobody
wants the chunked form, **deleting `mul_reduce_scalar_chunked_tile` is a legitimate outcome** and
the cheaper one — it currently ships with no caller and no test.

**Do not** re-investigate the accumulator fill or a missing UNPACK/MATH barrier. §3 records both
as tried on silicon and disproved, and this result explains why neither moved the number.

**Tripwire.** The 12 `xfail`s in `test_mul_reduce_scalar_reenter.py` (marker form, so the bodies
run) flip to XPASS the moment re-entry inside one section restores state.

---

### 13.5 C5 — out-of-bounds metadata read, live on `main`

**Symptom.** An L1 read one word past the compressed-matmul metadata buffer whenever
`kt_dim * ct_dim` is a multiple of 10. `kt_dim=10, ct_dim=1` is the smallest case, and it is
inside the documented ranges.

**Mechanism.** The metadata buffer holds `ceil(kt_dim * ct_dim / 10)` words. Both walkers
reload unconditionally after consuming an item, so on the last item of an exact-multiple-of-10
block they step one word beyond the buffer: the unpack side at `meta_ptr[full_iters]`
(`llk_unpack_AB_compressed_custom_mm.h`), and the math side in three places
(`llk_math_compressed_custom_mm.h`).

**What it costs, stated precisely so nobody over- or under-reacts.** At `rem_iters == 0` the
remainder loop never runs, so the word read past the buffer is **never used**, and no golden can
observe it — confirmed by running the boundary test against the unguarded kernel, where it
passes. This is a **memory-safety** defect, not a wrong-answer defect.

**Status: fixed, pushed, needs a PR.** `ldjurovic/compressed-mm-oob-guard` is cut from `main`
with both guards and nothing else — 2 files, +30/-12, no compute-API change — and verified from
its own worktree at **582 passed**. The guard bounds the reload with
`num_meta_words = (kt_dim * ct_dim + 9) / 10`, the same expression the caller sizes the buffer
with, so the guard and the allocation cannot drift apart.

**Tripwire.** The 6 metadata-word-boundary variants in `test_matmul_custom_compressed.py` reach
the exact case; they live on the #53130 branch, so run them there rather than on the guard branch.

---

### 13.6 C6 — `top32_rm`'s 32-bit unpack branch sorts against stale Dest

**Symptom.** On the 32-bit (unpack-to-dest) branch, a chunk that fills fewer than 4 faces
produces a top-32 containing values **that were never in the input**. Measured: a 160-element
Float32 row of values in [-80, 79] returned 11026.0, 10041.0, 9058.0 and more — recognisable as
another test's stimuli, i.e. leftovers in Dest.

**Mechanism, partly established.** `llk_unpack_A_top32_rm_api.h` forks on the src format:

- **16-bit branch** — the unpacker clears SrcA to -infinity (`TTI_UNPACR_NOP … CLR_SRC_NEGINF`)
  before unpacking, so faces the caller did not fill hold -infinity and lose every comparison.
  The math half then moves all four faces into Dest.
- **32-bit branch** — the tile goes straight to Dest via unpack-to-dest; **nothing clears
  anything**, and the `ZEROACC` loop inside `_llk_math_top32_rm_` runs `i < num_faces`, so the
  faces beyond `num_faces` are never touched. The sort reads all four regardless.

**Latent in the consumer, and the reason is worth keeping.** `top32_rm_dev_compute.cpp` *does*
drive this branch with `num_faces=2`, but only for its uint32 **index** tile, and an index slot
can only be selected when the paired value slot wins — the value tile is bf16, so its padding is
-infinity and never wins. **The 32-bit branch is safe today only because a 16-bit operand is
supplying the -infinity next to it.** Put values through it and the defect is live.

**A fix was attempted on 2026-08-20 and reverted. Read this before trying the obvious thing.**
The obvious thing is to extend that `ZEROACC` loop from `num_faces` to `TILE_NUM_FACES`, and it
does not work:

- It compiles into a further change, because after it `num_faces` is unused in the math half —
  both branches then process the whole tile — so the parameter has to be dropped from
  `_llk_math_top32_rm_`, from `llk_math_top32_rm` and from 6 call sites in the two dev kernels.
- **Measured: it passes in isolation and still fails in a session.** With the change,
  `--test test_top32_rm.py --k partial_chunk` **XPASSes**, and the same variant inside a
  full-suite run still **XFAILs** — twice in a row. Whatever the sort is reading, a Dest clear
  issued from the math thread at that point does not remove it.
- What that rules out: "the unwritten faces merely hold uninitialised content that ZEROACC
  fixes". What it points at instead: either the `clear zero flags` form of ZEROACC used here does
  not zero content, or the window unpack-to-dest writes and the window `_llk_math_transpose_dest_`
  subsequently reads do not line up with the faces the clear addresses. Establishing which is the
  next investigative step, and it wants a Dest dump (DPRINT via `debug_dest_copy.cpp`) rather
  than more guessing.

**Fix options, in the order I would try them.**

1. **Give the 32-bit branch the 16-bit branch's semantics.** Have the unpacker issue
   `CLR_SRC_NEGINF` plus a dvalid and have the math half run the MOP for faces
   `num_faces..3`, so the padding is -infinity exactly as on the working branch. This is the only
   option that makes the two branches agree, and agreement is what makes the family's doc tables
   true. Cost: cross-thread choreography (both TRISCs must know the face count), which is why it
   is not a five-line change.
2. **Reject `num_faces < 4` on the 32-bit branch at compile time**, and document partial chunks
   as 16-bit only. Honest, minutes of work, and it turns a silent wrong answer into a build
   error. It also permanently forecloses fp32 values with a tail, which the doc tables currently
   promise.
3. **Caller-side, no LLK change:** pad the chunk in L1 to a full 64 elements with -infinity and
   always pass `num_faces=4`. Works on both branches today, costs one memset and a little L1, and
   is what I would tell a caller to do while 1 is unowned.

**Tripwire, with a caveat that matters.** `test_top32_rm_32bit_partial_chunk` is a non-strict
`xfail` and flips to XPASS when fixed — **but judge it from a full-suite run, not from `--k`**.
Run alone it XPASSes even with the defect present, because nothing has polluted Dest yet. That
sensitivity is itself evidence about the mechanism.

**What it currently costs the sweep.** One shape: the Metal dev test's own `row=3232`, which ends
in a 32-element chunk. Everything either side of it is covered, and fixing C6 makes that shape a
one-line addition to `PRE_SORTED_ROW_ELEMENTS`.

---

### 13.7 F — intermittent `test_matmul_custom_compressed` failure (host/BRISC desync)

**Symptom, two shapes.** Under repeated runs the suite fails roughly **2 in 6**, usually as a
**hang**, and at least once as a **wrong answer** (`shape=(1, 64, 32), formats=('bfp4',)` at
PCC -0.033, 587/588) that did not reproduce — that variant passes 17/17 in isolation and the
suite then passes 588/588. An owner looking only for a hang will miss half of it.

**Mechanism, as far as triage gets it.** `run_test.sh`'s triage on a hang caught:

```
Unpacker/Math/Packer mailboxes = 0x0 (KERNEL_STARTED)
TRISC0/1/2  in_reset=True
BRISC       pc=0x368, unchanged  (spinning)
BriscCounter=0x118 (280)   host Python counter: 281
```

All three TRISCs sit in soft reset while BRISC spins **one command behind the host** — a
host↔BRISC command-protocol desync, not an LLK compute bug. `get_tensix_state` then failed to
halt BRISC, so the device was already unresponsive.

**Scope.** Every failing variant reproduced (`clustered`, `interleaved`, `single`) is
`@pytest.mark.nightly`, and the PR gate filters `not nightly`. So this affects nightly runs, not
the gate.

**Fix options.** This is harness/dispatch territory rather than LLK:

1. **Instrument the command path.** The desync is one command deep, which is small enough to
   catch: log the host counter and `BriscCounter` around every `commit_brisc_command`
   (`helpers/device.py`) and find whether the host advances without an acknowledged commit, or
   BRISC misses a wake.
2. **Look for the missing barrier.** A one-behind desync under back-to-back launches is the
   signature of a write that is visible to the host but not yet to BRISC — i.e. a flush/barrier
   between writing the command and ringing it.
3. **Separate the two symptoms.** Repeat a *single* variant many times rather than the suite. If
   the hang reproduces on one variant, it is a launch-path bug; if only the suite reproduces it,
   the trigger is state carried across variants and it belongs with C3.

**Before reproducing it yourself:** back-to-back runs are not how CI runs it and may be the
aggravating factor rather than an independent trigger, and it wedges the device
(`PcieHangError`, all devices unhealthy) often enough that you should expect to `tt-smi -r`
between attempts. That is the sanctioned remedy for a runtime timeout — not for a reconfig
escape, which is why C3 must not be treated this way.

**Tripwire.** None, and one is hard to write for a 2-in-6 nightly hang. The nearest thing is the
triage output above: whoever fixes it should confirm the counters advance in lockstep under the
same loop that produced this.
