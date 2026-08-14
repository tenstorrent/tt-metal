# tt-llk blaze promotions — OPEN work

> **What this is.** The remaining tt-llk test work for the blaze->tt-metal `experimental/`
> promotions (#52709, #52713, #52727). **5 items remain: 3 not started, 2 attempted and
> reverted.** Completed work has been moved out to
> **`BLAZE_PROMOTION_TESTS_DONE.md`** — check there before starting anything, since three
> of the plans below were already corrected by what those tests measured on silicon.
>
> Branch: `ldjurovic/llk-tests-blaze-promotions` (tt-metal), which merges the three PRs
> onto main so the promoted headers exist to compile against. Rebase onto main once they
> land; the test commits touch only `tt_metal/tt-llk/tests/`.
>
> Hardware: Blackhole p100a. Every item below is Blackhole-only — every promoted header is
> `#if defined(ARCH_BLACKHOLE)`-guarded — so each test carries `skip_for_wormhole` +
> `skip_for_quasar`.
>
> Run tests via `tt-llk/.claude/scripts/run_test.sh`, not raw pytest. See the tooling note
> at the end of the DONE document for two gotchas that cost time.

---

## Open work at a glance

| # | Item | PR | Est. | Notes |
|---|------|----|------|-------|
| 1 | `mul_reduce_scalar_chunked_tile` — new driver, not an extension | #52709 | ~1 d spent, not done | **§3. Attempted and reverted — read §3 before retrying.** Driver written and compiling; result is ~5-30x high and unexplained |
| 2 | `rmsnorm` bcast-scalar dest-reuse — new file | #52709 | ~2 d | **§4** |
| 3 | plain `custom_mm` matmul — new file | #52727 | ~2 d | **§8.** Also settles the `ct ∈ {7,9,11}` doc question |
| 4 | `top32_rm` — new file, two modes | #52713 | ~3-4 d | **§6.** Largest single effort; also the only coverage for the 7 newly-promoted SFPU wrappers |
| 5 | `eltwise_mul_scalar` HiFi init fix | #52709 | ? | **§9. Attempted and reverted — read §9 before retrying.** Hangs the device as first written |

> The sampling `recip_init` polluter test (formerly item 6) is **done** — moved to the DONE
> document. It found that the hazard is a ~1e-3 precision loss hidden by the suite's 2%
> reciprocal tolerance, not the garbage the PR wording implies.

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

## 3. OPEN #1 — `mul_reduce_scalar_chunked_tile` (ATTEMPTED AND REVERTED)

> **Status: a working driver exists in history but was reverted — it produced a wrong
> answer that was not diagnosed.** Do not start from scratch; start from what is below.

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

**Most likely cause — identified after the revert, and it supersedes the three suspects
below: the driver has no per-batch unpack/math synchronisation.**

The non-chunked op has exactly **one** phase transition (multiply everything, then
`switch_to_reduce`, then reduce), so the SrcA/SrcB dvalid handshake alone keeps UNPACK and
MATH ordered. The chunked form has **one transition per batch**, and nothing in the driver
stops the unpack thread running ahead: as soon as it finishes batch 0's
`switch_to_reduce` it re-inits and issues batch 1's `_llk_unpack_AB_` calls, which set
dvalid and overwrite SrcA/SrcB while MATH is still mid-reduce on batch 0. MATH then
column-reduces partially-clobbered source data.

That fits the symptom in a way none of the suspects below do: the result is inflated
(extra/duplicated accumulation) by a **non-integer, tile-count-dependent factor**, which is
what a race produces — a deterministic logic error would give a clean ratio.

The reason production does not hit this is that the real compute kernel gets the ordering
for free from circular-buffer flow control (`cb_wait_front` / `cb_pop_front` around each
chunk). The tt-llk harness has no CBs, so the driver must supply the barrier itself. The
harness does initialise the tensix semaphores for this — `helpers/include/boot.h` sets up
`UNPACK_TO_DEST`, `MATH_DONE`, `PACK_DONE` and `PACK_UNPACK` — and several existing
multi-phase drivers already sync by hand; `sources/sdpa_reinits_test.cpp` and
`sources/matmul_unpack_tilize_test.cpp` are the closest models.

**Concrete fix to try first:** make the unpack thread wait on a math-side semaphore post at
the end of each batch's reduce, so batch *n+1*'s unpacks cannot begin until batch *n*'s
reduce has consumed SrcA/SrcB. If that turns the numbers correct, the earlier suspect list
is moot. The `dst_capacity = 2, num_tiles = 3` bisect below is still the right first
measurement, because with `batch_size = 1` the race window is at its widest.

**Older suspects, kept only in case the sync fix does not resolve it:**

1. **Packing `DEST[ACCUMULATOR]` under the reduce mask.** The working non-chunked driver
   packs `DEST[0]`; this one packs the reserved slot with
   `_llk_pack_reduce_mask_config_<REDUCE_SCALAR>` active. If the mask interacts with
   `set_dst_write_addr(tile_index)` such that a non-zero tile index is not honoured, the
   packed datum is not the accumulator at all. **Cheapest decisive check:** pack `DEST[0]`
   instead and see whether the number moves. If it does not, the pack path is the problem.
2. **The SFPU add's DEST addressing.** `VectorMode::RC` walks four faces advancing the
   dst-write counter; whether `dst_index_out == dst_index_in0 == ACCUMULATOR` aliases
   correctly across that walk is unverified. Try a single-face `VectorMode` and 8 iterations.
3. **Per-batch state restore.** The driver re-inits the multiply and the unpack for
   `batch > 0`, matching the API's own `if (batch > 0)` guard, but the reduce phase may
   consume more state than those two inits restore.

A useful bisect before any of that: run with `dst_capacity = 2` and `num_tiles = 3`
(`batch_size = 1`, three single-tile batches) and compare against three separate
non-chunked runs. That isolates the cross-batch accumulation from the reduce itself.

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

## 4. OPEN #2 — `rmsnorm` bcast-scalar dest-reuse (new file)

- `tests/python_tests/test_rmsnorm_bcast_scalar_dest_reuse.py`
- `tests/sources/rmsnorm_bcast_scalar_dest_reuse_test.cpp`

**Why a new file, not an extension.** The op is a `num_tiles`-templated MOP
(`rmsnorm_bcast_scalar_dest_reuse_configure_mop<eltwise_binary_type, num_tiles, math_fidelity>`) driven from a
*single* unpack, with SrcB sourced from DEST via `MOVD2B` under a `WAIT_SFPU | SRCB_VLD` stall
(`rmsnorm_bcast_scalar_reuse_dest_as_src`). No existing test file has that structure: `test_bcast.py` does
one-tile-per-unpack broadcasts, `test_eltwise_binary.py` has no MOP-over-N-tiles axis and no
`num_tiles`-as-template-argument plumbing.

Kernel structure — replicate `api/compute/experimental/rmsnorm.h`'s call sequence at the llk layer, which is
the established tt-llk convention (see §6):

```
UNPACK: _llk_unpack_A_rmsnorm_init_<num_tiles, SCALAR, true, DEST_TO_SRCB>(transpose, transpose, ...)
        _llk_unpack_A_<SCALAR, true, DEST_TO_SRCB>(...)          // ONE unpack for all num_tiles
MATH:   _llk_math_rmsnorm_bcast_scalar_dest_reuse_init_<op, num_tiles, fidelity>(num_faces, acc_to_dest)
        _llk_math_rmsnorm_bcast_scalar_dest_reuse_<op, num_tiles, dest_acc, fidelity, clear_dest>(src, dst)
PACK:   pack num_tiles tiles
```

Golden: seed DEST with `num_tiles` known tiles (a datacopy pre-pass, same technique as
`_prepare_dest_reuse_inputs` in `test_eltwise_binary.py`), then apply the scalar at element `[0]` of the
unpacked operand element-wise across all `num_tiles` × all faces, in `ELWADD` or `ELWMUL`. Reuse
`helpers/golden_generators.py`'s binary golden with a bcast-scalar wrapper.

Sweep:

| axis | values | rationale |
|------|--------|-----------|
| `eltwise_binary_type` | `ELWADD`, `ELWMUL` | both instantiate distinct MOP branches (`TT_OP_ELWADD` vs `TT_OP_ELWMUL` at lines 31-64) |
| `num_tiles` | `1, 2, 3, 7, 8` (bf16) / `1..4` (fp32 DEST) | it is the MOP's outer-loop count; DEST half-sync capacity caps it |
| `math_fidelity` | `LoFi, HiFi2, HiFi4` | fidelity is a template arg on both init and execute |
| `clear_dest` | `False, True` | template arg with no other coverage |
| `dest_acc` | `No, Yes` | |
| `num_faces` | `1, 2, 4` | runtime arg to `_init_`; tiny-tile geometry |
| `unpack_full_transpose` | `False, True` | **the axis only blaze's version has** — the `_fidelity` init folds transpose into the unpack. This is new reachable surface, so it must be swept. |

Priority note: `unpack_full_transpose=True` and `clear_dest=True` are the two axes that exist *only* because
blaze's version won the reconciliation. They are the highest-value cells in this matrix.

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
| `SORT_DST_WRITE_OFFSET` | `SORT_DST_WRITE_OFFSET` | OPEN #4 — `top32_rm` shares the helper |
| `SAMPLING_PRGM0_HAZARD` | `SAMPLING_POLLUTE_PRGM0`, `SAMPLING_SKIP_RECIP_INIT` | template for any "prove the init is load-bearing" axis |

### Still to add

- **`helpers/test_variant_parameters.py`** — for OPEN #1: a compile-time `DST_CAPACITY` and
  the chunked `num_tiles` (both are template args in the chunked reduction, not runtime).
  For OPEN #2: `NUM_TILES_TEMPLATE` (rmsnorm's `num_tiles` is a *template* arg, unlike the
  existing runtime `TILE_COUNT`), `CLEAR_DEST`, `UNPACK_FULL_TRANSPOSE`. For OPEN #4:
  `TOP32_MODE`, `TOP32_TOP_MIN`, `SORT_DIRECTION`.
- **`helpers/golden_generators.py`** — `MulReduceScalarChunkedGolden` (OPEN #1),
  `RmsnormBcastScalarGolden` (OPEN #2), `Top32RmGolden` (OPEN #4, likely an extension of
  the existing `TopKXLGolden`). OPEN #3 should reuse the existing matmul golden and
  `helpers/matmul_sweep.py`.
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

---

## 11. Open questions for the PR authors

1. ~~**#52713** — is `llk_api/experimental/llk_sfpu/llk_math_deepseek_top32_rm.h` intentionally
   promoted?~~ **Answered** while building the branch: yes. The PR body is simply stale — commit
   `d577a2d4a5f "Promote llk_math_deepseek_top32_rm SFPU wrappers"` adds it deliberately, after the
   description was written. It remains seven public entry points with **no in-tree caller and no test**,
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
