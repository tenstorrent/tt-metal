# tt-llk blaze promotions — COMPLETED work (archive)

> Closed record of the tt-llk test work for the blaze->tt-metal `experimental/` promotions
> that is **done and passing on Blackhole p100a**. Split out of
> `BLAZE_PROMOTION_TEST_STRATEGY.md` on 2026-08-14 so that document only tracks
> outstanding work.
>
> Most of this needs no action. **Four items do**, and are cross-referenced from the
> open-work document:
>
> - **Finding 2** — the `dense_packing` W-stride defect, recorded as an `xfail` in a landed
>   test. Needs an owner decision. (Its `xfail` only became a working detector on
>   2026-08-18 — see Finding 10.)
> - **Finding 7** — the `eltwise_mul_scalar` HiFi workaround's stated mechanism does not
>   survive reading the code it calls. Needs the #52709 author.
> - **Finding 8** — a pre-existing `topk_xl` -> `eltwise_binary` reconfig escape. Unrelated
>   to the promotions, but it needs an owner and it will confuse OPEN #4.
> - **Finding 9 (NEW, 2026-08-18)** — `mul_reduce_scalar` re-entry needs a DEST-section
>   boundary. This is the located cause of the reverted `mul_reduce_scalar_chunked_tile`
>   driver, i.e. a defect in a shipping op. Needs an owner. Tracked as C4.
>
> **Updated 2026-08-18:** a second session added the A3 and A6 items below, closed the #53130
> review comments, and turned the old open A4 investigation into Finding 9.
>
> Branch: `ldjurovic/llk-tests-blaze-promotions` (tt-metal). #52709 merged 2026-08-14 and the
> branch was rebased onto main; #52713 and #52727 are **still open** as of 2026-08-18, so it
> still carries their promotion payload.

**PRs covered:** tt-metal #52747, #52745, #52713, #52727, #52709

---

## Summary

| | |
|---|---|
| Verification tier (V1-V4) | 4 of 4, all green |
| New test items landed | **8** — the original 5 (`add_rsqrt`, `custom_mm` `block_uninit`, sort-header coexistence, sampling Prgm0 hazard, rmsnorm bcast-scalar dest-reuse) plus 3 from 2026-08-18 (`set_dst_write_addr_offset` behaviour, compressed metadata-word boundary, `mul_reduce_scalar` re-entry) |
| Test results | **235 new variants passing / 13 xfailed** (42 + 15 + 2 + 12 + 114 + 14 + 6 + 36; xfails are 1 W-stride + 12 re-entry) |
| Files | 12 added (6 `tests/sources/*.cpp`, 6 `tests/python_tests/test_*.py`) + 3 extended (`sfpu_sampling_test.cpp`, `test_sfpu_sampling.py`, `test_matmul_custom_compressed.py`) + 2 LLK headers fixed to compile + 3 template params added |
| Product findings | **2 defects** (both need a decision) + 1 pre-existing reconfig escape + 1 unresolved intermittent + 8 behavioural constraints |

### Landed tests

| Item | Files | Result |
|------|-------|--------|
| `add_rsqrt` SFPU functor (#52709) | `tests/sources/sfpu_add_rsqrt_test.cpp`, `tests/python_tests/test_sfpu_add_rsqrt.py` | 42 passed, 14 skipped |
| `custom_mm`/`compressed_custom_mm` `block_uninit` (#52727) | `tests/sources/custom_mm_uninit_restore_test.cpp`, `tests/python_tests/test_custom_mm_uninit_restore.py` | 15 passed, 1 xfailed, 16 skipped |
| Sort-header coexistence (#52713) | `tests/sources/sort_headers_coexist_test.cpp`, `tests/python_tests/test_sort_headers_coexist.py` | 2 passed |
| **rmsnorm bcast-scalar dest-reuse (#52709)** | `tests/sources/rmsnorm_bcast_scalar_dest_reuse_test.cpp`, `tests/python_tests/test_rmsnorm_bcast_scalar_dest_reuse.py` | **114 passed, 114 skipped** (was 66; widened 2026-08-18, Finding 11) |
| Sampling `vConstFloatPrgm0` hazard (#52745) | `tests/sources/sfpu_sampling_test.cpp`, `tests/python_tests/test_sfpu_sampling.py` (extended) | 63 passed, 97 skipped (was 51 passed) |
| **`set_dst_write_addr_offset` behaviour (#52713)** — 2026-08-18 | `tests/sources/set_dst_write_addr_offset_test.cpp`, `tests/python_tests/test_set_dst_write_addr_offset.py` | **14 passed, 14 skipped** |
| **Compressed metadata-word boundary (#52727)** — 2026-08-18 | `tests/python_tests/test_matmul_custom_compressed.py` (extended) | **6 passed** (suite 582 -> 588) |
| **`mul_reduce_scalar` re-entry (#52709)** — 2026-08-18 | `tests/sources/mul_reduce_scalar_reenter_test.cpp`, `tests/python_tests/test_mul_reduce_scalar_reenter.py` | **36 passed, 12 xfailed** (Finding 9) |

### Verification tier — all green on the merged branch

| Suite | For | Result |
|---|---|---|
| `test_matmul_custom_compressed.py` | V1 / #52727 | 583 passed; **588 after the 2026-08-18 boundary test**, but see Finding 12 — one unreproduced failure in three runs |
| `test_topk_xl.py` | V2 / #52713 | 71 passed |
| `test_sfpu_sampling.py` | V3 / #52745 | 51 passed, 93 skipped as the baseline; **63 passed, 97 skipped** after the hazard test below was added |
| `test_generalized_moe_gate.py` | V4 / #52747 | 89 passed |
| `test_sfpu_generic_moe_gate_topk.py` | V4 / #52747 | 24 passed |
| `test_eltwise_binary.py` | regression baseline | 4388 passed, 72 skipped |

V3 and V4 confirm the verdict below that #52745 and #52747 need no new tt-llk tests: the
canonical targets they rewire onto are already fully covered.

---

## 1. Cleanup PRs #52747 / #52745 — checked and verified

Neither PR adds anything under an `experimental/` path, so neither *required* a new
test, and both canonical targets were verified green. #52745 later gained one anyway —
not for the promotion itself but to pin the cross-op hazard that motivated it (section
5).

### #52747 — Retire the demo `deepseek_moe_gate` fork onto canonical `generalized_moe_gate`

Adds **nothing** under any `experimental/` path. 13 headers deleted from the demo shadow tree; the only
non-deletion is `unified_kernels/deepseek_moe_gate.hpp` re-pointing at
`api/compute/experimental/generalized_moe_gate.h` with `GMG_UNGROUPED_TOP8 = 0`.

The canonical target is already the best-covered family in the tt-llk suite:

- `test_generalized_moe_gate.py` / `sources/generalized_moe_gate_test.cpp` — 12 test functions, including
  the **grouped** DeepSeek path that `GMG_UNGROUPED_TOP8 = 0` selects: `test_generalized_moe_gate_grouped`,
  `test_generalized_moe_gate_sigmoid[grouped=True]` (the sigmoid + grouped combination is the DeepSeek gate
  exactly), `test_generalized_moe_gate_ties`, `test_generalized_moe_gate_shipping_config`.
- `test_sfpu_generic_moe_gate_topk.py` — the SFPU top-k functors underneath.

**Verdict: no new test.** Run `test_generalized_moe_gate.py` and `test_sfpu_generic_moe_gate_topk.py`
unchanged on the branch as the regression gate.

**One thing worth flagging to the author (not a blocker):** tt-llk's own `test_deepseek_moe_gate.py` /
`sources/deepseek_moe_gate_test.cpp` do **not** consume the tree this PR deletes. They include a *third*
fork living under `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/deepseek_moe_gate/device/kernel_includes/`,
reachable via the `-I../../../ttnn/cpp/ttnn/operations/experimental` entry in
`tests/python_tests/helpers/test_config.py` (carrying the TODO "remove this after kernels get moved into
Metal experimental (#52837)"). That is the fork the PR body defers to a later batch — and it is the one
whose retirement *will* require rewiring a tt-llk test source. Queue that as a known follow-up.

### #52745 — Retire the demo fork of `ckernel_sfpu_sampling.h`

Adds nothing under `experimental/`. The canonical
`hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sampling.h` landed with its suite in
#52163; this PR deletes the demo copy and rewires `unified_kernels/sampling.hpp`.

`test_sfpu_sampling.py` + `sources/sfpu_sampling_test.cpp` already cover **every entry point**, including the
two things this PR's call-site changes touch:

- `sampling_recip_init<legacy_compat>` — called in the driver's math thread, swept both ways
  via `legacy_compat=[True, False]`.
- `calculate_sampling_binary_first_column<SamplingBinaryOp::{add,sub,mul}>` — the collapsed
  dispatch, driven from the driver's `run_sampling_op()`.

  (Line numbers deliberately omitted: both files were edited when the hazard test landed,
  and stale citations are worse than none.)

**Verdict for the promotion itself: no new test needed.** `test_sfpu_sampling.py` passes
unchanged (51 passed, 93 skipped) and that is the regression gate for the rewiring.

**A test was added on top**, for a different reason: the sweep proved the init *works* but
never that it is *required*, which is the hazard #52745 exists to fix. See section 5, and
Finding 4 for why that hazard was invisible to the existing tolerances.


---

## 2. `add_rsqrt` (#52709) — DONE

> **What landed:** `tests/python_tests/test_sfpu_add_rsqrt.py` +
> `tests/sources/sfpu_add_rsqrt_test.cpp`, 42 passed / 14 skipped on BH p100a.
>
> Deviation from the recommendation below: a **dedicated file**, not an extension of
> `test_sfpu_binop_scalar.py`. `calculate_add_rsqrt` carries two template axes that suite
> has no notion of (`APPROX` selecting the sqrt body, `FAST_APPROX` gating the negative
> guard), and it lives in the metal `experimental/llk_sfpu/` tree, which needs the
> `#define DST_ACCUM_MODE` / `constexpr bool APPROX` preamble. `test_sfpu_sampling.py` is
> the exact precedent — a dedicated file for an `experimental/llk_sfpu` header — so that
> shape was followed instead. `SFPU_UNARY_SCALAR` is still reused for the addend.
>
> Two further departures, both forced by measurement:
> * The `FAST_APPROX` case asserts a **sign** predicate, not `isnan`. The guard's NaN
>   arrives as `+inf` on the Float16_b path, so an isnan assertion fails while the guard
>   works correctly. What holds in all six live configurations is: guard on → no negative
>   lane, guard off → negative lanes present.
> * Tolerances are measured envelopes per (body, output width), 1e-6 … 2.0e-2, replacing
>   the "loosen for approx" sketch below — which as written would have *tightened* the
>   bf16 cases 25x below the format default and failed them.

`calculate_add_rsqrt<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, FAST_APPROX>(uint32_t param0)` is a
unary SFPU op with one bit-packed float scalar — exactly the shape of the
`test_sfpu_binop_scalar.py` / `sources/sfpu_binop_scalar_test.cpp` suite, which already uses
`SFPU_UNARY_SCALAR(scalar_bits)` and a `_bits()` host helper and has precedent for a host-transformed
scalar (`ScalarDiv` inverts on the host).

Work items:
1. `helpers/llk_params.py` — add `MathOperation.AddRsqrt`.
2. `sources/sfpu_binop_scalar_test.cpp` — add an `#elif defined(SFPU_OP_ADD_RSQRT)` arm including
   `experimental/llk_sfpu/ckernel_sfpu_add_rsqrt.h` and calling `init_add_rsqrt<APPROX>()` +
   `calculate_add_rsqrt<...>` through `_llk_math_eltwise_unary_sfpu_params_`.
3. `helpers/golden_generators.py` — golden is `torch.rsqrt(x + eps)` in fp32, then
   `round_to_dest_width` for the `!fp32_dest_acc_en` `convert<vFloat16b>(Nearest)` store.
4. Add to `_SCALAR_OPS`.

Sweep axes: `eps ∈ {0.0, 1e-6, 1.0}` (`1e-6` is the production RMSNorm epsilon; `0.0` cross-checks against
the plain `MathOperation.Rsqrt` result in `test_sfpu_unary.py`), `dest_acc ∈ {No, Yes}` (drives both the
`ITERATIONS` count and the truncation branch), `APPROX ∈ {False, True}`, `FAST_APPROX ∈ {False, True}`.

**Domain note.** The binop-scalar suite does not consume `helpers/sfpu_domains.py` — it falls back to
`default_spec_for_format` = `uniform(0.1, 1.1)`, i.e. positive-only. That is fine as the default (it keeps
`x + eps > 0`), but pass an explicit `spec_A` for two extra cases worth having: `x` near `0` with `eps = 0`
(result → `+inf`, assert the inf rather than a tolerance) and large `x` (~1e4, exercises the
`_calculate_sqrt_body_` exponent path). Do **not** feed negatives — `rsqrt` of a negative is undefined for
this functor and would only test garbage.

---

## 3. The `set_dst_write_addr_offset` extraction (#52713) — DONE

The PR's stated reason for the new shared header is that `ckernel_sfpu_topk_xl.h` already defined an
identical `set_dst_write_addr_offset`, so a kernel including both headers would hit a redefinition error
(blaze papers over it with `#ifndef` guards). **Nothing in the tree compiles both headers into the same
TRISC-math translation unit**, so the redefinition this PR fixes is currently unreachable by any test.

Add a variant to `sources/top32_rm_test.cpp` (or a dedicated compile-only case) that includes **both**
`sfpu/experimental/ckernel_sfpu_topk_xl.h` and `sfpu/experimental/ckernel_sfpu_deepseek_top32_rm.h` in the
math TU and calls one entry point from each. Cheap, and it is the only thing that actually pins the
extraction. Also note the PR's own observation: the shared copy keeps the demo's `LLK_ASSERT`, so `topk_xl`
callers gain a Dst-offset bounds check — run this case with `ENABLE_LLK_ASSERT` set to exercise it, and pass
an out-of-range offset in a negative variant if the harness supports expected-assert tests.

Plus: run `test_topk_xl.py` unchanged on the branch. It is the direct regression check for the header edit.

**Landed as `sort_headers_coexist_test.cpp` + `test_sort_headers_coexist.py` — 2 passed**, and
review corrected its scope. The original plan asserted that a datacopy after
`set_dst_write_addr_offset` proves the helper leaves no dirty Dst offset. It does not:
`_llk_math_eltwise_unary_datacopy_` itself calls
`math::set_dst_write_addr<Tile32x32, SrcRegs>(dst_index)` — the same
`DEST_TARGET_REG_CFG_MATH_Offset_ADDR32` the helper writes — **before** anything touches
DEST, so whatever the helper left is discarded and the copy lands identically whether the
helper is correct or deleted.

The 0/2/64 offset sweep and its `SORT_DST_WRITE_OFFSET` template parameter were therefore
removed; one call is kept so the helper is still code-generated, and `dest_acc` is swept
instead as the one axis that changes what the combined TU builds. **The compile-time
coexistence assertion is the real content of this test**, which is what the PR claims.
Observing the offset's effect needs a DEST consumer that does not reprogram that register
first — relevant to OPEN #4.

---

## 4. `custom_mm` / `compressed_custom_mm` `block_uninit` (#52727) — DONE

**This is the highest-value new test in the whole batch.** It is the only behavioral delta in #52727 and it
has zero coverage at any layer.

`custom_mm_block_uninit<dense_packing, restore_tile_pack_mop>` does two conditional state restores:
- `dense_packing` → `cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(...)` back to the 64-row stride;
- `restore_tile_pack_mop` → `_llk_pack_mop_config_<PackMode::Default>()`.

Both are classic cross-op state leaks, and tt-llk has an established pattern for exactly this shape:
`test_unpack_tilize_uninit_restore.py`, `..._block.py`, `..._tiny.py`, `test_unpack_bcastA_B_uninit_restore.py`.
Follow that pattern literally — its docstring even spells out the discipline ("NO existing test calls this
function at all").

Kernel shape:

```
run 0: custom_mm block, packer MOP replaced by pack_block_contiguous_init (± dense_packing stride)
       custom_mm_block_uninit<dense_packing, restore_tile_pack_mop>()
run 1: plain _llk_pack_<..., PackMode::Default> datacopy of a known tile, NO packer re-init
```

Assertion matrix — note that both polarities are assertions, not just the happy path:

| `restore_tile_pack_mop` | run-1 expectation |
|------------------------|-------------------|
| `true` | matches the datacopy golden — the restore works |
| `false` | **differs** from the datacopy golden — pins the documented "the MOP is owned by whichever init programmed it" contract, so a future accidental unconditional restore is caught |

Cross with `dense_packing ∈ {False, True}` to cover the `Wstride` RMW in the same kernel (same failure
shape, different config register — a `dense_packing` block followed by an unrestored default pack writes
tiles 32 rows apart instead of 64). Run the identical matrix for
`compressed_custom_mm_block_uninit<dense_packing, restore_tile_pack_mop>`.

> **Second discrepancy to raise with the author.** The PR body describes the fix as unconditional:
> "`*_block_uninit()` now restores the Default tile-pack MOP." At the current head it is **opt-in** —
> `template <bool dense_packing = false, bool restore_tile_pack_mop = false>`, defaulting to `false`, with a
> comment explaining that an unconditional `_llk_pack_mop_config_<Default>()` would install fixed 32×32
> geometry and clobber the 1×32 configuration this family targets. So the "all ten demo `*_block_uninit`
> callers are exercised" claim in the notes is about the *old* behavior unless those call sites were switched
> to `<..., true>`. Worth confirming which callers opt in — and it is a good argument for testing both
> polarities as above rather than assuming the restore is always on.

**Landed at 15 passed / 1 xfailed / 16 skipped**, half the original 30/2/32. Review found the
`family` axis (`custom_mm` vs `compressed_custom_mm`) never reached the build: `_run` had no
`family` parameter and `CUSTOM_MM_UNINIT` no family field, so both values produced the same
`variant_id`, the ELF was reused, and every surviving case simply ran twice on hardware with
three assertion f-strings differing.

The rationale was also backwards. The driver **replicates** the uninit body rather than
calling `custom_mm_block_uninit` / `compressed_custom_mm_block_uninit` — a tt-llk test cannot
include `tt_metal/hw/inc/api/compute` — so a future divergence between the two headers is
exactly what the axis could not catch: both ids would keep passing. Guarding that divergence
needs a test on the metal side that calls the real entry points; **that gap is still open and
unowned.**

The `restore_tile_pack_mop` opt-in question above was raised again in review and deliberately
left as-is: the flag defaults to `false`, nothing in the tree opts in, and switching the
family to a clean-state-on-entry contract is an API change that does not belong in a test PR.
Both polarities and the inert-at-matching-geometry case are pinned, which is what a future
contract change will need.

---

## 5. Sampling `vConstFloatPrgm0` cross-op hazard (#52745) — DONE

> **What landed:** two switches in `tests/sources/sfpu_sampling_test.cpp` plus
> `test_sfpu_sampling_recip_prgm0_hazard` in `tests/python_tests/test_sfpu_sampling.py`.
> Suite went from 51 passed to **63 passed, 97 skipped** on BH p100a.

The existing sweep covered every entry point but always called `sampling_recip_init`
immediately before the op, so it proved the init *works* and never that it is *needed*.
#52745's motivation is the opposite direction: the `legacy_compat=false` reciprocal reads
`vConstFloatPrgm0` as its Newton-Raphson constant, and only `sfpu_reciprocal_init` writes
the `2.0f` it expects, so a kernel that ran another `Prgm0`-owning op earlier is silently
wrong.

The driver now takes `SAMPLING_POLLUTE_PRGM0` (runs `log_init` first, standing in for that
earlier op) and `SAMPLING_SKIP_RECIP_INIT` (drops the repair). `log_init` is chosen because
it sets `Prgm0` to `LOG_TWO * 2^-23` — about nine orders of magnitude from `2.0f`.

Asserted matrix, with the polluter always active:

| `legacy_compat` | `recip_init` | expectation |
|---|---|---|
| true | either | correct — the legacy reciprocal never reads `Prgm0`, so it must be immune |
| false | called | correct — this is what the init exists for |
| false | skipped | **wrong** — the init is load-bearing |

The `(pollute=false, skip_init=true)` cell is deliberately not swept: with neither a
polluter nor an init, `Prgm0` holds whatever the invariant LLK SFPU init left, which is not
a defined value and so not assertable either way.

The test asserts on a hazard-specific **1e-3** threshold rather than the suite's 2%
reciprocal tolerance — **Finding 4** in section 6 explains why, and what that threshold
still cannot distinguish.

---

## 6. rmsnorm bcast-scalar dest-reuse (#52709) — DONE

`tests/sources/rmsnorm_bcast_scalar_dest_reuse_test.cpp` +
`tests/python_tests/test_rmsnorm_bcast_scalar_dest_reuse.py` — **66 passed, 66 skipped** on
BH p100a. Was OPEN item 2 in the strategy document.

The op is a `num_tiles`-templated MOP driven from a **single** unpack call, with SrcB
sourced from DEST via `MOVD2B` under a `WAIT_SFPU | SRCB_VLD` stall rather than from L1. No
existing test file has that shape, which is why it is a new driver rather than an extension:
`test_bcast.py` does one-tile-per-unpack broadcasts and `test_eltwise_binary.py` has neither
the `num_tiles`-as-template-argument plumbing nor a MOP-over-N-tiles axis.

### Driver shape

1. a plain A2D datacopy seeds DEST[0]; element [0] of that tile is what `MOVD2B` broadcasts,
   standing in for the `1/RMS` that `add_rsqrt` produces in the real kernel;
2. `_llk_math_rmsnorm_bcast_scalar_dest_reuse_` runs with `src_index == dst_index == 0`,
   which is exactly how `unified_kernels/rmsnorm.hpp:146` calls it;
3. `num_tiles` tiles are packed out.

The seed tile is deliberately non-uniform so a `MOVD2B` that picked up the wrong row or face
fails rather than silently agreeing with the golden.

### Sweep as built

| axis | values | note |
|---|---|---|
| `eltwise_binary_type` | `ELWADD`, `ELWMUL` | `ELWSUB` is accepted by the MOP but the compute API never instantiates it |
| `num_tiles` | 1, 2, 3, 4 | 3 is the odd count that catches an off-by-one in `num_tiles * num_faces / 2` |
| `math_fidelity` | LoFi, HiFi2, HiFi4 — **ELWMUL only** | the ELWADD MOP branch never reads the template argument, so sweeping it there builds identical ELFs |
| `clear_dest` | both for ELWADD; True in the main matrix for ELWMUL | see Finding 6 — the mul accumulates, so its False half is asserted against an accumulating golden in its own test |
| `num_faces` | 4 in the main matrix; 1 and 2 in a dedicated test | at <4 the pack still emits a full 4-face tile, so those variants are about the uncovered tail |
| `dest_acc` / formats | Float16_b/No, Float32/Yes | |
| `unpack_full_transpose` | False, True | the axis that exists only because blaze's version won the reconciliation — **it lands and passes** |

### Deviations from the original plan, and why

- **`unpack_to_dest` is off for every variant.** The `static_assert` in
  `_llk_unpack_A_rmsnorm_mop_config_` rejects it for this configuration
  (SCALAR + `acc_to_dest` + `DEST_TO_SRCB`). Driving only the *seed* unpack through it while
  the op itself cannot corrupts the fp32 result into alternating datums — correct values at
  odd indices, garbage at even. Worth recognising that signature.
- **No `RmsnormBcastScalarGolden` generator was added.** The op is elementwise against one
  broadcast scalar, so the golden is a few lines in the test file. It does reuse
  `EltwiseBinaryGolden._apply_fidelity_masking` for the ELWMUL phases — see Finding 6.

---

## 7. Findings from building these tests

### Finding 1 — `restore_tile_pack_mop` is only observable at mismatched tile geometry

§5.2 assumed a leftover block-contiguous MOP would corrupt any following
`_llk_pack_<Default>`. It does not. Measured: with the run-0 block MOP programmed at **4
faces** — the same geometry the restore installs — run 1 is byte-correct whether or not
the restore runs, so the flag is **unobservable**. The restore re-establishes *geometry*,
nothing broader.

Programming the run-0 MOP at **2 faces** (a 16x32 tiny tile) makes it observable, and is
the hazard the header's own comment names ("wrong for 1x32 follow-ons"): the un-restored
MOP then packs half of each tile, a 0.50 per-tile match. The landed test uses 2 faces for
the discriminating cases and keeps a 4-face test that pins the no-op-at-matching-geometry
behaviour, since that is *why* the flag is opt-in rather than unconditional.

### Finding 2 — DEFECT (needs an owner decision): the `dense_packing` W-stride is not format-aware

Found while building P0-1, and the most substantive result so far.

`cpack_common.h set_packer_strides` — the canonical writer — computes

```
w_stride = TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * datum_size_in_bytes(pack_src_format)
```

but both `custom_mm.h` and `compressed_custom_mm.h` spell the same expression with a
literal `* 2`, i.e. hardcoded for a **16-bit** pack source, at four sites (init + uninit
in each family):

```
init   dense:   (TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2  = 1024
uninit restore:  TILE_NUM_FACES      * FACE_C_DIM * FACE_R_DIM * 2  = 2048
```

With a Float32 pack source `datum_size_in_bytes` is 4, so the correct values are 2048 and
4096. The uninit therefore does not restore what `_llk_pack_init_` programmed. Measured
(Float32 in/out, `dest_acc=Yes`, `dense_packing=True`): run 1 matches on tile 0 only
(0.25 overall) **regardless of `restore_tile_pack_mop`** — the W-stride restore cannot
recover. The 16-bit path is fully correct.

Pre-existing rather than introduced by #52727 (the demo's structure was kept), but the
promotion ships it in packaged metalium via `HW_JIT_API_HEADERS`, widening the blast
radius. Landed as an `xfail` with the full explanation, so the suite stays green and
flips to XPASS when the constants become format-aware — or when a `static_assert`
restricts `dense_packing` to 16-bit pack sources, if that is the intended contract.
**This needs an owner decision; it is the one item here that is a product question, not a
test question.**

### Finding 3 — the two sort families cannot both be initialized in one kernel

§4.3 proposed calling an entry point from each header to prove they coexist. Compiling
both in one TU works (the extraction is sound — that is the assertion worth having), but
calling `_top32_rm_init_()` and `_topk_xl_init_<K, fused>()` in the same kernel **hangs
the math thread**: both program overlapping ADDR_MOD slots, the MOP and the REPLAY
buffer. Not a defect, and no real kernel does it, but it bounds the claim — the PR's
guarantee is translation-unit coexistence, not simultaneous liveness. Anyone fusing the
two families must re-init between them. The landed test therefore calls only the shared
helper, with the constraint documented in the driver.


### Finding 4 — why this hazard was invisible, and what it actually costs

The pollution does **not** produce garbage. `Prgm0` becomes ~8.3e-8 instead of `2.0f`, so
`t = x * y - Prgm0` comes out *positive* (`x * y` is ~1 since `y ≈ 1/x`), the
`v_if(t < 0)` refinement inside `sfpu_reciprocal_iter` never fires, and the raw
`approx_recip` result survives unrefined. Max relative error vs golden, measured:

| output / `dest_acc` | `recip_init` called | `recip_init` skipped |
|---|---|---|
| Float16_b / No | 0.0 | 6.2e-03 |
| Float32 / Yes | 1.1e-07 | 5.0e-03 |
| Float16_b / Yes | 2.3e-03 | 5.0e-03 |

So the cost is about **1e-3 relative** — real, but comfortably inside the suite's 2%
`RECIP_REL_TOL`, which is why no existing variant could have caught it. The test therefore
asserts on a hazard-specific **1e-3** threshold rather than the suite tolerance.

The third row is excluded from the strict check: there the packer's own fp32->bf16
conversion already costs 2.3e-3, only 2.2x from the unrefined error, so a strict assertion
would be measuring the packer rather than the hazard. Worth knowing if anyone tightens
`RECIP_REL_TOL` later — bf16 output with an fp32 DEST cannot distinguish the two.


### Finding 5 — the rmsnorm LLK headers did not compile under the tt-llk build

`llk_unpack_A_rmsnorm.h` and `llk_math_rmsnorm_bcast_scalar_dest_reuse.h` were promoted by
#52709 with metal-side consumers only. tt-llk builds with `-Werror`, and both headers fail
it outright on dead locals — so **no tt-llk test for this family could exist at all** until
they were fixed:

| Header | Dead symbols |
|---|---|
| `llk_unpack_A_rmsnorm.h` | 8 unreachable `UNPACR` / `SETADCZW` constants carried over from `llk_unpack_A.h`'s general `mop_config`, plus 2 unused format parameters the signature must keep |
| `llk_math_rmsnorm_bcast_scalar_dest_reuse.h` | `addr_mod`, `innerloop`, `outerloop`, `ZERO_ACC_MODE` — all superseded by values passed straight to `ckernel_template` or chosen per `DstSync` branch |

The unpack constants are unreachable *by construction*: the `static_assert`s pin that MOP to
SCALAR + `acc_to_dest` + `DEST_TO_SRCB`, so SrcB is never unpacked from L1 and the
unpack-to-dest opcodes cannot fire. Removal is behaviour-preserving — none was ever read.

**Generalise this:** expect a build fix as a prerequisite for any promoted header tt-llk
covers for the first time. It is part of why the ~2 d estimate on this item was optimistic,
and OPEN #3 / OPEN #4 should budget for it.

### Finding 6 — ELWMUL accumulates into DEST; ELWADD overwrites it

Both branches of `rmsnorm_bcast_scalar_dest_reuse_configure_mop` pass 0 in the instruction's
dest-accumulate slot — `TT_OP_ELWADD(0, acc_to_dest, ...)` with `acc_to_dest == 0`, and
`TT_OP_ELWMUL(0, 0, ...)` — so the two read as though they behave alike. Measured on BH
p100a with the `ZEROACC` suppressed (`clear_dest=False`), they do not:

| op | result with a seeded DEST |
|---|---|
| ELWADD | `A + scalar` — the seed is discarded |
| ELWMUL | `seed + A*scalar` — the seed is accumulated onto |

This is the contract `unified_kernels/rmsnorm.hpp:146` depends on when it passes
`clear_dest=true` to the mul, and it had no test. Both directions are now asserted, the add
case as an explicit negative control so the asymmetry is pinned as a property of the MOP
branch rather than of the driver.

**Do not assume a 0 in that slot means overwrite.**

Two corollaries for the remaining work:

- **`clear_dest` is a correctness requirement for ELWMUL, not a preference.** Any future
  caller of the mul path that omits it inherits whatever DEST held.
- **LoFi ELWMUL costs a few percent, not a few ULPs.** On a `uniform(-4, 4)` sweep the LoFi
  mantissa masking produced ~3% relative error, and it scales with the operand magnitude.
  Model it per phase with `EltwiseBinaryGolden._apply_fidelity_masking` the way
  `test_eltwise_binary.py` does, rather than widening `rtol` — a tolerance loose enough to
  absorb LoFi is loose enough to miss a real regression.

### Finding 7 — the `eltwise_mul_scalar` HiFi workaround's stated mechanism does not hold

Raised during review of the test PR, against `api/compute/experimental/eltwise_mul_scalar.h`
(now in main). The `deepseek_binary_dest_reuse_tiles_init` HiFi branch calls
`_llk_math_eltwise_binary_init_` with a hardcoded `DEFAULT_TENSOR_SHAPE`, and its comment
attributes a HiFi4 correctness fix to the shorthand init "mis-specialising the tile shape".
Reading the code it calls:

- `get_effective_math_fidelity<ELWMUL, f>()` is the **identity** for ELWMUL
  (`llk_math_common_api.h:123-125`), so the fidelity gate cannot be the difference;
- `acc_to_dest` is 0 in both arms;
- the shorthand resolves the shape as `get_operand_tensor_shape(operand_A)` regardless of
  fidelity (`llk_math_binary_api.h:31-42`).

So tensor shape is the *only* thing separating the two arms — and on a standard 4-face
32x32 CB `get_operand_tensor_shape` returns exactly `DEFAULT_TENSOR_SHAPE`, making the HiFi
arm bit-identical to the shorthand it replaces. Meanwhile the paired execute
(`deepseek_binary_dest_reuse_tiles`) uses the 4-arg overload, which *does* derive the shape
from the CB — so on any non-default geometry init and execute disagree.

Combined with the measured fact in §12.4 of the strategy document — forcing
`DEFAULT_TENSOR_SHAPE` on a 2-face tile deadlocks the MATH_PACK handshake rather than
corrupting the result — the workaround is either inert (4-face CB) or hangs (2-face CB).
There is no configuration in which it does what its comment says.

The failing config #52709 cites is `gated_local_reduce` at HiFi4 (M2 MoE HiFi4,
0.70 -> 0.9996). **That measurement is not explained by the stated mechanism**, so either
the comment needs correcting or the real cause is still unidentified. Flagged to the #52709
author; relevant to OPEN item 5.

### Finding 8 — a pre-existing reconfig escape between `topk_xl` and `eltwise_binary`

Not caused by any promotion — it reproduces on a clean checkout of main with every
blaze-promotion change stashed — but it will waste time for whoever picks up OPEN #4, since
`top32_rm` and `topk_xl` share the sort headers and a new failure there will look
self-inflicted.

```
pytest test_eltwise_binary.py   (alone)  -> 4388 passed, 72 skipped
pytest test_topk_xl.py          (alone)  ->   71 passed
pytest test_topk_xl.py <target>          ->    1 failed
```

where `<target>` is
`test_eltwise_binary[dest_acc:No-formats:Bfp4_b->Bfp4_b-broadcast_type:None_-math_op:Elwmul-math_fidelity:LoFi-transpose_srca:Yes-input_dimensions:[256, 32]-tile_dimensions:[32, 32]]`.

Per the tt-llk notes a reconfig escape is a real bug rather than a test-ordering nuisance,
and `tt-smi -r` must not be used to paper over it. **Needs an owner.**

### Finding 9 — DEFECT (needs an owner): `mul_reduce_scalar` re-entry needs a DEST-section boundary

Added 2026-08-18. This is the located cause of the reverted `mul_reduce_scalar_chunked_tile`
driver, and it closes what the open-work document tracked as A4. Tracked for decision as C4.

A ~40-line driver that runs the known-good non-chunked sequence twice over the same input,
re-issuing exactly what the chunked loop re-issues per batch, splits the behaviour cleanly:

| Configuration | Result on BH p100a |
|---|---|
| `passes=1`, either mode (the control) | correct |
| `passes=2`, DEST-section boundary between passes | correct, and **bit-identical** across passes |
| `passes=2`, one shared DEST section | **wrong — all 12 variants, 9.27x to 9.93x golden** |

So the family **is** re-enterable, which the old one-line hypothesis ("not re-enterable") got
wrong. What breaks is re-entry with no `dest_section_done` / `wait_for_dest_available` pair in
between — that handshake restores whatever the second `_llk_math_mul_reduce_scalar_init_` does
not.

And that is exactly the chunked op's structure. `mul_reduce_scalar_chunked_tile`
(`rmsnorm.h:105`) documents that the caller "must ... acquire DST before calling", then
re-enters every batch inside that single section, with `if (batch > 0)
mul_reduce_scalar_init(...)` as its only restoration attempt.

Same signature as the reverted driver — it reported 5-30x golden and "not a clean multiple of
anything"; this reproduces 9.3-9.9x, also non-integer — so very likely the same defect, now
with a minimal reproducer instead of a full chunked implementation.

This also explains why the two earlier fix attempts (the accumulator fill, and a missing
UNPACK/MATH barrier) both left the output **byte-identical**: neither touched the DEST-section
boundary, which is the variable that actually matters. Do not re-investigate either.

The 12 failing variants are `xfail` (marker form, so the body runs) and flip to XPASS the
moment re-entry inside one section restores state. The fix belongs in the LLK, or in the
compute API if the answer is that the chunked op must close its section per batch.

### Finding 10 — an `xfail` that could never have reported XPASS

`test_custom_mm_uninit_restore.py` recorded the Finding 2 W-stride defect with imperative
`pytest.xfail()`, which raises immediately and aborts the test body. So the `dense_packing`
fp32 variant never built, never ran, and **could never report XPASS** — functionally a `skip`
with a different label, while the surrounding comment promised it would "flip to XPASS the
moment the constants become format-aware". Finding 2's owner decision was resting on a detector
that was not armed.

Fixed 2026-08-18 by attaching the marker instead
(`request.node.add_marker(pytest.mark.xfail(reason=..., strict=False))`), which is what the rest
of the suite already did — `test_sfpu_unary.py`, `test_sfpu_binary.py` and
`test_sfpu_reduce.py` all use the marker form; this file was the only outlier. The run now
prints a real golden-vs-device comparison under XFAIL rather than a bare skip, which is how you
can tell the difference at a glance.

Note `pytest.param(..., marks=...)` is **not** available as an alternative here: the local
`parametrize()` helper builds raw tuples and calls `.name`/`str()` on each value, so it does not
unwrap a `ParameterSet`.

### Finding 11 — the rmsnorm partial-faces addr_mod skips to the next tile base

Established while widening the partial-faces test on 2026-08-18 (66 -> 114 variants). The
addr_mod that `_llk_math_rmsnorm_bcast_scalar_dest_reuse_init_` programs increments DEST by
`8 + (4 - num_faces) * 16`: after a tile's covered faces it **skips the uncovered ones and
lands on the next tile's base**. The unpack side, by contrast, reads `num_tiles * num_faces`
faces **contiguously** out of L1.

So for `num_faces < 4` the k-th face of the input goes to tile k's leading face-slot, and the
golden must slice the input contiguously while indexing the device output per tile. Assuming one
contiguous output run — the obvious reading — produces a wrong golden.

`num_tiles` is what pins the skip term: at `num_tiles=1` it is unobservable, since there is no
second tile for a wrong stride to land in. Halving it to `8 + (4 - num_faces) * 8` fails every
`num_tiles=2` variant and **no** `num_tiles=1` variant.

Two smaller facts from the same pass, both previously unswept anywhere in the suite:

- **HiFi3 was never exercised**, though `_FIDELITY_PHASES` defined it. It is the only fidelity
  whose phase count is not a power of two (3), so an implementation deriving the loop bound by
  shifting rather than from the table would pass LoFi/HiFi2/HiFi4 and fail only there.
- **The transpose-fold path ran at LoFi only.** For ELWMUL the replay buffer decides which face
  lands in which SrcA bank while each fidelity phase masks a different mantissa slice, and
  nothing ran the two together.

### Finding 12 — `test_matmul_custom_compressed.py` is not deterministic (unresolved)

On 2026-08-18, three runs of the same suite on the same commit gave 588 passed, then **587
passed / 1 failed**, then 588 passed. The failing variant was **not identified** — the run was
backgrounded through a `grep` that discarded the detail, and it has not recurred.

Not caused by the 2026-08-18 changes: the 6 new metadata-boundary variants passed in all three
runs, and nothing else in that suite's compile path was touched. Finding 8 already records a
pre-existing order-dependent reconfig escape in a neighbouring area, which is the obvious
suspect but is unconfirmed.

Recorded so a single green run of this suite is not read as proof. If it fires again, capture
the full log; the `perturb` skill exists for this shape of problem.

### Finding 13 — a test that passes first try has not been shown to test anything

Method note rather than a product finding, but it changed two conclusions on 2026-08-18, so it
is worth stating. Every test added that day passed on its first hardware run, and in every case
a deliberate mutation was what established it was not vacuous:

| Test | Mutation | Result |
|---|---|---|
| `set_dst_write_addr_offset` | helper discards its argument | 10 of 14 fail |
| `set_dst_write_addr_offset` | rows read as datums (`addr * 32`) | sub-tile spill check fires |
| rmsnorm partial faces | addr_mod skip term halved | every `num_tiles=2` variant fails |
| compressed metadata boundary | the OOB guard removed | **still passes** |

Two of those changed what the test could claim. The last one showed the boundary test cannot
detect the out-of-bounds read at all — at `rem_iters == 0` the word read past the buffer is
never used, so no golden comparison can see it — and the `set_dst_write_addr_offset` mutation
showed its `tile=0` variants prove nothing about the helper, since offset 0 is a no-op even when
broken. Both limitations are now written into the tests rather than left implied.

---

## 8. Note on tooling (applies to the remaining work too)

Run tests through `tt-llk/.claude/scripts/run_test.sh` (`count` / `compile` / `run`), not
raw pytest. Two gotchas cost time: a `--k` expression containing brackets or commas
mangles the pytest args and surfaces as an opaque xdist worker crash (use `--test-id`, or
a bracket-free `--k`); and `tests/.venv` must be created with
`source ./setup_external_testing_env.sh` — `setup_testing_env.sh` alone only fetches SFPI
and assumes the Docker image's Python environment.


---
