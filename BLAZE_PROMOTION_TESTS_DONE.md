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
> Branch: `ldjurovic/llk-tests-blaze-promotions` (tt-metal). **All three promotion PRs have
> merged** — #52709 on 2026-08-14, #52727 on 2026-08-18, #52713 by 2026-08-20 — and the branch
> has been rebased onto main, so it no longer carries their payload.
>
> **Updated 2026-08-20:** #52713's merge unblocked A2, and the `top32_rm` family went from zero
> coverage to 10 passing variants across both of its modes. Findings 17 and 18 are from that
> work; the narrative is in `REMAINING_WORK.md` § Closed on 2026-08-20.

**PRs covered:** tt-metal #52747, #52745, #52713, #52727, #52709

---

## Summary

| | |
|---|---|
| Verification tier (V1-V4) | 4 of 4, all green |
| New test items landed | **12** — the original 5 (`add_rsqrt`, `custom_mm` `block_uninit`, sort-header coexistence, sampling Prgm0 hazard, rmsnorm bcast-scalar dest-reuse), 3 from 2026-08-18 (`set_dst_write_addr_offset` behaviour, compressed metadata-word boundary, `mul_reduce_scalar` re-entry), the uninit parity guard and plain `custom_mm` from 2026-08-19, and the **`top32_rm` sort family** from 2026-08-20 |
| Test results | **287 new variants passing / 13 xfailed** (42 + 15 + 2 + 12 + 114 + 14 + 6 + 36 + 2 + 32 + 10; xfails are 1 W-stride + 12 re-entry) |
| Files | 14 added (7 `tests/sources/*.cpp`, 7 `tests/python_tests/test_*.py`) + 3 extended (`sfpu_sampling_test.cpp`, `test_sfpu_sampling.py`, `test_matmul_custom_compressed.py`) + **3** LLK headers fixed to compile + 4 template params added |
| Product findings | **4 defects** (all need an owner) + 1 pre-existing reconfig escape + 12 behavioural constraints |

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
| **custom_mm uninit parity guard (#52727)** — 2026-08-18 | `tests/python_tests/test_custom_mm_uninit_parity.py` | **2 passed** — static, device-free; B1's interim guard |
| **plain `custom_mm` matmul (#52727)** — 2026-08-19 | `tests/sources/matmul_custom_mm_test.cpp`, `tests/python_tests/test_matmul_custom_mm.py` | **32 passed**, PCC >= 0.99999 — closes A1's "no coverage at all" |
| **`top32_rm` sort family (#52713)** — 2026-08-20 | `tests/sources/top32_rm_test.cpp`, `tests/python_tests/test_top32_rm.py` | **10 passed** — 8 plain (4 row lengths x 2 `dest_acc`) + 2 pre-sorted 1024/2048; closes A2 |

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

### Finding 12 — DEFECT (needs an owner): `test_matmul_custom_compressed.py` hangs on repeat, host/BRISC desync

First seen as an unidentified single failure, then reproduced and diagnosed the same day. Six
back-to-back runs on BH p100a: runs 1 and 6 clean, runs 2 and 5 **hung** (exit 5, in
`_clustered` and `_interleaved`), run 4 failed 3 with `TTException` in `_single`, run 3 hit a
build-tree race (see below).

It is a **hang, not a mismatch**. `run_test.sh` triage on run 2:

```
Unpacker/Math/Packer mailboxes = 0x0 (KERNEL_STARTED)
TRISC0/1/2  in_reset=True
BRISC       pc=0x368, unchanged (spinning)
BriscCounter=0x118 (280)   host Python counter: 281
```

All three TRISCs in soft reset while BRISC spins one command behind the host — a host↔BRISC
command-protocol desync, not an LLK compute bug. `get_tensix_state` could not then halt BRISC.

**Nightly-only, so the PR gate is unaffected:** every failing variant (`clustered`,
`interleaved`, `single`) is `@pytest.mark.nightly`, and the gate filters `not nightly`.

Caveats worth carrying: back-to-back runs are not how CI runs the suite and may be the
aggravating factor rather than an independent trigger; and run 3's failure was **not** real —
`ld: cannot open output file .../elf/pack.elf` in the new metadata-boundary test, a
`/tmp/tt-llk-build` race left when run 2's hang handler killed the tree mid-compile. Six runs
also left the device at `PcieHangError`, needing `tt-smi -r` — sanctioned here (runtime timeout,
not a reconfig escape), but it means reproducing this is not free.

Probably distinct from Finding 8: that is a golden mismatch under a specific test ordering, this
is a hang.

### Finding 12b — the compressed-matmul intermittency can present as a wrong answer, 2026-08-20

Finding 12 records `test_matmul_custom_compressed.py` hanging on repeat (host/BRISC command
desync, nightly-only). On 2026-08-20 the same suite produced a **value** failure instead:
`shape=(1, 64, 32), formats=('bfp4',)` at **PCC -0.033**, 587/588. It did not reproduce — that
variant passes in isolation (17/17 for `single and bfp4`) and the full suite then passes
588/588 — and it appeared in the first run after a device-contention incident (a stray plain
`pytest` overlapping a `run_test.sh` run, which by itself produced 137 spurious failures and a
`TENSIX TIMED OUT` across two suites).

Worth attaching to Finding 12 rather than filing separately: same suite, same intermittency,
but the symptom is a wrong answer, not a hang. Whoever picks up item F should know the failure
mode has both shapes, because "it hangs" is a much easier thing to look for than "it
occasionally computes garbage".

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

### Finding 14 — DEFECT (needs an owner): #52727 merged without the out-of-bounds guard

The out-of-bounds remainder metadata read Copilot found on #53130 (Finding 12's neighbour, fixed
on this branch) **did not make it into the merge**. Verified 2026-08-19: `rem_iters != 0` does
not appear in main's `llk_unpack_AB_compressed_custom_mm.h`, so the unguarded
`meta_ptr[full_iters]` is live on main.

Reachable inside the documented ranges whenever `kt_dim * ct_dim` is a multiple of 10
(`kt_dim=10, ct_dim=1` is the smallest). Precisely what it costs: at `rem_iters == 0` the
remainder loop never runs, so the word read past the buffer is never used and no golden can see
it — an L1 memory-safety defect, not a wrong-answer one. Fix is a three-line guard; cherry-pick
`54e218ebbce`. Tracked as C5.

### Finding 15 — the merged uninit is not what #52727's promote branch had

Main's `*_block_uninit` restores **only** the `dense_packing` W-stride. Two earlier revisions on
the promote branch went further — one made the tile-pack MOP restore unconditional, the next put
it behind a `restore_tile_pack_mop` flag — and **neither survived review**. Main has no MOP
restore and no flag, and `pack_block_uninit.h` / `pack_block_contiguous_uninit()` are gone from
the compute API entirely, so the area was reshaped rather than just trimmed.

Two consequences worth carrying:

- **D3 is resolved by deletion**, not by the decision it was waiting on. The #53130 reviewer's
  suggestion — pair the fused caller with `pack_block_contiguous_uninit` rather than add a flag
  to the op uninit — effectively won upstream.
- **A replicating test can keep passing after its subject is deleted.**
  `test_custom_mm_uninit_restore.py` replicates the uninit body rather than calling the API, so
  post-merge it went on asserting a MOP restore nothing performs — green, measuring nothing.
  Narrowed in `1d06517c59f`. Note that `test_custom_mm_uninit_parity.py`, written for exactly
  this staleness class, did **not** catch it: it compares the two bodies to each other and the
  W-stride expressions to the headers, and neither changes when a knob the test drives
  disappears upstream. A guard's reach is worth stating as precisely as its claim.

### Finding 16 — the plain custom_mm doc tables overstate ct_dim

From building A1's test (`ed43f6f7b8f`). The tables say `ct_dim` is "any integer from 1 to 16".
The ct output tiles are all live in DEST at once and half-sync holds 8 bf16 tiles, so **ct_dim > 8
is unreachable from a single call** in that configuration; the upper half of the documented range
needs `DstSync::SyncFull` or a caller that splits the block. Swept 1, 3, 7, 8 — 3 and 7 being the
odd middle values the tables never pinned, which was A1's stated open question.

The companion constraint is real and now sourced: `kt_dim` must be **even**, because
`_llk_unpack_AB_custom_mm_run_` issues `TT_MOP(0, (kt_dim / 2) - 1, 0)`. That is where the
tables' "even number from 2 to 256" comes from.

---

### Finding 17 — `llk_math_top32_rm.h` did not compile under the tt-llk build

Third header of the class Finding 5 opened, and the same fix. `_llk_math_top32_rm_init_`'s
`num_faces` and `llk_math_top32_rm_configure_mop`'s `total_rows` are never read — the MOP is a
fixed two-instruction body walking 8 Dest rows per issue via `ADDR_MOD_2` — and the tt-llk test
build compiles with `-Werror=unused-parameter`, so **the first test to include this header
failed to build**. The JIT build does not treat that warning as an error, which is why a
promoted header could sit on main in this state.

Both parameters were dropped rather than suppressed, matching the call the #53130 reviewers
made on `llk_unpack_A_rmsnorm.h` and `llk_unpack_A_top32_rm.h`, and
`llk_math_top32_rm_api.h` — its only caller anywhere in the tree — was updated with it.

What the three instances have in common is worth stating once: **a promoted header that no
in-tree test compiles has not been compiled at all except by the JIT path, under a weaker
warning set.** Promotion reviews cannot see this; the first test can, and does, immediately.

### Finding 18 — the 7 `llk_math_deepseek_top32_rm_*` wrappers are on main with no caller

During #53130's review these seven metal wrappers were dropped from the branch because they
had no caller and no test. They are on main regardless — they arrived with #52713 — and the
recheck on 2026-08-20 says nothing has changed: `git grep` finds no caller anywhere outside
the header itself, and the in-tree consumers (`unified_kernels/sampling.hpp`, both
`top32_rm_dev_compute*.cpp`) still drive the underlying `ckernel::sfpu::_bitonic_top32_*` and
`_top32_rm_init_` primitives directly through `SFPU_UNARY_CALL`.

So A2's "this test also unblocks restoring the wrappers" is resolved by the wrappers never
having needed restoring, and what is left is coverage rather than a promotion decision:
`test_top32_rm.py` drives the same primitives the wrappers wrap, but through the LLK layer,
which is where a tt-llk test can reach. Covering the **wrapper layer** needs a metal-side
test, the same shape and the same blocker as B1.

## 8. Note on tooling (applies to the remaining work too)

Run tests through `tt-llk/.claude/scripts/run_test.sh` (`count` / `compile` / `run`), not
raw pytest. Two gotchas cost time: a `--k` expression containing brackets or commas
mangles the pytest args and surfaces as an opaque xdist worker crash (use `--test-id`, or
a bracket-free `--k`); and `tests/.venv` must be created with
`source ./setup_external_testing_env.sh` — `setup_testing_env.sh` alone only fetches SFPI
and assumes the Docker image's Python environment.

**Measured cost of ignoring the first sentence (2026-08-20).** A background plain
`pytest test_matmul_custom_compressed.py test_topk_xl.py` was left running while a
`run_test.sh` run took the device. Result: **137 failures out of 688 and a `TENSIX TIMED OUT`**
(BRISC command poll, python counter 1239 vs brisc 157). Serially afterwards the same two suites
are **100/100** and **588/588**. The `flock` inside `run_test.sh` is what makes concurrent
agents' numbers mean anything; without it the failure mode is a large, entirely spurious
failure count that reads exactly like a real regression — and it also leaves the device in a
state where the *next* run can produce a one-off wrong answer (Finding 12b).


---

## 9. `top32_rm` sort family (#52713) — DONE

Unblocked by #52713 merging; landed 2026-08-20 in `5b768385ee1` (plain path) and `edcdc8f4157`
(pre-sorted path). `tests/sources/top32_rm_test.cpp` +
`tests/python_tests/test_top32_rm.py`. **BH p100a: 10 passed.**

Before this, **every** entry point the family exposes was uncalled from `tests/sources`, and
`_top32_rm_init_` looked covered on a grep only because its sole occurrence in the test tree was
inside a comment in `sort_headers_coexist_test.cpp`.

### Both of the consumer's modes

| Mode | Entry points | Shape |
|---|---|---|
| plain (`< 1024`) | `_llk_unpack_A_top32_rm_init_`/`_`, `_llk_math_top32_rm_init_`/`_`, `_top32_rm_init_`, `_bitonic_top32_phases_steps_`, `_bitonic_top32_merge_`, `_bitonic_top32_rebuild_` | `top32_rm_dev_compute.cpp`: 64 row-major elements at a time into a Dest **column**, sort, merge the running top32 across tiles |
| pre-sorted (`>= 1024`) | `_bitonic_top32_of_1024_rm_pre_sorted_{prep,combine,final}_` | `top32_rm_dev_compute_v2.cpp`: whole 32x32 tiles transposed into Dest, 16 columns reduced at once |

Sweep: plain at row lengths 64 / 128 / 160 / 256 x `dest_acc` No / Yes (8 variants, Float16_b);
pre-sorted at 1024 / 2048 (2 variants, Float32).

### What made it non-obvious

`_llk_unpack_A_top32_rm_` takes 64 consecutive **row-major** elements — not a tilized tile — and
lands them in the first COLUMN of 64 Dest rows, 16 per face, transposed within the face. That
column is what the bitonic sort addresses, with distances in Dest ROWS (8/16/32/64). The index
operand must sit exactly +2 tiles away because `load16`/`store16` hardcode
`dst_indices_offset = 128`, and the pack side narrows the packer to one datum per row
(`SETADCXX PAC`) to turn the surviving column back into 32 contiguous L1 elements.

The pre-sorted mode does not use the family's unpack at all: it goes through `transpose_tile`'s
LLK sequence, where `transpose_of_faces` is always on but the within-face 16x16 transpose and
`acc_to_dest` belong to the **non**-32-bit path only — on the 32-bit path the within-face half is
`_llk_math_transpose_dest_` on the math thread, and `acc_to_dest` with `unpack_to_dest` is
static_asserted out (`llk_unpack_A.h:64`). Init interleaving follows the consumer: the
datacopy/transpose init is reinstated before each chunk's transposes while the SFPU family init
runs once, which is coherent because `_top32_rm_init_` owns `ADDR_MOD_6` and the SFPU control
register's index-tracking bit while the datacopy owns the MOP and the other ADDR_MODs.

### Two deliberate deviations from the consumer

- **One format for both operands.** The consumer runs bf16 values + uint32 indices, needing a
  srcA reconfig between the two unpacks and a pack reconfig on the way out. The test carries
  indices as floats holding the integer itself, so it measures the sort rather than the reconfig
  sequence — C3 already owns the reconfig question. In the plain mode that caps the row at 256
  (bf16 has 8 mantissa bits); the pre-sorted mode is Float32, where it does not bind.
- **Distinct values straddling zero.** Distinct means "the top 32" determines the indices, so
  indices are asserted **exactly** rather than tolerating hardware tie order the way
  `test_topk.py` has to. Straddling zero makes the -inf padding load-bearing: a 32-element tail
  chunk fills two faces and the other two arrive as -inf from `CLR_SRC_NEGINF`, so with
  non-negative stimuli the `row=160` case would pass on padding alone.

**`dest_acc` turned out to be a real axis.** It selects the index word width inside the sort
(`InstrModLoadStore::INT32` vs `LO16`) *and* the Dest-move opcode in
`llk_math_top32_rm_configure_mop` (ELWADD against a zeroed SrcB at fp32 Dest, MOVA2D otherwise).
The consumer only ever builds fp32 Dest, so the `dest_acc=No` cells are the first exercise the
16-bit half of this family has had — and they pass.

### Discrimination (Finding 13)

- Plain: rebuilding the driver with a 4-byte chunk stride instead of 2 makes it re-read
  overlapping chunks, and the answer comes back `[63, 62, 61, 59, 57, ...]` against a golden of
  `[63, 62, 61, 60, 59, ...]`. Both assertions fire.
- Pre-sorted: discriminates by construction at 2048, which is why that length is in the sweep —
  the tiebreak is a permutation over all 64 runs, so **17 of the 32 winning indices land in the
  second 1024-chunk**. A driver that dropped the second chunk, or a `combine` that did not merge
  across tiles, cannot produce that answer. The 1024 case is the complement: one tile, prep then
  final, no combine at all.

### Regression run for the 2026-08-20 change set

Serially through `run_test.sh` on BH p100a, all PASS: `test_top32_rm.py` **10**;
`test_rmsnorm_bcast_scalar_dest_reuse.py` 114 (+114 skipped);
`test_sfpu_sampling.py` 63 (+97 skipped); `test_custom_mm_uninit_restore.py` 7 (+8 skipped,
**1 xfailed** — C1, expected); `test_custom_mm_uninit_parity.py` 2;
`test_matmul_custom_mm.py` 32; `test_sort_headers_coexist.py` 2;
`test_set_dst_write_addr_offset.py` 14 (+14 skipped); `test_topk_xl.py` 100;
`test_matmul_custom_compressed.py` 588.

---
