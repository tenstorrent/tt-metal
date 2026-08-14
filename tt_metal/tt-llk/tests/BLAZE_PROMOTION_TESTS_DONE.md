# tt-llk blaze promotions — COMPLETED work (archive)

> Closed record of the tt-llk test work for the blaze->tt-metal `experimental/` promotions
> that is **done and passing on Blackhole p100a**. Split out of
> `BLAZE_PROMOTION_TEST_STRATEGY.md` on 2026-08-14 so that document only tracks
> outstanding work.
>
> Nothing here needs action, with **one exception**: the defect in Finding 2 is recorded as
> an `xfail` in a landed test and still needs an owner decision. It is cross-referenced
> from the open-work document.
>
> Branch: `ldjurovic/llk-tests-blaze-promotions` (tt-metal), which merges #52709 + #52713 +
> #52727 onto main so the promoted headers exist to compile against.

**PRs covered:** tt-metal #52747, #52745, #52713, #52727, #52709

---

## Summary

| | |
|---|---|
| Verification tier (V1-V4) | 4 of 4, all green |
| New test items landed | 3 (`add_rsqrt`, `custom_mm` `block_uninit`, sort-header coexistence) |
| Test results | **75 passing / 2 xfailed** across 3 drivers + 3 python suites |
| Files added | 6 (3 `tests/sources/*.cpp`, 3 `tests/python_tests/test_*.py`) |
| Product findings | 1 defect (needs a decision) + 3 behavioural constraints |

### Landed tests

| Item | Files | Result |
|------|-------|--------|
| `add_rsqrt` SFPU functor (#52709) | `tests/sources/sfpu_add_rsqrt_test.cpp`, `tests/python_tests/test_sfpu_add_rsqrt.py` | 42 passed, 14 skipped |
| `custom_mm`/`compressed_custom_mm` `block_uninit` (#52727) | `tests/sources/custom_mm_uninit_restore_test.cpp`, `tests/python_tests/test_custom_mm_uninit_restore.py` | 30 passed, 2 xfailed, 32 skipped |
| Sort-header coexistence (#52713) | `tests/sources/sort_headers_coexist_test.cpp`, `tests/python_tests/test_sort_headers_coexist.py` | 3 passed |

### Verification tier — all green on the merged branch

| Suite | For | Result |
|---|---|---|
| `test_matmul_custom_compressed.py` | V1 / #52727 | 582 passed |
| `test_topk_xl.py` | V2 / #52713 | 71 passed |
| `test_sfpu_sampling.py` | V3 / #52745 | 51 passed, 93 skipped |
| `test_generalized_moe_gate.py` | V4 / #52747 | 89 passed |
| `test_sfpu_generic_moe_gate_topk.py` | V4 / #52747 | 24 passed |
| `test_eltwise_binary.py` | regression baseline | 4388 passed, 72 skipped |

V3 and V4 confirm the verdict below that #52745 and #52747 need no new tt-llk tests: the
canonical targets they rewire onto are already fully covered.

---

## 1. Cleanup PRs #52747 / #52745 — checked, verified, no tt-llk work needed

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

- `sampling_recip_init<legacy_compat>` — called at `sfpu_sampling_test.cpp:148`, swept both ways via
  `legacy_compat=[True, False]` (test_sfpu_sampling.py:212).
- `calculate_sampling_binary_first_column<SamplingBinaryOp::{add,sub,mul}>` — the collapsed dispatch,
  driven at `sfpu_sampling_test.cpp:122-126`.

**Verdict: no new test.** Run `test_sfpu_sampling.py` unchanged.

**One optional hardening idea remains open** — a polluter test proving
`sampling_recip_init` is *necessary*, not merely present. It is carried in the
open-work document, not here.


---

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

---

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

---

## 5. Findings from building these tests

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


---

## 6. Note on tooling (applies to the remaining work too)

Run tests through `tt-llk/.claude/scripts/run_test.sh` (`count` / `compile` / `run`), not
raw pytest. Two gotchas cost time: a `--k` expression containing brackets or commas
mangles the pytest args and surfaces as an opaque xdist worker crash (use `--test-id`, or
a bracket-free `--k`); and `tests/.venv` must be created with
`source ./setup_external_testing_env.sh` — `setup_testing_env.sh` alone only fetches SFPI
and assumes the Docker image's Python environment.


---
