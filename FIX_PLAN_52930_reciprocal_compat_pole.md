# Fix plan — `RsqrtCompat(0)` returns 1.7e38 instead of `inf` (`_reciprocal_compat_` has no pole guard)

**Issue:** [#52930](https://github.com/tenstorrent/tt-metal/issues/52930), finding 4.
**Evidence:** [ISSUE_52930_INVESTIGATION.md](ISSUE_52930_INVESTIGATION.md) §3, §4.4.
**Classification:** kernel defect. `rsqrt_compat` is a pure software composition, so the ISA prescribes no
answer — the issue is right that nothing documents either result.

> **Read §4 before scoping this.** The broken function is `_reciprocal_compat_`, and the public compute API
> `recip()` **defaults to it** (`recip.h:17,37` — `legacy_compat = true`). This is very likely not a
> legacy-corner fix. §4 says what to measure first.

---

## 1. The defect

`rsqrt_compat(0)` returns a large finite where the golden gives `+inf`, on **all 8** format combinations
(8/8 XFAIL reproduced on Wormhole n300).

| input → output | `dest_acc` | hardware |
|---|---|---|
| `Float16_b → Float16_b` | No / Yes | `0x7F000000` = 1.7014118e38 |
| `Float32 → Float32` | Yes | **`0x7EFFFD9E` = 1.7013500e38** |

**Correction to the issue text:** the recorded `0x7F000000` is the *bf16-rounded view*. The single computed value
is `0x7EFFFD9E`; rounding it to bf16 gives `0x7F00`. Any fix must be verified against `0x7EFFFD9E` on the fp32
pipeline, not against `0x7F000000`.

Plain `Rsqrt` over the same probe does **not** diverge — two implementations of one function disagreeing at their
shared pole. §3 explains why, and it is the whole argument for the preferred fix.

## 2. Root cause

`tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_rsqrt_compat.h`.

`_calculate_rsqrt_compat_(0)` first calls `_sqrt_compat_(0)`, which *does* guard the pole
(`v_if (val != 0.0f)` at line 36) and correctly returns `0.0`. That `0.0` is then handed to
`_reciprocal_compat_`, which has **no zero guard at all**:

```c
60:  sfpi_inline sfpi::vFloat _reciprocal_compat_(const sfpi::vFloat in) {
63:      sfpi::vFloat val = sfpi::setsgn(in, 1);   // -0.0
65:      val = setexp(val, 126);                   // -0.5  <-- the input's magnitude is now gone
         ...                                      // Newton-Raphson converges to 1/0.5 = 2.0 (slightly under)
78:      sfpi::vInt orig_exp = exexp(in);           // exexp(0.0)      = -127
         sfpi::vInt new_exp  = exexp(result);       // exexp(1.99997…) =    0
83:      new_exp -= orig_exp;                       // 0 - (-127)      =  127
84:      new_exp += 126;                            //                 =  253
86:      v_if (new_exp < 0) { result = 0.0F; new_exp = 0; }   // guards overflow only
96:      return setexp(result, new_exp);
     }
```

`exexp(0.0)` is `0 - 127 = -127` — `SFPEXEXP.md` extracts the raw exponent field (0 for a zero input) and
subtracts the bias. So the exponent difference on lines 83-84 lands on **253**: an ordinary finite exponent, one
short of the 255 that would mean infinity. `setexp` writes the field, the surviving mantissa ≈ 1.99997 rides
along, and the result is `1.99997 × 2^(253-127) = 1.70135e38` = `0x7EFFFD9E` — matching the measurement to the
bit.

**So this is not saturation and no clamp is involved.** It is `126 - exexp(in)` evaluated at `exexp(0.0)`, in a
function whose only guard (line 86) covers the *opposite* direction — an input so large that the reciprocal
underflows. The zero/infinity end was never handled.

The same arithmetic means `_reciprocal_compat_(±inf)` is also suspect: `exexp(inf) = 128`, so
`new_exp = 0 - 128 + 126 = -2 < 0`, which line 86 catches and returns `0.0`. That end happens to be right.

## 3. Why plain `Rsqrt` is correct, and why that decides the fix

`Rsqrt` (and the modern `Reciprocal`) use `sfpu_reciprocal_iter` in
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h`, which builds its scale factor as
`~in.Exp` **specifically** so the poles fall out for free. Its own comment:

> "Not only is `255-in.Exp` more efficient via `SFPNOT`, but it also ensures that `in.Exp == 0` results in
> `±inf`, and `in.Exp == 255` results in `±0`."

That kernel handles both poles by construction and needs no guard. A correct implementation of this function
already exists in the tree — which is why redirecting is preferred over patching (§5).

## 4. Blast radius — measure this first

`_reciprocal_compat_` is reached by more than `RsqrtCompat`:

| Consumer | Route | Default? |
|---|---|---|
| `RsqrtCompat` (the failing op) | `calculate_rsqrt<…, legacy_compat=true>` → `_calculate_rsqrt_compat_` (`ckernel_sfpu_rsqrt.h:19-21`) | `rsqrt.h:18` defaults `legacy_compat = **false**`, so production `rsqrt` is on the *good* path |
| **`recip` (public compute API)** | `recip_init` / `calculate_reciprocal<…, legacy_compat>` (`ckernel_sfpu_recip.h:113-131`) → `_calculate_reciprocal_compat_` | **`recip.h:17,37` defaults `legacy_compat = true`** ⚠️ |
| SDPA softmax reciprocal | `compute_common.hpp:252` — `calculate_recip_first_column<legacy_compat>` | **defaults `true`** ⚠️ |
| `SqrtCompat` | `_calculate_sqrt_compat_` — uses `_sqrt_compat_` only, **not** `_reciprocal_compat_` | unaffected |
| Sampling `recip_scalar` | `test_sfpu_sampling.py:212` drives both `legacy_compat` values | both |

**The tt-llk suite's `MathOperation.Reciprocal` exercises `legacy_compat = false`** — `sfpu_operations.h:972`
calls `calculate_reciprocal<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>` and takes the default `false`. So the
*default production reciprocal path is not covered by the reciprocal edge test at all*, and by the §2 derivation
`recip(0)` on that path should return ≈`1.7e38` rather than `+inf`.

**Step 0 of this plan is to measure that, not assume it.** Two existing vehicles need no new op enum:

```bash
cd tt_metal/tt-llk
# recip_scalar already parametrizes legacy_compat; drive a 0 through it.
bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole --test test_sfpu_sampling.py \
    --k "recip_scalar"
# and the SDPA legacy path:  sfpu_sdpa_test.cpp OP_RECIP_LEGACY
```

If `recip(0)` on the default path does return a finite, this stops being a legacy-kernel cleanup and becomes a
**correctness bug in the default reciprocal**, with its own issue and its own priority. Scope the rest of this
plan on that answer.

## 5. The fix — two options

### Option A (preferred): redirect the compat consumers to `sfpu_reciprocal_iter`

Make `_calculate_rsqrt_compat_` / `_calculate_reciprocal_compat_` use the modern reciprocal, keeping only
`_sqrt_compat_`'s distinct square-root approximation where that is the point of "compat".

* **Pro:** deletes the defect rather than patching it, removes the second implementation, and inherits a pole
  handling that is correct by construction and already tested.
* **Con:** changes numerics for every `legacy_compat = true` caller — which is the *opposite* of what a flag
  named `legacy_compat` promises. `_reciprocal_compat_`'s Grayskull-derived first guess (1.44) exists explicitly
  "for consistency", so bit-compatibility with older results may be a deliberate contract for some caller.
* **Therefore:** Option A is only viable once someone can say who depends on the legacy bit pattern. That is a
  question for the kernel owners, and it is the decision this plan is blocked on. Do not guess.

### Option B (safe, always correct): add the missing pole guard in place

Preserves every existing value except the pole:

```c
    // in == 0 makes setexp(val, 126) discard the magnitude, so the exponent difference below
    // lands on 126 - exexp(0) = 253 -- an ordinary finite -- rather than the 255 that means
    // infinity. The v_if(new_exp < 0) guard covers only the opposite (underflow) end.
    v_if (in == 0.0F) { result = std::numeric_limits<float>::infinity(); new_exp = 255; }
    v_endif;
```

placed alongside the existing `v_if (new_exp < 0)` block at line 86, before the `setexp` on line 96. The
caller-side `v_if (in < 0.0) { out = -out; }` in `_calculate_rsqrt_compat_` (line 108-110) already gives
`1/-0 = -inf` if a real `-0.0` arrives; note that on all but the unpack-to-dest pipelines it will not (see the
investigation's §4.1), so `+inf` is the answer to expect there.

Take care with the mechanics: writing `result = inf` and then `setexp(result, new_exp)` would overwrite the
exponent field, so `new_exp` must be set to 255 in the same block — or the guard must be applied *after* the
`setexp`, restructuring the `return`. The second shape is less error-prone; prefer it.

**Recommendation:** ship **Option B** now — it is behaviour-preserving everywhere except the one wrong value, so
it does not need the ownership question answered. Open Option A as a follow-up once §4 step 0 establishes how
widely `_reciprocal_compat_` is actually relied on, since if `recip` really does default to a broken pole then
consolidating on one implementation is the right end state.

## 6. Arch propagation

`_reciprocal_compat_` lives in the **tt-llk lib**, so the
`tt_metal/tt-llk/.claude/references/metal-integration.md` checklist applies. The 4-layer stack must stay
consistent:

| Layer | Action |
|---|---|
| `tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_rsqrt_compat.h` | the fix |
| `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_rsqrt_compat.h` | **identical file** — same `_reciprocal_compat_`, same missing guard. Same fix, same PR |
| `tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_sqrt.h` | different structure (semantic naming); check whether it has an equivalent compat reciprocal before assuming it is unaffected |
| CKernels LLK API (`hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_rsqrt.h`, `_recip.h`) | no signature change, so no edit expected — confirm |
| Compute API (`hw/inc/api/compute/eltwise_unary/{recip,rsqrt}.h`) | no signature change; the `legacy_compat` defaults are **not** changed by Option B |

No signature changes means no TTNN bypass-file work, but re-read the checklist rather than trusting this table.

## 7. Test changes

Current behaviour is pinned by non-strict xfails; the fix turns them into XPASS. Remove in the same commit:

1. **`tt_metal/tt-llk/tests/python_tests/test_sfpu_unary.py:609-618`** — the whole `MathOperation.RsqrtCompat`
   entry in `_EDGE_KNOWN_DIVERGENCES` (all 8 combinations).
2. **`test_sfpu_unary.py:673-675`** — the `MathOperation.RsqrtCompat` entry in `_EDGE_DIVERGENCE_REASON`.
3. **`test_sfpu_unary.py:591-599`** — the "STILL OPEN — not explained by the ISA" comment block loses its
   `rsqrt at 0` paragraph. Keep the file's habit of recording *why* it moved, with a pointer to this plan.

After removal, `test_eltwise_unary_sfpu_edges[…-RsqrtCompat-…]` **asserts** `rsqrt_compat(0) == inf` on all 8
combinations — `_OP_SINGULARITIES` at `sfpu_domains.py:1536` already puts `0.0` in the probe, so no new stimulus
is needed.

Coverage gaps this exposes, worth closing alongside:

* **`RsqrtCompat` is not in `SPECIALS_READY_OPS`**, so `rsqrt_compat(±inf)` and `(NaN)` are never driven. §2 shows
  the `+inf` end is handled by accident; pin it.
* **The default (`legacy_compat = true`) reciprocal has no edge test.** If §4 step 0 confirms the pole is wrong
  there, that needs a `MathOperation.ReciprocalCompat` (or an equivalent `legacy_compat` axis on the existing
  `Reciprocal` variants) so the default production path is covered at all. This is the largest gap the finding
  uncovers.

## 8. Verification

```bash
cd tt_metal/tt-llk

# 0. Scope first (§4): does the DEFAULT reciprocal path share the bug?
bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole --test test_sfpu_sampling.py --k "recip_scalar"

# 1. Baseline the exact value, fp32 pipeline (must read 0x7EFFFD9E, not 0x7F000000).
bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole \
    --test test_sfpu_wh_issue52930_probe.py --k "unary_edge_values and RsqrtCompat"

# 2. After the fix: 0 -> 0x7F800000 on all 8, and every non-pole row unchanged.
# 3. Shipped sweep with the tables removed, both arches.
bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole \
    --test test_sfpu_unary.py --k "edges and (RsqrtCompat or Rsqrt or SqrtCompat)"
bash .claude/scripts/run_test.sh run --worktree $PWD --arch blackhole \
    --test test_sfpu_unary.py --k "edges and (RsqrtCompat or Rsqrt or SqrtCompat)"

# 4. The consumers that share the function, so a numerics change cannot slip through.
bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole --test test_sfpu_sampling.py
bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole --test test_sfpu_sdpa.py
```

Acceptance:
- `rsqrt_compat(0) == +inf` (`0x7F800000`) on all 8 combinations, both arches.
- **Every other value bit-identical to baseline**, including the known ~0.15 % accuracy figure
  `rsqrt_compat(2.384e-07) = 2044.9895` (see §9 — not this fix's job).
- `SqrtCompat`, `Reciprocal`, sampling and SDPA suites unchanged.
- Both xfail entries removed; `_assert_signed_zero_partition_valid()` and the other collection-time assertions
  still pass.
- `perf_sfpu_div_wh.py` / `perf_eltwise_unary_sfpu.py` show no regression from the added guard.

## 9. Explicitly out of scope

`rsqrt_compat(2.384e-07)` = `2044.9895` against a golden of `2048.0` — ~0.15 % relative error on the fp32
pipeline, visible in the same probe. That is the legacy kernel's inherent accuracy, not the pole, and it is
already tolerated by `CUSTOM_TOLERANCES`. Do not try to fix it here; if it matters, it is another argument for
Option A.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Option B's guard interacts with the `setexp` on line 96 and silently produces a wrong exponent | Prefer the post-`setexp` restructuring (§5); verify the literal bit pattern `0x7F800000`, not `isinf` |
| Option A changes numerics for a caller that depends on legacy bits | Do not take Option A without an owner's answer; ship B first |
| The guard costs cycles in `recip`, which is on the SDPA softmax hot path | §8 step 4 runs the SDPA suite; check `perf_sfpu_reduce_sdpa.py`. One `SFPSETCC` on a kernel already doing 3 Newton iterations should be noise, but SDPA is the one place to confirm it |
| Fix lands WH-only; Blackhole XPASSes with the tables removed | Both arches in one PR; §8 step 3 runs both |
| §4 step 0 reveals a broader `recip` bug and the PR grows without bound | Keep this PR to `RsqrtCompat`; file the `recip` finding separately and link both to #52930 |

## 11. Effort

Option B: small — one guard per arch (2 files), two table deletions, plus the §4 step-0 measurement.
The step-0 result may spawn a second, larger issue; that is a feature of this plan, not scope creep in it.
