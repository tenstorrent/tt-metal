# PR #52938 — replies to the review bot

Paste each as a reply on its thread, then resolve. Every one is fixed in code; none is a
"won't fix". Head at the time of writing: `d1366655516` + the follow-up below.

The bot last reviewed `14ebb6b5815`, so its right-shift comment was already fixed by the time it
landed — the reply says so rather than pretending it prompted the fix.

---

## 1. `golden_generators.py` — "the unary right-shift kernel does not zero out-of-range shifts"

> **Fixed** in `d1366655516`, which landed just before this review — so this was already in flight
> rather than prompted by it, and the credit for spotting it is still the bot's from the previous
> round.
>
> You were right, and it was worse than one wrong branch: the shared `_shift` helper asserted a single
> out-of-range rule for two kernels that do not share one.
>
> ```cpp
> calculate_left_shift:   out_of_range ? vInt(0) : (v << amt)        // → 0
> calculate_right_shift:  eff = (shift_amt >= 32) ? 31u : shift_amt  // clamps, then sign-extends
> ```
>
> So an out-of-range right shift of a negative operand gives `-1`. The helper is gone; each golden
> implements its own kernel's rule. `BinarySFPUGolden._right_shift` already documented that the
> *binary* op really does zero both signs, which is what made the unary claim checkable and wrong.
>
> One thing worth recording: the corrected branch is **unreachable today**. I re-tested rather than
> assumed — restoring negatives to the stimulus still fails 23 of 35 variants, in range as well as
> out, which identifies it as sign-magnitude Dst delivery rather than shift arithmetic. So the golden
> is now right where it was wrong, but the assertion still only covers the positive half where both
> rules coincide at 0, and the docstring now says that instead of implying otherwise.

---

## 2. `test_variant_parameters.py` — "the `u` suffix wraps a negative amount"

> **Fixed.** Correct: `SFPU_SHIFT_AMOUNT` emits the amount with a `u` suffix and both kernels branch
> on `shift_amt >= 32` as unsigned, so all four negative amounts arrive as large unsigned values and
> take the same out-of-range path as `32, 33, 40, ...` — four amounts, one code path, eight redundant
> silicon variants.
>
> The unary sweep now collapses them to a single representative (`_UNARY_SHIFT_AMOUNTS`), taking it
> from 36 variants to 30. **One is kept rather than none** because the unsigned wrap is load-bearing:
> if `SHIFT_AMOUNT` ever became signed, `-1` would compare as in-range and `v << -1` is undefined
> behaviour rather than a wrong answer, so it is worth one variant to pin.
>
> The shared `SHIFT_EDGE_AMOUNTS` list keeps all four, because the *binary* shift ops take their
> amount as a signed operand where the four are genuinely distinct. Only this consumer narrows.

---

## 3. `test_sfpu_ternary.py` — "`_TERNARY_NONZERO` carries `seed=0` ... both operands draw the identical stream"

> **Fixed, and it was a real coverage hole.** Confirmed before changing anything: one spec used for
> both operands makes them bit-identical, because the seed is per-spec.
>
> ```
> A[:5] = [-0.99779, 0.86285, -0.87007, -0.91684, -0.89403]
> B[:5] = [-0.99779, 0.86285, -0.87007, -0.91684, -0.89403]   # torch.equal(A, B) is True
> ```
>
> The pole on `c` was still being reached, which is why this passed — but the operands stopped being
> independent. `snake_beta` degenerates from `sin(b*a)` to `sin(a²)`, and a kernel that read the wrong
> one of the two operands would have been invisible.
>
> Now two specs differing only in seed (0 and 1), so the streams stay reproducible while being
> different from each other. Verified `A != B` with both still satisfying `|x| >= 0.5`; the ternary
> suite is unchanged at 39 passed / 25 skipped.
>
> Worth noting the neighbouring code was already fine: `_ternary_default_specs` leaves its spec
> unseeded, and an unseeded spec does draw different streams for A and B. It was the explicit
> `seed=0` that caused this.

---

## 4. `sfpu_domains.py` — "should this compare against `_truncate_mantissa(point, step_fmt)`?"

> **Fixed.** Yes it should. The narrow datapath truncates the boundary as well as the probe, so the
> question `probe_beside` is asking — are these two still distinct as the kernel sees them? — is only
> answered correctly if both sides are truncated.
>
> Checked before changing it: every point registered today (all 16 across `_OP_SINGULARITIES` and
> `_OP_EDGE_POINTS`) is bfloat16-exact, so `_truncate_mantissa(point, step_fmt) == point` and the two
> forms are equivalent right now. **The change is behaviour-preserving and purely defensive** — which
> is exactly why it was worth making: with the bare form, registering a pole that is not
> bfloat16-exact would make the condition silently always true, the widening would never trigger, and
> the probe would collapse back onto the boundary with nothing reporting it.

---

## 5. Suppressed: `spec_C` is added but no consumer reads it

> **Fixed** in `d1366655516`. `OperandSpecs` grew `spec_C` in this branch, but
> `_ternary_default_specs` still returned `spec_B` for C — with a comment saying "OperandSpecs carries
> only A and B", which had stopped being true — and `exclude_undefined_pair` never subtracted C's
> undefined ranges.
>
> Both paths are dead today, since no ternary op has an `_OP_DOMAIN_REGISTRY` entry, which is exactly
> when a silent drop goes unnoticed: the first person to register a ternary domain would have had
> their C spec discarded on the one code path that exists to honour it. Both now read C.

---

## 6. Suppressed: the PR body does not explain this PR's changes

> **Fixed** — the description is rewritten to cover what the PR does, what it deliberately leaves out,
> and what is unverified. Summary of the last: 30 unary ops remain outside cat B and none of them is
> per-op work; cat F is untouched; and **nothing is verified on Wormhole**, which matters most for the
> seven comparison goldens, since the total order they model is documented for Blackhole only.
