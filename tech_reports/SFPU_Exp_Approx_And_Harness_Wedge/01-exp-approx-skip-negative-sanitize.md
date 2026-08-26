# 01 — Approximate exp: opt-in `SKIP_NEGATIVE_SANITIZE`

## Wormhole

File: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`

## What the approximation costs today

`calculate_exponential<APPROXIMATION_MODE=true, CLAMP_NEGATIVE=true>` — the default, and what
SDPA's softmax uses — spends **24 SFPU issue slots on 8 datums (3.0/datum)**, in two passes:

| pass | instructions | purpose |
|---|---|---|
| sanitize | 8 `SFPLOADMACRO` + 7 `SFPNOP` = **15** | clamp every input to >= -88.5, store back to Dst |
| compute | 8 `SFPLOADMACRO` + 1 `SFPNOP` = **9** | the actual exponential |

The compute pass is already optimal: `exp_init`'s Macro Sequence Register 0 fuses
`LOAD, MAD, ROUND(STOCHRND), SIMPLE(SHFT), STORE` into a single issue, using all four
sub-unit slots. **Five of every eight slots this function spends are the clamp.**

`SKIP_NEGATIVE_SANITIZE` (new, defaults to `false`) drops the sanitize pass:
**9 slots per 8 datums = 1.125/datum, a 2.67x reduction.** No new macro configuration is
needed — Macro Sequence Register 1 is simply left configured and unused.

For every input >= -88.5 the result is **bit-identical**, because the clamp is a `max()` that
leaves such inputs untouched.

## Why the clamp cannot simply be fused away

Two things were checked before adding a flag rather than a fix.

**The rounding step does not saturate.** The obvious idea is to let the fp32->int conversion
clamp negatives to zero and delete the `SFPSWAP`. It does not work.
`SFPSTOCHRND_MOD1_FP32_TO_UINT16` sets `KeepSign = false`, which means it takes **|i|** and
then clamps to 65535. So for x = -1000, `i = A*x + (B-C)` is -336829, `|i|` saturates to
65535, and `65535 << 15` is ~3.4e38. The failure is in the *opposite* direction from a
missing clamp: the result is enormous, not small, so it poisons any downstream sum. The
clamp is load-bearing.

**The clamp cannot move into the compute macro.** `SFPLOADMACRO` schedules at most one
instruction per sub-unit column. The clamp is `SFPSWAP`, which is a **Simple** instruction,
and the compute macro already needs Simple for `SFPSHFT`. Moving the shift to the Round
column is blocked too — `SFPSHFT2` lives there but so does `SFPSTOCHRND`. Either way one
column is doubled, so a clamping variant needs two macro issues. `SFPSWAP` additionally
forces an `SFPNOP` on the MAD sub-unit for the same cycle and requires Simple and Round idle
on the next, which is where the 7 `SFPNOP`s in the sanitize pass come from.

So 3.0 slots/datum is the floor *while clamping*, and 1.125 is the floor without. There is no
middle.

## The contract, and why it is not enabled anywhere

```
the caller must guarantee every input is >= -88.5
```

This is a caller contract, not a hint, and violating it does not degrade gracefully — see
above. The documented valid range of the underlying Schraudolph approximation is already
`[-88, 0.72]`, so on its own stated domain nothing is lost.

**It is not turned on at any call site**, deliberately. The obvious candidate is
`tt_metal/hw/inc/api/compute/experimental/sdpa.h:186`, which calls
`calculate_exponential<true, DST_ACCUM_MODE, true, 8, true>()`. Flash attention subtracts the
running row max, so x <= 0 holds — but an **additive** attention mask (-1e9, or -inf) survives
that subtraction and lands far below -88.5. Whether that is reachable depends on how masking
is applied in each caller, which is a question for the owners of those kernels, not something
to decide from inside the LLK.

## Verification

The default path is unchanged, and that is what was verified: the full approx-exp sweep
passes on Wormhole n300, and the diff with whitespace ignored is 38 insertions confined to
the template signature, a `static_assert`, and two `if constexpr` brackets. The 163-line raw
diff is clang-format reindenting the bracketed block.

```
pytest test_sfpu_unary.py -k "Exp and approx_mode:Yes"
  ->  98 passed, 56 skipped, 6 xpassed
```

**The 6 XPASS are pre-existing and unrelated.** They are exactly the six entries of
`_APPROX_EXP_ACCURACY_XFAIL` (3 format combinations x 2 tile shapes), which expects approx
exp to breach the 5% rtol above an argument of ~8 on Wormhole. Verified by stashing
`ckernel_sfpu_exp.h` back to HEAD and re-running: the same 6 XPASS. The xfail set is stale on
this board and wants its own look, independent of this branch.

`SKIP_NEGATIVE_SANITIZE = true` compiles (both `is_fp32_dest_acc_en` values instantiated) but
is **not** covered by a hardware test, because no test parametrises it — the tt-llk kernel
template `helpers/include/sfpu_operations.h` plumbs `APPROX_MODE`, `ITERATIONS` and
`CLAMP_NEGATIVE`, not this flag. Anyone enabling it should add that parameter and a case that
feeds inputs below -88.5 to pin the documented behaviour.

## Blackhole

`SKIP_NEGATIVE_SANITIZE` is added to
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h` as well. The
Blackhole `APPROXIMATION_MODE && CLAMP_NEGATIVE` path is structurally identical to the Wormhole
one — the same two macro passes, the same `Macro Sequence Register 1` for the clamp, differing
only in `ADDR_MOD_7` where Wormhole uses `ADDR_MOD_3` — so this is a transliteration, and the
same numbers hold. Confirmed from the compiler, not by inspection: with the flag set, both
`is_fp32_dest_acc_en` instantiations emit exactly

```
sfploadmacro 0,L0,0,0,7      sfploadmacro 0,L1,2,0,7
sfploadmacro 0,L2,4,0,7      sfploadmacro 0,L3,6,0,7
sfploadmacro 0,L0,8,0,7      sfploadmacro 0,L1,10,0,7
sfploadmacro 0,L2,12,0,7     sfploadmacro 0,L3,14,0,7
sfpnop
```

**9 slots for 8 datums = 1.125/datum**, against 24 = 3.0/datum with the clamp — the same 2.67x.
The `static_assert` is live on this arch too (it fires if the flag is set outside the
`APPROXIMATION_MODE && CLAMP_NEGATIVE` path).

### There is no cycles number for this on Blackhole, and here is why

Unlike tanh and sigmoid, this one could not be put on silicon at all, for two independent
harness reasons:

1. **`perf_eltwise_unary_sfpu.py` drives `clamp_negative=False` for exp on this arch.** The
   `clamp_negative` column of every collected row is `False`, which means the perf test measures
   Blackhole's *other* approx-exp path — the replay-buffer kernel at `ITERATIONS == 8` / `== 32`
   — and not the sanitize path this flag touches. That path measures 93.59 cycles/tile and is
   unaffected by this change: 93.59 cycles/tile with and without it, measured either side of
   the flag being added.
2. **The clamped path cannot run at the iteration count the perf test pins.** Its body is
   hand-unrolled for 8 datums and does not respect `ITERATIONS` (there is a commented-out
   `static_assert(ITERATIONS == 8)` in the source and a TODO, tt-llk#1486), while the perf
   parametrisation fixes `iterations=32`. Forcing the branch on and running it wedges the
   TRISC — `TENSIX TIMED OUT ... Polling brisc command timed out` — which is the expected
   consequence, not a new bug.

So the Blackhole result for this item is a slot count from the compiler, matching Wormhole's,
and the runtime coverage gap is the same one already recorded above: no test parametrises
`SKIP_NEGATIVE_SANITIZE`, because the tt-llk kernel template
`helpers/include/sfpu_operations.h` plumbs `APPROX_MODE`, `ITERATIONS` and `CLAMP_NEGATIVE` and
not this flag. Anyone enabling it should add that parameter and a case feeding inputs below
-88.5. Fixing tt-llk#1486 first would also make the clamped path measurable at
`ITERATIONS = 32`.

The default remains `false` and the default path is byte-identical on both arches; that is what
the exp sweep below verifies.
