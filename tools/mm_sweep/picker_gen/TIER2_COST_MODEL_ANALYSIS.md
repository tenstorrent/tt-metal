# Tier 2: what the picker's cost model actually gets wrong (and why refitting it does not help)

Ground truth: 3,091 measured configs across 24-25 shapes (exhaustive over the feasible set), HEAD @ the
12-entry table update. The C++ cost model was mirrored verbatim in Python and validated by reproducing the
deployed fallback picks exactly (e.g. 512x6144x768 -> (12,1,1,2,1), 512x2304x6144 -> (3,4,1,1,1)).

## 1. The deployed fallback's real regret: median +7.7%, mean +9.6%, worst +36.1%

10 of 25 shapes are picked optimally; 15 are not. Two distinct error modes:

* **Over-parallelisation.** The model picks MORE cores than optimal on 7 shapes and FEWER on **0** --
  a strictly one-sided error. Median regret on those: +14.5%.
* **Wrong decomposition SHAPE at equal core count.** e.g. 512x6144x768 is +24.7% while using the same 96
  cores (Pk12/Sm1/nsb1 vs the optimal Pk6/Sm2/nsb3).

Distributional bias is stark:
| | nsb=1 | Sm=1 |
|---|---|---|
| model picks | 21/25 | 21/25 |
| measured best | 10/25 | 13/25 |

**Mechanism for the nsb bias:** `ovlT = kOvl * comp_pc / Nbpc` with `Nbpc = ceil(Nown/nsb)`. A larger nsb
means fewer N-blocks, so Nbpc shrinks and ovlT GROWS -- the model actively penalises wide sub-blocks.
Measurement says wide sub-blocks usually win (larger output writes; and nsb>=2 is what lets the
reduce-scatter gate fire at all). `kAcap=6` compounds it by capping the credit for sub-block area.

**Mechanism for the Sm bias:** structural. The fallback ranks an Sm=1 ANCHOR and only considers Sm>1
through the narrow-N hysteresis (`Nband <= 2`). Note the anchor scaffolding is *load-bearing*: with the
deployed constants, plain argmin over all configs scores median +25.8% / mean +29.0%, far worse than the
guarded +7.7%. The scaffolding exists to compensate for a cost function that cannot rank Sm>1.

## 2. Refitting the constants DOES NOT beat the deployed picker (negative result)

Grid search (76,800 param sets) over ground truth, adding a new `kCon` core-CONTENTION term:

| | median | mean | worst |
|---|---|---|---|
| deployed picker (guarded) | +7.7% | +9.6% | +36.1% |
| refit, in-sample | +5.1% | +5.6% | +22.6% |
| **refit, LEAVE-ONE-OUT held out** | **+6.2%** | **+10.6%** | **+65.3%** |

The in-sample gain was overfitting (9 free params, 24 shapes). Out-of-sample the refit is better on median,
WORSE on mean, and much worse in the tail. **Do not ship a refit.**

Parameter selection was nonetheless extremely stable across all 24 folds -- `csat=16` 24/24, `ovl=0.0`
23/24, `acap=64` 22/24, `kbcap=2` 24/24, `wst=0.5` 24/24, `rk=0.4` 22/24 -- which says the CONSTANTS are
well determined and the **functional form is the binding limitation**. Refitting moves mean regret
9.6% -> 10.6%: nothing.

**`kCon` was NOT selected (0.0 in 22/24 folds).** The core-contention hypothesis is REFUTED: the
over-parallelisation error is a symptom of the nsb/area mis-weighting (ovl, acap), not of missing
contention. Zeroing ovl and uncapping acap does the work.

## 2b. ONE structural term DOES beat the deployed picker (positive result)

Refitting constants fails, but the constants were never the problem -- the FORM was. Tested an extended
form that keeps the LOOCV-stable read/compute constants fixed (csat=16, acap=64, ovl=0, kk=0.5, aa=2,
kbcap=2) and adds three physically-motivated terms, fitting only 4-5 new params:

| model | median | mean | worst |
|---|---|---|---|
| deployed picker | +7.7% | +9.6% | +36.1% |
| 9-param constant refit (LOOCV) | +6.2% | +10.6% | +65.3% |
| **extended form (LOOCV held out)** | **+5.6%** | **+7.2%** | **+32.7%** |

Better than deployed on all three metrics under LOOCV. **⚠️ BUT THIS DID NOT SURVIVE A TRULY FRESH TEST
SET -- see section 4. LOOCV held the CONSTANTS out but the model FORM and the grid were chosen knowing
those 24 shapes. Do not adopt this term on the strength of the LOOCV number.**

Of the three added terms, only ONE survived the fit:

* ✅ **Strategy-differentiated reduction cost.** `rk_ch=0.4` (23/24 folds) charged as `rk_ch*(Pk-1)*out`
  for the CHAIN, vs `rk_rs=0.8` (22/24) charged as `rk_rs*(Pk-1)/Pk*out` for the RING. At Pk=6 that is
  2.0 vs 0.67 -- the ring is charged ~3x LESS. Physically obvious in hindsight: the chain serialises Pk-1
  full-block adds while the ring moves 1/Pk of a block per round, yet the deployed model charges both the
  same. This single term is the entire improvement.
* ❌ **Output-write / lone-page penalty** (`kw=0.0` in 23/24, `lone=0.0` in **24/24**). REFUTED, despite
  "nsb=1 makes every output write a lone page" being repeatedly observed this campaign. The nsb effect is
  already captured by removing the ovl penalty and the area cap; an explicit write term adds nothing.
* ❌ **M-split in1 forwarding cost** (`kf=0.0` in **24/24**). REFUTED, even though Sm is the most
  mis-picked knob.

Lesson: the mis-picks were caused by terms that were WRONG (ovl penalising wide sub-blocks, acap capping
sub-block credit, one reduction cost for two very different strategies) -- not by terms that were MISSING.

## 3. The actionable result: the model is a good SHORTLISTER, a bad PICKER

Regret of best-of-top-N, N ranked by model cost:

| N | deployed params | refit params |
|---|---|---|
| 1 (what ships today) | 16.4% med / 19.7% mean | 5.3% / 6.1% |
| 2 | 9.2% / 13.2% | 3.1% / 4.2% |
| 5 | 5.8% / 7.0% | 2.9% / 3.7% |
| 8 | 3.2% / 5.5% | 0.1% / 2.1% |
| 12 | 0.6% / 3.6% | 0.0% / 1.7% |

The refit column is in-sample and optimistic. The DEPLOYED column is not fitted on these shapes at all and
still reaches 3.2% median at N=8, 0.6% at N=12 -- so this conclusion is robust to the overfitting above:
**even a mediocre cost model produces a shortlist that contains the true optimum.**

## 4. TRUE out-of-sample test on 16 fresh shapes -- the extended form does NOT generalise at N=1

The Tier 1 sweep produced 16 shapes that were used in NO fit. Scored on them:

| model | median | mean | worst |
|---|---|---|---|
| real deployed picker (table-or-fallback, anchor+hysteresis) | +7.1% | **+10.5%** | -- |
| deployed cost FORM, plain argmin N=1 | +5.9% | +35.0% | +136.2% |
| extended form, N=1 | +4.3% | **+15.9%** | +61.4% |
| deployed form, top-8 shortlist | +2.5% | +4.0% | +12.3% |
| **extended form, top-8 shortlist** | **+0.1%** | **+1.2%** | **+5.4%** |

**The extended form beats the deployed picker on MEDIAN but LOSES on MEAN (+15.9% vs +10.5%)**, with
outliers at +61% (32x6144x768), +54% (64x2048x1536), +46% (64x6144x768). Its LOOCV advantage was an
artifact of choosing the form on those 24 shapes. **RETRACTED: do not ship the strategy-split reduction
term as a picker change on this evidence.**

Note also how badly the deployed cost FORM does without its anchor/hysteresis guard on fresh shapes
(+35% mean, +136% worst) -- further confirmation that the guard is load-bearing.

**What DOES survive the fresh test is the shortlist result**, and it survives strongly: top-8 is
~0% median / 1.2% mean / 5.4% worst with the extended form, and 2.5% / 4.0% / 12.3% with the deployed one.

### Recommendation (revised after the fresh-set test)

Do NOT ship a new cost model -- neither a constant refit nor the extended form generalises at N=1.
Capture the perf with:

1. **Measured table entries** for shapes that matter -- already demonstrated: 12 entries, median 10.7% gain.
   This is why the table exists and why it must be re-validated whenever the kernel stack moves (7 of the
   12 were stale rows).
2. **A first-run/build-time AUTOTUNER over a model-shortlisted candidate set of ~8**, caching the winner per
   (Mt,Kt,Nt). Validated on 16 FRESH shapes: ~0% median / 1.2% mean / 5.4% worst regret. At ~1.4 s/config
   that is ~11 s once per distinct shape. This is the only approach here that generalises -- the model's job
   becomes shortlisting (which it does well even unmodified) instead of picking (which it does ~10% badly).
   The extended form is still worth using FOR THE SHORTLIST (5.4% vs 12.3% worst case), just not as a picker.

Tooling: `tier2/model.py` (verbatim mirror), `refit.py` (grid search), `loocv.py` (held-out validation).


## 5. Shortlist sizing, and a refuted shortcut (19 fresh shapes)

| shortlist | median | mean | worst | measurements |
|---|---|---|---|---|
| plain top-1 (ships today) | +10.9% | +18.1% | +61.4% | 1 |
| plain top-4 | +2.2% | +3.7% | +16.6% | 4 |
| **plain top-8** | **+0.1%** | **+1.1%** | **+5.4%** | 8 |
| bias-corrected (top-2 + max-nsb + Sm=2 siblings) | +2.9% | +7.8% | +23.6% | 2.1 |

❌ **The bias-corrected shortlist is REFUTED.** Since the model's error is directional (under-selects nsb
and Sm), it seemed likely that adding the known-under-ranked siblings would let a ~4-element shortlist match
a top-8. It does not: the corrected set collapses to ~2 configs (the max-nsb sibling is usually already the
top pick) and it loses to plain top-4 at half the budget. **Use a plain top-N.** top-8 for ~0 regret, top-4
if the autotune budget must be halved.
