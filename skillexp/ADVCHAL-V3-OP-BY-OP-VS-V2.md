# advchal-v3 vs v2 — op by op, layout by layout, on the three cells where v3 lost the most

Companion to [`RESULTS`](ADVCHAL-V3-RESULTS.md). The headline percentages say v3 delivered less than v2 on four
of five comparable cells. They do not say **why**, and the why is different on each cell. This file goes down to
the individual device op — its core count, its memory layout, its microseconds — and to the lines of
`tt/optimized_decoder.py` each version actually shipped.

**Every number here is read from a committed artefact of the run that produced it**: `tracy/*_ops.csv` and
`perf_reports/*/decode_perf_report.csv` for per-op device time, `measurements/*.json` for harness medians,
`measurements/*.policy.json` and `tt/optimized_decoder.py` at each version's own tag for the layouts.

| cell | v2 µs/model | v3 µs/model | gap | root cause, one line |
|---|---:|---:|---:|---|
| **gemma-4-26B `-onA`** | −7,105.4 | −1,198.3 | **−5,907** | v3 found **better** grids than v2 on both kinds, then vetoed 17 sliding measurements from **one** PCC sample |
| **phi-3.5 `nofuse-noadvise`** | −1,284.9 | 0 | **−1,285** | the two versions shipped **different code** for "the same" change; v3's is interleaved where v2's is sharded, and returns the key in the query's layout |
| **phi-3.5 `fuse-noadvise`** | −1,267.5 | −278.2 | **−989** | v3 **could not express** v2's change at all (the capture substitutes `_decode_rope`), so it shipped a smaller one |
| phi-3.5 `-onA` | −1,594.1 | −1,254.4 | −340 | not analysed here; see `RESULTS` §3 |
| gemma-4-26B `fuse-noadvise` | −791.7 | **−986.4** | **+195** | v3 ahead |

Together the three cells below are **8,181 µs of the 8,326 µs total gap — 98 %.**

---

# 1. gemma-4-26B `nofuse-noadvise-onA` — the ops agree, the verdict does not

Both versions start from the **identical** frozen incumbent: `model_estimate.before_us` = **54,633.6 µs** in both
`final.json` files, and per kind **44,728.325** (sliding, 25 layers) / **9,905.275** (full, 5 layers) in both.
Same host, same optimizer pin. So every difference below is a decision, not a baseline.

## 1.1 The op the whole cell turns on: eleven `LayerNormDeviceOperation` calls, all on **one core**

From v3's `tracy/{kind}/incumbent_ops.csv`. Each row is one norm call in one decoder layer:

| kind | 11 norm calls, device µs each | total |
|---|---|---:|
| `full_attention` | 44.12 · 9.50 · 9.53 · 8.04 · **44.20 · 44.08 · 43.98 · 34.67 · 44.02 · 39.85 · 44.05** | **366.03 µs** |
| `sliding_attention` | 43.88 · 6.15 · 6.20 · 5.30 · **44.66 · 43.98 · 44.38 · 34.55 · 44.18 · 39.82 · 44.09** | **357.20 µs** |

The shape is the same on both kinds: **eight expensive norms at 34–45 µs and three cheap ones at 5–10 µs**, and
every one of the eleven runs on `Cores=1`. On a 1,981 µs / 1,789 µs layer that is **18.5 % / 20.0 % of the layer
spent in a reduction on a single core.**

## 1.2 What v3 shipped on `full_attention`: eight cores, and it paid for the resharding

`tracy/full_attention/shipped_ops.csv` against `incumbent_ops.csv`, whole trace:

| | incumbent | shipped | Δ |
|---|---:|---:|---:|
| 11 `LayerNorm` calls | **366.03 µs** | **87.91 µs** | **−278.12** |
| `InterleavedToSharded` + `ShardedToInterleaved` | 8 ops, 8.83 µs | **31 ops, 37.86 µs** | **+29.03** |
| whole traced layer | **1,981.06 µs** | **1,732.35 µs** | **−248.71 (−12.6 %)** |

Per norm, `Cores=1` → `Cores=8`: **44.12 → 7.31 · 44.20 → 8.03 · 44.08 → 7.56 · 43.98 → 7.94 · 34.67 → 6.69 ·
44.02 → 7.90 · 39.85 → 7.67 · 44.05 → 7.64.** A flat **5.5–6.0×** on each. The three cheap norms (9.58 / 9.52 /
8.06) were **left on one core** — correctly: at 8–10 µs they would not repay a reshard.

**The reshard cost is the real price of the change and it is visible:** 15 new `InterleavedToSharded` and 8 new
`ShardedToInterleaved` calls, +29.0 µs. Net −248.7 µs, which matches the harness's 2.0118 → 1.7721 ms = −239.7 µs
to within the noise floor. The op-level and end-to-end numbers close.

## 1.3 The grid ladder — and v3 beat v2's grid on **both** kinds

All 33 candidate measurements, from `measurements/`. `residual_norm_policy_code` in each `.policy.json` encodes
the core count (`800` = 8 cores, `1100` = 11).

| rung | `full_attention` ms | vs 2.0118 | `sliding_attention` ms | vs 1.8242 |
|---|---:|---|---:|---|
| 1 (incumbent) | 2.011792 | — | 1.824205 | — |
| 2 | 1.835801 | −8.76 % | 1.648651 | −9.62 % |
| 4 | 1.794113 | −10.83 % | 1.601660 | −12.20 % |
| **8** | **1.772132** | **−11.92 % ← kept** | 1.583809 | −13.18 % |
| **11** | 1.773512 | −11.85 % | **1.581980** | **−13.28 % ← the kind's best** |
| 22 | 1.779293 | −11.55 % | 1.585306 | −13.10 % |
| 44 | 1.775298 | −11.75 % | 1.586481 | −13.03 % |
| advised plan verbatim | 1.796079 | −10.72 % | 1.601292 | −12.22 % |
| 7 × single-norm ablation | 1.8008–1.8091 | | 1.6080–1.6179 | |
| **88 — the advised grid, and v2's winner** | **never tried** | v2 measured **1.7767** | **never tried** | v2 measured **1.5873** |

Two things follow, and the first is a straight v3 win:

- **v3's grid is better than v2's on both kinds.** full: **1.772132 (8 cores) vs v2's 1.7767 (88)**, −0.26 %.
  sliding: **1.581980 (11 cores) vs v2's 1.5873 (88)**, −0.33 %. The ladder did exactly what it was added for —
  the advisor's 88 is not the optimum, and the response is flat from 8 cores up, so the value is entirely in the
  *first* step off one core. That reproduces the shakedown's threshold finding on a second model.
- **88 is missing from the ladder.** It stops at 44. The advised value and v2's shipped value were the one rung
  never measured — which is also why §1.5's correctness question cannot be closed from this run's data.

## 1.4 Why sliding shipped nothing: **one** PCC sample vetoed **seventeen** measurements

All 17 sliding measurements carry `verdict = rejected_kind_by_absolute_oracle`. Sixteen of them carry, verbatim:

```
"oracle_pcc": "not_retested; sliding kind vetoed at its best measured rung"
```

Exactly one has a number — `ladder_11_sliding`, the kind's fastest at 1.581980, at **PCC 0.99457296** against a
0.995 bar. So the mechanism is:

> build the oracle on the kind's **best** rung → it misses the bar by **0.0003** → **generalise the verdict to
> the entire layer kind** and stop.

⚠ **This corrects two earlier entries of mine.** [`PCC-BY-GRID`](ADVCHAL-V3-PCC-BY-GRID.md) §2 and
[`DEVIATIONS`](ADVCHAL-V3-DEVIATIONS.md) §3.2a both said the cell *"tried one rung and stopped searching that
kind"*. **The ladder was fully swept — six rungs, seven ablations, the advised plan, 17 measurements.** The
performance search was complete and correct. What failed is narrower and worse: **the correctness verdict from
one configuration was applied to sixteen others that were never tested.** Two of the untested rungs (8 and 22)
are within 0.25 % of the tested one and may well pass. That is a defect in how the oracle is *scoped*, not in how
the ladder is *swept*, and it is the single most expensive line in the run.

**What it cost:** 1.581980 vs 1.824205 = **−242.2 µs/layer × 25 layers = −6,055 µs/model**, on a cell that
shipped −1,198. That one veto is **73 % of the entire v3-vs-v2 gap across all five comparable cells.**

## 1.5 The correctness question — settled for the op, and v2's own data confirms it

Whole-layer PCC on the sliding kind, each from its own artefacts:

| version | grid | source | **whole-layer PCC** | verdict |
|---|---:|---|---:|---|
| v3 | 1 *(incumbent, interleaved)* | `evidence_sliding_attention.json` | **0.99962801** | baseline |
| **v2** | **88** | `oracle/norm88/pcc_layer0_sliding_attention_shared1.json`, `decode_pcc` | **0.99962934** | **shipped** |
| v3 | **11** | `evidence_sliding_attention.json` | **0.99457296** | **vetoed the kind** |
| v3 | 2, 4, 8, 22, 44 | — | **never measured** | vetoed by inheritance |

**v2's 88-core candidate scores the same as v3's untouched incumbent — 0.99962934 against 0.99962801, a
difference of 1.3 × 10⁻⁶.** So at 88 cores the norm re-grid is, at layer scope, a numerical no-op. That is an
*independent confirmation of the sweep below* from v2's own oracle file, and it makes the 11-core 0.99457 an
outlier against both the isolated op and v2's whole-layer measurement at a different grid.

⚠ **One caveat that cannot be closed from the artefacts.** v2's oracle records its scope —
`decode_current_pos=32, sequence_length=32, shared_physical_cache=true`, plus a separate
`prefill_pcc=0.99880996`. **v3's sliding oracle records no such scope**, only the value: this is RUN-LOG **P4**,
`oracle_reference` left advisory rather than CRITICAL. So "11 cores flips something 88 does not" and "the two
oracles evaluate different positions" both remain live, and **the artefacts cannot distinguish them.** That is
the cost of P4, priced at 6,055 µs/model.

**✅ Settled — the sweep was run.** [`NORM-GRID-SWEEP`](ADVCHAL-V3-NORM-GRID-SWEEP.md) measured this op at
1/2/4/8/11/22/44/88 cores, in four rectangles the model cannot emit, at three `subblock_w` values, with and
without a weight, at dynamic ranges to 1 × 10¹² — **79 configurations.** Every one returns
**PCC 0.999998–1.000000**; the worst deviation anywhere is **7.3 × 10⁻⁷**, and 11 and 88 are numerically
indistinguishable from each other and from the interleaved incumbent.

**So neither version's 0.9990 nor 0.9946 is about this op.** Both are whole-layer numbers, and the op contributes
≤ 10⁻⁷ of a 5.06 × 10⁻³ gap. The remaining mechanism is **discontinuity, not rounding**: the router norm feeds
`ttnn.topk`, and a 10⁻⁷ perturbation near a routing boundary flips an expert selection — which is why the drop is
a step to the same value at every rung rather than a slide. **A 0.995 whole-layer bar on a sparse-MoE decoder
measures expert-selection agreement and cannot gate a placement change.**

One more thing the sweep found: **88 cores costs 69.15 µs against the interleaved incumbent's 44.68** on this op
in isolation — the advised grid is **55 % slower than doing nothing**, and the slowest of all 79. The cheapest is
4 cores as **2×2 at 19.71 µs**, a rectangle the model's `(C,1)`/`(11,h)` rule cannot produce. But the isolated
ranking and the end-to-end ladder **disagree** (end-to-end preferred 8 and 11, with 2 and 4 worse), which is the
deeper point: **per-op cost does not compose, so a per-op advisor cannot be trusted to pick the grid.** That is
the argument for the ladder, measured.

## 1.6 What v3 did better here, stated plainly: it could see the op at all

The per-op evidence each version committed for this cell:

| | rows | device µs in the CSV | layer µs | coverage | `LayerNorm` rows |
|---|---:|---:|---:|---:|---:|
| **v2** `tracy/norm88_sliding_ops.csv` | 25 | **138.82** | 1,789 | **7.8 %** | **0** |
| **v2** `tracy/norm88_full_ops.csv` | 30 | **303.79** | 1,981 | **15.3 %** | **0** |
| **v3** `tracy/sliding_attention/incumbent_ops.csv` | 74 | **1,789.13** | 1,789 | **100.0 %** | **11** |
| **v3** `tracy/full_attention/incumbent_ops.csv` | 76 | **1,981.06** | 1,981 | **100.0 %** | **11** |

Both v2 CSVs sum to 100 % of their own `Total %` column, so they are complete traces of a *fragment*. **Neither
contains a single norm op — the op v2 shipped.** That is the mechanical reason its reconciliation reported a
**0.000 µs advisor-attributable ceiling** and its `rms_norm` rows read `below_threshold`, and it is why the
step-0 capacity metric built from those artefacts returned a 9.6 % "upper bound" on a cell that had already
delivered 13.0 % ([`RESULTS`](ADVCHAL-V3-RESULTS.md) §1b).

So on this cell the two versions failed at **opposite ends**: v2 shipped the right change with no op-level
evidence for it; v3 produced complete op-level evidence, found a better grid than v2's on both kinds, and then
threw the larger kind away on an untested inference.

---

# 2. phi-3.5 `nofuse-noadvise` — same idea, different code, and v3's is both slower and wrong

## 2.1 The baseline is the same execution, op for op

| | harness incumbent | per-op rows | per-op total |
|---|---:|---:|---:|
| v2 (`incumbent_perf_report.csv`) | 0.788610 ms | 60 | 699.77 µs |
| v3 (`perf_reports/incumbent/decode_perf_report.csv`) | **0.788347 ms** | **60** | **696.96 µs** |

Same 60 ops, same `(op, cores, memory)` triples, every op within 0.5 %: `Matmul` ×4 on 12 cores L1-width-sharded
215.12 / 214.91 µs; `SdpaDecode` on 110 cores L1-height-sharded 192.19 / 191.94; `UntilizeWithUnpadding` ×4 on 32
cores DRAM 40.62 / 40.59. **The comparison below is between two candidates on a shared baseline.**

## 2.2 The candidates: 0.9990 against 0.9173, on the same knob name

| | v2 | v3 |
|---|---|---|
| measurement | `rope_l1_chain` | `rope_l1_compatible_geometry` |
| median | **0.748709 ms** (−5.09 %) | **0.755911 ms** (−4.11 %) |
| confirmation | 0.748458 | 0.756321 |
| oracle | real weights, recorded target activations, `CACHE_SOURCE reference_fill recorded_target_prefix prefix_length=127 batch=32` | **identical string** |
| **PCC** | **0.998993** | **0.917313** |
| verdict | **shipped** | **`rejected_correctness`** |

Same oracle harness, same cache source, same prefix length, same batch. **The 0.0817 PCC difference is in the
code under test.**

## 2.3 The two implementations, side by side — this is the whole finding

Each cell wrote its own knob into `tt/optimized_decoder.py`. They are not the same knob and not the same code:

| | v2, `advisor_rope_l1_chain` (line 901) | v3, `advisor_rope_l1` (line 885) |
|---|---|---|
| `cos`/`sin` placement | `to_memory_config(cos, **query_memory_config**)` — the query's own **sharded** config | `to_memory_config(cos, **ttnn.L1_MEMORY_CONFIG**)` — L1 **interleaved** |
| the two half-slices | `memory_config=ttnn.L1_MEMORY_CONFIG` | `memory_config=ttnn.L1_MEMORY_CONFIG` — **identical** |
| `concat(neg(second), first)` | L1 interleaved, **then `to_memory_config(rotated, query_memory_config)`** | L1 interleaved, **left interleaved** |
| `multiply` / `add` | `memory_config=**query_memory_config**` — sharded arithmetic | `memory_config=**ttnn.L1_MEMORY_CONFIG**` — interleaved arithmetic |
| what is returned | `query, ttnn.to_memory_config(key, **key_memory_config**)` — **each tensor restored to its own layout** | `apply_l1(query, **rope_memory_config**), apply_l1(key, **rope_memory_config**)` — **both restored to the same layout** |

Neither version implements the advised `[32,64]` output shard on the slices — v3's own comment says tt-metal
rejects it and points at `isolate_advised_rope_slice.py`. So both substituted. **The substitutions differ in
exactly two places, and each place explains one of the two gaps:**

- **the slower 0.96 %:** v2 does the `multiply`/`add` in the query's 32-way height shard; v3 does them
  interleaved and then converts. v2's profile shows it directly — `BinaryNgDeviceOperation` ×6 at
  **`L1_HEIGHT_SHARDED`, 13.25 µs**, where the incumbent had ×7 at `DRAM_INTERLEAVED`, 25.71 µs. v3's path
  cannot produce that row.
- **the 0.917:** v2 captures `key_memory_config` before the transform and restores the key to **it**. v3 returns
  **both** query and key in `rope_memory_config` — the query's. A consumer reading the key under the wrong shard
  spec is the shape of a 0.917, and it is one line.

## 2.4 What v2's change did per op, for scale

`incumbent_perf_report.csv` → `candidate_rope_l1_perf_report.csv`, 699.77 → **624.04 µs (−75.73)**:

| op | incumbent | v2 candidate | Δ |
|---|---|---|---:|
| `Concat` ×2 | 24.47 µs `DRAM_INTERLEAVED` | 10.26 µs `L1_INTERLEAVED` | **−14.21** |
| `Slice` ×4 | 19.70 µs `DRAM_INTERLEAVED` | 6.44 (×2 DRAM) + 5.33 (×2 L1) | **−7.93** |
| `BinaryNg` ×7 | 25.71 µs `DRAM_INTERLEAVED` | 13.25 (×6 **`L1_HEIGHT_SHARDED`**) + 7.28 (×1 DRAM) | **−5.18** |
| `UntilizeWithUnpadding` ×4 | 40.62 µs `DRAM_INTERLEAVED` | 36.56 µs `L1_INTERLEAVED` | **−4.06** |
| `TilizeWithValPadding` ×4 | 15.33 µs `DRAM_INTERLEAVED` | 13.99 µs `L1_INTERLEAVED` | **−1.34** |
| `Permute` 64 ×4 / 96 ×2 | 23.43 + 15.13 µs | dropped out of the trace | **−38.6** |
| `Matmul` ×4, `SdpaDecode` | 215.12 / 192.19 µs | 215.27 / 191.67 µs | ±0.2 — **untouched** |

The change is **pure placement**: the arithmetic ops are bit-for-bit the same cost, and everything saved is
DRAM→L1 plus the elimination of six `Permute` calls. Which is precisely what the advisor advised, and precisely
what v3 substituted away from.

## 2.5 So [`DEVIATIONS`](ADVCHAL-V3-DEVIATIONS.md) §3.2's conclusion is wrong and is now corrected

§3.2 concluded that phiB's `no_change` was **correct**, because *"the only faster thing it found breaks the
model"*, and left open whether v2's −5.74 % had shipped the same broken geometry. **It did not.** v2's
implementation is on file at PCC **0.998993** with a −5.09 % measurement and a fresh-process confirmation. The
correct reading is:

> **v3's `no_change` is right about its own candidate and wrong about the cell.** It rejected a defective
> implementation of a change that is achievable, achieved, and correct in the other version's tree — and the
> defect is two lines of its own substitution, not a property of the model or of tt-metal.

---

# 3. phi-3.5 `fuse-noadvise` — v3 shipped a different change because it could not express v2's

No layout dispute here; the two versions shipped different things.

| | v2 | v3 |
|---|---|---|
| knob shipped | `advisor_rope_l1: str = "query_key"` | `input_norm_cores: int = 11` |
| incumbent → final | 0.807152 → **0.767542** | 0.806756 → **0.798063** |
| Δ per layer | **−4.91 %** | **−1.08 %** |
| Δ per model | **−1,267.5 µs** | **−278.2 µs** |
| oracle kind | **differential**, reference = *"frozen incumbent"* | **absolute**, reference = Torch functional decoder |
| PCC reported | **1.0** — against itself | incumbent **0.9989039**, candidate **0.9989078**, differential 0.9999986 |

**Why v3 shipped the smaller change.** Its `inexpressible[]` names the blocker with a price:
`dense:3, profile_cost_us 16.759, advisor_removes_us 14.636, reason: capture-substituted rope concat/multiply
placement`. The capture monkey-patches `_decode_rope`, so the advice for that region is advice about a stand-in
and there is no bridge back to the real decoder. v2 got there by hand-writing the knob — which is a capability
v3 deliberately does not have, and the corpus's own STG-9.

**And v3's oracle is strictly stronger.** v2's PCC **1.0 is against the frozen incumbent** — a differential
oracle, which by construction cannot detect that the incumbent and candidate are *both* drifting from the model.
v3 reports the absolute number for both sides (0.9989039 / 0.9989078) and shows the candidate is very slightly
**more** accurate than the incumbent. So v2's −4.91 % is a real speed result whose **absolute** correctness is
not established to v3's standard, and v3's −1.08 % is a smaller result that is.

**Net:** −989 µs of the gap, of which the recoverable part is a **capability** gap (express the rope placement
without the capture substitution), not a decision defect.

---

# 4. What each version did better, in one table

| | v2 | v3 |
|---|---|---|
| **per-op coverage** | g26onA: 25/30 rows, **7.8 % / 15.3 %** of the layer, **zero norm rows** | **100 %** of both layers, all 11 norms, per-call µs and core counts |
| **grid search** | the advised grid only (88) | **6-rung ladder + 7 ablations + advised verbatim, both kinds**, and it found **better grids than v2 on both** |
| **grid actually shipped** | 88 cores | **8 cores — 0.26 % faster than v2's 88** |
| **oracle** | phiFN **differential vs itself, PCC 1.0**; phiB absolute 0.9990 | **absolute on every cell**, both sides reported, bar sourced to the model's own test file |
| **decision recording** | shipped without op-level evidence for the shipped op | every rejection carries a verdict; this document is buildable *because* of that |
| **decision quality** | shipped the big win on both kinds | **vetoed 17 sliding measurements from 1 untested inference — 73 % of the total gap** |
| **implementation fidelity** | hand-wrote a sharded, correct rope chain | substituted an interleaved one that is slower and PCC 0.917 |
| **what it cannot do** | see the op it changed | express the rope placement (capture substitution) |

**The pattern across all three cells: v3's *measurement* is better than v2's everywhere, and v3's *action* on
its own measurements is worse.** Two of the three losses are decisions taken against data v3 had in hand
(§1.4, §2.3); the third is a capability it never had (§3). None of the three is a case of v2 measuring something
v3 could not find.

# 5. The four changes this file argues for

1. **An absolute-oracle rejection must not propagate to untested configurations** (§1.4). Test the rungs, or
   record them as `not_screened_after_correctness_rejection`. Worth ~6,055 µs/model on one cell.
2. **Put the advised grid and the prior version's shipped grid on the ladder** (§1.3). 88 was the one rung never
   measured and it is the rung both open questions turn on.
3. **A substituted geometry must be diffed against the advice and against the prior implementation before it is
   measured** (§2.3). Two lines separate −5.09 % at PCC 0.9990 from −4.11 % at PCC 0.917.
4. ~~Isolated single-op PCC sweep~~ — **done**: [`NORM-GRID-SWEEP`](ADVCHAL-V3-NORM-GRID-SWEEP.md), 79
   configurations. §1.4's veto was **simply wrong**, not over-reaching: the op cannot produce the deviation it was
   rejected for, by a factor of 6,879.
5. **Gate a placement candidate on the op's own output**, not only on the layer's PCC — the change the sweep
   argues for, and the one that would have prevented §1.4.
