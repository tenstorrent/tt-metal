# advchal-v3 — every core grid that moved PCC, and the candidate tt-metal defects in it

The v2 corpus said this explicitly and nothing collected it: *"some ops compute the wrong answer under
particular shard specs, and changing placement is exactly what triggers such a bug… report the op and the
shard spec as a tt-metal bug. That is a real result of this stage, not a nuisance."*

This is that collection, from all 11 cells of the v3 run. It answers three questions asked of the run.

## 1. When is PCC checked?

**Per candidate, after it is timed.** Every measurement record carries `median_ms` and, where an oracle was
built, `oracle_pcc`. So the order is **measure → oracle → verdict**, and a materially wrong grid gets timed
before it is caught. That is the right order for device cost — you do not pay oracle time on candidates you
will not ship — and it is why the timing data below exists at all.

## 2. When a grid produces a bad PCC, is another grid tried?

**No — and the reason is worse than "it stopped looking".** ⚠ An earlier revision of this section said
gemma-4-26B `-onA` *"tried one rung and stopped searching that kind"*. **That is wrong.** Reading all 33
measurement files shows the sliding ladder was **fully swept**: rungs 2, 4, 8, 11, 22, 44, plus 7 single-norm
ablations and the advised plan verbatim — **17 sliding measurements**.

What happened is narrower and more expensive. All 17 carry `verdict = rejected_kind_by_absolute_oracle`, and
**sixteen of them carry, verbatim:**

```
"oracle_pcc": "not_retested; sliding kind vetoed at its best measured rung"
```

Exactly one has a number: `ladder_11_sliding`, the kind's fastest at 1.581980 ms, at **PCC 0.99457296** against a
0.995 bar. So:

> the oracle was built on the kind's **best** rung → it missed the bar by **0.0003** → **the verdict was
> generalised to the entire layer kind**, and sixteen configurations were rejected without being tested.

**The performance search was complete and correct; the correctness verdict was extrapolated.** Two of the
untested rungs (8 and 22) are within 0.25 % of the tested one. And **88 — the advised grid and v2's winner — is
not on the ladder at all**, which stops at 44.

Cost: 1.581980 vs 1.824205 = **−242.2 µs/layer × 25 layers = −6,055 µs/model**, on a cell that shipped −1,198.
**73 % of the entire v3-vs-v2 gap across all five comparable cells, from one untested inference.**
[`OP-BY-OP`](ADVCHAL-V3-OP-BY-OP-VS-V2.md) §1.3–1.5 has the full ladder and the layouts.

## 3. Do we have every case collected? Not yet, and the reason is a gate of mine

| | |
|---|---|
| measurements in the run | **187** |
| carrying `oracle_pcc` | **60 (32 %)** |
| carrying `op_under_test` | **0 (0 %)** |

`op_under_test {name, incumbent_grid, candidate_grid, legal_ladder}` is the field added specifically to make
"which op, which grid" mechanical — and it is populated in **none** of the 187 measurement records, because
my gate made it **advisory**. The table below is recoverable only because `reconcile.py` fills the same field
inside `cliff_candidates`, which gives **49 rows with an op name and 42 with a PCC**.

**Required change: `op_under_test` and `oracle_pcc` become CRITICAL per screened candidate.** Without them a
grid-dependent correctness bug is visible as a number with nothing attached to it.

---

# 4. ⚠⚠ RETRACTED — the sweep was run and there is no bug here

**[`NORM-GRID-SWEEP`](ADVCHAL-V3-NORM-GRID-SWEEP.md) ran the experiment §4 asked for, on 79 configurations, and
the hypothesis below is false.** The op is clean at every core count, every rectangle, every `subblock_w`, with
and without a weight, at dynamic ranges to 1 × 10¹²: **worst PCC deviation anywhere 7.3 × 10⁻⁷, against the
5.06 × 10⁻³ whole-layer drop it was blamed for — 6,879× too small.** 11 cores and 88 cores are numerically
indistinguishable from each other and from the interleaved incumbent.

**And the "three independent reproductions" below were not independent**: all three cells ran the same candidate
policy, so that is one observation reported three times. The clustering was real; the attribution was mine and
it was wrong. **No tt-metal bug to file.**

The section is kept verbatim below because the *whole-layer* numbers in it are still the measurements, and
because the way the inference failed is the point: **a whole-layer PCC was attributed to a single op with no
per-op evidence, and nothing in the artefacts could contradict it because `op_under_test` was advisory.**

## 4 (superseded). The original hypothesis: `rms_norm` widened off 1 core degrades PCC on the *sliding* kind

**Three independent cells, three different arms, same model, same op, same layer kind, same result:**

| cell | arm | kind | grid | **PCC** | verdict |
|---|---|---|---|---:|---|
| gemma-4-26B `-onA` | nofuse-noadvise-onA | `sliding_attention` | 1 → **11** | **0.99457296** | `rejected_correctness` |
| gemma-4-26B `fuse-noadvise` | fuse-noadvise | `sliding_attention` | 1 → *(unrecorded)* | **0.99467277** | `rejected` |
| gemma-4-26B `nofuse-noadvise` | nofuse-noadvise | `sliding_attention` | 1 → **22** | **0.99469421** | `rejected_absolute_oracle` |

**Clustered within 0.00012 of each other, all just below the model's own 0.995 bar.** Three cells that never
saw each other's artefacts (blob-identity checked, 156 refs parked) produced the same number.

**And the same op, on the same model, on the *other* layer kind, is clean:**

| cell | kind | grid | PCC |
|---|---|---|---:|
| gemma-4-26B `-onA` | `full_attention` | 1 → 8 | 0.99980000 |
| gemma-4-26B `nofuse-noadvise` | `full_attention` | 1 → 22 | 0.99985746 |
| gemma-4-26B `fuse-noadvise` | `full_attention` | 1 → *(unrecorded)* | 0.99989244 |

So it is **not** the widening as such, and **not** the model. It is specific to the **sliding-attention path's
norm**, which points at that path's input — mask, shape or dtype — rather than at the core count.

**Why this is a candidate defect and not just a rejection.** v2 measured its shipped 88-core sliding config at
**0.999629** on this same cell. So 88 cores is clean and 11/22 are not, on the same op and kind. A correctness
result that depends on the *core count* of a reduction, non-monotonically, is the shard-spec bug signature the
corpus described.

**And v2's number for the same kind is now precise: PCC 0.998993 at 88 cores**, from its own `final.json`
(real-weight HF layer reference). So 88 passes at 0.9990 while 11 fails at 0.9946 on the same op, model, kind and
host — if that holds, PCC is **non-monotonic in a reduction's core count**, which is a tt-metal shard-spec
question rather than a stage question.

**What is needed to file it**, and none of it is expensive: an isolated single-op test of gemma-4-26B's
sliding-attention `rms_norm` at 1, **8**, 11, 22, 44 and **88** cores against a fixed reference, on this host,
with the shard shape from `final_ir.mlir` — 8 and 88 added because they are the two grids that decide it. If 11 and 22 degrade while 1 and 88 do not, it is a tt-metal report with a
reproduction. If all of them degrade, then **v2's 0.999629 is the number in question** and the −12.98 % it
shipped needs re-examining.

# 5. The other materially wrong case

| cell | candidate | PCC | notes |
|---|---|---:|---|
| phi-3.5 `nofuse-noadvise` | `rope_l1_compatible_geometry` | **0.91731302** | recorded **twice** (candidate + confirmation), so reproducible |

**0.917 is not reassociation, it is wrong output** — and it was the cell's *fastest* measurement at −4.11 %,
so clause 1 of the oracle earned its place here. But the name is the finding: *`compatible_geometry`* means
the advised geometry was **not expressible**, so the cell built a nearby one — and no `op_under_test` was
recorded, so **the op and grid that produced 0.917 are not in the artefacts.** This is the case the §3 gap
costs us most.

# 6. Everything else, for completeness

The remaining 30-odd rows sit at 0.9995–0.99999 — reassociation, not error. Two are worth noting as the
counter-example that keeps the bar honest:

| cell | op | grid | PCC | outcome |
|---|---|---|---:|---|
| north-mini `-onA` | `rms_norm` | 1 → 16 | 0.99500000 | **kept** — exactly at the bar |
| north-mini `-onA` | `rms_norm` | 1 → 16 | 0.99597765 | kept |
| phi-3.5 `fuse-noadvise` | `rms_norm` | 1 → 11 | 0.99890780 | kept, shipped |
| north-mini `fuse-noadvise` | `rms_norm` | 1 → 22 | 0.99952112 | kept, shipped |

So the same transformation — a norm off one core — is **clean on north-mini and phi at every grid tried, and
degraded on gemma-4-26B's sliding kind at every grid tried.** That contrast is what makes the gemma case worth
filing rather than writing off as a tight bar.

# 7. Actions

1. **`op_under_test` and `oracle_pcc` CRITICAL per screened candidate** — without them this document cannot be
   built from the artefacts, and I had to reconstruct it from a different file.
2. **A correctness rejection must not propagate to configurations that were never tested.** The ladder is
   swept; the oracle is not. Test the rungs, or record them as `not_screened_after_correctness_rejection` so the
   gap is visible. Worth ~6,055 µs/model on g26onA alone.
2b. **Put the advised grid and the previous version's shipped grid on the ladder.** 88 was the one rung never
   measured, and both open questions here turn on it.
3. ~~Isolated single-op test for §4~~ — **done**, [`NORM-GRID-SWEEP`](ADVCHAL-V3-NORM-GRID-SWEEP.md). §4 retracted.
3b. **Gate a placement candidate on the op's own output, not only the layer's PCC** — the change the sweep argues
   for. On a sparse-MoE layer a 0.995 whole-layer bar measures expert-selection agreement, not arithmetic.
4. ~~v2's and v3's sliding norm results are mutually inconsistent~~ — **resolved.** They are not inconsistent
   about the *op*: it is clean at both grids. They differ about a **whole-layer** number that the op cannot have
   caused. v2's 0.9990 and v3's 0.9946 are both plausible readings of a layer whose PCC is a discontinuous
   function of tiny perturbations (expert selection), which is why neither is a valid gate for a placement
   change.
