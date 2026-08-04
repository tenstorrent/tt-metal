# advchal-v2 — zooming out: what is the advisor actually worth?

The stage measures *"how much faster did the decoder get when we followed the advisor?"* This file asks a
different question: **for the wins that were found, was the advisor necessary?**

Three independent tests, all from measured data. Then the accounting defect that hides the advisor's one real
strength, two starved op classes it steers away from, and what 7.4 hours of device time bought.

Companion to [`ADVCHAL-V2-COUNTERFACTUALS.md`](ADVCHAL-V2-COUNTERFACTUALS.md).

---

## 1. Was the advisor needed to *find* the win?

The corpus's four win cells all turned on the same thing: a reduction stuck on ≤2 cores. Two detection rules
over the cells' own per-op profiles:

| rule | what it looks at | cells flagged | win cells caught | non-win cells flagged |
|---|---|---|---|---|
| **A** | shipped grid ≤2 cores **AND the advisor wants more** AND op ≥2 % of window | 5 | **4 of 4** | 0 |
| **B** | shipped grid ≤2 cores AND op ≥2 % of window — **no advisor at all** | 7 | **4 of 4** | 2 |

**Both find every win.** The advisor's contribution to detection is *precision* — narrowing 7 candidates to
5 — not recall. Rule B needs only the shipped profile, which the stage already has before it runs the advisor
at all.

---

## 2. Was the advisor needed to *choose the grid*?

Three selectors, scored on the ladders I measured:

- **advisor** — the grid `ttnn-advise` recommended
- **dumb** — the legal grid closest to **16 cores**, chosen with no reference to the advisor whatsoever
- **best** — the fastest legal grid (hindsight)

| cell | control ms | advisor | dumb (~16c) | best legal | dumb − advisor |
|---|---|---|---|---|---|
| phi FN | 0.808757 | 11 → −7.66 % | 16 → −7.47 % | 11 → −7.66 % | +0.19 pp |
| nm FN | 0.577971 | 22 → −6.04 % | **16 → −11.28 %** | 16 → −11.28 % | **−5.25 pp** |
| g26 onA | 1.823508 | 88 → −12.94 % | **11 → −13.67 %** | 11 → −13.67 % | **−0.73 pp** |
| g26 B | 1.258327 | *not expressible in that knob* | 11 → −10.00 % | 22 → −12.44 % | — |

Summed over the three cells where both are defined:

| selector | summed layer-level improvement | share of achievable |
|---|---|---|
| best legal (hindsight) | **−32.61 pp** | 100 % |
| **dumb (~16 cores, no advisor)** | **−32.42 pp** | **99.4 %** |
| the advisor's own grid | −26.64 pp | **82 %** |

**A fixed heuristic that never consults the advisor captures 99.4 % of the achievable win. The advisor's own
recommendation captures 82 %.** It wins the grid choice in 1 of 3 cells, by 0.19 pp.

This is consistent with the mechanism: `LayoutScore` has **no latency term**, and for normalization ops the
core-count term is overridden with a value independent of the candidate
([`ADVISOR-INTERNALS`](ADVCHAL-V2-ADVISOR-INTERNALS.md) §2–3). There is nothing in the objective that could
have made its grid a throughput optimum.

---

## 3. When its advice *was* measured, how often was it right?

Every per-op row in the corpus carrying a verdict that implies a device measurement — 118 of 483 rows:

| advisor's direction | won | lost | hit rate |
|---|---|---|---|
| **FEWER cores** | 38 | 37 | **51 %** |
| **to DRAM** | 19 | 15 | **56 %** |
| **MORE cores** | 1 | 8 | **11 %** |
| **overall** | **58** | **60** | **49 %** |

**A coin flip.** And its "use more cores" advice — the direction that matters for the starved-reduction class —
loses 8 times out of 9 *as recorded*.

By op class, the pattern is that the advisor is right about cheap movement ops and wrong about the expensive
ones: `slice_static` 16–10, `multiply` 14–8, `rotary_embedding` 4–0, `neg` 6–2 … against `linear` **1–8** and
`repeat` **0–8**.

---

## 4. The accounting defect that hides its one real strength

The table above says "MORE cores: 11 %". But the corpus's two shipped double-digit wins *are* "more cores" on
a starved reduction. Why don't they appear?

**Because none of them is recorded as `kept`.** All 37 low-core `rms_norm` rows the advisor wanted widened
(1,030 µs of per-layer op cost across 6 cells):

| verdict | rows |
|---|---|
| `below_threshold` | **30** |
| `not_measurable` | 5 |
| `rejected` | 2 |
| **`kept`** | **0** |

That includes **gemma-4-26B onA** and **north-mini FN** — the two cells that shipped exactly this change and
booked **−12.98 %** and **−10.23 %** for it. Their `rms_norm` rows read `below_threshold`.

The cause is the ceiling: it prices only *boundary conversions the advice does not place*, so an in-chain
re-grid is worth 0.000 µs and lands below any screening threshold
([`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E12). Three consequences:

1. **A reader of the per-op data would conclude this class was screened and found worthless** — the exact
   opposite of the truth.
2. **It is visibly how gemma-4-26B B missed its −12.44 % win**: its rows say `below_threshold` for 1→88. The
   cell looked, priced them below threshold, and moved on.
3. **The advisor's hit rate is computed on the wrong population** — only the boundary candidates, where it is
   a coin flip. The direction it gets right is unrepresentable.

**So: at the cell level the advisor's "widen this starved reduction" direction was validated in 4 of 4 cells
where anyone measured it. At the op-row level its own accounting records it 0 times out of 37.**

---

## 5. Two starved op classes the advisor steers *away* from

`rms_norm` is the **only** starved class the advisor wants widened. Every op sitting on ≤2 cores anywhere in
the corpus:

| op on ≤2 cores | rows | sum µs | cells | max share of window | what the advisor advises |
|---|---|---|---|---|---|
| `rms_norm` | 37 | 1,030 | 6 | 9.48 % | **8 / 11 / 22 / 88 cores — widen** ✅ |
| `nlp_create_qkv_heads_decode` | 13 | 283 | 5 | **9.27 %** | **1 core — keep it starved** ❌ |
| `concatenate_heads` | 2 | 154 | 1 | **7.79 %** | **move to DRAM** ❌ |
| `rotary_embedding` | 18 | 88 | 5 | 2.23 % | 2 cores |
| `embedding` | 20 | 52 | 7 | 0.43 % | DRAM |
| `typecast` | 5 | 4 | 5 | 0.15 % | 1 core |

Filtering to "advisor wants it widened and it is ≥1 % of the window" leaves **`rms_norm` alone** — 23 rows,
925 µs.

So the two next-largest starved classes — one at **9.27 %** of a layer window, one at **7.79 %** — are ops the
advisor explicitly declines to widen. Nobody screened either in v2, because both are recorded as agreement or
DRAM-advice and neither appears in the ranked worklist.

**This is the corpus's largest untested hypothesis:** the starved-reduction class paid 6–13 % per layer every
time it was measured, and there are two more starved classes of comparable share that were never tried
*because the advisor said no*. v1's analysis independently flagged `concatenate_heads` on one core as its
single largest miss.

---

## 6. What 7.4 hours bought

| cell | hours | | cell | hours |
|---|---|---|---|---|
| gemma-4-12B | 1.23 | | g26 FN | 0.50 |
| nm FN | 0.78 | | gemma-4-26B B | 0.45 |
| qwen B | 0.64 | | phi FN | 0.34 |
| nm B | 0.62 | | llama-8B / llama-1B | 0.33 / 0.31 |
| g26 onA | 0.57 | | nm onA | 0.29 |
| qwen FN | 0.55 | | phi exp17 / A / B | 0.27 / 0.26 / 0.26 |

**Total 7.37 wall-clock hours, 123 MB of transcripts, 15 cells.**

| | µs/model | per hour |
|---|---|---|
| what the stage shipped | 13,601 | **1,847 µs/h** |
| what was reachable from the same directions | 20,225 | 2,746 µs/h |

The static cliff check in §1 costs **zero device time**, runs on data `reconcile.py` already produces, and
flags every cell that had a win.

---

## 7. So what is the advisor for?

Stating it as narrowly as the evidence allows.

**What the evidence does not support:**

- that it is needed to *find* the corpus's dominant win class — §1, a profile-only rule finds all four;
- that it is needed to *choose the grid* — §2, a fixed ~16-core heuristic gets 99.4 % vs its 82 %;
- that its measured advice is better than chance on boundary candidates — §3, 49 %.

**What it did contribute, concretely:**

- **Precision in detection.** Rule A flags 5 cells where rule B flags 7 — it filters out ops that are on few
  cores for a good reason. On a larger corpus that could matter a lot more than it did here.
- **The direction on starved reductions, 4 for 4.** Wherever it said "widen this reduction", widening it won —
  even though its specific number was second-best twice. It is a good *detector of a defect* and a poor
  *chooser of a fix*, which is exactly what an objective with no latency term should be.
- **Naming the legality walls early.** `Sharded output not supported for GQA`,
  `nlp_concat_heads_decode requires a sharded input`, `Cos must be sharded in decode mode` — the advisor's
  `unfixable_ops` and the constraint errors it surfaces saved cells from blind alleys.

**The honest framing:** on this corpus the advisor is a **defect detector with a broken cost model**, and the
stage that measures it is priced to reward the half of its output that is a coin flip while recording the half
that works as `below_threshold`. The cheapest improvements are therefore not to the advisor at all — they are
to the detection rule (§1), the grid sweep (§2), and the accounting (§4).

⚠ **Scope.** 15 cells, 5 model families, one host, decode only, batch 32 / batch 1. Every quantitative claim
here is measured, but "the advisor is not needed for this class" is a claim about *this* class on *this*
corpus — the starved-reduction pattern happened to dominate. A corpus where boundary conversions dominated
would score it differently, and §5's untested classes could change the picture again.
