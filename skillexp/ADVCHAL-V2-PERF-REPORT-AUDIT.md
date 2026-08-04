# advchal-v2 — the perf report the stage runs and then throws away

Four questions, answered from the artefacts and the tool's source:

1. Does `tt-perf-report` give a per-op *% of total time*, and a compute-vs-movement split? **Yes — and more.**
2. Are its outputs saved? **The CSV always; everything else in 4–5 of 14 cells.**
3. Is any of it used? **No. The stage reads three columns.**
4. How does each cell score on "non-compute should be minimal, compute should use sensible cores"?
   **Scorecard in §3.** Mean non-compute **18.5 %**; worst cell **53.3 %**.

---

## 1. What the tool actually produces

`tt-perf-report` (`python_env/.../tt_perf_report/perf_report.py`) emits four things. Flags:
`--group-by {op,memory,category}`, `--summary-file`, `--stacked-csv`, `--no-advice`, `--no-stacked-report`.

**(a) A per-op table**, whose CSV form carries — per op —
`ID, Total %, Bound, OP Code, Device, Device Time, Op-to-Op Gap, Cores, DRAM, DRAM %, FLOPs, FLOPs %,
Math Fidelity, Output Datatype, Input 0/1 Datatype, DRAM Sharded, Input 0 Memory, Inner Dim Block Size,
Output Subblock H, …`

So **`Total %` per op and `Cores` per op are already in an artefact every cell saved.**

**(b) A stacked report** with exactly the compute-vs-movement classification, e.g. on phi FN:

```
Total %  Op Code                                              Device Time Sum  Op Count  Op Category
29.11 %  MatmulDeviceOperation (in0:dram_interleaved)              8,493.99 μs       100      Compute
10.85 %  MatmulDeviceOperation (in0:width_sharded)                 3,167.22 μs        85      Compute
 8.16 %  SdpaDecodeDeviceOperation (in0:height_sharded)            2,382.46 μs        53        Other
 7.19 %  PagedUpdateCacheDeviceOperation                           2,099.05 μs        64           DM
 5.00 %  PermuteDeviceOperation                                    1,458.89 μs       249           TM
 4.39 %  TilizeDeviceOperation                                     1,280.61 μs         5           TM
 3.81 %  UntilizeWithUnpaddingDeviceOperation                      1,113.10 μs       114           TM
 3.65 %  SliceDeviceOperation                                      1,065.33 μs       290           TM
```

`Op Category` ∈ **`Compute` / `TM` (tensor manipulation) / `DM` (data movement) / `Other`**.

**(c) A roofline summary** — `99 GB/s, 19.0 %` of DRAM peak for the window.

**(d) Per-op advice, on by default.** Verbatim examples from a corpus CSV:

```
- Increase grid size (currently using 2.0)
- No output subblock size found
- Use HiFi2 or HiFi4 with BF16 activations for improved accuracy
✅ Optimized
```

**The tool already tells you which compute ops are under-parallelised.** That is the second half of the
rubric, produced for free, in the same run.

---

## 2. What the stage keeps, and what it uses

| artefact | cells that have it |
|---|---|
| bounded per-op CSV (`--csv`) | **14 / 14** |
| terminal output (`.txt`) | 4 / 14 |
| `--summary-file` | 5 / 14 |
| `--stacked-csv` | 4 / 14 |
| `.png` stacked chart | 8 / 14 |

**And what consumes them: only `reconcile.py --perf`, which reads three columns** —
`OP CODE`, `DEVICE KERNEL DURATION [ns]`, `CORE COUNT`.

Searched `SKILL.md` and `02b-advisor-challenger.check.sh` for `--group-by`, `category`, `--summary-file`,
`--stacked-csv`, `roofline`, `advice`: **no matches.** The gate never asks for a summary or a stacked report;
the skill mentions `tt-perf-report` only as "the CSV".

**So the stage runs a full performance analysis, saves a fraction of it, and reads three columns out of the
fraction.** `Total %` is sitting in the file it does read, unused.

---

## 3. Scorecard: how each cell scores on the rubric

Classified with the tool's own `OPERATION_CATEGORIES`, on each cell's **bounded decode window**:

| cell | Compute % | **TM %** | DM % | Other % | cost-weighted cores |
|---|---|---|---|---|---|
| phi B | 38.6 | **23.3** | 1.9 | 36.2 | 25.2 |
| **qwen B** | 42.6 | **53.3** | 1.5 | 2.6 | 44.1 |
| phi A | 49.0 | **26.3** | 11.4 | 13.3 | 24.8 |
| phi FN | 52.9 | **21.5** | 10.9 | 14.8 | 60.4 |
| phi exp17 | 54.5 | 17.1 | 7.0 | 21.4 | 52.3 |
| g26 B | 75.0 | 8.6 | 2.4 | 14.0 | 28.3 |
| gemma-12B | 78.3 | 4.3 | 2.7 | 14.7 | 56.8 |
| qwen FN | 78.5 | 8.5 | 4.5 | 8.4 | 20.4 |
| **llama-1B** | **80.7** | **1.1** | 8.6 | 9.6 | 17.1 |
| **llama-8B** | **81.4** | **0.9** | 5.4 | 12.3 | 15.3 |
| **mean** | **63.2** | **16.5** | **5.6** | **14.7** | **34.5** |

**The rubric predicts the corpus's outcomes better than the advisor does.** The two cells that returned
*honest zeros* after exhaustive screening — llama-8B and llama-1B — are exactly the two with the **lowest TM
(0.9 %, 1.1 %) and highest Compute (81.4 %, 80.7 %)**. They had nothing to win because they were already
clean. The cells where the big wins were found or missed (phi at 21–26 %, qwen B at **53.3 %**) are the dirty
ones.

⚠ `cost-weighted cores` must be read with care: a DRAM-sharded matmul reports *storage* cores, and
[E20](ADVCHAL-V2-COUNTERFACTUALS.md) measured that widening those is **+65 %** slower. A low number is not
automatically a defect.

---

## 4. Could the advisor help? Split the non-compute time and it answers itself

| bucket | ops | can a *layout assigner* remove it? |
|---|---|---|
| **layout-induced** | tilize, untilize, tilize-with-val-padding, untilize-with-unpadding, typecast, fill_pad, interleaved↔sharded, reshard, copy | **potentially** — these exist because of layout/dtype choices |
| **graph-structural** | permute, transpose, reshape, reshape-view, slice, concat, split, head create/concat | **no** — the op exists because of how the graph is written |

| cell | non-compute % | **layout-induced %** | **graph-structural %** | biggest single non-compute op |
|---|---|---|---|---|
| phi A | 30.8 | **13.0** | 17.8 | `UntilizeWithUnpadding` 7.1 % |
| phi FN | 30.6 | 10.3 | **20.4** | `NLPCreateQKVHeadsDecode` 7.8 % |
| phi B | 28.3 | 11.1 | 17.2 | `UntilizeWithUnpadding` 5.8 % |
| phi exp17 | 21.7 | 8.6 | 13.1 | `UntilizeWithUnpadding` 4.0 % |
| qwen FN | 16.1 | 4.1 | 12.0 | `ReshapeView` 4.0 % |
| gemma-12B | 15.7 | 2.0 | **13.7** | `ReshapeView` **8.8 %** |
| g26 B | 12.6 | 3.3 | 9.3 | `Transpose` 6.2 % |
| llama-1B | 6.5 | 1.2 | 5.3 | `NLPCreateQKVHeadsDecode` 4.2 % |
| llama-8B | 4.5 | 0.9 | 3.6 | `NLPCreateQKVHeadsDecode` 2.7 % |
| **mean** | **18.5** | **6.0** | **12.5** |

**Answer: about one third advisor-reachable, two thirds not.**

- **6.0 pp of window is layout-induced.** An advisor with a *conversion cost model* could reason about it.
  Today it cannot: `requiresReshard` is a **boolean** and `LayoutScore` has no tilize/element-type term, so an
  819 µs untilize and a 1.5 µs regrid are the same value to it
  ([E22](ADVCHAL-V2-COUNTERFACTUALS.md)). This is action **D0**.
- **12.5 pp is graph-structural.** No layout assigner can remove a `permute` or a `reshape` — that is a
  *rewrite*, and it belongs to `$optimize` or a new tt-mlir rewrite pass, **not** to the advisor. This is what
  qwen's 191 ms `retilize` turned out to be
  ([E21–E22](ADVCHAL-V2-COUNTERFACTUALS.md)): a shape choice, fixed by changing the chain.

One item stands out as possibly trivial: **`ReshapeViewDeviceOperation` costs 8.8 % of gemma-12B's window**
(and 4.0 % of qwen FN's). A *view* reshape should be metadata-only. Either it is materialising, or it is
mis-attributed in the profile. Worth one look; if it is materialising, it is free money.

---

## 5. Best way to attack it — in the requested order of preference

### First: change our approach. Free, immediate, and it would have caught everything

The information is already produced and already on disk. Four changes to the stage, none needing a build:

1. **Require the full report, not just the CSV.** Add `--group-by category --summary-file … --stacked-csv …`
   to the mandated invocation, and have the gate check all three exist. Cost: one flag change.
2. **Put a movement budget in the gate.** Compute Compute/TM/DM/Other from the CSV the cell already saves
   (`Total %` + the tool's own `classify_operation`) and **warn above ~10 % non-compute, fail above ~25 %
   without a recorded reason.** On this corpus that flags qwen B (53.3 %) and all four phi arms (17–26 %)
   immediately, and passes both llamas — which is exactly the right answer.
3. **Rank the profile's own conversion ops by cost, independently of the advisor** (action **B0a**). qwen B's
   191 ms `retilize` was filed as `boundary` with an advisor ceiling of 0.000 µs and therefore never
   actionable. Its own profile says 25 % of the layer.
4. **Surface the tool's advice.** It already prints *"Increase grid size (currently using 2.0)"* per op. Record
   it next to each screening decision; where the stage and the tool disagree, that disagreement is a result.
   (⚠ and treat it as a hypothesis, not truth — it says "increase grid size" for DS matmuls, where E20
   measured widening at **+65 %** slower.)

### Second: tt-mlir optimizer changes — worth it, and scoped to the 6.0 pp

- **D0: price conversions as a cost, not a boolean**, and let `LayoutScore` see element type. This is the only
  change that lets the advisor reason about the layout-induced third at all.
- **D0b: enumerate row-major.** `rowMajorEnabled` defaults `false` and the advisor never sets it; it is a
  one-line change to its option string to find out what difference it makes. Cheap experiment, no rebuild —
  `ShardAdvisor` already accepts `extra_pipeline_options`.
- Both are prerequisites, not solutions: neither would have found qwen's 191 ms, because that is a rewrite.

### Third: tt-metal — one trivial fix worth proposing, and nothing else

**The trivial one, and it is worth doing:** `OPERATION_CATEGORIES` in `tt_perf_report/perf_report.py` is
missing four op codes that are live in this corpus, so their time lands in the meaningless `Other` bucket —
**14.7 % of measured time on average, up to 36.2 % (phi B)**. The whole `Other` bucket is these four:

| unclassified op code | share of the `Other` bucket | should be |
|---|---|---|
| `SdpaDecodeDeviceOperation` | 41.5 % | `Compute` |
| `ReshapeViewDeviceOperation` | 40.4 % | `TM` |
| `RotaryEmbeddingDeviceOperation` | 12.8 % | `Compute` |
| `NLPCreateQKVHeadsDecodeDeviceOperation` | 5.2 % | `TM` |

Four strings. The tool already prints a warning naming each one
(*"Please add to OPERATION_CATEGORIES for proper classification"*), so it is a known gap. With them added, the
category split becomes usable without a wrapper.

**Everything else on the tt-metal side is documented as a recommendation only** and not expected to ship:
tiled-input variants of the conv/recurrent composites, sharded output for GQA SDPA (which gates both two
cells' top candidate and the `concatenate_heads` fix), and `ttnn.sparse_matmul` / mutable-state `ttnn.copy`
tracer support.

---

## 6. The one-line summary

**The stage already measures the thing that best predicts where the wins are, and reads three columns out of
it.** Making the movement budget a first-class gate check costs nothing, needs no build, would have flagged
qwen B's 53 % and passed both llamas, and is strictly more predictive on this corpus than the advisor's own
output ([`ADVISOR-VALUE`](ADVCHAL-V2-ADVISOR-VALUE.md)).
