# Mistakes made in this analysis, and since corrected

Thirty claims that were published as fact during this work and later retracted, downgraded or re-derived,
grouped by **the error pattern that produced them**. Anyone working from the same artefacts will be tempted the
same way, which is why the patterns are worth more than the individual corrections.

Each entry gives what was claimed, what was true, how it was caught, and **the check that would have caught it
first**. Defects in the stage and the advisor are in [`READ-THIS`](ADVCHAL-V2-READ-THIS.md) §3; this file is
about the analysis.

About 5 of the ~30 recommendations made here were later refuted by follow-up measurement — a useful base rate.

---

## Pattern 1 — I read a summary field instead of the authoritative artefact

The single most productive error class. `$shard-advise` emits both `report.json` (a summary) and
`final_ir.mlir` (what the advisor actually decided). **They do not carry the same information, and the summary
is lossy in ways that are not flagged.**

| I claimed | what was true | how it was caught |
|---|---|---|
| "The advisor advised **22** cores for phi's rope" (and 11/22 generally, across many findings) | **It advised 32.** `report.json` renders `cores=(x0,y0)-(x1,y1)` as the **first range only** of a multi-range `CoreRangeSet`; the real set was `[(0,0)-(10,1), (0,2)-(9,2)]` = 22 + 10. **58.3 % of advised core counts corpus-wide are understated this way** | Cross-checking the `AxB` grid string printed *beside* `cores=` against the decision trace's `beam[0].score.coreCount` — the grid product was right 49/49, the bounding box 5/49 |
| Implicitly, that `report.json` describes the advised layout | **It contains no shard shape at all.** Only `final_ir.mlir`'s `memref<1x2x!ttcore.tile<32x32,bf16>>` does. Not reading it caused Pattern 2 below | Trying to implement the advice and needing a shard shape |
| My own aggregate `advchal-v2-data.json` covered the corpus | **14 rows, not 15** — gemma-4-26B FN missing. Any count from it alone is one cell short | Rebuilding the cell list from the branches instead of from my own file |

**The check:** for anything you intend to *implement* or *quantify*, read `final_ir.mlir`. Treat `report.json`
as a human summary. Specifically distrust `cores=`, and never let a derived field (`advised_cores`) stand in for
the artefact when the artefact is one file away.

| I claimed | what was true | how it was caught |
|---|---|---|
| The capture template's *"`ttnn.sparse_matmul` is terminal in the tracer"* must be stale, since a `sparse_matmul` handler exists at `interception_tracer.py:546` and its `patch_ttnn` installs it | **There are two tracers in `ttnn-jit` and I read the wrong one.** `shard_advisor.py:191` uses `ttnn_emit_tracer.trace_ttnn`, which has no `sparse_matmul` coverage. The template was **correct** | Re-running the capture and reading the *traceback frames*, which name the tracer actually in use |

**The check:** when a component has two implementations of the same role, find out which one runs before
concluding anything from reading either. A traceback names it in one line; source-reading does not.

---

## Pattern 2 — I blamed the tool for my own reconstruction's failure

**The worst error in this corpus, and it survived two revisions.** I escalated it from "the cell took a
shortcut" to "the advice is illegal" to "there is a tt-mlir ↔ tt-metal validation gap" — a claim about two
codebases — on the strength of a probe I had written myself.

| I claimed | what was true |
|---|---|
| The advised `l1/height_sharded/32x1` for phi's RoPE **cannot run**; two `TT_FATAL`s prove it | **Every op in the advisor's real plan runs.** I had used shard **(32,48)** — the logical width. The advisor specified **(32,64)** — padded, tile-aligned. It never specified (32,48) anywhere |
| "The op model accepts a shard the runtime rejects" — a validation gap | **No evidence of any gap.** `OpModel<…>::getOpConstraints` queries tt-metal's *own* constraint machinery. The config it accepted is the config that runs |
| `l1/interleaved` was therefore the only legal placement | **False, and it is the slowest of three legal forms.** The advised plan implemented verbatim is **−10.43 % at PCC 1.0** against the **−4.88 %** the cell shipped |
| Earlier still: the failure was because my probe left the *slices* sharded | Also wrong — a different wrong explanation for the same self-inflicted failure |

Related, same pattern, smaller: I concluded a bounding-box filter in the advisor was why 32 cores is never
advised. **The decision traces disprove it** — the filter is real but did not exclude 32 from any candidate set
I inspected.

**The check, in order — this is the one rule I would most want a future analyst to have:**

1. Read the exact layout from **`final_ir.mlir`**, not from a summary and not from your memory of it.
2. Test the **single op, in isolation, in that exact config**, on device.
3. Only if that fails, look at the optimizer implementation.

**A `TT_FATAL` from your own reconstruction is evidence about your reconstruction.** Assume the stage's fault
before the advisor's, and your own before either.

---

## Pattern 3 — I read tool-recorded guesses as findings

| I claimed | what was true | how it was caught |
|---|---|---|
| Seven per-op rows "do not follow the advice" / "the op was removed" | **Undecidable.** All seven are `pair_confidence: position` — which the tool's own `limitations[]` documents as *"a positional pair is a guess"*. **23.2 % of pairings corpus-wide are positional** | Reading the field I had been printing all along |

**The check:** any field named `confidence`, `pair_confidence`, `limitations`, `note` — read it, and *propagate*
it. If a tool tells you a datum is a guess, it cannot support a finding. It can still support cost attribution,
where a wrong name still lands in the right chain.

---

## Pattern 4 — I trusted the stage's own accounting as ground truth

The reconciliation's buckets are **derived** quantities, computed by `reconcile.py` from fields that are
themselves wrong (Pattern 1). I quoted them as measurements for a long time.

| I claimed | what was true |
|---|---|
| The `chain` bucket is the set of genuine disagreements | **59 of 334 `chain` rows are phantom** — the advisor and the shipped code agree once the core count is corrected. That is **34.4 % of the disagreed-on µs**. It also flipped the headline "how much of the advice was followed" from 10.6 % to **16.2 %** |
| `kept` means the advice was followed | **It means the chain shipped.** Of phi FN's 12 kept chains, 6 implemented only the buffer type and dropped the advised sharding |
| "improved" means the advice was followed | **It does not.** Two of the three largest wins were booked by cells that recorded **0 kept chains** and whose feasibility verdict said `not_measurable` |
| `below_threshold` means "measured and too small" | **It is a self-reported dismissal with no measurement.** 70 of the 134 dismissed chains are **≥5× their own cell's noise floor**, up to 282× |
| `dram_resident` rows are advice — "leave this in DRAM" | **41 of 54 are `unfixable` fallbacks.** The advisor had already declared those ops unplaceable, with the exact `TT_FATAL` |
| An `agrees_with_shipped` row means the placement matches | The test compares **core count or DS-family only — never the memory space.** 1 of phi FN's 2 such rows disagrees on L1-vs-DRAM |

**The check:** recompute the buckets from the artefacts before building anything on them. And when a bucket
name sounds like a measurement (`below_threshold`, `not_measurable`), find out whether anything was measured.

---

## Pattern 5 — I proposed fixes that my own follow-up measurements refuted

These are the healthy ones: hypotheses, tested, discarded. Listed so nobody re-chases them.

| my proposal | refuted by |
|---|---|
| Re-measure an overlapping candidate at **4× replays** to separate it | 250 → 1,800 replays made the noise floor **3–4× worse** (0.4–0.7 → 1.3–3.0 µs) and still did not separate it. Drift does not average down; ~50 replays/block is near optimal |
| Make the advisor **enumerate row-major** layouts (action D0b) | `row-major-enabled=true` yields **zero** row-major layouts. They are already enumerated and then rejected by op constraints. Withdrawn |
| **Screen `ds_family` grid mismatches** — "a DS match at a different grid is a candidate, not an agreement" | Turning off DRAM sharding on north-mini's projections — exactly the advisor's 12 → 77-core direction — is **+65.2 % slower**. DS matmuls are DRAM-bandwidth-bound, so core count is not the limiting resource. **1 win in 7** measured matmul-widening candidates |
| north-mini should have tried **44 and 88** cores | Both are **illegal** (`TT_FATAL` in shard-spec validation). But **16** was legal, untried, and better than what shipped |
| phi FN missed the **32-core exactly-dividing** grid | 11 → 48 cores is a **plateau**. Exact tile division buys nothing there; nothing was missed |
| The advisor's **matmul** advice is a large untapped win | Implemented verbatim on the biggest matmul in the layer: **neutral**, inside the noise floor, both standalone and stacked |

**The check:** treat your own recommendations as hypotheses with a real refutation rate. Mine was ~1 in 6.

---

## Pattern 6 — labelling and attribution errors

Cheap to make, expensive downstream, because everything built on them inherits the error.

| I claimed | what was true | how it was caught |
|---|---|---|
| Arm labels: which cell was `FN` and which was `B` | **Inverted.** `FN` = fuse-noadvise, `B` = nofuse-noadvise | The driver's own `CLAIMED` lines |
| gemma-4-26B: "the fusing arm had already fixed it" | The **fastest** arm is a `nofuse` arm. The variable is stage-02 quality, not fusing | Comparing the arms' controls |
| The advisor has a "**fewer-cores bias**" | Its ordering prefers **more** cores, at level 6 of 7. The low values come from somewhere else — still an open question | Reading `LayoutScore::operator>` in the tt-mlir source |
| `nlp_create_qkv_heads_decode` on 1 core is a **starved reduction** | **Not a defect.** The op height-shards over *batch*, so its core count **is** the batch size — exactly, across all 23 corpus rows (batch 1 → 1 core, batch 32 → 32). The advisor advising 1 is **correct** | Checking the core count against the batch across every cell |
| qwen's unreachable linear layers are "~91 %" of its model time | **97 %** is the layer kind's share of model time | Recomputing from its own per-kind medians and layer counts |
| …and then that 97 % was "never advised on" | **Wrong — the trace stops *inside* the layer.** Of qwen B's 15,833 µs `linear_attention` window, 63.5 % is `untraced`; the residual/norm/MLP envelope around the token mixer **is** captured. The advisor saw ≈36 % of it, so **≈62 %** of the model was never advised on, not 97 % | Reading the layer's own `accounting` block instead of inferring it from "the kind is uncapturable" |
| DS-matmul advice never wins | One **did** — gemma-4-12B, `linear` 12 → 55 cores, kept | Reading the cells' kept lists rather than the skill's claim |

**The check:** before calling a low core count a defect, ask what the op's sharding *semantics* are. And derive
arm/config labels from the driver's own records, never from directory names or memory.

---

## Pattern 7 — provenance and staleness in my own numbers

| the problem | what it looked like |
|---|---|
| I quoted one quantity at **three different values** across sections without saying they were different runs or scopes | phi FN shipped as −4.88 / −4.90 / −4.91 % (three runs); north-mini FN as −9.26 / −10.23 / −10.37 % (three **scopes** — a multi-layer harness, the cell's model estimate, and per-layer). Reads as sloppiness or error; is neither |
| I did not re-derive dependent totals when a component improved | "≈8.0 ms/model on the table" → **9.2**; "20,225 µs reachable / 67 % credited" → **21,368 / 64 %**. Three files carried the stale pair |
| I replayed each cell's **committed** advice onto a decoder I had progressively modified, and reported the result without saying so | Later tested: the advice is **byte-identical** on the diverged graphs, because the advisor discards input memory configs and responds to *topology*. So the replay was valid **for those changes** — but it needed testing, not assuming. A topology-changing edit would not be invariant |

**The check:** state the scope on every number (per-layer vs per-model vs multi-layer harness). When a component
number changes, grep for every total derived from it. And if you apply advice to a graph that has since moved,
either re-advise (**~18 s**) or say plainly that you did not.

---

## What is still unverified

Being explicit, so nobody inherits these as facts.

| open | status |
|---|---|
| Why the advisor chose `1x22` for llama's MLP norm when 32 and 64 were valid and outrank it on both documented tiebreakers | **Unexplained.** The decision trace does not answer it; resolving it needs the pass rebuilt, which I did not do |
| Where the advisor's low core-count values actually come from, given its ordering prefers more cores | Open — see Pattern 6 |
| "What applying everything would have meant" for 14 of the 15 cells | **Analysis from artefacts, not measurement.** Only phi FN was measured |
| Whether the remaining unapplied advised items (qkv `linear`, `o_proj`, MLP `multiply`, cos/sin embeddings) change phi FN's total | **Expectation, not measurement** — the one matmul I did apply was neutral, so I expect little, and say so |
| Whether the advisor would advise differently with a latency term, or on a topology-changed graph | Untested |
| north-mini onA's sparse MoE tail | Untraceable, so never advised on — its headroom is unknown, not zero |
| qwen FN's `linear_attention` kind | Declared tracer-unreachable, so no reconciliation exists. It very likely carries the same ~191 ms/model `retilize` cost qwen B does, **unmeasured** |

---

## For calibration: what held up

So the file is not read as "distrust everything". These survived every subsequent test:

- The **static check** (op on ≤2 cores, advisor wants more, ≥2 % of the window) — used to *predict* gemma-4-26B B's unscreened win before measuring it, and every double-digit win in the corpus is in a flagged cell.
- The **channel-1 / channel-2** split, and that channel-2 wins exist only under traced replay — confirmed on two independent models.
- The **`retilize` 191 ms/model** finding, and that no layout advisor could have found it.
- The **oracle contradiction**, strengthened rather than weakened by a second cell and an absolute reference.
- The advisor's **direction** on starved reductions: right in 4 of 4 cells where anyone measured it.
