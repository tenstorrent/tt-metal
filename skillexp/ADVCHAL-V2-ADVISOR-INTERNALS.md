# advchal-v2 — how the advisor actually decides

Why `$shard-advise` / `ttnn-advise` recommends what it recommends, read from the tt-mlir source at the
pinned commit and from the advisor's own decision traces preserved in the cells' artifacts.

This file exists because the corpus kept producing one result — *measure grids above the advised core
count and you often win* — without explaining it. This is the explanation, and one part of it is an
**open question** that the traces do not answer.

**Sources.** tt-mlir at pin `618cd4e75d` (`/home/mvasiljevic/tt-mlir`, no rebuild performed), and the
decision traces committed under
`models/autoports/<model>/doc/advisor_challenger/shard_advise/<kind>/decision_trace/`.

---

## 1. The advisor never runs anything

From `tools/ttnn-jit/_src/shard_advisor.py`, class docstring:

> Trace a ttnn function, run the greedy optimizer, report layout decisions.
> **Analysis-only: never lowers to a flatbuffer and never executes on device.**
> The optimizer itself runs on the mock device from a saved system descriptor.

The pipeline options it sets (`_build_options`):

```
enable-optimizer=true  enable-greedy-optimizer=true
memory-layout-analysis-enabled=true          # at optimization_level >= 2
enable-decision-trace=true  decision-trace-dir=<dir>
system-desc-path=$SYSTEM_DESC_PATH
```

**Consequence.** Every number the advisor produces is analytic. Nothing in its advice has been timed.
That is by design — but it means the stage's premise (*measure the advice*) is not redundant work; it is
the only place a latency fact enters the loop.

---

## 2. The objective function — and what is missing from it

`include/ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h` defines `LayoutScore`;
`lib/Dialect/TTNN/Analysis/OpModelStrategy.cpp` defines its ordering. Higher is better, strictly
lexicographic:

| level | criterion | comment in source |
|---|---|---|
| 1 | `isL1` | L1 > DRAM (highest priority) |
| 2 | `isSharded` | Sharded > Interleaved |
| 3 | `isDRAMShardedCandidate` | "architecturally superior for decode … **regardless of core count**" |
| 3a | `hasCanonicalDSIn0` | tiebreaker within DS: in0 already canonical `1×kNumStorageCores` |
| 4 | `inputDramBytes` **lower** | total bytes transferred from DRAM across all inputs |
| 5 | `requiresReshard` **false** | penalise reshard |
| 6 | `coreCount` **higher** | "More cores > fewer cores" |
| 7 | `outputL1Usage` **lower** | leaves more room for other tensors |

**There is no latency, runtime, or throughput term at any level.** The struct's own comment calls
core count "Primary: more cores = better parallelism", but in the ordering it is the **sixth**
criterion — below avoiding a single reshard.

Two direct consequences that the corpus measured:

- **Level 3 outranks core count**, so a DRAM-sharded matmul wins over an L1-sharded one at any grid.
  This is the mechanism behind the reconciliation quirk documented in
  [`ADVCHAL-V2-PER-OP.md`](ADVCHAL-V2-PER-OP.md): `12→99, agrees` means *both are DS*, not that 87
  cores were left on the table.
- **Level 5 outranks core count**, so the advisor will accept a much narrower grid to avoid one
  reshard — even where the corpus priced a reshard at only 1.4–1.9 µs.

---

## 3. Normalization ops are scored on their *input*, not on the candidate

`lib/Dialect/TTNN/Analysis/OpRules/NormalizationRules.cpp`, `RmsNormRuleBook::adjustScore`:

```cpp
if (isShardedMemoryLayout(ml.getValue())) {
  // Sharded-input path: the sharded layernorm kernel parallelizes across
  // the input shard grid. […] force it to the input grid's volume […]
  // Also clear `requiresReshard` — the 3x+ kernel speedup over the interleaved
  // variant dominates the few-µs reshard cost on decode-shape tensors.
  base.coreCount = ttmlir::utils::volume(operand0Layout.getGridShape());
  base.requiresReshard = false;
} else {
  // Interleaved-input path: LayerNormMultiCoreProgramFactory parallelizes
  // by tile-row count.
  base.coreCount = inputOperandTileHeight(op, /*operandIdx=*/0);
}
```

So for a norm, `coreCount` is **overridden to a value that does not depend on the candidate's own output
grid** — it is the input's grid volume, or (interleaved input) the input's tile height.

`inputOperandTileHeight` = `ceil(shape[-2] / 32) × ∏ leading dims`. **On a decode-shaped activation
`[32, hidden]` that is exactly 1.** Verified in phi FN's trace: every one of the 360 enumerated
candidates for `ttnn.rms_norm` op0 — including `width_sharded/1x96` — carries `"coreCount": 1`.

**Consequence.** For a decode-shape norm, score level 6 is constant across every candidate. It cannot
express "more cores is faster" for the single most valuable op class in this corpus — the class where
the corpus measured 26.1 µs → 5.6 µs (4.65×, north-mini) and 44 µs → single-digit (gemma-4-26B).

*This is my reading of the code, and it is consistent with the traces; I have not run the pass in
isolation to confirm no other code path repopulates the field.*

### Hard constraints on norm layouts (same file)

- **Height-sharded is rejected** by the sharded layernorm kernel
  (source comment cites `layernorm_device_operation.cpp:151`).
- Block- and width-sharded are accepted **only if the grid is a full rectangular bounding box** —
  `isFullBboxSharded` in `LayoutFilterUtils.h` requires `numCores == bbox_num_cores`.

With row-major core placement on Blackhole's 10×11 worker grid (`deriveCanonicalL1CoreRangeSet` →
`buildRowMajorCoreRanges`, mapping shard *m* to core `(m/11, m%11)`), a width-shard of *M* cores is a
full bbox only when **M ≤ 11** or **M is a multiple of 11**. So 12, 24, 32, 96 are not full bboxes.

⚠ **I initially concluded this filter was why 32 cores is never advised. The traces disprove that** —
see §5. The filter is real, but it did not exclude 32 from the candidate sets I inspected.

---

## 4. How the candidate set is built — and pruned

`lib/Dialect/TTNN/Analysis/LegalTensorLayoutAnalysis.cpp`:

1. **Enumerate everything**: DRAM interleaved; L1 interleaved at the full worker grid; block-sharded for
   every `height ≤ 10, width ≤ 11`; height-sharded `1..110`; width-sharded `1..110`.
2. **Deduplicate by shard shape** — `checkIfShardShapeExists`. Because the loops ascend, **the smallest
   grid that produces a given shard shape survives and larger equivalents are dropped.** The source
   calls this "a good heuristic to reduce the search space".
3. **Drop shapes with fewer tiles than grid cells** (`tiledShape[i] < gridShape[i]`), with a `TODO`
   noting the check may be removable but currently causes test failures if removed.
4. **Sort descending by grid volume, keep the top `maxLegalLayouts`** — default **64**
   (`GreedyMemoryLayoutPropagation.h:19`).

**Consequence.** The set of advisable core counts for a tensor is the set of *minimal* grids per distinct
shard shape. For a 128-tile-wide tensor that is `{1..13, 15, 16, 19, 22, 26, 32, 43, 64, 110}` — verified
exactly against llama-8B's trace (§5). Counts absent from that ladder can never be advised, however fast
they measure.

---

## 4a. The validation the candidate set is filtered against is not the runtime's

Every enumerated layout is checked by `op_constraint_validation::validateOperation` against the **op model**,
on a **mock device** built from `SYSTEM_DESC_PATH`. Nothing here executes (§1), so "valid" means "the op model
accepts this", not "this runs".

**Those two are not the same set.** phi's RoPE body, from the decision trace:

| op | evaluations | valid | `height_sharded/32x1` valid? | runs on device? |
|---|---|---|---|---|
| `ttnn.neg` op10 | 296 | **296** | yes (1 of 112 valid HS candidates) | **no** |
| `ttnn.concat` op11 | 512 | 256 | yes | **no** |

The heads are 96 wide and split at 48, so the shards are `(32, 96)` and `(32, 48)`. On device:

```
TT_FATAL: Physical shard shape (32, 48) must be tile {32, 32} sized!
TT_FATAL: Cannot concat interleaved inputs into a sharded output.
          Either shard the inputs first or use an interleaved output memory config.
```

**The op model does not enforce the runtime's tile-sized-shard rule.** A 48-wide height shard is therefore
enumerated, validated, scored, ranked first, and emitted as advice — and then cannot be run.

Two consequences for reading any `report.json`:

1. **`valid` in the decision trace is a weaker claim than it looks.** 296/296 valid does not mean 296 runnable.
2. It interacts with the pruning in §4: `generateAllPossibleLayouts` dedups by shard shape keeping the smallest
   grid **before** per-op rulebooks run, so the surviving representative of a shard shape can be exactly the
   unrunnable one.

This is a genuine consistency gap between tt-mlir's validation and tt-metal's runtime, and it is separate from
the scoring problems in §2 — here the ranking is doing its job on a candidate that should never have been in
the set. → [`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md) §D6.

## 5. What the decision traces actually show

Selection is a **beam search, width 8**, over the op chain — not an independent per-op argmax. Per-op
cross-products reach 16,320 combinations.

### llama-3.1-8B, dense layer, `ttnn.rms_norm` op12 (the MLP norm)

| | |
|---|---|
| evaluations | 3,120 of a 16,320 cross-product; **3,096 valid** |
| core counts offered | 1–13, 15, 16, 19, **22**, 26, **32**, 43, **64**, 110 — all valid |
| **chosen** | `l1/width_sharded/1x22`, score `coreCount 22, inputDramBytes 0, requiresReshard false, outputL1Usage 12288` |
| shipped by the decoder | **32 cores** |

The advised chain around it:

```
op11  ttnn.add       l1/width_sharded/1x64      <- producer
op12  ttnn.rms_norm  l1/width_sharded/1x22      <- chosen
op13  ttnn.linear    l1/width_sharded/1x90      <- consumer
```

**Three facts worth holding together:**

1. **32 cores was a valid, offered candidate and was not chosen.** So "the advisor cannot express 32" is
   false for this op — the bbox story in §3 does not apply here.
2. **22 matches neither neighbour** (producer 64, consumer 90), so it is not a reshard-avoiding choice
   in any obvious sense.
3. Under the documented ordering, `1x64` should outrank `1x22` on **both** remaining tiebreakers —
   level 6 (64 > 22 cores) and level 7 (`outputL1Usage` 4096 < 12288). It did not win.

**Open question — I could not determine from the trace why `1x22` was selected.** Candidate
explanations I could not confirm or refute: the beam's joint scoring across ops may prefer it for a
downstream reason the per-op score does not show; `adjustScore` may equalise `coreCount` across all
candidates and then some pre-beam ordering decides; or the recorded per-evaluation score may not be the
value actually compared. Resolving this needs the pass run in isolation with logging, which would mean
rebuilding tt-mlir — **not done**, per the standing constraint against rebuilding mid-experiment.

### phi-3.5-mini, dense layer, `ttnn.rms_norm` op0

| | |
|---|---|
| evaluations | 360 of a 1,600 cross-product; **360 valid, 0 invalid** |
| **chosen** | `l1/block_sharded/1x11`, beam score `coreCount 11, outputL1Usage 18432` |
| per-evaluation scores | **every** candidate recorded `coreCount: 1`, `inputDramBytes: 393216`, `requiresReshard: false`; only `outputL1Usage` varied |
| shipped by the decoder | **1 core** |

`outputL1Usage` was *lowest* for the widest candidate (`1x96` → 2048 bytes) and the chosen `1x11` is
18432 — so level 7 did not select the winner either. Same open question as above.

The advised chain shows the norm sitting between two 96-core width-sharded ops:

```
op28  ttnn.add       l1/width_sharded/1x96
op29  ttnn.rms_norm  l1/block_sharded/1x11
op30  ttnn.linear    l1/width_sharded/1x103
```

so the advised plan itself contains a 96 → 11 → 103 re-grid across the norm.

---

## 6. What this explains about the corpus

| corpus observation | explanation from the source |
|---|---|
| Grids above the advised count often win (nm FN 22→**32** for −6.40 %; phi FN 11/12/24 all beat 1 core) | The advised count comes from a lexicographic score with **no latency term**, selected by a width-8 beam over the chain. It is one point in a legal set, not a throughput optimum. Measuring above it is therefore expected to pay sometimes. |
| The advisor sometimes wants **fewer** cores than shipped (llama 32→22, phi 32→11) | Not a "fewer-cores bias" in the ordering — level 6 prefers more. It is the interaction of the minimal-grid-per-shard-shape ladder, the normalization `coreCount` override that ignores the candidate's grid, and the beam. **The mechanism that selects the specific lower value is the open question in §5.** |
| `12→99, agrees` in the reconciliation | Level 3: a DRAM-sharded matmul outranks a normal L1-sharded one *regardless of core count*, so the tool treats a DS-family match as agreement even when grids differ. |
| The advisor accepts a re-grid inside a chain that costs a reshard | For norms, `adjustScore` **clears `requiresReshard` outright**, with the stated justification that the 3×+ sharded-layernorm kernel speedup dominates a few-µs reshard. |
| A ceiling of 0.000 µs next to a measured 236.8 µs/layer win (g26 onA) | The ceiling counts *boundary conversions the advice does not place*. Nothing in `LayoutScore` or the reconciliation prices a re-grid of an op that stays inside its chain, so that whole class is invisible to the accounting. |
| Height-sharded norm candidates never appear | Explicitly rejected by the layernorm kernel; the rulebook filters them out. |

---

## 7. Actionable, in advisor terms

1. **Add a latency term.** `getOpRuntime` exists in `lib/OpModel/TTNN/TTNNOpModel.cpp` and is already
   wired for many ops, but `LayoutScore` never consults it. A single runtime comparison above
   `coreCount` would let the model rank grids the way the hardware does.
2. **Make the normalization `coreCount` override candidate-aware.** Modelling the *kernel's* actual
   parallelism is right; discarding the candidate's own grid from the comparison is what makes the
   term inert on decode shapes.
3. **Emit the advisable ladder in the report.** The set of legal core counts per op is computable and
   is exactly what a challenger needs to sweep. Every cell in this corpus rediscovered it by trial and
   error, and two cells wasted device time on counts that hard-failed.
4. **Run the op-specific legality filters before the shard-shape dedup**, so the surviving
   representative of a shard shape is one the op can actually accept.
5. **Keep the decision trace as a stage deliverable.** phi FN's is 118 MB uncompressed (7 MB gzipped)
   and is the only artifact that answers "why this grid". llama-8B's was committed **uncompressed at
   51 MB**; two of the traces are the largest single files in the cells' trees.
