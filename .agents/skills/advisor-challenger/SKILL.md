# `$advisor-challenger` — turn a finished optimized decoder into a faster one, using `$shard-advise` as a challenger that cannot lose

**Input:** a *finished, tagged* `optimized_decoder.py` produced by an optimize stage that never ran the
advisor (a `*-noadvise` arm cell).
**Output:** a **new `optimized_decoder.py`** that is measured to be **the best of the incumbent and
everything the advisor suggested** — never worse than the incumbent, by construction.

This skill exists because the advisor was previously used as a **seed** at the *start* of optimization,
and that ordering is what produced every problem worth naming: it biased the search toward whatever it
said first, it froze a dtype it had itself made conditional, and it was never obliged to be measured
against anything. Run *after* a finished decoder instead, it becomes a challenger — and a challenger
with a frozen incumbent and ties going to the incumbent cannot make the model slower. It can only cost
time.

## The invariant — this is the whole point, do not weaken it

```
final_ms <= incumbent_ms          for every measured configuration
```

It holds because of three mechanical facts, not because anyone is careful:

1. the incumbent's latency is **measured and frozen before the advisor runs at all**;
2. every change is measured against that frozen number;
3. **the incumbent wins ties** against a noise floor derived from its own repeats.

If you re-order these, you lose the guarantee. In particular: do not run the capture first "to know
where to look". A capture is cheap to run and expensive to unsee.

## What the advisor is actually good at — use it for that and nothing else

Measured across four models (see `SHARD-ADVISOR-FINDINGS.md`):

| what it produces | track record |
|---|---|
| **the *set* of ops it would shard that you did not** | **right every time** — DS legal on phi's 5 linears; norms should be sharded on qwen (+152 µs/layer, proven) and gemma (363 µs/layer, nobody did it); DS legal on north's gate/up + down |
| the *specific geometry* for those ops | **lost** on phi (−34 µs), **tied** on qwen gate/up DS, **untested-above** on north; neutral ±5 % on its own surface at matched dtype |

**Its precision is poor; its recall is good.** So the query you are running is:

> *"What did I leave interleaved, or on ≤2 cores, that is legally shardable?"*

**not** *"what config should I use?"* Take the **set difference** and derive your own geometry. If you do
sweep the advisor's geometry, sweep **above** its value as well as below — north's sweep that appeared to
vindicate its block widths only tested points *beneath* them on a monotone-improving axis, so its value
"won" by being the largest anyone tried. On phi, where a sweep actually bracketed the advisor's value,
the block width mattered 0.1–0.3 % — inside noise, in both directions.

## Ordered procedure

### Step 1 — Freeze the incumbent (before touching the advisor)

Measure the incumbent decoder as it ships, **3 repeats minimum**, on this machine, with its own perf
harness. Write `doc/advisor_challenger/incumbent.json`:

```json
{ "cell_tag": "...", "commit": "...", "harness": "tests/optimized_decoder_perf.py",
  "decode_batch": 1, "repeats_ms": [1.2721, 1.2734, 1.2719],
  "incumbent_ms": 1.2721, "noise_floor_ms": 0.0015,
  "shipped_policy": { ... }, "shipped_policy_source": "..." }
```

- `incumbent_ms` is the **best** repeat, not the mean — a challenger must beat the incumbent's good day,
  otherwise you ratchet on noise.
- `noise_floor_ms` = spread across repeats. Observed same-config spreads in this corpus were
  **0.2–31 µs**. Any delta inside the floor is a **tie**, and ties go to the incumbent.

**`shipped_policy` must come from what EXECUTED**, and `shipped_policy_source` must name that artifact:
the final `tt-perf-report` CSV columns, or the selected candidate JSON. **Never from
`resolved_policy.constructor_defaults`** — those are the class's default arguments, not the run's
effective config, and reading them as policy is a documented error (gemma's defaults print
`dense_decode_dram_sharded: False` on a cell whose CSV shows `DRAM Sharded = True`).

### Step 2 — Capture, once per layer kind, at the shipped precision

Copy `scripts/capture_template.py`, adapt it per model, and **construct the decoder with the shipped
policy**. This is the single most common defect in existing capture scripts: **3 of 4 call
`from_state_dict(...)` with no dtype/policy argument** and silently trace the class defaults. north
traced `bf16` attention and shipped `bfp8`, so two matmuls were excluded for a dtype the model never
used. `advise_qwen.py` is the only correct existing template — it builds `POLICIES["default"]` and
passes `policy=policy`.

- **One capture per layer kind.** A single capture on one kind lets the whole search follow it there:
  qwen's advise arm ran candidate variants on `full_attention` only, while the `linear_attention` kind
  carrying **48 of 64 layers** got nothing but a default measurement.
- **Pass `--pipeline-options allow-bf16-dram-sharded-matmul=true` whenever a traced weight is bf16.**
  Otherwise DS is declined *by policy, not by capability* — bf16 DS runs at PCC 1.0000. This is exactly
  how gemma got `dram_sharded_considered = 0` on 5 of 5.
- Some ops are **terminal in the tracer** and no sequencing reaches them: `ttnn.sparse_matmul` (all MoE
  experts) and SSM/gated-delta ops (`softplus`, `prefix_scan`, `hc_sum_reduce`, `assign`). Record which
  layer kinds are therefore uncapturable and what share of layers they carry.

### Step 3 — Read the advice correctly

- `final_ir.mlir` is **authoritative** for program configs, required input layouts and the advisor's own
  reverts. `report.json` gives per-op layout and the program-config *family* but **omits block widths and
  `per_core_N`**.
- **`compute_config` / `math_fidelity` in the IR is not advice.** It mirrors whatever the captured graph
  already had. Taking it literally cost north **30.8 µs (−10.7 %)**. Override it with your own
  fidelity choice, always.
- **`dram_sharded_considered == 0` has two causes and you must say which:** (a) a wrong-precision
  capture, or (b) the bf16-DS-off-by-default eligibility gate. OPT-015 names only (a), which sends you
  hunting a capture bug that may not exist. The report states its own reason — quote it.

### Step 4 — Reconcile: build the disagreement list

Run `scripts/reconcile.py`. It diffs advised layout/program-config per op against the shipped graph and
emits `reconciliation.json`, one row per disagreement, each carrying that op's share of the measured
device window so you can rank by what is actually worth measuring.

Rank by window share and apply a **materiality threshold** (default 1 % of the decode window). Below it,
record the row as `below_threshold` with its number — do not silently drop it.

### Step 5 — Measure every material disagreement, one variable at a time

For each row above threshold: build the variant, measure it against the frozen incumbent, record a
number.

- **Change one variable per measurement.** phi's arm rejected the advisor's geometry in an A/B that moved
  core count *and* block width together, and so never learned which one mattered.
- **A prose rejection is not a result.** Every row ends with a measured number or an explicit
  `below_threshold`. qwen's arm rejected the advised RMSNorm sharding in one sentence with no
  measurement; that advice was worth **152 µs per layer**.
- **Do not mix dtype/fidelity policies.** A geometry measured under a different dtype does not reject the
  final dtype policy, and vice versa.
- **Correctness is a gate on every kept change, not a formality.** Re-run the incumbent's own test suite
  at the incumbent's own PCC bar and record which oracle was used. Prefer a real-weight oracle where the
  cell has one; note explicitly when only synthetic weights are available, because diagonal synthetic
  weights provably cannot see some defects (qwen's contiguous-vs-per-head Q/gate split was invisible to
  them). **A faster decoder that fails its oracle is not a result — it is a regression with a good
  number.**

### Step 6 — Iterate, bounded

You may iterate: apply what won, then re-capture and go again. Two rules keep iteration from becoming
the search:

- **Re-capture only after a topology rewrite that changes an op's shape.** A dtype change is not a
  rewrite, and under this ordering dtype is already final.
- **Cap it (default ≤3 captures per layer kind per cell)** and record each capture's trigger in
  `iterations[]`. Extra captures are not free: one advisor seed application hit
  `TT_FATAL: Sharded inputs require sharded outputs` and **wedged PCIe access**, needing a `tt-smi`
  reset; and qwen's second capture returned **byte-identical** program configs — pure waste.

Stop when an iteration produces no material disagreement, or the cap is reached.

### Step 7 — Ship the better decoder

Write the winning configuration into `models/autoports/<model>/tt/optimized_decoder.py` and record:

- `final.json`: `final_ms`, `incumbent_ms`, `delta_ms`, `delta_pct`, the winning config, the oracle that
  passed, and the per-iteration history;
- `README.md`: what was advised, what was kept, what was rejected **with its number**, and what was
  uncapturable.

**A no-change outcome is a valid, publishable result.** If nothing beat the incumbent, ship the
incumbent unchanged and say so with the numbers. That is the honest answer to "does the advisor earn its
cost on this model", and it is the outcome the invariant is designed to make safe. Do not manufacture a
change to look productive.

## Running this as an ordinary pipeline stage

The gate `02b-advisor-challenger.check.sh` is self-contained: it reads only committed-or-working-tree
artifacts under `doc/advisor_challenger/`, takes either `<model_dir>` or `models/autoports/<model_dir>`,
and needs **no environment from any orchestrator**. An earlier revision required a `CHALLENGER_DECODE_BATCH`
env var that only one experiment's driver exported, which made the stage fail for everyone else — a gate
that cannot pass in the pipeline is a wall, not a gate. The batch now comes from the stage's own
`requested_decode_batch`, and an orchestrator that *does* export `CHALLENGER_DECODE_BATCH` gets one extra
cross-check for free.

**What the gate enforces on its own:** the three batches agree; the incumbent was frozen *before* the
capture (`captured_at` after `measured_at`); traced dtypes equal shipped dtypes; `dram_sharded_considered`
of 0 is classified; every material chain has a measured number; `final_ms <= incumbent_ms` with ties to the
incumbent; the oracle passed; iterations are capped and triggered.

**What it cannot enforce, and therefore what an orchestrator must add.** The gate may run *during* the
stage, when artifacts are still uncommitted, so it cannot use git history. Git-level freshness — that this
run built its own artifacts instead of recovering a previous attempt's — has to come from whatever launches
the stage. A previous re-run in this project passed a clean-working-tree preflight and then cherry-picked
its predecessor's commits out of the object store two minutes later, so this is not hypothetical. The three
checks worth wiring in, in the orchestrator:

1. every commit touching `doc/advisor_challenger/` authored inside this stage's own window — cherry-pick and
   rebase preserve author dates, so inherited history is visible;
2. no `cherry-pick` or `rebase` entry in the work branch's reflog;
3. no byte-identical artifact shared with a parked copy of the same cell — this is what catches files
   copied by hand, which have fresh commits but stale content.

`skillexp-logs/run_challenger.sh` implements all three as `challenger_is_fresh()` if you want a reference.

## What this cannot reach — state it, do not discover it

The largest single win in the reference corpus is **not reachable by this skill**: gemma's routed
expert-down grid at 8 cores against a legal 44, worth **+117.7 µs/layer at identical BFLOAT8_B** across
all 30 layers, is `ttnn.sparse_matmul` and terminal in the tracer. If a model's time is in its experts,
this skill will be quiet and correct while the real win sits untouched. Sweeping norm and expert grids
directly, without the advisor, is a separate and often better-paying activity.
