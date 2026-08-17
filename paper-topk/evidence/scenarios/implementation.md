# MODEL_SCENARIOS layer — implementation report

Branch `nkapre/sorting` @ HEAD 3b7df85bb9eb (uncommitted working tree), Blackhole p150a,
2026-08-16. All device runs serial under `flock /tmp/tt-device.lock` with the sweep's own
900 s per-cell watchdog inside an outer `timeout 7200`. 18/18 cells MEASURED, zero WRONG,
zero FAILED, zero UNSUPPORTED, zero SKIPPED_SLOW (no tier-C/D cell was needed this run).
Provenance uniform across all 18 result JSONs:
`head=3b7df85bb9eb tree_diff_md5=8ad0a891 so_md5=fabc04ea`.

## What was implemented (additive-only)

### 1. `tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py`
- `--model-scenarios` mode per wiring.md: `SCENARIO_ENGINES` /
  `SCENARIO_ENGINE_ORDER` / `SCENARIO_STOCK_NS_PER_ELEM` constants + built-in
  `MODEL_SCENARIOS` grid (8 scenarios, overridable via `--scenarios-file`, subsettable
  via `--scenarios`); `load_scenario_specs` (name regex + reserved-prefix + engine +
  today_engine validation), `_scenario_iters` (tier A/B/C/D bounding — tier B folds rows
  into the competition's `w*k >= 2^24` rule; tier C single-sample; tier D writes a
  SKIPPED_SLOW result JSON with a linear-model estimate that never enters numeric
  columns), `build_scenario_cells` (cell ids `scen_{name}_{engine}`, pinned per-engine
  seed indices identical to the competition's, so coinciding cells see identical
  tensors), `run_scenarios` (engine-major, resume-aware incl. SKIPPED_SLOW, table
  rewritten after every cell, **no header edits ever**), `build_scenario_table` +
  `write_scenario_reports` (`scenarios_table.csv` / `.md`, `{e}_us` filled only from
  MEASURED, provenance-drift check like the competition's).
- The only touch inside existing measurement code: `cell_warmup = cell.get("warmup",
  warmup)` in `run_child` (+ the loop uses it) — absent key = old behavior, so every
  classic/competition cell is byte-identical. Plus `import re` and the CLI/dispatch
  additions (`--model-scenarios` / `--scenarios-file` / `--scenarios`, loud exit when
  combined with `--competition`).

### 2. `tests/ttnn/unit_tests/operations/reduction/_topk_ledger_render.py`
- `load_scenarios()` (reads `scenarios_table.csv`), `render_model_scenarios()`
  (columns: scenario | model + callsite | shape | today engine | today µs | routed µs |
  op µs | best speedup | note; `us_fmt` em-dash rule; SKIPPED_SLOW today renders as
  `≈ X ms †`; free-text fields HTML-escaped; "win" styling only when an engine actually
  beats today — a routed cell slower than the model's current engine renders plain),
  optional `--scenarios-dir` with a conditional `MODEL_SCENARIOS` splice.

### 3. `TOPK_LEDGER.html`
- One-time `MODEL_SCENARIOS` details block after the P-sweep block, hand-written intro
  stating the two structural findings (canonical-form caveat for sampling rows;
  qwen36's pad-to-exactly-65536 falling off the bitonic gate) plus the largest=False
  stock-proxy convention and the †/— legend.

## Byte-identical verification (all passed)
- `--competition --with-blaze --report --out comp3` after the patch:
  `competition_table.csv` and `.md` byte-identical (`cmp` clean). (Plain `--report`
  without `--with-blaze` differs — pre-existing flag-dependent schema, not the patch;
  comp3 was originally built with blaze.)
- Old-style render (`--competition-dir comp3 --psweep-dir psweep4`, no scenarios):
  `git diff --stat TOPK_LEDGER.html` = **22 insertions, 0 deletions** — exactly the new
  empty details block.
- Final render with `--scenarios-dir scenarios1`: **35 insertions, 0 deletions** — the
  block + the spliced 8-row table; every pre-existing line intact.
- `scenarios_table.csv` copied to
  `tests/ttnn/unit_tests/operations/reduction/baselines/comp3/scenarios1_table.csv`.
  NOTE: `.gitignore:8 (*.csv)` covers it — the sibling baseline CSVs were force-added,
  so commit-time needs `git add -f` (left uncommitted per instructions).

## Measured scenario table (Tracy device-kernel, composite-summed, correctness-gated; µs)

| scenario | shape (rows×N, k) | today engine | today µs | routed (canonical ttnn.topk) | op direct | best speedup | per-cell status |
|---|---|---|---|---|---|---|---|
| sampling_qwen36_tp4 | 32×65536, k=32 | stocknow (1 core) | 9,596.3 | 217.0 (130c) | 193.6 (130c) | **49.6×** | all MEASURED, err=0 |
| sampling_tp8_pow2 | 32×32768, k=32 | stocknow = bitonic (65c) | 171.3 | 171.3 (65c, same bitonic) | 120.7 (130c) | **1.4×** | all MEASURED, err=0 |
| sampling_1chip_split | 32×64128, k=32 (×2 calls/token) | stocknow (1 core) | 8,923.5 | 215.2 (130c) | 187.9 (130c) | **47.5×** | all MEASURED, err=0 |
| dsa_indexer_k2048 | 160×65536, k=2048 | **op** (row-parallel, 130c) | 712.5 | 892.7 | 712.5 | 1.0× (already best) | MEASURED, err=0; stocknow omitted (no call site) |
| dsa_indexer_v4_k512 | 160×65536, k=512 | **op** (130c) | 559.5 | 691.1 | 559.5 | 1.0× (already best) | MEASURED, err=0 |
| msa_blocks_k16 | 1×8192, k=16 | **op** (16c) | 8.4 | 116.1 (bitonic, 65c) | 8.4 | 1.0× (already best) | MEASURED, err=0 (stocknow context: 116.1) |
| gate_gptoss_k4 | 32×128, k=4 | stocknow (1 core) | 24.2 | — | — | — | control, MEASURED, err=0 |
| gate_qwen35_k10 | 32×512, k=10 | stocknow (1 core) | 77.5 | — | — | — | control, MEASURED, err=0 |

Every number traces to
`generated/canonical_sweep/scenarios1/results/scen_{name}_{engine}.{engine}.t0.json`
(`ns_median`/1000). Anchors verified in the notes fields: routed sampling cells anchor
on `TopkLargeIndicesDeviceOperation` (routing engaged); the tp8_pow2/msa routed cells
anchor on `TopKDeviceOperation` (bitonic — routing correctly does not engage);
stock cells anchor on `TopKDeviceOperation`. No estimates were rendered (no tier-D
cell); the † legend in the block covers future SKIPPED_SLOW cells.

## What routed-vs-today means per row, and the model-side change per win

- **sampling_qwen36_tp4 (44–50× on the table, NOT a free win).** Today's 9.6 ms/token/
  device is the stock single-core insertion kernel (N=65536 is the one pow2 that fails
  the bitonic's `W < 65535` gate, `topk_device_operation.cpp:70`). The routed 217 µs is
  the CANONICAL call — production passes `indices_tensor=` + `sub_core_grids=` +
  `stable=True` (`tt_sampling.py:127,819,859`), each of which independently disqualifies
  routing (`topk.cpp:271–279`). **Model change to capture:** in
  `models/common/sampling/tt_sampling.py:847` drop `indices_tensor`/`sub_core_grids`/
  `stable=True` (requires resolving the stable-tiebreak contract, issue #33492, and the
  index-dtype shift uint16→uint32 the sampling chain assumes at `tt_sampling.py:101-105`)
  — or call `ttnn.experimental.topk_large_indices` directly for the extra 12%
  (193.6 µs). Also viable: run this model TP=8 (shard pads to 32768 → bitonic), which
  is what every other TP=8 model gets.
- **sampling_tp8_pow2 (the honest already-fast row).** Today IS the multi-core bitonic
  at 171 µs; routed measures the *same* bitonic at 171.3 (ratio 1.00 — the routing
  no-regression proof on the hottest production shape class). The 1.4× is only
  available by calling the op directly (120.7 µs); routing will never fire here by
  design. Marginal — a call-site rewrite for 50 µs/token is likely not worth it alone,
  but comes for free if the qwen36 fix standardizes on the direct op.
- **sampling_1chip_split (41–48×, ×2 calls/token).** Same canonical-form caveat;
  today ≈ 17.8 ms/token in stock topk (two 8.9 ms halves). `sampling_1d.py` does NOT
  pass `stable`, so its blockers are only `indices_tensor` + `sub_core_grids`
  (`sampling_1d.py:530,568`) — the easier call-site to relax first. Post-change the
  split-in-half workaround itself becomes unnecessary (the split exists purely to dodge
  stock's non-pow2 cliff).
- **dsa_indexer_k2048 / _v4_k512 (no change needed — and the routed column proves it).**
  The models already call the op by name; rows=160 takes the row-parallel factory with
  the chunk skip live (712.5 µs ≈ 2 rows/core × ~357 µs per-row single-core — exactly
  the rows>cores schedule). Routed (what `ttnn.topk` would give) is 25% SLOWER (the
  untilize/tilize/gather envelope costs ~180 µs at 160 rows): direct-op call sites
  should stay direct.
- **msa_blocks_k16 (no change needed, biggest routed-vs-op contrast).** The op's
  column-parallel factory does 1×8192 k=16 in 8.4 µs on 16 cores; the generic
  `ttnn.topk` entry (bitonic) costs 116 µs. MiniMax's choice to call the op directly is
  a 13.8× win over the generic path — keep it.
- **gate_gptoss_k4 / gate_qwen35_k10 (controls, as labeled).** k=4/k=10 violate the
  op's k≥16 / k%16==0 gate and N=128/512 sit below every routing threshold — routing
  provably cannot fire, and the rows exist as no-change proof. 24.2 / 77.5 µs stock
  single-core; unchanged by this branch.

## Findings worth keeping

1. **rows≤32 rides free on the stock single-core kernel.** The predicted
   "32 × 9.5 ms ≈ 300 ms/iter" class never materialized: stock TopK at (32 rows,
   65536) measured 9,596 µs ≈ the batch-1 time (9.49 ms) on 1 core — a 32-row batch is
   one tile height and the SFPU datapath covers all 32 rows in the same tile pass. So
   production per-token sampling cost today is ~9.6 ms (bad), not ~300 ms (catastrophic)
   — the ledger row states the measured number. The tier-B bound (3 iters, warmup 1)
   fired as designed; tier C/D were not needed.
2. **`SCENARIO_STOCK_NS_PER_ELEM` is a small-k rate** (~137–145 ns/elem at k≈32); the
   per-elem rate scales ~linearly with k (~9,600 ns/elem at k=2048). Documented in the
   constant: a future large-k stocknow scenario must scale it — the built-in grid only
   runs stock at k≤32, where the model is calibrated (and it OVER-estimates thanks to
   finding 1, which only errs toward more conservative bounding).
3. comp3's `--report` schema depends on `--with-blaze`; rebuilds of that dir must pass
   the flag (pre-existing behavior).

## Working tree (left uncommitted, as instructed)

Modified: `tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py`,
`tests/ttnn/unit_tests/operations/reduction/_topk_ledger_render.py`, `TOPK_LEDGER.html`
(35 additive lines). New (gitignored, needs `git add -f` at commit time):
`baselines/comp3/scenarios1_table.csv`. Run artifacts:
`generated/canonical_sweep/scenarios1/` (grid, 18 result JSONs, per-cell logs,
`scenarios_table.csv/.md`), run log at `scratchpad/scenarios1.log`.

## Rerun / extend

```bash
source python_env/bin/activate
flock /tmp/tt-device.lock python tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py \
    --model-scenarios --out generated/canonical_sweep/scenarios1 --resume
python tests/ttnn/unit_tests/operations/reduction/_topk_ledger_render.py \
    --competition-dir generated/canonical_sweep/comp3 --psweep-dir generated/canonical_sweep/psweep4 \
    --scenarios-dir generated/canonical_sweep/scenarios1 --ledger TOPK_LEDGER.html
```
New scenarios: edit `MODEL_SCENARIOS` in the sweep or pass `--scenarios-file` (JSON
`{"scenarios": [...]}`); subset with `--scenarios name1,name2`.
