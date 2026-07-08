---
name: mla-perf-report
description: >
  Build (or update) the interactive HTML performance report for the DeepSeek V3.2 / GLM sparse
  (DSA) MLA vs the dense v3.1 baseline, from Tracy dumps of test_sparse_mla_perf.py. Use when asked
  to "make/update the sparse MLA perf report", turn tracy profiler CSVs into a self-contained report,
  or produce a code-verified op dataflow graph with per-op device-kernel timings. Handles the Galaxy
  vs LoudBox/QuietBox proxy framing, sparse/dense × warm/cold/long scenarios, per-call attribution,
  and a two-level (semantic ↔ ops) dataflow graph with pan/zoom.
---

# Sparse MLA performance report

Turns Tracy device-profiler dumps into a **single self-contained interactive HTML artifact**: per-op
tables, a sparse-vs-dense comparison, a cold-prefill growth view, and a **code-verified dataflow graph**
whose node/op durations are **real per-call Tracy times**. The graph is verified against `mla.py` /
`indexer.py`, never inferred from an existing report.

**The one rule that matters:** every number comes from a Tracy report; every graph node is traced to
source (`file:line` + snippet). If you can't verify it against code or the raw report, don't assert it.

Spec of what the report must contain: `references/test_report.md` (the 11 requirements + the "Agreed
decisions" section). Read it first. Reference implementations live in `scripts/` next to this file —
they are **repo-specific templates**: reuse the machinery, re-derive the data, re-author the block
metadata against current source.

---

## Prerequisites

```bash
source python_env/bin/activate          # pandas, tracy live here (venv has no pip; see memory)
./build_metal.sh                         # ensure the build matches current code before any run
```
- Perf test: `models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py`.
- Tracy dumps land under `generated/profiler/deepseek_v32_{sparse,dense}_mla_perf/`:
  per-`(scenario,mode)` summary CSVs at the top level, raw per-op-per-device reports under
  `reports/<timestamp>/ops_perf_results_*.csv`.
- Node for headless validation may be at `/proj_sw/user_dev/bsheikh/nodejs/bin/node` (no system node).

## Workflow

### 1. Establish the workload and VERIFY dumps match current code
- Detect the box from device count (4=QuietBox SP1×TP4, 8=LoudBox SP2×TP4, 32=Galaxy SP8×TP4). All are
  the **Galaxy per-chip proxy**; frame measured numbers against the Galaxy target (see the test docstring).
- **Trust dumps only if they postdate the last code change.** `git log -1 --format=%ci` for HEAD, and the
  last commit touching `tt/mla/` or the perf test; compare against the dump dir mtimes. If code changed
  after the dumps, re-run tracy (`pytest -m perf ...::test_mla_chunked_perf -k "<scenario> and <mode>"`).
- **GOTCHA (cost me an hour):** the summary CSVs are **overwritten per (scenario,mode)** on every run.
  A saved `..._cold.csv` may be from a *different* run (different `DS_PERF_CHUNK`/cache) than the others,
  making a sparse-vs-dense comparison invalid. **Verify each dump's real parameters from its raw report**,
  don't trust filenames: `LayerNorm INPUT_0_Y_PAD` = per-chip seq (chunk); count the signature op
  (SparseSDPA / RingJointSDPA) per DEVICE ID = number of forwards (iterations). See `references/gotchas.md`.

### 2. Recover a matched run if a summary was clobbered
The matched run often still exists in an older `reports/<timestamp>/`. Re-derive its summary with the
driver's own post-processing — **do not hand-roll the device-collapse**:
`post_process_ops_log` slices the `signpost("start")..signpost("stop")` region, then
`merge_device_rows` (from `models/tt_transformers/tests/test_utils.py`) collapses the 8 chips:
**compute ops → max (critical path), collectives → avg**. See `scripts/recover_cold.py`.

### 3. Per-call attribution → real per-block / per-op durations
Tracy aggregates by op **code**, but the graph needs time per **semantic block** and per **op instance**.
Assign each execution-ordered, device-collapsed call to a block+op by walking the call stream against a
code-verified op template. See `scripts/parse_percall.py`. Two hard-won rules baked in there:
- **Alias relabels:** ttnn renames some CCL ops by layout/topology — the prefix gather surfaces as
  `AllBroadcast`, minimal reduce-scatter as `ReduceScatter`, indexer query rope as `RotaryEmbeddingLlama`.
  Accept alias sets when matching.
- **Pin async anchors as LABELS ONLY — do not advance the walk pointer on them.** The profiler lists
  async ops (topk, CCL) out of program order; advancing the pointer on a pinned anchor orphans the
  in-order ops after it and races the whole tail into the wrong blocks (symptom: the big `AllBroadcast`
  gather lands in the sdpa block). Validate: block sums == scenario total to the ns, and the unique
  anchors (SparseSDPA, RingJointSDPA, IndexerScore, Topk) match the summary totals exactly.

### 4. Author the dataflow graph from source (the correctness-critical part)
Trace `ttMLA.forward` for **both** paths and produce, per semantic block: `file:line`, a verbatim
snippet, the ttnn op codes it emits (reconciled against the per-forward Tracy counts), weights, and
input/output tensors (shape · dtype · layout · distribution). Delegate the breadth to parallel agents
(one per mode) but **spot-check the load-bearing claims yourself** against `mla.py`/`indexer.py`.
- Sparse (`has_indexer=True`): q_a → indexer[key-cache write + query/score/top-k] → q_stem → kv_stem →
  [kvpe write → prefix gather → sparse_sdpa] → wkv_b2 → o_proj.
- Dense (`has_indexer=False`): NullIndexer (0 device ops) + `ring_mla` (RingJointSDPA) over the full prefix.
- **Distribution notation** (unambiguous): `SP↕<factor> <axis>(dim<N>) · TP↕<factor> <axis>(dim<N>)`,
  `replicated` where an axis isn't sharded. The `↕N` is the mesh factor (LoudBox SP=2, TP=4), not the dim.

### 5. Build the interactive HTML artifact
`scripts/build_html.py` is the generator (data injected from `perf_data.json`; block metadata + edges +
per-op tensor annotations authored inline). Before writing UI, load the `artifact-design` skill. Report
conventions settled this session (all in the script):
- **Mode colours report-wide:** teal = sparse, amber = dense (toggle, comparison pills, graph pill).
- **Comparison headline:** total critical-path ms per scenario, sparse vs dense; expect the crossover
  (dense cheaper at short prefix, sparse wins as prefix grows — `ring_mla` scales with prefix, sparse is
  bounded by top-k).
- **Two-level graph, top→down:** semantic blocks with a global Semantic/Ops toggle + per-node ＋ expand
  into the intra-block op dataflow (real per-call times, heat-coloured, composites dashed). Lay out nodes
  in aligned columns (index / query / kv) so edges don't cross.
- **Pan/zoom** in a fixed-height viewport with a reset-to-fit control; preserve the view across
  same-layout redraws (e.g. opening the node drawer) and reset only on layout change.
- Node heat = white→red by share of trace; expanded-op heat = share within the run.
- Appendices: per-node source+snippet, full Tracy tables, branch/commit/HW metadata, and the attribution
  method. Collapse the data-integrity and appendix sections by default.

### 6. Validate headlessly, then publish
No browser here, so run the JS through a stubbed-DOM harness (see `references/validation.md`): eval the
inline script with a Proxy/JSDOM-ish `document`, fire every scenario×mode + view toggle + expand + drawer,
and assert no throw, no `NaN`/`undefined`, block sums exact, and node/onode counts > 0. Then `Artifact`
the file to the **same URL** (republish the same path) with a version `label`.
- **CSS gotcha:** the report is one big stylesheet — watch for **class-name collisions** (an "info"
  severity caveat once inherited an unrelated `.info` tooltip-button rule and collapsed to a 19px flex
  box). Scope utility classes (`.infobtn`, not `.info`).

## Checklist before calling it done
- [ ] Every dump's parameters verified from its raw report; comparison uses matched runs.
- [ ] Block sums == scenario total (ns-exact); unique anchors match summary totals.
- [ ] Every graph node has `file:line` + snippet; op counts reconcile with the per-forward Tracy counts.
- [ ] Data-integrity caveats stated (clobbered/recovered runs, any N/A like dense-long).
- [ ] Headless harness passes; no NaN/undefined; all toggles/expand/pan-zoom/drawer exercised.
- [ ] Branch, commit, and Blackhole box (LoudBox/QuietBox/Galaxy, card) recorded in the report.

See `references/gotchas.md` for the full list of traps and `references/requirements.md` for the report spec.
