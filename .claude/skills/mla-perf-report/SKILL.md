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
- Perf test: `models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py`. It sweeps
  **variant × scenario × mode** = `[deepseek_v32, glm_5_1] × [warm, cold, long] × [sparse, dense]`.
- Tracy dumps land under `generated/profiler/{variant}_{sparse,dense}_mla_perf/`: per-`(scenario)`
  summary CSVs at the top level, raw per-op-per-device reports under `reports/<timestamp>/`, and — since
  the run-manifest change — a **`run_manifest.json`** in each `reports/<timestamp>/` (see below).
- **COMMIT FIRST.** The manifest records the commit but has NO dirty flag — it trusts that the working
  tree equals the commit. So the tree MUST be committed before profiling (a dirty run's manifest lies).
  When driving unattended, the skill is responsible for enforcing this: `git status --porcelain` must be
  empty, else commit (or refuse) before running tracy.
- **Swapping code states is Python-only.** The MLA impl (`tt/mla/*.py`) and the perf test are Python, so
  switching between two branches/commits that differ only there (e.g. branch vs its baseline) needs **no
  `./build_metal.sh`** — the existing `_ttnn.so` serves both. Only rebuild when C++ (`ttnn/…`) changed.
- Node for headless validation may be at `/proj_sw/user_dev/bsheikh/nodejs/bin/node` (no system node).

## Workflow

### 1. Establish the workload and VERIFY dumps match current code
- Detect the box from device count (4=QuietBox SP1×TP4, 8=LoudBox SP2×TP4, 32=Galaxy SP8×TP4). All are
  the **Galaxy per-chip proxy**; frame measured numbers against the Galaxy target (see the test docstring).
- **Trust a dump only via its `run_manifest.json`, not mtimes.** The manifest (written by the perf-test
  driver into each `reports/<ts>/`) is the source of truth for what produced the dump: `commit`, `branch`,
  `device{num_devices,box,mesh,fabric}`, `build.so_mtime`, and a copy-paste `command`. Require
  `manifest.commit == <the commit you mean to report>` — a git-log/mtime check is blind to a dirty tree,
  which is exactly how an uncommitted SP×TP working tree once got mistaken for its committed TP-head-sharded
  parent. If the manifest commit ≠ the intended commit, re-run tracy after committing.
- **GOTCHA (cost me an hour):** the summary CSVs are **overwritten per (scenario,mode)** on every run.
  A saved `..._cold.csv` may be from a *different* run (different `DS_PERF_CHUNK`/cache) than the others,
  making a comparison invalid. **Verify each dump's real parameters from its raw report** (or its manifest),
  don't trust filenames: `LayerNorm INPUT_0_Y_PAD` = per-chip seq (chunk); count the signature op
  (SparseSDPA / RingJointSDPA) per DEVICE ID = number of forwards (iterations). See `references/gotchas.md`.

### 1b. Measure a BASELINE (the "before"), not just the current branch
A perf report needs something to compare the branch against, or a "before" can silently be another build
of the same change (this bit us: the first dumps were already the SP×TP indexer, so before≈after and the
real win was invisible). The baseline is **the merge-base with `origin/main`** — the last commit on the
branch that's also in main:
```bash
git fetch origin main; BASE=$(git merge-base HEAD origin/main)   # or an explicit baseline branch the user names
```
- **Measure once, reuse while iterating.** Cache the baseline sweep keyed by its commit; re-measure only
  when the merge-base moves (a rebase), and prompt before spending the time — baseline runs are expensive.
- Because swapping is Python-only (see Prerequisites), you often don't even rebuild: check out the baseline
  commit/branch, sweep; check out the branch, sweep; the same `_ttnn.so` serves both. Restore the user's
  original checkout when done.
- The report then shows **baseline (before) vs branch (after)** per scenario×mode, alongside sparse-vs-dense.

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
  gather lands in the sdpa block).
- **Match STRICTLY at the pointer (auto-skipping anchor nodes) — never scan the whole block for a code.**
  A scan-ahead walk lets an unmodeled composite whose code equals a *later* named node (a rope-internal
  `MeshPartition` matching a later `mesh_partition` node; an RS-internal `Concat` matching a later concat)
  jump the pointer and orphan everything between. This is the nastiest trap: **block sums STILL equal the
  total and anchors STILL match, so the asserts PASS while the distribution is silently scrambled** (a
  block collapses to ~10% of its real time). Always eyeball the per-block node distribution (dump the
  s3/s4 nodes), not just block-sum==total. Fixed in `parse_percall.py` (strict-at-ptr + `skip_anchors`).
- Validate: block sums == scenario total to the ns, unique anchors (SparseSDPA, RingJointSDPA,
  IndexerScore, Topk) match the summary totals exactly, AND the named nodes of the load-bearing blocks
  (s3 indexer, s4 q-stem) carry sane per-node times.
- **`build_html.py` needs the percall→DATA merge** (`build_data.py` does NOT add it): the JS reads
  `data.block_timing[m][s]` and `data.expanded[m][s]`; fold `percall.json` in at the top of `build_html.py`
  or the graph renders blank.

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
- **Per-variant graph differences (deepseek_v32 vs glm_5_1).** The sparse graph differs in exactly three
  blocks — re-derive them per variant from the variant's own dump, don't reuse deepseek's:
  - GLM has `index_rope_interleave=True` → `_rope_perm is None` → **no rope-permute matmul** (drop the
    perm/rperm nodes in s2 write_k and s3 query).
  - GLM has 64 q-heads → 16/chip at TP=4 (<32) → `_needs_head_to_seq_reshard` is **True** → **s8 gains the
    thin-head transpose** ops (all_gather(q,dim1)+mesh_partition before sparse_sdpa; mesh_partition/
    all_gather/mesh_partition after). deepseek (32/chip) skips these.
  Dense is structurally shared (only config dims differ). Make `SPARSE_TMPL`/`SPARSE_BLOCKS` keyed by
  `(variant, mode)`. Model dims come from `reference/{deepseek_v3_2,glm_5_1}_config.py`, never hardcoded.

### 5. Build the interactive HTML artifact
`scripts/build_html.py` is the generator (data injected from `perf_data.json`; block metadata + edges +
per-op tensor annotations authored inline). Before writing UI, load the `artifact-design` skill. Report
conventions settled this session (all in the script):
- **Mode colours report-wide:** teal = sparse, amber = dense (toggle, comparison pills, graph pill).
- **Comparison headline:** total critical-path ms per scenario, sparse vs dense; expect the crossover
  (dense cheaper at short prefix, sparse wins as prefix grows — `ring_mla` scales with prefix, sparse is
  bounded by top-k).
- **Baseline (before/after) axis:** when a baseline was measured (step 1b), also show baseline-vs-branch
  ms per scenario×mode — this is what surfaces the branch's actual effect (e.g. the SP×TP indexer's
  ~12.4→8.9 ms warm win, invisible if both dump sets are the same code). `build_data.py`/`parse_percall.py`
  ingest both dump sets (baseline dir + branch dir); `build_html.py` renders the delta.
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
- [ ] Tree was committed before profiling; each dump's `run_manifest.json` commit == the intended commit.
- [ ] Baseline measured (merge-base with origin/main, or the named baseline branch); report shows
      baseline-vs-branch, not just one code state.
- [ ] Every dump's parameters verified from its raw report/manifest; comparison uses matched runs.
- [ ] Block sums == scenario total (ns-exact); unique anchors match summary totals; AND s3/s4 per-node
      distribution eyeballed (strict-walk trap: sums can be right while distribution is scrambled).
- [ ] Every graph node has `file:line` + snippet; op counts reconcile with the per-forward Tracy counts;
      per-variant s2/s3/s8 differences re-derived from that variant's own dump.
- [ ] Data-integrity caveats stated (clobbered/recovered runs, any N/A, GLM sweep pending, etc.).
- [ ] Headless harness passes; no NaN/undefined; all toggles/expand/pan-zoom/drawer exercised.
- [ ] Branch, commit (from the manifest), and Blackhole box (LoudBox/QuietBox/Galaxy, card) in the report.

See `references/gotchas.md` for the full list of traps and `references/requirements.md` for the report spec.
