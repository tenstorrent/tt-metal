# Scripts — reference implementations

These are the exact scripts that produced the report this skill is based on (LoudBox, SP=2×TP=4). They are
**templates, not turnkey**: they hardcode a scratchpad output dir, the `reports/<timestamp>` → scenario
mapping, and the block metadata. Reuse the machinery; re-derive the data and re-author the metadata against
current source. Run with `source python_env/bin/activate` first.

Pipeline order: `discover.py <commit>` (map dumps → dirs/totals per code-state via run-manifest; run once
per commit — branch and baseline) → `parse_percall.py` (branch graph) → `build_data.py` (assemble
`perf_data.json`, incl. the baseline axis) → `build_html.py`.

### `discover.py` — manifest-driven dump discovery (the run-store idea)
Scans every `reports/<ts>/run_manifest.json`, groups by `(commit, scenario, mode)`, and computes the
device-collapsed signposted total per group via `merge_device_rows` — no dependence on the clobber-prone
top-level summary CSVs. Run `python discover.py <baseline_commit>` and `python discover.py <branch_commit>`;
each writes `totals_<commit>.json` (dir + total + calls + iters per `mode/scenario`). This is how baseline
vs branch dumps are told apart after both sweeps write into the same profiler dirs. **Adapt:** nothing —
it's driven by the manifests the perf-test writes.

### `parse_percall.py` — per-call attribution (the core)
Assigns each execution-ordered, device-collapsed op call to a semantic block + internal op node with its
REAL merged duration. Contains the ordered op **templates** per mode, the CCL/rope **alias sets**, and the
**anchor-pinning** walk (label-only, no pointer advance — see gotchas). **Adapt:** the `RPT` dict mapping
each `(mode,scenario)` to the *correct verified* `reports/<timestamp>` dir; the templates if the op graph
changed. Validates block sums == total and anchors == summary totals; keep those assertions.

### `recover_cold.py` — re-derive a clobbered summary
Reprocesses a specific raw `reports/<timestamp>/ops_perf_results_*.csv` with `merge_device_rows` to rebuild
a matched summary + by-iteration CSV without re-running the board. **Adapt:** the `RPT` path and the chunk.

### `build_data.py` — assemble `perf_data.json`
Reads the summary + by-iter CSVs into the JSON the HTML embeds (per-op tables, totals, cold-by-iter). The
final report also folds in `parse_percall.py`'s output as `block_timing` (semantic-node real durations) and
`expanded` (per-block ordered op nodes). Also ingests the **baseline axis**: set `BASELINE_TOTALS` to the
baseline commit's `totals_<commit>.json` (from `discover.py`) → `data.baseline` + `data.baseline_meta`
(before/after). **Adapt:** paths; the baseline commit; which run is authoritative for each cell.
`build_html.py`'s `drawBaseline()` renders the baseline-vs-branch panel (dense is the control — should be
~flat; the sparse delta is the indexer's effect).

### `build_html.py` — the generator (produces the self-contained report)
Holds the block metadata (`SPARSE_BLOCKS`/`DENSE_BLOCKS`: `file:line`, snippet, weights, io tensors, col/row
layout), the edge lists with distribution notation, per-op tensor annotations (`TENSOR`), the `LAYOUT`
lane-assignment (crossing-free columns), `META`/`CAVEATS`, and the full CSS/HTML/JS (two-level pan/zoom
graph, toggles, tables, cold charts, drawer, appendices). **Adapt:** all block metadata to current source
(this is the code-verified part — re-trace, don't copy blindly); `META` (branch/commit/box); `CAVEATS`.
Injects `perf_data.json` as the inline payload. Emits a single `.html` — publish it with the `Artifact` tool
after loading the `artifact-design` skill.

Validate the output with the stubbed-DOM node harness in `../references/validation.md` before publishing.
