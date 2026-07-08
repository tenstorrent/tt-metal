# Performance tests, sparse attention in MLA


## Performance tests
Performance test has 3 scenarios:
- warm (cache) - cache with 50k tokens, we get new 5k tokens
- cold (cache) - cache is empty, fills in iteratively in 5k chunks until it reaches 55k tokens
- long - same as warm, just cache is filled to 0.5M tokens

Modes:
- Sparse - being developed, sparse attention
- Dense - dense attention, baseline used for comparison

After running tests, tracy reports will be generated.

## Goal

Create interactive HTML artifact with performance results.

Report should contain:
1. List of operations and their duration (absolute and relative), number of calls. List can be sorted by execution order or duration.
2. Graph of the MLA model operations.
    2.1. Nodes are operations, edges tensors that flow between nodes.
    2.2. Nodes should have labeled internal parameters (tensors) - weights
    2.3. Edges shuold have labeled corresponding tensors
    2.4. All tensors should have labeled dimensions, data types, mem layout (tile is default, row major shuold be clearly indicated) and distribution (SP|TP|replicated|etc.).
    2.5. Nodes should be colored in shade of red - longest red, shortest white
    2.6. Nodes should be labeled w/ their duration - both absolute and relative to the full trace
3. A toggle that controls which scenario is presented in 1. and 2.
4. Dense mode should be used as a baseline - there shuold be a clear report how much better/worse sparse is compared to dense
5. If cold scenario is selected, additional information is needed - tracking of top N (default = 10) longest operations across iterations and total duration of each iteration.
6. Graph *must be* verified agains the actual code, not as a proxy to existing report or assumption or whatever
7. Reported durations *must be* based on tracy reports.
8. Add appendix sections where I can walk through details
    8.1. for nodes, link file path and line number where I can check more details and add critical code snippet
    8.2. for duration of operations, present complete tracy reports
9. Report shoudl be self sufficient - present evidence and copy reports.
10. Commit number, branch name and short description in bullet points what are the key changes in the branch and what the experiment is about.
11. Hardware information - detials about device where measurements took place - LoudBox, QuiteBox, Galaxy, Blackhole, Wormhole, card name etc.


## Agreed decisions (v1 build)

Resolved in session 2026-07-08 before building. These pin down the ambiguous requirements above.

**Measured hardware (this run).** LoudBox — 8× Blackhole, mesh SP=2×TP=4, i.e. the **1/4-Galaxy proxy**
(per-chip compute shapes equal Galaxy; sequence length is 1/4). Box-local sizes: chunk=1280,
warm/cold cache=12800 (11 chunks), long cache=128000. The report presents these measured numbers and
frames them against the Galaxy target (chunk=5120, cache=50k / 0.5M, SP=8×TP=4) exactly as the test
docstring does. (req 11)

**Tracy source (req 33, 48).** Trust the existing dumps under
`generated/profiler/deepseek_v32_{sparse,dense}_mla_perf/` rather than re-running everything. Evidence:
the last commit touching the MLA impl or perf test is `3e94dfa` (committed 2026-07-07 14:47 UTC); every
dump was generated after it (earliest 07-07 23:24, latest 07-08 07:42), and HEAD `099251b` is an
unrelated Exabox script that does not touch MLA code — so the dumps correspond to current code.
Present: sparse {warm, cold, long}, dense {warm, cold}. **Missing: dense `long`** — to be generated
after the v1 report is built; shown as N/A / placeholder until then.

**Graph model (req 2, 6).** Nodes are **semantic blocks authored by reading the code** (not raw tracy
op codes). Each block's duration = sum of the tracy ops it maps to (abs + rel, red-shaded: longest red →
shortest white). Edges carry the tensors flowing between blocks, labeled with shape / dtype / layout
(TILE default, ROW_MAJOR flagged) / distribution (SP | TP | replicated). Blocks also label internal
weights. Every block is verified against actual source and linked to file:line (not inferred from the
CSV or existing reports).

**Graph toggles.** Two independent toggles: **scenario** (warm | cold | long) × **mode**
(sparse | dense). Sparse and dense are structurally different forwards, so mode swaps between **two
distinct graphs**, each drawn from its own code path with durations from its own dumps.

**Comparison (req 4).** Headline = **total critical-path device-kernel ms, sparse vs dense, per
scenario** (expected crossover: dense cheaper at short prefix, sparse wins as the prefix grows), with
per-op-code deltas below.

**Scope.** Full v1 covering all 11 requirements in one pass, then iterate on polish.


## Measurements

- Performance test for sparse attention can be found here: models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py
- If tracy dumps are not present run tracy.
- Make sure that build is up to date by running `./build_metal.sh`.
- Proper python environment is set by running `source python_env/bin/activate`
- Make sure that tracy measurements correspond to current code status.

## How

- We will probably iterate on the report a few times to polish which data should be presented and how and on the overall design.
- After we get a good report, you should create a skill based on this file and the session on creating perf report.
