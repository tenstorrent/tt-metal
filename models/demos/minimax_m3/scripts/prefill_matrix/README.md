# MiniMax-M3 pipeline-prefill perf matrix

Tracks prefill performance of the intra-/multi-galaxy pipeline runner over a fixed
**(new tokens) x (cached tokens)** matrix so runs can be compared over time.

| | values |
|---|---|
| new tokens (the request) | 640 (p25), 1600 (p50), 3072 (p75), 5120 (one chunk), 6900 (p90), 32768, 51200 (p99) |
| cached tokens (prefix already in the KV cache) | 0, 61440 (p25), 143360 (p50), 312320 (p75), 552960 (p90), 860160 (p99) |

The values are percentiles of the **Agent X dataset** (`semianalysisai/cc-traces-weka-062126`):
new-prefill tokens per request p25 / p50 / p75 / p90 / p99 and cached context p25 / p50 / p75 / p90 / p99,
rounded to chunk multiples (chunk = 5120), plus 5120 (exactly one chunk) and 32768 as extra
points. Per cell the script measures:

* **Idle pipeline** — one request at a time, pipeline drained between requests: TTFT from the
  producer's push of the first chunk to the last rank finishing the last chunk, and
  `new tok/s = new / TTFT`. Median of the timed iterations after the first (iteration 0 is cold).
* **Loaded pipeline** (`USERS>0`) — `USERS` users, each with the cached prefix, stream
  `USERS x M` requests back to back, round-robin at chunk granularity, never waiting:
  **steady-state aggregate tok/s** from the last rank's chunk cadence with the first and last
  `stages` chunks dropped (pipeline fill/drain excluded; reported as *new* tokens and as
  *processed* 5120-token chunks), aggregate incl. tails, and per-request **TTFT under load**
  (median / p90). `M` defaults to enough requests for ~240 chunks per cell (`--target-chunks`).

The cached prefix is prefilled once per row (per user slot); each cell then appends only its new
chunks at that offset, so the prefix stays intact and every cell sees exactly `cached` tokens.
Each row gets its own runner launch with the KV capacity fitted to `cached + 51200`: the dense
layers gather the whole cache shard per chunk, so an over-sized cache would skew the numbers.

## Prerequisites

* A Slurm allocation holding the galaxies (4 for 16 stages, 3 for 12), reachable over ssh from
  the rank-0 host (the pipeline launcher uses mpirun over ssh; see
  `models/demos/common/prefill/runners/run_pipeline_prefill.sh`). Scripts and logs must live on
  storage shared by all hosts (home dirs are node-local on exabox).
* Weights + the `[2,4]` tilized cache on weka (defaults:
  `HF_MODEL=/mnt/weka/model-weights/llm/minimax/MiniMax-M3`,
  `TT_CACHE_PATH=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill`) and a golden
  trace with `token_ids` (`PREFILL_TRACE_DIR`, default the weka `golden/longbook_56320`; its
  tokens are tiled cyclically to the cache capacity — real text keeps MoE routing realistic).
  A cold cache build inside the runner is slow and currently hangs on non-first ranks; populate the
  cache first.
* The host wiring in `binding_16stage_quad.yaml.in` / `quad_bh_galaxy_16stage_2x4_z_chain.textproto`
  is the exabox "straight" chain (tray order 0-7 -> 24-31 -> 16-23 -> 8-15 per host, host h's 8-15 ->
  host h+1's 0-7). Other quads may need a different binding.

## Run

```bash
cd $TT_METAL_HOME
# 16 stages on 4 galaxies, rank-0 host first; idle + loaded (4 users), 3 idle iterations per cell
JOB=<slurm job id> HOSTS=bh-glx-120-b09u02,bh-glx-120-b09u08,bh-glx-120-b08u08,bh-glx-120-b08u02 \
  models/demos/minimax_m3/scripts/prefill_matrix/run_matrix.sh
```

Takes ~15 min bring-up (galaxy resets + first weight load) plus ~5-8 min per cached row
(six rows; the 860k row needs a 911360-token KV capacity per slot, ~2.8 GB/chip with 4 users). Knobs: `USERS` (0 = idle only), `ITERS`, `CACHED` / `NEW` lists, `REPS` (repeat the
whole matrix for error bars), `STAGES=12` with three hosts, `WORK` (default
`generated/m3_prefill_matrix/<stamp>`), `OUT` (results JSONL).

Results: `OUT` holds one JSON line per idle iteration (`mode: idle`) and per loaded cell
(`mode: loaded`). Print the tables again with

```bash
models/demos/minimax_m3/scripts/prefill_matrix/matrix_table.py <results.jsonl> [--csv out.csv]
```

Per-run artefacts in `WORK`: `runner*.log` (all ranks), `producer*.log`, `timing_*/rank<r>.csv`
(one row per chunk per rank: `rank,c,compute_start,compute_ms`, from `PREFILL_SYNC_PER_CHUNK=1`).

## Notes for comparing runs

* Quote the galaxies and the commit: single-galaxy M3 prefill numbers differ by ~10% between
  galaxies and a few % day to day.
* The runner path uses the 2D fabric, on which the MoE dispatch's `sparse_mcast` fast path is
  disabled (the log says `sparse_mcast DISABLED`), so stage times are not comparable to the
  single-mesh harness (`tests/galaxy_prefill_kv_pcc.py`).
* `PREFILL_SYNC_PER_CHUNK=1` (needed for the per-chunk timing rows) costs a few % of
  throughput versus the production runner.
* Small requests (< 5120 new tokens) always cost one full chunk, so their tok/s is `new / one
  chunk latency`; under load the pipeline still processes a full chunk per period, so their
  "processed" tok/s is the pipeline ceiling while "new" tok/s shows the padding waste.

## Files

| file | role |
|---|---|
| `run_matrix.sh` | top-level: loops the cached rows (and `REPS`), prints the tables |
| `matrix_row.sh` | one row: shutdown previous runner, (reset,) launch runner, wait ready, run producer |
| `matrix_runner.sh` | node side: manifest + rendered binding, exec `run_pipeline_prefill.sh` |
| `matrix_producer.py` | pushes chunks over H2D, measures from the ranks' timing CSVs, writes JSONL |
| `matrix_shutdown.py` | sends the SHUTDOWN sentinel to a live runner |
| `matrix_table.py` | tables / CSV from a results JSONL |
| `matrix_common.sh` | shared helpers sourced by the shell scripts: ulimit raise, the producer/shutdown env, scoped runner kill |
| `binding_16stage_quad.yaml.in`, `quad_bh_galaxy_16stage_2x4_z_chain.textproto` | 4-galaxy 16-stage topology |
| `binding_12stage_tri.yaml.in`, `tri_bh_galaxy_12stage_2x4_z_chain.textproto` | 3-galaxy 12-stage fallback |
