# MiniMax-M3 pipeline prefill — running & testing

Multi-galaxy pipeline-parallel prefill for MiniMax-M3 via the common prefill runner. Serving is always
request mode (runner + producer); accuracy (KV PCC) and throughput differ only in the producer flags.
A single galaxy carved into sub-meshes runs the same machinery — see
[Intragalaxy pipeline](#intragalaxy-pipeline-single-galaxy-carved-into-sub-meshes).

## Setup

On the slurm node: build and activate the venv.
```bash
cd <tt-metal>
git submodule update --init --recursive
./build_metal.sh
source python_env/bin/activate
```
Prereqs on every host: same clone path + commit, a populated tilized weight cache, and the golden trace
(the adapter defaults to `/mnt/models/MiniMaxAI/MiniMax-M3-ref/golden/longbook_10240`;
`PREFILL_TRACE_DIR` overrides).

## Allocate the galaxies (slurm)
```bash
salloc -N 4 --nodelist=bh-glx-b08u02,bh-glx-b08u08,bh-glx-b09u08,bh-glx-b09u02 \
  --exclusive -t 04:00:00        # add your -p <partition> / -A <account>
```
Run the commands below from inside the allocation.

## Accuracy — KV PCC

Same two processes as the perf run below (runner under `tt-run`, producer on rank 0's host); the producer
reads the KV back and PCCs it against the golden trace. Single-rank, `PREFILL_MOCK_MIGRATION=1` makes the
runner publish its KV chunk table for the read-back gate (without it the producer has nothing to PCC).
Multi-rank, bare `PREFILL_MOCK_MIGRATION=1` is rejected (each rank would publish a table covering only its
own layer slice); set `PREFILL_ENABLE_MIGRATION: "1"` ALONGSIDE it in the binding's `global_env` to select
the merged mock — every rank joins the stage-layout all-gather, rank 0 publishes ONE table spanning all
layers, and each rank writes a rank-scoped device map (`<stem>_r<rank>.json`; the producer merges the
local ones). The table path must then be on shared storage (`PREFILL_MIGRATION_TABLE_PATH`, not `/tmp`).
On the producer (Process 2) add `PREFILL_PRODUCER_CHECK_PCC=1` and set `PREFILL_PRODUCER_MAX_REQUESTS=1`
so every slot's KV is still resident when it is read back. PASS = `[producer] KV cache PCC PASSED`
(threshold `PREFILL_STANDALONE_CHUNKED_PCC`, default `0.93`).

Both `PREFILL_MANIFEST` and `PREFILL_MOCK_MIGRATION` are shell-forwarded with `mpirun -x`, which lands on
the launch-host rank only — fine at 1 galaxy (one rank), but at 2+ galaxies the remote ranks never see
them and silently run the default model without publishing their table. For multi-host, put both in the
request binding's `global_env` (the same `_minimax.yaml` copy made under Process 1, plus
`PREFILL_MOCK_MIGRATION: "1"` and `PREFILL_ENABLE_MIGRATION: "1"` — see above) and drop them from the
shell. A ready-made 2-rank example (intragalaxy) is
`models/demos/minimax_m3/tt/runners/manifests/m3_binding_mock_migration_intragalaxy_2rank.yaml`.

### 1 galaxy
```bash
PREFILL_MANIFEST=models/demos/minimax_m3/tt/runners/manifests/minimax_m3.json \
PREFILL_MOCK_MIGRATION=1 \
  ./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_request_1rank.yaml \
  bh-glx-b08u02:1
```

### 2 galaxies
```bash
# binding copy carries PREFILL_MANIFEST (absolute) + PREFILL_MOCK_MIGRATION: "1" in global_env
./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_request_2rank_minimax.yaml \
  bh-glx-b08u02:1,bh-glx-b08u08:1
```

### 4 galaxies
```bash
./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_request_4rank_minimax.yaml \
  bh-glx-b09u02:1,bh-glx-b09u08:1,bh-glx-b08u08:1,bh-glx-b08u02:1
```

## Perf — throughput + overlap plot

Two processes: the runner (blocks waiting for input) and the producer on rank 0's host (the first
`--host`).

### Process 1 — runner (tee to a log)

Shell-exported `PREFILL_MANIFEST` is forwarded with `mpirun -x`, which lands on the launch-host rank only.
That is fine at 1 galaxy (one rank), but at 2+ galaxies the remote ranks never see it and silently fall
back to the default model, so they disagree on the chunk plan. For multi-host, copy the request binding
and set `PREFILL_MANIFEST` to the **absolute** manifest path in its `global_env` (every rank reads the
binding), then run the copy with no shell `PREFILL_MANIFEST=`:
```yaml
# pipeline_prefill_request_2rank_minimax.yaml (copy of ..._request_2rank.yaml)
global_env:
  PREFILL_MANIFEST: "<tt-metal>/models/demos/minimax_m3/tt/runners/manifests/minimax_m3.json"
  # ... rest unchanged
```

1 galaxy (single host, shell export reaches the one rank):
```bash
PREFILL_MANIFEST=models/demos/minimax_m3/tt/runners/manifests/minimax_m3.json \
  ./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_request_1rank.yaml \
  bh-glx-b09u02:1 \
  2>&1 | tee /data/philei/health/pp_1rank.log
```

2 galaxies (manifest in the binding's `global_env`, per the note above):
```bash
./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_request_2rank_minimax.yaml \
  bh-glx-b09u02:1,bh-glx-b09u08:1 \
  2>&1 | tee /data/philei/health/pp_2rank.log
```

4 galaxies (manifest in the binding's `global_env`, per the note above):
```bash
./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_request_4rank_minimax.yaml \
  bh-glx-b09u02:1,bh-glx-b09u08:1,bh-glx-b08u08:1,bh-glx-b08u02:1 \
  2>&1 | tee /data/philei/health/pp_4rank.log
```

Wait for `[pp rank 0] [h2d] descriptor …` before starting Process 2.

### Process 2 — producer (on rank 0's host = first `--host`)

Second terminal; attach a shell to rank 0's node — `b09u02` for every config (it's the first `--host` in all of them):
```bash
squeue --me                                                     # get JOBID
srun --jobid=<JOBID> --nodelist=<rank0-host> --overlap --pty bash
```
Then, on that node:
```bash
cd $TT_METAL_HOME && source python_env/bin/activate
LOGURU_LEVEL=INFO \
PREFILL_MODEL=minimax_m3 \
PREFILL_TRACE_DIR=/mnt/models/MiniMaxAI/MiniMax-M3-ref/golden/longbook_56320 \
PREFILL_H2D_SERVICE_ID=ds_prefill \
PREFILL_SP=8 PREFILL_TP=4 PREFILL_NUM_LAYERS=60 \
PREFILL_CHUNK_SIZE=5120 PREFILL_MAX_SEQ_LEN=56320 \
PREFILL_NUM_USERS=2 \
PREFILL_PRODUCER_CHUNKS=11 \
PREFILL_PRODUCER_MAX_REQUESTS=10 \
PREFILL_PRODUCER_INTERLEAVE=round_robin \
PREFILL_SEND_SHUTDOWN=1 \
  python3 -m models.demos.common.prefill.runners.prefill_producer
```
Per config, change `PREFILL_NUM_USERS` / `PREFILL_PRODUCER_MAX_REQUESTS` (and the attach host) — see the
table below. Transport env (`SP`/`TP`/`CHUNK_SIZE`/`MAX_SEQ_LEN`/`NUM_LAYERS`) must match the runner.

### Plot
```bash
python -m models.demos.deepseek_v3_d_p.scripts.plot_pipeline_trace \
  /data/philei/health/pp_2rank.log -o /data/philei/health/pp_2rank.png
```
`parse_iteration_times.py <log>` prints per-iteration numbers. Ignore chunk 0 (first iteration recompiles).
`PREFILL_SYNC_PER_CHUNK=1` on Process 1 gives exact per-chunk compute but disables overlap (timing only).

### 1 vs 2 vs 4

| galaxies | request binding | `--host` order | NUM_USERS | MAX_REQUESTS | rank0 host | log |
|---|---|---|---|---|---|---|
| 1 | `..._request_1rank.yaml` | `b09u02` | 1 | 5 | b09u02 | pp_1rank.log |
| 2 | `..._request_2rank.yaml` | `b09u02, b09u08` | 2 | 10 | b09u02 | pp_2rank.log |
| 4 | `..._request_4rank.yaml` | `b09u02, b09u08, b08u08, b08u02` | 4 | 20 | b09u02 | pp_4rank.log |

Hold `PREFILL_CHUNK_SIZE`, the trace, and per-stage `NUM_USERS` fixed across runs; compare `E2E_CLOCK`
(per rank) or `parse_iteration_times.py`.

## Intragalaxy pipeline (single galaxy carved into sub-meshes)

The same rank/stage machinery runs on ONE 8×4 galaxy split into Z-linked sub-meshes:
2 stages of [4,4] (30 layers each, `pipeline_prefill_request_intragalaxy_2rank.yaml`) or 4 stages of
[2,4] (15 layers each, `..._intragalaxy_4rank.yaml`), both under
`models/demos/common/prefill/runners/topology_configuration/`. All ranks are co-located, so shell
exports reach every rank and no binding copy is needed for the manifest.

Three prerequisites specific to this mode:

- **Per-mesh-shape weight cache.** The tilized cache is keyed by mesh shape — the sub-meshes need
  `tensor_cache_bfp8_MeshShape([4, 4])` / `([2, 4])`, which the default checkpoint dir does not carry
  (populating takes ~1 h per shape, once; the ranks populate their own slices in parallel). Point
  `TT_CACHE_PATH` at a root that has them — both shapes are currently populated at
  `/data/zbaczewski/m3_pp_cache` (its `[8, 4]` dir is empty, so do NOT use it for whole-galaxy runs).
- **PRTE slot fix.** Multi-rank on one host under a Slurm allocation fails with "All nodes which are
  allocated for this job are already filled" (the galaxy advertises `CPUTot=1`). Before launching:
  ```bash
  unset $(env | sed -n 's/^\(SLURM[^=]*\)=.*/\1/p')
  export PRTE_MCA_ras="^slurm" PRTE_MCA_plm="^slurm"
  ```
- **Raise RLIMIT_NPROC first.** The limit is per-user node-wide and a live multi-rank runner's threads
  exhaust it — a later terminal then can't even fork. Make `ulimit -u <big>` (e.g. `2318144`) the FIRST
  line of every terminal/step on the node.

### Perf

Same two-process flow as multi-galaxy; both terminals are on the one node. The producer's transport env
must match the SUB-MESH: `PREFILL_SP=4` (2-stage) or `PREFILL_SP=2` (4-stage), `PREFILL_TP=4`.

```bash
# 2-stage runner
TT_CACHE_PATH=<pp-cache-root> PREFILL_MANIFEST=models/demos/minimax_m3/tt/runners/manifests/minimax_m3.json \
  ./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_request_intragalaxy_2rank.yaml \
  $(hostname -s):2 2>&1 | tee pp_intra2.log

# 4-stage runner
TT_CACHE_PATH=<pp-cache-root> PREFILL_MANIFEST=models/demos/minimax_m3/tt/runners/manifests/minimax_m3.json \
  ./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_request_intragalaxy_4rank.yaml \
  $(hostname -s):4 2>&1 | tee pp_intra4.log
```

The producer command is the multi-galaxy one with `PREFILL_SP=4` (2-stage) / `PREFILL_SP=2` (4-stage) —
same plot/readout (`parse_iteration_times.py`, `plot_pipeline_trace`).

### Accuracy — KV PCC (merged mock)

Ready-made manifests (10240 ISL, 2 chunks/slot; the merged-mock env, table on shared storage, producer
PCC threshold set to M3's 0.88 gate):

```bash
# 2-stage
TT_CACHE_PATH=<pp-cache-root> ./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  models/demos/minimax_m3/tt/runners/manifests/m3_binding_mock_migration_intragalaxy_2rank.yaml $(hostname -s):2
python -m models.demos.common.prefill.runners.prefill_producer \
  --manifest models/demos/minimax_m3/tt/runners/manifests/m3_producer_mock_migration_2rank.yaml

# 4-stage
TT_CACHE_PATH=<pp-cache-root> ./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  models/demos/minimax_m3/tt/runners/manifests/m3_binding_mock_migration_intragalaxy_4rank.yaml $(hostname -s):4
PREFILL_SP=2 python -m models.demos.common.prefill.runners.prefill_producer \
  --manifest models/demos/minimax_m3/tt/runners/manifests/m3_producer_mock_migration_2rank.yaml
```

**Expect:** rank 0 logs the merged 60-layer 9-config table; each rank writes a rank-scoped device map
(`/tmp/m3_kv_device_map_r<rank>.json`, merged by the producer); per-layer K/V/index_k over
`60/60 local layers`, then `KV cache PCC PASSED`.

### Loopback migration (real endpoint + worker)

The intragalaxy 2-stage loopback-migration gate (P2) — endpoint, 2-rank runner, migration driver with
`--verify-migration both` — is documented next to the manifests in
`models/demos/minimax_m3/tt/runners/PREFILL_MIGRATION_TESTING.md` (Gates P0/P1/P2). Validated 2026-08-25:
source PCC 60/60 layers, dst==src byte-identical over all 345600 chunks, migrated slots ≥ 0.88 vs golden.

## Env knobs

- `M3_WEIGHTS_FROM_CACHE=1` — force the tilized-cache load (skip the bf16 source).
- `PREFILL_TRACE_DIR=<dir>` — override the golden trace.
- `PREFILL_PP_LAYER_COUNTS="a,b,..."` — override the even layer split (must sum to 60, one per rank).
- `PREFILL_SYNC_PER_CHUNK=1` — exact per-chunk compute (disables overlap).
- `LOGURU_LEVEL=INFO` — silence the model's DEBUG logs. Already in the request bindings' `global_env` for
  the runner ranks; set it on the producer command too (shown above).
