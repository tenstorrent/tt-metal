# MiniMax-M3 pipeline prefill — running & testing

Multi-galaxy pipeline-parallel prefill for MiniMax-M3 via the common prefill runner. Accuracy (KV PCC)
uses the standalone bindings; throughput + overlap plots use request mode + the producer.

## Setup

On the slurm node: build and activate the venv.
```bash
cd <tt-metal>
git submodule update --init --recursive
./build_metal.sh
source python_env/bin/activate
```
Prereqs on every host: same clone path + commit, a populated tilized weight cache, and the golden trace
(the adapter defaults to `/data/philei/models/minimax-m3-prefill-cache/golden/longbook_10240`;
`PREFILL_TRACE_DIR` overrides).

## Allocate the galaxies (slurm)
```bash
salloc -N 4 --nodelist=bh-glx-b08u02,bh-glx-b08u08,bh-glx-b09u08,bh-glx-b09u02 \
  --exclusive -t 04:00:00        # add your -p <partition> / -A <account>
```
Run the commands below from inside the allocation.

## Accuracy — KV PCC

Run from `TT_METAL_HOME`. PASS = every rank's KV PCC passes.

### 1 galaxy
```bash
PREFILL_MANIFEST=models/demos/minimax_m3/tt/runners/manifests/minimax_m3.json \
  ./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_real_1galaxy_1rank.yaml \
  bh-glx-b08u02:1
```

### 2 galaxies
```bash
PREFILL_MANIFEST=models/demos/minimax_m3/tt/runners/manifests/minimax_m3.json \
  ./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_rank_binding_2rank_d2d.yaml \
  bh-glx-b08u02:1,bh-glx-b08u08:1
```

### 4 galaxies
```bash
PREFILL_MANIFEST=models/demos/minimax_m3/tt/runners/manifests/minimax_m3.json \
  ./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_rank_binding_4rank_d2d.yaml \
  bh-glx-b09u02:1,bh-glx-b09u08:1,bh-glx-b08u08:1,bh-glx-b08u02:1
```

## Perf — throughput + overlap plot

Two processes: the runner (blocks waiting for input) and the producer on rank 0's host (the first
`--host`).

### Process 1 — runner (tee to a log)

1 galaxy:
```bash
PREFILL_MANIFEST=models/demos/minimax_m3/tt/runners/manifests/minimax_m3.json \
  ./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_request_1rank.yaml \
  bh-glx-b08u02:1 \
  2>&1 | tee /data/philei/health/pp_1rank.log
```

2 galaxies:
```bash
PREFILL_MANIFEST=models/demos/minimax_m3/tt/runners/manifests/minimax_m3.json \
  ./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_request_2rank.yaml \
  bh-glx-b08u02:1,bh-glx-b08u08:1 \
  2>&1 | tee /data/philei/health/pp_2rank.log
```

4 galaxies:
```bash
PREFILL_MANIFEST=models/demos/minimax_m3/tt/runners/manifests/minimax_m3.json \
  ./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_request_4rank.yaml \
  bh-glx-b09u02:1,bh-glx-b09u08:1,bh-glx-b08u08:1,bh-glx-b08u02:1 \
  2>&1 | tee /data/philei/health/pp_4rank.log
```

Wait for `[pp rank 0] [h2d] descriptor …` before starting Process 2.

### Process 2 — producer (on rank 0's host = first `--host`)

Second terminal; attach a shell to rank 0's node (`b08u02` for 1/2 galaxies, `b09u02` for 4):
```bash
squeue --me                                                     # get JOBID
srun --jobid=<JOBID> --nodelist=<rank0-host> --overlap --pty bash
```
Then, on that node:
```bash
cd $TT_METAL_HOME && source python_env/bin/activate
LOGURU_LEVEL=INFO \
PREFILL_MODEL=minimax_m3 \
PREFILL_TRACE_DIR=/data/philei/models/minimax-m3-prefill-cache/golden/longbook_56320 \
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
| 1 | `..._request_1rank.yaml` | `b08u02` | 1 | 5 | b08u02 | pp_1rank.log |
| 2 | `..._request_2rank.yaml` | `b08u02, b08u08` | 2 | 10 | b08u02 | pp_2rank.log |
| 4 | `..._request_4rank.yaml` | `b09u02, b09u08, b08u08, b08u02` | 4 | 20 | b09u02 | pp_4rank.log |

Hold `PREFILL_CHUNK_SIZE`, the trace, and per-stage `NUM_USERS` fixed across runs; compare `E2E_CLOCK`
(per rank) or `parse_iteration_times.py`.

## Env knobs

- `M3_WEIGHTS_FROM_CACHE=1` — force the tilized-cache load (skip the bf16 source).
- `PREFILL_TRACE_DIR=<dir>` — override the golden trace.
- `PREFILL_PP_LAYER_COUNTS="a,b,..."` — override the even layer split (must sum to 60, one per rank).
- `PREFILL_SYNC_PER_CHUNK=1` — exact per-chunk compute (disables overlap).
- `LOGURU_LEVEL=INFO` — silence the model's DEBUG logs. Already in the request bindings' `global_env` for
  the runner ranks; set it on the producer command too (shown above).
