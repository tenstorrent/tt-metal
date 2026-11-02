# strands-decider System One server (local)

FastAPI server for the strands decider on one Blackhole chip. It serves the upstream `POST /v1/systemone` API on the local host only.
One device process batches the questions of concurrent requests into shared forwards.
Tokenization runs in the server process (a 4-thread pool), so it overlaps the device.

## Run
```bash
source python_env/bin/activate
export TT_METAL_HOME=$PWD ARCH_NAME=blackhole
export TT_CACHE_PATH=$HOME/tt_cache/strands_agent

# QB2 (P300x2) only: open one of its 4 chips as a single P150
export TT_VISIBLE_DEVICES=0
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0

python -m models.demos.strands_agent.serve.server --port 8000
```
The server answers at once. `/health` returns 503 `starting` until the tokenizer is loaded, the model is built and every traced shape is warm.

## Request
```bash
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days!",
  "questions": {"urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}
}'
```

## Client
```bash
python -m models.demos.strands_agent.serve.client --wait-ready 900 --save /tmp/answers.json
python -m models.demos.strands_agent.serve.client --load 200 --concurrency 16 --route --tokens 512
# open loop: Poisson arrivals at 100 req/s, in-flight cap --concurrency (default 4096)
python -m models.demos.strands_agent.serve.client --load 2000 --rate 100 --route --tokens 512 --seed 0
# split the load over 4 client processes (load N/4, concurrency C/4, rate R/4, seed+k each; one merged report)
python -m models.demos.strands_agent.serve.client --load 3000 --concurrency 128 --route --tokens 128 --procs 4
```

## Endpoints
- `POST /v1/systemone`: `state` (str, dict or list), `questions`, optional `model` (ignored), `images` (must be empty).
  Returns `model`, `answers`, `usage`, `latency_ms`. 422 for bad questions, 503 while starting or after a worker failure.
- `GET /health`: status and settings.
- `GET /metrics?reset=false`: request counts, percentiles (latency, encode, queue wait, device), batch size histograms (one count per batch).

## Flags
`--host` (127.0.0.1), `--port` (8000), `--max-seq-len` (4096), `--batch-wait-ms` (0: take what is already queued, no wait; a
positive value waits that long for more items), `--max-batch-questions` (32), `--no-trace`,
`--eager-buckets` (also serve prompts above the traced buckets, eagerly; default off, traced buckets only),
`--warmup-eager` (also compile the eager buckets; needs `--eager-buckets`), `--keep-alive-s` (30, HTTP keep-alive),
`--model-name`, `--verbose`.

## Tests
`pytest -q models/demos/strands_agent/serve/test_batcher.py` (CPU only).
