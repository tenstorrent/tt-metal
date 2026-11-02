# strands-decider System One server

FastAPI server for the strands decider on one Blackhole chip. It serves the upstream `POST /v1/systemone` API on 127.0.0.1 only.
One device worker process batches the questions of concurrent requests into shared forwards; tokenization runs in the server process.

## Start the server
1. Build tt-metal and create `python_env` (`./build_metal.sh`, `./create_venv.sh`; see [INSTALLING.md](../../../../INSTALLING.md)).
2. Install the server packages into `python_env` (tested with fastapi 0.162.0 and uvicorn 0.54.0):
   ```bash
   source python_env/bin/activate
   uv pip install fastapi "uvicorn[standard]"
   ```
3. Set the environment from the tt-metal root:
   ```bash
   export TT_METAL_HOME=$PWD ARCH_NAME=blackhole
   export TT_CACHE_PATH=$HOME/tt_cache/strands_agent

   # QB2 (P300x2) only: open one of its 4 chips as a single P150
   export TT_VISIBLE_DEVICES=0
   export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
   export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
   ```
4. Start the server:
   ```bash
   python -m models.demos.strands_agent.serve.server --port 8000
   ```
   The defaults give the best measured performance; `--help` lists the other options.
   The first start downloads `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` and `Qwen/Qwen3.5-2B-Base` from Hugging Face
   and converts the weights into `TT_CACHE_PATH`, so it takes longer. Later starts are ready in about 22 s.
   After the first download, `HF_HUB_OFFLINE=1` runs without network. `STRANDS_DECIDER_MODEL` and `HF_MODEL` select local checkpoint dirs.
5. Wait until the log prints `model ready after X s`. `GET /health` returns 503 `starting` until then, and 200 `"status": "ok"` after.
   ```bash
   curl -s localhost:8000/health
   ```
6. Send a request:
   ```bash
   curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
     "state": "Help! My payouts have been failing for 3 days!",
     "questions": {"urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}
   }'
   ```
7. Stop the server with Ctrl+C. It closes the device before it exits.

## Benchmark
`client.py` checks the answers and measures latency and throughput. The server does not need it; any HTTP client works.
```bash
# quick check: send the example requests and save the answers
python -m models.demos.strands_agent.serve.client --wait-ready 900 --save /tmp/answers.json
# load test: 3000 requests of 128 tokens, 128 in flight over 4 client processes; prints latency percentiles and req/s
python -m models.demos.strands_agent.serve.client --load 3000 --concurrency 128 --route --tokens 128 --procs 4
```
`--help` lists the other options (open-loop `--rate`, `--questions` per request).

## Endpoints
- `POST /v1/systemone`: `state` (str, dict or list), `questions`, optional `model` (ignored), `images` (must be empty).
  Returns `model`, `answers`, `usage`, `latency_ms`. 422 for bad questions, 503 while starting or after a worker failure.
- `GET /health`: status and settings.
- `GET /metrics?reset=false`: request counts, percentiles (latency, encode, queue wait, device), batch size histograms (one count per batch).

## Tests
`pytest -q models/demos/strands_agent/serve/test_batcher.py` (CPU only).
