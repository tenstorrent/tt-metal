# llama-nemotron-rerank-1b-v2 rerank server

FastAPI server for the reranker on Blackhole, one device worker per chip, with cross-request batching. Request and
response shapes follow tt-media-server's `/v1/rerank` and `/v1/score` (vLLM / Cohere style). The runtime design
(503 while starting, watchdog, pair-capped batching, one tokenizer per thread) follows the strands-decider server.
`demo/serve.py` is the simpler bge-m3-style server; this one is meant for deployment.

## Files
- `rerank_server.py`: HTTP layer, worker processes, watchdog, metrics.
- `rerank_batcher.py`: device-free batching (`collect_batch`, `run_batch`); `test_rerank_batcher.py` tests it on CPU.
- `rerank_client.py`: reference check (`--check ref.json`, Pearson against HF fp32 logits) and load test (`--load`).
- `Dockerfile`, `entrypoint.sh`, `requirements.txt`: server image (tt-metal and caches mount at runtime).

## Run from a tt-metal checkout
```bash
source python_env/bin/activate
uv pip install -r models/demos/blackhole/llama_nemotron_rerank_1b/serve/requirements.txt
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD TT_MM_THROTTLE_PERF=0
python models/demos/blackhole/llama_nemotron_rerank_1b/serve/rerank_server.py --chips 0000:86:00.0 --port 8112
```
`--chips` takes PCI addresses, one worker per chip (pin the chip you were given, not an index).
`GET /health` returns 503 `starting` until every trace is captured, then 200 `ok`.

## Image
```bash
docker build --network host -f models/demos/blackhole/llama_nemotron_rerank_1b/serve/Dockerfile \
             -t nemotron-rerank-server models/demos/blackhole/llama_nemotron_rerank_1b
docker run --rm -d --name nemotron-rerank --device /dev/tenstorrent/31 \
  -v /dev/hugepages-1G:/dev/hugepages-1G -v $TT_METAL_HOME:/work:ro -v /path/to/cache:/cache \
  -e RR_CHIPS=0000:86:00.0 -p 127.0.0.1:8112:8112 nemotron-rerank-server
```
The first start downloads the weights and tokenizer into `/cache` (run once with `-e HF_HUB_OFFLINE=0`).

## Endpoints
- `POST /v1/rerank` (`/rerank`): `query`, `documents` (1-1000 strings or `{"text": ...}`), optional `top_n`
  (0 = all), `return_documents`, `model`. Results sorted best first with `relevance_score` = sigmoid(logit) and `logit`.
- `POST /v1/score` (`/score`): `text_1` (query), `text_2` (one document or a list); scores in input order.
- `GET /health`, `GET /v1/models`, `GET /metrics?reset=false` (counts, latency / encode / queue-wait / device
  percentiles, per-batch histograms, mid-serving trace captures).
422 for a bad request, 503 while starting, after a worker exit or a device hang (`--hang-timeout-s`).

## Tests
`pytest -q models/demos/blackhole/llama_nemotron_rerank_1b/serve/test_rerank_batcher.py` (CPU only).
