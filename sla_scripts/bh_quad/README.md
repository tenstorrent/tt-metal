# Llama-3.3-70B on 4 BH Galaxies: 128 concurrent users

One vLLM server per Galaxy (the single-Galaxy SLA configuration from `../bh_tp32`, 32 slots each:
DP4 x `max_num_seqs` 8) behind `lb.py`, an OpenAI-compatible load balancer. Clients use the balancer exactly like
one vLLM server; it adds no model-side change, so accuracy is the single-Galaxy accuracy.

```
clients ──> lb.py :9000 ──┬─> Galaxy A vLLM :8000  (32 slots)
                          ├─> Galaxy B vLLM :8000  (32 slots)
                          ├─> Galaxy C vLLM :8000  (32 slots)
                          └─> Galaxy D vLLM :8000  (32 slots)
```

## Files

| File | What |
|---|---|
| `lb.py` | the balancer (aiohttp + uvloop, one process) |
| `lb_selftest.py` | 24 checks against 4 fake backends, CPU only, ~1 min |
| `serve.sh` | one Galaxy's server in the shipped config, on 0.0.0.0:8000 |
| `quad_bench.sh` | the SLA lines through the balancer: 128 x 4K, 128 x 16K, 4 x 240 RPM, needle, sanity |

## Run

On each Galaxy (work dir `J` with `tt-metal/`, `weights/`, `cache/llama70b/` (222 GB tensor cache),
`vllm-tt-plugin-gio/`):

```bash
J=/mnt/nvme0/jerrywang bash sla_scripts/bh_quad/serve.sh
```

On any host that reaches all four:

```bash
python sla_scripts/bh_quad/lb.py --backends http://10.82.97.43:8000,http://10.82.97.44:8000,http://10.82.97.45:8000,http://10.82.97.46:8000
```

```bash
HF_MODEL=/path/to/Llama-3.3-70B-Instruct OUT=runs/quad bash sla_scripts/bh_quad/quad_bench.sh
```

`GET /lb/stats` shows per-backend in-flight, prefill backlog, served, errors, TTFT EWMA.
`POST /lb/backend {"url": "...", "enabled": false}` drains a Galaxy (no new requests; running ones finish).

## How the balancer routes

- **Slots:** at most `--slots` (32) requests in flight per backend. Requests past the total wait in one FIFO
  queue and take the first slot that frees on any backend. No backend is ever over-subscribed, so each Galaxy
  sees at most the load the single-Galaxy SLA numbers were measured at.
- **Cost:** a request goes to the backend with the lowest
  `prefill backlog (prompt tokens waiting for their first output token) + 256 x running requests + the prompt
  tokens this backend would have to prefill`. A burst spreads evenly; a backend busy with long prompts gets the
  short ones last; prompts whose prefix a backend probably still caches (hashed at 2K/8K/32K/128K chars) count
  only their uncached part there. That is how conversations stay with their history without piling a burst of
  identical prompts onto one Galaxy.
- **Failures:** connection refused, a reset, or a 5xx before the first byte: the backend is marked down and the
  request retried elsewhere (2 retries), invisible to the client. A backend dying mid-stream cuts the client
  connection (the client sees an error, never a clean but short answer). Health checks (`/health` every 2 s)
  bring a backend back. A client that disconnects closes the upstream connection, so vLLM aborts the request.
- **Not routed:** `GET /v1/models`, `/tokenize`, `/version`... go to any healthy backend without a slot.

## Self-test (lb_selftest.py, Nov 23, all 24 pass)

128 concurrent streams split exactly 32/32/32/32 with every byte in order; 300 concurrent queue and never exceed
32 per backend; pending-prefill accounting returns to 0; chat, non-streaming and 4xx pass through (4xx not
retried); short requests avoid a backend prefilling 100K tokens; shared-prefix requests stick while a burst of
identical prompts still spreads 2/2/2/2; a backend returning 503 or killed: all requests succeed elsewhere, and it
rejoins (32 of the next 128) when healthy; a backend dying mid-stream gives those clients an error and leaves the
rest intact; 16 abandoned client streams abort upstream and free their slots; a drained backend gets nothing.
Overhead: 1000 tiny requests take the same time through the balancer as direct.
