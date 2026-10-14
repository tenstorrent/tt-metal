# AgentX replay for Gemma4 prefill (tt-metal#60408)

Host-only tools that replay the AgentX agentic-coding traces through a cost model of the Gemma4 chunked prefill on a BH Galaxy (8x4, CP8 x TP4). Their purpose is to choose a chunking design and chunk size. A silicon driver checks the simulator's predictions on slices of the trace.

| File | What it does |
|---|---|
| `cost_model.py` | `Model.step(lanes)`: device time of one prefill step = dense(rows) + attention per request at its own depth. `simulate()` covers synthetic workloads. |
| `model_data_revc.json` | The measured data behind the cost model (see "Where the device timing comes from"). |
| `agentx_sim.py` | Replays the whole AgentX corpus through a prefix cache and a scheduler, priced by `cost_model`. Reports the metrics from #60408. |
| `agentx_sweep.py` | Runs `agentx_sim` over a grid of settings in parallel and writes one JSON line per run. |
| `sweep_60408.sh` | The calibrated sweep behind the #60408 results. |
| `agentx_slice.py` | `build`: cut a time window out of the simulated schedule. `predict`: price the demo scheduler's policy on it. `compare`: check a silicon replay against the prediction. |
| `../demo/agentx_replay.py` | The silicon driver: plays a slice through the batching PoC's demo scheduler (`demo/batching_server.py`) in every mode. |

## Data
The traces are the 256k variant of the HF dataset `semianalysisai/cc-traces-weka-062126-256k`: 393 sessions and 68,266 requests. Requests with input + output > 256k tokens are already dropped. Download it with:

```
hf download semianalysisai/cc-traces-weka-062126-256k --repo-type dataset --local-dir <dir>
```

The first run caches the parsed traces next to the file as `traces.jsonl.sim.pkl`.

Each request carries:
- `hash_ids`: 64-token prefix blocks, scoped to the session;
- `t`, `api_time`, `out`: used for pacing.

The ideal (infinite-cache) hit rate on this variant is 96.2%.

## How the simulator works
- **Workload (closed loop):**
  - Each session is a DAG: main-agent requests chain in order, and each sub-agent chain forks off the main request before it.
  - A request is sent when its dependencies have answered (prefill done + `out / --decode-tps`), plus the recorded think gap capped at `--max-think-s`.
  - `--concurrency` sessions run at once; when one finishes, the next one starts.
- **Prefix cache:**
  - Keyed by (session, 64-token block); infinite or LRU over `--cache` tokens.
  - A request resumes at its cached depth rounded down to the chunk unit (`--align chunk`, as the block-cyclic layout requires) or exactly (`--align exact`, an upper bound).
  - Its last chunk is padded to the unit.
- **Scheduler:** `UNITxBUDGET:POLICY`.
  - `8192x8192:fcfs` is today's one request per step at 8k.
  - `2048x8192:fcfs` is the batching PoC: 4 lanes of 2k, FCFS.
  - `2048x8192:pack` gives spare lanes to the request with the fewest chunks left. It starves long requests under overload.
- **Metrics:**
  - TTFT p50/p90/p99, from send to prefill done;
  - TTFT p50/p99 by new tokens (≤1K, 1–4K, 4–16K, >16K);
  - total and useful tok/s while busy;
  - padding % and recompute % (cached tokens re-run because of chunk rounding);
  - cache hit, raw and usable after rounding;
  - steady-state (`ss_*`) values from the window where `--concurrency` sessions are live.

## Where the device timing comes from
`model_data_revc.json` was measured on a RevC Galaxy (bh-glx-110-d07u08, 130 W):
- one Tracy capture per chunk size (1k–16k) of a global and a sliding layer, which gives dense cost by rows;
- a host-timed replay of each layer type at every chunk index up to 256k, which gives global attention as a line in prefix depth per chunk size. Sliding attention is flat with depth.

A step is priced as 10 global + 50 sliding layers.

## Validation on silicon (bh-glx-110-d06u02, 2026-10-14, this branch: main 836bf0b + the batching PoC)
- **Whole-model perf test (`text_demo_prefill.py::test_prefill_chunk_sweep_traced`):** first chunk 53.1 / 74.1 / 113.2 ms and 256k device time 10.31 / 7.46 / 6.73 s at 2k / 4k / 8k. 4k and 8k are within 0.5% of the fit box. 2k at 256k is 3.5% slower than the fit box, which had the 2k q128 / K-split global attention of #59939.
- **KV PCC (`test_prefill_migration[mock-256k]`):** overall 0.98264 / 0.98025 / 0.98085 and worst head 0.941 / 0.935 / 0.939 at 2k / 4k / 8k.
- **Batched 2k steps:** 12 lane/depth cells measured with the batching PoC (`test_batched_step_perf`). The cost model (pessimistic attention) is within −6.8% to +8.0%. The worst cells are deep prefixes (128k+, +5% to +8%): the fit was taken with #59939's 2k global attention, and the per-mode calibration below absorbs this for AgentX's depth mix.
- **AgentX slices:** 308, 474 and 1,756 requests over 2 to 10 minutes, replayed with `demo/agentx_replay.py` in 8k, 4k, 2k and batched mode.
  - Step counts match the prediction exactly; batched is within 0.2%.
  - Measured corrections (`agentx_slice.py` `CAL_SCALE`, `CAL_HOST_MS`):
    - device scale per mode: 8k 1.125 (power cap at long context), 4k 1.057, 2k 1.009, batched 1.054;
    - host overhead per step: 3.3 ms serial, 6.3 ms batched.
  - With those corrections (`--scale`, `--host-ms`; `agentx_slice.py --calibrated`), predicted busy time is within ±0.5% in every mode.
  - TTFT p50 is within ~4%, except 8k near saturation, which reads 12–19% optimistic.
- **Not modelled:**
  - the cost of loading a cached prefix into the device (`--hit-load-us-per-ktok`, default 0, not measured);
  - clock throttling, beyond the calibration factor;
  - the decode's first token.

## Running
```
# whole corpus, one config (~30 s)
./agentx_sim.py --traces <dir>/traces.jsonl --config 8192x8192:fcfs --concurrency 64 --scale 1.125 --host-ms 3.3

# the #60408 sweep (~2.5 min on 24 cores)
./sweep_60408.sh <dir>/traces.jsonl out.jsonl

# silicon check on a slice
./agentx_slice.py build --traces <dir>/traces.jsonl --out slice.json --concurrency 48 --start 600 --window 120
./agentx_slice.py predict --slice slice.json --calibrated
AGENTX_SLICE=slice.json AGENTX_OUT=replay.json pytest models/demos/gemma4_d_p/demo/agentx_replay.py -s   # needs the Galaxy
./agentx_slice.py compare --slice slice.json --measured replay.json --calibrated
```

## Cross-check with the tt-d-gen mock
Andrew Vorozcovs's harness is tt-d-gen `adapters/dynamo/tools/eviction_study/run_agentx.sh`. It runs the real Dynamo engine and the official AIPerf AgentX client against a CPU mock device. We ran it with `TOPOLOGY=pd`, `--chunk-size 8192 --chunk-occupancy-us 170050` and an ack-latency curve from this cost model.

At a matched request rate (think cap 1800 s), the simulator agrees with it:
- cache hit 92.4% vs 92.9%;
- TTFT p50 0.27 s in both.

The mock charges a flat time per chunk with no batching or padding cost, so it cannot rank chunk designs.
