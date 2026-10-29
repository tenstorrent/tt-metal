# AgentX simulator (Gemma4 prefill)

Replays the AgentX coding-agent traces against a step-cost model of Gemma4 prefill on a BH Galaxy (8x4), to compare chunk sizes, batching and cache sizes on a realistic workload in seconds instead of hours of silicon. Host only. Run from this directory.

The step cost is built from per-layer device measurements (`model_data_revc.json`) and calibrated against silicon replays of AgentX slices. Sessions run closed loop: `--concurrency` sessions at a time, each request sent when the ones it depends on finish plus its recorded think time, capped at `--max-think-s`.

```
# traces
hf download semianalysisai/cc-traces-weka-062126-256k --repo-type dataset --local-dir <dir>

# configs x loads over the whole corpus (every knob takes a comma list)
./agentx_sim.py --traces <dir>/traces.jsonl --config 8192x8192:fcfs,2048x8192:pack --concurrency 32,64 -j 4

# the tt-metal#60408 sweep
./sweep_60408.sh <dir>/traces.jsonl sweep.jsonl

# a slice for a silicon replay, its prediction, and the comparison with the measured replay
./agentx_slice.py build --traces <dir>/traces.jsonl --out slice.json --concurrency 48 --start 600 --window 120
./agentx_slice.py predict --slice slice.json --measured replay.json
```

## Reading the output

One row per run (`--json` writes every metric):
- `TTFT p50 p90 p99`: seconds from send to prefill done, every request of the run.
- `tok/s`, `useful`: rows processed per busy second, padding included, and new (uncached) tokens per busy second.
- `pad%`: padded rows. `recmp%`: cached tokens prefilled again because the resume point is rounded down to the chunk.
- `hit`: prefix-cache hit rate. `used`: the part of it kept after rounding to the chunk.
- `ss ...`: the steady window, while all `--concurrency` sessions are running: device utilisation, new tokens per wall second (against an unlimited cache) and TTFT. Compare designs here at a fixed load; `ss util` near 1.0 means saturated.

## TTFT distributions

`--ttft-out` writes each run's per-request TTFT and new tokens; `plot_ttft_cdf.py` draws one CDF per run, labelled by the knobs that differ (needs matplotlib, `pip install matplotlib`).

```
./agentx_sim.py --traces <dir>/traces.jsonl --config 8192x8192:fcfs,4096x4096:fcfs,2048x2048:fcfs,2048x8192:fcfs,2048x8192:pack \
    --concurrency 64 --max-think-s 60,1800 -j 10 --ttft-out ttft.jsonl
./plot_ttft_cdf.py ttft.jsonl --where max_think_s=60 --out ttft_cdf.png            # all requests
./plot_ttft_cdf.py ttft.jsonl --where max_think_s=60 --by-size --out ttft_cdf_by_size.png  # one panel per new-token bucket
./plot_ttft_cdf.py ttft.jsonl --where max_think_s=60 --xmax 30 --out ttft_cdf_linear.png     # linear axis, cut at 30 s
```

## Other configurations

`--config UNITxBUDGET:POLICY`: UNIT is the chunk size, BUDGET / UNIT the lanes per step (UNIT == BUDGET is unbatched), POLICY `fcfs` or `pack` (fewest rows left first).

```
# chunk sizes, unbatched
./agentx_sim.py --traces <dir>/traces.jsonl --config 1024x1024:fcfs,2048x2048:fcfs,4096x4096:fcfs,8192x8192:fcfs,16384x16384:fcfs -j 5

# lane counts and widths
./agentx_sim.py --traces <dir>/traces.jsonl --config 2048x4096:fcfs,2048x8192:fcfs,2048x16384:fcfs,4096x16384:pack -j 4
```

- Fitted on silicon: unbatched 8192, 4096, 2048; 2048-wide lanes, 1 to 4 per step.
- Anything else (unbatched 1024 or 16384, wider or narrower lanes, more than 4 lanes) runs on the measured per-layer data with the nearest calibration, and is marked `*` with the reason. Treat it as an unvalidated estimate.
- Per-layer data covers five chunk sizes, 1024, 2048, 4096, 8192 and 16384, and at most 16384 tokens per step. UNIT must be one of the five and BUDGET a multiple of it; any other `--config` is rejected before the run starts.

## Serving through tt-d-gen

The serving engine keeps prefill KV in a few whole slots (Gemma4: 6 of 256K tokens), seats a request only when a slot is free, and schedules one chunk at a time round robin over its seated requests. `--cache slotN` models its slots (`SlotCache`):
- a request is looked up when it is seated;
- a slot holding its prefix is reused and cut back to the match;
- otherwise a whole slot is evicted, least recently used first.

`:rr` models the scheduling: the engine queues chunks round robin up to `--inject-depth` ahead of the device (9 on silicon), so a newly seated request's first chunk waits behind the ones already queued. Through tt-d-gen on silicon (tt-metal#60869), chunks ran at the calibrated step time, and each request took about 95 ms more: frontend, admission, and the handoff after its last chunk (`--request-overhead-ms 95`). These values come from the runs' per-stage timestamps, not from fitting TTFT.

Replaying each silicon run's own requests with this model (send to prefill done, seconds):

| run | p50 | p90 | p99 | max | cache hit |
|---|---|---|---|---|---|
| 12 sessions, measured | 0.29 | 0.89 | 3.0 | 6.8 | 87.4% |
| 12 sessions, slot6 + rr | 0.29 | 0.86 | 3.0 | 6.7 | 87.5% |
| 12 sessions, token LRU + fcfs | 0.28 | 0.43 | 1.7 | 2.3 | 93.7% |
| 24 sessions, measured | 0.72 | 4.8 | 14.7 | 23.1 | 67.7% |
| 24 sessions, slot6 + rr | 0.86 | 5.9 | 15.7 | 25.2 | 63.6% |
| 24 sessions, token LRU + fcfs | 0.28 | 0.70 | 3.3 | 6.6 | 92.0% |

At 24 sessions the slot cache thrashes, and the replay is sensitive to small timing changes: across inject depths 7 to 9 it gives 63.6–68.5% hit. A run over the whole corpus is not the same workload as an AIPerf run, which samples a 20-minute window of sessions it starts partway through. Over the whole corpus, six slots fare worse: 80%, 66% and 31% hit at 6, 12 and 24 sessions.

```
./agentx_sim.py --traces <dir>/traces.jsonl --config 8192x8192:rr --cache slot6 --align resume --max-think-s 1800 \
    --request-overhead-ms 95 --inject-depth 9 --concurrency 6,12,24 -j 3
```

## Accuracy

- On held-out silicon replays, batched busy time is within 0.2% and TTFT p50 / p90 / p99 within 1.7%. They ran on bh-glx-120-c05u08, the box the calibration was fitted on, with different slices. Unbatched 8k / 4k were refit on 2026-10-18 for LoFi projections, as a server runs them. On three slice replays on another Galaxy (bh-glx-120-b08u02, about 1% faster than the calibration box) step time reads 2.0-2.5% high, TTFT p50 4-8% high and p90 / p99 1-14% high. On a saturated 8k window that 2% compounds: p50 reads 4.7 s against 3.7 s measured.
- The calibration is from one Galaxy; another box moved saturated TTFT tails by up to 10%.
- The model goes stale as prefill gets faster. To check it, replay one slice on silicon (`../demo/agentx_replay.py`, added with the continuous-batching PR) and compare with `agentx_slice.py predict --measured`.
- Not modelled: loading a cached prefix into the device, and the first decode token.

`--help` on each script lists its options and the other knobs: cache size, resume rounding (chunk, Gemma4's resume granularity, or exact), the extra cost of a chunk that starts mid-chunk, think-time cap, decode rate.
