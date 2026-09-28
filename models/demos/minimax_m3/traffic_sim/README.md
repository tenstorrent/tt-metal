# MiniMax-M3 prefill: AgentX traffic roofline

Traffic-level model of the M3 pipeline-prefill system on Blackhole galaxies. It replays the **AgentX 1M corpus** the way
AIPerf's `inferencex-agentx-mvp` scenario does and runs every request through a **per-op roofline** calibrated to the
16-stage hardware runs of #57827. The goal is to rank features and hyperparameters by what they do for the whole system
on realistic traffic, not by single-cell benchmarks.

* Interactive artifact (presets, live sweeps, roadmap, calibration): https://claude.ai/artifact/3Yresb3qbWJQpvpQMabaq6
* Tracking issue: https://github.com/tenstorrent/tt-metal/issues/57827

## Metric

**Goodput = useful tok/s at p90 TTFT ≤ 10 s**, maximised over concurrency, interpolated at the SLO crossing.
*Useful* tokens are the tokens an infinite prefix cache would still have to prefill (`in - 64·lcp_best`). Re-prefilled tokens
(evicted, misaligned, never materialised) and padding count as processed but not useful. Each result also reports the same
configuration with an infinite cache, TTFT p50/p90, the hit rate against the ∞-cache hit rate, and the split of processed
tokens into useful, re-prefill and padding.

## Files

| file | what |
|---|---|
| `sim_core.js` | **The whole model** (no dependencies): per-op roofline + calibration, stage plan and memory, KV residency (static slots / paged pool with lanes / host tier / paging / ∞), and the closed-loop replay. The artifact inlines it and node requires it. |
| `prep_traffic.py` | Corpus → `traffic.bin` + `traffic.json`: DAG, end-to-start delays, sub-agent spawn/join, prefix-tree pieces. Takes 14 s on 32 cores. |
| `collect_calib.py` | Measured data → `calib_data.json` (committed): zone profiles, per-rank per-chunk-position medians of runs A/B/C, matrix tables. |
| `validate.js` | Calibration report: fitted efficiencies, per-rank stage times, and all 105 matrix cells, model vs measured. |
| `run.js` | One configuration or a concurrency sweep from the CLI (`--preset`, `--set key=value`, `--conc`). |
| `study.js` | Greedy feature roadmap, leave-one-out, topology/budget/lane grid and sensitivity, over {4, 8} galaxies × {today's kernels, roofline kernels}. |
| `analyze.js`, `tools/study_detail.js` | Print study results. |
| `lib/pool.js` | Worker-thread pool plus the goodput rule (early stop past the cache cliff). |
| `build_artifact.js`, `artifact/template.html` | Build the single-file artifact: the core, calibration, 4 MB of traffic as base64, the study summary and the presets. |
| `on_node.sh` | `JOB=<slurm id> ./on_node.sh <cmd>` runs on the compute node with soft ulimits raised to the hard limits. |
| `tools/` | Debug helpers: `dump_cells.js` (per-cell stage medians), `traffic_stats.js`, `smoke.sh` (every feature path). |

Data that does not belong in git lives in `/data/philei/m3_traffic_sim/` on exabox:
* `data/`: `traffic.bin/json`, plus `zones/` holding copies of the `parse_zone_perf.py` outputs.
* `results/study.json`: the full study.
* `artifact/`: the built HTML.

## How to run (exabox, cpu_only allocation; never on the login node)

```bash
NODE=/data/philei/tools/node-v22.11.0-linux-x64/bin/node   # node 22 tarball, no install needed
PY=/data/philei/tt-metal/python_env/bin/python3
cd models/demos/minimax_m3/traffic_sim
JOB=<id> ./on_node.sh $PY prep_traffic.py --procs 32              # once per corpus version
python3 collect_calib.py                                          # after new hardware runs
$NODE validate.js                                                 # check the calibration
JOB=<id> ./on_node.sh $NODE run.js --preset today-C --conc 16,64,256 --set cache=pool --set hostTier=true
JOB=<id> ./on_node.sh $NODE study.js --workers 36 --out /data/philei/m3_traffic_sim/results/study.json   # 5 min
$NODE analyze.js /data/philei/m3_traffic_sim/results/study.json
$NODE build_artifact.js    # then republish artifact/m3_agentx_lab.html to the artifact URL above
```

One simulation of 1800 s of traffic at C=1024 takes 0.3–0.5 s. The full study (749 configurations, about 12k runs) takes 278 s on 36 threads.

## Traffic replay (AIPerf `inferencex-agentx-mvp` semantics)

The replay rules below were read from the AIPerf source (`ai-dynamo/aiperf`, `timing/strategies/agentic_replay.py`, `dataset/loader/weka_trace.py`).

* **Corpus.** `semianalysisai/cc-traces-weka-062126`, full variant: 393 sessions, 98,827 requests, 21.6B input tokens, contexts up to 990k. The ∞-cache hit rate is 98.3%.
* **Lanes.** `concurrency` = N lanes, each replaying one tree (the main chain plus its sub-agents).
  * **Start.** t* is uniform over the session (AgentX uses a 0–1 ratio). Every stream active at t* gets a primer: its last turn before t*, as a real prefill with `max_tokens=1`. Profiling starts when all primers are done.
  * **Recycle.** A drained tree recycles to the next session, sequentially, from turn 0 with a new cache-bust. Trees share nothing.
* **Delays.** Each next request waits `max(0, t_k − t_{k−1} − api_time_{k−1})` after the live end of the previous one. The delay is **not capped**. Main-agent gaps have a median of 4.9 s, a mean of 395 s and a p99 of 54 min. The only cap is AIPerf's 10 s system-idle cap. `gapCap` exists as a knob, default off.
* **Sub-agents.** They spawn when the spawning turn returns, or at issue on the overlap path. The join turn fires immediately once the last child ends.
* **Decode.** A request ends at prefill done + `out / decodeTps` (default 180 tok/s).
* **Window.** The profiling window is 1800 s.

**Consequence:** an AgentX lane is mostly idle. Saturating a pipeline therefore takes hundreds to thousands of lanes. The KV working set then grows with concurrency, and throughput is limited by a **cache cliff**: past it, evictions turn into re-prefill, which raises TTFT, which leaves more KV idle and evicted.

## Cost model

Each layer is decomposed into the ops the implementation runs.

* **MoE/MSA layer:** 2× norm all-gather, qkv, index branch, all-gathers of the K/V and index_k prefix, indexer, sparse SDPA, o_proj + RS, shared expert (+RS), router, dispatch, experts, combine, moe_reduce (+RS), misc.
* **Dense layer:** ring-joint SDPA = max(compute ∝ T·(k+T/2), scan of the whole lane shard ∝ capacity/SP), plus the MLP and its CCLs.

**Roofline.** Each op gets a roofline time from FLOPs, DRAM bytes and ethernet bytes on the stage mesh. The Blackhole numbers come from tt-moe-nappkin `lib/system.py`: 608/304 TF (LoFi/HiFi2), 512 GB/s, 100/50 GB/s CCL bi/uni.

**Calibration.**
1. **Zone profiles.** `eff = roofline / (zone time − latency floor)`, using the zone profiles (chunk 5120, 51k cached) for [2,4], [8,4] and [4,2]. [4,4] is the geometric midpoint of [2,4] and [8,4].
2. **Pipeline fit.** A fit on 7,840 per-rank, per-chunk-position medians of the 16×[2,4] runs A/B/C produces:
   * an MoE multiplier of 1.03 (2D fabric);
   * dense ring-joint efficiencies: compute 30%, capacity scan 1.4%, i.e. about 100 ms per 1M-token lane per chunk;
   * embedding 1.5 ms;
   * a blocking send of 18 ms at 5120 and 7.5 ms at 2048;
   * a hop latency of 16 ms at 5120 and 14 ms at 2048.

   Two findings came out of the fit:
   * **Padded chunk tails are cheaper.** Routed MoE ops use *actual* tokens (the `actual_isl` trim), so a 640-token request in a 5120 chunk is 18% cheaper than a full chunk.
   * **Today's expert kernel behaves like weight read + compute,** not the max of the two.
3. **Validation.** Replaying all 105 matrix cells gives |error| of 3.9% median, 8.6% p90 and 13.9% max (loaded new tok/s and idle TTFT). Run `validate.js` to see it.

**`opEff` knob.** It moves each op geometrically from its measured efficiency to a target (70% matmul, 80% DRAM/link, overlapped expert weight reads) and lowers the latency floors. 0 means today's kernels; 1 means roofline kernels.

**Batching.** Several requests share one chunk: MoE and projections run on the padded total, and attention runs per request (`seq`) or once per chunk (`fused`), with a 110-core wave-quantization factor. Variable layout pads each segment to 32·SP tokens and adds an all-to-all KV write per layer.

**Memory per stage.**
* weights: bf4 experts, bf8 attention, bf16 shared/dense MLP, embedding on stage 0, LM head on the last stage;
* `reserveGB` per chip plus activation buffers;
* KV: 1088 B/token/layer for K/V (bf8), plus index_k at 128 × (2 B bf16 | 1.0625 B bf8) × (TP replicas | 1).

KV capacity is the minimum over stages. At 4 galaxies with bf16 index_k replicated ×4, that is 21.5M tokens, or 20 static 1M slots.

## KV residency

* **`slots`** (today): one 1M slot per stream; LRU over idle slots. The hit is the prefix shared with the stream's previous request, floored to a chunk multiple unless `unaligned`.
* **`pool`**: lanes (fixed `lanes`×1M per stage, or request-sized in an arena) plus a content-addressed paged pool with **exact LRU**.
  * The prefix tree is compressed into pieces cut at branch points and request ends, so every request touches whole pieces. That makes LRU over pieces identical to LRU over 64-token pages.
  * Copy-in/out is DRAM-bound per stage and overlapped with a 25% contention charge.
  * With `hostTier`, device evictions are demoted to host DRAM. Host hits are fetched over PCIe before admission, and write-backs share the link.
* **`paging`**: an ideal paged kernel (no lanes, no copies).
* **`inf`**: infinite cache.

## Findings (study of Sep 28 2026, decode 180 tok/s, `results/study.json`)

Goodput in useful tok/s at p90 TTFT ≤ 10 s. "Today" = 16×[2,4] (or 32×[2,4]), chunk 2048, auto split, static 1M slots.
The earlier decode-60 study is kept as `results/study_decode60.json`; the ranking is the same.

| scenario | today | greedy full stack | best grid config | same with ∞ cache |
|---|---|---|---|---|
| 4 gx, today's kernels | 2.4k | 37.2k | **40.8k** (16×[4,2], 4 lanes, budget 16k) | 57.9k |
| 4 gx, roofline kernels | 4.0k | 74.6k | **78.3k** (16×[4,2], 2M arena, budget 32k) | 208k |
| 8 gx, today's kernels | 4.1k | 89.4k | **95.9k** (32×[4,2], 2M arena, budget 8k) | 131k |
| 8 gx, roofline kernels | 14.4k | 155.5k | **161.1k** (32×[4,2], 2M arena, budget 8k) | 373k |

Each cell below is "G / LOO". G is the gain at the greedy step where the feature was added. LOO is the leave-one-out loss when the feature is removed from the full stack.

Complexity is a rough judgement of how much of the stack a feature touches:
* **low**: one op or the scheduler, plus validation;
* **med**: a new op or allocator, or changes across a few ops;
* **high**: cross-cutting changes to the runtime, KV layout, several kernels and the scheduler.

It is not a time estimate and has not been checked with the code owners.

| tier | feature | 4gx today | 4gx roofline | 8gx today | 8gx roofline | complexity |
|---|---|---|---|---|---|---|
| P0 | slot lanes + paged pool | ×2.98 / ×6.9 | ×4.17 / ×6.4 | ×2.12 / ×5.5 | ×3.02 / ×6.0 | high |
| P0 | host-DRAM KV tier (1 TB/gx) | ×1.47 / ×2.0 | ×1.85 / ×1.7 | ×1.48 / ×1.6 | ×1.38 / ×1.6 | high |
| P0 | bounded dense gather (while lanes are 1M) | ×1.66 / ×1.00 | ×1.00 / ×1.00 | ×3.32 / ×1.00 | ×0.99 / ×0.99 | low |
| P0 | async stage handoff | ×1.12 / ×1.13 | ×1.31 / ×1.32 | ×1.20 / ×1.25 | ×1.49 / ×1.51 | med |
| P0 | multi-request batching (8–32k budget) | ×1.20 / ×1.30 | ×1.22 / ×1.33 | ×1.16 / ×1.20 | ×1.08 / ×1.23 | high |
| P0 | index_k stored once (not ×TP) | ×1.18 / ×1.23 | ×1.20 / ×1.24 | ×1.17 / ×1.18 | ×1.22 / ×1.27 | med |
| P1 | variable chunk (a2a KV write) | ×1.18 / ×1.08 | ×1.09 / ×1.01 | ×1.14 / ×1.08 | ×1.14 / ×1.09 | high |
| P1 | index_k bf8 | ×1.07 / ×1.08 | ×1.13 / ×1.04 | ×1.04 / ×1.07 | ×1.05 / ×1.10 | low (PCC) |
| P1 | shortest-first scheduling | ×1.03 / ×1.02 | ×1.00 / ×1.00 | ×1.03 / ×1.06 | ×1.04 / ×1.10 | low |
| P2 | MSA SP-local indexer | ×1.03 / ×1.03 | ×1.01 / ×1.00 | ×1.02 / ×1.02 | ×1.00 / ×1.00 | high |
| P2 | variable-size lane arena | ×0.99 / ×0.99 | ×1.00 / ×1.00 | ×1.02 / ×1.01 | ×1.03 / ×1.03 | med |
| P2 | fused multi-user attention | ×1.01 / ×1.01 | ×1.00 / ×1.00 | ×1.01 / ×1.01 | ×1.02 / ×1.02 | high |
| P2 | unaligned resume (subsumed by var) | ×1.00 (up to ×1.14 before var) | ×1.00 | ×1.00 | ×1.00 | low |

Takeaways:
1. **On AgentX, KV capacity sets the throughput, not compute.** Even the best stacks reach only 38–73% of their own ∞-cache goodput.
   * The pool (vs static 1M slots), the host tier and index_k de-replication are strong in every scenario.
   * Host DRAM size matters much more than PCIe bandwidth: 0.5 → 2 TB/galaxy moves 4-gx goodput 33.7k → 51.2k (today's kernels) and 62k → 103k (roofline kernels). 16 → 256 GB/s moves it by at most 7%.
2. **Bounded dense gather is P0 while lanes are 1M slots:** ×1.7 at 4 gx, ×3.3 at 8 gx with today's kernels. Every chunk scans the whole lane, about 100 ms per dense layer with today's kernel, which caps run-C-like pipelines at about 18k processed tok/s. A request-sized arena removes the same cost, so do one or the other. With roofline kernels the scan is link-rate and cheap.
3. **Batching is P0/P1: ×1.08–1.33.** It pays once capacity is fixed, and more with roofline kernels (bigger MoE token counts). Sequential per-request attention is as good as fused, so the proposed plan (batch the MoE, attention per request) is the right one; fused attention is not worth building. Best budget: 8k at 8 gx, 16k (today's kernels) / 32k (roofline kernels) at 4 gx.
4. **Async stage handoff grows with pipeline depth and kernel speed:** ×1.12 at 4 gx with today's kernels, ×1.49 at 8 gx with roofline kernels. The measured blocking send is 7.5–18 ms per chunk per stage.
5. **Variable chunk / a2a KV write is worth 1–18%.** It removes padding (15% of processed tokens at chunk 2048) and alignment loss; the modelled a2a cost is small (about 0.1 ms per layer).
6. **Topology.**
   * 8 gx: one 32-stage pipeline beats 2×16.
   * [4,2] beats [2,4] by 3–9%; it needs KV heads sharded 2 per chip.
   * [4,4] and [8,4] stages lose.
   * Auto-split keeps the three dense layers on single-layer stages.
7. **Faster decode raises prefill goodput** (90 → 360 tok/s: 38.4k → 43.8k at 4 gx) because it shrinks the live KV working set per unit of load.
8. **Paging vs pool.** Full paging with the same host tier beats slot lanes + pool only slightly (for example 37.0k vs 36.2k useful tok/s at C=512, 4 gx), because 4 fixed 1M lanes cost 4M of the 21M-token capacity. The pool captures almost all of the benefit of paging.

## Assumptions to revisit

* Host tier: 1 TB DRAM and 64 GB/s PCIe per galaxy, modelled as an ideal page DMA. Both are unverified; see the sensitivity table in the artifact.
* Pool copies are page-list gathers at 50% DRAM efficiency. Arena fragmentation is not modelled.
* Decode is a fixed per-request rate. KV migration to decode is not modelled.
* Meshes without a profile ([4,4], [1,4], …) are extrapolated. TP=2 stages use the [4,2] single-stage profile.
* Batched chunks larger than 5120 are extrapolated from the roofline scaling of each op.

## Extending (notes for the next agent)

* **New feature.** Add a knob to `DEFAULTS` in `sim_core.js`, use it in `roofTok` / `roofSeg` / `layerMs` (cost), `makePlan` (memory), or the scheduler (`formChunk`, `tryStart`). Then add a `FEATURES` entry in `study.js`, a control in `artifact/template.html` (`FIELDS`) and a complexity line in `build_artifact.js` (`SCOPE`).
* **New hardware data.** Rerun `collect_calib.py`. It segments the timing CSVs per cell with `results_16stage.jsonl`. Then run `validate.js` and check the error summary before trusting a study.
* **New corpus.** Rerun `prep_traffic.py`. hash_ids must stay prefix-chained and topologically increasing (checked on 19.6M block pairs).
* `lib/pool.js` `summarize` and the copy in `artifact/template.html` must stay identical.
