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
* **Decode.** A request ends at prefill done + `out / decodeTps` (default 60 tok/s).
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

## Findings (study of Sep 28 2026, `results/study.json`)

Goodput in useful tok/s at p90 TTFT ≤ 10 s. "Today" = 16×[2,4] (or 32×[2,4]), chunk 2048, auto split, static 1M slots.

| scenario | today | greedy full stack | best grid config | same with ∞ cache |
|---|---|---|---|---|
| 4 gx, today's kernels | 1.8k | 31.8k | **36.1k** (16×[4,2], 4 lanes, budget 8k) | 56.7k |
| 4 gx, roofline kernels | 3.2k | 61.2k | **65.0k** (16×[4,2], 2M arena, budget 16k) | 197k |
| 8 gx, today's kernels | 3.5k | 73.5k | **80.5k** (32×[4,2], 2M arena, budget 8k) | 136k |
| 8 gx, roofline kernels | 6.1k | 125.6k | **128.2k** (32×[4,2], 2M arena, budget 8k) | 301k |

Greedy order: gain at the step the feature was added, followed by leave-one-out loss when removing it from the full stack.

Complexity is a rough judgement of how much of the stack a feature touches:
* **low**: one op or the scheduler, plus validation;
* **med**: a new op or allocator, or changes across a few ops;
* **high**: cross-cutting changes to the runtime, KV layout, several kernels and the scheduler.

It is not a time estimate and has not been checked with the code owners.

| tier | feature | 4gx today | 4gx roofline | 8gx today | 8gx roofline | complexity |
|---|---|---|---|---|---|---|
| P0 | slot lanes + paged pool | ×3.68 / ×8 | ×4.04 / ×5.6 | ×2.40 / ×6.3 | ×5.75 / ×6.6 | high |
| P0 | host-DRAM KV tier (1 TB/gx) | ×1.65 / ×2.0 | ×2.02 / ×1.9 | ×1.63 / ×1.6 | ×1.51 / ×1.7 | high |
| P0 | bounded dense gather (while lanes are 1M) | ×1.42 | ×1.00 | ×2.83 | ×1.00 | low |
| P0 | index_k stored once (not ×TP) | ×1.32 / ×1.28 | ×1.31 / ×1.29 | ×1.20 / ×1.15 | ×1.27 / ×1.32 | med |
| P1 | async stage handoff | ×1.09 / ×1.12 | ×1.18 / ×1.25 | ×1.22 / ×1.15 | ×1.38 / ×1.43 | med |
| P1 | multi-request batching (8–16k budget) | ×1.09 / ×1.19 | ×1.17 / ×1.30 | ×1.07 / ×1.07 | ×1.07 / ×1.15 | high |
| P1 | variable chunk (a2a KV write) | ×1.14 / ×1.04 | ×1.05 / ×1.05 | ×1.11 / ×1.02 | ×1.09 / ×1.02 | high |
| P1 | index_k bf8 | ×1.11 / ×1.07 | ×1.14 / ×1.10 | ×1.03 / ×1.06 | ×1.08 / ×1.12 | low (PCC) |
| P2 | shortest-first scheduling | ×1.03 | ×1.02 | ×1.05 | ×1.07 | low |
| P2 | MSA SP-local indexer | ×1.01 | ×1.02 | ×1.01 | ×1.00 | high |
| P2 | fused multi-user attention | ×1.00 | ×1.01 | ×1.00 | ×1.00 | high |
| P2 | unaligned resume (subsumed by var) | ×1.00 (×1.07–1.12 before var) | ×1.00 | ×1.00 | ×1.00 | low |
| P2 | variable-size lane arena | ×0.99 | ×1.02 | ×1.00 | ×1.00 | med |

Takeaways:
1. **On AgentX, KV capacity sets the throughput, not compute.** Even the best stacks reach only 30–65% of their own ∞-cache goodput.
   * The pool (vs static 1M slots), the host tier and index_k de-replication are the top features in every scenario.
   * Host DRAM size matters much more than PCIe bandwidth: 0.5 → 2 TB/galaxy moves 4-gx goodput 28k → 46k (today's kernels) and 48k → 85k (roofline kernels). 16 → 256 GB/s moves it by at most 15%.
2. **Bounded dense gather is P0 while lanes are 1M slots.** Every chunk scans the whole lane, about 100 ms per dense layer with today's kernel, and that caps run-C-like pipelines at about 18k processed tok/s. A request-sized arena removes the same cost, so do one or the other. With roofline kernels the scan is link-rate and cheap.
3. **Batching is P1: 7–30%.** It pays once capacity is fixed and more with roofline kernels (bigger MoE token counts). Sequential per-request attention is as good as fused, so the proposed plan (batch MoE, attention per request) is the right one; fused attention is not worth building. The best budget is 8k with today's kernels and 16k at 4 gx with roofline kernels.
4. **Variable chunk / a2a KV write is worth 2–14%.** It removes padding (15% of processed tokens at chunk 2048) and alignment loss; the modelled a2a cost is small (about 0.1 ms per layer).
5. **Async stage handoff grows with pipeline depth and kernel speed:** ×1.09 at 4 gx with today's kernels, ×1.38 at 8 gx with roofline kernels. The measured blocking send is 7.5–18 ms per chunk per stage.
6. **Topology.**
   * 8 gx: one 32-stage pipeline beats 2×16.
   * [4,2] beats [2,4] by 1–10%; it needs KV heads sharded 2 per chip.
   * [4,4] and [8,4] stages lose.
   * Auto-split keeps the three dense layers on single-layer stages.
7. **Faster decode raises prefill goodput** (30 → 120 tok/s: 27.6k → 39.4k at 4 gx) because it shrinks the live KV working set per unit of load.

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
