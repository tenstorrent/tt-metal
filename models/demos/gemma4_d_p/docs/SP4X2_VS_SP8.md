# Gemma4 prefill: two SP4 halves vs one SP8 Galaxy (AgentX)

Study, 2026-10-27/28. Question: serve Gemma4-31B prefill on a BH Galaxy as one CP8 x TP4 model (SP8, today) or as two independent CP4 x TP4 model copies, one per 4x4 half (SP4 x 2), and how that compares with the sliding-attention lanes change (#61237) on SP8. Focus: AgentX under load.

## TL;DR

- **Under load, two SP4 halves serve AgentX faster than SP8 with lanes.** On silicon (ring fabric), six AgentX slice replays that keep SP8 lanes 92-99% busy all improve on the halves: TTFT p50 by 2-4x (e.g. 1.30 -> 0.46 s, 2.56 -> 0.62 s), p90 by 12-61%, and p99 in five of six (the 300 s window's p99 is worse, 8.8 -> 10.1 s). In the AgentX simulator over the whole 256k corpus, on step tables measured on ring fabric, the halves give +16-21% useful tokens/s once the Galaxy saturates (+9-11% with the measured 1.7M-token prefix cache). Without lanes the halves still give +14-19% (+7-10%): most of the gain needs no op change.
- **Below saturation they tie, and long requests pay.** A lone request runs on 16 chips instead of 32: 1.41x slower per 2k chunk (76.7 vs 54.3 ms). A lone 256k prefill is 1.37x slower at 2k chunks (14.6 vs 10.6 s); at each layout's best chunk that serves accurately (SP8 8k, a half 4k) it is 2.0x slower (13.0 vs 6.5 s). On silicon at 73-84% SP8 load the halves still cut p50 by 25-30% with p90 / p99 within about +-15%; at 60-64% load they tie, except requests above 16K new tokens, which are slower in two of three slices (1.35 -> 1.73 s p50). The sim agrees: below ~64 AgentX sessions throughput is equal and p99 is worse on the halves.
- **Why it wins:** a 2k chunk split 8 ways leaves 256 rows per chip; fixed per-layer costs, 8-rank rings and an idle SDPA grid (16 attention units on 110 cores) dominate. Two halves give each request 512 rows per chip and 4-rank rings, and run two requests at once. The gain grows with prefix depth (global attention on shorter rings): -11% per 4-chunk step at a 2k prefix, -17% at 128k, -19% at 240k.
- **Same KV capacity; accuracy follows the per-chip slab.** Both layouts hold six 256k slots per Galaxy (~1.7M tokens; a half holds three). A half at chunk C is as accurate as SP8 at chunk 2C (same per-chip Q slab, same global attention config): at the serving chunk (2k lanes) a half matches SP8 at 4k and passes every golden gate at 128k and 256k, but it is measurably below SP8 at 2k (256k: overall PCC 0.9801 vs 0.9829, ΔNLL 64-256K 0.31 vs 0.14). At 8k chunks a half cannot get segmented accumulation (it overflows L1) and fails the 128k gates, so a half should not run 8k chunks.
- **Lanes still help on a half (~4% per step, +2-3% in the sim), but most of their SP8 gain comes free from the split.**
- **Cost: serving changes.** Two prefill workers per Galaxy with session affinity, KV migration from 16 chips (the chunk table today requires CP8 / TP4), a 4x4 weight cache, and fixes to the places that assume 8x4 (listed below). None of that exists today.

## Layouts

| | SP8 (today) | SP4 x 2 |
|---|---|---|
| Model copies per Galaxy | 1 on 8x4 | 2, each on a 4x4 half |
| Chips per request | 32 | 16 |
| Rows per chip, one 2k chunk | 256 | 512 |
| Sliding attention units per request | 16 | 32 |
| Ring length (CP collectives, ring SDPA) | 8 | 4 |
| Weights per chip | TP4 shard | same TP4 shard (no extra weight memory) |
| KV per cached token per chip | 1x | 2x (spread over 4 ranks) |
| 256k slots (measured DRAM) | 6 | 3 per half, 6 per Galaxy |
| Max lanes per step | 4 (5-6 hit an L1 clash at compile) | 3 per half |

## Silicon results

Every comparison is within one box and one build: PR 5 head `6c3cb363de7` plus this branch's knobs (kernels unchanged), BH Galaxy at 130 W.

- **Ring fabric (production):** bh-glx-120-b08u08 on 10-28 after its ring links trained, and bh-glx-120-b09u08 on 10-27 (the first measurements, 1 and 4 lanes on SP8, 1 and 2 per half). The two boxes agree within 0.5 ms on SP8 and 2 ms on a half.
- **Linear fabric:** bh-glx-120-b08u08 on 10-28 before that. FABRIC_1D_RING would not map (every link reported 2 of 4 Ethernet channels; two resets did not help), so the morning ran FABRIC_1D with `single_bh_galaxy_mesh_graph_descriptor.textproto` (`GEMMA4_FABRIC=FABRIC_1D`). A reset after an Ethernet link dropped in the evening brought the ring links up. Linear fabric is 14-21% slower in absolute terms, but SP4 x 2 vs SP8 deltas match ring within a point, so its grid, capacity, accuracy and replay results are kept.

### Step time (traced trace replay, ms per step, 2048-token lanes at equal prefixes)

Ring fabric, full grid (lanes, b08u08), prefix 2k / 32k / 128k / 240k:
- SP8, 1-4 lanes: 54.3 / 60.8 / 80.6 / 105.2; 81.7 / 94.9 / 134.6 / 184.2; 104.5 / 124.0 / 184.4 / 260.5; 130.6 / 156.4 / 237.7 / 341.4
- SP4 x 2, 1-3 lanes per half (halves in parallel): 76.7 / 85.4 / 112.8 / 150.2; 115.6 / 132.8 / 196.2 / 276.4; 174.9 / 202.4 / 310.8 / 421.1
- 4 chunks per step, SP4 x 2 at 2 per half vs SP8 at 4: -11% / -15% / -17% / -19%. 3 per half adds no throughput per chunk (174.9 ms for 6 chunks vs 115.6 for 4).
- One request alone: 1.41x at 2k (76.7 vs 54.3 ms), 1.40x at 128k.

Ring fabric, one sliding SDPA per request (no lanes, b08u08), prefix 2k / 32k / 128k / 240k:
- SP8, 1-4 lanes: 54.3 / 60.8 / 80.5 / 105.2; 87.4 / 100.8 / 140.2 / 189.7; 113.4 / 133.3 / 193.1 / 268.0; 142.1 / 168.5 / 249.1 / 351.1 (lanes save 6-12 ms at 4)
- SP4 x 2, 1-2 per half: 76.5 / 85.1 / 112.4 / 150.4; 120.8 / 137.7 / 200.4 / 274.8 (lanes save ~5 ms at 2)
- So SP4 x 2 without lanes (120.8 ms for 4 chunks at 2k) is already faster than SP8 with lanes (130.6).

Linear fabric, full grid (lanes), prefix 2k / 32k / 128k / 240k:
- SP8, 1-4 lanes: 61.8 / 68.2 / 88.1 / 112.6; 94.8 / 107.7 / 147.4 / 196.6; 124.6 / 143.6 / 203.8 / 280.1; 158.0 / 183.3 / 263.3 / 362.5
- SP4 x 2, 1-3 lanes per half: 89.2 / 97.8 / 125.3 / 158.4; 140.4 / 157.5 / 213.7 / 289.7; 211.5 / 237.3 / 331.3 / 439.7
- SP4 x 2 at 2 per half vs SP8 at 4: -11% at 2k, -19% at 128k (ring: -12%, -19%). 3 per half adds no throughput per chunk (211.5 ms for 6 chunks vs 140.4 for 4).
- Both halves at once vs one half alone: within 0-2 ms per step.
- Step time is linear in prefix depth per lane to 240k; the fit `F[n] + slope[n] * sum(prefix)` is within 1 ms of every point.

### Lone 256k prefill (test_model_perf_glx, wall time)

Ring fabric (b08u08 after its links trained; SP8 lands inside its CI baselines):
- chunk 2048: SP8 10.6 s, one half 14.6 s (1.37x)
- chunk 4096: SP8 7.4 s, one half 13.0 s (1.76x)
- chunk 8192: SP8 6.5 s; a half's 8k chunk is not accurate enough to serve (see Accuracy)
- Best accurate chunk each: SP8 6.5 s (8k) vs a half 13.0 s (4k), 2.0x.

Linear fabric: 2048 11.5 vs 16.0 s (1.39x), 4096 8.2 vs 14.3 s (1.75x), 8192 7.1 vs 13.9 s (1.95x).

### KV capacity (DRAM free per chip after building N slots of 262144 tokens)

- SP8, 4 slots: 8.68 GiB free; 6 slots: 1.63 GiB free.
- SP4 half, 3 slots: 1.63 GiB free; 4 slots: out of memory at build.
- So a 256k slot costs 3.5 GiB per chip on SP8 and 7.0 GiB on a half, over ~8.5 GiB of weights and buffers: 6 slots per Galaxy either way (~1.7M tokens).

### Accuracy

- Batched vs unbatched KV (`test_batched_kv_matches_unbatched`, B = 2 and 4, gates: min head in layers 0-15 >= 0.99 and the overall floor): passes on SP8 and on a half. Min head in layers 0-15: SP8 0.99849-0.99966, half 0.99833-0.99966; overall PCC SP8 0.9777-0.9972, half 0.9753-0.9960; the untouched slot stays bit-identical on both.
- Golden KV + next-token likelihood vs the GPU reference (`test_prefill_accuracy`, gutenberg, 128k context; `*` misses its gate). Min per-head PCC (>= 0.928) / overall PCC (>= 0.978) / RRMSE (< 0.208) / top-1, ΔNLL, top-20 KL at 64-128K (>= 0.87, <= 0.39, <= 0.19):
  - SP8, chunk 2048: 0.9434 / 0.9828 / 0.1856 / 0.911, 0.145, 0.120
  - SP8, chunk 4096: 0.9356 / 0.9798 / 0.2012 / 0.891, 0.352, 0.171
  - SP8, chunk 8192: 0.9378 / 0.9802 / 0.1993 / 0.891, 0.250, 0.159
  - half, chunk 2048: 0.9357 / 0.9797 / 0.2017 / 0.893, 0.345, 0.166 (= SP8 at 4096)
  - half, chunk 4096: 0.9357 / 0.9793 / 0.2037 / 0.887, 0.359, 0.193*
  - half, chunk 8192: 0.9280* / 0.9777* / 0.2114* / 0.862*, 0.384, 0.235*
  - 256k context, chunk 2048: SP8 0.9422 / 0.9829 / 0.1852 / top-1, ΔNLL, KL at 64-256K 0.912, 0.144, 0.101; half 0.9356 / 0.9801 / 0.1996 / 0.890, 0.310, 0.146. Both pass every gate; the half is SP8-at-4k quality as at 128k.
  - At 8k context every config misses only the min per-head gate (0.907-0.923; base-limited at short context), with overall PCC 0.979-0.982.
- Why a half's 8k chunk is worse: its 2048-row Q slab gets q96 with no K split and no segmented accumulation (176 Q units > 110 cores); one Q chunk per core needs q_chunk >= 160, whose circular buffers overflow L1 (1.67 MB at q160, 1.87 MB at q192, of 1.57 MB). SP8 uses segmented accumulation at 8k (1024-row slab, q96, 88 units). The half's 2k and 4k chunks take the tuned table's configs keyed by per-chip slab (`GEMMA4_HALF_TUNED_SDPA`, on by default with `GEMMA4_ALLOW_SUBMESH=1`); at 2k they equal the fallback and change no step time (within 1 ms).

### Issues found

- SP8 at 5-6 lanes: 6 slots fit DRAM, but compiling a 5- or 6-lane step fails with "statically allocated dataflow buffers clash with L1 buffers" (`dataflow_buffer.cpp:3096`). SP8 lanes are capped at 4 today.
- SP4 x 2 with one sliding SDPA per request at 3 lanes per half hung (device timeout) while filling the second half's slots on the linear box. The lanes path at 3 per half ran clean; not chased further.
- The global ring SDPA's tuned configs were gated to CP8 x TP4 (`ring_prefill.py`); this branch lets a half use them by per-chip slab. At a half's 8k chunk no config in L1 gets segmented accumulation (see Accuracy).

## AgentX

### Simulator

`agentx_sim/agentx_designs.py` prices each design from its measured step table (`agentx_sim/sp_designs_1028.json`, `designs.py`) instead of the calibrated 8x4 cost model, and runs `agentx_sim.run()` unchanged once per prefill server: SP4 x 2 is two servers, each with every other session (a session's KV stays on its half), half the concurrency and half the prefix cache. Host time per step and the scheduler gap come from the existing batched calibration, the same for every design. Every ring-fabric design is measured at every lane count and prefix (the JSON keeps the linear-fabric tables as `*_linear` for the linear-fabric replays).

Whole 256k corpus, think cap 60 s, SP8 lanes (4 lanes) vs SP4 x 2 lanes (2 per half), both from ring-fabric grids measured at every lane count and prefix; steady-state useful tok/s and TTFT p50 / p90 / p99 (s):

- Infinite cache, FCFS: 32 sessions -1% (p90 1.06 -> 1.66, p99 11.2 -> 18.4); 64: -1% (p90 3.24 -> 3.41, p99 16.3 -> 23.6); 96: +6% (p50 1.89 -> 0.81); 128: +16% (p50 6.13 -> 2.66, p90 14.7 -> 12.0, p99 34.3 -> 34.1); 192: +19% (p50 13.4 -> 8.1, p90 28.7 -> 22.8, p99 47.5 -> 57.5).
- Infinite cache, pack (fewest rows left first): -1% at <= 64, +8% / +21% / +21% at 96 / 128 / 192; p90 0.87 -> 1.17 and p99 36 -> 58 at 128.
- 1.7M-token cache (the measured capacity; saturated from 32 sessions; hit rate 0.65 -> 0.62 from splitting the cache): FCFS +9-11% with lower p50 / p90; pack +7-9% with p90 roughly doubled (5.5 -> 10.5 s).
- 3 lanes per half instead of 2: 0 to -2%. SP8 without lanes (one SDPA per request): -0% to -6% vs SP8 lanes.
- SP4 x 2 without lanes vs SP8 with lanes: -1% at <= 64 sessions, +5% / +14% / +16% at 96 / 128 / 192 (infinite cache, FCFS), +7-10% with the 1.7M-token cache. The split alone carries most of the gain.
- The linear-fabric tables give the same picture (+18% at 128 sessions, infinite cache, FCFS).

### Silicon replay (checks the simulator and the design)

Five 2-minute AgentX slices (`agentx_slice.py build` at 96 sessions, starts 300 / 600 / 900 / 1500 / 2100 s, 481-532 requests and 1.7-2.2M new tokens each) and one 5-minute window (600-900 s, 1238 requests, 5.1M new tokens), each replayed open loop by `demo/agentx_replay.py` at its own rate (1x) and with arrivals compressed to 1.25x and 1.67x. SP8 lanes run with 4 slots on 8x4; each half replays its share (every other request, `AGENTX_HALF`) on one 4x4 half with 2 slots, one half after the other (both halves at once measure within 0-2 ms per step of one alone). `DEMO_BATCHED_ONLY=1` captures 2k layouts only. Ring fabric, b08u08, 10-28; SP8 lanes -> SP4 x 2 lanes, sorted by SP8 load (the halves run 13-19 points less busy):

| Slice start | Rate | SP8 busy | TTFT p50 (s) | p90 (s) | p99 (s) | > 16K new tokens p50 (s, n) |
|---|---|---|---|---|---|---|
| 1500 s | 1.67x | 99% | 2.56 -> 0.62 | 6.30 -> 2.47 | 8.30 -> 5.43 | 5.99 -> 3.43 (21) |
| 600 s, 300 s window | 1.67x | 98% | 1.84 -> 0.62 | 4.52 -> 2.91 | 8.79 -> 10.12 | 4.36 -> 2.92 (54) |
| 600 s | 1.67x | 97% | 1.30 -> 0.46 | 2.36 -> 1.80 | 3.69 -> 2.86 | 3.34 -> 2.10 (19) |
| 300 s | 1.67x | 95% | 1.99 -> 0.71 | 4.53 -> 2.47 | 5.54 -> 3.40 | 3.52 -> 1.93 (32) |
| 2100 s | 1.67x | 95% | 1.40 -> 0.59 | 4.25 -> 2.34 | 5.55 -> 3.65 | 4.68 -> 2.44 (30) |
| 900 s | 1.67x | 92% | 0.83 -> 0.42 | 2.34 -> 2.07 | 4.77 -> 4.18 | 3.55 -> 3.12 (19) |
| 1500 s | 1.25x | 84% | 0.43 -> 0.31 | 1.41 -> 1.42 | 3.99 -> 3.74 | 2.66 -> 2.48 (21) |
| 600 s, 300 s window | 1.25x | 79% | 0.36 -> 0.29 | 1.31 -> 1.29 | 3.93 -> 5.07 | 2.07 -> 2.09 (54) |
| 300 s | 1.25x | 78% | 0.37 -> 0.27 | 1.44 -> 1.28 | 2.74 -> 2.30 | 1.81 -> 1.60 (32) |
| 2100 s | 1.25x | 75% | 0.36 -> 0.26 | 1.26 -> 1.17 | 2.81 -> 2.48 | 1.98 -> 1.84 (30) |
| 900 s | 1.25x | 73% | 0.34 -> 0.26 | 0.95 -> 1.09 | 2.54 -> 2.85 | 1.94 -> 2.15 (19) |
| 300 s | 1x | 64% | 0.23 -> 0.22 | 0.96 -> 0.94 | 2.48 -> 2.14 | 1.37 -> 1.38 (32) |
| 600 s | 1x | 63% | 0.22 -> 0.21 | 0.65 -> 0.67 | 1.89 -> 1.85 | 1.35 -> 1.73 (19) |
| 900 s | 1x | 60% | 0.27 -> 0.21 | 0.70 -> 0.73 | 2.25 -> 2.43 | 1.38 -> 1.88 (19) |

- Saturated (SP8 92-99% busy): the halves win p50 by 2-4x, p90 by 12-61%, and p99 in five of six slices; requests above 16K new tokens get 12-49% faster.
- Moderate (73-84%): p50 25-30% lower in every slice; p90 and p99 within about +-15% either way.
- Light (60-64%): a tie for most requests; requests above 16K new tokens are slower on a half in two of three slices (1.35 -> 1.73, 1.38 -> 1.88 s), as a lone long request is.
- Simulator vs replay (`agentx_slice.py predict --design sp8_lanes|sp4_lanes --split k/2`, `--measured`): step time 3-5% high for both designs; predicted TTFT percentiles are high by a similar margin for both, so the predicted winner matches the measured one at every load.
- Linear fabric (the morning, before the ring links trained; slower steps, so a slice loads the box harder): at 1.43x on the 600 s slice, SP8 96% busy, p50 1.04 -> 0.46 / p90 2.36 -> 1.90 / p99 4.41 -> 3.26 s. 3 slots per half instead of 2: p50 0.50, p90 1.81, p99 4.21 s; a 3-lane step costs as much per chunk as two 2-lane steps and the longer steps hurt the tail, so 2 slots per half is the better setting.

## Serving impact

- Two prefill workers per Galaxy: the engine must route a session's requests to the half that holds its prefix (a miss re-prefills it), and balance load across halves.
- KV migration: `kv_chunk_table.py` rejects anything but CP8 / TP4; the chunk-table contract itself (pairs by config / slot / layer / position, chip and mesh free) needs no change, but each request's KV leaves from 16 chips, so a single handoff carries 2x per chip.
- Model code that assumes 8x4 (this branch gates each behind `GEMMA4_ALLOW_SUBMESH=1` for the study): `config.validate_galaxy_mesh`, `MeshConfig.total_devices`, `kv_validation.read_cache_tensor`, the adapter's `weight_cache_path` and `validate_params`, and the tuned global SDPA configs.
- A 4x4 weight cache (~35 GB) next to the 8x4 one.
- Running the halves as two processes (as serving would) is not measured; this study drives both from one process.

## Open questions

- Lanes accuracy at 3 per half (only B = 2 and 4 on one half were checked).
- A scheduler that places sessions on halves by load instead of alternation, and mixed-depth steps.
- Two processes per Galaxy (as serving would run the halves); this study drives both from one process and replays the halves one at a time.
- The SP8 5-6 lane L1 clash: SP8 with 6 lanes might close part of the gap under load.
- KV migration time from 16 chips.

## Reproduce

Tests and tools on this branch (env knobs default off):

```bash
# SP8 step grid (lanes / GEMMA4_SLIDING_LANES=0 for the loop)
PERF_LANE_COUNTS=1,2,3,4 PERF_MIXES=shallow,u32k,mixed,deep,d240k PERF_SKIP_ALONE=1 \
  pytest "models/demos/gemma4_d_p/tests/test_batched_prefill.py::test_batched_step_perf" -s
# SP4 x 2 step grid (needs a TT_CACHE_PATH with tensor_cache_bf16_mesh4x4; GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1 builds it)
TT_CACHE_PATH=<4x4 cache root> SP4X2_LANES=3 SP4X2_MIXES=shallow,u32k,mixed,deep,d240k \
  pytest models/demos/gemma4_d_p/tests/test_sp4x2_step_perf.py -s
# One half: accuracy, lone 256k
GEMMA4_TEST_SP4=1 TT_CACHE_PATH=<4x4 cache root> pytest "models/demos/gemma4_d_p/tests/test_batched_prefill.py::test_batched_kv_matches_unbatched" -s
GEMMA4_TEST_SP4=1 TT_CACHE_PATH=<4x4 cache root> pytest models/demos/gemma4_d_p/tests/test_prefill_accuracy.py -k "8k or 128k" -s
GEMMA4_TEST_SP4=1 TT_CACHE_PATH=<4x4 cache root> GEMMA4_PERF_CHUNKS=2048,8192 pytest models/demos/gemma4_d_p/tests/test_model_perf_glx.py -s
# A Galaxy whose ring links do not train: GEMMA4_FABRIC=FABRIC_1D TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto

# AgentX (host only)
cd models/demos/gemma4_d_p/agentx_sim
./agentx_designs.py --traces <dir>/traces.jsonl --designs sp8_lanes:1x4,sp4_lanes:2x2 --concurrency 32,64,96,128,192 --cache inf,1.7e6 --policy fcfs,pack -j 16
./agentx_slice.py build --traces <dir>/traces.jsonl --out slice.json --concurrency 96 --start 600 --window 120
# replay: SP8, then each half's share
AGENTX_SLICE=slice.json AGENTX_OUT=sp8.json AGENTX_MODES=batched DEMO_SLOTS=4 DEMO_BATCHED_ONLY=1 pytest models/demos/gemma4_d_p/demo/agentx_replay.py -s
GEMMA4_TEST_SP4=1 AGENTX_HALF=0 AGENTX_SLICE=slice.json AGENTX_OUT=half0.json AGENTX_MODES=batched DEMO_SLOTS=2 DEMO_BATCHED_ONLY=1 TT_CACHE_PATH=<4x4 cache root> pytest models/demos/gemma4_d_p/demo/agentx_replay.py -s
./agentx_slice.py predict --slice slice.json --design sp4_lanes --slots 2 --split 0/2 --measured half0.json
```
