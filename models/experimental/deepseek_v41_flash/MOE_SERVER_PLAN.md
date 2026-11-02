# Disaggregated MoE server: plan review

Status: proposal, reviewed against the checkpoint config, the hardware and the measured profiles in
`DSPARK_100TPS.md`, `DECODE_SDPA_IMPROVEMENTS.md` and `DECODE_LIGHTNING_INDEXER_IMPROVEMENTS.md`. Every
number is *measured* (with its source), *exact* (from the config) or *estimated* (with how). Attention at
TP1 and TP4 was profiled for this document (Nov 2, `tests/decode/test_attention.py -k profile`, section
2.2); everything else on the device side comes from earlier profiles.

## 0. Verdict

**The plan as written will not speed up per-user decode.** At one row it is estimated at ~30-39 ms per
token, against 27.9 ms measured today and the 18-20 ms that `DSPARK_100TPS.md` WS3 targets on the
current layout. It also moves 100 tok/s/user further away. Three reasons:

1. **At one row, expert parallelism cannot use more chips than today.** A token picks 6 experts, so it
   reads from at most 6 chips. With random placement the busiest chip holds 1.9 of them (EP12) or 1.5
   (EP24), about 30-38 MB. That is the same as today's TP4, which reads 30 MB per chip because every
   chip holds a quarter of every expert (section 2.1).
2. **Attention on one chip is measured at 2x the time of TP4**: 468 µs per layer against 238 µs at
   32k context (18.7 ms against 9.5 ms per token). The head-sharded projections, SDPA and the output
   RoPE all take ~4x longer on one chip. The ~150 µs per layer of non-attention work (mHC, MixStreams,
   norms, step context) would also land on the attention chip (section 2.2).
3. **Every layer adds 3 serial hops**: attention to router, router to experts, experts back to
   attention. That is 120 hops per token where today there are 7 stage hops.

**It can raise aggregate throughput at high concurrency, but only in the 24-MoE-chip variant.** The real
gain is that one global MoE pass batches every user through each expert once, where an 8-stage pipeline
splits the users into 8 rows that each read the experts on their own. In a bandwidth-only model, EP24 is
~15% behind today up to 64 users and gives ~1.4x at 128 users and ~2.4x at 256 users. EP12 does worse
than today below ~200 users, because it gives the experts 12 chips' bandwidth where they have 32 today
(section 4).

**Some parts of the plan are worth doing on their own:** persistent kernels with fewer trace boundaries,
routing on the attention chip, and a sequence-sharded indexer for long context. Section 6 orders them.

## 1. Facts used

| what | value | source |
|---|---|---|
| hardware | 32 Blackhole chips (Galaxy, `1e52:b140`), 32 GB GDDR6 and ~512 GB/s each | `lspci`, spec |
| layers | 40 (+3 DSpark MTP stages) | config |
| hidden size \(D\), expert intermediate \(I\) | 5120, 2304 | config |
| routed experts | 384, top-6, plus 1 shared expert | config |
| one routed expert, bfp4 | \(3 \cdot 2304 \cdot 5120 \cdot 0.5625\) B = **19.9 MB** | exact |
| all routed experts, 40 layers, bfp4 | \(384 \cdot 40 \cdot 19.9\) MB = **306 GB** | exact |
| shared expert, bf8 | 38 MB per layer, 1.5 GB in total | exact |
| attention weights per layer | ~130 M params, ~73 MB at bfp4 (+ bf16 compressor on compressed layers) | `DSPARK_100TPS.md` 1.1 |
| current layout | 8 stages of 1x4 (TP4), 5 layers per stage | `tt/config.py` |
| current MoE split | every chip of a stage holds \(I/4\) of all 384 experts (TP inside the expert) | `tt/decode/moe.py` |
| layer kernel time at TP4 | ~530 µs: `fused_experts` ~170, replicated work ~208 (mHC 84, MixStreams + norms ~40, step context ~25, router ~16, `q_a` / `kv` ~25), attention the rest | measured Oct 27 |
| token latency, one row | 27.9 ms; ~21 ms kernels, ~6 ms hops, trace boundaries and host | measured Oct 29 |
| attention module per layer, 32k, averaged over the 40 layers | 468 µs at TP1, 238 µs at TP4 | measured Nov 2 (section 2.2) |
| SDPA, 640 keys | 24.6 µs at TP1 (64 heads, 64 cores), 12.9 µs at TP4 (16 heads, 16 cores) | measured Nov 2 (the 68.8 µs in `DECODE_SDPA_IMPROVEMENTS.md` predates its fixes) |
| lightning select at 32k, layer 20 | 134 µs at TP1; 74 + 17 (all-gather) + 46 (merge) µs key-split at TP4 | measured Nov 2 |
| KV / index caches | bf16, `head_dim` 512, `index_head_dim` 128 | `build_decode_caches` |
| D2D hop latency on this Galaxy | **not measured**; kept as \(h\), with 5-20 µs as the working range | none |

## 2. Assumptions, checked

### 2.1 "MoE scales better with more chips, DRAM bandwidth bound"

**Partly right. It scales with chips for throughput, not for one row's latency, and only if the chips
hold slices of the same experts.**

* Today's `fused_experts` runs at ~170 µs against a ~60-75 µs bandwidth floor. It is about 2.3x off the
  floor, so it is partly bound by latency and per-op overhead, not by bandwidth alone. The whole layer
  is 4.6x off its floor (`DSPARK_100TPS.md` 1.1).
* For one row, the bytes per token are fixed (6 experts, ~120 MB). Splitting them over more chips only
  helps if each chip reads a slice of *each* selected expert, which is TP inside the expert (today's
  scheme). Under expert parallelism a token touches at most 6 chips, and the busiest chip sets the
  latency. Simulated over random placements:

  | layout | experts on the busiest chip | bytes on the busiest chip |
  |---|---|---|
  | today, TP4 | 6 x 1/4 | 30 MB |
  | EP12 | 1.88 on average (all 6 on different chips 23% of the time) | 37 MB |
  | EP24 | 1.50 on average (53%) | 30 MB |
  | EP32 | 1.38 on average (63%) | 27 MB |

  So EP gives no bandwidth gain at one row. On top of that, today's op splits one expert over 16
  cores. With 1-2 experts per chip only 16-32 of ~120 cores would stream, far below the chip's DRAM
  bandwidth, so the op would have to be redesigned to put one expert on the whole grid.
* With many tokens, expert parallelism does pay: each expert is read once for every token routed to it
  (section 4).

**Correction:** MoE is weight-bandwidth bound *in the limit*, and it scales with chips *in aggregate
throughput*. Per-row latency scales only with TP inside the expert, and at today's op overheads not
even with that (`DSPARK_100TPS.md` 1.1: wider TP adds collectives to a latency-bound layer).

### 2.2 "Attention doesn't scale well"

**Mostly wrong for the attention module itself: TP4 is measured at 2.0x faster than one chip.**

Profile (Nov 2): `test_attention_decode_profile` (one chip) and `test_attention_decode_profile_tp4` (1x4
submesh), bfp4 weights, one layer of each decode variant, the caches filled with random values as if the
context had been decoded, 4 profiled positions per layer after 4 warm-up positions. Device kernel time
per op; under TP4 each op is the slowest of the 4 chips. Eager dispatch, so gaps between ops are not
counted. Reports: `generated/profiler/reports/2026_11_02_06_33_21` (TP1) and `2026_11_02_06_35_35` (TP4).

Per layer, by variant:

| layer (variant) | layers like it | TP1, 32k | TP4, 32k | TP1 / TP4 |
|---|---|---|---|---|
| 0 (sliding only) | 2 | 413 µs | 190 µs | 2.2 |
| 2 (ratio-2 kv + index source) | 3 | 683 µs | 453 µs | 1.5 |
| 3 (ratio-2 consumer) | 15 | 426 µs | 195 µs | 2.2 |
| 20 (ratio-1 candidate source) | 1 | 713 µs | 486 µs | 1.5 |
| 21 (ratio-1 consumer) | 15 | 426 µs | 195 µs | 2.2 |
| 24 (ratio-1 index-only source) | 4 | 592 µs | 363 µs | 1.6 |
| **40-layer token** | | **18.7 ms** | **9.5 ms** | **2.0** |

At 4k the token is 18.4 / 9.1 ms, and at 128k it is 19.1 / 9.7 ms. The context only reaches the index
layers, and slowly: the select is random-key, best case (real keys cluster).

Op by op, for a consumer layer (layer 21, 32k):

| op | TP1 | TP4 | scales? |
|---|---|---|---|
| `q_a` (replicated) | 28.0 µs | 28.0 µs | no |
| `q_b` (head-sharded) | 72.5 µs (325 GB/s) | 19.6 µs | 3.7x |
| `kv` (replicated) | 15.9 µs | 15.9 µs | no |
| q RoPE / kv RoPE / ring write | 15.9 µs | 10.5 µs | partly |
| SDPA, 640 keys | 24.6 µs | 12.9 µs | 1.9x |
| output RoPE, 1 core | 89.1 µs | 23.1 µs | 3.9x (one core, 4x the heads) |
| `o_a` (output groups) | 101.5 µs (186 GB/s) | 28.2 µs | 3.6x |
| `o_b` (row-parallel) | 71.2 µs (332 GB/s) | 18.4 µs | 3.9x |
| all-reduce of `o_b` | none | 15-34 µs (fastest / slowest chip) | TP cost |
| the rest (all-gathers for matmul) | 7.6 µs | 4.6 µs | |
| **total** | **426 µs** | **195 µs** | **2.2x** |

What does not scale is `q_a`, `kv` and the per-layer index and compressor work (~250 µs extra on an
index source, the same on both). The select's key split under TP4 buys nothing at 32k: 134 µs on one
chip against 74 + 17 + 46 µs split. The projections, SDPA and the output RoPE scale 3.6-3.9x, and the
TP4 all-reduce gives back 15-34 µs of that. The ~150 µs per layer of non-attention work in the Oct 27
profile (mHC 84, MixStreams + norms ~40, step context ~25) is not in this test. It is replicated at
TP4 and would move whole to the attention chip.

Two TP1 costs are kernel problems rather than limits: the output RoPE on one core (89 µs; on 64 cores
the q RoPE does the same shape in ~10 µs) and `o_a` at 186 GB/s where `q_b` and `o_b` reach ~330. Fixing
both brings a TP1 consumer layer to ~300 µs. The same RoPE fix brings TP4 to ~175 µs, so TP4 stays
~1.7x ahead.

**Correction:** the attention module scales about 2x from one chip to TP4, 3.6-3.9x on its head-sharded
parts. What doesn't scale is `q_a` / `kv`, the index and compressor work, and the non-attention
per-layer work. Going to one chip keeps all of those and only drops one all-reduce.

### 2.3 "DeepInfer uses expert parallelism, but attention is on a single chip"

I could not identify "DeepInfer". I assume this means DeepSeek's published inference system (V3
technical report section 3.4, and the Feb 2025 "DeepSeek-V3/R1 Inference System Overview"). The figures
below are from memory of those public reports and should be checked before they are quoted.

* Decode uses EP for the routed experts (EP144, ~2 routed experts per GPU) and DP for MLA and the shared
  expert (DP144). **Attention and experts sit on the same GPUs.** There are no attention-only GPUs and no
  router GPU.
* It is a **throughput** design: large batches per GPU, two micro-batches overlapping attention with
  dispatch / combine, and redundant copies of hot experts for load balance. The published average output
  speed is ~20-22 tok/s per user.
* DP attention pays off there because MLA's KV reads dominate at long context and large batch, and DP
  avoids replicating KV. V4.1's CSA makes each user's SDPA read only 640 rows (~24 MB per token over 40
  layers at bf16). At V4.1's batch sizes the cost is the **weights**, and DP replicas each read all of
  them.

**Correction:** the reference design co-locates attention and experts on every chip, targets aggregate
throughput rather than per-user speed, and relies on large batches. It is not evidence that
single-chip attention lowers per-user latency.

### 2.4 "TP4 has too few chips for MoE and too many for attention"

**Not right as stated.** Today the experts are already spread over all 32 chips: each stage holds 5
layers' experts, ~9.5 GB per chip. The issue is not too few MoE chips. At one row, 28 of 32 chips sit
idle while one stage works, so the pipeline is underused. More rows in flight fix that; moving experts
to fewer chips does not. And attention at TP4 is not wasted (2.2).

### 2.5 Claims in the plan itself

| claim | check |
|---|---|
| "12 / 24 chips, 16 / 32 experts per chip" | Read as 12 or 24 MoE chips. 384 experts give 32 or 16 per chip **per layer**, so 1280 or 640 per chip over 40 layers. |
| 12 MoE chips hold all experts | 306 GB / 12 = **25.5 GB per chip** of routed experts, plus DSpark experts (128 x 3 x 19.9 MB = 7.6 GB at bfp4, 0.64 GB per chip) and buffers. That fills a 32 GB chip, leaves no room for bf8 experts or for redundant hot experts, and is risky for prefill activations. 24 chips need 12.7 GB per chip, which is comfortable. |
| a router chip | It works, but it adds one serial hop per layer (40 per token), and one chip has to serve every replica's every layer. The gate is 384 x 5120 (2 MB at bf8, 80 MB for 40 layers). Route on the attention chip and dispatch directly. |
| one chip does attention for all layers | It fits: ~3 GB of attention weights at bfp4, 1.5 GB of shared experts if they live there, and the 372 MB `lm_head`. Its cost is latency (2.2, section 3). With 12 MoE chips + 1 router there are 19 such chips, so it is DP attention with 19 replicas, each reading every attention weight per step. |
| CSA limits SDPA to 128 + 512 = 640 rows | **Right** for compressed layers (`sliding_window` 128 + `index_topk` 512). Sliding-only layers see 128. But SDPA is not the only per-context cost (next row). |
| long context needs indexer / compressor chips | **Right for the indexer, not for the compressor.** The compressor is O(1) per step (it compresses the newest 1-2 positions). The indexer is O(N) on 4 layers only: 2, 8 and 14 score N/2 keys, and 20 (the candidate source) scores N. Layers 24-36 select from a pool capped at 2048 x 8 = 16384 candidates. |

## 3. Latency model, one row

Per layer at 32k. Attention is measured (section 2.2). The non-attention work is from the Oct 27
profile. Experts, routing and the shared expert are estimated at ~400 GB/s achieved (78% of peak):

| piece | today, TP4 | plan |
|---|---|---|
| attention module (measured, 40-layer average) | 238 µs | 468 µs on one chip |
| non-attention per-layer work (mHC, MixStreams, norms, step context) | ~150 µs, replicated | ~150 µs on the attention chip |
| routing | ~16 µs | ~20 µs on the router chip |
| routed experts | ~170 µs | ~95-115 µs with a full-grid op (1.5-1.9 experts x 19.9 MB); ~170+ µs with today's 16-core split |
| shared expert (bf8, 38 MB) | TP4-split, ~9.5 MB per chip, inside the per-layer total | ~95 µs serial on the attention chip, or ~0 if it runs on MoE chips alongside the routed ones |
| hops | TP collectives, inside the numbers above | 3h, with \(h\) the D2D hop latency |
| **per layer** | **~530 µs** measured Oct 27; ~575 µs with the Nov 2 attention number in place of that profile's | **~735-905 µs + 3h** |

Over 40 layers, with \(h\) = 5-20 µs, the plan's kernels and hops come to **~30-39 ms**. Persistent
kernels remove trace boundaries, which takes back part of today's ~6 ms of gaps. Today is **27.9 ms**,
and the same gap removal applied to today's layout gives ~22 ms. **There is no per-user speedup.** The
plan is slower than today, and further behind an improved baseline.

The plan would only win per-user latency if all of these held at once: the attention chip at ≤ 350 µs
per layer including the non-attention work (the attention module alone is 468 µs today, ~300 µs after
the RoPE and `o_a` fixes, so the ~150 µs of non-attention work would also have to go almost entirely),
a full-grid expert op at ≤ 100 µs, the shared expert off the critical path, and \(h\) ≤ 5 µs. Every one
of those except \(h\) helps today's layout just as much.

## 4. Throughput model, many users

Bandwidth-only model: \(U\) concurrent users, every user advances one token per step, perfect overlap of
attention and MoE, no load imbalance or queueing, ~400 GB/s per chip, 32k context. A step can't be
shorter than one user's serial chain: 27.9 ms today, and ~33 ms for the plan (the middle of section 3's
range). Expert reads per layer are the union of
the experts the tokens of one pass pick: \(384(1-(1-6/384)^n)\) for \(n\) tokens.

* Today: 8 rows of \(U/8\) users, so \(8 \times\) the union over \(U/8\) tokens. Attention weights are read once per row.
* Plan: one global pass, so the union over \(U\) tokens, spread over the MoE chips. Each DP attention
  replica reads all 40 layers' attention and shared-expert weights (~4.5 GB) per step, plus ~60 MB of
  KV and index keys per user.

| \(U\) | expert reads per layer: today / plan | today tok/s | EP12 (+19 attention) tok/s | EP24 (+7 attention) tok/s |
|---|---|---|---|---|
| 8 | 48 / 46 | ~290 | ~240 | ~240 |
| 32 | 188 / 152 | ~1150 | ~970 | ~970 |
| 64 | 364 / 244 | ~2290 | ~1580 | ~1940 |
| 128 | 684 / 333 | ~2760 | ~2320 | ~3880 |
| 256 | 1216 / 377 | ~3200 | ~4090 | ~7760 |

Up to ~64 users both layouts are bound by their serial chains, and the plan's is longer, so the plan
is ~15% behind. Beyond that, today's
pipeline reads every expert several times per step (once per row), and the global pass reads it once.
That is the plan's real advantage, and it only shows with enough MoE chips. With 12 the experts get 37.5%
of today's bandwidth.

These are upper bounds for the plan. Real EP loses some of the gain to load imbalance (hot experts;
DeepSeek replicates them), to queueing when 7-19 replicas arrive at different layers, and to dispatch /
combine overhead at large token counts. Today's layout has not been run at 32 users per row either.

## 5. Memory

| chip | plan, EP12 | plan, EP24 |
|---|---|---|
| MoE chip | 25.5 GB routed + ~0.6 GB DSpark + buffers: near the 32 GB limit | 12.7 GB + ~0.3 GB: fine; room for redundant experts or bf8 |
| attention chip | ~3 GB attention + 1.5 GB shared + 0.4 GB head, plus KV | same |

KV and index keys at bf16 are ~4.2 KB per position per user (`comp_kv` for kv sources 2, 8, 14 at N/2
and 20 at N; index keys for 8 index sources). That is ~134 MB at 32k, ~540 MB at 128k and ~4.3 GB at 1M
per user. One attention chip holds ~40 users at 128k, and ~5 at 1M.

## 6. Revised plan

### Phase 0: measure before building (device runs, by hand)

1. D2D socket latency between Galaxy chips at 10-40 KB payloads, adjacent and far. This is \(h\), and
   the plan's per-layer cost is 3h.
2. ~~One layer's attention at TP1, per op, against TP4.~~ Done Nov 2 (section 2.2): 468 µs against
   238 µs per layer at 32k.
3. A `fused_experts` variant with 1-2 experts on a full grid, and its time per expert.
4. The expert-union size in real traffic: routing correlation across users can make the union smaller
   than the random-routing numbers in section 4.

Go / no-go: if (2) is above 400 µs or (1) above 10 µs, the latency case is closed, and only the
throughput case (Phase 3) remains. (2) came in at 468 µs, so the latency case is closed. (1) and (3)
still matter for the throughput case.

### Phase 1: the parts that pay on today's layout

* WS3 from `DSPARK_100TPS.md`: cut the ~208 µs of replicated work, `fused_experts` towards ~110 µs,
  balance the stages. Target L = 18-20 ms.
* Persistent / fused per-layer kernels and fewer trace boundaries: the "persistent kernel" idea of this
  plan, applied inside the pipeline (`tt-blaze`'s fused DeepSeek-V4.1 layer kernels are prior art).

### Phase 2: long context (independent of EP)

* Shard the index keys **and** `comp_kv` of the O(N) layers only (2, 8, 14, 20) across S indexer chips
  by position. Each chip scores its slice and returns its local top-512 (scores, rows). The attention
  chip merges, then attends 512 + 128 rows. That adds one round trip per index-source layer: 4 per
  token, not 40.
* Leave the compressor on the attention / stage chips. It is O(1) per step.
* Leave layers 24-36 alone below 1M. Their pool is capped at 16384 candidates.

### Phase 3: high-concurrency throughput (only if that is the target)

If the goal is aggregate tok/s at ≥ 128 users rather than tok/s/user:

* **24 MoE chips, not 12.** 12 is close to the memory limit and gives the experts too little bandwidth.
* **Route on the attention chip** and dispatch straight to the expert chips. Drop the router chip
  (frees a chip and 40 hops per token).
* Put the shared expert on the attention chip only if the attention chip has slack; otherwise TP-split
  it over the expert chips, alongside the routed experts.
* Two micro-batches per attention replica, so attention overlaps dispatch, experts and combine.
* Redundant copies of the hottest experts, using the room EP24 leaves.
* Keep prefill off the decode MoE server, or give decode priority per layer. A 16k-token prompt is ~7
  TFLOP of routed-expert work per layer (estimated), enough to stall decode for tens of ms. Single-chip
  attention also makes prefill attention ~4x slower than TP4, so prefill wants its own TP attention.

## 7. Risks and open questions

* \(h\) is unknown, and 120 hops per token make it the most sensitive parameter.
* A persistent MoE server serving 40 layers takes requests of variable size, (layer, expert, tokens),
  without traces. That is a new op class: weight addresses are computed from the layer index, with
  dynamic per-expert token counts. It is the largest engineering item in the plan.
* Tail latency: combine waits for the slowest of up to 6 chips, and with many replicas a popular expert
  queues.
* bf16 ranking: today's router subtracts `bias.max()` so the bf16 ranking matches fp32
  (`PORTING.md`). Any new router has to keep that.
* The section 3 and 4 numbers are estimates built on a 20-layer, 32k-context profile. Phase 0 replaces
  them.
