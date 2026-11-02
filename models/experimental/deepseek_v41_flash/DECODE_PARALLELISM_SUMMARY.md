# Decode parallelism for V4.1: summary of the Nov 2 review

Status: summary. The detailed review is `MOE_SERVER_PLAN.md`. This document says what was proposed, what
the numbers show, and what to do next. Numbers are *measured* (with the run), *exact* (from the config) or
*estimated* (with how). Hop latency between chips is still unmeasured; it appears as \(h\).

## 1. The bottom line

* **Per-user decode speed (tok/s/user) is set by attention and per-layer overhead, not by how the experts
  are spread.** At one token, expert parallelism has no bandwidth advantage over today's TP4, because a
  token reads only 6 experts.
* **Keep attention at TP4.** Measured, it is 2x faster than one-chip attention.
* **Expert parallelism over a dedicated set of MoE chips is a throughput design.** It only pays off at
  ≥ ~128-256 concurrent users, and only with enough MoE chips. With 12 it is slower than today between
  ~64 and ~200 users.
* **The biggest available per-user win is attention itself:** 9.5 ms of a 27.9 ms token today, and ~2.5-3
  ms looks reachable (section 6). That is larger than any MoE layout change considered.

## 2. The proposal that was reviewed

Problem statements as given:

1. MoE scales with more chips and is DRAM-bandwidth bound.
2. Attention doesn't scale well.
3. "DeepInfer" uses expert parallelism with attention on a single chip.
4. Today's TP4 has too few chips for MoE and too many for attention.

Plan as given: 12 (or 24) chips run a persistent MoE server for decode and prefill, 32 (or 16) experts per
chip per layer. The other chips run attention, one chip per user for all 40 layers, and send
`(layer, hidden)` over D2D sockets to a router chip, which forwards each token to its experts. CSA bounds
SDPA at 128 + 512 = 640 keys, and long context gets its own indexer / compressor chips.

## 3. Facts that drove the analysis

| fact | value | kind |
|---|---|---|
| hardware | 32 Blackhole chips (Galaxy), 32 GB and ~512 GB/s each | `lspci` |
| model | 40 layers, D = 5120, 384 routed experts top-6 (I = 2304), 1 shared expert | config |
| one routed expert, bfp4 | 19.9 MB; all 40 layers: 306 GB | exact |
| today's layout | 8 pipeline stages of 1x4 (TP4), 5 layers each; every chip holds I/4 of all experts of its layers | code |
| today's token | 27.9 ms at one row: ~21-23 ms kernels, ~6 ms hops, trace boundaries and host | measured Oct 29 |
| today's layer | ~530 µs: `fused_experts` ~170, ~150 µs of replicated non-attention work (mHC, norms, step context), router ~16, attention the rest | measured Oct 27 |
| attention module per layer, 32k context | **238 µs at TP4, 468 µs on one chip** | **measured Nov 2** |

## 4. The assumptions, corrected

| assumption | finding |
|---|---|
| MoE scales with chips, bandwidth bound | It scales for **throughput** (many tokens per expert read), not for one token's latency. A token's 6 experts sit on at most 6 chips; with 12 MoE chips the busiest one holds 1.9 of them on average (37 MB), about the same as TP4's 30 MB per chip. `fused_experts` runs at ~170 µs against a ~60-75 µs floor, so it is partly overhead-bound too. |
| attention doesn't scale | **Wrong for the attention module**: TP4 is 2.0x faster than one chip (measured). The head-sharded projections, SDPA and the output RoPE scale 1.9-3.9x. What doesn't scale is the replicated `q_a` / `kv`, the index and compressor work, and the non-attention per-layer work. |
| DeepInfer: EP + single-chip attention | "DeepInfer" not identified; compared against DeepSeek's published system (from memory, to verify). There, attention and experts **share** every GPU, batches are large, and per-user speed is ~20-22 tok/s. It is a throughput design, not evidence for faster per-user decode. V4.1's CSA makes KV reads small, so the cost is weights, which every attention replica reads again. |
| TP4: too few MoE chips, too many attention chips | Not right. The experts already span all 32 chips (~9.5 GB each). At one row the problem is that 28 of 32 chips are idle while one stage works. |
| router chip | Adds a serial hop per layer and a single hotspot. The gate is 2 MB; route on the attention chips. |
| 12 MoE chips hold all experts | 25.5 GB per chip of a 32 GB chip: no room for redundant hot experts, bf8 weights or prefill buffers. 24 chips: 12.7 GB, comfortable. |
| CSA bounds SDPA at 640 keys | Correct. |
| long context needs indexer / compressor chips | Right for the indexer, not the compressor (constant cost per step). Only layers 2, 8, 14 (N/2 keys) and 20 (N keys) grow with context; 24-36 are capped at 16384 candidates. KV + index keys are ~4.2 KB per position per user (bf16): ~540 MB at 128k, ~4.3 GB at 1M. |

## 5. The layouts, estimated

Per-user latency at one row and 32k context. The plan's totals assume persistent kernels remove most of
today's ~6 ms of gaps; today's layout would gain from the same change (~22 ms).

| layout | per layer | per token | vs today (27.9 ms) |
|---|---|---|---|
| today: 8 stages x TP4 | ~530-575 µs | 27.9 ms (measured) | |
| original plan: one-chip attention + EP12 MoE + router chip | ~735-905 µs + 3h | ~30-39 ms | **slower** |
| TP4 attention (5 groups) + EP12 MoE, routing on the attention group, shared expert overlapped | ~505-525 µs + 2h with a new full-grid expert kernel; ~580 µs + 2h with today's | ~25.5-27.5 ms | 0-9% faster, all of it from the new kernel |
| today's layout + the same expert kernel (WS3: `fused_experts` → ~110 µs) | ~515 µs | ~25.5 ms | as good, no extra hops |

Throughput, bandwidth-only model (tok/s; no load imbalance, perfect overlap):

| users | today | original plan, EP24 | TP4 attention + EP12 |
|---|---|---|---|
| 8 | ~290 | ~240 | ~300 |
| 64 | ~2290 | ~1940 | ~1580 |
| 128 | ~2760 | ~3880 | ~2320 |
| 256 | ~3200 | ~7760 | ~4090 |

The real advantage of a global MoE pass is that each expert is read once for all users, where the
8-stage pipeline reads it once per row. It needs MoE bandwidth to show: 12 MoE chips give the experts 37.5%
of today's. If throughput becomes the goal, the better layout is TP4 attention with experts spread over
**all 32 chips**, sharing chips with attention (as DeepSeek does): same per-user latency, no 64-200 user
regression, ~9.6 GB of experts per chip.

## 6. Attention profile and how to make it faster

Measured Nov 2 with the new `test_attention_decode_profile` / `_tp4` tests (section 8):

| layer type (how many of 40) | one chip | TP4 |
|---|---|---|
| sliding only (2) | 413 µs | 190 µs |
| compressed consumer (30) | 426 µs | 195 µs |
| ratio-2 kv + index source (3) | 683 µs | 453 µs |
| candidate source, layer 20 (1) | 713 µs | 486 µs |
| index-only source (4) | 592 µs | 363 µs |
| **40-layer token** | **18.7 ms** | **9.5 ms** |

Context barely matters: 4k gives 18.4 / 9.1 ms and 128k 19.1 / 9.7 ms.

A TP4 consumer layer, op by op: `q_a` 28.0, `kv` 15.9 (both replicated), `q_b` 19.6, q/kv RoPE + ring
write 10.5, SDPA 12.9, output RoPE 23.1 (one core), `o_a` 28.2, `o_b` 18.4, `o_b` all-reduce 15-34
(fastest / slowest chip), all-gathers 4.6. On one chip the head-sharded parts are ~4x, and two are
kernel problems: the output RoPE on one core (89 µs) and `o_a` at 186 GB/s (`q_b` and `o_b` reach ~330).

**Getting a TP4 layer below 50 µs.** Each chip reads 21.7 MB of attention weights per layer, ~54 µs at a
realistic 400 GB/s. So with weights streamed from DRAM during attention, the floor is ~70-78 µs even with
perfect kernels. Below 50 needs the weights off the critical path:

1. **Preload the next attention block into L1** during the ~150 µs of non-attention work before it, which
   barely touches DRAM (a layer reads ~62 MB per chip, ~150 µs of DRAM time in a ~575 µs layer). 21.7 MB is
   ~180 KB per core over 120 cores. The prefetcher and its rings exist, but they are 2-4 pages deep, so
   they stream with the matmul rather than ahead of it. This needs a whole-block preload mode, an L1 budget
   (tight next to Engram and the whole-grid select), and mHC's `fn` weight kept resident.
2. **Cut the fixed-cost ops**: output RoPE 23 → ~3 µs (16 cores, or in the SDPA reducer); q/kv RoPE and
   cache write fused into matmul epilogues, 10.5 → ~3; the all-reduce fused with `o_b`, → ~8; SDPA more
   cores per head, → ~8; all-gathers folded in, → ~2.
3. Optionally shard `q_a` / `kv` instead of replicating them (−3.9 MB per chip, plus one tiny all-gather).

Estimated TP4 consumer layer: ~78 µs with 2 alone, ~72 µs with 2 and 3, **~40-45 µs with 1 and 2**. The 8
index layers carry ~250 µs more (select 134, head replication 31, compressor ~56, indexer projections
~25); the indexer plan in `DECODE_LIGHTNING_INDEXER_IMPROVEMENTS.md` leaves them at ~150-200 µs. The
40-layer average then lands at ~60-70 µs, and attention at ~2.5-3 ms per token instead of 9.5.

## 7. Next steps

| # | step | why |
|---|---|---|
| 1 | Re-profile attention with the prefetcher on (`use_prefetcher=True` inside `tensor_prefetcher_session`) | the Nov 2 profile ran without it; production runs with it |
| 2 | Prototype the whole-block L1 preload on one consumer layer; measure matmul time from L1 and the L1 headroom | the lever behind < 50 µs |
| 3 | Output RoPE spread, RoPE / cache fusion, all-reduce fusion, SDPA cores per head | ~60-90 µs per layer, on any layout |
| 4 | `DECODE_LIGHTNING_INDEXER_IMPROVEMENTS.md` steps 1-5 | the index layers |
| 5 | WS3 in `DSPARK_100TPS.md`: the non-attention per-layer work and `fused_experts` → ~110 µs | per-user latency on today's layout |
| 6 | Only if aggregate throughput at ≥ 128 users becomes the target: measure D2D hop latency \(h\), a full-grid expert kernel and the real expert-union size, then build EP over all 32 chips with TP4 attention | `MOE_SERVER_PLAN.md` sections 4 and 6 |

## 8. Artifacts

* `MOE_SERVER_PLAN.md`: the full review, updated with the Nov 2 measurements.
* `tests/decode/test_attention.py`: `test_attention_decode_profile` (one chip) and
  `test_attention_decode_profile_tp4` (1x4 submesh). One layer of each decode variant at 4k / 32k / 128k,
  caches filled with random values as if decoded, 4 warm-up and 4 profiled positions, per-layer
  `ATTN_L<layer>_START` / `_END` signposts. Run under Tracy:

  ```
  DEEPSEEK_V4_TRACED_DECODE=0 python -m tracy -p -r -v -m pytest 'models/experimental/deepseek_v41_flash/tests/decode/test_attention.py::test_attention_decode_profile'
  DEEPSEEK_V4_TRACED_DECODE=0 python -m tracy -p -r -v -m pytest 'models/experimental/deepseek_v41_flash/tests/decode/test_attention.py::test_attention_decode_profile_tp4'
  ```

* Reports: `generated/profiler/reports/2026_11_02_06_33_21` (one chip) and `2026_11_02_06_35_35` (TP4).
  The TP4 CSV holds each op once per chip, with different call IDs, so it has to be read per device; the
  numbers above take the slowest chip for each op.

Caveats on all of the above: the profile is eager (gaps between ops not counted), random index keys make
the select a slight best case, the non-attention per-layer work comes from the Oct 27 profile, and
everything about EP, hops and the L1 preload is estimated until steps 1, 2 and 6 run.
