# Where Gemma4 prefill time actually goes, per chunk size

**Question this answers.** Chunk 2048 gives a 1.85x better TTFT than today's 8192 but is
**2.09x slower over a 256k prompt** — "hard pass" as a deployed setting. Why? And: an earlier
session's claim that the global-attention op is *fabric-bound* (from "it runs at 47% of the
same-layer matmul's per-core rate") was challenged as an unverified inference. This re-derives
the whole picture from measurement.

Measured on a BH Galaxy, mesh 8x4 (CP8 / TP4), branch `kmabee/gemma4-swa-multihop-halo`
@ `23df2b38ba4`. Gemma4-31B: 60 layers = 50 sliding + 10 full attention, hidden 5376,
32 q heads / 16 kv heads, head_dim 256, window 1024.

> **Forwarding this to someone?** Send [`README.md`](README.md) instead — it is the same answer
> written to be read cold in two minutes, with the results and the methodology up front. This
> file is the full record behind it.

---

## 0. TL;DR — the results

| | |
|---|---|
| **The question** | why is chunk 2048 **2.09x** slower than 8192 over a 256k prompt? |
| **The answer** | **two independent ~2x effects, in different layers.** (1) A fixed **~94 ms per-chunk cost** paid 4x more often — **2.16x**. (2) The prefix attention leaves **71% of the core grid idle** at chunk 2048 — **1.99x**. |
| **The floor is** | matmul weight reads ~30 ms (127 MB/layer/device, read once per chunk) + `rms_norm` ~18 ms (row-parallel only ⇒ 8 of 120 cores) + sliding halo ~10 ms + ~12 ms other. **84% of it is the 50 sliding layers, by count.** |
| **The prefix term is** | **100% the 10 full-attention layers' `RingJointSDPA`.** Sliding layers are exactly flat (slope −0.0001 ms/index over 256k). |
| **That op is bound by** | **MAC throughput on QK^T and PV.** Not bytes (−47% K/V bytes ⇒ 0.998x), not softmax (⇒ 1.007x), not fabric. Fidelity LoFi ⇒ 0.786x, HiFi4 ⇒ 1.755x. |
| **Most actionable fact** | **chunk 4096's prefix term is within 2% of 8192's** — its whole 1.25x penalty is the floor, so 4096 buys a 1.39x better TTFT for 1.25x throughput. 2048 is not viable at long context. |
| **Corrections to prior work** | the "~53% of the op is fabric movement" claim does **not** hold (§4). Two of this investigation's own conclusions were also retracted (§3.3, §5.1). |

**How, in one line:** the backbone is whole-model per-chunk device timings fitted to
`t_i = a + slope·i` (R² ≥ 0.9996), plus **layer-count differencing** for per-layer vs non-layer
cost, **per-layer-type depth curves** to separate global from sliding, **Tracy** for per-op
composition only, and **causal ablations** for every mechanism claim. Tracy could not answer
"why" — its utilization columns are empty on this path (§8). Full method table in
[`README.md`](README.md#how-we-measured-it).

---

## 1. Headline

The 2.09x is **two independent ~2x effects that live in different layers and need different
fixes**. Neither is the sliding-window halo, and neither needs a fabric explanation.

| | 2048 vs 8192 | cause | whose |
|---|---|---|---|
| **per-chunk term** `N·a(C)` | **2.16x** | a **~94 ms cost** that doesn't shrink with the chunk, paid 4x more often (128 chunks × 131 ms; chunk is 4x smaller but `a` only falls 1.85x) | **~84%** the **50 sliding** layers, by count |
| **prefix term** `slope·N(N−1)/2` | **1.99x** | the attention op leaves **71% of the core grid idle** at chunk 2048 (32 work units on ~110 cores) and pays 4x as many steps | **100%** the **10 full-attention** layers' `RingJointSDPA` |
| | **= 2.09x** | | |

Both are structural. What each one is:

* the per-chunk floor is **~26 ms of `rms_norm` that barely shrinks with the chunk (width-bound)** plus
  **~27 ms of non-scaling matmul weight-DRAM reads (≥159 GB/s achieved, ~a third of peak)**;
* the prefix slope is set by **SDPA work-unit granularity**: at chunk 2048 the op puts only 32
  work units on a ~110-core grid (**29% occupancy**) and the idle cores still march through the
  loop. `depth(C)·q/C` predicts every measured chunk size with zero fitted parameters.
  `q_chunk_size` is hardcoded to 64 and is optimal only at chunk 4096 — but retuning it is worth
  only 8–16%, not the 34–50% a pure-occupancy reading suggests (§3.3).

Two things this does **not** establish. It is **not the fabric and not the sliding-window halo**
(the halo costs ~4–5 ms at chunk 4096). But it also does **not** yield a movement-vs-math split
for the global SDPA — an earlier attempt to extract one is retracted in §3.3, and that question
is still open.

---

## 2. The decomposition, from measured data only

Per-chunk device times are linear in chunk index, so one ctx-32k run per chunk size gives both
coefficients, and `T(ISL,C) = N·a + slope·N(N−1)/2` (validated against independent 256k runs to
0.9%). `scripts/why_2x.py`:

| chunk | hops | N | a (=TTFT) | slope | per-chunk term | prefix term | TOTAL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 4 | 128 | 131.3 ms | 1.458 | 16.81 s | 11.85 s | 28.66 s |
| 4096 | 2 | 64 | 174.2 ms | 2.995 | 11.15 s | 6.04 s | 17.19 s |
| 8192 | 1 | 32 | 242.7 ms | 11.990 | 7.77 s | 5.95 s | 13.71 s |
| 16384 | 1 | 16 | 436.2 ms | 40.33 | 6.98 s | 4.84 s | 11.82 s * |
| 32768 | 1 | 8 | 917.9 ms | 140.69 | 7.34 s | 3.94 s | 11.28 s * |

\* Re-measured 2026-09-18; the originally published slopes (37.100 / 126.2) were low by ~9-10%. Bad original fits, not a regression -- two builds agree on the new values. See the README footnote and the addendum below.

Ratios to chunk 8192:

| chunk | per-chunk | prefix | total |
|---:|---:|---:|---:|
| 2048 | 2.16x | 1.99x | 2.09x |
| 4096 | 1.44x | **1.02x** | 1.25x |
| 16384 | 0.91x | 0.75x | 0.84x |
| 32768 | 0.96x | 0.59x | 0.80x |

**Read this line first: chunk 4096's prefix term is already within 2% of chunk 8192's.** All of
4096's 1.25x long-context penalty is the per-chunk floor. The prefix penalty appears only
*below* 4096.

---

## 3. Term 2 (prefix slope) is entirely the global layers, and it is occupancy

### 3.1 It is the global layers — measured directly, 256 points

Depth curves for one global and one sliding layer, every chunk index of a 256k prompt at chunk
2048 (`curve_both_c2048`, 256 measured replays):

| layer type | a (index 0) | slope, ms/index | R² | index 0 → 127 |
|---|---:|---:|---:|---|
| **global** | 2.79 ms | **0.1455** | 1.0000 | 2.77 → 21.24 ms (7.68x) |
| **sliding (SWA)** | 2.54 ms | **−0.0001** | 0.013 | 2.51 → **2.51 ms (1.00x)** |

**A sliding layer is exactly flat across a full 256k prefill** — its slope is indistinguishable
from zero, as it must be: the halo is constant-size by construction. Reconstructing the
whole-model slope from the two gives `10×0.1455 + 50×(−0.0001) = 1.449 ms/index` against the
**measured 1.458** — agreement **0.6%**.

The same conclusion from a completely different measurement: a per-op tracy capture of one
isolated global layer at chunk 8192 gives `RingJointSDPA` growing 1.301 ms (index 0) →
37.222 ms (index 31) = 1.159 ms/index; ×10 = 11.59 against the measured 11.990 — **−3.4%**.

So the prefix term is **100% the 10 global layers**, two ways, at two chunk sizes.

### 3.2 It is work-unit quantization, not bandwidth

From `ring_joint_sdpa_program_factory.cpp`:

```cpp
const uint32_t all_heads_num_q_chunks = B * NH * num_q_chunks;
const uint32_t max_q_per_core = tt::div_up(all_heads_num_q_chunks, num_cores);
```

`num_q_chunks = ceil((chunk/CP) / q_chunk_size)`, `NH` = 8 local q heads (32 / TP4), and the SDPA
grid is `(compute_grid.x − 1) × compute_grid.y` ≈ **110 cores** (`AVAILABLE WORKER CORE COUNT`
is 120; `RingJointSDPA` reports 113–114 including its fused CCL workers). With `q_chunk_size=64`
the unit count is just `chunk/64`.

Critically, **cores with no work are not skipped.** The factory says so:

> "A core with no Q chunks (`global_q_start == global_q_end`) is **NOT dead**: … it runs padded
> handshake iterations (`loop_q_count = *_max_q_per_core`)"

So runtime = `max_q_per_core` iterations of a handshake-synchronised loop for *every* core. That
makes the cost `depth × q_chunk × C` per chunk-index, and the whole-prompt prefix cost
`∝ depth(C)/C`. **Zero fitted parameters** — `depth` is integer division:

| chunk | units | depth | occupancy | predicted prefix vs 8192 | measured |
|---:|---:|---:|---:|---:|---:|
| 2048 | 32 | 1 | **29%** | 2.00x | 1.99x |
| 4096 | 64 | 1 | 58% | 1.00x | 1.02x |
| 8192 | 128 | 2 | 58% | 1.00x | 1.00x |
| 16384 | 256 | 3 | 78% | 0.75x | 0.75x |
| 32768 | 512 | 5 | 93% | 0.62x | 0.59x |

and the adjacent slope ratios, which no power law fits (2.05x then 4.00x then 3.09x then 3.40x),
come out at 2.00 / 4.00 / 3.00 / 3.33 — within 3%. `scripts/occupancy_model.py`.

This is why chunk 2048 loses at long context: it wastes 71% of the grid *and* pays 4x as many
chunk-index steps.

Note the model is `depth(C,q) × C × (q·m′ + r′)`; with `q` fixed at 64 the bracket is a constant,
which is why the cross-chunk-size predictions above need no knowledge of `m′` or `r′` at all.

### 3.3 The q_chunk ablations — and a retraction

Two ablations, 128- and 32-point depth curves on one isolated global layer, same session and
binary:

| chunk | q_chunk | units | depth | rows/core | units/core | slope, ms/index/layer | vs q=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 64 (today) | 32 | 1 | 64 | 1 | 0.1460 | — |
| 2048 | **32** | 64 | 1 | 32 | 1 | **0.1233** | **0.845x** |
| 8192 | 64 (today) | 128 | 2 | 128 | 2 | 1.2272 | — |
| 8192 | **128** | 64 | 1 | 128 | 1 | **1.1239** | **0.916x** |

Pure occupancy predicts 0.50x and 0.66x respectively. Both came in far milder. **So the `div_up`
depth penalty is real but costs much less than a full extra unit of work — a core holding 2 units
overlaps most of the second one.**

> **RETRACTED.** An earlier version of this document read the chunk-2048 ablation through
> `cost = depth × (q·m′ + r′)` and concluded "69% of the cost is per-work-unit prefix K/V
> re-streaming, 31% is math". **That does not survive the second ablation.** Chunk 2048 demands
> `r′ = 142·m′` and chunk 8192 demands `r′ = 13·m′` — an order of magnitude apart, so no single
> two-term model fits both, and the split is unfounded.

The chunk-2048 ablation changes rows/core **and** `q_chunk` together, so it cannot separate
"per-unit overhead" from "per-row efficiency depends on `q_chunk`" — at `q_chunk=32`
(`Sq_chunk_t==1`) the QK/softmax/PV work per row is simply less efficient. The **chunk-8192
ablation is the better design**: it holds rows/core fixed at 128 and changes only units/core
(2 → 1), isolating the per-unit overhead. It gives

> **per-unit overhead ≈ 17% of the op's cost at `q_chunk=64`; per-row work ≈ 83%.**

*A confound checked and ruled out:* `q_chunk=32` makes `Sq_chunk_t==1`, which gates the factory's
in-place latent-V path — but `kt_inplace_v_enabled(v_shares_k_buffer, Sq_chunk_t)` also requires
`v_shares_k_buffer = has_latent_v() = !input_v.has_value()`. Gemma4 passes `cache_v` explicitly
at `ring_prefill.py:406`, so it is always false and that path never engages.

### 3.4 The KV-dtype ablation: the op is NOT bandwidth-bound (2026-09-17)

The decisive causal test, needing no FLOP anchor, no spec-sheet bandwidth and no utilization
column: change the **bytes** the global layers' K/V occupy, holding shapes and work-unit geometry
identical. `GEMMA4_KV_DTYPE`, applied to the **global** cache only (the sliding path hard-requires
BFP8_B at `ring_joint_sdpa_device_operation.cpp:589`, but that block is gated on
`if (args.has_sliding_window())` at line 553 — and the prefix term is 100% global layers anyway).
Whole-model `ctx_32k`, chunk 8192:

| global KV dtype | bytes/tile | slope | T(256k) | vs bfp8 |
|---|---:|---:|---:|---:|
| bfp8_b (today) | 1088 B | 11.83 | 13.64 s | — |
| **bfp4_b** | **576 B (−47%)** | **11.81** | 13.61 s | **0.998x** |
| bf16 | 2176 B | — | — | L1 overflow: CBs grow to 1 635 456 B vs 1 572 864 B max |

**47% fewer K/V bytes → 0.2% change in the prefix slope.** The bf16 failure is the control that
makes this trustworthy: it proves the dtype propagates into the op's own L1 circular buffers, so
bfp4 really did halve the bytes through DRAM, the fabric gather, L1 *and* the per-work-unit
streaming.

> **The global `RingJointSDPA` is not bandwidth-bound.** Its prefix-proportional cost scales with
> element **count**, not byte count.

Three claims die here:
1. the original **"53% is fabric movement"** — already undermined by §3.2, now directly refuted;
2. the retracted **"69% per-unit K/V re-streaming"** of §3.3 — independently refuted, since a
   movement term would have shrunk with the bytes;
3. **"cut the redundant prefix K/V streaming"** as a lever — if bytes do not matter,
   deduplicating the streaming will not help either.

What survives as the candidate: **per-element work in the compute path** — unpack rate, MAC
issue, softmax — or fixed per-tile overheads. That is where kernel-level `DeviceZone`
instrumentation should look.

### 3.5 What the global SDPA actually spends its time on — ANSWERED (2026-09-17)

Four cheap causal ablations, each varying one cost term while holding the rest fixed, all on the
whole model (`ctx_32k`, the slope is 100% global-layer SDPA). No FLOP anchor, no spec-sheet
bandwidth, no utilization column, **and no kernel instrumentation needed**.

| ablation | what it varies | slope | vs base | verdict |
|---|---|---:|---:|---|
| base (HiFi2, `exp_approx=False`, k=256) | — | 11.93 | — | ships today |
| `GEMMA4_KV_DTYPE=bfp4` | K/V **bytes** −47% | 11.81 | **0.998x** | **bytes: ruled out** |
| `GEMMA4_SDPA_EXP_APPROX=1` | softmax **exp** path | 12.01 | **1.007x** | **softmax: ruled out** |
| `GEMMA4_SDPA_FIDELITY=lofi` | **MAC passes** 2→1 | **9.38** | **0.786x** | **dominant** |
| `GEMMA4_SDPA_FIDELITY=hifi4` | MAC passes 2→4 | **20.94** | **1.755x** | **dominant** |
| LoFi + exp_approx | both | 9.35 | 0.784x | = LoFi alone, so exp really is ~0 |
| `GEMMA4_GLOBAL_K_CHUNK=128` | inner-loop blocking | 13.79 | 1.156x | 256 is better |
| `GEMMA4_GLOBAL_K_CHUNK=512` | inner-loop blocking | — | — | L1 overflow (1 733 760 B vs 1 572 864 B) |

> **The global `RingJointSDPA`'s prefix cost is MAC-throughput-bound on QK^T and PV.**

**Generalises across occupancy regimes.** LoFi gives **0.786x at chunk 8192** (128 units, depth 2,
58% occupancy) and **0.764x at chunk 2048** (32 units, depth 1, 29% occupancy) — same mechanism,
slightly stronger at the small chunk.

**No precise MAC percentage is claimed.** Fitting `time = passes·k + c` on (LoFi, HiFi2) gives a
43% MAC share; on (HiFi2, HiFi4) it gives 76%. The two disagree, so the relationship is not linear
in fidelity passes and quoting a single share would repeat the two-point-model error of §3.3.
What is certain: MAC passes are the dominant term and the only knob with a large effect.

**`k_chunk=256` is the practical optimum** — bounded below by per-k-block overhead (128 is 16%
worse) and above by L1 (512 does not fit). Nothing to gain here.

**Why this closes the question without `DeviceZone` zones.** The kernel work was queued to find
what starves the math. The answer is that nothing does: the op *is* the math. Zones would only
split QK^T from PV, and both are MACs, so the result would not change any decision. That is a
genuine outcome, not an omission — if fidelity had behaved inconsistently across chunk sizes, or
if `k_chunk` had mattered, zones would be the next step.

**The consequence is a policy question, not a perf bug.** The biggest single lever on long-context
throughput is the SDPA math fidelity Gemma4 ships (`HiFi2`, set in
`models/demos/gemma4_d_p/tt/attention/__init__.py`): LoFi is **−21% on the prefix slope** and
**−9.9% on a 256k prefill**. That is an accuracy decision — see §7.2d for the measured cost.

**What remains open.** Occupancy explains the chunk-size *scaling* (§3.2) but gives no
movement-vs-math split, and nothing else measured here does either. So **what actually consumes
the global SDPA's time is not settled** — which is exactly the original objection's point: a
matmul-rate anchor cannot tell you, and the profiler's utilization columns are empty (§4).

---

## 4. The disputed "47% ⇒ fabric-bound" claim

**Verdict: the methodological objection was right; the substantive conclusion survives a proper
test, but with a different mechanism.**

The original argument was: the op's FLOPs at the same-layer matmul's per-core rate would take
17.57 ms; it takes 37.22 ms; therefore 53% is "fabric movement + softmax".

Three things are wrong with that:

1. **Anchoring on a matmul rate and attributing the residual to fabric assumes nothing else can
   starve the math.** At chunk 8192 the op runs 128 work units on 110 cores at depth 2 —
   occupancy 128/(2×110) = **58%** — so ~42% of core-time is *structurally* idle, burning the
   padded handshake iterations of §3.2, before any byte moves. Correcting for that alone moves
   the "ideal" from 17.57 ms to ~30.1 ms, leaving ~19%, not 53%, unexplained.
2. **The two "independent" estimates were not independent** — the matmul-rate anchor and the
   cost-scaling fit were both calibrated on the same chunk-8192/16384 pair.
3. **The profiler cannot settle it.** `NOC UTIL (%)`, `DRAM BW UTIL (%)`, `ETH BW UTIL (%)`,
   `DEVICE COMPUTE CB WAIT FRONT [ns]` and the per-core min/max duration columns **exist in the
   CSV but are entirely empty** on this path (`PM IDEAL [ns]` median is 1 ns, a stub). They need
   `--analyze-noc-traces` *and* a built tt-npe, and even then ETH BW UTIL is *modelled* from NoC
   event traces, not measured. Any claim that process_ops_logs.py "reports ETH BW UTIL" for this
   workload is wrong.

**What replaces it — partially.** The occupancy model (§3.2) explains the chunk-size *scaling*
with no fabric term and no fitted parameters, which is enough to say the original mechanism was
wrong. It is **not** enough to give a movement/math split: two `q_chunk` ablations demand
per-unit-overhead values an order of magnitude apart (§3.3), so any such split from this data is
unfounded, and an earlier version of this document claiming "69% movement" is retracted. The
cleanest thing measurable here is that **per-unit overhead is ~17%** of the op at `q_chunk=64`.

So the honest position is the objection's own: **nobody has yet measured what starves the math
in this op.** Doing so needs either tt-npe NoC traces or a targeted ablation that holds the
work-unit geometry fixed while varying only bytes moved (e.g. KV cache dtype).

Consistent with that, the **multi-hop halo costs ~4–5 ms at chunk 4096** (measured by holding the
chunk fixed and halving the window so the hop count drops, with chunk 8192 as a 1-hop control),
and fabric links, a fabric mux and a line-multicast halo were all measured to be worth ~nothing.

---

## 5. Term 1 (the per-chunk floor) attributed to ops

Per-op device kernel time for one sliding and one global layer at **chunk index 0** (no prefix,
so this is purely the floor + intra-chunk work). Three profiled captures, **all on this branch**
(`floor_c2048` / `floor_c4096` / `floor_c8192`). `scripts/floor_ops.py`.

Sliding layer, µs:

| op | 2048 | 4096 | 8192 | ratio, 4x tokens |
|---|---:|---:|---:|---:|
| **Matmul ×5** | 843.9 | 1137.1 | 1242.7 | **1.47** |
| **LayerNorm ×7** | 464.9 | 475.5 | 601.5 | **1.29** |
| AllGather ×3 | 166.9 | 327.2 | 561.7 | 3.36 |
| ReduceScatter ×2 | 140.6 | 245.1 | 465.6 | 3.31 |
| RingJointSDPA | 265.6 | 358.4 | 391.4 | 1.47 |
| BinaryNg ×5 | 92.3 | 178.0 | 340.2 | 3.69 |
| *(rest)* | 306.9 | 429.4 | 611.5 | |
| **TOTAL** | **2281.1** | **3150.7** | **4215.1** | **1.85** |

A sliding layer is **2.16x less efficient per token** at chunk 2048. Reconstructing a 60-layer
chunk (10 global + 50 sliding) gives 140.6 ms at 2048 and 268 ms at 8192, against whole-model `a`
of 131.3 / 242.7 ms — the isolated-layer benchmark inflates by ~1.07x / ~1.10x, so use these for
**attribution and ratios**, not absolute totals.

The two ops that barely respond to chunk size (Matmul 1.47x and LayerNorm 1.29x for 4x the
tokens) are the floor; the ones near 3.3–3.7x scale properly and are not the problem.

### 5.0 The layer-count difference: the cleanest handle on the floor

**Do this before any profiling.** Running the whole model at several layer counts and
differencing gives the per-layer cost and the non-layer cost with no profiler, no isolated-layer
inflation and nothing missing from the graph. `a(C,N) = F(C) + N·L(C)`; the HF `layer_types`
pattern is a regular `[sliding×5, full]` repeat, so truncating to a multiple of 6 preserves the
5:1 ratio exactly. `scripts/parse_nlayers.py`.

| chunk | N=60 | N=12 | N=6 | F (non-layer) | L (per layer) |
|---:|---:|---:|---:|---:|---:|
| 8192 | 242.7 ms | 49.3 ms | 25.1 ms | **0.9 ms** | **4.029 ms** |
| 2048 | 131.3 ms | 26.6 ms | 13.5 ms | **0.5 ms** | **2.181 ms** |

Linearity in N is **tested, not assumed**: solving on (6,12) and on (12,60) gives F within 5–6%,
and `60·L + F` reproduces the measured `a(C)` exactly.

**Two conclusions:**

1. **Embedding, final norm, LM head and inter-layer CCL together are 0.4% of TTFT** (0.9 ms of
   242.7). An earlier draft of this document flagged "~10–20% of TTFT attributed to nothing
   verifiable" as a major open gap — **that is now closed and the hypothesis was wrong.** The
   whole 7–11% discrepancy was the isolated-layer benchmark's own per-replay overhead, measured
   here as **1.07x at chunk 2048 and 1.11x at 8192**. Scale isolated-layer absolutes by that;
   ratios are unaffected.
2. **The floor is 70.4 ms**, from whole-model numbers alone: per-layer excess over perfect token
   scaling `2.181 − 4.029/4 = 1.174 ms`, ×60 = **70.4 ms**, against the measured whole-model
   excess `131.3 − 242.7/4 = 70.6 ms` — **0.3%**. Third independent route to that number.

Applying the per-op shares of §5.3 to that 70.4 ms: matmul **30.3 ms**, `rms_norm` **18.3 ms**,
sliding `RingJointSDPA` 9.9 ms, heads/rope/tilize 6.3 ms, **TP collectives 2.8 ms**.

The same runs give the prefix slope per global layer a third way — 11.880/10, 2.430/2 and
1.230/1 = **1.188 / 1.215 / 1.230 ms/index/layer** across N=60/12/6, confirming the prefix term
is the global layers and showing the isolated-layer figure (1.2272) was 3.3% high.

### 5.0b How to quantify "fixed" per op — and how NOT to

Three-point depth curves (`curve_both_c{2048,4096,8192}`) give `a` per layer type at three chunk
sizes, which lets the affine form be **tested** rather than assumed. It fails:

| layer | intercept from (2048,4096) | from (4096,8192) | spread |
|---|---:|---:|---:|
| sliding | 1.71 ms | 2.26 ms | 12% of a(8192) |
| global | 1.69 ms | 1.91 ms | 4% |

`a(C)` is **concave**, not affine. The same test per op, on three same-branch profiled captures,
agrees: solving the intercept on (2048,4096) vs (4096,8192) disagrees by **39% for Matmul, 39%
for RingJointSDPA and 17% for LayerNorm** — and Matmul's (4096,8192) intercept, 1031 µs, even
*exceeds* its total measured cost at chunk 2048 (844 µs), which is impossible for a real fixed
component. The whole-layer intercept lands anywhere in **1.41–2.09 ms** depending on the pair.
So **solving a "fixed cost" by extrapolating to C = 0 is not justified** — the same trap that
produced two wrong conclusions in earlier sessions. Everything
below therefore uses an in-range measure with no extrapolation: the **excess over perfect token
scaling**, `t(2048) − t(8192)/4`. (Where an op is measured to be ~100% chunk-invariant, as
`rms_norm` is at ratio 1.29 for 4x the tokens, i.e. 0.32x per token, its *whole* cost is close
to the saving available from fixing it — a different and equally valid question.)

The decomposition closes on that basis:

| | a(2048) | a(8192)/4 | excess | × layers | total |
|---|---:|---:|---:|---:|---:|
| global | 2.79 ms | 1.47 ms | 1.32 ms | 10 | 13.2 ms |
| **sliding** | 2.54 ms | 1.12 ms | 1.42 ms | **50** | **71.0 ms** |

De-inflated by the measured 1.18x isolated-layer factor gives **71.4 ms**, against the
whole-model measured excess `a(2048) − a(8192)/4 = 131.3 − 60.7 = 70.6 ms` — **+1.1%**, with
**84% of it in the 50 sliding layers**. That reproduces the independently fitted ~69 ms floor
from a different measurement path entirely.

### 5.1 `rms_norm` is width-bound, so it barely shrinks with the chunk — ~26 ms/chunk

`ttnn.rms_norm` parallelises over **rows only**. The per-device slab is `chunk/CP` rows, so the
four hidden-width `(rows × 5376)` norms get one core per 32 rows, each reducing a 168-tile-wide
row-block at HiFi4 + `fp32_dest_acc_en`:

| capture | rows | row-tiles | CORE COUNT | µs per call (4 calls, ±2%) |
|---|---:|---:|---:|---:|
| chunk 2048, this branch | 256 | 8 | **8** | **110.3** |
| chunk 8192, this branch | 1024 | 32 | **32** | **134.2** |
| chunk 8192, 2026-09-09 build | 1024 | 32 | 32 | 99.3 |

**4x the rows on 4x the cores for only 1.22x the time — 0.30x per token.** The per-head q/k/v
norms are `(rows × 256)` and cheap (6–27 µs).

| | chunk 2048 | chunk 8192 |
|---|---:|---:|
| total, 4 norms × 60 layers | **26.5 ms/chunk** | 32.2 ms/chunk |
| non-scaling part (`t − t8192/4`) | **18.4 ms/chunk** | — |

26.5 ms is **20% of chunk 2048's entire 131 ms TTFT**, spent on 8 of 120 cores.

> **Correction.** An earlier version of this document put the ratio at **0.98 — "exactly
> chunk-invariant"** — and called it the most solidly measured number here. That came from
> comparing the chunk-2048 capture against the **2026-09-09** chunk-8192 capture, where the same
> norm on the same shape and core count took 99.3 µs instead of 134.2 µs. Same-branch it is
> 1.22x, not 0.98x. The mechanism (row-parallel, 8 vs 32 cores) is unaffected and directly
> observed in all three captures; only the tightness of the invariance claim was wrong. **Never
> compare per-op timings across builds** — §8.

The fix is precision-neutral: give the norm a width-sharded program config so the 5376-wide
reduction splits across cores instead of one core per 32 rows.
`LayerNormShardedMultiCoreProgramConfig` (`compute_with_storage_grid_size`, `block_h`, `block_w`,
`subblock_w`) exists in
`ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_types.hpp`; it needs a
block-sharded input activation. (Dropping HiFi4/fp32 would also help but is an accuracy decision
— `rms_norm.py` deliberately matches the reference's FP32 prefill norm.)

### 5.2 Matmul is weight-DRAM-read bound — the largest single item

Three chunk sizes, same branch, the 5 matmuls of one sliding layer (M = rows per device):

| M | µs | vs previous M | math grew | implied achieved rate (≥) |
|---:|---:|---:|---:|---:|
| 256 | 843.9 | — | — | **150 GB/s** |
| 512 | 1137.1 | 1.35x | 2.00x | 112 GB/s |
| 1024 | 1242.7 | **1.09x** | 2.00x | 102 GB/s |

**2x the math from M=512 to M=1024 costs only 1.09x the time**, so the matmul is already nearly
saturated at the weight-read bound by M=512. The weights are 119.7 M params = **127 MB** per layer
per device (TP=4, bfloat8_b) and identical at every M, so the smallest measured time is an upper
bound on the read: achieved weight-read bandwidth is **≥ 150 GB/s**, roughly a third of plausible
BH peak. No extrapolation is involved, and it is not an irreducible physical floor.

In-range excess at chunk 2048: `843.9 − 1242.7/4 = 533 µs` per layer → **26.7 ms** over 50
sliding layers.

### 5.3 The floor, attributed per op, in range

Excess over perfect token scaling (`t(2048) − t(8192)/4`) for one sliding layer, **both captures
on this branch**, ×50 layers:

| op | @2048 | @8192 | ratio | excess/layer | × 50 | share |
|---|---:|---:|---:|---:|---:|---:|
| **Matmul ×5** | 843.9 µs | 1242.7 µs | 1.47x | 533 µs | **26.7 ms** | 43% |
| **LayerNorm ×7** | 464.9 µs | 601.5 µs | 1.29x | 315 µs | **15.7 ms** | 26% |
| RingJointSDPA (constant halo) | 265.6 µs | 391.4 µs | 1.47x | 168 µs | 8.4 ms | 14% |
| NlpCreateHeads / RotaryEmb / Tilize / ConcatHeads | | | 1.1–2.2x | 116 µs | 5.8 ms | 9% |
| AllGather ×3 + ReduceScatter ×2 (TP) | 307.5 µs | 1027.3 µs | **3.3x** | 51 µs | 2.5 ms | 4% |
| rest | | | 2.3–3.7x | 45 µs | 2.3 ms | 4% |
| **TOTAL** | **2281 µs** | **4215 µs** | **1.85x** | **1227 µs** | **61.4 ms** | |

**So the floor is matmul weight movement (~43%) plus `rms_norm` (~26%).** Two things worth
stating explicitly:

* **The TP collectives are only ~4% of it.** They scale with the chunk almost properly (3.3x for
  4x tokens). "CCL is 30% of a layer" from an earlier session is true of a layer's *total* cost
  at depth, but the collectives are **not** what makes small chunks inefficient.
* `GatherCodegenDeviceOperation` now appears in both captures (70.6 → 185.7 µs), so the
  cross-commit `Gather` rename confound noted in earlier drafts is resolved.

---

## 6. Answering the SWA-vs-global question directly

> "let's see if it's the global layers that suck here at smaller chunks, and if SWA is good,
> we're in a good place"

Half right, and the half that is right is not about the halo:

* **SWA *attention* is fine.** The multi-hop halo costs ~4–5 ms at chunk 4096 (~0.09 ms per
  sliding layer). Hop count is not a performance problem; 4 hops at chunk 2048 works and is
  near the floor for that chunk size.
* **But the sliding *layers* dominate the per-chunk floor** — not because each is expensive, but
  because there are 50 of them. 50 × 2.28 ms = 114 of the 141 ms reconstructed chunk at 2048.
  Their cost is chassis (matmul weight reads + norms + TP collectives), not attention.
* **The global layers own the entire long-context penalty**, via SDPA occupancy (§3).

So: it is both, but for unrelated reasons, and neither is the halo.

---

## 7. Levers, with projected effect

### 7.1 `q_chunk_size` per chunk size — measured, all three optima

The `q ∈ {64,128}` allowlist sits inside `if (args.has_sliding_window())`, so **global layers are
unrestricted**; `q_chunk` is hardcoded to 64 today. Global-layer depth curves:

| chunk | q_chunk | a (idx 0) | slope ms/index | vs q=64 | |
|---:|---:|---:|---:|---:|---|
| 2048 | **32** | 2.65 ms | **0.1233** | **0.845x** | ← best |
| 2048 | 64 | 2.77 ms | 0.1460 | 1.000x | |
| 4096 | **64** | 3.88 ms | **0.3011** | 1.000x | ← best (today's value) |
| 4096 | 128 | 4.18 ms | 0.5169 | **1.717x** | much worse |
| 8192 | 64 | 5.87 ms | 1.2272 | 1.000x | |
| 8192 | **128** | 5.34 ms | **1.1239** | **0.916x** | ← best |

**Measured optimum: q=32 at chunk 2048, q=64 at 4096, q=128 at 8192.** The chunk-4096 row is the
deliberate wrong-direction control — the model predicted q=128 would be *worse* there, and it is
(1.72x, against a predicted 1.31x). So the model is a **reliable direction predictor and an
unreliable magnitude predictor**: it got the sign right at all three chunk sizes and the size
wrong at two of them.

This also explains the in-tree comment "q=64 is a true optimum, worse in both directions": that
sweep was taken **at chunk 4096**, where it is correct. It does not transfer.

Whole-model effect of using the measured optimum, `q_chunk` as the only change:

| chunk | a → | slope → | T(256k) → | delta |
|---:|---:|---:|---:|---:|
| 2048 | 131.3 → 130.3 ms | 1.458 → 1.231 | 28.66 → **26.68 s** | **−6.9%** |
| 4096 | unchanged | unchanged | unchanged | 0% |
| 8192 | 242.7 → **238.1 ms** | 11.990 → **10.957** | 13.71 → **13.06 s** | **−4.8%** |

### 7.2 End-to-end verification (whole model, not single-layer deltas)

Every `q_chunk` number in §7.1 is an isolated-layer delta. Confirmed on the whole model,
`ctx_32k`, same branch:

| chunk | q_chunk | TTFT | slope | T(256k) | vs today |
|---:|---:|---:|---:|---:|---:|
| 8192 | 64 (today) | 242.9 ms | 11.87 | 13.66 s | — |
| 8192 | **128** | 241.0 ms | **10.73** | **13.03 s** | **−4.6%** |
| 2048 | 64 (today) | 131.3 ms | 1.4543 | 28.62 s | — |
| 2048 | **32** | 131.0 ms | **1.1782** | **26.34 s** | **−8.0%** |

Single-layer projected −4.8% / −6.9%; measured −4.6% / −8.0%. Slope ratios projected
0.916 / 0.845, measured 0.904 / 0.810 — **within 1–4%**, which also validates projecting
single-layer deltas to whole-model in general. **TTFT is unchanged** in both cases (−0.8%,
−0.2%): this is a pure throughput win with no latency cost.

### 7.2b Deployment table, on verified numbers

q_chunk at its measured optimum per chunk size, then with the `rms_norm` fix on top
(−20 ms from `a` at every chunk size):

| chunk | q | TTFT | T(256k) | + norm fix: TTFT | T(256k) | vs an also-fixed 8192 |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 32 | 131.0 ms | 26.34 s | **110.8 ms** | 23.76 s | 1.92x — **still a hard pass** |
| 4096 | 64 | 174.2 ms | 17.19 s | **154.0 ms** | 15.90 s | **1.28x** |
| 8192 | 128 | 241.0 ms | 13.03 s | 220.8 ms | **12.39 s** | 1.00x |

Against **today** (chunk 8192, q=64: 242.9 ms / 13.66 s):
* **8192 + q=128 + norm fix → 221 ms TTFT (1.10x better) and 12.39 s (−9%)**, with no chunk-size
  decision at all. Lowest-risk win on the table, and half of it (the q_chunk half) is verified.
* **4096 + norm fix → 154 ms TTFT (1.58x better) for 15.90 s (1.28x worse throughput)** — the
  tradeoff to argue about, if TTFT below 8k tokens is worth 28% of long-context throughput.
* **2048 stays out**: 1.92x even with both levers, because its penalty is dominated by paying the
  per-chunk floor 128 times.

### 7.2c The norm lever, re-scoped on measured data (2026-09-17)

Standalone microbenchmarks at the exact shapes (`scripts/micro_floor.py`, `scripts/micro_norm.py`,
single device, no model build) settled two things and downgraded both:

**`rms_norm` is width-bound and precision-insensitive.** Default row-parallel: 170 µs at
(256×5376) and 166 µs at (1024×5376) — **4x the rows for 0.98x the time**, confirming the
mechanism off-model. **LoFi buys only 19%** (170 → 138 µs), so HiFi4/fp32 is *not* the cost and
dropping precision is not a lever.

**Width-sharding works, but gives ~2x, not the ~8x an earlier draft assumed:**

| rows | config | ms | vs default |
|---:|---|---:|---:|
| 256 | default row-parallel | 0.170 | — |
| 256 | **6x8 = 48 cores, block 1x28t, sub 2** | **0.076** | **2.25x** |
| 256 | 12x8 = 96 cores, block 1x14t | 0.177 | 0.96x — *slower* |
| 1024 | default | 0.166 | — |
| 1024 | **8x8 = 64 cores, block 4x21t, sub 3** | **0.091** | **1.83x** |

More cores is **not** better (96 cores loses to 48), and some cells disagree between precisions,
so treat **~2x** as the expectation rather than the best cell. Note `subblock_w ≤ 3` in fp32 mode
(`dst_full_sync_en` false) — the first attempt failed on exactly that.

At ~2x the norm saves **12.6 ms/chunk**, not the 20–22 ms earlier drafts projected:

| chunk | TTFT | T(256k) | vs a fixed 8192 |
|---:|---:|---:|---:|
| 2048 | 118.4 ms | 24.73 s | 1.96x |
| 4096 | 161.6 ms | 16.38 s | **1.30x** |
| 8192 | 228.4 ms | **12.63 s** | 1.00x |

Against **today** (242.9 ms / 13.66 s): **8192 + q=128 + norm sharding → 228 ms TTFT and 12.63 s
(−8%)**. And it is not a one-line change — the norm needs a **block-sharded input**, so the
surrounding ops must produce and consume sharded activations. Moderate integration work for
~12 ms/chunk.

**Matmul is downgraded further.** A naive standalone `M=256, K=N=5376` bfp8 matmul takes **243 µs**
against the in-model **193 µs**, so the model's config is already better than a default and the
"~150 GB/s vs peak suggests easy headroom" framing was too optimistic. Real gains need genuine
matmul tuning. Rank it below the norm work.

### 7.2d The fidelity lever's ACCURACY cost — measured, and it blocks the lever

The largest single throughput lever found (§3.5) is the SDPA math fidelity. Its accuracy cost,
from the same op-level gate the multi-hop halo work used
(`tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py`, PCC/RMSE vs torch, 1/2/4-hop configs):

| fidelity | PCC | RMSE | prefix slope | T(256k) |
|---|---:|---:|---:|---:|
| **HiFi2** (ships) | 0.99963–0.99968 | **0.0057–0.0062** | 11.93 | 13.68 s |
| **LoFi** | 0.99811–0.99880 | **0.0177–0.0179** (**3x worse**) | 9.38 | **12.32 s (−9.9%)** |
| HiFi4 | 0.99967–0.99969 | 0.0051–0.0058 | 20.94 | 18.0 s (+31%) |

**Recommendation: do not take LoFi on this evidence.** All three pass the gate's threshold, but
LoFi **triples single-layer RMSE**, and that error compounds across 60 layers. A −9.9% throughput
gain is not worth an unquantified end-to-end accuracy risk. The correct gate is a real eval
(perplexity or task metrics), which nobody has run. Status: **a genuine ~10% lever, blocked on an
accuracy evaluation** — not a recommendation.

**HiFi4 is strictly dominated.** It buys nothing measurable (PCC 0.99967 vs 0.99965; RMSE 0.0055
vs 0.0058 — inside run-to-run noise) for **1.755x** the prefix cost. Gemma4's SDPA correctly uses
HiFi2; this is worth knowing for any other path that reached for HiFi4 expecting accuracy.

### 7.3 Ranked (revised 2026-09-17)

1. **`q_chunk_size` per chunk size** (§7.1) — program-config only, all three optima measured:
   **−4.8% at 256k at the deployed 8192** (and −6.9% at 2048), independent of any chunk-size
   decision.
2. **Width-shard `rms_norm`** — **~2x measured** (§7.2c) => saves ~12.6 ms/chunk, precision-
   neutral, helps every chunk size and every context. Needs a **block-sharded activation**
   plumbed through the surrounding ops, so it is moderate integration work, not a config flip.
3. ~~Find out what consumes the global SDPA's time~~ — **ANSWERED (§3.5): MAC-throughput-bound
   on QK^T and PV.** Bytes, softmax exp and `k_chunk` are all ruled out. The follow-on lever is
   fidelity, which is **blocked on an accuracy eval** (§7.2d), not on more perf work. No
   `DeviceZone` kernel instrumentation needed — zones would only split QK from PV, both MACs.
4. **Matmul DRAM efficiency at small M** — biggest single floor item (~41 ms of the ~94 ms); ≥159 GB/s achieved
   suggests headroom, not physics, but it is the hardest.
5. Do **not** spend time on fabric links, a fabric mux, a line-multicast halo, or the rendezvous
   protocol — all measured to be worth ~nothing: the halo payload is invariant at
   ~2 MiB/device/layer whatever the hop count, and the multi-hop protocol costs ~4–5 ms.

---

## 8. Method notes worth not re-paying

* **The utilization columns are empty.** See §4. Do not plan an investigation around
  `ETH BW UTIL` / `NOC UTIL` / `DRAM BW UTIL` / `CB WAIT FRONT` on this path.
* **`python -m tracy` now launches a blocking Tracy WASM web-UI server** before generating the
  ops report, and a profiled chunk-0 run writes a **4.8 GB** `profile_log_device.csv` and a
  **5.7 GB** `tracy_ops_times.csv`, then spends minutes post-processing. Budget ~15 min per
  profiled run and a timeout well above it, or the report is killed after the device work is
  already done. (`--process-logs-only` can recover a killed report from the `.logs` folder.)
* **Never compare per-op timings across builds — it cost a headline claim here.** Two things
  differed between the 2026-09-09 capture and this branch: `GatherDeviceOperation` was replaced
  by `GatherCodegenDeviceOperation` (590 → 176 µs), and, less obviously, the *same* `rms_norm` on
  the *same* shape and core count took 99.3 µs instead of 134.2 µs. The second one turned a
  measured 1.22x into an apparent 0.98x and made "exactly chunk-invariant" look like the most
  solid number in the investigation. Always re-profile the baseline on the branch under test;
  a same-shape, same-core-count op is **not** a safe cross-build anchor.
* Still true from the prior sessions: `ninja <target>` does not install (use
  `cmake --build build_Release --target install`); never pass `--device-trace-profiler`; never
  extrapolate a fitted cost model outside its fitted range.

---

## 9. Reproduce

```bash
source /data/kmabee/gemma4_runs/env.sh    # TT_METAL_HOME, HF paths, mesh-8x4 tt_cache
cd $TT_METAL_HOME
cmake --build build_Release --target install     # NOT `ninja <target>`

# per-op floor attribution at chunk index 0 (one global + one sliding layer)
timeout -k 10 3000 ./python_env/bin/python3 -m tracy -r -p -v -o <out>/profiler \
  -m pytest models/demos/gemma4_d_p/demo/text_demo_prefill.py::"\
test_prefill_layer_perf_chunk_n[blackhole-chunk0-both-sz2048-ctx_256k-8x4]" -sv

# per-layer-type depth curves -> a() and slope() for global and sliding separately
timeout -k 10 2700 ./python_env/bin/python3 -m pytest \
  models/demos/gemma4_d_p/demo/text_demo_prefill.py::"\
test_prefill_layer_perf_chunk_n[blackhole-chunkall-both-sz2048-ctx_256k-8x4]" -sv

# q_chunk ablation on global layers (diagnostic env var, not committed)
GEMMA4_GLOBAL_Q_CHUNK=32 timeout -k 10 2700 ./python_env/bin/python3 -m pytest \
  models/demos/gemma4_d_p/demo/text_demo_prefill.py::"\
test_prefill_layer_perf_chunk_n[blackhole-chunkall-global-sz2048-ctx_256k-8x4]" -sv
```

Analysis scripts are **not in-tree**; they live in `~/debug-docs/gemma4_chunk_size_anatomy-noissue/scripts/` (private repo `kmabeeTT/debug-docs`): `why_2x.py` (the two-term split), `occupancy_model.py` (the
zero-parameter prefix model), `floor_ops.py` (per-op fixed/proportional split from two or three
chunk sizes), `project_fixes.py` (lever projections), `parse_curves.py` (depth curves per layer
type).

Prior records: `gemma4_prefill_chunk_scaling-noissue/` (private repo `kmabeeTT/debug-docs`) (the 8k-vs-16k work this corrects) and
`gemma4_swa_multihop_halo-noissue/` (same private repo) plus the sibling report
[`../SlidingWindowMultiHopHalo/`](../SlidingWindowMultiHopHalo/README.md) (the multi-hop halo and the chunk-size sweep).

---

## Addendum (2026-09-17) — per-op tables, and the floor's basis

See [`PER_OP_TABLES.md`](PER_OP_TABLES.md) for `tt-perf-report` views of chunk 2048 / 4096 /
8192 side by side, captured both at chunk index 0 and at a matched prior context of 49152
tokens. Three things there supersede or sharpen this document:

1. **The floor is quoted on a new basis.** §"the floor is 70.4 ms" derives the per-layer
   *excess over perfect token scaling*, `2.181 − 4.029/4 = 1.174 ms`, ×60. For
   `cost = F + k·tokens` that excess is exactly `¾·F`, so the chunk-invariant cost is
   `70.4/0.75 = 93.9 ms`. Confirmed by an affine fit of the measured `a(C)` over 2048–8192
   (94.2 ms) and by a per-op fit (105–113 ms). All component figures in this document that
   sum to 70.4 ms are on the old basis; multiply by 1.335 for the chunk-invariant basis.
2. **"The prefix term is 100% global layers" is now a per-op measurement, not an inference.**
   99.8–100.0% of a global layer's growth between prior context 0 and 49152 tokens is
   `RingJointSDPADeviceOperation`; the sliding layer moves by +0.2% / +0.8% / −0.1%.
3. **The occupancy model passed a pre-registered discriminating test.** At matched prior
   context, chunk 2048 does 0.25x the prefix work and takes 0.443x the time, against 0.50x
   predicted from work-unit depth and 0.25x for an efficiency-neutral op.

---

## Addendum (2026-09-18) — two corrections found by validating the skill

Both were found by the `prefill-perf-debug` skill's validation pass, which re-derives this
document's numbers from the raw captures rather than from its tables. Neither touches the
2048-vs-8192 answer.

### 1. The 16384 and 32768 slopes were low by ~9–10%

| chunk | published | refit from 2026-09-09 logs (`d3064a5fd6b`, pre-halo) | measured 2026-09-18 (current build) |
|---:|---:|---:|---:|
| 16384 | 37.100 | 40.91 | **40.33** |
| 32768 | 126.2 | 139.7 | **140.69** |

The two builds agree to **1.4%** and **0.7%**; the published values are the outliers, so this
is a **bad original fit, not a regression**. Consistent with the per-op regression check at
chunk 8192, which found every op within 2% across the same two branches.

Root cause: the large-chunk slopes had been fitted from too few points. At chunk 32768 a
`ctx_32k` run has **N=1**, from which no slope can be fitted at all, and `ctx_64k` gives N=2.
The fresh runs are N=8 and N=16 at ctx 256k, R² ≥ 0.9996, max residual 11.7 ms.

Consequences: totals at 256k become **11.82 s** (16384) and **11.28 s** (32768). The
throughput ranking is unchanged. The chunk 2048 / 4096 / 8192 rows reproduce to **0.03%**, so
2.09x = 2.16x × 1.99x stands.

**The occupancy model's range is now honest.** `cost ∝ rounds/C`, calibrated at chunk 8192,
predicts the prefix term to **+2.8% / +0.1% / 0.0%** at 2048 / 4096 / 8192 but **−10.8%** at
16384 and **−14.8%** at 32768. The earlier "all five within 5%" was an artifact of the two
bad slopes. The mechanism is established in the range it was tested; do not extrapolate it.

**Method lesson:** a fit is only as good as its point count, and the original table mixed
slopes fitted from 128 points with slopes fitted from 2–4. Record N alongside every fitted
parameter, and refuse to report a slope from N < 3.

### 2. The per-op floor fit clamps one op's intercept, and the absolute floor is estimator-dependent

`cmp_ops.py` fitted `cost = F + k·tokens` per op with `F` clamped at zero. The clamp fires on
exactly **one** op — the global layer's `RingJointSDPA`, least-squares intercept **−290.4 µs**
— inflating the global per-layer fixed cost from 1651 to 1942 µs (**+17.6%**).

The negative intercept is not noise. At chunk index 0 that op attends the chunk against
itself, so its cost is roughly **quadratic** in the chunk (~C²/CP); an affine-in-C model is
misspecified for it, and clamping hides that rather than fixing it. The right treatment is to
exclude it from a floor fit — it is chunk-dependent attention work, not floor.

| quantity | per-op (clamped) | per-op (unclamped) | excess-in-range | whole-model affine |
|---|---:|---:|---:|---:|
| local share of the floor | 82.8% | 85.0% | 84.8% | — |
| absolute floor | 112.9 ms | 110 ms | 79.0 ms | **94.2 ms** |

**The share is robust (82.8–85.0%) and that agreement is the evidence. The absolute floor is
not** — a 79–113 ms spread. Quote the **whole-model affine fit (94.2 ms)**: it is anchored on
end-to-end per-chunk device times, whereas every per-op sum inherits the isolated-layer
harness's missing inter-layer overlap and its staging ops.

`scripts/cmp_ops.py` no longer clamps silently; it reports the raw intercept and flags any op
whose fit is misspecified.
