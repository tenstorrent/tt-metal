# Per-op comparison: chunk 2048 vs 4096 vs 8192

`tt-perf-report` Device Time, **ms per layer**, on `mmanzoor/svuckovic/gemma4-L1-activations`
@ `9dc8e32a2` (+ multi-hop halo), `GEMMA4_PREFILL_L1_ACT=1`, mesh 8x4, ctx 262,144.

Two depths per width, at **matched prior context 57,344 tokens** — chunk index 28 / 14 / 7 for
2048 / 4096 / 8192. Matching prior context (not chunk index) is what makes the prefix work
exactly proportional to chunk size.

**`r2048` = chunk-2048 time / chunk-8192 time. Ideal is 0.250** (a quarter of the tokens should
be a quarter of the work). Anything well above 0.250 does not scale with chunk width.

The single-cell tables these are built from are in `reports/` — see the index in the summary.


## Sliding layer (x50 in the model)

| op | n | 2048 first | 2048 depth | 4096 first | 4096 depth | 8192 first | 8192 depth | `r2048` | cores 2048→8192 |
|---|---|---|---|---|---|---|---|---|---|
| `MatmulDeviceOperation _x5376x5376` | 3 | 0.567 | 0.567 | 0.862 | 0.862 | 0.636 | 0.635 | **0.891** ⚠️ | 84 → 96 |
| `AllGatherDeviceOperation` | 3 | 0.159 | 0.161 | 0.313 | 0.314 | 0.564 | 0.566 | **0.282** | 10,2 → 34 |
| `LayerNormDeviceOperation` | 7 | 0.462 | 0.463 | 0.468 | 0.468 | 0.523 | 0.522 | **0.884** ⚠️ | 64,32,8 → 120,32 |
| `ReduceScatterDeviceOperation` | 2 | 0.144 | 0.145 | 0.255 | 0.257 | 0.483 | 0.484 | **0.299** | 18 → 34 |
| `RingJointSDPADeviceOperation` | 1 | 0.403 | 0.402 | 0.421 | 0.443 | 0.450 | 0.461 | **0.895** ⚠️ | 114 → 112 |
| `MatmulDeviceOperation _x5376x4096` | 1 | 0.178 | 0.178 | 0.121 | 0.121 | 0.226 | 0.227 | **0.789** ⚠️ | 64 → 96 |
| `GatherCodegenDeviceOperation` | 2 | 0.072 | 0.072 | 0.125 | 0.125 | 0.193 | 0.193 | **0.375** | 64 → 120 |
| `MatmulDeviceOperation _x2048x5376` | 1 | 0.086 | 0.086 | 0.135 | 0.135 | 0.164 | 0.164 | **0.521** ⚠️ | 84 → 96 |
| `BinaryNgDeviceOperation` | 4 | 0.042 | 0.041 | 0.079 | 0.078 | 0.149 | 0.147 | **0.280** | 120 → 120 |
| `TilizeDeviceOperation` | 1 | 0.057 | 0.056 | 0.090 | 0.090 | 0.122 | 0.123 | **0.466** ⚠️ | 84 → 84 |
| `NlpCreateHeadsDeviceOperation` | 1 | 0.056 | 0.056 | 0.057 | 0.057 | 0.060 | 0.059 | **0.939** ⚠️ | 8 → 32 |

- **chunk 2048: layer total 2.329 ms first chunk → 2.330 ms at depth** (+0.1%)

- **chunk 4096: layer total 3.041 ms first chunk → 3.063 ms at depth** (+0.7%)

- **chunk 8192: layer total 3.716 ms first chunk → 3.724 ms at depth** (+0.2%)

## Global layer (x10 in the model)

| op | n | 2048 first | 2048 depth | 4096 first | 4096 depth | 8192 first | 8192 depth | `r2048` | cores 2048→8192 |
|---|---|---|---|---|---|---|---|---|---|
| `RingJointSDPADeviceOperation` | 1 | 0.189 | 4.249 | 0.356 | 4.410 | 1.314 | 9.426 | **0.144** | 114 → 114 |
| `MatmulDeviceOperation _x5376x5376` | 3 | 0.567 | 0.567 | 0.862 | 0.862 | 0.635 | 0.635 | **0.893** ⚠️ | 84 → 96 |
| `AllGatherDeviceOperation` | 3 | 0.159 | 0.161 | 0.315 | 0.314 | 0.567 | 0.569 | **0.281** | 10,2 → 34 |
| `LayerNormDeviceOperation` | 6 | 0.462 | 0.462 | 0.470 | 0.470 | 0.521 | 0.520 | **0.886** ⚠️ | 64,8 → 120,32 |
| `ReduceScatterDeviceOperation` | 2 | 0.144 | 0.145 | 0.260 | 0.265 | 0.499 | 0.506 | **0.288** | 18 → 34 |
| `GatherCodegenDeviceOperation` | 5 | 0.176 | 0.176 | 0.213 | 0.213 | 0.374 | 0.362 | **0.472** ⚠️ | 32 → 120 |
| `MatmulDeviceOperation _x4096x5376` | 1 | 0.152 | 0.152 | 0.234 | 0.233 | 0.280 | 0.280 | **0.545** ⚠️ | 84 → 96 |
| `MatmulDeviceOperation _x5376x4608` | 1 | 0.186 | 0.185 | 0.283 | 0.283 | 0.249 | 0.248 | **0.746** ⚠️ | 72 → 96 |
| `SliceDeviceOperation` | 5 | 0.208 | 0.210 | 0.210 | 0.210 | 0.219 | 0.220 | **0.948** ⚠️ | 120,32 → 120 |
| `BinaryNgDeviceOperation` | 5 | 0.044 | 0.045 | 0.081 | 0.081 | 0.149 | 0.150 | **0.296** | 120 → 120 |
| `TilizeDeviceOperation` | 1 | 0.057 | 0.057 | 0.090 | 0.090 | 0.122 | 0.122 | **0.471** ⚠️ | 84 → 84 |

- **chunk 2048: layer total 2.548 ms first chunk → 6.611 ms at depth** (+159.5%)

- **chunk 4096: layer total 3.597 ms first chunk → 7.651 ms at depth** (+112.7%)

- **chunk 8192: layer total 5.195 ms first chunk → 13.299 ms at depth** (+156.0%)


## Which layer type owns the floor — 85 / 15, and it is the layer count

The two tables above look comparably bad, and reading them side by side is misleading: **there
are 50 sliding layers and 10 global ones.** Per layer the chunk-invariant cost is nearly the
same; the 5x count is the whole story.

| layer | count | `F` per layer | `F` x count | scaled to the e2e fit | share |
|---|---|---|---|---|---|
| **sliding** | 50 | 1.866 ms | 93.3 ms | **84.7 ms** | **84.9%** |
| global | 10 | 1.666 ms | 16.7 ms | 15.1 ms | 15.1% |
| | | | 110.0 ms | 99.8 ms | (per-op sum overstates by 10%) |

**So debug effort belongs in the sliding layer.** The same fix is worth 5x more there.

### Per-op `F`, split by layer type

Scaled to the e2e fit. This is the table to pick work from:

| op | sliding x50 | global x10 | total | share |
|---|---|---|---|---|
| Matmul `_x5376x5376` | **24.7 ms** | 4.9 ms | 29.6 | 29.7% |
| LayerNorm | **20.1 ms** | 4.0 ms | 24.1 | 24.1% |
| RingJointSDPA | **17.6 ms** | **−1.7 ms** | 15.9 | 15.9% |
| Matmul `_x5376x4096` | **7.4 ms** | 0.0 ms | 7.4 | 7.4% |
| NlpCreateHeads | 2.5 ms | 0.6 ms | 3.1 | 3.1% |
| Matmul `_x2048x5376` | 2.7 ms | 0.0 ms | 2.7 | 2.7% |
| GatherCodegen | 1.5 ms | 1.0 ms | 2.5 | 2.5% |
| RotaryEmbeddingLlama | 1.9 ms | 0.2 ms | 2.1 | 2.1% |

Three things follow that the ratio tables above do not show:

1. **The top three ops are 70% of the floor, and 62.4 of their 69.6 ms is in the sliding
   layer.** Fix any of them and it pays five times over.
2. **The global SDPA's contribution to the floor is NEGATIVE (−1.7 ms)** — see below.
3. **`_x5376x4096` is sliding-only** (7.4 ms, zero in global): it is the sliding attention's
   qkv shape. Small overall but exclusively in the 50x layer, so better value than 7.4% looks.

### Why the global SDPA's floor ratio is 0.144, i.e. *better* than ideal

It is not a typo and it is not a contradiction. **That op has two different problems and only
one of them is a floor problem.**

- **At the floor (chunk index 0) there is no history**, so it computes only the causal diagonal
  block of the chunk against itself. That work is roughly quadratic in the chunk, not linear,
  so quartering the chunk cuts it by more than 4x — hence 0.144 against a linear ideal of
  0.250, and hence a *negative* contribution to the chunk-invariant cost.
- **At depth it is the dominant cost in the model**, ~100% of the global layer's growth, and
  that is where its 71%-idle-grid problem lives (see the occupancy note above).

So: **do not target this op for floor work** — it is already better than linear there. Target it
for *prefix* work, at chunk 2048, where it is 1.9x off C^2 scaling. The ⚠️ marks in the tables
above are on `r2048 > 0.45` and correctly skip this row.

## How to read it

**⚠️ marks `r2048 > 0.45`** — ops that are nowhere near scaling with chunk width. Those are the
per-chunk floor, and they are why `a(2048)` is 126 ms rather than 205/4 = 51 ms.

**The sliding layer barely moves between "first chunk" and "at depth" at any width** — that is
the control. A windowed layer cannot see history, and it doesn't. It makes the global layer's
growth credible rather than a harness artifact.

**The global layer's growth is one op**, `RingJointSDPADeviceOperation`, ~100% of the delta.
Growth ratios 0.490 / 0.501 / 0.978 at 2048 / 4096 / 8192 against occupancy's predicted
0.500 / 0.500 / 1.000 — so 2048 and 4096 cost the *same* for prefix work.

## Why the `Cores` column is not occupancy — with the evidence

This matters because the `Cores` column is the first thing anyone reads off these tables, and
for the op that dominates the prefix term it is **actively misleading**.

**The SDPA's cost varies 50x while `Cores` never moves.** Same op, all six cells in
[`per_op/`](per_op/):

| cell | Device Time | `Cores` |
|---|---|---|
| `c2048_floor_global` | 0.189 ms | **114** |
| `c4096_floor_global` | 0.356 ms | **114** |
| `c8192_floor_global` | 1.314 ms | **114** |
| `c2048_deep_global` | 4.249 ms | **114** |
| `c4096_deep_global` | 4.410 ms | **114** |
| `c8192_deep_global` | 9.426 ms | **114** |

A column that reads an identical 114 while the op ranges over **0.189 → 9.426 ms** is reporting
the *grid it was given*, not the work it did. Meanwhile the real occupancy nearly doubles across
those widths and the column cannot show it:

| chunk | work units | grid passes | useful |
|---|---|---|---|
| 2048 | 32 | 1 | **29.1%** |
| 4096 | 64 | 1 | 58.2% |
| 8192 | 128 | 2 | 58.2% |

**The source says why explicitly.**
`ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp:834`:

> *"A core with no Q chunks (`global_q_start == global_q_end`) is **NOT dead**: in the GQA /
> shared-K row-wide multicast path it runs padded handshake iterations
> (`loop_q_count = *_max_q_per_core`) so the injector's mcast rectangle never targets a silent
> worker."*

So every core in the rectangle is genuinely dispatched to and genuinely reports in — they just
have no Q chunk to work on. The profiler counts them because they ran. Occupancy has to come
from the factory's own work-unit math (`all_heads_num_q_chunks = B * NH * num_q_chunks`,
`max_q_per_core = div_up(all_heads_num_q_chunks, num_cores)`, same file, line 1306).

**By contrast `Cores` IS informative for LayerNorm** — and this is the useful half of the story:

| cell | Device Time | `Cores` |
|---|---|---|
| `c2048_floor_local` | 0.462 ms | **64,32,8** |
| `c4096_floor_local` | 0.468 ms | **120,64,16** |
| `c8192_floor_local` | 0.523 ms | **120,32** |

The small entry is the hidden-width norm, and it reads **8 / 16 / 32** — exactly
`(chunk/CP)/32`, one core per row-tile. Here the column moves, it matches the row-parallel
decomposition, and the op's time barely moves because so few cores are doing it. That is a
defect you can read straight off the table, and it is the one this branch's norm-sharding
commit fixes (8 → 16 cores at chunk 2048, 4.36x on the op).

**Rule of thumb:** trust `Cores` when it *varies* with the shape in a way the op's
decomposition predicts. Distrust it when it is pinned to the grid size — that means idle cores
are being counted, and you need the op's work-unit math instead.
