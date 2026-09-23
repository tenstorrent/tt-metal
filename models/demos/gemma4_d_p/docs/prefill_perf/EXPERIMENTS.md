# Experiment ledger

Every experiment run against the Gemma4 prefill per-chunk floor, including the negatives.
Negatives are listed because most of them steered the work: the fix worth the most
(`GEMMA4_MLP_MM_CFG`, −26.9 ms at chunk 4096) exists because a *null* result eliminated the
competing explanation.

> ### Naming
> Experiments were originally numbered Exp 0–6, and a later session started a second Exp 6
> and Exp 7. **Those numbers collide and are retired.** The env flag is the canonical ID:
> unique, present in the code, and in every commit message. Old-number mapping is in the
> last column so earlier documents remain readable.

## Landed

| flag | what it changes | 2048 | 4096 | 8192 | ship? | was |
|---|---|---|---|---|---|---|
| `GEMMA4_NORM_SHARD` | block-shard the prefill RMSNorm instead of row-parallelising | **−16.7 ms** | −13.6 | −8.7 | ✅ **yes** | Exp 4 |
| `GEMMA4_MLP_MM_CFG` | explicit matmul program config on the 3 MLP projections | **−10.7 ms** | **−26.9** | −1.7 | ✅ **yes** | Exp 1, re-tested |
| `GEMMA4_ATTN_MM_PC` | same, on the 4 attention projections | **−6.3 ms** | −6.5 | **−11.5** | ✅ **yes** | (new) |
| `GEMMA4_ATTN_MM_CFG` | `core_grid` only, on the 4 attention projections | **+1.3 ms** | +1.5 | −4.1 | ❌ **no** — superseded by `ATTN_MM_PC`; code removed | Exp 2 |

All three shipping fixes are **call-site configuration**. No op or kernel implementation was
modified.

## Negative and closed

| experiment | result | what it bought |
|---|---|---|
| MLP weight bytes bfp8 → bf16 | **null** — 1.88x the bytes cost **+0.6 ms** vs a predicted +30 | Killed "byte-bound" for 30% of the floor and pointed straight at `MLP_MM_CFG` |
| sliding SDPA `q_chunk` sweep | falsified — q=128 is a 1.0% *regression*, q=32 is a `TT_FATAL` | Closed a tempting knob |
| sliding window 512 / 1024 / 2048 | **diagnostic**: `a = 94.9 + 0.01515·window`, ~88% of the op's floor is halo | Identified the #1 remaining item |
| `GEMMA4_NUM_LINKS` 3, 4 | **impossible** — `TT_FATAL: 2 ethernet channels available` | `num_links` is at the hardware max; no CCL lever |
| `GEMMA4_CCL_ASYNC=1` | **worse** — +0.7 ms (≈23σ) | Closes CCL entirely; its docstring's "measure it on the target board" is now done, and it is red |
| `ATTN_MM_CFG` size gate (`rows/device ≥ 1024`) | no-op, ±0.5 ms | Reverted rather than shipped on ambiguous data |
| chunk 1536 | illegal — `(chunk/CP) % k_chunk ≠ 0` | Corrected the published legality rule to `chunk % 1024 == 0` |
| variable chunk width PoC | dropped — loses to fixed 8192 mid-range | — |
| per-head q/k/v RMSNorm tuning | **not worth it** — 17 calls per 6 layers, 114 µs, i.e. ~19 µs/layer (~1.1 ms per chunk at 2048) in the 6-layer traced capture | The LayerNorm share is the 4 hidden-size norms PR2 already block-shards, not the per-head ones |
| drop the block-sharded norm's reshards | **deferred (decided 2026-09-23)** — the reshards cost 188 + 179 µs per 6 layers (~3.7 ms per chunk at 2048), already netted out of PR2's gain. Keeping the output sharded into the two residual adds would save only ~0.9 ms (~1%); removing them all needs a sharded residual stream | Revisit only alongside a sharded-residual refactor; the SDPA halo and CCL are worth more |
| whole-model tracy capture | **infeasible at 60 layers** — 1.19e9 zones, OOM-killed while saving. **Feasible at 6 layers, ctx 32k** (~1.5e7 zones, 254 MB CSV, ~20 min host post-processing) | Why per-op numbers are isolated-layer × 50/10; the 6-layer traced capture (2026-09-23) replaced that for attribution |

## Open

| item | size @2048 | status |
|---|---|---|
| sliding SDPA halo | 21.3 ms (26%) | **Diagnosed 2026-09-23** (6-layer traced capture): the reader waits for the whole halo before computing; the halo is ~75% of a sliding SDPA call (156-324 us of 277-402 us at chunk 2048) and is link-contention bound: h multi-hop unicasts put 10 slab-crossings on 2 links where a line multicast needs 4. Compute after the halo is a constant ~77 us on 32 of 110 cores. Plan: confirm with an op-level 2-hop vs 4-hop capture, then multicast the halo with a source-keyed compact buffer (est. -7 to -9 ms per chunk at 2048) |
| ~~residual outside the layer loop~~ | ~~11.4 ms~~ | **Resolved 2026-09-23: it is not outside the loop.** See below |
| `MLP_MM_CFG` slope at chunk 4096 | +3.07% (39σ) | Real, and **in-layer**: the layer-count fit's slope intercept is ~0 (`slope(L) = 0.0015 + 0.0246·L`). Zero impact when ISL ≤ chunk |

## Resolved: the "11.4 ms outside the layer loop" (2026-09-23)

The 11.4 ms was `a` minus isolated-layer per-op sums × 50/10 — a subtraction, never a measurement. Two
measurements replace it (chunk 2048, all fixes on):

- **Layer-count fit** (traced demo, ctx 32k, `L` = 6 / 12 / 30 / 60 layers built from the warm cache):
  `a(L) = 0.54 ms + 1.503 ms/layer · L`, residuals < 0.1 ms. Only **~0.5 ms** per chunk sits outside the
  layer loop.
- **6-layer traced tracy capture** (PR2's final config on this branch): kernel-time sum per chunk is
  9.12-9.25 ms across the 32 devices, against 9.5 ms unprofiled, so **≥ 96% of a chunk is inside op
  kernels**. Op-to-op gaps are a median of 0.6 µs, p99 2.0 µs per op, about 2 ms per 60-layer chunk.

So the residual was never dispatch or glue. Ops run slower inside the full model than in the isolated-layer
captures, ~1.52 vs ~1.36 ms/layer. **Isolated-layer per-op captures understate in-model op time.** Use them
to rank ops, not to size the floor.

Chunk-0 kernel time by op in the 6-layer capture: Matmul 36.5%, RingJointSDPA 18.7%, ReduceScatter +
AllGather 17.2%, LayerNorm 7.9% (41 calls per 6 layers: the 4 hidden-size norms plus the per-head q/k/v
norms), NlpCreateHeads 3.8%.

## Two results that are easy to misread

**The matmuls are done as a tuning target.** After `MLP_MM_CFG` + `ATTN_MM_PC`, all five
distinct matmul shapes sit at **96 cores and 50–52% of the DRAM roof**. 96 is a hard ceiling
at chunk 2048: `M = chunk/CP = 256` is 8 tile-rows, so only 8 of the grid's 10 rows can carry
work (8 × 12 = 96) and the other 24 cores are unreachable by this decomposition at any
setting. Do not re-sweep `in0_block_w` expecting a third win.

**Two different "bad matmul config" failure modes exist and need opposite fixes.**
*Over-parallelisation*: `_x2048x5376` is half the MACs of the next smallest op and goes
**+41%** on a full grid — fewer cores is better. *Bad blocking*: the MLP matmul is fixed by
`in0_block_w` and is **indifferent to core count**. They look identical in a `Cores` column.
