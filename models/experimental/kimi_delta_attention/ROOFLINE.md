# KDA prefill roofline — compute vs collectives (per KDA layer, per chip)

> Phase 8 companion to `DISTRIBUTION.md`. The yardstick the implementation is measured against.
> Utilization targets: **compute ≥ 60%** of matmul peak, **CCL ≥ 40%** of the fabric roofline (CCL
> targets are aspirational — collectives usually land lower). All numbers from in-repo constants
> (cited); a serial compute+CCL sum is an **upper bound** (ignores overlap).
>
> **Correction (supersedes DISTRIBUTION.md §3):** an earlier estimate used 400 GB/s effective fabric —
> a bits/bytes error. The CCL roofline (`test_sparse_mla_ccl_perf.py:88-97`) is `link_Gbps × links ×
> dirs / 8`: LoudBox **100 GB/s**, Galaxy **50 GB/s** (Linear). With the correct BW, KDA prefill is
> **CCL-bound**, not compute-bound. That aligns with the stack's known reality (dominant cost = CCL).

## Constants (provenance)

| Quantity | Value | Source |
|---|---|---|
| matmul FLOP/cycle/core | LoFi 4096 · HiFi2 2048 · HiFi4 1024 | `test_indexer_score.py:809` |
| BH clock | 1.35 GHz | `test_indexer_score.py:808` |
| compute grid | (11,10) = 110 cores | `mla_config.py:19`, `deepseek_v3_matmul_config.py:20` |
| **→ peak/device** | **LoFi 608 · HiFi2 304 · HiFi4 152 TFLOP/s** | derived |
| link BW / dir / link | Galaxy 200 · LoudBox 400 Gbps | `test_sparse_mla_ccl_perf.py:40-41` |
| links, topology | 2 links, `Topology.Linear` (sustained ×1) | `:36`, `mla.py:259` |
| **→ eff fabric BW** | **Galaxy 50 · LoudBox 100 GB/s** | `:88-93` (`/8`) |
| CCL critical path | `local_input_bytes × (participants−1)` | `:70-71,96-97` |

**Projection fidelity:** headline **HiFi2** (bf16 act × bfp8 weight — the deployed indexer path,
`test_indexer_score.py:826`). LoFi (bfp8×bfp8) and HiFi4 (bf16×bf16) bracket it.

## Workload (per chip)

Per-chip query rows are **640** on every box (box-invariant design: `chunk/SP`; LoudBox 1280/2,
Galaxy 5120/8). Heads/chip = 32/TP = **8** at TP=4. Projections = **95%** of MACs (39.5 M/token);
recurrence 5% (2.1 M). Per-chip FLOP/forward = 2·(41.6 M/TP)·640 = **13.3 GFLOP**.

**KDA note — warm ≡ long.** KDA's "cache" is its fixed-size state, so a query chunk costs the same
regardless of cache depth (50k or 512k). warm and long are the *same* per-forward roofline (the
linear-attention headline); only **cold** (full prefill = ~11 chunks) scales up. Contrast MLA, where
long ≫ warm.

## Roofline (per KDA layer, per chip, HiFi2 projections)

### LoudBox proxy (SP=2 × TP=4, 100 GB/s) — the measurement box

| | ideal | target |
|---|---|---|
| **compute** (13.3 GFLOP) | 44 µs | **73 µs** @60% |
| **CCL**: TP all-gather 22 + TP reduce-scatter 22 + SP state-scan 5 | **49 µs** | **124 µs** @40% |
| ideal C : CCL | **0.88 : 1** | balanced, slightly CCL-leaning |
| serial upper bound (warm/long) | 93 µs ideal | ~197 µs @targets |

### Galaxy target (SP=8 × TP=4, 50 GB/s)

| | ideal | target |
|---|---|---|
| **compute** (13.3 GFLOP) | 44 µs | 73 µs @60% |
| **CCL**: TP all-gather 44 + TP reduce-scatter 44 + SP state-scan **73** | **162 µs** | **405 µs** @40% |
| ideal C : CCL | **0.27 : 1** | **strongly CCL-bound** |
| serial upper bound (warm/long) | 206 µs ideal | ~478 µs @targets |

**cold (full prefill, ~11 chunks):** LoudBox compute 0.80 ms / CCL 1.36 ms @targets; Galaxy compute
0.80 ms / CCL **4.45 ms** @targets.

## Reading it — what to optimize

1. **CCL dominates, and more so on Galaxy** (0.27:1). Halved BW (50 vs 100 GB/s) *and* the SP=8
   state-scan (×7 hops) make collectives the bottleneck. This is the expected shape for this stack.
2. **The SP state-scan is NOT negligible on Galaxy** (73 µs, the largest single CCL term) — I was wrong
   earlier (that claim also rode the 400 GB/s error). A **sequential line-scan is 7 hops**; an
   **associative/tree scan (~⌈log₂8⌉=3 hops) roughly halves it** → the first CCL optimization to reach for.
3. **TP projection collectives** (all-gather in + reduce-scatter out, 88 µs on Galaxy) are the other
   half — candidates for matmul-fused collectives (all-gather-matmul) like the MLA q_a stem.
4. **Compute target (60%)** is a normal matmul goal; the projections are well-shaped
   `[640,2304]×[2304,4096]`-class matmuls. The recurrence's 5% is where the un-fused-kernel penalty
   lands (a fused kernel keeps it ~5%; the composed version inflates it — measured separately).

## Targets the implementation is held to

- **Compute:** projection matmuls ≥ **60%** of HiFi2 peak (304 TFLOP/s) → ≤ 73 µs/layer/chip (warm).
- **CCL:** collectives ≥ **40%** of roofline → ≤ 124 µs (LoudBox) / ≤ 405 µs (Galaxy) per layer/chip (warm).
- **Recurrence:** reported separately; un-fused inflates it, so it's excluded from the compute-util
  gate until the fused diagonal-gate kernel exists (immediate perf-phase follow-up).
