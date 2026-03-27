# 32K Prefill Single-Layer: Combined Fused Ops Analysis

Analysis of `both/32k/prefill.csv` vs `rsmm_only` and `agmm_only_32k` to verify RS+MM slowdown when combined with AG+MM.

---

## Total Time Comparison (32K, 1 layer)

| Config | Total (ms) | vs Baseline |
|--------|------------|-------------|
| Baseline | 56.41 | — |
| AG+MM only | 55.47 | −0.94 ms |
| RS+MM only | 54.96 | −1.45 ms |
| **Both** | **54.02** | **−2.39 ms** |

**Expected if additive:** −0.94 − 1.45 = **−2.39 ms**
**Actual:** −2.39 ms
**Gap:** 0 ms — savings are additive at 32K.

---

## Fused-Op Duration Comparison

| Fused Op | RS+MM only | AG+MM only | **Both** | Δ vs single |
|----------|------------|------------|----------|-------------|
| MinimalMatmulStridedReduceScatterAsync (FF1+FF3) | 9,866 us | — | **9,909 us** | **+43 us (+0.44%)** |
| AllGatherMinimalMatmulAsyncOp (FF2) | — | 5,693 us | **5,690 us** | ~0 |

**Finding:** RS+MM fused op is **~0.44% slower** when both fusions are enabled at 32K. AG+MM is unchanged. The slowdown is **much smaller** than at 64K (+676 us, +3.4%).

---

## Per-Op Breakdown (Both run, lines 34–38)

| Line | OP | Duration (us) |
|------|-----|---------------|
| 34 | MinimalMatmulStridedReduceScatterAsync (FF1) | 4,975 |
| 35 | MinimalMatmulStridedReduceScatterAsync (FF3) | 4,934 |
| 36 | BinaryNg (mul) | 786 |
| 37 | **AllGatherMinimalMatmulAsyncOp** (FF2) | **5,690** |
| 38 | ReduceScatterMinimalAsyncDeviceOperation (FF2 out) | 2,143 |

---

## 32K vs 64K Comparison

| Metric | 32K | 64K |
|--------|-----|-----|
| RS+MM slowdown when combined | +43 us (+0.44%) | +676 us (+3.4%) |
| AG+MM when combined | ~0 | ~0 |
| Expected vs actual gap | 0 ms (additive) | 0.29 ms (sub-additive) |

---

## Conclusion

1. **RS+MM is still slower when combined** at 32K, but the effect is **~15× smaller** than at 64K (+43 us vs +676 us).
2. **AG+MM timing is stable** when combined with RS+MM at both 32K and 64K.
3. At 32K, total savings are **additive** — no measurable gap between expected and actual.
4. **Resource contention scales with sequence length:** at 64K, both fused ops handle 2× the data; ring links, DRAM bandwidth, and core usage are more stressed, leading to the larger RS+MM slowdown when both run in the same MLP block.

---

## Data Sources

- Baseline: `profiler_sweep_results/baseline_32k/32k/prefill.csv`
- AG+MM only: `profiler_sweep_results/agmm_only_32k/32k/prefill.csv`
- RS+MM only: `profiler_sweep_results/rsmm_only/32k/prefill.csv`
- Both: `profiler_sweep_results/both/32k/prefill.csv`
