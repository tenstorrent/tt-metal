# Why Combined Fused Ops Don't Give Additive Duration Reduction

Analysis of `both/64k/prefill.csv` vs `rsmm_only` and `agmm_only` to find the gap.

---

## Total Time Comparison

| Config | Total (ms) | vs Baseline |
|--------|------------|-------------|
| Baseline | 135.87 | — |
| AG+MM only | 134.06 | −1.81 ms |
| RS+MM only | 132.75 | −3.12 ms |
| **Both** | **131.23** | **−4.64 ms** |

**Expected if additive:** −1.81 − 3.12 = **−4.93 ms**
**Actual:** −4.64 ms
**Gap:** 0.29 ms (small at total level)

---

## Fused-Op Duration Comparison (the real issue)

| Fused Op | RS+MM only | AG+MM only | **Both** | Δ vs single |
|----------|------------|------------|----------|-------------|
| MinimalMatmulStridedReduceScatterAsync (FF1+FF3) | 19,951 us | — | **20,627 us** | **+676 us (+3.4%)** |
| AllGatherMinimalMatmulAsyncOp (FF2) | — | 11,338 us | **11,335 us** | ~0 |

**Finding:** RS+MM fused op is **~3.4% slower** when both fusions are enabled. AG+MM is unchanged.

---

## Per-Op Breakdown (Both run, lines 32–36)

| Line | OP | Duration (us) |
|------|-----|---------------|
| 32 | MinimalMatmulStridedReduceScatterAsync (FF1) | 10,214 |
| 33 | MinimalMatmulStridedReduceScatterAsync (FF3) | 10,414 |
| 34 | BinaryNg (mul) | 1,568 |
| 35 | **AllGatherMinimalMatmulAsyncOp** (FF2) | **11,335** |
| 36 | ReduceScatterMinimalAsyncDeviceOperation (FF2 out) | 3,346 |

---

## Likely Causes for RS+MM Slowdown When Both Enabled

1. **Resource contention**
   - RS+MM and AG+MM both use ring CCL (cluster_axis=1, num_links)
   - When both run in the same layer, they may contend for:
     - Ring links / network bandwidth
     - DRAM bandwidth
     - Worker cores (66 vs 54 cores each)

2. **Memory / cache pressure**
   - FF1+FF3 outputs feed FF2; with both fused, more data may be live
   - Possible cache thrashing or DRAM pressure between fused ops

3. **Scheduling / overlap**
   - RS+MM and AG+MM may be scheduled differently when both are present
   - Slight serialization or worse overlap could add latency

---

## Summary

- **AG+MM** duration is stable when combined with RS+MM.
- **RS+MM** duration increases by ~676 us (~3.4%) when both are enabled.
- That ~676 us explains most of the gap between expected (−4.93 ms) and actual (−4.64 ms) total savings.
- Root cause: resource contention or scheduling when both fused ops run in the same MLP block.

---

## Next Steps to Investigate

1. **Profiler:** Check if RS+MM and AG+MM overlap in time or run back-to-back.
2. **Grid / links:** Confirm whether both use the same ring (cluster_axis=1) and if link sharing causes contention.
3. **Core usage:** RS+MM uses 66 cores, AG+MM 54; check for core overlap or scheduling effects.
