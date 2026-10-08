# rmsnorm-prefill campaign: final report

Op: ttnn.experimental.dit_fused_distributed_rmsnorm, prefill shapes, 1x4 Blackhole mesh (bh-qbge-12), bf16.
Metric: per-chip mean device kernel time over 10 measured calls; score = geomean speedup vs baseline.
Accuracy rule: all compute at MathFidelity::HiFi4 (human decision 2026-10-08), PCC >= 0.99999, max_abs <= 0.05.

## Result
Best valid node: r05-b01-a01 (tag dream/rmsnorm-prefill/n/r05-b01-a01), score 1.5696.

| shape | baseline µs | best µs | speedup |
|---|---|---|---|
| kimi-k3 h3584 (local 896) | 16.99 | 11.19 | 1.518x |
| deepseek-v4 h4096 (local 1024) | 18.26 | 12.08 | 1.512x |
| glm-5 h6144 (local 1536) | 23.48 | 14.59 | 1.610x |
| kimi-k2 h7168 (local 1792) | 26.03 | 15.84 | 1.643x |

Accuracy at the best node: PCC 0.9999985 (same as baseline), max_abs 0.020-0.024, mean_abs ~0.0012 (baseline 0.00125).

## Progress per round
| round | policy | root | attempts | best |
|---|---|---|---|---|
| r01 | v0 | campaign root | 16 | 1.2132 |
| r02 | v1 | r01-b04-a04 | 7 | 1.2396 |
| r03 | v2 | r02-b02-a03 | 12 | 1.3333 |
| r04 | v3 | r03-b02-a02 | 12 | 1.3997 (1.4475 with HiFi2, invalidated) |
| r05 | v4 | r04-b04-a02 | 8 | 1.5696 |
| r06 | v5 | r05-b01-a01 | 3 (+1 lost, stopped) | 1.5512 |

58 committed attempts (55 valid under the final rules, 3 invalidated for reduced fidelity), 1 lost, 5 dreaming phases.
Approximate model cost: ~$225 (workers + policy agents). Wall clock: about 7 hours.

## What made it faster (lineage of the best node)
1. x*gamma pre-pass under the cross-chip all-gather wait; gamma read on the idle writer RISC; trid-pipelined input reads (r01).
2. sum(x^2) accumulated in DST, one pack per row; path-aware dual-NoC output drain (r01-r02).
3. Post-AG stat finalize on row 0 only (fused SFPU add_rsqrt); stats scratch in L1 (r03).
4. Posted (no-ack) output writes; ack-free stick push; streamed gamma chunks (r04).
5. k=2 column split run as two 20-core all-gather waves, overlapping each wave's read/drain with the other's AG (r05).

## Dead ends (don't retry without new evidence)
- Reduced math fidelity (rule). More output-drain parallelism (second cmd buf, VC round-robin, congestion-aware NoC choice).
- Column-major worker placement. 4 waves (needs the stick-push un-gate; still below 2 waves). Uneven 9/11 wave split.

## Open leads
- Port r05-b03-a03's stick-push un-gate fix (no wait on gamma) to the 2-wave best (r05-b01-a01).
- Cross-device launch skew (about 2 µs between chips at h7168) dominates the max-over-chips latency.

## Policy evolution (dreaming)
v0 parallel refine -> v1 gap-to-leader close -> v2 depth 3 -> v3 stall close + no lone trailing step -> v4 no single-item step
-> v5 earned depth. Known limit: the replay objective rewards only the best score inside a round and charges every attempt,
so it can't value discoveries that pay off in a later round, and each dreaming phase added pruning.
