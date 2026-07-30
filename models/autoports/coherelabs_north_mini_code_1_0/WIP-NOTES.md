# north-mini-code-1-0 — arm `nofuse-noadvise`, PARKED (blocked, batch-1 result IS valid)

Parked per MONITORING.md §6b. **Not tagged** — the stage did not pass its gate. But unlike the
earlier gemma park, the numbers in here are **real and usable**: the block is batch-32-only.

## Provenance
- base `b9e6c242a34` · arm `nofuse-noadvise` `51b17c3da34` · FD tag `skillexp/fd-ready/coherelabs_north_mini_code_1_0` (`78dbd88bec7`)
- run2 `acadb63da7d` (2h21m, DECODE_BATCH=32) → resume-1 `3ba45973e79` (11m, DECODE_BATCH=1 MOE=1 STEER=1)
- `.agents` tree `33ada6f46a4` == arm, verified before AND after both runs. No factor drift.

## Batch 1 is complete and correct
seq 128, 3 warmups, 20 samples, complete-forward trace replay. Functional = stage-entry baseline.

| Layer kind | Phase | Functional b1 | Optimized b1 | Δ |
|---|---|---:|---:|---:|
| dense/full/forced-RoPE | decode | 0.356 ms | 0.187 ms | −47.5% |
| sliding/RoPE/MoE | decode | 9.528 ms | 0.792 ms | **−91.7%** |
| full/no-RoPE/MoE | decode | 9.524 ms | 0.795 ms | **−91.7%** |
| sliding/RoPE/MoE | prefill | 14.908 ms | 14.191 ms | −4.8% |
| full/no-RoPE/MoE | prefill | 14.655 ms | 14.264 ms | −2.7% |

Decode PCC 0.99664833; prefill 0.99896813 / 0.99897853. Full 500,000 context, no reduction.
Attention precision: all 16 BFP4 candidates failed the 0.995 bar (best 0.99474771); BFP8/LoFi
selected. Sparse output subblocks (cumulative 1×2, 12 cores gate/up + 32 down) gave a further
−8.37% on b1 decode (0.791673 → 0.725390 ms).

## Why it is blocked — batch 32 only, and it is a RUNTIME dependency
On a 1×1 mesh the routed-MoE combine needs `MoEComputePath::FullLocal` (fabric-free local
selective-reduce-combine, drains each rolling expert output while live). This checkout's HEAD has
only `Full` and `ComputeOnly`. `FullLocal` lands in upstream **`50c56281566`**
("Feature: Add single-device fused moe_compute support (#49886)"), present on `origin/main`, not
here. It spans **18 shared-TTNN files**.

**Deliberately not backported.** Changing the runtime under a perf ablation is the exact hazard the
experiment ruled on when it chose `base` over `gpt-oss-trace-tracker`, and it would desynchronise
this machine's build from machine A's, making every cross-machine cell incomparable.

**Expect this to hit all four cells**, including A's advise arms: the dependency is in TTNN, not in
either skill. Tell A before it spends device time on north-mini.

## Why the resume did not close the gate, and what supersedes this park
`MOE=1` put batch 32 out of scope and the scoped objective **did** reach the thread (44 occurrences
of the carve-out and the steering note; thread objective 3914 chars, no "serving batch 32"). It still
produced `AUTODEBUG_B32_FRESH.md`. A resumed thread carrying 2h21m of batch-32-focused context was
not redirected by a changed objective — a real limitation of scoping-by-resume.

**Superseded by a FRESH run** (fresh branch + fresh log dir) with `MOE=1` from the first turn, so no
b32 context ever accumulates. This park exists so that rerun can be compared against it, and so the
b1 numbers and the `50c56281566` diagnosis survive regardless of how the rerun goes.
