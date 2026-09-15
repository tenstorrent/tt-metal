# Improved FP32 streaming SDPA — 3.32-second target

## Result

The final independently rebuilt candidate is **3601.68 ms at 0.487625% L2**, down from **4597.94 ms** in perf-v3: **21.67% less time, 1.277x speedup**. The **3320 ms target was not reached**: another 281.68 ms, or 7.82% of the current time, must be removed. A valid refreshed main control is **3329.98 ms at 8.798303% L2**; the improved candidate is **8.16% slower**.

Both QK and PV remain **HiFi2**, with approximate exp, FP32 destination/online state, BF16 Q/K/V/output, and the same host Q preprocessing (six-bit bit-ceil and scale compensation 1.0027). This is now a **compute_streaming FP32 implementation**. The best configuration changes K chunk size from 512 to **1024**, retaining Q chunk 128.

Three normal-input seeds pass the explicit aggregate relative-L2 gate: **0.488857%, 0.489988%, 0.487625%**. Full-operator tail output and sampled-device tail output give exactly the same 0.487624912112% L2 for seed 1236. This is **not** a guarantee for every row or distribution; see the stress results below.

## Measurement contract

Blackhole P100A on yyzo-bh-26, reservation 214149, 11x10 = 110 compute cores; **not Galaxy**. Worktree branch cglagovich/blackhole-work-20260908, main base 2ba6fc2339d53300ae87c5202f335ef56492cfb3. Nominal clock 1350 MHz, 150 W card limit, neither changed.

Primary attention: noncausal B=1, H=10, Q=K=262144, D=128. FP64 reference is computed from the **original BF16 inputs**, on 128 tail queries per head; L2 is not over every output element. Relative L2 is 100*norm(output-reference)/norm(reference).

Timing is median **blocking trace replay wall time**, with 40 warmups and 10 measured replays, not a claim of raw device-counter latency. It excludes compilation, transfers, reference work, and approximately 106 ms of host Q preprocessing. The public packer_l1_acc field is false, but the kernel explicitly uses packer L1 accumulation internally.

| Candidate | Q/K chunks | Median ms | L2 % |
|---|---|---:|---:|
| Previous retained perf-v3 | 128/512 | 4597.940 | 0.491831 |
| Exact main, forced rebuild, fresh cache | 128/512 | 3329.980 | 8.798303 |
| FP32 streaming + identity correction | 128/1024 | 4028.048 | 0.488587 |
| Format caching + effective-weight FPU denominator | 128/1024 | 3763.050 | 0.487625 |
| Paired SFPU replay refiner, initial full run | 128/1024 | 3603.439 | 0.487625 |
| Final retained, forced rebuild, isolated cache | 128/1024 | **3601.677** | **0.487625** |

The initial replay-refiner run spans 3593.85–3609.07 ms. Raw results: [stream-identity-full256.jsonl](stream-identity-full256.jsonl), [stream-denom-full256.jsonl](stream-denom-full256.jsonl), [stream-replay-full256.jsonl](stream-replay-full256.jsonl). Exp replay scheduling produces exactly unchanged numerical metrics/output versus the denominator candidate. The final independently rebuilt run spans **3593.74–3613.25 ms**, with exactly the same output metrics: [final-forced-w40.jsonl](final-forced-w40.jsonl). Host Q preprocessing is 106.03 ms in that run.

The valid main control spans 3325.50–3334.64 ms: [main-forced-w40.jsonl](main-forced-w40.jsonl). Main and candidate use the same original input tensors/reference and seed, but only the candidate preprocesses Q, and chunk layouts differ. Main with Q/K chunks 128/1024 fails L1 allocation: **1,909,760 B requested vs 1,572,864 B available** ([main-forced-k1024.log](main-forced-k1024.log)). The new single-buffered layout enables this chunk size; there is no valid exact-main 128/1024 timing on this card.

## Retained implementation

- Port the improved FP32 algorithm into streaming compute while preserving full FP32 recurrent numerator/denominator arithmetic through direct-to-DST unpack and SFPU correction. Merely allocating FP32 CBs is insufficient: ordinary FPU state arithmetic loses low mantissa bits.
- Use 1x4 QK/PV subblocks, batched exponential processing, and single-buffered K/V at K chunk 1024 to fit L1. Larger chunks reduce the number of online updates.
- Form the denominator with a LoFi P-times-column-of-ones matmul. It sees the same six-bit effective SrcB P weights as HiFi2 PV; this removes the explicit SFPU mantissa mask and SFPU denominator reduction. LoFi applies only to this sum, not either large matmul.
- Keep the biased cubic exp refiner. Its grid and polynomial use separate regions of Blackhole's **32-entry per-thread replay buffer**: slots 0–7 for the grid macro pattern, 8–21 for two independent Horner chains. Address modifiers and the final stores preserve the original traversal.
- Cache PACK format/width configuration and avoid redundant UNPACK/MATH reconfiguration within this BF16/FP32-only specialization. Exact identity max corrections skip the rescale calculation.
- Restrict activation to validated long-context noncausal geometry: BF16 Q/K/V, FP32 DST, HiFi2, approximate math/exp, D=128, Q chunk=128, K chunk=512 or 1024, K length at least 32768, no generated padding, mask, causal/windowed/chunked/MLA/sink features. Other shapes retain the previous FP32 implementation. Previous BF16 compensation is unchanged.

## Counter profile and remaining gap

On full **128K**, to avoid 32-bit counter overflow, the final candidate's median active-core utilization is **FPU 42.01%, SFPU 44.94%, MATH union 68.80%**. Mean-core values imply about **18.11 percentage points of FPU/SFPU overlap** (41.93 + 44.85 - 68.67). The prior perf-v3 profile was FPU 31.8%, SFPU 35.87%, MATH 67.67%, with essentially no overlap. These are measured activity counters, not achieved FLOP/s divided by peak, and not a full-256K counter measurement. Raw data: [final-profile.log](final-profile.log) and final-profile/.

[Timestamp validation](final-counter-timestamps.jsonl) confirms zero reference-counter wraps on all 110 active cores. Its separate full_chip_pct fields use the profiler header's 120-core denominator; the utilization quoted above is the active-core median and should not be conflated with those fields.

Streaming materially increased overlap, but substantial serialization and non-math time remain. Reaching 3.32 s still needs another 7.82% elapsed-time reduction. The tested polynomial, chunk, tiling, and throttling tweaks did not close that gap without losing accuracy or speed. A more fine-grained pipeline/state-update redesign remains a plausible next investigation; the failed aliased-state prototype is not evidence of a valid speedup or a fundamental limit.

## Accuracy and regression checks

[validate.sh](validate.sh) completed **22 results**: three normal seeds; seven spread-sampled distributions; six common-mode cases; BF16 streaming; causal and D=64 fallbacks; and 32K/64K/128K lengths. Finite output, ordinary/trace output equality, and FP64 reference self-checks are enforced by the repro. The 0.5% gate applies to normal seeds and lengths, not to known stress failures.

Three additional K-chunk=512 seed checks passed after the final forced rebuild: **0.488753%, 0.489911%, 0.487661%**, bringing the correctness suite to **25 results**. They are included in the updated validation script. Raw combined results: [final-accuracy-all.jsonl](final-accuracy-all.jsonl); the extra original files are final-k512-1234/1235/1236.jsonl. These are sampled-device correctness checks, not full-operation timing.

| Spread-sampled distribution, seed 1236 | L2 % | Raw maximum relative error % |
|---|---:|---:|
| uniform_constant_v | 0.00000 | 0.00000 |
| common_q +8 | 1.14972 | 2.27545e+6 |
| common_q +32 | 2.75634 | 119626 |
| common_k +8 | 0.570001 | 9092.96 |
| common_k +32 | 1.11865 | 89200.6 |
| common_v +8 | 0.0406054 | 0.196375 |
| common_v +32 | 0.0105102 | 0.0513371 |
| normal | 0.487760 | 103514 |
| scaled_qk | 0.851475 | 112817 |
| outliers | 0.648438 | 409242 |
| biased_v | 0.173003 | 0.397695 |
| uniform | 0.171883 | 9.56690 |
| constant_v | 3.62115e-14 | 1.11022e-13 |

Maximum elementwise relative error is unbounded near reference zeros; the JSON also reports the maximum restricted to reference magnitudes above 1% of reference RMS. Constant-V outputs are exact BF16 one (the tiny residual is reference arithmetic).

32K/64K/128K L2: **0.491996%, 0.488898%, 0.484609%**. BF16 streaming, with no Q preprocessing, remains **3.162034261959%**, exactly matching perf-v3's corresponding spread-sampled result. Causal and D=64 fallback L2 are 0.108631% and 0.481489% respectively.

Large Q/K offsets remain a numerical limitation, not a performance regression fix. Outlier L2 improves from perf-v3's 0.683394% to 0.648438%, but still misses 0.5%.

## Rejected and diagnostic experiments

Short-Q/all-core screenings are **not** full 256K timing results. The all-core proxy uses Q=1408 (one 128-row chunk per core) or Q=2816 for Q chunk 256; 2000 warmups/20 measurements. It does not reproduce the full operation's sustained timing exactly.

| Experiment | Observation |
|---|---|
| Q chunk 256 / K chunk 512 | 39.746 ms for twice the proxy Q work; slower per query than 18.508 ms for 128/1024 |
| Q chunk 128 / K chunk 512 | 20.378 ms all-core proxy; slower than 128/1024 |
| MM_THROTTLE=1 | 20.034 ms proxy vs 18.508 ms control, same output; not retained |
| Two-row PV tiling | 18.536 ms proxy; no useful improvement |
| Two-row QK tiling | 19.734 ms proxy; slower |
| Quadratic exp replay, both matmuls HiFi2 | Short-Q 15.961 ms, but L2 **0.526820%**; rejected on accuracy |
| In-place aliased numerator/denominator state | Incorrect across K chunks, including attempts to repair synchronization/initialization; rejected |
| Mixed HiFi3 PV + quadratic exp | Correct, but did not beat HiFi2 cubic; not retained |
| Original raw fused exp; rolling PV scheduling | Slower |
| Smaller core grids | Slower full-128K timings; no useful power-limit benefit |
| FP16A score / BF16 P alias | Rejected by JIT mixed-exponent-format validation; no global JIT changes made |

Early v4 JSON files incorrectly recorded streaming=false because the repro metadata predated the new path; their labels and source variants identify FP32 streaming. New runs use --record-fp32-streaming, which records provenance only and does not select a kernel. The mixed-HiFi3 experiment's JSON configuration fidelity remains HiFi2; its PV kernel override is explicitly experimental and not retained.

The first fresh-main attempts (main-w40, main-k1024, main-fresh-short) are **invalid controls**: restoring the older snapshot with tar preserved older source timestamps, so the transformer host object was not rebuilt even though kernel sources changed. They produced all-zero output. A new isolated kernel cache did not fix this; forcing the host translation unit to rebuild did. The main-build log shows only a core version-object rebuild, whereas main-forced-build shows the transformer object being compiled. Only main-forced results are valid comparisons. Snapshot restores now explicitly force recompilation; [measure.sh](measure.sh) implements that precaution. The prior passing candidate runs used newly edited source timestamps and are unaffected.

[inplace-rejected.patch](inplace-rejected.patch) preserves the last failed aliased-state prototype as a full delta from main, for follow-up debugging. It is not the retained implementation.

## Reproduce / review

Use the existing built remote repository and environment documented in the root report:

```bash
python_env/bin/python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \\
  --kv-lens 262144 --full --heads 10 --q-chunk 128 --k-chunks 1024 \\
  --variants fp32_hifi2 --q-round-bits 6 --q-bitceil --q-prescale 1.0027 \\
  --seed 1236 --record-fp32-streaming --max-l2-pct 0.5 \\
  --benchmark-warmup 40 --benchmark-iters 10 \\
  --label final-recheck --output experiments/sdpa-l2/perf-v4/final-recheck.jsonl
bash experiments/sdpa-l2/perf-v4/validate.sh recheck
```

For a snapshot restored with preserved timestamps, use `bash experiments/sdpa-l2/perf-v4/measure.sh fresh-label` to force the host rebuild before the same full measurement. `TT_METAL_CACHE` can select a separate cache without deleting the existing one.

[incremental-fp32.patch](incremental-fp32.patch) is the delta from the prior retained perf-v3 source; [final-candidate.patch](final-candidate.patch) is the complete four-file delta from main, including the previous BF16 work. Both pass reverse-apply checks. No commits were made.

The incremental change is three C++ files, 483 insertions and 22 deletions. C++ changed hunks were clang-formatted; the repro passes Black 23.10.1, Python syntax checks, and git diff --check. [source-sha256.txt](source-sha256.txt) matches the restored remote source. This remains an experimental specialization, not a general SDPA acceptance claim; nondefault RISC L1 data-cache configurations have not been validated.

Handoff: the remote machine is left on the retained candidate, forcibly rebuilt and tested, not on main or a rejected variant. All v4 raw logs, profiles, patches, and JSON results are copied to the local worktree. The final independent run used the isolated cache /localdev/cglagovich/sdpa-perf-v4-cache.QCDd6Y; the existing cache was not removed. Reservation 214149 remains allocated and expires around 2026-09-10 04:04 UTC (00:04 Eastern), unless extended separately.
