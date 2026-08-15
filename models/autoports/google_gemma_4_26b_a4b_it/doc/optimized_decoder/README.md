# Gemma 4 26B A4B optimized decoder

This directory records the single-device optimized-decoder stage for
`google/gemma-4-26B-A4B-it`. The stage is deliberately limited to one decoder
layer; it does not start multichip, full-model, or vLLM work.

## Selected runtime

`OptimizedDecoder` preserves the functional public API, logical shapes, paged
KV-cache ownership, trace mutability, and deterministic replay. The default
reuses the functional layer orchestration, while every material measured
projection/expert path dispatches an optimized override and is verified by
runtime counters and method-identity tests.

The selected policy is:

- BF16 activations, router, sliding-attention weights, and caller-owned KV
  cache; BFP8 full-attention, dense-MLP, and expert weights.
- HiFi4 sliding attention and LoFi full attention, dense MLP, and
  routed-expert matmuls.
- One packed same-input dense gate/up projection.
- Routed prefill in logical 32-token chunks with separate large full-chunk and
  legal tail sparse programs.
- Expert decode `in0_block_w=11`, `per_core_M=1`, `per_core_N=2`, with its
  input resident in L1.
- Both layer kinds use DRAM-sharded packed gate/up and down decode weights
  with `in0_block_w=11`, the fastest correct geometry that did not regress
  either measured batch.
- Both attention kinds use a DRAM-sharded O projection. Full attention keeps
  its precision-locked BFP8/LoFi policy; the advisor-challenger stage later
  established a strict batch-1 win for sliding attention's BF16/HiFi4 O path
  and shipped its measured role-specific `in0_block_w=1` geometry.

Tests assert optimized-method identity and material path counters, so a
functional fallback cannot satisfy the suite.

## Correctness and context

Every meaningful layer kind remains above the functional PCC bar of `0.995`.

| Case | Sliding attention | Full attention |
| --- | ---: | ---: |
| Real-weight prefill | 0.998545 | 0.996870 |
| Real-weight decode | 0.999617 | 0.999754 |
| Traced batch-32 aggregate | 0.999465 | 0.999754 |
| Traced batch-32 minimum user | 0.998390 | 0.999754 |
| Eager/replay and repeat replay | 1.000000 | 1.000000 |
| Lowest boundary prefill | 0.995281 | 0.996123 |
| Batch-2 prefill | 0.997983 | 0.998438 |

Coverage includes natural and shared paged-cache views, mutable A/B/A traced
buffers, repeated replay, and both representative layer kinds. Logical
lengths `1, 31, 32, 33, 63/127, 64/128, 65/129, 1023, 1024, 1025` pass without
a public divisibility restriction.

The existing 262,144-token context contract is unchanged. Traced decode at
position 262,143 passes both layer kinds with cache-history sentinels and
repeat PCC 1.0. A physical non-aligned prefill of 262,143 tokens also passes
both cache geometries. The selected optimization does not change the public
KV dtype, page geometry, or capacity, so `doc/context_contract.json` did not
need an update.

## Before and after performance

Measurements use one Blackhole P300C at sequence/current position 1024.
Prefill is warmed host latency; decode is warmed trace replay. The context
contract’s serving batch is 32, and batch 1 is the primary decode target.

| Workload | Functional | Optimized | Change |
| --- | ---: | ---: | ---: |
| sliding prefill, batch 1 | 680.885 ms | 95.194 ms | -86.02% |
| full prefill, batch 1 | 681.819 ms | 106.328 ms | -84.41% |
| sliding decode, batch 1 | 3.034 ms | 1.272 ms | -58.07% |
| full decode, batch 1 | 3.210 ms | 1.270 ms | -60.45% |
| sliding prefill, batch 32 | 21781.406 ms | 2995.823 ms | -86.25% |
| full prefill, batch 32 | 21818.454 ms | 3386.442 ms | -84.48% |
| sliding decode, batch 32 | 68.825 ms | 15.474 ms | -77.52% |
| full decode, batch 32 | 68.703 ms | 15.148 ms | -77.95% |

The final primary batch-1 result beats the clean functional baseline for both
layer kinds. Full attention also beats the prior correct optimized policy;
the unchanged sliding path is within 0.7% run-to-run variation of its best
candidate at both measured batches. Candidate-specific measurements and exact
environment overrides are under `candidate_runs/`.

## Candidate decisions

| Candidate | Correctness | Batch-1 / batch-32 evidence | Decision |
| --- | --- | --- | --- |
| packed dense gate/up | passes both kinds | improves cumulative decode | selected |
| full MLP DRAM block 1 | full decode 0.999812 | 1.421 / 15.293 ms | superseded |
| MLP DRAM block 11 | sliding/full decode 0.999617/0.999754 | final 1.272/15.474 and 1.270/15.148 ms with full O | selected both kinds |
| packed block 22 + down block 11 | passes both kinds | 1.261/15.508 sliding; 1.387/15.251 full | rejected; serving regression |
| packed block 22 + down block 3 | passes both kinds | 1.265/15.470 sliding; 1.420/15.253 full | rejected; no cross-batch win |
| packed block 22 + down block 33 | API accepts geometry | static CB end 1,455,936 overlaps L1 allocation at 1,370,112 | rejected; hard device L1 limit |
| gate/up BFP4 with down BFP8 | all real-weight cases miss PCC | precision-locked mixed trial | rejected |
| down-only BFP4 | prefill misses PCC; full decode reaches 0.995113 | precision-locked mixed trial | rejected |
| full QKV + selected MLP | prefill 0.996728, decode 0.968413 | current cumulative policy | rejected on decode PCC |
| full O + selected MLP | prefill 0.996870, decode 0.999754 | 1.266/15.124 ms candidate; 1.270/15.148 final rerun | selected for full attention |
| sliding O + MLP DRAM block 11 | 0.998380/0.999456 | 1.307 / 15.534 ms | rejected; slower than MLP-only |
| sliding O + final LoFi MLP/down | 0.997935/0.999520 | 1.274 / 15.464 ms | rejected; regresses primary batch 1 |
| all dense roles DRAM-sharded | full decode 0.985799 | 1.392 / 15.254 ms | rejected on PCC |
| attention roles DRAM-sharded | sliding 0.993140; full 0.985759 | isolated real-weight runs | rejected |
| QKV only DRAM-sharded | sliding 0.993052; full passes | full 1.487 / 15.345 ms | rejected |
| O only DRAM-sharded | sliding passes; old BF16/full trial 0.990663 | HiFi4 and block-1 retries also fail | superseded by precision-locked full BFP8/LoFi pass |
| coherent residual R11/R22 on final policy | sliding 0.994795/0.994694; full 0.999854/0.999857 | B1 1.378/1.347 ms; B32 hits L1 CB clash | rejected |
| MLP BFP4/LoFi on DRAM block 11 | sliding 0.979019/0.988925; full 0.989108/0.991148 | real-weight prefill/decode | rejected |
| MLP BFP8/LoFi on DRAM block 11 | sliding/full pass | 1.259/15.489 and 1.391/15.249 ms candidates | selected |
| sliding attention BFP8 | length-63 minimum 0.993506 | otherwise faster | rejected |
| sliding BF16/HiFi2 or LoFi | trace min-user 0.993700 / 0.948959 | prospective final MLP topology | rejected |
| full attention BFP8/LoFi | real, batch-2, boundary, and trace-32 pass | 1.389/15.257 ms candidate | selected |
| attention BFP4/LoFi | misses PCC | current cumulative policy | rejected |
| prefill experts BFP4/LoFi | sliding boundary 0.993363; full boundary 0.993806 | faster at seq 1024 | rejected |
| decode experts BFP4/LoFi | sliding batch-32 min-user 0.993908 | block-22 retry 0.993876 | rejected |
| KV cache BFP8 | sliding 0.998631/0.999396; full 0.998337/0.999807 | current cumulative policy passes | supported candidate, not selected because cache allocation/dtype is caller-owned BF16 |

The DRAM block-4/3 MLP trial hit the exact divisibility constraint that a
22-tile local shard width is not divisible by block 4. Legal block 11 was then
run end to end and selected; the API error was not treated as a rejection.
The DRAM program-config API exposes `in0_block_w` and per-core M/N but no
separate output-subblock control.

## Profiler conclusions

Final compact Tracy and advice-enabled `tt-perf-report` artifacts are in
`tracy_final_retry/` and `tracy_full_final/`; raw captures remain under
`/tmp/gemma4-tracy-sliding-final-v2-20260729` and
`/tmp/gemma4-tracy-full-final-v2-20260729`.

- Sliding decode totals 1.196 ms of signposted device operations: 23.96% dense
  matmul, 29.86% layer norm, 18.44% sparse matmul, and 3.54% SDPA. Its modeled
  DRAM roofline is 22.5% (115 GB/s).
- Full decode totals 1.203 ms: 18.34% dense matmul, including the selected
  width-sharded O and MLP ops, 30.43% layer norm, 18.21% sparse matmul, and
  4.44% SDPA. Width-sharded matmuls reach 39.19% weighted FLOP utilization.
  Modeled DRAM roofline is 19.5% (100 GB/s).
- Same-run traced host latency is 1.326 ms sliding and 1.324 ms full. The
  difference from summed device time is trace/dispatch and synchronization
  overhead.
- Prefill remains routed-sparse dominated: 78.31% sliding and 79.83% full.
- An approximate weight-plus-KV payload lower bound is 0.286 ms sliding and
  0.226 ms full at the P300C’s 512 GB/s peak. The larger observed device
  latency is consistent with sparse/low-utilization matmuls, norms, layout
  composites, and non-ideal DRAM utilization shown by the report.
- The measured path has no Torch conversion, host fallback, redundant reshard
  loop, or avoidable tilize/untilize round trip. Remaining movement is at
  attention-head, paged-cache, sparse packing/reduction, and logical-unpadding
  contract boundaries.

## Verification

- Normal suite: `18 passed, 12 skipped in 169.00s`; opt-in skips were exercised
  separately.
- Advertised context: `2 passed in 26.35s`.
- Non-aligned 262,143-token capacity: `2 passed in 183.46s`.
- Serving-batch prefill: `2 passed in 40.62s`.
- Final watcher run: `7 passed, 23 deselected in 102.83s`, covering real
  weights, batch-32 trace, and mutable A/B/A replay. The 2,171-line watcher log
  preserved as `final_watcher_device_log.txt` contains no
  error/assert/hang/timeout/illegal-NoC report.
- Post-watcher `tt-smi -s`: all four P300C devices healthy, DRAM status true,
  and zero corrected or uncorrected GDDR errors; the derived machine-readable
  result is `post_watcher_health_summary.json`.

The persistent `/dev/shm` capacity warning is an environment limitation; every
gate completed and device health stayed clean. See `work_log.md` for exact
commands, topology actions, checklist, review history, and local commit SHAs.
