# Ornith-1.5-9B optimized decoder

Stage 3 device gates pass. Independent [round 1 review](stage_review_round1.md)
returned `clean-pass` with no required work. Local commit provenance is recorded
in [work_log.md](work_log.md).

[OptimizedDecoder](../../tt/optimized_decoder.py) is a standalone single-device
implementation. The delivered [pytest](../../tests/test_optimized_decoder.py) runs
this class with real checkpoint weights and rejects diagnostic/fallback reports.
Target revision: `489cb97981b8654bcfcf30ce1f94ed1b62e07b53` of
`ornith-ai/Ornith-1.5-9B`. Hardware: one P300c Blackhole chip, ID 0, eight DRAM banks,
11x10 compute grid. All device jobs were serial. No resets, multichip, full-model
or vLLM implementation were performed.

## Final performance and accuracy

Matched fused baseline and final default, S1024, milliseconds per layer:

| Layer kind | Warmed prefill before → after | Traced warmed decode before → after | Decode reduction |
|---|---:|---:|---:|
| linear | 5.9326 → 3.1744 | 1.20773 → 0.64394 | 46.7% |
| full | 6.5435 → 3.6649 | 1.13024 → 0.62467 | 44.7% |

PCC is shown as output / residual contribution; both retain the 0.995 acceptance bar:

| Layer/phase | Fused | Optimized |
|---|---:|---:|
| linear prefill | 0.999920 / 0.999922 | 0.999650 / 0.999649 |
| linear traced | 0.999847 / 0.999810 | 0.999531 / 0.999498 |
| full prefill | 0.999843 / 0.999862 | 0.999458 / 0.999477 |
| full traced | 0.999893 / 0.999886 | 0.999518 / 0.999520 |

The minimum across the final correctness, watcher, fused-state comparison,
128-token trajectory and maximum-context suites is **0.997881**.
The modest PCC change follows BFP8 weight/cache quantization and MLP LoFi;
FP32 recurrent state and its HiFi4 compute remain unchanged. No gate was relaxed.
These are real-weight decoder tests with seeded BF16 hidden-state probes, not
full-model language/accuracy results. The HF reference uses CPU torch separately.

[final_performance.csv](final_performance.csv) includes S128,1023,1024,1025,8192;
[final_accuracy.csv](final_accuracy.csv) indexes the gates. Batch-32 traced latency
falls from 38.509 to
20.409 ms for linear and
35.037 to
19.375 ms for full attention.
This batch API executes independent rows serially in one trace, with mixed lengths,
starts and page tables; it does not claim a dense-batch speedup.

Timing uses seven warmed prefill samples and seven groups of 50 nonblocking decode
replays, with synchronization once per group. Weight loading, compilation, trace
capture, caller input/position refresh, state restoration and readback are excluded.
Repeated timing inputs/positions are fixed; changing-state correctness has separate
trajectory and trace-reuse tests. No model TTFT or token-feedback latency is claimed.

## Selected configuration

- All projection weights default to BFP8; activations and norms are BF16.
  Attention matmuls use HiFi2; MLP matmuls use LoFi. FP32 destination accumulation
  stays enabled. Recurrent math is HiFi4/FP32, with approximation disabled.
- Decode uses DRAM-sharded projections and a 64-core L1 residual/norm stream.
  Readers per bank: packed input 8, output 4, gate/up 6, down 2. Down uses K block 2;
  the others use block 1. Packed attention input beats tuned separate projections;
  separate gate/up beats packed decode MLP after activation/layout overhead.
- Prefill uses explicit minimal matmul M4/K16/N8, subblock1x2, grid11x10;
  the packed MLP additionally fuses SwiGLU. Legal adapted 2D kernels and L1 input
  placement were measured and lost. Default physical chunk size is 1024.
- Linear attention uses causal-convolution and GDN composites, FP32 recurrent-state
  aliasing during decode, row-major three-row history, direct untilize/unpadding and
  slice into the persistent history destination. Scratch does not escape forward.
- Full attention uses BFP8 paged KV, Qchunk32, Kchunk128 prefill / 256 decode,
  fused K/V updates on disjoint cores and a tiled RoPE lookup. All RoPE positions,
  page maps and state remain device-resident during forward.

The source contains no runtime torch/from_torch/to_torch or host fallback. Required
attention boundaries still use conversions; coherent native-grid, fused-head and
other lower-movement alternatives have measured before/after results in the work
log. Matmul's computed compact grid differs from the requested rectangle; retaining
the rectangular residual boundary is faster than the tested all-native layout.

## Context, traces and validation

The [context contract](../context_contract.json) remains **262144**. Full prefill
executes all tokens at 262139,262143,262144; reference checks use first/middle/end
windows with complete HF prior state/KV. The final token at position 262143 passes
traced replay. Short and medium tests compare every output row. Physical cache
capacity uses 32-token pages; logical sequence lengths have no chunk/page-alignment
restriction. State allocation capacity is page-aligned, as in the fused contract.

The final battery covers lengths 1,2,3,4,31,32,33,1023,1024,1025,2051,17,
nonzero-start continuation with a failing empty-state negative control, shuffled
page maps, batch32 and deterministic replay. It also checks state/output equivalence
to the fused decoder and advances 128 distinct tokens without restoring state.
Watcher runs use TT_METAL_WATCHER=10, separately from profiling. Allocation tracking
includes program-cache allocations and an intentionally failing positive control.
Only exact returned output tensors are acknowledged as corruptible; their values
are consumed before replay. No whole-forward/warmup exemption is used.

[final_audit.json](final_audit.json) verifies final artifact source hashes, numerical
bars, actual matmul precision rows, context and roofline. Production SHA256:
`b678803e78b74ede66d61e7609c6ca9147b2977dbba702f8431eb621ef997c4e`.

## Profiles, optimization decisions and stopping point

Advice-enabled reports and CSVs:

- [Linear prefill](tracy_final/linear/prefill_perf_report.txt), [decode](tracy_final/linear/decode_perf_report.txt), [prefill CSV](tracy_final/linear/prefill_perf_report.csv), [decode CSV](tracy_final/linear/decode_perf_report.csv).
- [Full prefill](tracy_final/full/prefill_perf_report.txt), [decode](tracy_final/full/decode_perf_report.txt), [prefill CSV](tracy_final/full/prefill_perf_report.csv), [decode CSV](tracy_final/full/decode_perf_report.csv).

[perf_summary.json](perf_summary.json) reconciles kernel time, op gaps and host time
from each same-run signpost. Instrumented single-replay host timing is separate
from unprofiled grouped replay. Header FLOPs/core utilization miscounts multi-reader
pipelines; raw durations and explicit byte accounting determine the conclusions.
Conservative S1024 roofline fractions are 71.90% linear and
70.39% full; batch32 fractions are
72.59% / 71.88%.
The last two kept decode changes gain 0.80% and 1.20% on measured batch32 ablations.
This satisfies Stop Rule 2; no quality requirement is waived.

[work_log.md](work_log.md) contains the initial topology audit, full checklist,
commands, adapted failures, rejected options and `Not tried (stop rule)` hand-offs.
[candidate_measurements.csv](candidate_measurements.csv) ranks whole-layer trials;
[matmul_geometry.csv](matmul_geometry.csv) contains 1,200 per-role BFP8/BFP4 cases.
Explicit prefill config/grid and L1 advice was tested after the first profiler pass.
Fixed-signature caller prefill tracing was measured; decode was already traced.
Dense MLP architecture makes MoE routing/expert optimization inapplicable.

BFP4/LoFi is faster but all-MLP policies fail PCC. Down-only BFP4 passes on the
linear representative and narrowly fails on the full-attention representative.
[Stage-8 hand-off](stage8_precision_handoff.json) retains every diagnostic outcome,
and [real BFP4 kernel rows](bfp4_profile/tracy_final/linear/decode_perf_report.csv)
prove the candidate precision reached execution. The requested default remains
BFP8; stage 8 and its P1 precision probe decide the full-model policy.

## Reproduction and retained evidence

```bash
bash models/autoports/ornith_ai_ornith_1_5_9b/tests/run_optimized_final.sh
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.audit_optimized_evidence
```

Use the configured bring-up environment and local pinned HF checkpoint. The final
selected-path rerun used `--selected-only` to reuse unchanged fused-baseline evidence.
Experiment commands and status are in queue manifests and queue_results.jsonl.
Raw logs/large CSVs have lossless `.gz` archives and SHA256 entries in
[archive_manifest.json](archive_manifest.json); raw Tracy captures remain local.
`pre_advice/` is explicitly superseded evidence with its own source snapshot.
The profiler kernel overlay consists only of byte-identical source symlinks.
