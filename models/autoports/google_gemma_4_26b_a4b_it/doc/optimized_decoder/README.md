# Gemma 4 26B A4B optimized decoder

Status: complete; the independent remediation rereview returned `clean-pass`.

This stage owns `tt/optimized_decoder.py`, `tests/test_optimized_decoder.py`,
and this directory. `OptimizedDecoder` inherits the completed `FusedDecoder`
and owns both runtime entry points; the tests reject a functional alias or
fallback. No multichip, full-model, LM-head, sampling, or vLLM work is included.

The selected single-device Blackhole policy uses BF16 attention and dense
weights, BFP8 experts, packed dense gate/up, fused router scales, exact sparse
grids, layer-specific DRAM-sharded QKV, and a batch-32 DRAM-sharded packed
dense projection. Prefill MoE is tiled internally in logical 32-token chunks;
one packed 1408-wide sparse gate/up uses an exact 11x4 grid, while down uses
11x8. Both use K block 11 and L1 outputs.
The public API retains arbitrary logical sequence lengths and the inherited
padding/slicing contract.

## Correctness and context

The unchanged PCC threshold is 0.995.

| Layer/cache kind | Prefill PCC | Decode PCC | Result |
| --- | ---: | ---: | --- |
| sliding/shared | 0.998599 | 0.999521 | pass |
| full/natural | 0.997994 | 0.999804 | pass |
| full/shared HMA | 0.997994 | 0.999804 | pass |

The prefill change from the prior 0.998634/0.998006 bar is -0.000035 sliding
and -0.000012 full, with substantial margin above threshold. Both layer kinds
pass logical lengths 1, 31, 32, 33, 63, 65, 127, 129, 1023, and 1025. Bounded
modulo-tail integrity, mutable A/B/A trace buffers, batch-1 and batch-32 trace
replay, and three repeated traces all pass. Advertised-position decode passes
at token 262143, so `doc/context_contract.json` is unchanged.

Attention BFP4/LoFi fails real weights (sliding prefill/decode
0.969190/0.979630; full 0.988143/0.979503). Expert BFP4 passes aligned cases
but fails logical length 31 at PCC 0.994344. An adapted 88-core sharded RMSNorm
also fails at 0.993390 sliding and 0.990832 full. These results retain BF16
attention/dense, BFP8 experts, and HiFi4 sparse gate; sparse up/down use LoFi.

## Warmed performance

All values below are host medians. Decode is 50 warmed trace replays at
sequence/current position 1024. Batch-1 prefill is 50 warmed seq-1024 samples;
batch-32 prefill is one warmed full batch. The decode baseline is the best
correct frozen-source policy before the new DRAM roles, also measured for 50
replays.

| Layer | Mode | Batch | Before (ms) | Final (ms) | Change |
| --- | --- | ---: | ---: | ---: | ---: |
| sliding | prefill | 1 | 680.575 fused | 82.473 | -87.88% |
| full | prefill | 1 | 681.700 fused | 83.621 | -87.73% |
| sliding | prefill | 32 | 21779.750 functional | 2637.948 | -87.89% |
| full | prefill | 32 | 21818.296 functional | 2675.148 | -87.74% |
| sliding | traced decode | 1 | 1.369657 | 1.351698 | -1.31% |
| full | traced decode | 1 | 1.538101 | 1.503703 | -2.24% |
| sliding | traced decode | 32 | 19.477006 | 19.444872 | -0.16% |
| full | traced decode | 32 | 19.343033 | 19.342580 | -0.00% |

The primary batch-1 target beats the best correct pre-AutoFix baseline for both
meaningful layer kinds; batch 32 does not regress either kind. The batch-1
baseline artifacts end in `_autofix_decode_baseline50.json`; current-source
batch-32 controls end in `_final_source_decode_baseline50.json`.

## Device profile and movement

Fresh natural-cache Tracy windows and `tt-perf-report` outputs are under
`tracy/current_fused_final/`.

| Window | Host (ms) | Device ops (ms) | Op gaps (ms) | Device+gaps (ms) | Modeled DRAM roofline |
| --- | ---: | ---: | ---: | ---: | ---: |
| sliding decode | 1.407 | 1.306 | 0.077 | 1.382 | 23.51% |
| full decode | 1.557 | 1.461 | 0.072 | 1.532 | 25.65% |
| sliding prefill seq256 | 21.560 | 21.161 | 0.299 | 21.459 | 59.08% |
| full prefill seq256 | 21.746 | 21.527 | 0.116 | 21.643 | 58.42% |

These are same-run host/device figures. The remaining host-minus-device/gap
overhead is 0.024/0.025 ms decode and 0.100/0.103 ms prefill. The exact modeled
bytes, 512-GB/s theoretical roofline times (0.306976/0.374560 ms decode and
12.501504/12.575232 ms prefill), equations, and readable op tables are indexed
by `profiler_accounting.md`. Full attention
uses its natural non-shared physical cache; sliding uses its natural shared
cache. Profiler invocation counters prove the optimized methods executed.

The measured forwards contain no Torch, `from_torch`, `to_torch`, or host
fallback. Device-side tilize/untilize operations remain at the required
row-major TopK/scatter, sparse routing, and tile-aligned packed-slice consumer
boundaries; their exact counts and microseconds are in
`profiler_accounting.md`. Sparse prefill replaces two same-input sparse
projections with one packed exact-grid operation and retains exact-grid down.

## Candidate decisions

- Batch-1 DRAM QKV width 1 is selected only for sliding attention; global
  width 1 fails full PCC (0.983679). Width 2 is selected only for full;
  global width 2 fails the stricter sliding real-weight case (0.993311).
- Batch-32 DRAM packed-dense width 4 is selected and trace-correct. DRAM QKV
  width 2 is selected only for sliding attention (trace PCC 0.999451); the
  generic/full form is rejected because full PCC is 0.984152.
  DRAM down width 3 is correct but its mixed 0.02--0.03 ms effect is not robust.
- Batch-1 DRAM packed widths 8 and 4 and dense-down widths 3 and 6 all regress.
- Prefill exact-grid separate gate+up+down with K block 11/L1 reduces seq-256
  from 168.959 to 32.093 ms. Packing gate/up further reduces it to 21.490 ms
  with unchanged PCC. A 64-token chunk regresses to 54.815 ms. The first
  128-token L1 config error was adapted to DRAM and ran correctly at about
  103 ms, so it is rejected by measurement rather than API failure.
- Large packed-dense prefill programs, BFP4 policies, sharded RMSNorm, and
  sparse-gate LoFi are retained in `candidate_matrix.json` as rejected evidence.

## Evidence and limitations

- `final_correctness.xml`: 14/14; `final_stress.xml`: 4/4;
  `final_perf.xml`: 4/4; `final_batch32_prefill.xml`: 4/4.
- `final_watcher_tests.xml` and `final_watcher.log`: 9/9 and no watcher
  error/assert/hang signature.
- `final_suite.xml`: 41 collected, 23 passed, 18 explicitly opt-in skipped.
- `operation_topology_audit.md`, `candidate_matrix.json`, `work_log.md`, and
  `AUTOFIX.md` retain topology, commands, rejected candidates, and remediation.
- Frozen source/test SHA256: `608da0656b1d4f0b8c3b3c812b032cfdcb6cd99631a32f1f3bb7cfa58a53a747`
  and `cc62897949aba36ec7019313ed81372bbff514d0cee3a4ca8322336a8267a5e6`.

Evidence is single-device Blackhole. Batch-32 MoE serializes independent
routing rows because the expert tensor owns the 128-expert batch dimension.
Raw Tracy traces are not retained; enriched CSV, compact reports, summaries,
plots, source-hashed JSON, and JUnit evidence are retained.
