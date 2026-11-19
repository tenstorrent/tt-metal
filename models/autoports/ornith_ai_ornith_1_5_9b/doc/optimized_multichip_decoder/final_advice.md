# Final profile findings and advice disposition

All five advice-enabled tables describe production `b1d5016dd769ab65fce82e27d483f73a33fa2e746e29e21952d521052948d2d8`:

- `tracy_final_v2/0/{prefill,decode}_perf_report.txt`
- `tracy_final_v2/3/{prefill,decode}_perf_report.txt`
- `tracy_final_v2_target/stack32/decode_perf_report.txt`

Each has CSV, source-CSV SHA256, command and measurement provenance. Watcher was disabled for profiles. Clean timing and profiler timing are separate.

## Target decode profile

The B32 two-kind stack at positions 4115–4118 has 55 operations per rank. The final timed trace is replay session 6: four validated replays, one profile warmup, then one timed replay. Raw host timestamps remain the capture timestamps, so replay identity—not a raw host-timestamp filter—selects its runtime rows. `selected_runtime_rows.json` preserves the actual attributes and kernel sources.

| Family | Merged kernel time, μs |
|---|---:|
| Ten DRAM-sharded matmuls | 256.293 |
| Paged SDPA | 210.024 |
| Ordered recurrence plus normalization/gating | 118.119 |
| Partial RMSNorm/RoPE | 22.191 |
| Convolution/gate preparation | 12.885 |
| Four async all-reduces | 40.957 |
| All 55 operations | 771.478 |
| Reported inter-operation gaps | 46.060 |

The report chooses the slowest rank for ordinary operations and averages ranks for CCL. Therefore 817.538 μs (kernel sum plus gaps) is report accounting, not an exact critical path. Independent per-chip FW spans, using the captured 1350 MHz clock, are 812.335/808.259/806.976/803.181 μs. Same-run host time is 880.320 μs; host minus maximum device span is 67.985 μs, including API/synchronization and profiler overhead. This remainder is not classified as a TTNN limitation.

The same-run useful-bandwidth floor is 0.430016 ms, versus profiled host 0.880320 ms. The clean primary workload ends at positions 5135–5138 and has its own floor/measurement in `final_roofline.json`; mixing those positions with this profile would be incorrect.

Runtime rows show BF16 inputs, BFP8 weights and BF16 outputs for all ten matmuls: four attention matmuls use HiFi2, six MLP matmuls use LoFi. Raw program attributes show two workers per DRAM bank for packed input and gate/up, four for output/down. The report's Cores=8 field is not workers per bank. Its >100% FLOPs estimates for some multiworker rows are not used as hardware-utilization claims. The modeled-ops DRAM percentage (27.1%) omits important custom/SDPA traffic and is not the whole-path stop metric. See `final_profile_summary.json`.

## Advice and evidence

| Advice | Trial or disposition |
|---|---|
| Target recurrent GenericOp has an 8.859 μs preceding gap; tracing or compile-time arguments could save about 3 μs | This path is already traced. `compile_args_jobs.json` tested runtime-to-compile-time arguments against the fast-zero helper: long-answer 0.863355864 vs 0.863660616 ms, about 0.035%; no material gain. The final fused helper's additional argument specialization was not remeasured after the stop rule and is a handoff, not a measured rejection. |
| Prefill host gaps: trace fixed-shape inputs | `prefill_trace_0.json` and `prefill_trace_3.json` test fresh inputs/page tables, state checks and nonaligned lengths through 8193 under allocation tracking. At S1024, earlier eager/traced measurements were 2.059/2.001 ms linear and 2.175/2.129 ms full. Caller-managed prefill tracing is a documented handoff; the default prefill API remains eager. These earlier trials are not final-default timing claims. |
| Increase packed-input matmul active grid beyond 88 linear / 80 full cores at S1024 | Precision-matched geometry and complete-layer trials are in `phase_prefill_matched_jobs.json` and the prefill sweep records. The selected program grids are 11×10 and 11×8; active cores are lower because of M/N block ceilings. Further M/N padding to occupy every configured core was not measured after the stop rule. Its ideal gain before padding overhead is at most roughly 29–35 μs per layer for these rows; actual gain is unknown. Carry it, do not call it rejected. |
| SLOW output/down projection rows; blocks/subblocks already look good | Role-specific geometry/fidelity sweeps and the phase-input/output/both complete-layer comparisons select K16, 8×8 and subblock 1×2. Higher-fidelity accuracy suggestions are covered by precision trials and the unchanged real-weight/recorded-HF gates; they are not evidence that a slower policy should replace the passing selected one. See `prefill_precision_jobs.json`, `precision_long_jobs.json`, `recorded_precision_jobs.json`, `recorded_long_precision_jobs.json` and the work-log search tables. |
| MinimalMatmul says no program_config specified | The op uses its own explicit `MinimalMatmulConfig`, not the generic matmul program-config field inspected by the report. Production supplies M4/N8/K16 and grid11×10; sweep evidence varies these actual controls. Further exhaustive large-prefill configurations are listed under the stop-rule handoff. |
| All ten decode matmuls marked optimized | Preserve the measured per-role configs; this label does not certify whole-path roofline. |

The eight-hour search stop fired before these final profiles. Remaining performance ideas above are handoffs under that rule, not failed quality gates. The later 128K accuracy failure was investigated and repaired despite the timebox; its unchanged gates now pass.
