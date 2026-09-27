# Stage08 datatype sweep work log

Target: google/gemma-4-26B-A4B-it at revision
4d7ae4984b7db7de8f8457170b3f1a419ee76d52. Starting checkout
0cfab7802b, branch gemma-4-26b-a4b-it; clean tracked worktree at startup.
Four Blackhole P300c ASICs are available and listed by
`timeout 60 tt-smi -ls --local`; selected mesh is 1x4. No reset was needed.

## Contract and baseline refresh

Acceptance: full-model prefill and traced teacher-forcing top1 >= 90%,
top5 >= 98%, top100 = 100%, on the existing 100-position AIME24 chat reference.
No context reduction is permitted without physical-limit evidence; inherited
supported context is 262144. B1 capacity is distinct from B32 short prompts.
Teacher-forcing ranking and 4096/128 B1 C1 token-out headline are separate.

Baseline command:
`OMP_NUM_THREADS=8 timeout 1800 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_full_model --output-dir models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/baseline`
Output: `baseline/run.log`; readiness, shared qualitative checks and warmed
streaming/buffered token-out measurements are produced by the same Stage07 runner.

## Precision propagation remediation

AutoFix source-only diagnosis is delegated to a fresh xhigh agent, with hardware
access forbidden while baseline runs. See `AUTODEBUG_precision.md` for verified
constructor gaps and `AUTOFIX_precision.md` for subsequent validation. The
selected artifact must affect real runtime objects, not only metadata. Stage09
vLLM implementation is excluded; later adapters must call the normal constructor.

`tests/run_datatype_candidate.py` evaluates an explicit policy, saves actual
runtime summaries, checks all 30 layers and 100 continuation positions, requires
99 model/sampling trace replays, and separates first-run from three warmed
teacher-forcing requests. No eager timings enter ranking.

Baseline refresh exited 0: prefill 96/100/100%, traced decode 94/100/100%, 99 model and sampling replays. Decode 49.9878 t/s/u on 161/100 readiness. Warmed 4096/128 B1 C1 buffered token-out: median TTFT 2115.2172 ms, decode 49.35446 t/s/u; all output tokens equal streaming controls. Shared qualitative suite has exact token equality against Stage07 for all six prompts; HF chat-format controls are inherited unchanged. Exact artifacts: baseline/readiness.json, baseline/performance.json, baseline/performance_comparison.json, baseline/qualitative_tt.json.

## Construction audit checks

`run_datatype_smoke --config .../configs/baseline_mixed_lofi.json --output .../smoke_baseline.json --lengths 33` exited 0, validating the actual layer0/layer5 runtime objects and non-aligned generation. Host `pytest -q .../tests/test_precision_policy.py` initially passed all eight schema/override/default-loading/mismatch tests. Formatting found the repository expect_error convention; test code was adapted accordingly. Full baseline reconstruction uses TT_METAL_TRACE_ALLOC_TRACKING=1. Initial smoke allocation warnings are inherited split-trace lifecycle warnings; allocation tracking in the all-layer check is the discriminating corruption check.

The first reconstructed baseline attempt is retained in
`diagnostics/baseline_tracker_accumulator.{json,log}`. Its first complete
100-token pass agrees at 94/100/100% with 99 traced model/sampling replays
under allocation tracking. Tracking slows host submission substantially and is
excluded from ranking. The second repetition exposed a runner bug: TokenAccuracy
retains predictions across calls, so a reused collector reported 200/100 tokens.
The runner now creates a fresh collector for each repetition. The generator
correctly made 100 callbacks; this was an evidence-harness accumulation error.

AutoFix also removed redundant same-dtype weight casts that allocated duplicate
BFP8 buffers. New baseline and candidates use the final alias-preserving path.
The diagnostic attempt does not enter the Pareto matrix. Ordinary performance
runs disable tracker/watcher/profiler instrumentation and retain trace counters.

Final-source reconstructed baseline passes all four repeated traced readiness
requests: top1/top5/top100 =94/100/100%, prefill96/100/100%. Median warmed
teacher-forcing decode50.1086455 t/s/u. `results/baseline_mixed_lofi.json`
contains all runtime tensor/kernel policies, source hashes and per-repeat metrics.

Pure-host memory accounting in `tests/precision_memory.py` reconciles the
inherited decoder resident10,986,081,280 B/device and peak28,467,973,120 B/device.
`memory_candidates.json` records candidate deltas including retained prefill
copies. BF16 cache conservative peak36,017,720,320 B exceeds the32GB planning
budget; this is not proof of physical impossibility and does not authorize
reducing context. No candidate is selected from that estimate alone.

The coarse matrix runs each config in a separate process and serialized mesh lifecycle. For each ID the exact invocation is `OMP_NUM_THREADS=8 timeout 1200 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_datatype_candidate --config models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/configs/ID.json --output models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/results/ID.json`; allocation tracking, Watcher and device profiling are unset. Matching `.log` retains console output. IDs are baseline_mixed_lofi, head_bfp8_hifi2, head_bfp8_lofi, head_bfp4_lofi, head_bfp4_hifi2, inner_bfp4_lofi, inner_expert_gate_bfp4_lofi, inner_qkv_bfp4_lofi, inner_output_bfp4_lofi, shared_down_bfp4_lofi, activation_bfp8, ccl_bfp8, mixed_hifi2, decode_bfp8_hifi2, decode_bfp8_lofi. Results, not this planned list, establish execution.

## BF16 cache long-prefill remediation

`run_datatype_smoke --config .../configs/kv_bf16.json --output .../smoke_kv_bf16.json`
passed31/32/33/1023/1024/1025, then4097 failed before device execution:
SDPA static CB end1,590,848 B exceeds Blackhole L1 limit1,572,864 B.
Fresh AutoDebug/AutoFix diagnosis: `AUTODEBUG_bf16_cache.md`. Exact factory
allocation arithmetic matches the error; BF16 K/V double buffers need a smaller
K chunk for512-wide heads. A conditional use of the existing Q64/K128 program
reduces the bound to1,033,792 B, without changing BFP8 selection or logical
prefill chunking. Host-lowered before/after cases establish BFP8 equivalence.
Device confirmation remains required and is queued after the serialized matrix.

Interim results: BFP4/LoFi head97/100/100% and51.42739 t/s/u; BFP4/HiFi2 head97/100/100% and51.33303. All-inner BFP4 QKV/output/expert-gate trial92/100/100%,50.67938; isolated inner expert-gate BFP4 preserves94/100/100%,50.46543. These are candidate teacher-forcing results, not final selection or token-out numbers. Per-repeat accuracy counts are stable. Head fidelity timing separation is near noise (one HiFi2 sample has a roughly2% timing outlier), so final combination/default verification is required.

Prefill shared-MLP `library_default` means the existing closure calls `ttnn.linear(x, weight)` with BF16 inputs/weights and no grid/program/compute override (`models/demos/gemma4/tt/shared_mlp.py`). Current `create_matmul_attributes` in `ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp` resolves that exact signature to HiFi2, approximation=false, FP32 accumulation=false, L1 accumulation=true. This fixed prefill policy is unchanged; the selected runtime summary separately proves the configurable shared decode fidelity.

Remaining work after the initial matrix: repaired BF16 cache smoke including
2047/2048/2049 and4095/4096/4097; full kv_bf16, inner_bfp4_hifi2,
shared_down_bfp4_hifi2 and decode_bf16_hifi4 controls; combinations
head4_inner_gate4, head4_inner_qkv_gate4 and head4_inner_all4, plus surviving
activation/CCL/shared choices when material. No winning artifact exists yet.
Final selected policy must run the normal-default optimized full-model runner,
shared qualitative suite, non-aligned checks and maximum-context capacity check.
`run_full_capacity.py --output-dir` now permits fresh evidence without overwriting
Stage06 artifacts. Independent clean-pass review and local commits remain pending.

## Precision-matched head geometry

Ran `OMP_NUM_THREADS=8 timeout 600 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_full_lm_head --fixture models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/terminal_input.pt --output models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/head_bfp4_lofi_geometry.json --interleaved-only --weight-dtype bfloat4_b --fidelity LoFi --grids 11x10 11x8 8x8 --blocks 4 8 11 22 44 88 --rounds 5 --replays 20`.
Exit0. The real target weights and recorded real reduced-model input compare
same-precision complete terminal paths. 11x10/K4 is fastest at389.785 us;
11x10/K8 is399.207 us. 11x10 permits K88 but is slower; smaller grids reject
some larger blocks for exact L1 bounds. All executed geometries pass the same-
precision PCC check. This is isolated host-wall traced timing, not full-model
performance or device time. Keep the existing geometry; no runtime geometry
change is introduced. `head_bfp4_lofi_geometry.{json,log}` contains all19 rows,
including the K8 helper baseline and18 explicit geometry cases.

The second serialized batch uses the same candidate command template for:
kv_bf16, inner_bfp4_hifi2, shared_down_bfp4_hifi2, decode_bf16_hifi4,
head4_inner_gate4, head4_inner_qkv_gate4 and head4_inner_all4.
The target has no named canonical accuracy/performance entry in
`models/demos/gemma4/precision_overrides.json`; uniform BFP8 LoFi/HiFi2 and
BF16/HiFi4 decode controls are the closest recovery policies under the accepted
optimized TP4/EP4 architecture. Prefill remains explicitly fixed in those controls.

## Completed recovery and combination controls

The repaired BF16 cache smoke exited0 and all12 boundary lengths passed;
`smoke_kv_bf16_fixed.{json,log}` closes the earlier pending device confirmation.
Full BF16-cache readiness passes94/100/100% at50.06457 t/s/u. Inner BFP4+HiFi2
passes92/100/100% at50.39578; shared-down BFP4+HiFi2 passes95/100/100%
at49.84240; uniform BF16/HiFi4 decode passes98/100/100% at44.40083.

Head4 plus inner gate/up4 passes95/100/100% at51.97184; adding inner QKV4
passes97/100/100% at52.06739; adding inner attention-output4 passes94/100/100%
at52.12325. All are warmed traced teacher-forcing medians, not token-out.
The final compatible combination also sets shared-down toBFP4/LoFi:
`head4_inner_all4_shared_down4`, using the same command template and regime.

The user explicitly requires the fastest evaluated passing configuration.
Selection therefore uses the maximum measured median, with close timing
separations disclosed rather than claiming statistical significance or a
universally optimal precision. Rejected activation8/CCL8 trials were slower;
no recovery to a slower higher precision is justified by these passing gates.

## Final selection and default-path validation

The final shared-down combination passes97/100/100% prefill and94/100/100%
decode, with median warmed traced teacher-forcing TTFT149.2465697 ms and
52.30587745 t/s/u. It is the fastest of23 evaluated passing configs.
`selected_precision_config.json` is an exact copy of its complete policy.
No precision override is supplied to the final runners: normal factory loading
must reproduce this artifact. The host policy regression tests pass8/8 after
repository-fixture adaptation (`policy_tests.log`).

Final token-out/readiness/quality command:
`OMP_NUM_THREADS=8 timeout 1800 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_full_model --output-dir models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/selected`.
Output: `selected/run.log`, `readiness.json`, `precision_runtime.json`,
`qualitative_tt.json`, `autoregressive/`, `performance_comparison.json` and
`performance.json`. The runner now asserts the existing90% top1 gate as well
as top5/top100; it does not weaken the candidate gate.

Default-path final runner exited0. Readiness reproduces prefill97/100/100%
and decode94/100/100%; first-request traced decode52.18254 t/s/u. Warmed
post-selection autoregressive token-out at4096/128/B1/C1 is51.64728059 t/s/u
and2115.2476519 ms TTFT, median3. Relative to refreshed baseline49.35445763,
decode improves4.65%; TTFT is unchanged within measurement variation. This
post-selection token-out result is the headline for later reports/vLLM comparisons.
Counters prove127 model, sampling and output replays; one final token readback,
zero full-logit readbacks and no per-token token/position/cache-position refreshes.
Streaming and buffered sequences match exactly across repeated requests.

All six shared-suite selected outputs differ from baseline tokens but remain
coherent under manual HF/baseline comparison; exact outputs and specific verdicts
are in `selected/qualitative_verdict.md`. The sky-explanation degeneration checker
exits0 with no findings. Command:
`python models/common/readiness_check/check_degenerate_output.py models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/selected/autoregressive --missing-artifacts critical --scope autoregressive --json models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/selected/degeneracy.json`.

`TT_METAL_TRACE_ALLOC_TRACKING=1 OMP_NUM_THREADS=8 timeout 900 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_datatype_smoke --output models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/selected_nonaligned.json --lengths 31 32 33 1023 1024 1025 2047 2048 2049 4095 4096 4097`
exited0, all12 cases passed. The instrumented two-layer shape/trace test is not a
performance result. Both layer types report the selected default policy.

The all-layer batch32 command is
`OMP_NUM_THREADS=8 timeout 1200 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.check_full_trace --batch 32 --all-layers --output models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/selected_trace_batch32.json`.
The helper now records its actual precision summary to tie this capability check
to the default artifact.

All-layer batch32 exited0: mixed31/127-token prompts, persistent feedback and
positions pass; isolated slots0 and31 have exact1.0 prefill/decode/second-decode
PCC and matching top5. `selected_trace_batch32.json` records30 layers and actual
selected-policy runtime settings. `selected/source_manifest.json` proves the
runtime source hashes and full policy equal the winning candidate's records.

The throughput workload's periodic nine-token continuation is controlled:
all128 selected tokens exactly equal the refreshed baseline on the same repeated
4096-token synthetic input. `selected/performance_output_control.json` records
anomaly, affected path, control, investigation and resolution. It is not used
as a text-quality claim. Prompt-correct shared-suite checks remain separate.

Maximum-context command:
`OMP_NUM_THREADS=8 timeout 1800 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_full_capacity --output-dir models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/selected_capacity`.
All device jobs are serialized. `pre-commit run --files <stage-owned files>`
passes after normalizing helper formatting and completion-text whitespace;
original generated token IDs remain in JSON. Python-only changes require no
C++ build. Logs are retained locally; compact result JSON/source files are
versioned evidence.

Review also classified an initialization warning in decode_bf16_hifi4,
head4_inner_all4 and shared_down_bfp4_hifi2: AICLK settled at1343MHz versus
requested1350MHz (within5%, runtime proceeds). This0.52% difference reinforces
that sub-percent median ordering is not statistically established. No sustained
clock measurement is claimed. The winner is the maximum observed passing
median, as explicitly requested, not proof of a hardware-noise-free ordering.


Maximum-context runner exited0. All30 layers pass262143-token nonaligned
prefill plus final supported position262143 decode, and262144-token aligned
prefill with finite logits. Elapsed host check times150.976/150.499 seconds are
capacity-test durations, not headline performance. The selected runtime summary
confirms all30 cache pairs useBFP8. Context contract recomputed: conservative
peak26,652,351,488 bytes/device, unchanged262144 context, no alignment restriction,
B32 short-request coverage retained. Final `timeout 60 tt-smi -ls --local` exits0
and enumerates the same four Blackhole P300c ASICs. No resets were needed.

Fresh independent xhigh stage reviewer `/root/datatype_stage_review` audits the
live worktree against the original stage contract; final verdict is pending.


## Independent review and checkpoint

Fresh xhigh `/root/datatype_stage_review` returned **clean-pass**, no required
work. `stage_review.md` independently rederives23 candidate aggregates, checks
actual runtime policy/source propagation, default token-out, chat controls,
full capacity, memory accounting, plots and the exact-template local telemetry
packet. All observed anomalies are classified. Stage08 is ready for local
checkpoint; no vLLM work and no push.


Local implementation/evidence checkpoint: repository `/workspace/tt-metal`,
branch `gemma-4-26b-a4b-it`, commit **`e9d0a0a5f9584060e4b4b905c1f2c6621086815d`**.
All commit-time pre-commit hooks pass. The required CSV is explicitly tracked
despite the repository's general CSV ignore rule. No unrelated changes are
included and nothing was pushed. This provenance-only follow-up records that
checkpoint; its final HEAD is logged with both SHAs in the local telemetry packet
and run-level `datatype_sweep_checkpoints.json` (under the telemetry directory).
