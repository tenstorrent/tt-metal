# Stage Review

Verdict: clean-pass

Independent final review of stage07 optimized-full-model for Qwen/Qwen3.8-27B,
2026-10-25. Reviewed the live `exp26-qwen38-bringup` worktree based on
`6bcbce2d6e26a194e58be1fe23eef340d1a08dff`. This verdict accepts the inspected
implementation and evidence. The prescribed local checkpoint and its receipts
follow this review; their absence before review is not a missing gate. No push is
authorized or implied.

## Required Work

- None. The three required findings in `interim_stage_review.md` are resolved:
  the redundant validation wrapper is removed and all64 validation completes;
  tracker-enabled B32 all-active/mixed/all-active collector replay passes for
  greedy and sampled modes; and paired full-model EOS measurements support the
  revised16-step EOS window instead of the rejected128-step default.

## Other Concerns

- The measured throughput improvement is small: B1 increases from35.186165 to
  35.252433 tokens/s/user, about0.19%. All four final4K/128 groups have three
  repeats and matching baseline prompt/output hashes. This supports the reported
  measurements and removal of per-token readback; it does not establish a large
  speedup or a statistical confidence interval across independent runs.
- EOS-aware buffering remains a latency tradeoff. The exact47-token factual
  answer takes1739.872636ms streamed,4064.639178ms with128-step windows, and
  1797.595501ms with16-step windows, using paired three-repeat medians. The
  selected policy schedules49 outputs and costs57.722865ms/3.32% versus streaming.
  It can finish up to15 extra device steps after EOS. Explicit streaming remains
  available; fixed-length performance uses a128-step window.
- The original4096 repeated-text quality fixture still delays its final answer:
  TT exits reasoning at887 and truncates at1024, whereas HF begins prose much
  earlier. Every current token matches the accepted stage06 control. The nearby
  4095 case completes coherent prose at EOS892; the independent4096 factual case
  returns all three correct JSON facts at EOS47. This is a retained quality and
  answer-latency limitation, not HF parity or evidence of a new stage07 regression.

## Hard-Check Gaps

- The automated degeneration check covers the recognized AIME artifacts. It does
  not substitute for direct reading of the shared and long outputs. I read the
  HF/TT AIME, code, explanation, structured and long controls, and independently
  compared current versus stage06 token arrays for all shared and long cases.
- Full-model device time and rooflines are unknown. The accepted reduced profiles
  contain real layers0/3 plus the complete terminal/sampler/collector path, with
  synthetic history at decode position4223. Their measurements are correctly
  excluded from full-model device telemetry. Inherited layer host medians are an
  approximate guide, explicitly including per-layer replay overhead and predating
  the rotary correction; current reduced whole-layer envelopes provide separate
  attribution rather than a fabricated all64 device result.
- Native262144 execution/capacity evidence is inherited from the unchanged B1
  model/cache/position path. Stage07 freshly allocates that B1 context and exercises
  current full-stack smaller-context generation. This is not new native-context
  HF accuracy. B32's32768/128 workload has one measured request after warmup;
  its allocation-event audit uses4096-token prompts. Those scopes are disclosed.
- External ownership checks sample selected rows/pages on all four devices.
  Current reduced B32 preparation/replay checks complement explicitly inherited
  full64 external-pool capacity checkpoints. They are not a complete pool download.
- These limits do not conceal an unmet stage07 gate. Context and degeneration
  checks pass. The agentic checker has only six inherited weight-representation
  advisories, each with an explicit measured retention reason and byte accounting.

## Anomaly Ledger

- **Observed anomaly:** Readiness teacher forcing aborted because a wrapper hid
  the explicit `enable_trace` argument. **Evidence:**
  `full_validation_signature_failure.log`, final `full_validation.log`,
  `tests/run_full_validation.py`. **Affected path:** Validation orchestration.
  **Control or comparison:** The actual generator already exposes the required
  argument. **Likely subsystem:** Python signature introspection.
  **Investigation performed:** Inspected wrapper removal and completed prefill,
  teacher-forcing, AR/shared/long outputs with current runtime hashes.
  **Resolution:** Fixed; both accuracy phases have99/100 top1 and100/100 top5/top100.

- **Observed anomaly:** Initial collector evidence omitted the B32 all-active
  model trace. **Evidence:** `buffered_final_b32_transition.json`,
  `watcher_buffered_b32.json`, `tests/probe_buffered_output.py`.
  **Affected path:** Shared collector with multiple live model traces.
  **Control or comparison:** Exact streamed/buffered output for all-active ->
  mixed -> all-active at k1 and k8/p0.9; stable trace IDs and capture count.
  **Likely subsystem:** Trace lifetime coverage.
  **Investigation performed:** Read test ordering and actual records; tracking
  is enabled with program-cache allocations included. **Resolution:** Fixed.

- **Observed anomaly:**128-step EOS collection schedules129 outputs for a
  47-token answer. **Evidence:** `benchmark_b1_perf.json:eos_window_control`,
  `tt/generator.py::_generate_buffered_tail`. **Affected path:** Request latency.
  **Control or comparison:** Streamed,128-step and16-step exact-token arms.
  **Likely subsystem:** Output batching. **Investigation performed:** Recomputed
  medians and scheduled/returned counts; inspected bounded cursor resets and
  fixed-length separation. **Resolution:** Controlled by the measured16-step
  default; residual3.32% cost remains explicit above.

- **Observed anomaly:** Head candidates first failed optional input configuration
  and L1 capacity. **Evidence:** `head_c8192_r1_k5.log`,
  `head_c8192_r1_k5_s8.log`, `head_generic_k16.log`, `head_candidates.csv`,
  `tt/lm_head.py`. **Affected path:** Terminal optimization experiments.
  **Control or comparison:** Explicit input reshard, legal reader1/2/3 and larger
  K16/20/32 DRAM-sharded trials, plus split-interleaved K16/K20 controls.
  **Likely subsystem:** Common-head input contract and matmul geometry.
  **Investigation performed:** Inspected adaptation and19 successful real-hidden
  candidate rows. Whole terminal reshards/slices/concats are timed. Material
  alternatives are slower under the same BF16/HiFi4 policy. **Resolution:**
  Controlled, earned rejection; no first-error rejection or rejected decoder
  precision policy is substituted. The roughly3us genericK4 micro-improvement
  does not justify a claimed full-path gain and is not selected.

- **Observed anomaly:** First watcher launch exceeded the ACTIVE_ETH config
  buffer. **Evidence:** `watcher_inline_firmware_size_failure.log`,
  `watcher_inline_firmware_watcher.log`, `post_watcher_size_*`,
  `watcher_summary.json`. **Affected path:** Diagnostic firmware build/launch.
  **Control or comparison:** Established `TT_METAL_WATCHER_NOINLINE=1` configuration.
  **Likely subsystem:** Inline instrumentation size. **Investigation performed:**
  Read reset/list/ring-smoke completion and final B1/B32 watcher logs. Final14
  cases have exact outputs, tracking1, skip-program-cache0, watcher10 and no
  disabled Ethernet checks. Raw watcher logs contain no error/exception/overflow/
  invalid/corrupt marker. **Resolution:** Fixed without suppressing watcher checks.

- **Observed anomaly:** Generic fused-matmul fidelity columns say HiFi4 despite
  LoFi inner matmuls; B32 profile warns about unreplayed masked trace1.
  **Evidence:** Raw prefill/decode CSV attributes, `kernel_policy_audit.json`,
  `tracy/b32/scope.json`, `profile_b32.log`. **Affected path:** Profiler reporting.
  **Control or comparison:** Embedded compute configuration and signposted
  all-active trace0/buffered sampler3; separate masked correctness tests.
  **Likely subsystem:** Nested-op reporting and unused prepared trace.
  **Investigation performed:** Inspected BF4/LoFi runtime rows, head BF16/HiFi4,
  native prefillK4/K17 and decodeK6/K17/K24/K17; verified nonempty cycle data in
  every measured device row. **Resolution:** Controlled, correctly scoped.

- **Observed anomaly:** Delayed/truncated long-fixture output persists.
  **Evidence:** `default_long_quality_cases.json`, stage06
  `qualitative_verdict.md`, `benchmark_hf_control_512.json`,
  `long_quality_cases_hf.json`. **Affected path:** Quantized full-model text
  quality. **Control or comparison:** Exact prior-stage tokens and nearby/factual
  HF controls. **Likely subsystem:** Inherited numerical trajectory sensitivity,
  not an established new collector defect. **Investigation performed:** Read
  the full outputs and verified current/prior token equality. The prior literal
  copying failure remains repaired. **Resolution:** Controlled inherited
  limitation; no equal-budget parity claim.

- **Observed anomaly:** Timing runs emit generic potentially unsafe-allocation
  warnings; legacy gathers emit deprecation, L1-semaphore and packet-size advice;
  packed views drop ND metadata; startup/tooling emit motherboard, OpenMPI,
  clock-settling and matplotlib notices. **Evidence:** Raw benchmark, validation,
  diagnostic and profile logs; `runtime_audit.md`; prior stage06 review.
  **Affected path:** Allocation diagnostics, CCL compatibility, discovery and
  report generation. **Control or comparison:** Current tracker-on all64 B1 and
  reduced B32 transitions; exact mixed-slot tests; matched gather alternatives;
  completed four-device smoke and measurements. **Likely subsystem:** Runtime
  reporting/configuration. **Investigation performed:** Confirmed no broad
  corruptible-allocation exemptions, unchanged equivalent WIDTH_SHARDED layout,
  no measured CPU fallback, and complete raw profile windows. No packet-size or
  clock optimization benefit is claimed. **Resolution:** Controlled; warnings
  do not establish corruption, missing devices or failed measured work.

## Scope Inspected

- **Goal/skills:** Parent-provided stage07 contract and installed
  `prompts/model_bringup_multigoal/07-optimized-full-model.txt`; `stage-review`,
  `multichip`, `optimize`, `agentic-serving`, `tt-device-usage`, `full-model`,
  `tt-enable-tracing`, `qualitative-check`, and bringup startup under
  `.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21/`; repository
  `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`.
- **Code:** Stage diff, full generator preparation/prefill/decode/streamed and
  buffered generation, model terminal/layout/loading, sampler CCL adapter and
  common Sampling1D contract, candidate head, benchmark, validation, collector,
  prefill-packing, sampling, trace, external-ownership, memory/profile and summary
  harnesses. Decoder implementation remains unchanged. Collector allocations
  precede every trace; all traces share persistent tokens/logits/positions/page
  maps. No exact-tensor acknowledgment or broad allocation scope was added.
- **Identity:** Final generator SHA256
  `6cfca2c02aae45abcfc377aa586d7d1745061a031c63cfa0a62e9a33ca70fc04`;
  sampler adapter `1c652512bb91cec3692d95211c49808fbe1553173155365bf3c73d4414421694`;
  recorded loaded backend
  `aca153162839c6b5134ac9b446dd32d1574441a9c58fd6bc9a6ffceec17028d4`.
  Final validation, all four benchmarks, both profile scopes and current
  ownership/mixed-slot source hashes match local files. All227 final manifest
  entries have matching sizes and hashes. Baseline and reused HF-control receipt
  hashes also match. Metadata was reconciled to stage07 during review without
  changing measured runtime source.
- **Accuracy/quality:** `prefill_accuracy.json`, `teacher_forcing_accuracy.json`,
  AIME reference metadata/hash, actual autoregressive text, every shared HF/TT
  output, all three current long outputs and their HF/prior-stage controls.
  Each phase compares100 next-token positions from one AIME problem; top5/top100
  gates pass. Prompt templates and rendering identities are recorded.
- **Trace/capability:** Final collector/transition/watcher JSONs and logs,
  `reduced_trace_contract.json`, `external_prepare_reduced_b32.json`,
  `mixed_slots.json`, full64 benchmark mixed-slot records, packed-prefill33/65/4097
  controls, native-context inherited evidence, and context/agentic ledgers.
  Greedy/sample alternation,130-token collector crossing,16-step boundaries,
  changed/unchanged page maps, continuation tails, external failure cleanup,
  nonzero inactive states and stable captures are covered. Finished EOS rows
  remain physical slots; low-level inactive cache preservation is separately tested.
- **Performance:** Recomputed medians and verified stage06/07 prompt and output
  hashes for every4K row. Final warmed all64 autoregressive results are:

  |Batch/concurrency|TTFT ms|Tokens/s/user|
  |---|---:|---:|
  |1/1|431.360885|35.252433|
  |8/8|3564.809091|21.015013|
  |16/16|7302.092257|13.724910|
  |32/32|14715.772549|8.844420|

  All use4096 input/128 output and diagnostics off. Each127-step interval records
  zero input refreshes/captures/host copies, nonblocking split traces and one final
  read. B32's32768/128 result is144643.644184ms/7.441522 tokens/s/user. The separate
  clean203/100 AIME teacher-forced token-out control is114.055946ms/35.524841.
  It is correctly excluded from the headline. Layer-stack guide excess is at most
  4.49% before terminal allowance; no evidenced10–15% avoidable gap remains.
- **Profiles/candidates:** Advice-enabled prefill/decode tables and raw/filtered
  CSVs under `tracy/b1` and `tracy/b32`, runtime policy audit, complete per-device
  whole-layer attribution, head candidate table and sampling controls. Recomputed
  reduced envelopes: B1 prefill16073.86us/decode3120.088148us; B32
  prefill440997.339259us/decode5551.250370us. Every component attribution matches
  raw cycles. Fast TopK routes and canonical sampling are present; full-vocab
  gather, force-argmax and generic one-core TopK are absent from measured decode.
  Persistent sampler clones lose the measured controls; decoder persistent CCL
  pooling remains enabled. Inherited same-policy residual-grid experiments support
  retention of the non-dominant fused rows flagged by generic advice.
- **Memory:** Independently rebuilt live allocation sets and peaks from all four
  compressed event captures. All final additional live sets are empty and hashes/
  allocator deltas reconcile. B32's4096 request bound is28,440,140,288bytes/device
  versus34,178,731,008 capacity, and its1,056,768-token pool covers32x32768.
  B1 native-context prepared allocation is9,189,426,176bytes/device. New collector
  and sampling trace are accounted; trace-release bytes are separately scoped.
- **Telemetry:** The required local packet
  `/home/mvasiljev/agentic-research-data/experiment-26/bringup-logs/qwen38-27b/20261023-astra-medium-exp26-clean/telemetry/packets/7e42486c-136c-4a99-b106-d41352ac0a38.json`
  names stage07 and the supplied attempt/run IDs, matches the exact4096/128/B1
  autoregressive medians, records one AIME problem with100 compared positions,
  and leaves unavailable full-model device/roofline metrics null with reasons.
  Raw evidence is not embedded in that packet.
- **Commands:** Read-only `git status/diff/rev-parse`, `rg`, `cat`, `sed`, `tail`,
  and small Python JSON/CSV/hash/token/median/allocation analyses. No TT import,
  device access/reset, model execution, benchmark, server or subagent was launched
  by this reviewer. Only this review file was written.

## Residual Risk

- Accuracy evidence remains one AIME problem and a small qualitative suite;
  long-context task accuracy and broad dataset accuracy are unproven.
- Native context is demonstrated at B1. Larger batches use their recorded pools;
  their maximum feasible contexts and a fresh native-context HF comparison are
  not established. External preparation capacity and native execution are
  inherited only where the unchanged path permits it.
- EOS batching is not a streaming API and retains bounded excess device work.
  Explicit low-level serving semantics are preserved, but vLLM integration is
  outside this review.
- The recorded backend rotary correction remains required. Runtime, precision,
  cache, trace-lifetime or sampler changes invalidate affected evidence and need
  appropriate revalidation. Current timing gains should not be generalized beyond
  the measured workloads.
