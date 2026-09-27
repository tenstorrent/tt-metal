# Stage Review

Verdict: clean-pass

Independent Stage 06 full-model review for `google/gemma-4-26B-A4B-it`,
2026-09-27. Reviewed the live `gemma-4-26b-a4b-it` worktree based on
`6b1776a0bc3904efc4b00ab8d71c0fcd358e7c3d`. This supersedes the initial
more-work-needed review. Paths below are relative to the model directory
unless explicitly rooted at `models/common/`, `ttnn/`, or `bringup/`.

## Required Work

None. The initial review's seed, host-policy, independent-cache and incomplete
acceptance findings are closed by the inspected source and final artifacts.
The owner must perform the skill's post-review local checkpoint and record its
SHA; that follows this verdict and was not performed by the reviewer.

## Other Concerns

- The native build is **unverified**. The required
  `.github/scripts/copilot-build.sh --build-ttnn-tests` attempt failed because
  Docker is unavailable (`doc/full_model/native_build.log`). The user-provided
  AGENTS instructions explicitly permit this environmental limitation when
  disclosed. The affected kernels did JIT-compile in passing linear/ring and
  integrated Watcher tests. This is narrower evidence than a complete build.
- Optional TP4 log-probability output is explicitly unsupported. Both common
  samplers share the topology restriction in
  `models/common/sampling/tt_log_probs.py:419`. Validation at
  `tt/generator.py:94` rejects requests before mutation. This is a documented
  optional API limit, not a replacement of required device sampling.
- The headline is a warmed request with trace reuse: TTFT 2115.067 ms and
  49.2247 token/s/user at exactly 4096 input, 128 output, batch 1, concurrency 1.
  It includes caller token readbacks. Counter `synchronizations=0` counts
  explicit synchronization calls; it does not eliminate token-readback waits.
  The separate teacher-forcing throughput is not an autoregressive result.
- The 3078.76 microsecond device window and 512.37 microsecond sampling trace
  belong to the reduced two-layer profile. They are not full-stack device-time
  or roofline measurements. The telemetry packet leaves those full-stack
  quantities null, with an explanation.

## Hard-Check Gaps

- No required acceptance gap remains. Maximum-context evidence establishes
  allocation/execution and finite final logits, not a full-context HF accuracy
  benchmark. The 100-position AIME gate covers one chat-formatted prompt.
- Full-stack B32 independent-cache comparisons cover endpoint slots 0 and 31;
  the reduced real-layer/terminal test independently checks all 32 slots. The
  mixed B3 test checks inactive cache preservation. These scopes are explicit
  and supported by the actual test code and reports.
- Local checkpoint, final status reconciliation and telemetry acceptance
  status are post-review owner steps. No remote push is part of this review.

## Acceptance Evidence

- `tt/model.py` loads all 30 real decoder layers, scaled embeddings, final
  RMSNorm, a tied-weight vocabulary-sharded LM head and logit softcap. The
  decoder source is unchanged from the accepted baseline. Its precision,
  fidelity, cache split, residual layout and rejection ledger are preserved.
  Model-owned scratch sharing is across serial layers; downstream router
  outputs are copied before the shared scratch is reused.
- `tt/generator.py:194` exposes logical prompt lengths and caller-owned cache,
  page table and slot state. `decode_forward` handles persistent bindings and
  shape/content changes. `generate` has explicit `enable_trace`; standalone
  reset clears only owned caches. Both 262143 and 262144 prompt lengths pass
  on all 30 layers, with a maximum-context cache and traced decode at position
  262143 (`capacity.json`). Public nonaligned prompt probes include 31/33,
  1023/1025 and 4097 tokens (`prompt_lengths.json`). The full-stack context
  calculation includes both embedding/head representations, all KV caches,
  rotary tables, page table, scratch/trace reserve and persistent CCL storage.
- `readiness_final.json` and its completed log report prefill top1/top5/top100
  of 0.96/1.0/1.0 and traced decode of 0.94/1.0/1.0, each over 100 positions.
  The fresh AIME chat reference's SHA256 matches `reference_metadata.json`.
  The recorded command generated 100 continuation tokens and top100 candidates.
- The common sampler receives padded 32-row vocabulary shards and persistent
  `tt_out_tok`. `tt/generator.py:243` advances positions and sampled seed state
  on device; capture restores warmup-mutated inputs before use. The measured
  default loop performs 127 model and 127 sampling replays, zero steady token,
  position or page-table uploads, and zero full-logit readbacks.
  `trace_watcher.json` checks feedback snapshots, changed/unchanged page tables,
  reset identities and changed-prompt trace reuse against fresh capture.
- The initial batch-control finding is fixed in `tests/check_full_trace.py:115`:
  reference prefill and repeated decode use separately allocated caches.
  `trace_mixed_slots.json`, `trace_batch32.json` and `trace_full_batch32.json`
  show minimum recorded PCC 0.99999994 and passing top5 comparisons. An
  `inactive_cache_unchanged=false` field in older B32 reports means that no
  inactive-slot case was run there; B32 has every slot active. B3 provides the
  actual inactive-cache assertion.
- Final `sampling_contract.json` and `sampling_contract_pass.log` pass fresh
  unseeded entropy, explicit seed repeatability, alternating modes, sampled
  token/seed feedback, penalty counts, host compatibility, and early logprob
  rejection without trace/token/position mutation. Runtime auditing rejects
  host tensor execution during model prefill, decode, sampler precompile and
  replay. The common sampler's normal semantically greedy path is selected
  over the measured slower force-argmax alternative.
- All twelve shared-suite HF/TT completions were read directly, along with
  both standard autoregressive completions. Chat metadata and rendered/token
  prompts are present. The standard runner's prompt IDs exactly match
  `autoregressive/prompt_format.json`; both implementations emit 128 tokens.
  `degeneracy.json` has no findings and exit code 0. The shared answers remain
  coherent; incomplete long answers are capped in both controls.
- `profile_terminal/decode_perf_report.csv` contains 281 operation rows and
  matches the real two-layer wrapper, terminal and split greedy sampling path.
  Its raw source report exists. The LM head is BF16/HiFi4 and vocabulary
  sharded; sampling is 16.6% of the complete reduced window and does not
  dominate. `layer_stack_comparison.json` labels its host-wall accounting as
  diagnostic rather than device-time subtraction.
- Host tests report 10 passed. Recorded pre-commit hooks pass, including
  clang-format; independent `git diff --check` also passed. The compact packet
  `bringup/artifacts/multigoal-runs/20260925T171711Z/telemetry/packets/
  4c7850c3-ede9-4ed7-9ffc-f5c36cd19e58.json` reproduces the final workload,
  accuracy and performance values without substituting reduced device times.

## Anomaly Ledger

- Observed anomaly: HF chat-template output was a mapping rather than flat IDs.
  Evidence: reference failure/control reports and `reference_host_tests_final.log`.
  Affected path: common reference prompt tokenization.
  Control or comparison: real HF tokenizer unit test and original reference run.
  Likely subsystem: tokenizer return type.
  Investigation performed: explicit `return_dict=False` repair and regression.
  Resolution: fixed.
- Observed anomaly: full-stack prefill L1 collision.
  Evidence: `AUTODEBUG_prefill_l1.md`, router-allocation reproducer logs and
  completed full-stack readiness/capacity runs.
  Affected path: full-model persistent router scratch.
  Control or comparison: extra semaphore-only probe passes; exact router
  allocations reproduce the collision.
  Likely subsystem: persistent L1 address frontier.
  Investigation performed: serial scratch sharing, preserving decoder policy.
  Resolution: fixed.
- Observed anomaly: B3 sampler broadcast failure and unsafe public output allocation.
  Evidence: raw/padded sampler controls, trace failure logs, final B3/B32 reports.
  Affected path: sampler row padding and token output lifetime.
  Control or comparison: known-winner greedy/sampled B1/B3/B32 and independent caches.
  Likely subsystem: logical 32-row sampler contract and trace allocation lifetime.
  Investigation performed: physical padding and persistent public-output storage.
  Resolution: fixed; allocation-tracked integration tests pass.
- Observed anomaly: deterministic omitted seeds and inconsistent host policy.
  Evidence: initial review, `tt/generator.py:94`, `sampling_contract_pass.log`.
  Affected path: request sampling setup and host compatibility.
  Control or comparison: direct seed-state checks, explicit-seed repeats and
  unsupported host-policy rejection.
  Likely subsystem: generator request state.
  Investigation performed: fresh setup entropy and uniform host greedy behavior.
  Resolution: fixed.
- Observed anomaly: reduced two-layer host greedy output collapses to EOS while
  common device greedy selects other tokens.
  Evidence: final `sampling_contract.json` and `sampling_greedy_oracle.json`.
  Affected path: incomplete-stack diagnostic logits, not full-model text output.
  Control or comparison: initial and same-prefix decode oracles find both tokens
  equal to the maximum softcapped logit 30, with tens of thousands of ties.
  Likely subsystem: reduced-stack softcap saturation and valid tie selection.
  Investigation performed: actual full-logit maximum checks at each tested step.
  Resolution: controlled; no exact host/device tie-order claim.
- Observed anomaly: optional logprob requests returned no result on TP4.
  Evidence: `AUTODEBUG_logprobs.md`, shared calculator topology guard and final test.
  Affected path: optional logprob API.
  Control or comparison: both common sampler implementations share the same gate.
  Likely subsystem: unsupported common-calculator topology.
  Investigation performed: explicit rejection before mutation; device sampling
  and penalties remain tested independently.
  Resolution: fixed API behavior; support limitation documented.
- Observed anomaly: native Watcher assertion in unused scatter initialization.
  Evidence: `watcher_failure/AUTODEBUG.md`, BF16-pass/uint32-fail baseline,
  linear/ring after-controls and `trace_watcher_fixed.log`.
  Affected path: multicast all-gather with one 4096-byte page per packet.
  Control or comparison: 2048/4096-byte eager and changed-input traced cases on
  all four devices, with alternate-route template evidence.
  Likely subsystem: unconditional construction of a scatter header with one chunk.
  Investigation performed: guard both initialization sites with the existing
  `use_scatter_write` condition; preserve unconditional unicast initialization.
  Resolution: fixed in tested device paths. Full native build remains unverified.
- Observed anomaly: allocator warning with live traces and an initial Watcher
  Ethernet firmware size failure.
  Evidence: untracked generation logs, final tracked tests and work-log recovery.
  Affected path: trace lifetime diagnostic and Watcher instrumentation.
  Control or comparison: allocation tracking passes; NOINLINE Watcher setup
  reaches and passes the model after the separate native repair.
  Likely subsystem: generic allocation warning and instrumentation code size.
  Investigation performed: tracked lifecycle tests and bounded device recovery.
  Resolution: controlled/fixed within recorded scopes.
- Observed anomaly: repeated headline performance completion and differing chat wording.
  Evidence: `performance.json`, `tests/run_full_readiness.py` prompt construction,
  shared HF/TT text, divergence ranks and standard autoregressive artifacts.
  Affected path: synthetic repeated-document performance prompt; chat generation.
  Control or comparison: performance input itself repeats the same sentence;
  properly rendered HF/TT chat controls remain coherent with matched token caps.
  Likely subsystem: prompt continuation and numerical greedy-choice differences.
  Investigation performed: direct output reading and same-prefix HF ranks.
  Resolution: controlled; performance text is not used as the quality verdict.

## Scope Inspected

- Supplied Stage 06 contract; `.agents/skills/{stage-review,full-model,
  tt-enable-tracing,qualitative-check,tt-device-usage}/SKILL.md`; installed
  review core, router, model-bringup and trace review instructions.
- Full `tt/model.py`, `tt/generator.py`, new runners/probes, changed common
  readiness helpers/tests, native multicast helper, relevant accepted decoder
  paths and common sampler state/trace/logprob code.
- Full-model README, work log, context contract, initial review, AutoDebug/AutoFix
  reports, readiness/reference metadata, actual qualitative/autoregressive text,
  capacity, sampling, trace, Watcher, runtime, profiler, build/lint/test evidence
  and compact telemetry packet.
- Read-only git/status/diff/check commands, `find`, `cat`, `sed`, `grep`, `nl`,
  `sha256sum` and small Python JSON/CSV analyses. `rg` is unavailable. No hardware,
  resets, servers, tests or long experiments were run by this reviewer.
  The only reviewer write is this report.
- Source SHA256 at final read: model
  `42cf1bd4802f8c773d29570f4dc6b19988a7bb19cb5d906c7c81c5c3fea29672`;
  generator `32ae510b3102d713e53efea01c5b4bad8dfaad6d961f3f9a123c11abea1ae03e`;
  trace test `d8b33b1522b18e0d44e8357ae88cd14e91f167c877d948b41aaa75a82961c07d`;
  sampling test `3ba9fcdb5faf92c83476929ade5ecf6b11c26e19914742e28e371e280d463ba1`;
  multicast helper `c3e41e9d927627cd739e5546b7bd42010955d5aec20c89474697391b6f4b9783`.

## Residual Risk

This is Stage 06 full-model acceptance on the selected TP4 hardware and tested
contracts. It is not vLLM serving validation, whole-dataset accuracy, full-context
HF equivalence, exhaustive sampling-distribution qualification, or a complete
native build. No unmeasured performance improvement is asserted. Subsequent
implementation changes require checks appropriate to the affected path.
