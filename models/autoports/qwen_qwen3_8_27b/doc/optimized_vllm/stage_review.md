# Stage Review

Verdict: clean-pass

Independent review of Stage 10 optimized-vLLM, Qwen/Qwen3.8-27B, completed
2026-09-14. This final review supersedes the interim pending-gate verdicts.
The reviewer inspected source, raw artifacts and generated text, performed
small local artifact analyses, and wrote this report. No server, device,
model experiment, reset or profiler was run by the reviewer.

## Required Work

- None. The missing precision artifact was restored and verified. Final native
  quality, concurrent/seed/lifecycle controls, matched benchmarks, complete
  compatibility sampling profile and cleanup evidence pass review.

## Other Concerns

- The full shared sampling profile is **73 passed**, with no failures, skips
  or expected failures, in explicit **all-host compatibility** at max sequences
  32. Its 962.82-second log covers broad sampling/API tests; it does not prove
  native device sampling performance. Native supported sampling has separate
  final-source greedy, stochastic, qualitative, async and benchmark evidence.
- Primary native **S128/G128/N1**, max sequences 1, concurrency 1, temperature
  0, ignore EOS and one same-shaped warmup: TTFT improves **81.3734 to
  62.9547 ms**, mean TPOT changes **24.20018 to 24.20525 ms**, and decode is
  **41.3220 to 41.3134 tokens/s/user**. All 128 output tokens arrive. Raw and
  normalized results agree; command arrays and actual server configs match.
  The final default reproduces the candidate's warmed TTFT improvement.
- Secondary native **S100/G100/N32**, max sequences 32, unbounded burst and
  zero explicit burst warmups, completes 32/32 and 3200 output tokens before
  and after. Aggregate throughput is **138.0904 to 138.1700 tokens/s**;
  median TTFT is **3287.20 to 3372.12 ms**. The matched preceding cold
  B32 S128/G128/N1 TTFT is **521.11 to 567.87 ms**. The report discloses
  these cold costs and does not claim burst acceleration.
- Primary decode is comparable to the selected full-model queued token-out
  reference, **41.0955 tokens/s** at S128/G128/B1. Different prompt data,
  cache geometry and output boundaries prevent a direct speedup claim.
- Source inspection supports the serving contract: exact external cache
  binding; generator-owned prefill/sampling; persistent token, position,
  RoPE, page and sampling inputs; page copies only on changes; device token
  feedback and seed/position advancement; separate model and sampling trace
  replay with `blocking=False`; minimal deferred token read and host formatting.
  Native measured paths contain no host argmax or full-logits readback.

## Hard-Check Gaps

- Primary P99 TTFT/TPOT represent one measured request, not population tails.
  Tiny decode and burst-throughput differences have no statistical-significance
  claim. The clear primary improvement is warmed TTFT.
- HTTP submission order does not force particular scheduler rows. The diverse
  full-model control covers allocator-driven growth and reordered concurrent
  submission; the separate direct adapter probe forces a drained permutation
  and checks pending snapshots, inactive state and physical page tensors.
- Full64 prefill boundary probes cover 31, 33, 128, 129, 4095, 4096 and 4097
  tokens. Maximum-context execution is inherited from the unchanged full-model
  and selected-policy paths. This stage preserves served context 262144 and
  bounds new persistent storage; it does not rerun every context/batch pairing.
- Lifecycle gauges measure scheduler/cache occupancy and API-process RSS,
  not exhaustive device DRAM allocation. Tracker evidence, bounded ownership
  and finite repeat checks support the changed paths, with this limit stated.
- The shared degeneracy checker does not establish answer completeness or
  correct haiku meter. All twelve shared and eight extended outputs were read
  directly. The eight extensions stop coherently; three inspected Fibonacci
  functions pass six cases each under restricted local execution.

## Anomaly Ledger

- Observed anomaly: reduced adapter/multirow streams are only token 220 and
  provide a weak token-sensitivity oracle.
  Evidence: `adapter_final.json`, `watcher_prefill.json`, final
  `concurrent_control.json`, `seed_continuity.json` and tracked server log.
  Affected path: async feedback, automatic page growth and request changes.
  Control or comparison: four diverse full64 chat prompts, native async,
  logprob-forced synchronous greedy and reordered native requests.
  Likely subsystem: reduced-model output sensitivity, not established cache
  corruption.
  Investigation performed: independently compared all 12 actual text/token
  arrays; four distinct 100-token streams match through positions 96/128/160.
  Six native seeded continuation prefixes match across companion admission
  and departure. The control server's 1123 model minus 1024 sampler replays
  equal 99 host decode steps; 103 full-logit reads equal those 99 plus four
  control prefills. Native-only benchmark/quality servers are separate.
  Resolution: fixed evidence-sensitivity gap.

- Observed anomaly: periodic lifecycle continuation text and possible retained
  allocation concerns after new shapes.
  Evidence: `lifecycle.json`, `lifecycle_concurrent.json`, tracked full64 B32
  server and bounded generator-owned prefill/staging storage.
  Affected path: new-shape and repeated-request trace/cache lifetime.
  Control or comparison: sequential and concurrent S31/33/127/129/31/33,
  G70 each, after concurrent and seed controls on the same live server.
  Likely subsystem: periodic raw input continuation and allocation lifetime.
  Investigation performed: all 12 HTTP responses deliver 70 tokens. Before
  and after running/waiting/KV gauges are zero. API RSS rises 131072 bytes
  sequentially and is unchanged across the concurrent sequence. One owned
  shape and one cache-bound sampling input bound persistent storage; tracked
  replay reports no violation. Periodic raw requests are structural stress,
  separately labeled from the prompt-correct quality suite.
  Resolution: controlled finite lifecycle behavior; no general memory guarantee.

- Observed anomaly: precision confirmation citation initially referenced an
  absent local artifact.
  Evidence: `restored_evidence.json` and
  `../datatype_sweep/selected_confirmation.json`.
  Affected path: selected-policy evidence provenance.
  Control or comparison: historical git object
  `3f48cac393ac8609517b9e8dee76828a87821e5d`.
  Likely subsystem: archived artifact retention.
  Investigation performed: all 325099 restored bytes and SHA256 match that
  object. All 64 layers record 256 BFLOAT4_B uploaded tensors and 320
  LoFi/FP32-accumulation compute configurations; head BFLOAT4_B/LoFi and
  final norm HiFi2 agree with the selected policy. Runtime policy equals the
  selected configuration. README labels this inherited restoration correctly;
  `propagation_check.json` is described as a summary.
  Resolution: fixed; no new confirmation experiment is claimed.

- Observed anomaly: shared output truncation, visible closing thinking marker
  and sampled haiku meter 5/7/4.
  Evidence: final shared/extended outputs, prompt metadata, control comparison,
  degeneracy result and native quality server log.
  Affected path: prompt format, generation budget and instruction quality.
  Control or comparison: exact Stage 9 greedy and seeded extension controls.
  Likely subsystem: checkpoint reasoning template/budget and syllable counting.
  Investigation performed: read all 20 outputs and independently compared all
  six greedy texts plus all eight extended token/text/request/input arrays.
  They match prior controls exactly; all extensions finish with EOS. Prompt
  metadata/source hash and real chat endpoint agree. The sampled final line
  “answers from noise” has four syllables, exactly as in the prior control.
  Stories finish coherently, science/translation remain correct, and inspected
  Fibonacci functions execute correctly. Fresh unseeded samples are coherent
  but are not asserted to reproduce prior unseeded wording.
  Resolution: controlled; no serving-quality regression established.

- Observed anomaly: generic live-trace allocation corruption advisory.
  Evidence: `after/primary_server.log`, `prefill_full.log/.json`, reduced
  Watcher evidence, tracked native B32 server and temporary-lifetime tests.
  Affected path: prefill and first-token sampling trace storage.
  Control or comparison: exact eager/traced full64 logits/tokens, changed
  prompts/pages, stable identities and allocation tracking.
  Likely subsystem: allocations while trace scratch addresses are reserved.
  Investigation performed: inspected creation of persistent destinations before
  capture, coordinated trace release, public/packed temporary destruction,
  short/fallback transitions and partial-capture cleanup. Boundary, Watcher and
  full64 concurrent/lifecycle tracker runs complete without a violation.
  Resolution: controlled for the tested paths. The generic warning alone is
  neither proof of corruption nor proof of safety.

- Observed anomaly: nanobind reference warnings at interpreter shutdown.
  Evidence: before/final server logs, worker/UMD close markers, guard cleanup,
  `final_process_audit.json` and `final_device_health.log`.
  Affected path: Python/C++ binding teardown.
  Control or comparison: warnings also occur in the unchanged baseline.
  Likely subsystem: binding reference lifetime.
  Investigation performed: checked mesh-close ordering, all nine recorded
  launch identities, the owner's empty process audit and an independent
  read-only `/proc` scan. No serving/owned processes remain. The subsequent
  bounded listing records all four Blackhole chips; no reset was required.
  Resolution: controlled inherited teardown behavior; no leak-free claim.

- Observed anomaly: cold CI and B32 single-request TTFT increase.
  Evidence: matched before/after raw and normalized benchmark results.
  Affected path: cold prefill and first-use trace/staging work.
  Control or comparison: same workload, preceding cold request and native
  counters separating first-use samples from decode sampling replay.
  Likely subsystem: trace preparation/capture and run variation.
  Investigation performed: re-derived metrics, checked exact launch/benchmark
  configuration equality and unchanged math hashes. The report states the
  cold tradeoff and disclaims burst acceleration or isolated causal timing.
  Resolution: controlled tradeoff; primary warmed TTFT improves and capacity
  remains intact. Exact cold-cost attribution is not proven.

## Scope Inspected

- Goal/skills: parent-supplied Stage 10 contract; repository stage-review,
  vllm-integration, optimize, tt-enable-tracing, qualitative-check and
  tt-device-usage; installed vllm-integration allocator/lifecycle requirements.
- Artifacts: final README/work log/checklist/perf and validation summaries;
  AutoDebug/AutoFix records; before/candidate/final benchmarks and logs;
  full64/reduced/Watcher prefill and adapter probes; host and pre-commit logs;
  all final native quality/concurrent/seed/lifecycle responses and controls;
  restored precision evidence; context contract; full sampling/server/runner
  logs; process/health audits; source/config and artifact manifests.
- Code: `tt/generator.py`, `tt/generator_vllm.py`, changed host tests,
  `check_vllm_prefill_tracing.py`, concurrent/seed/lifecycle runners, serving
  launch script, common sampler, sibling plugin async/drain logic and relevant
  cache/model interfaces. Operator-owned `PIPELINE_BLOCKERS.md` and root
  `AUTODEBUG.md` are excluded.
- Commands: read-only git status/diff/show, rg, sed, cat and small Python
  JSON/hash/config/token/metric analyses, restricted execution of inspected
  generated Fibonacci functions and a read-only process scan. All 89 immutable
  artifact hashes/byte counts verify; all four final runtime hashes still match
  primary timing. Five unchanged math/policy hashes and actual before/after
  launch configs verify. Scoped `git diff --check` passes. All report Markdown
  link targets checked exist.
- Verification provenance: the owner ran 59 host tests, Python compilation,
  Black and final pre-commit; logs pass. The initial pre-commit whitespace fix
  is retained before its passing rerun. The reviewer counted all 73 passing
  pytest cases and no nonpassing cases. Three pytest warnings are missing
  source-checkout version metadata and SWIG deprecations. Python/docs-only
  changes do not require a C++/CMake build.
- Repositories: live branch `mvasiljevic/qwen38-full-bringup`; tt-metal review
  base `929c84f1afb0296d82d2e39fe1e80a669d621314`. Sibling vLLM is clean and
  unchanged at `5dfd818f4f0f5444533331d85f4711c43f8f2f2b`.

## Residual Risk

- This is a serving optimization pass with finite quality and lifecycle
  coverage, not release-scale evaluation or an arbitrary-shape allocator soak.
- New shapes, cache bindings and active-slot changes can require cold sampling
  and recapture. One retained prefill shape bounds memory and does not make
  every request mix warm. Prefix caching remains disabled.
- Native sampling retains its documented subset. Broader API sampling uses
  explicit compatibility; shared compatibility success is not native timing.
- Runtime JSON/log evidence remains local under the autoport ignore policy.
  Source commits alone do not carry the complete 89-artifact evidence set.
- No Tracy, tt-perf-report, ReadDeviceProfiler or serving profiling is required
  for this pass; device-time/roofline fields are null with the explicit reason.
- Post-review bookkeeping is to record this clean pass and create isolated local
  checkpoint commits for stage-owned source/tests/docs, excluding operator
  files. This does not require another experiment; no push is authorized.
