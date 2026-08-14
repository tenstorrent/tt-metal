# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- None.

## Hard-Check Gaps

- The preserved `06-06-full-model.check-1.log` predates the failed-probe
  metadata renames and still reports two historical near-empty artifacts. A
  fresh execution of the authoritative checker against the live worktree exits
  0 and discovers only the healthy canonical autoregressive evidence.
- Full-wrapper Watcher coverage is unavailable because instrumented ACTIVE_ETH
  firmware exceeds the kernel-config buffer. The stage records the exact
  pre-construction failure and substitutes separate profiler,
  runtime-integrity, deterministic trace, and lifecycle evidence without
  changing the TP4 path.

## Anomaly Ledger

- Observed anomaly: Two historical sampler-regression probes contained
  canonical `autoregressive_meta.json` filenames and were interpreted as
  current completion evidence.
  Evidence: The preserved runner log reports near-empty completions under
  `autoregressive_feedback_fix_smoke_v2` and `autoregressive_post_sampler`; both
  now retain metadata as `autoregressive_meta.failed_probe.json`.
  Affected path: Recursive runner-side autoregressive artifact discovery, not
  the final runtime implementation.
  Control or comparison: Fresh authoritative checker exits 0;
  `autoregressive_feedback_shape_exact_smoke` changes feedback tokens, and
  `autoregressive_active_rm_final` produces a coherent 100-token completion
  with zero adjacent duplication and 5.26% trigram overlap.
  Likely subsystem: Evidence classification/naming.
  Investigation performed: Inspected both failed probes, final autoregressive
  metadata and text, work-log chronology, and reran the exact canonical
  checker.
  Resolution: fixed.

- Observed anomaly: The S128 performance artifact begins with control-token
  fragments.
  Evidence: `full_model_perf_active_rm_final_128.json` contains
  `<|im_end|>/<|im_start|>` fragments before coherent reasoning.
  Affected path: Performance-only S128 workload.
  Control or comparison: The benchmark deliberately truncates the 161-token
  rendered chat prompt; fresh 100-token AIME24 and six-prompt 200-token HF/TT
  qualitative artifacts use complete chat templates and remain coherent and
  prompt-relevant.
  Likely subsystem: Benchmark prompt truncation.
  Investigation performed: Compared performance JSON text, reference metadata,
  prompt token count, final autoregressive text, and all six qualitative HF/TT
  outputs.
  Resolution: controlled.

- Observed anomaly: Watcher cannot initialize the full TP4 fabric.
  Evidence: `full_model_watcher_reduced_final.log` records ACTIVE_ETH firmware
  size 27,920 bytes against a 25,600-byte kernel-config limit before model
  construction.
  Affected path: Watcher instrumentation only.
  Control or comparison: Separate named device profiling, traced correctness,
  zero-refresh counters, exact semantic-greedy feedback, stable three-cycle
  allocator lifecycle, and inherited unchanged-decoder Watcher coverage.
  Likely subsystem: Instrumented fabric firmware capacity.
  Investigation performed: Reviewed the failure classification and independent
  runtime-integrity evidence.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: stage-06 contract; `.agents/skills/full-model/SKILL.md`;
  `.agents/skills/tt-device-usage/SKILL.md`;
  `.agents/skills/stage-review/SKILL.md`.
- Artifact paths: full-model README/work log; context contract; runner
  prompt/check log; accuracy logs; final and failed-probe autoregressive
  artifacts; 200-token qualitative suite; long-prefill, capacity,
  semantic-greedy, streaming-overlap, performance, profiler, lifecycle,
  fallback-audit, and Watcher evidence.
- Code paths: `tt/model.py`, `tt/generator.py`, `tt/multichip_decoder.py`,
  `models/common/sampling/tt_sampling.py`, and full-model public-contract tests.
- Commands run: read-only `sed`, `find`, `rg`, `git status/diff`, `jq`, artifact
  text inspection, and the authoritative `06-full-model.check.sh` with the
  specified model variables.

## Residual Risk

- The 200-token qualitative window ends during Qwen3.6's control-matched
  visible reasoning preamble, so it establishes coherence and HF/TT behavioral
  parity rather than completed final answers.
- Batch-32 full-context residency is physically impossible with terminal
  weights; the measured supported batch-32 boundary is C72,192, while batch-1
  preserves the full 262,144-token context.
