# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- None blocking. The optimized TTFT sample is 0.12% slower than the matched
  force-argmax baseline, while token-out decode is 0.82% faster. The report
  discloses both results and does not claim a TTFT win.

## Hard-Check Gaps

- The Tracy slice uses representative linear/full-attention layers rather than
  all 64 layers so profiler buffers retain the terminal and sampler rows. This
  does not hide the selected policy: the all-64 final benchmark supplies the
  headline latency, inherited optimized-multichip artifacts supply every layer
  median and policy/rejection evidence, and the representative profile shows
  the claimed BF16/BFP4/BFP8 fidelity rows plus the complete terminal path.
- ACTIVE_ETH Watcher instrumentation is disabled because its Blackhole firmware
  image exceeds local firmware capacity. The final mixed-slot run retains host
  and Tensix Watcher coverage on all four devices; the limitation is documented
  and does not alter the measured runtime path.

## Anomaly Ledger

- Observed anomaly: the first performance artifacts performed one sampled-token
  readback inside each timed replay.
  Evidence: the initial benchmark source and superseded artifacts reported 131
  total readbacks for 128 replays.
  Affected path: benchmark timing/provenance, not generator feedback semantics.
  Control or comparison: corrected force-argmax and default-split runs use the
  same B1 S128/G128 workload and report four setup/probe/reporting readbacks,
  all outside the measured interval.
  Likely subsystem: benchmark orchestration.
  Investigation performed: inspected the timed loop and re-derived counters and
  rates from `baseline_full_force_argmax_no_readback_b1_s128_g128.json` and
  `final_default_split_no_readback_b1_s128_g128.json`.
  Resolution: fixed.
- Observed anomaly: the first documentation revision cited a mixed-slot JSON
  that the runner did not yet write.
  Evidence: Watcher log contained the pass while the cited JSON was absent.
  Affected path: evidence provenance only.
  Control or comparison: the runner now has explicit `--output` handling and
  `mixed_slots_split_watcher.json` records S65/S63, inactive-row state,
  top-k=5/top-p=.9, exact inactive KV preservation, and reset/reuse.
  Likely subsystem: test artifact serialization.
  Investigation performed: compared runner arguments, log, and artifact path.
  Resolution: fixed.
- Observed anomaly: the inherited six-prompt suite predated the final split
  sampler default.
  Evidence: optimized-full-model initially contained only the AIME24
  autoregressive output.
  Affected path: final-default qualitative coverage.
  Control or comparison: fresh exact-revision chat-template HF and TT outputs
  in `full_model_qualitative_50.json`; three cases are token-exact through 50
  tokens and the other three remain coherent, task-relevant English after
  expected low-precision greedy divergence.
  Likely subsystem: qualitative evidence freshness.
  Investigation performed: inspected all six rendered prompts and both outputs.
  Resolution: fixed.
- Observed anomaly: Qwen does not expose the Galaxy-only keyed
  `line_all_gather` persistent-buffer API.
  Evidence: `TT_CCL` lacks `line_all_gather`, so sampling uses ordinary TTNN
  candidate all-gathers.
  Affected path: candidate CCL implementation.
  Control or comparison: candidate tensors/programs are stable allocations
  owned by the captured sampler trace, semaphores are cached, the two gathers
  measure 15/13 us, and neither dominates token-out. The inherited decoder CCL
  policy/rejection ledger remains unchanged.
  Likely subsystem: platform-specific CCL API availability.
  Investigation performed: inspected `SamplingArgs`, `_perform_all_gather`,
  `TT_CCL`, trace ownership, and profiler rows.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: optimized-full-model user contract; `multichip/SKILL.md`,
  `optimize/SKILL.md`, `tt-device-usage/SKILL.md`,
  `tt-enable-tracing/SKILL.md`, `qualitative-check/SKILL.md`, and
  `stage-review/SKILL.md`.
- Artifact paths: optimized-full-model README/work log, context contract,
  baseline/final/reduced benchmark JSON and logs, runtime fallback audit,
  AIME24 prefill/teacher-forcing/autoregressive evidence, mixed-slot Watcher
  JSON/log, AutoDebug/AutoFix reports, six-prompt qualitative JSON/log, Tracy
  raw CSV and tt-perf-report machine/human outputs, and inherited selected-policy
  optimized-multichip/full-model evidence.
- Code paths: Qwen generator, full-model performance/mixed-slot/sampler probes,
  common sampling implementation and planner tests, teacher-forcing runner and
  tests, model terminal path, and CCL helper.
- Commands run: read-only `git status/diff`, `rg`, `sed`, `jq`, JSON/PyTorch
  artifact inspection, process/artifact provenance checks, `git diff --check`,
  and the focused seven-test host suite. No TT hardware was opened by review.

## Residual Risk

- The six-prompt suite is capped at 50 generated tokens, so several Qwen
  thinking-first responses end before their final answer. Matching HF controls,
  exact prompt formatting, the separate 100-token AIME24 run, and the inherited
  200-token full-model suite make this a bounded evidence-length risk rather
  than visible stage wrongness.
- Final token-out is 7.61% above the decoder-layer-only lower bound. This is
  within the contract's 10-15% split gate, and the terminal profile attributes
  the remainder without a dominant avoidable sampler, full-vocabulary gather,
  force-argmax, or host-boundary row.
