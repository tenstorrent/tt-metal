# Stage Review

Verdict: more-work-needed

Independent Stage 06 review of `google/gemma-4-26B-A4B-it`, 2026-09-27.
Live branch `gemma-4-26b-a4b-it`, HEAD
`6b1776a0bc3904efc4b00ab8d71c0fcd358e7c3d`. Implementation and evidence were
changing during inspection. This is an initial review, not a final source freeze.
Paths below are relative to the autoport unless they start with `models/common/`.

## Required Work

- **P2: Distinguish an omitted sampling seed from an explicit seed of zero.**
  Evidence: `tt/generator.py:91-94` converts every `None` to zero before hashing,
  and `_forward` at lines 202-203 advances every sampled request from that same
  initial state. The selected common implementation explicitly initializes
  unseeded requests with per-user entropy
  (`models/common/sampling/generator.py:768-783,830-843`).
  Why this matters: repeated unseeded requests follow a fixed stream, and
  unseeded lanes start from identical state. This silently changes the common
  sampling API's unseeded behavior.
  Required next step: generate fresh per-request/per-slot seeds at setup for
  `None`, retain reproducibility for explicit seeds (including zero), and keep
  advancement on device. Check seed state directly; do not require a tiny
  stochastic sample to produce different tokens by chance.

- **P2: Host compatibility silently changes a sampled request after its first token.**
  Evidence: `tt/generator.py:314-329` accepts and applies `sampling_params` for
  prefill regardless of `host_sampling`; `_replay` at lines 280-284 always uses
  raw argmax in host mode. Temperature, top-k/top-p and penalties are ignored
  for subsequent tokens.
  Why this matters: one request can start with sampled/penalized behavior and
  then switch to unpenalized greedy decode without a diagnostic.
  Required next step: either implement the requested host policy consistently,
  or explicitly reject unsupported non-greedy/penalty configurations and
  document this as greedy-only compatibility. Keep the device sampled path.

- **P2: Finish batch-position correctness with an independently prefetched cache.**
  Evidence: the initial `tests/check_full_trace.py` batch branch checked
  feedback/positions and inactive cache preservation only. The live revision
  now adds distinct prompts and isolated B1 decode at lines 95-108, but passes
  the same batched-prefilled `caches` into the reference call at line 101.
  Why this matters: this controls decode batching but can reproduce a prefill
  slot/page contamination in both sides. The full-model skill explicitly
  requires logits to be deterministic across batch positions and larger-batch
  prefill/cache correctness.
  Required next step: compare final prefill logits and repeated decode outputs
  against independently allocated/prefilled single-request caches, or an
  equivalent independent slot-permutation control. Exercise distinct prompts,
  mixed lengths, fixed inactive slots and a tested batch through 32. A reduced
  real-layer/terminal control is suitable for isolating this interface.

- **P1: Complete the acknowledged Stage 06 acceptance gates.**
  Evidence: `trace_mixed_slots.log` fails in the canonical sampler;
  `AUTODEBUG_sampler_batch.md` identifies logical B3 versus B32 sampler state.
  At the last inspection `capacity.json` contains only the passing 262143 row
  with final-position decode; aligned 262144 is still running. There is no
  passing sampled/runtime-audit report, standard autoregressive/degeneracy
  output, or reduced terminal perf report. `sampler_comparison.json` contains
  allocation-tracking timings, which the owner correctly labels diagnostic.
  Required next step: finish the focused sampler padding repair and regressions,
  capacity run, sampled/runtime audit, standard autoregressive plus degeneracy
  gate, reduced one-layer-per-kind profile, and an untracked semantically greedy
  candidate comparison. Update the context contract and evidence to the final
  implementation, then request independent rereview. These are already active
  owner work, not new diagnoses.

## Other Concerns

- `tt/generator.py:248-265` rebinds on cache identity, batch and active-slot mask,
  but not page-table shape. A caller changing table width with the same cache
  reaches a copy into the old tensor. The same-shape content-change test is
  covered. State the fixed-shape requirement or include table shape in rebinding;
  the different-shape case was not run in this read-only review.
- `sampling_comparison.md:6-8` still says explicit request seeds disable the
  sampling trace, while the current wrapper deliberately owns device seed
  state and leaves the common seed manager inactive. Reconcile this after the
  sampled trace checks. The work log also predates completed readiness and
  qualitative evidence.
- The direct performance result reports 127 token readbacks and zero explicit
  synchronization calls. Reading tokens is a synchronization boundary; this is
  caller-visible output collection, not host token feedback. Do not describe
  the measured generator as having no readbacks or no synchronization cost.

## Hard-Check Gaps

- Preserve evidence connecting final sampler normalization/API changes to the
  accepted B1 readiness and performance path. Existing numbers were obtained
  before the pending normalization repair.
- Final pre-commit status, local stage checkpoint commit and compact telemetry
  remain owner completion work. Python-only changes do not require a C++ build.
- Classify the allocator warning in `generation_run.log:64` and `capacity.log`
  against the allocation-tracked controls. The warning alone does not prove
  corruption; the passing B1 trace evidence should be cited with its scope.

## Anomaly Ledger

- Observed anomaly: B3 sampler subtile broadcast failure.
  Evidence: `trace_mixed_slots.log`, `AUTODEBUG_sampler_batch.md`.
  Affected path: low-level mixed/fixed-slot decode.
  Control or comparison: B1 sampling passes; dimensions in the source explain
  why singleton broadcasting differs.
  Likely subsystem: sampler-bound logical batch padding.
  Investigation performed: existing source diagnosis inspected independently.
  Resolution: more-work-needed; focused repair/experiment pending.

- Observed anomaly: headline performance completion repeats the same sentence.
  Evidence: `performance.json`; `tests/run_full_readiness.py:68` constructs a
  raw prompt by repeating that sentence 1024 times before taking 4096 tokens.
  Affected path: synthetic performance workload only.
  Control or comparison: all six properly rendered shared chat prompts have
  coherent HF and TT completions; this workload is not their quality evidence.
  Likely subsystem: prompt continuation behavior; source establishes the
  repetition was deliberately present in the input.
  Investigation performed: inspected prompt construction and actual output.
  Resolution: controlled as a synthetic continuation workload; label its prompt
  mode explicitly and do not use this completion as a quality verdict.

- Observed anomaly: shared-prompt wording differs from HF, including a rank-7
  first divergence in `shared_2`.
  Evidence: actual completions in `qualitative_hf.json`, `qualitative_tt.json`
  and same-prefix ranks in `qualitative_divergence.json`.
  Affected path: full 30-layer greedy generation.
  Control or comparison: matched pinned model/tokenizer/chat prompts; HF ranks
  at first divergence are 2,3,7,2,2,2.
  Likely subsystem: numerical differences changing greedy selection.
  Investigation performed: read all twelve completions. Both sides remain
  coherent; long answers are capped at the same 128-token budget.
  Resolution: controlled qualitative divergence, not an exact-output claim.

## Scope Inspected

- Goal/skills: supplied Stage 06 contract; `.agents/skills/{stage-review,
  full-model,tt-enable-tracing,qualitative-check,tt-device-usage}/SKILL.md`;
  installed `tt-review-core/SKILL.md`; `models/common/readiness_check/contract.py`.
- Artifacts: Stage 06 README/work log/context contract, readiness/reference
  metadata and logs, performance, trace controls, sampler comparison/diagnosis,
  HF/TT qualitative completions/divergence, partial capacity result, existing
  host test log (10 passed).
- Code: complete `tt/model.py` and `tt/generator.py`, new Stage 06 runners and
  probes, changed readiness helpers/tests, relevant decoder/cache/router paths,
  common sampler tracing, parameter normalization and seed management.
- Commands: read-only `git status`, `git diff`, `git rev-parse`, `find`, `cat`,
  `sed`, `grep`, `nl`, `sha256sum`, and Python JSON inspection. `rg` is unavailable.
  No hardware commands, tests, servers or implementation mutations were run.
  The only reviewer write is this report.
- Source hashes at final read: `tt/model.py`
  `0ad27cedfaddf6d68977f7d7bf17cadc8dfcc75ad180038ef8d89567d2c88ef4`;
  `tt/generator.py`
  `474bd41e85354efbf050d515af5280407a92da631b984aa0e714432bfd4fd42b`;
  `tests/check_full_trace.py`
  `acd8e8ab2dc058763bc42c7521b468443f7918547cb5a552518dbb60da750857`.

## Residual Risk

The existing all-layer AIME readiness report supports top5=100% and top100=100%
for both prefill and 100-position traced decode. The measured all-layer
4096/128/B1/C1 path reports TTFT 2118.34 ms and 49.32 tokens/s, with device token
feedback and position advancement. Those successes do not close the pending
sampled/batch/capacity/profile gates. No new decoder dtype, fidelity, KV or
residual-policy regression was identified in the wrapper source. The live
worktree must be rereviewed after the outstanding fixes and checks.
