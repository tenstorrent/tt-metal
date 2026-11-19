# Stage Review

Verdict: clean-pass

Stage 9 vLLM integration, `ornith-ai/Ornith-1.5-9B`, independent follow-up
round 2 of 3. This review checks the authorization/status delta only. The live
checkout is `main` at `cc9b199e26521aac5729f7f446e9d7b5088da8e5`;
implementation and evidence checkpoints remain
`aa3d320cd6f0f1b64c0ddd9a5034c15b0f8b68bf` and
`e4746774a229575bd4bcd08c6be7cdba9692cdf6` respectively.
Paths below are relative to `models/autoports/ornith_ai_ornith_1_5_9b/`
unless stated otherwise.

## Required Work

- None. Prior P2, "Record operator acceptance of the default `top_k=20`
  serving recipe": **closed**. The authoritative owner message supplied to
  this reviewer explicitly approves the tested default for requests that omit
  `top_k`. The exact message appears in
  `doc/vllm_integration/serving_default_approval.json`, beside the serving
  command in `doc/vllm_integration/README.md`, and in the appended
  `doc/vllm_integration/work_log.md` entry. The JSON record and
  `doc/full_model/serving_contract.json:current_stage9.serving_default_approval`
  record the same approved scope and date. This is separate from Entry 1.
  A read-only comparison establishes that the shell-parsed README command,
  approval-record argv, and
  `readiness_vllm/final64_b32_untracked/vllm_run_summary.json:server_command`
  are exactly equal, including `--override-generation-config '{"top_k":20}'`.
  The model-card rationale is explicitly attributed to the owner; the delta
  makes no new independent model-card verification claim. The prior finding
  required no implementation change or hardware rerun following acceptance.

## Other Concerns

- Prior strict forced-preemption parity concern: **control retained**.
  Round 1's canonical sampled/greedy and aligned151 controls remain the basis
  for classification. The approval delta does not relabel strict parity as a
  pass, establish harmless numerical tolerance, or claim a recomputation fix.
  The existing `Carried to next stage` entry retains investigation of canonical
  prefill/decode numerical sensitivity. Exact recomputation equivalence remains
  an open investigation, with no specific adapter defect established by round 1.
- Prior disabled dense batched-prefill concern: **control retained**.
  The serial path and completed burst remain the accepted Stage 9 scope.
  Batched prefill remains an optional later optimization, conditional on
  batched-versus-solo proof. The changed documents do not claim a new
  batched/serial performance ratio or broaden the measured capability.
- Prior arbitrary-many-equal-local-maxima and physical-page-telemetry concerns:
  **control retained**. Both remain explicit next-stage ideas in the work log.
  This delta does not extend the tested greedy/capacity64/page-growth contracts
  or claim previously unavailable per-request physical-block telemetry.
- Completion bookkeeping is intentionally pending at review time. The README
  and current contract still distinguish owner approval from this follow-up
  verdict; historical round-1/pending entries are not new blockers. The stage
  owner should record this result and the final local documentation checkpoint
  when closing the stage, as required by the stage-review workflow.

## Hard-Check Gaps

- No new gap is introduced by the approval-only delta. The prior completed
  checks and runtime evidence were not rerun or re-reviewed. There is no
  production-source, command-recipe, raw measurement, or runtime-gate change
  requiring new device evidence.
- A supplementary `doc/vllm_integration/closure_process_cleanup.json` appeared
  during review. It records four empty `fuser` results with exit 1, no serving
  processes, and no device open/reset. The stage owner identified it as a fresh
  read-only `fuser`/`ps` audit. It is cleanup corroboration, not replacement
  correctness or performance evidence; this reviewer did not run that audit.

## Anomaly Ledger

- Observed anomaly: The tested omitted-`top_k` default had no explicit operator
  acceptance at round 1.
  Evidence: Round 1 Required Work; exact owner quote in the three current
  approval records; matching actual B32 launch argv.
  Affected path: Serving requests that omit `top_k`.
  Control or comparison: Owner message compared verbatim with recorded text;
  README shell argv compared with approval JSON and actual launch summary.
  Likely subsystem: Serving-default authorization and documentation.
  Investigation performed: Read-only diff, JSON parsing, shell argument
  parsing, equality checks, and referenced-artifact existence checks.
  Resolution: fixed by explicit owner approval and its accurate recording.
- Observed anomaly: Previously documented strict forced-preemption trajectory
  differences remain unresolved as exact recomputation equivalence.
  Evidence: Round 1 Other Concerns and unchanged next-stage work-log note.
  Affected path: Reduced forced-preemption diagnostic modes.
  Control or comparison: Round 1's canonical sampled/greedy and aligned151
  controls, retained unchanged.
  Likely subsystem: Canonical prefill/decode numerical sensitivity.
  Investigation performed: Checked that the approval delta preserves the
  previous classification and does not assert a parity repair.
  Resolution: controlled; next-stage investigation retained.

## Scope Inspected

- Goal/skill paths: `/home/ttuser/ornith-9b/multigoal_logs/09-09-vllm.prompt.txt`;
  installed `tt-model-bringup/0.1.24/skills/stage-review/SKILL.md`;
  `model-bringup/SKILL.md` startup instructions;
  `vllm-integration/SKILL.md` and relevant served-default guidance in
  `references/eval-readiness-requests.md`. Installed plugin metadata and host
  configuration confirm version 0.1.24 and enabled `tt-autodebug` dependency.
- Previous findings: `doc/vllm_integration/stage_review_round1.md`, precisely
  its Required Work and Other Concerns sections. This was not a whole-stage
  re-review.
- Changed artifacts: `doc/vllm_integration/README.md`,
  `doc/vllm_integration/work_log.md`,
  `doc/vllm_integration/serving_default_approval.json`,
  `doc/full_model/serving_contract.json`; supplementary cleanup JSON above.
- Comparison artifacts:
  `readiness_vllm/final64_b32_untracked/vllm_run_summary.json`;
  existence of `doc/vllm_integration/final64_measurement_execution/environment.json`
  and `commands.sh`; retained `Carried to next stage` work-log entry.
- Code paths: None changed or reviewed in this follow-up.
- Commands run: Read-only `cat`, `sed`, `rg`, `git status --short`, `git diff`,
  `git diff --stat`, `git diff --name-only`, `git ls-files --others
  --exclude-standard`, `git rev-parse HEAD`, and `git branch --show-current`;
  standard-library Python JSON/`shlex`/path comparisons. Only this review
  report was written. No TT imports, hardware access, server, reset, test,
  benchmark, or push was performed.

## Residual Risk

- The unchanged controlled concerns remain bounded by round 1's evidence and
  the existing next-stage notes. This authorization closes no numerical or
  performance issue and introduces no new capability claim.
- Follow-up review relies on round 1 for the unchanged implementation and
  runtime gates. Completion now requires the ordinary verdict/checkpoint
  bookkeeping, not another review or hardware run.
- Push state follows the supplied run plan: stage checkpoints are local;
  the stage agent must not push. No push is requested or performed here.
