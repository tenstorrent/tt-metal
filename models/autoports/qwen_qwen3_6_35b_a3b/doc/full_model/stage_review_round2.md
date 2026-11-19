# Stage Review

Verdict: clean-pass

Independent round 2 of 3, 2026-11-16, for stage 6/full-model,
`Qwen/Qwen3.6-35B-A3B`. This is a diff-only follow-up against
`35169ad54c7acbf0016ab155c5a4c08f9b1ec016` on `main`, in the live worktree.
The original goal is amended by the explicit owner decision reproduced in
[owner_decision.md](owner_decision.md). Acceptance of the recorded numerical
limitation closes the previous blocker; this verdict does not assert that PCC
improved or that the original numerical check passed.

## Required Work

None under the owner-revised contract.

**Previous P1 — closed: Required resumed-prefill acceptance is still failing;
the corresponding serving contract remains incomplete.** The raw numerical
failure remains, while its acceptance disposition has changed by user authority.

- I compared the runner message at
  `/home/ttuser/qwen36-35b-a3b/multigoal_logs/06-06-full-model.resume-1.message.txt`
  with the decision copied into [owner_acceptance.json](owner_acceptance.json)
  and [limitations.json](../eval_readiness/limitations.json). The text matches.
  The owner explicitly accepts the PCC failure and requests no further work on
  this gate. The limitation has `original_model: "same"`, existing evidence
  paths, and a waiver scope confined to the three recorded PCC failures.
- All seven case copies in the acceptance record match the
  [raw TT result](../../readiness/prefill_resume_check.failed.json) exactly.
  The result covers layers 0 through 39 at context 2048. All seven top-1
  criteria meet the unchanged 0.98 threshold, and every final top-1 token
  matches. Length 257/start 128 has 64/65 agreement, not every-row equality.
- I independently evaluated all ten stored negative-control metrics: both
  `no_history` and `fresh_suffix` fail the unchanged criteria at
  257/32, 257/128, 1025/32, 1025/128, and 1025/260. The two singleton cases
  have no applicable controls under the recorded minimum-window rule.
  Prepared and final program-cache counts are both 9602.
- The [HF result](qualitative/hf_resume_control/result.json) uses the pinned
  original model, default BF16 CPU execution, the same natural-book token
  construction, selected positions, and numerical helper criteria. Recomputing
  its mean/minimum PCC from the saved per-position arrays reproduces its
  summary. The three corresponding case-level PCC failures are:

  | Length / start | TT mean / minimum PCC | HF mean / minimum PCC |
  | --- | --- | --- |
  | 257 / 128 | 0.9935122896 / 0.8585700580 | 0.9870273797 / 0.4963426935 |
  | 1025 / 260 | 0.9931221526 / 0.9686653286 | 0.9916174265 / 0.9528563800 |
  | 1025 / 1024 | 0.9870205088 / 0.9870205088 | 0.9884654890 / 0.9884654890 |

- The raw TT and HF results and HF verdict are byte-identical to the review
  base. Their SHA256 values match the preserved HF verdict. The original
  review is also unchanged. No passing `readiness/prefill_resume_check.json`
  exists. The acceptance record is expressly an owner disposition over existing
  results, not a new device run or numerical pass.
- All four formerly missing rows in
  [serving_contract.json](serving_contract.json) are now `supported` with the
  explicit acceptance evidence: `prefill_logits_continuation`,
  `supports_chunked_prefill`, `chunked_prefill`, and `resumed_prefill`.
  Independently counted totals are 56 supported, 30 not supported, and zero
  missing. Every supported row's evidence path exists.
- Static AST inspection of [serving_capabilities.py](../../tt/serving_capabilities.py)
  confirms `supports_chunked_prefill=True`, no pending capability entry, and
  `supports_prefix_caching=False`. `QwenGenerator` inherits this declaration.
  The prefix-cache exclusion still names `recurrent_state`; accepting retained
  recurrent/KV continuation does not advertise automatic KV-only prefix reuse.
  The existing continuation and absolute-position prefill implementations are
  unchanged. This declaration file is the only changed code file in the diff.
- The [CPU stage06 log](logs/owner_acceptance_stage06.log) and its
  [exit-code artifact](logs/owner_acceptance_stage06.exit_code) report exit 0:
  degeneracy, context 262144, and the 86-row serving inventory pass. Inspection
  of the installed shell checker confirms these are artifact/contract checks;
  they do not rerun numerical prefill. The historical exit 2 and failed numerical
  result remain preserved. Applicable hooks pass in
  [the final pre-commit log](logs/owner_acceptance_all_precommit_final.log).

No new correctness or acceptance defect was found in the changed files.

## Other Concerns

Disposition of every previous Other Concern:

- **HF continuation sensitivity — closed as an acceptance finding; retained as
  a known limitation.** The owner record correctly limits the claim to matching
  case-level numerical failures. Exact TT/HF worst-row correspondence in the
  two multirow cases remains unestablished. The HF control is one deterministic
  observation per case reused under ten seed labels, not ten independent
  sampled quality trials. Neither stronger claim was added.
- **Qualitative smoke scope — closed/no new work.** No prompts, generated
  outputs, or qualitative verdicts changed in this follow-up. The round-1
  assessment remains applicable: six paired 128-token reasoning-prefix smokes
  do not establish final-answer accuracy or distributional quality. This
  follow-up does not repeat that whole-stage output review.
- **Performance alternatives — open as existing nonblocking hand-offs.**
  The prior rejection ledger and owner-accepted stage-5 limitation are unchanged.
  Unmeasured embedding/head and terminal alternatives remain candidates, with no
  newly claimed performance result or requirement for another experiment here.
- **Historical documentation wording — open as a nonblocking cleanup note.**
  Earlier source comments and nested evidence snapshots retain historical
  pending/missing language. The current owner update, raw outcomes, and current
  serving counts establish the applicable disposition. Historical HF records
  saying the gate was not waived are preserved pre-decision records, not a
  contradiction of the later explicit owner decision.

## Hard-Check Gaps

- No new required gap in this follow-up. Stage06 exit 0 establishes the CPU
  checks listed above; it does not establish a numerical repair. The owner
  decision supplies the acceptance authority for the preserved PCC failure.
- Reduced layers 0/3 replay, Watcher, and serving-command evidence remains
  scoped to its recorded geometry. It is not upgraded into an all-40-layer
  server validation by promoting the declaration. No vLLM validation is claimed.

## Anomaly Ledger

- Observed anomaly: Three retained-state resumed-prefill cases miss the original
  PCC criteria while all seven top-1 criteria and ten negative controls pass.
  Evidence: `readiness/prefill_resume_check.failed.json`,
  `doc/full_model/qualitative/hf_resume_control/result.json` and `verdict.json`.
  Affected path: All-40-layer uninterrupted versus retained-state resumed
  natural-book prefill, lengths 257/1025 at context 2048.
  Control or comparison: Original pinned HF reproduces the corresponding
  case-level numerical failures under the same numerical criteria.
  Likely subsystem: Full-versus-split numerical sensitivity; no specific TT
  root cause or exact multirow failure-position equivalence is established.
  Investigation performed: Existing HF control and unsuccessful AutoFix are
  preserved. This review re-derived stored metrics, controls, counts, hashes,
  declaration values, and the scope of the explicit owner decision.
  Resolution: controlled; accepted known limitation under the revised owner
  contract, with raw failure and thresholds unchanged. No further work on this
  gate is requested.

## Scope Inspected

- Goal/skill paths: `doc/full_model/stage_goal.txt`, `owner_decision.md`, the
  runner decision message, and `stage_review_round2_brief.md`; installed
  tt-model-bringup 0.1.24 `stage-review/SKILL.md`, startup requirements, and
  relevant `full-model`, `qualitative-check`, and `tt-enable-tracing` sections.
  The brief contains the prior Required Work and Other Concerns verbatim.
- Artifact paths: Every changed root/full-model README, acceptance/checklist,
  limitation, wrap-up, runtime-audit, serving-contract, brief, and work-log file;
  owner-acceptance validation logs; the unchanged raw TT resume result and HF
  control result/verdict/source; supporting
  `generator_replay_probe/lifetime_fixed_watcher.json`,
  `prefill_equivalence/resume_legacy_parity_drain.json`, and
  `serving_commands_probe/result.json`.
- Code paths: The complete declaration diff in `tt/serving_capabilities.py`,
  generator inheritance and continuation dispatch, the relevant prefill/serving
  implementations, and the installed `06-full-model.check.sh`.
- Commands run: Read-only `rg`, `cat`, `sed`, `git status`, `git diff` against
  the supplied base, `git show`, and small Python standard-library analyses
  using JSON, AST, SHA256, path checks, and arithmetic over existing artifacts.
  No target imports, tests, device use, resets, servers, or vLLM runs. Only this
  review report was written.
- Push state: The supplied run plan permits a local acceptance checkpoint after
  review and specifies no push. This review neither commits nor pushes.

## Residual Risk

- The accepted PCC failures still exist; this verdict cannot be read as stronger
  numerical equivalence or a repaired cache/position algorithm.
- The HF control does not establish a statistical quality waiver or exact
  multirow TT/HF error correspondence. The limitation record states both limits.
- Serving integration must preserve retained recurrent/conv state with KV and
  positions, the documented trace preparation/lifetime contract, and the
  automatic prefix-cache exclusion. Actual vLLM validation remains a later stage.
- The stage owner still needs the routine final review-status bookkeeping and
  local acceptance commit. Those follow this passing review and require no new
  device evidence or another review round for bookkeeping alone.
