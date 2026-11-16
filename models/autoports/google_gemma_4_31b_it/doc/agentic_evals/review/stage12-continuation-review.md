# Stage Review

Verdict: more-work-needed

Independent continuation review of stage 12, google/gemma-4-31B-it, live branch `v5-extended-run`, supplied serving HEAD `ac79c7b5b83e77148a2248bc05c4fa4427f9ac26` plus closure edits. Earlier evidence is covered by `stage12-final-review.md`; this review checks the authorized four-hour continuation and its closure. The operator's conditional closure instruction is satisfied: candidate validation, actual launch memory and relevant serving rates were collected, and the retained85 rerun was not admitted because its projection does not fit. This is an honest bounded closure, not extended-mode release readiness. No further experiment or request for more time is required or authorized by this review.

## Required Work

- P1: Extended SWE task quality remains below its reference band; no improved scored policy is demonstrated.
  Evidence: `current/swe-extended85-head-bfp8/collected-final/final-score-with-adjudication.json` retains 85 rows, 23 confirmed passes, 59 definitive unsuccessful outcomes and three unknown outcomes. Even 26/85 has a Wilson upper bound of 0.410458, below the 0.648 reference. The continuation validates full-BF16 serving, but `current/continuation-full-bf16/retained85-projection.json` records `admit:false`; no new scores exist. `review/continuation-gate-check.json` exits 2 and reports both the missing scored precision-fix rerun and unknown `score.solved`.
  Why this matters: diagnostic improvements and serving measurements do not establish task accuracy or satisfy the extended-mode pass contract. Retaining the scored head-BFP8 default does not cure its observed task failures.
  Required next step: retain **not passed yet**, the unchanged first-attempt accounting, and this **more-work-needed** verdict in the final report and checkpoint. Any future quality-repair evaluation belongs to separately authorized work; do not launch it or ask for additional time in this run. The current operator instruction explicitly permits closure without this rerun and does not turn the missing result into a pass.

- P2: Three Requests verifier outcomes remain unresolved.
  Evidence: the same accounting artifact preserves null outcomes for `psf__requests-1724`, `psf__requests-1766` and `psf__requests-2317`; `score.solved`, point value and point interval remain null. Existing recovery limitations are retained in the prior review and report.
  Why this matters: a complete point score cannot be asserted. Resolving these grades alone cannot put the suite inside the reference band.
  Required next step: preserve the 23–26/85 bounds and explicitly distinguish their Wilson envelope from an observed-score interval at closure. Faithful verifier recovery is future work, not permission to invent grades, drop tasks, modify the harness or extend this run.

## Other Concerns

- No new substantive defect was found in the continuation's decision to reject KV-only BFP8 or withhold the rerun. The prior time-authorization blocker has been answered; it must not be reopened.
- Full-BF16 broad acceptance inherits the previously reviewed wrapper143/unknown-child-exit qualification. Completed output evidence is usable with that qualification; repeating it merely to obtain exit0 is unwarranted.
- The seven rate rows contain historical `predicted` fields as well as the actual continuation prediction sidecars and comparisons. Use the sidecars for the continuation forecast; the historical fields are not the forecast that earned the reported range hits.
- Planning uses a shared mesh allocator measurement uniformly applied across four devices, not four independent physical allocator samples. This limitation is explicit. The conservative 131072-token planning working set is not a reduced served context.

## Hard-Check Gaps

- The saved gate is not green. Both critical findings are now disclosed in REPORT and the gate interpretation; the operator's conditional no-rerun instruction allows truthful closure, not checker success. No checker/schema modification is needed to conceal this result.
- Final runner-owned usage and the new evidence checkpoint SHA are written at terminal closure. Verify those records after the runner writes them; do not fabricate terminal usage or claim the older checkpoint contains the new continuation. Local commit only; no push, CI or GitHub action is authorized.
- Telemetry retains the scored 4096/128 B1/C1 vLLM headline (755.631794ms TTFT, 34.770637 tokens/s/user); unavailable device metrics stay null. Candidate B8/C1 rates are separate evidence, not a replacement headline.

## Anomaly Ledger

- Observed anomaly: KV-only BFP8 regresses two matched diagnostic continuations to caret loops.
  Evidence: `setup/continuation-kv-bfp8/fixed108/runs/`, indices37 and42; index42's extended output; `root-semantic-review.json` and `decision.json`.
  Affected path: saved-history generation with only KV reduced to BFP8.
  Control or comparison: same-fixture/seed full-BF16 panel and completed extensions from the previously reviewed baseline.
  Likely subsystem: precision-sensitive generation; this does not prove a general native incidence rate.
  Investigation performed: rehashed all125 admission pins and108 original result files with no mismatches; all108 runtime policies agree. Read actual regression outputs. Recounted88 normal/20 motifs/0 other/0 inconclusive. Compared all three extension token prefixes, each exact; extensions have1024,494,1033 tokens. The first remains a loop, while the latter two finish with tool handoff.
  Resolution: controlled, candidate rejected. No selected-policy cure claimed.

- Observed anomaly: case55 initially marked normal despite contraction duplication.
  Evidence: actual continuation begins `’ve tried everything I can`; the root review records the exact saved contraction boundary and corrected verdict.
  Affected path: diagnostic classification.
  Control or comparison: identical reviewed full-BF16 token sequence and its motif classification.
  Likely subsystem: coordinator classification error.
  Investigation performed: read raw continuation and correction record, verified final counts include the motif.
  Resolution: fixed before selection; no task grade changed.

- Observed anomaly: retained85 cannot fit the remaining authorized window on the measured candidate configuration.
  Evidence: `current/continuation-full-bf16/retained85-projection.json`, seven raw benchmark rows and actual launch sizing.
  Affected path: admission planning, not an executed evaluation.
  Control or comparison: primary three-slot replay19.968950h, optimistic C1-rate replay7.877950h, remaining3.521644h; even five-slot output-only sensitivity3.928307h.
  Likely subsystem: full-precision memory capacity and throughput.
  Investigation performed: verified source calls hash, recomputed1202253/(5*17.00270371035)/3600 =3.928307h, and checked all seven raw rows completed with zero failures. Every recorded decode rate equals1000/raw mean_tpot_ms. Read the projection's concurrency/length extrapolation and non-self-consistent scheduling qualifications. This is an evidence-backed forecast, not a physical lower bound for changed trajectories.
  Resolution: controlled admission decision; rerun not done as directed by the operator.

- Observed anomaly: original extended SWE remains below band with three unknown grades.
  Evidence: final85-row adjudication and unchanged score bounds.
  Affected path: task quality and external verification.
  Control or comparison: previous review's served/full/HF investigations and this continuation's candidate controls; no scored full-BF16 comparison.
  Investigation performed: checked final counts and null-score status against REPORT, results and learned records.
  Resolution: more-work-needed for release readiness; bounded run closes honestly without a pass claim.

## Scope Inspected

- Goal/skill paths: `bringup/run_contract.json`, `bringup/agentic_evals/inputs.json`, supplied AGENTS instructions; installed stage-review, agentic-serving, datatype-sweep and tt-device-usage skills. The latest operator instruction supersedes older requests for a12-hour window and stale run-contract infrastructure descriptions.
- Artifact paths: `REPORT.md`, `results.json`, `learned.json`, `work_log.md`, prior final review; continuation KV candidate admission/results/extensions/decision and memory plan; full-BF16 actual launch sizing, logging check, runtime summary, rate sidecars/raw benchmark results and retained85 projection; canonical benchmark rate catalog; saved gate; current telemetry packet.
- Re-derived evidence:125 source hashes and108 result hashes match; all three original extension prefixes preserved. Seven rate rows complete32 measured requests, zero failures and4096 generated tokens. Raw server log independently yields645 host-prefill,62 device-prefill and2479 device-decode steps, no host-decode steps. Launch records the autoport adapter,60 layers, pinned revision, full precision and262144 context; READY probe reports19 input/2 output tokens and stop. Pool10752 IDs consumes14092861440B/device against16059728896B free; contiguous free15631973376B, recorded reserve968507904B and additional512MiB headroom leave33733120B. Capacity is three planning agents on smallest configured validated batch8.
- Code paths: no new implementation changes reviewed beyond the earlier independent review's baseline; inspected continuation artifact provenance and benchmark calculation inputs. No claim of a complete new kernel audit.
- Commands run: read-only cat/git status and small Python standard-library JSON/hash/arithmetic analyses. No device, server, HF, task-container, network or long-test execution. Only this review file was written.

## Residual Risk

Saved-history diagnostics cannot establish spontaneous anomaly incidence or SWE task success. Full-BF16 may change trajectories and costs; the projection retains old histories and explicitly labels its approximations. The previous broad-process status limitation and three verifier unknowns remain. The continuation is sufficient for the operator-directed bounded closure but does not satisfy extended-mode release readiness. Preserve the more-work-needed verdict and end without further time requests.
