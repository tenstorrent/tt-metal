# Stage Review

Verdict: clean-pass

Independent rereview of the sole required-work item in `interim-phase12.md`: Astropy reference reproduction. This is limited to the completed phase 1/phase 2 scope, not final extended-stage closure. Current checkout HEAD is `30d59bb033bd09472618ea32be3d9b36f46dfefe`; HF control execution is pinned to `67f50e0d140100d3c478829fd056fba83c0cdfb2` with source hashes verified against the current files. Paths below are relative to `doc/agentic_evals/anomalies/swe-astropy-precision/` unless stated otherwise.

## Required Work

The prior Astropy HF finding is resolved by positive reference reproduction on both independently reproduced shortened diagnostic inputs. No further model run is required for this diagnostic decision. No required work remains in this completed-phase scope. This does not close extended stage 12.

## Evidence and Decision

- `short-hf-batch/hf.exit` is 0. Both actual HF result files have status complete, one seed-42 continuation of 512 generated IDs, and matching streamer counts. Reference revision is `842da3794eaa0b77d5f08bae87a17459d91ff475`, BF16 CPU. Temperature 1, top_p 0.95, top_k 20 match the diagnostic TT recipe.
- Verified all five source/case hashes recorded by the launch command. Each HF case hash matches its saved 512-token input. Each served/full TT result names that same case file. Saved chosen-token sequences match all 64 teacher-forced positions exactly.
- Read both complete HF texts. First case repeatedly cycles through checking lines, performing an edit, switching between sed and Python, and running reproduction; “Wait, I'll check the lines first.” occurs five times, with repeated multi-paragraph command-planning blocks. Second case repeats the same `super().__getattribute__(attr)` intention and “Let's go” cycle twenty times. These are generated text cycles with no tool execution and no sustained progress. Both have detector flag false; that does not contradict semantic degeneration.
- Read the corresponding served/full shortened TT continuations. They reproduce the same type of within-response planning/phrase repetition. The first HF continuation closely resembles the full-TT first-case planning cycle; the second reproduces the full-TT alternating phrase cycle. This establishes positive reference reproduction for both diagnostic inputs.
- Independently recomputed 64-position HF/TT top-1 agreement and chosen-sequence log-probability deltas (HF minus TT): first served **62/64, +3.2699904357**; first full **64/64, -0.0183779946**; second served **61/64, +13.9580245549**; second full **61/64, +5.1201648228**. These match `short-hf-batch/tf_comparison_summary.json`. Numerical agreement supports the comparison; the actual free-running texts establish the semantic reproduction.
- The original long-prefix HF attempt remains exit 124, with no completed free-running sample. It is not reclassified as successful. The shortened inputs remove earlier history and reset position origin, and reconstruct an onset whose native sampled IDs were unavailable. Neither exact original-request equivalence nor equal anomaly incidence is established.
- The datatype-sweep Anomalies procedure explicitly permits shortening a prefix while it still reproduces. Both shortened cases were reproduced in served/full TT before their matching HF runs. Positive repetition in one HF sample per case establishes existence; additional seeds are not necessary to turn these positive observations into a diagnostic reproduction. This does not justify estimating a population failure rate from one sample or claiming that precision cannot affect earlier reasoning.
- No precision change is selected from these controls, so no additional precision-triggered list rerun or rate remeasurement follows from this finding. Original Astropy failures, two capped responses, first-attempt score, and measured timing remain intact.

## Documentation Crosscheck

The reconciled `doc/agentic_evals/results.json`, `REPORT.md`, and `learned.json` now record phase 2 pass with unchanged 4/10 first-attempt score, 1.4036113561 measured hours and approximate 1.5230584178 explained hours. They explicitly retain the native task failure, both capped replies, the long-prefix timeout, one seed per shortened HF case, detector-negative semantic repetition, and limits on equivalence and incidence. Pipeline status remains in progress and extended traffic is not started. The documented decision matches the raw evidence reviewed here.

## Other Concerns

- The first independent triage labels the text `looping_actions`; the coordinator explicitly corrects this to `anomaly` because no actions were executed. The underlying evidence is consistent. The second triage calls it `repeating_tokens`, also classified as anomaly by the coordinator. These labels do not conceal a progressing limit cut.
- Immutable case metadata retains its preparation-time `untested_candidate` status. The completed execution files, comparison summary, and reproduction records supersede that historical preparation status; changing hashed case inputs retrospectively is unnecessary.
- The launch manifest now discloses overlapping Docker preparation. HF elapsed time is diagnostic cost only and is not used to rank serving performance.
- Historical concerns from the first interim review remain: missing SWE prelaunch full-list forecast and initial preparation-network deviation. This rereview does not manufacture a retrospective prediction or erase either record.

## Hard-Check Gaps

No additional hard-check requirement is introduced. Mechanical flag false is insufficient to clear a reply, and semantic inspection was performed. Extended admission, executed extended results, final telemetry and local closure commits remain outside this review.

## Anomaly Ledger

- Observed anomaly: two Astropy capped native replies and repeated planning/phrase cycles in TT controls.
  Evidence: actual `short-hf-batch/results/astropy.{first,second}.suffix512.hf.private.json`, matched served/full TT results specified by `short-hf-batch/commands.json`, and retained long-onset evidence from the first review.
  Affected path: reasoning text before tool execution.
  Control or comparison: exact same saved shortened diagnostic IDs at served TT, full BF16/HiFi4 TT, and pinned BF16 HF; 64-position teacher forcing and free-running text.
  Likely subsystem: reference-model behavior is demonstrated on these diagnostic inputs; exclusive native-run causality and incidence equivalence are not established.
  Investigation performed: independently read actual HF and TT continuations; verified case/source hashes, completion/exit status, token counts and chosen-token identity; re-derived numerical comparisons; inspected coordinator and fresh independent triages.
  Resolution: controlled. This closes the previous required-work item for the reviewed completed-phase scope.

## Scope Inspected

- Contract: original stage-review, agentic-serving and datatype-sweep requirements and prior `review/interim-phase12.md`; no change to the original fixed-list scores or timing gates.
- Raw artifacts: short-HF launch/commands/cases/exit, both complete HF result files, all referenced served/full shortened controls, independent triages, coordinator decisions, TF summary, and retained long-prefix timeout evidence.
- Code: source hashes for `tools/precision_repro_hf.py` and `tests/generate_full_reference.py` match the executed manifest. No new implementation review or approval of later unrelated changes is implied.
- Commands: read-only `rg`, `git rev-parse`, Python JSON inspection, SHA256 validation, phrase counts, exact-token comparisons and arithmetic. No device, server, model, benchmark, Docker or network runs. Only this authorized report was written.
- Push state: local-only operator contract unchanged; reviewer performed no commit or push.

## Residual Risk

A shortened single-seed reference reproduction is a bounded control, not proof of exact long-prefix causality, equal incidence, or general absence of TT defects. Existing first-attempt scores remain the evidence for scored quality. New material anomalies in extended traffic require their own triage. A clean result for this interim scope cannot close extended stage 12.
