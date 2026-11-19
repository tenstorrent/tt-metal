# Stage Review

Verdict: more-work-needed

Independent follow-up **round 2 of 3**, stage 5 `optimized-multichip-decoder`, Qwen/Qwen3.6-27B. This is a diff-only review of the round-one findings and the listed documentation/manifest changes. No new correctness, accuracy, missing-device-evidence, or implementation finding was identified. The documentation remedies are closed; the performance uncertainty and reporting-tool correction remain nonblocking carry items. The installed skill reserves `pass-with-carried` for round 3, so the remaining work is review closure, not further optimization or device testing.

The live checkout remains `main` at `8502f42bca20dcad1507fbc8cebf7032220b7d8e`. **D** below means `models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/`.

## Required Work

- **No model correctness, accuracy, or missing required device evidence work.** The previous round's disposition remains applicable. The six tracked files changed after round one are documentation or manifest metadata; the manifest changes only `_note` and `review`. The decoder still hashes to `2d4ba9dd20c86f6ae9d710c5fe79c259bdd66fe6007b6af4be5437ca01554c0b`. This follow-up did not re-review the complete stage or independently repeat round one's full numerical/source audit.
- **P2: Finish the prescribed review and checkpoint sequence — open, procedural.**
  Evidence: `D/final_manifest.json` correctly records round one as `more-work-needed`, one report, nonblocking carried items, and `local_commit.sha: PENDING`. The outstanding substantive items below are still carried, not fixed. Round-two `pass-with-carried` is unavailable under the installed stage-review rule.
  Why this matters: neither a failed statistical confidence check nor a review still in progress may be silently converted into an unconditional stage pass.
  Required next step: record this report and obtain the final fresh, diff-only round-three review. If no correctness or accuracy defect remains, that review may accept the recorded nonblocking carry items. Preserve the original `final_gate.json`; record review acceptance separately. Then create and log the authorized local checkpoints. No new hardware run, optional repetition, source change, threshold change, or push is requested.

## Other Concerns

- **P2: B1 prefill confidence — documentation closed; uncertainty open as nonblocking carry.**
  The README, handoff, selection, work log, and manifest consistently retain an inconclusive result for both layer kinds. I recalculated all five pairs per kind from the unchanged twenty raw timing JSON files: GDN means are **2.395564131 / 2.194059826 ms**, median final-minus-control difference **−0.216594121 ms**, maximum **+0.037668118 ms**; full-attention means are **1.761019820 / 1.701234902 ms**, median **−0.062272574 ms**, maximum **+0.057086038 ms**. Four of five pairs favor final in each kind. The preserved statistics report the maximum-based bound with **0.96875** coverage. No documentation claims that zero-margin nonregression passed or that attention passed because the checker stops first at GDN. The prespecified future confirmation series remains optional work for a later authorized stage. This is not evidence of a demonstrated population regression or a numerical gate waiver.

- **P2: DRAM-sharded compute annotation — disclosure closed; reporting correction open as nonblocking carry.**
  New `D/final29_profiler_limitations.md:3` and `:5`, the README, and work log explicitly exclude the affected `Cores`, `FLOPs [%]`, and compute-bound labels from utilization and rejection claims. The unchanged example CSV row is `32 x 5120 x 7168`, eight reported cores, **175.65269384774476%** FLOPs. Raw attributes select two workers per DRAM bank; the native factory derives its actual worker set from reader assignments, whereas installed `tt_perf_report/perf_report.py:1337` substitutes the architecture's fixed count. The two-reader example therefore uses sixteen compute workers while its report models eight. The explanation matches the inspected source and row. All six final decode CSVs have empty Advice columns, consistent with the new note. The docs retain measured latencies and separately derived byte-based accounting without asserting a 70% stop-rule result. Tool correction remains future work; no profiler rerun is needed to classify this limitation.

- **Untried optimization and lint inventory — disposition closed; experiments/cleanup remain carried.**
  `D/work_log.md:522` already records `Carried to next stage`, including final-fidelity geometry, norm capacity, recurrence compositions, long-prefill movement, and import cleanup. The revised handoff still distinguishes untried alternatives from measured rejections, and the selected minimal/DiT norm path remains off. Frozen measured sources were not reformatted. No new experiment is required under the user's cap.

- **Stale preparation wording — requested status correction closed, with a minor remaining wording note.**
  The manifest now records completed measurements/validation and the actual round-one verdict. The changed inter-layer paragraph agrees with `final29_capacity_update.log`, the capacity report, and the context contract's validated stage section: support remains 262144 with a final 32K smoke and inherited maximum-context evidence. The plan contains 28 timing jobs and six profiles, consistent with the revised selection text; the validation summary records 68 jobs and `pass: true`. Some unchanged earlier paragraphs still use preparation tense (for example, “Final29 must rerun” in the inter-layer note), while the final status paragraph records completion. This is a wording note, not missing evidence or grounds for another device run. At final closure, consolidate the profiler correction with the other items under the existing `Carried to next stage` heading; its disposition is already explicit in the new round-one work-log paragraph.

## Hard-Check Gaps

- **Original checker failure preserved — closed.** `D/final_gate.json` is byte-identical to its staged pre-review artifact (SHA256 `0ddbb0607936b7033cb1e7e8c8b7d57fe5348d2c8b9dbc41a60e8f8503105ea4`): `pass: false`, 25 checked groups, and the original `b1_prefill_regression` and `review_commit` errors. Both B1 summary files and all twenty raw pair JSONs also match their staged pre-review bytes. The updated manifest does not relabel that checker result.
- The changed documents introduce no new gate, alter no acceptance threshold, and supply no replacement numerical evidence. The unchanged defensive mixed-list count check, representative-weight resource limitation, and runtime-sentinel scope recorded in round one remain unchanged; this diff does not reopen or resolve them.
- Raw profiler CSVs are present but are not in the Git index, so this review did not establish their pre-round byte identity by an index comparison. The specific CSV/source explanation above was checked directly; this is not a missing-profile finding or a request to regenerate evidence.

## Anomaly Ledger

- Observed anomaly: positive B1 confidence bounds despite improved mean timings.
  Evidence: all `D/final29_pair_l{0,3}_p{0..4}_{control,final}.json`, `D/final29_b1_statistics.json`, and `D/final29_b1_uncertainty.json`.
  Affected path: statistical inference for warmed B1/S512 prefill timing.
  Control or comparison: five fixed alternating control/final process pairs per layer kind.
  Likely subsystem: sampling uncertainty; these artifacts do not isolate its cause.
  Investigation performed: recalculated means, paired medians, maxima, and favorable-pair counts; checked unchanged artifacts and revised claims.
  Resolution: **controlled as disclosed nonblocking uncertainty**; not a confidence pass. Final carry acceptance awaits round three.

- Observed anomaly: reported DRAM-sharded compute utilization exceeds 100%.
  Evidence: the cited GDN B1 decode CSV/raw attributes, native worker construction, and reporting-tool core-count override.
  Affected path: compute-core/FLOP and derived compute-bound annotations.
  Control or comparison: selected two readers per each of eight banks versus fixed eight-core reporting.
  Likely subsystem: profiler accounting model.
  Investigation performed: checked the example row and source explanation against the added limitation document; verified the six decode Advice columns are empty.
  Resolution: **controlled by explicit reporting limitation**; correction carried, measured latency retained.

- No new anomaly was introduced by the inspected documentation/manifest changes. Earlier numerical, trace, warning, and excluded-candidate anomaly dispositions remain as recorded in round one; they were not re-reviewed here.

## Scope Inspected

- Goal/skill paths: original `/home/ttuser/qwen36-27b/multigoal_logs/05-05-optimized-multichip-decoder.prompt.txt`; installed 0.1.24 `stage-review/SKILL.md`, `model-bringup/SKILL.md` startup, and `optimize/SKILL.md`; `D/final29_review_brief.md`, `D/final29_review_round2_brief.md`, and `D/final29_review_round1.md`, including the quoted latest user cap. Enabled dependency inventory and packaged environment resolution were checked.
- Changed artifact paths: `D/README.md`, `D/work_log.md`, `D/final29_interlayer_contract.md`, `D/final29_handoff.md`, `D/final29_selection.md`, `D/final_manifest.json`, and new `D/final29_profiler_limitations.md`.
- Targeted supporting artifacts: original final gate, B1 pair/summary JSONs, final plan/validation summary, capacity report/update log, adjacent context contract, GDN B1 raw ops/decode CSV, and the other five final decode CSV Advice columns. These reads verify the previous findings and changed claims only.
- Code paths: bounded inspection of the DRAM-sharded matmul factory's worker-count construction and installed reporting-tool override; decoder SHA256 only. No implementation was edited.
- Commands run: read-only `git diff`, `git show`, `git rev-parse`, `git branch`, `rg`, `cat`, `sed`, `nl`, plugin inventory/environment inspection, and small standard-library JSON/CSV/hash/statistics scripts. One attempted index comparison correctly reported that raw profile CSVs are not staged; direct inspection continued. Only this requested report was written. No hardware, server, test, benchmark, reset, or subagent was started.
- Push state: the supplied run plan authorizes local checkpoint commits after passing review and prohibits stage-agent push. Pending commits are correctly pending; no push state was inferred from reflog history.

## Residual Risk

- B1 statistical nonregression remains unestablished by the fixed series; documentation does not remove this uncertainty.
- Affected profiler compute annotations remain unsuitable for utilization-based decisions until corrected.
- Round one's unchanged stage limitations remain in force. This diff-only follow-up is not a fresh whole-stage numerical or source review.
