# Stage Review

Verdict: pass-with-carried

Independent follow-up **round 3 of 3**, stage 5 `optimized-multichip-decoder`, Qwen/Qwen3.6-27B. No correctness, accuracy, wrong-number, or missing-required-gate defect was found in the changed claims. The prior documentation findings are closed. B1 statistical uncertainty, the profiler annotation correction, and untried optimization/import cleanup remain explicitly carried; this verdict does not turn them into passed tests or completed work.

The inspected live checkout was `main` at `8502f42bca20dcad1507fbc8cebf7032220b7d8e`. **D** means `models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/`.

## Required Work

- None within this diff-only review. The previous finding of no model correctness, accuracy, or missing required device evidence remains closed. No new device run, tuning, source change, or threshold change is required by this review.

## Other Concerns

- **Review/checkpoint sequence — review requirement closed by this report; local checkpoint and acceptance recording remain open procedural follow-up.** `D/final_manifest.json` accurately records two completed `more-work-needed` reviews and `local_commit.sha: PENDING` at review time. Both listed reports exist and have those verdicts. The round-two work-log entry accurately describes its procedural disposition. The owner should now record this third report and its accepted carry items separately from the original checker result, create the authorized local checkpoints, and log repo/branch/SHAs before stage completion. Pending commits are expected because the skill orders them after passing review. The run plan permits local commits only; no push is authorized.

- **P2: B1 prefill confidence — documentation closed; uncertainty open and accepted as nonblocking carry.** `D/work_log.md` keeps both layer kinds inconclusive under `Carried to next stage`; manifest carry metadata retains `quality_gate_waiver: false`. Independent arithmetic over the unchanged five raw pairs per kind reproduces control/final means of **2.395564131/2.194059826 ms** for GDN and **1.761019820/1.701234902 ms** for full attention. Median final-minus-control differences are **−0.216594121** and **−0.062272574 ms**; maximum differences are **+0.037668118** and **+0.057086038 ms**. Four of five pairs favor final in each kind. The saved bound coverage is **0.96875**. Zero-margin statistical nonregression remains unestablished. This is accepted performance uncertainty under the owner cap, not output wrongness or demonstrated population regression. Optional prespecified confirmation belongs to a later authorized stage.

- **P2: DRAM-sharded compute annotation — disclosure/consolidation closed; tool correction open and accepted as nonblocking carry.** The new first paragraph under `Carried to next stage` explicitly excludes affected `Cores`, `FLOPs [%]`, and compute-bound labels from utilization/rejection decisions and links `D/final29_profiler_limitations.md`. The example decode CSV still reports eight cores and **175.65269384774476%** FLOPs for `32 x 5120 x 7168`. Raw op attributes select two workers per DRAM bank; the native factory constructs its compute worker set from those assignments, while the installed reporter substitutes the architecture's fixed core count. This supports the documented sixteen-worker versus eight-worker discrepancy. Correct the reporting tool later and retain the raw output and measured latencies. All six final decode CSVs have empty Advice columns. The 30 final raw ops/report CSV/table files checked are byte-identical to their staged copies.

- **Untried optimization and lint inventory — disposition closed; experiments/cleanup remain accepted carry.** The work log retains the inventory under `Carried to next stage`, now with the explicit `Not tried (stop rule)` subheading and a statement that integrated gains are unmeasured. This correctly avoids converting component gains or untried alternatives into whole-layer wins or measured rejections. Final-fidelity geometry, norm capacity, recurrence combinations, long-prefill movement, and import cleanup remain future work. The selected minimal/DiT norm path remains off in the context contract. No experiment or source reformat is required under the owner's no-tuning instruction.

- **Stale preparation wording — closed.** `D/final29_interlayer_contract.md` now says final29 completed its own final-default performance, independent contract, context/owner/trace/stress/watcher, and provenance work. The existing final validation summary and manifest agree on 68 jobs and a common signature. All 68 referenced receipts exist, report `pass: true` and exit code zero, and record successful native-build verification before/after; all record tracking `1` with skip-program-cache `0`. The TP4 receipts include ten correctness, two contract, two stack, two resource, two context, eight stress, and twelve watcher jobs. Six final measurement artifacts pass and pin the current decoder SHA. The capacity report's 80 recorded input hashes match disk. Its updater log and the validated context-contract section preserve **262144** supported context, distinguish the final **32768** smoke from historical maximum-context evidence, and explicitly do not claim a new maximum-context run or full-model allocation proof. These artifacts support completed-status wording without relabelling final28 receipts.

## Hard-Check Gaps

- The original `D/final_gate.json` remains **`pass: false`**, SHA256 `0ddbb0607936b7033cb1e7e8c8b7d57fe5348d2c8b9dbc41a60e8f8503105ea4`, matching its staged bytes. It records the B1 confidence failure and the review/commit status at the time of execution. This review accepts the documented nonblocking confidence uncertainty; it does not claim that checker passed. Preserve that artifact and record final review acceptance/checkpoint metadata separately.
- No new hard-check gap was introduced by the inspected changes. This follow-up does not repeat round one's whole-stage numerical/source audit or independently reconfirm every unchanged implementation claim.

## Anomaly Ledger

- Observed anomaly: improved B1 means coexist with positive one-sided confidence bounds.
  Evidence: twenty `D/final29_pair_l{0,3}_p{0..4}_{control,final}.json` files, `D/final29_b1_statistics.json`, and original final gate.
  Affected path: B1 prefill timing inference for both layer kinds.
  Control or comparison: all five fixed control/final pairs for each kind.
  Likely subsystem: timing variability/statistical inference, not demonstrated output corruption.
  Investigation performed: independently recomputed means, paired medians, maxima, and favorable-pair counts; inspected the carried status and retained false checker result.
  Resolution: **controlled as explicit nonblocking uncertainty**; confirmation remains carried, confidence test remains unpassed.

- Observed anomaly: reported compute utilization exceeds 100% on a two-reader DRAM-sharded matmul.
  Evidence: `D/perf/final29_profile_l0_b1_s512/{ops.csv,decode_perf_report.csv}`, native worker construction, and installed reporting-tool override.
  Affected path: profiler core/FLOP/compute-bound annotations.
  Control or comparison: selected reader assignments versus fixed eight-core report model.
  Likely subsystem: reporting-tool worker accounting.
  Investigation performed: checked the row, raw attributes, source explanation, carry paragraph, and preservation of final raw reports.
  Resolution: **controlled by explicit reporting limitation**; correction remains carried and measured latency remains available.

- No new anomaly was found in the changed documentation or metadata. Earlier unchanged anomaly dispositions remain those of the previous reviews.

## Scope Inspected

- Goal/skill paths: original `/home/ttuser/qwen36-27b/multigoal_logs/05-05-optimized-multichip-decoder.prompt.txt`; installed 0.1.24 `stage-review/SKILL.md`, `model-bringup/SKILL.md` startup, relevant `optimize/SKILL.md` stop rule and `tt-device-usage/SKILL.md` stage budget; original and round-two/round-three briefs; round-two report. Enabled plugin inventory and packaged environment resolution were checked.
- Changed files: `D/work_log.md`, `D/final29_interlayer_contract.md`, and `D/final_manifest.json`, including the additional wording-only `Not tried (stop rule)` change communicated during review. The live index was being populated by the owner; staging was not treated as new numerical evidence.
- Supporting artifacts: original final gate; B1 pair/statistics files; validation summary/manifest and their 68 receipts; six final measurement JSONs; capacity report/update log and its input hashes; adjacent context contract; profiler limitation note; final profile raw ops/CSV/tables and decode Advice fields. These reads were bounded to the prior findings and changed completion claims.
- Code paths: decoder SHA256 only, still `2d4ba9dd20c86f6ae9d710c5fe79c259bdd66fe6007b6af4be5437ca01554c0b`; bounded native DRAM-sharded worker construction and installed `tt_perf_report/perf_report.py` core-count override. No implementation was edited.
- Commands run: read-only `cat`, `sed`, `rg`, `git status/diff/show/rev-parse`, plugin inventory/environment inspection, and standard-library JSON/CSV/hash/statistics scripts. Only this requested report was written. No hardware, server, test, benchmark, reset, or other agent was started.
- Push state: local-only checkpoints follow this review under the supplied run plan; no push state was inferred from reflog history.

## Residual Risk

- B1 statistical nonregression is still unestablished; affected profiler compute annotations remain unsuitable for utilization decisions until corrected.
- Full-model allocation and a new final-source maximum-context execution are not established by this decoder-stage evidence and are not claimed here.
- This is a fresh diff-only round-three review, not another full-stage review. The owner must still record acceptance and local checkpoint SHAs before declaring the stage complete.
