# Stage Review

Verdict: clean-pass

Fresh independent follow-up, round 2 of 3, Stage 9 vLLM integration for `Qwen/Qwen3.6-35B-A3B`. This is a diff-only review against round 1 and the owner's explicit 2026-11-17 decision. Paths below are relative to `models/autoports/qwen_qwen3_6_35b_a3b` unless absolute. The verdict accepts the reviewed stage under that amended contract; it does not convert the constrained-thinking comparison into a passing quality result.

## Required Work

None. Both previous required findings are closed for Stage 9. No new correctness or missing-evidence blocker was found in the changed implementation and affected artifacts.

### Previous required findings

- **Round 1 P1 — closed by explicit, narrowly scoped owner acceptance, not by repair.** I read the original `/home/ttuser/qwen36-35b-a3b/multigoal_logs/09-09-vllm.resume-6.message.txt`, `doc/vllm_integration/owner_decisions_pending.md`, and the verbatim decision in `owner_decisions_20261117.md`. The instruction expressly accepts the recorded prompt, constraints, sampling recipe, 1024-token cap and seeds 0–9 for Stage 9 closure, with numerical localization deferred. `readiness_vllm/sampling_controls/card_unblinded_summary.json` still says `FAIL_COMPARISON`: HF 2/10 versus TT 6/10 secondary failures, including two TT loops, a 40-point difference against the frozen 20-point margin. The new limitations entry correctly retains `original_model: "better"`; it does not claim HF equivalence. Native reproduction and unsuccessful AutoFix remain attribution/investigation evidence, not the basis for acceptance. No broader quality waiver follows.
- **Round 1 P2 — closed by restoration and affected validation.** `tests/run_vllm_server.sh` no longer supplies a server-wide template override. The actual API-server arguments in `readiness_vllm/checkpoint_default/execution.json`, its server summary, and raw `server/server.log:14` confirm the measured server omitted it. In `default_behavior.json`, default, explicit-on and default-after-off requests have identical prompt tokens and identical 158-token greedy responses ending in `4`, with reasoning present. Explicit-off returns `4` in two tokens without reasoning. The check source asserts all four server tokenizations equal the pinned tokenizer's rendering. Fresh P3 has 15 passing checks and real completed responses, including thinking, tools, history, streaming, JSON and stopping. The shared default suite and supplemental explicit-mode sampling results are assessed below. The restoration meets the owner's requested behavior; the rejected direct-answer default is not retained.

## Other Concerns

Disposition of every round 1 concern:

- **Admission-dependent seeded routing — closed as a review follow-up; the technical limitation remains.** The README, work-log handoff and serving-contract seed row retain the observed host/device transition limitation and the old individually unattributed `TopP:` mismatch. No repair or universal route-invariance claim was added. The new explicit-mode feature tests do not waive this limitation or the separate quality result.
- **Stale capability descriptions — closed for the current advertised behavior.** `tt/serving_capabilities.py::serving_limits` now describes top-k 1–64, standalone versus serving buckets, and the mixed-route limitation. `doc/full_model/serving_contract.json` adds the current top-k behavior and a seed limitation pointer while retaining historical baseline evidence. The stage-9 adapter section remains the authority for serving behavior. Some inherited baseline prose still refers to the former interface and fixed bucket; that is a documentation qualification, not evidence of a changed execution path or a new review blocker.
- **SwiGLU hang attribution — closed as a documentation follow-up; causal uncertainty remains.** README and the new work-log handoff preserve the distinction between the repaired completion contract and unproven causality of the original hang. This restoration adds no native-kernel change or new causal claim.
- **Small performance sample sizes — closed as a reporting follow-up; statistical limitation remains.** README still states that the primary workload contains one request and the matched HTTP comparison two measured repetitions. It labels the percentiles as descriptive and limits the overhead conclusion to the measured B2 path. The retained measurements use completions and are unaffected by removal of a chat-template default.

The new 12-output default suite supports coherence of capped reasoning, not complete-answer accuracy. I read all 12 served outputs, all 12 fresh native controls, and the two six-output HF suites. All remain task-directed; the French in q05 is requested, and q03's `thinking thinking sequence` opening also occurs in both HF controls. All 12 served and all 12 HF outputs reach the 256-token cap before final answers. Six greedy served texts exactly equal fresh native text; sampled text differs without an observed sustained loop, gibberish, unintended language drift or contamination. These samples do not establish broad HF-equivalent accuracy.

The supplemental bad-word responses include poor content despite passing the feature assertion: literal thinking text and an unrelated invented instruction occur. I compared all five new response contents and finish reasons to `sampling_controls/http_bad_words_off_100.json`; all five match exactly. The existing same-recipe, ten-seed comparison in `off_unblinded_summary.json` reports HF 7/10 and TT 6/10 secondary failures within its frozen margin. This is controlled prior behavior, not a newly introduced restoration regression or an extension of the owner's thinking-on waiver. The separate model-card nonthinking comparison also remains poor on both sides. Passing the ban assertion is not an absolute quality pass.

## Hard-Check Gaps

Disposition of every round 1 gap:

- **Missing prefill serial reference — closed as already classified advisory.** The new `doc/vllm_integration/logs/stage_check_checkpoint_default.log` and `stage_check_checkpoint_default.json` preserve exit 1 and identify the same advisory. Context 262144, 86-row serving contract, degeneracy and eval-readiness checks pass. The installed `vllm-integration/references/prefill-features.md` permits serial prefill and requires a serial comparison when a batched path exists. No new batched implementation or speedup is claimed, so no redundant self-comparison is required.
- **Full sampling skips and coverage — closed as an evidence-scope clarification.** I parsed the retained full JUnit: 142 cases, 68 passed, 74 skipped, no failures/errors; 70 skips require speculation, one speculative artifacts, and three the chunked-prefill test argument. The earlier independently reviewed lifecycle/slot/growth evidence remains the chunked-prefill coverage. The historical full pass used thinking-off server defaults. The new JUnit has only three passing tests; it is supplemental explicit-request coverage, not a fresh unchanged-default full-suite pass.
- **Degeneracy checker scope — closed as an interpretation gap.** Its passing result covers the shared qualitative artifacts; it does not rescind the separately retained constrained-thinking failure. Current README and limitations preserve that distinction and the exact owner acceptance.
- **Unknown original P5 exit — closed as a provenance qualification.** The current README/inventory retain the recovered-result description and unknown original exit; no invented zero exit or fresh P5 execution is claimed. The unchanged nine-case result remains covered by round 1. No rerun is requested.

For the three new canonical assertions, I inspected the model-owned helper and the unchanged plugin tests. The helper adds only the existing request field `chat_template_kwargs.enable_thinking=false`, preserves other request fields, and refuses to overwrite an explicit template setting. I independently checked all 101 recorded requests/responses: five bad-word requests at the original cap/seeds, 32 mixed-capacity requests and 64 slot-reuse requests; all original response assertions pass. Structured cases retain caps 8/16/64, their grammars and the choice penalty; plain cases retain cap 16. This validates those request features in explicit nonthinking mode without changing the server default or shared plugin source.

P2 reuse is appropriately bounded. The work log explicitly acknowledges that three chat profiles render different prompt tokens after restoration. Sampling parameters, routing predicates, fixed decode shapes and production execution code are unchanged; fresh default rendering/state checks, P3 and the default shared suite cover the changed template behavior. Historical P2 metrics retain their original provenance. This review does not treat them as newly measured thinking-on performance, and does not request a blanket P2/P5/context/performance rerun.

## Anomaly Ledger

- Observed anomaly: TT constrained thinking has more failures than HF, including two loops.
  Evidence: `sampling_controls/card_unblinded_summary.json`, retained raw/blind/native evidence, owner decision source.
  Affected path: exact recorded constrained-thinking greeting corpus.
  Control or comparison: the prior independently reviewed ten-seed HF comparison and exact native TT attribution.
  Likely subsystem: inherited full-model numerics; exact cause remains unproven.
  Investigation performed: prior AutoFix and native/boundary controls; no new numerical experiment in this round.
  Resolution: controlled for Stage 9 disposition by explicit owner acceptance of this exact limitation; failed scores and unresolved technical defect retained.

- Observed anomaly: ordinary chat had an unapproved thinking-off default.
  Evidence: old launch records versus current launcher, actual process arguments and `checkpoint_default/default_behavior.json`.
  Affected path: omitted `enable_thinking`.
  Control or comparison: omitted/on/off/omitted-after-off rendering and exact greedy results; full P3.
  Likely subsystem: launcher policy.
  Investigation performed: source inspection, affected serving validation and raw response review.
  Resolution: fixed; normal checkpoint thinking is restored and explicit-off remains request-local.

- Observed anomaly: all restored-default shared outputs stop during reasoning at 256 tokens; q03 repeats a word in its opening.
  Evidence: new shared outputs, fresh full40 native controls, matching pinned HF controls.
  Affected path: bounded default-mode qualitative suite.
  Control or comparison: all prompt-token arrays agree; all HF outputs also have 256 tokens; q03 opening reproduced by HF; all six greedy served texts equal native.
  Likely subsystem: checkpoint behavior and token budget.
  Investigation performed: full text reading and direct artifact comparisons.
  Resolution: controlled for this budget; no complete-answer accuracy claim.

- Observed anomaly: passing bad-word feature tests contain poor answers and literal thinking text.
  Evidence: `checkpoint_default/explicit_off_requests.jsonl` and `sampling_controls/{http_bad_words_off_100,off_unblinded_summary}.json`.
  Affected path: explicit-off constrained greeting with the canonical test recipe.
  Control or comparison: all five new contents/finish reasons exactly reproduce the earlier controlled run; existing ten-seed HF/TT comparison is within its predeclared margin.
  Likely subsystem: constrained model behavior; these assertions check bans, not general answer quality.
  Investigation performed: raw response reading, exact comparison and inspection of test assertions/settings.
  Resolution: controlled prior limitation; not an absolute quality pass and not a new owner waiver.

## Scope Inspected

- Goal/skills: original Stage 9 goal, exact resume-6 owner instruction, round-2 brief and round-1 report; installed `stage-review`, `vllm-integration`, `qualitative-check`, model-bringup startup and relevant prefill/P2 references. No re-review of unchanged production subsystems.
- Changed code: `tests/run_vllm_server.sh`, `tests/check_vllm_checkpoint_default.py`, `tests/checkpoint_default_test_requests.py`, `doc/vllm_integration/run_checkpoint_default_validation.py`, descriptive serving-capability/contract updates. Read the collector's existing audit helpers and canonical plugin tests only to assess the new evidence.
- Artifacts: owner decision records; updated README/work log/manifest/inventory/limitations/checker; `readiness_vllm/checkpoint_default/` execution, actual server arguments/raw log, default responses, P3 raw responses, qualitative outputs, explicit request log and JUnit; fresh native controls and reused HF controls; retained full JUnit and directly relevant constrained controls. Verified promoted qualitative files equal the new files and preserved old copies equal the former canonical artifacts.
- Commands: read-only `cat`, `sed`, `tail`, `rg`, `git status/diff/rev-parse`, and short standard-library Python JSON/XML/text analyses. One exploratory script assumed the wrong HF prompt-id key; the corrected comparison verified all HF/server prompt-token arrays. No model imports, inference, server, hardware operation, reset or test suite was run. Only this report was written.
- Recomputed evidence: all four jobs exit 0; raw log intervals contain respectively 476/0, 664/14, 3060/0 and 2/137 device/host decode steps, totaling 4202/151. Raw shutdown events equal the execution record and show four captures/four releases. There are no raw error/traceback lines. Recorded cleanup has launcher exit 0, no remaining owned PIDs and empty fuser output with exit 1. Native controls use the exact selected precision dictionary and all 12 token arrays match the earlier native default controls. All 101 supplemental assertions independently pass.
- Repository state: live tt-metal `main` baseline `47c0bbd606a691e6111b46c5ee1eb9e01fd423d8` plus stage-owned work; plugin `5d1785b76d17a8989c9495f6939abea6cef62f09`, clean. Local tt-metal checkpoint follows this review as required; its absence before review is not a finding. The run plan authorizes local commits and forbids stage-agent pushes. Precommit log `logs/checkpoint_default_all_precommit.log` records passing applicable hooks.

## Residual Risk

- The owner's exact acceptance closes Stage 9's constrained-thinking blocker; it does not repair numerical quality, extend to other prompts/recipes, or waive later accuracy gates.
- Default-mode qualitative evidence is capped reasoning. Full canonical sampling retains historical thinking-off provenance; the three new tests prove explicitly configured request behavior.
- Admission-sensitive seeded routing, serial prefill, the recurrent-state prefix-cache exclusion, prior scoped resumed-prefill limitations, limited performance sample sizes and provisional original-hang attribution remain documented boundaries. This follow-up introduces no broader claims about them.
