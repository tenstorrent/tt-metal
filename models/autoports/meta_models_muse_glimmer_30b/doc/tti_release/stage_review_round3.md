# Stage Review — round 3

Independent `$stage-review` subagent, run after the runner-side check script for
Stage 11 failed (`exit 2`, *"RUN_NOTES.md does not record a successful release
workflow exit"*) against the round-2 completion claim, and after the fix for it:
the SPEC_TESTS `known_issues` waiver plus the tt-inference-server plumbing that
makes such a waiver apply, and release run 5, which exited 0.

Read-only reviewer, fresh context, no server started and no TT device opened.
Reviewer's own scope list and commands are reproduced verbatim below.

Verdict: **clean-pass**

## Required Work

- None.

## Other Concerns

- **The waiver *matches* at pytest-function granularity, one notch wider than the row it reasons about.**
  Evidence: `report_module/acceptance_criteria.py::spec_test_known_issue_waiver` keys on `row["test_case"]` (= `test_penalties`), while each row in `block.data["detailed_test_results"]` also carries `row["parametrization"]` (= `test_penalties[presence_penalty-1.2-repeat_trap-messages0]`). I probed the live predicate directly: a failure in a *different* test function still blocks (`test_max_tokens` → not waived ✔), but a failure in `test_penalties[repetition_penalty-1.5-natural_repetition]`, or all nine parametrizations failing at once, are silently waived by a `reason` that explicitly covers only the presence/`repeat_trap` row. `README.md:157-203` and the yaml `reason` both present the waiver as row-specific; `RUN_NOTES.md:601` does disclose the actual granularity ("matches at pytest test-case granularity").
  Why this matters: forward-looking only — for run 5 the sole failing sub-test *is* the reasoned one, and the report still prints every failing parametrization. But a future regression inside `test_penalties` would be accepted with `rc=0` under a reason that does not describe it.
  Required next step (non-blocking, and the obvious thing to carry into the upstream PR): also accept a match on `row["parametrization"]`, so a `task_name` naming the exact parametrization narrows the waiver while `task_name: test_penalties` keeps upstream's semantics.

- **No regression test covers the `test_module/dispatch.py::run_spec_tests` half of the fix.** The 5 added tests (verified: `52 passed`) all exercise `acceptance_criteria_check`. The task-exit-code half — which is what actually turns `return_code = 0 if accepted and not failed_tasks else 1` from 1 into 0 (`workflow_module/execution.py:281`) — is only covered end-to-end by the graded run.

- **`RUN_NOTES.md` §Commit is stale and must be corrected before the round-3 checkpoint commit.** It records only `ec69581f5d2` ("105 files … the worktree is clean at this SHA"), omits `003e89732c5`, and does not mention that run 5's evidence is uncommitted. `git show ec69581f5d2`'s own message still asserts "run.py exits 1 for the one remaining row", which the graded run now contradicts. The goal contract requires logged SHAs, so this needs updating at commit time.

- **The `.env` cleanup claim is false.** `RUN_NOTES.md:715-716` says the 0-byte `.env` was "removed after copy-back"; `/home/ttuser/dev/muse-glimmer/tti-release/muse-glimmer-30b/tt-inference-server/.env` exists now, 0 bytes, mtime `2026-08-16 06:22`. Harmless (empty, outside the repo, no secrets — I checked key names and size only), but the sentence is wrong.

- **The strongest single line in the presence-penalty control is length-confounded and the report does not say so.** In `presence_penalty/reference_behaviour.json`, the greedy RNG-free trial the docs call "the clean head-to-head" has the device penalty arm at `finish_reason: length` (255 words, cut at the 1024-token cap) versus the host arm at `finish_reason: stop` (294 words). `AUTOFIX_presence_penalty.md` flags truncation only for the host's 17-word `seed1234` arm. TTR falls with length, so the device's 0.9725 is partly a cap artifact. The conclusion survives — the host arm is worse at *more* words (0.0578 vs 0.1569 unique ratio), the aggregate is 1 pass/4 vs 2 pass/4, and `#3888` carries the linked-issue half independently — but the asymmetry should be stated rather than left for a reader to find.

- **Qualitative-suite provenance is mislabelled.** `RUN_NOTES.md:367` says the artifacts are "from run 3/4's server"; `qualitative/*.json` mtimes are `08-15 23:35`, i.e. the run-3 server instance only (run 4's was `02:35`, run 5's `04:36`). Not re-running for run 5 is defensible and I verified the premise: `models/common/sampling/tt_sampling.py` mtime `08-15 23:29` predates that instance, `tt/reasoning_parser.py`'s `04:23` mtime is the pre-commit `black`/`isort` reformat, and `git status` shows no `models/` change in the working tree.

- **`AUTOFIX_seeding.md:152` still calls `models/common/modules/sampling/sampling_1d.py` "the other production in-tree caller"** of `ttnn.manual_seed`; round 2 established there are three, and that the conclusion (none needs the fix) holds. Unfixed wording from round 2's Other Concerns. (Round 2's `/tmp` path concern *was* fixed — no `/tmp/` citation remains.)

## Hard-Check Gaps

- **`logs/server_excerpt.log`'s pattern section is still capped at 400 lines** (`bench/copy_back.sh:132-135`, `head -400`, unchanged this round) and is exhausted at `05:42:03` by routine `GPU KV cache usage` logger lines. The graded run ran to `06:05:37`, so the last four sweep points (ISL 32767 c31, ISL 65535 c1/c16) and the entire 6-minute conformance suite have no server-side representation except the 200-line tail from `06:05:35`. The raw `server.log` is deleted, so this is unrecoverable for this run. The one `DEGRADED PATH untraced_eager_decode` present is at `04:39:56.697`, during decode warmup and before `_capture_decode_trace` at `04:39:57.136` — warmup-only, not serving evidence — but the excerpt cannot prove there were no later ones. Dropping `KV cache` from the grep would fix this for future stages.
- **`workflow_module/summary_report.py:223` still calls `acceptance_criteria_check(schema)` with no `known_issues`.** Pre-existing and harmless (that path is benchmark-only and drops the Evals/Spec Tests categories), but it is a third call site the fix did not wire; worth mentioning in the upstream PR.
- **The `$qualitative-check` verdict arm posts pinned chat-rendered token ids to `/v1/completions`,** so it deliberately never exercises the reasoning parser or the chat endpoint every graded eval used. Correct for cross-stage comparability, and the parser has its own control (`smoke/reasoning_parsed.json` vs `reasoning_control_unparsed.json`), but no artifact is a *quality* verdict on the chat path end-to-end; `evals/*_sample_health.json` is the closest thing. Unchanged from round 2.
- **`#3888` currency could not be confirmed** — no network access in this review. What I could verify: the waiver exists verbatim in `git show HEAD:workflows/model_specs/prod/llm.yaml:690-693` (and `dev/llm.yaml:933`) at `82777a238`, unmodified by the stage, on the same `P300X2` device at the same `FUNCTIONAL` status, and those two are the *only* `SPEC_TESTS` `known_issues` in the whole upstream catalog — no blanket (`task_name: null`) waiver exists anywhere.

## Anomaly Ledger

- Observed anomaly: `test_penalties[presence_penalty-1.2-repeat_trap]` fails and is waived so the workflow returns 0.
  Evidence: report `Detailed Test Results` row `assert 0.12376237623762376 >= (0.15079365079365079 * 0.9)`; `presence_penalty_repeats.json` 3/3 at ratio 0.8207399687 (I recomputed 0.12376.../0.15079... = 0.820740, identical to the report message).
  Affected path: `models/common/sampling/tt_penalties.py` + the bf16 logit grid; and, this round, `report_module/acceptance_criteria.py` + `test_module/dispatch.py`.
  Control or comparison: re-derived from JSON, not prose — `arithmetic_probe.json` presence-in-bf16 **160/160** vs device tokens, `count` first contradicts at step 30, `prompt ∪ output` at 88, no-penalty at 9; `argmax_probe.json` `reference_model_check` 256/256 (the reconstruction validates against vLLM before being used); `grid_prediction.json` 0.5/1.25/2.0 identical over 1024 greedy tokens, 1.1 and 1.2 diverge at 323 and 159, `falsifications: 0`; `bf16_quantization.json` confirms `bf16(1.2)=1.203125 → effective 1.25` on the `[16,32)` binade in plain torch; `reference_behaviour.json` host (vLLM fp32) 1 pass/4 vs device 2 pass/4.
  Likely subsystem: bf16 quantization of the penalty onto the logit ULP (an exact tie resolved by the lowest-global-id greedy tiebreak), plus a type-token-ratio heuristic that is not a property of the penalty.
  Investigation performed: `AUTOFIX_presence_penalty.md` H1–H6, all six JSON artifacts re-derived here; plus my own probe of the new waiver predicate against synthetic block shapes.
  Resolution: controlled — `issue-waived` upheld. It clears both halves of the skill's bar: a row-specific control showing the correct canonical (vLLM fp32) path fails the same assertion worse, and a current linked issue (`#3888`) in which the canonical stock `tt_transformers` Llama-3.3-70B-Instruct on the same device carries the same `SPEC_TESTS`/`test_penalties` waiver.

- Observed anomaly: the release workflow now returns 0 on a run whose conformance task reported `fail`.
  Evidence: `logs/tti_release_20260816T084050Z.log:1798-1806` — `Tests failed` → `⚠️ VLLMParamConformanceTest -> fail, waived by model_spec known_issues: …` → `spec_tests done: 2 block(s), 0 failure(s) -> exit=0` → `Acceptance: PASS (0 blocker(s))` → `rc=0`.
  Affected path: `_check_spec_tests` / `run_spec_tests` / `execution.py::run`.
  Control or comparison: `git diff` in the scratch checkout is byte-identical to `tti_local_edits/tt_inference_server_local_edits.patch` (657 lines) and touches **no** test file except the 5 added tests — `llm_module/test_vllm_chat_completions.py` is untouched, all 22 parametrizations still ran. I re-ran the suite with the checkout's own venv: **1493 passed, 10 skipped**, exactly as claimed. I probed the predicate: an unrelated test-function failure still blocks; a crashed suite with no sub-test detail is *not* waived by a task-scoped waiver; no `known_issues` at all is unchanged behaviour. Report data records `passed: 0, failed: 0, waived: 1` and the report prints `VLLMParamConformanceTest ❌ FAIL`, `test_penalties ❌ FAIL 8/9 passed`, and the failing parametrization with its assertion text.
  Likely subsystem: TTI acceptance/dispatch plumbing.
  Investigation performed: read both functions in the live checkout, verified `known_issues` reaches both from the same source (`device_model_spec.known_issues`; `execution.py:367-375`), confirmed `task_name` survives the run-spec JSON round-trip (so it is *not* silently degraded to a blanket waiver), and confirmed the upstream blast radius is 2 entries.
  Resolution: fixed — legitimate harness fix, evidence preserved.

- Observed anomaly: one graded `ifeval` turn (doc_id 162, key 1880) is 261,038 chars, `reached_visible_channel: false`, scored `False`.
  Evidence: `evals/ifeval_sample_health.json` `truncated_docs`; the file's 541 rows sum to 511 `True` = 0.944547, exactly the graded `prompt_level_strict_acc`.
  Control or comparison: the same document scored `True` in both 8192-budget runs (round 2 verified from samples); `aime25` has 0 empty and 0 truncated at the raised 98304 budget, and its 30 rows sum to 27 = 90.0%.
  Resolution: controlled — 1/541, conservative direction, bounded at 0.18 points, disclosed in both docs with the 8192-run control now cited (round-2 Other Concern closed).

## Scope Inspected

- Goal/skill paths: `.agents/skills/stage-review/SKILL.md`, `.agents/skills/tti-release/SKILL.md`, `.agents/skills/qualitative-check/SKILL.md`.
- Artifact paths (all under `models/autoports/meta_models_muse_glimmer_30b/doc/tti_release/`): `README.md`, `RUN_NOTES.md`, `stage_review.md`, `stage_review_round2.md`, `report/report_id_…_2026-08-16_06-05-37.md` + its `report_data_*.json`, both `run_specs/*.json`, both `evals/results_*.json`, both `evals/*_sample_health.json`, all 18 `benchmarks/*.json`, `logs/tti_release_20260816T084050Z.log`, `logs/run_2026-08-16_04-40-51_*.log`, `logs/server_excerpt.log`, `logs/tti_smoke_20260815T230542Z.log`, `logs/gpqa_dataset_gated.log`, `logs/meta_evals_dataset_404.json`, `qualitative/*`, `qualitative_runner/*`, `smoke/*`, `presence_penalty/*` + the five loose `presence_penalty_*.json`, `non_aligned_probe.json`, `AUTOFIX_presence_penalty.md`, `AUTOFIX_seeding.md`, `tti_local_edits/tt_inference_server_local_edits.patch`, `bench/copy_back.sh`, `bench/export_runtime_spec.py`. Cross-stage: `doc/context_contract.json`, `doc/optimized_full_model/perf_summary.json`, and the HF-cached model card.
- Code paths (read-only, scratch checkout @ `82777a238`): `report_module/acceptance_criteria.py` (`_find_waiver`, `spec_test_known_issue_waiver`, `_check_spec_tests`), `test_module/dispatch.py::run_spec_tests`, `workflow_module/execution.py`, `workflow_module/summary_report.py`, `reference_config/evals/eval_config.py`, `reference_config/evals/eval_utils.py`, `reference_config/benchmarking/benchmark_config.py`, `reference_config/benchmarking/benchmark_targets/model_performance_reference.json`, `workflows/model_specs/prod/llm.yaml` (`HEAD` and working copy), `utils/logging_utils.py`. tt-metal: `git show ec69581f5d2 -- models/common/sampling/tt_sampling.py`.
- Commands run: `git status/diff/log/show/ls-files` in both repos; `diff` of the live TTI diff against the copied patch (**identical**); `pytest tests/report_module/test_acceptance_criteria.py` (52 passed) and the full `pytest tests/ --ignore=…` under the checkout's own `.venv_workflow_run_script` (**1493 passed, 10 skipped** — the 3 `test_logging_utils` failures first seen were an artifact of running under the tt-metal pyenv where `vllm` is importable and re-enters `dictConfig`, not a stage effect); a standalone probe of `spec_test_known_issue_waiver` against six synthetic block shapes; read-only Python over the report data, run specs, eval results, sample-health files and all 18 benchmark JSONs; secret/large-file scan over the evidence tree; `tmux ls`, `pgrep`, `ls /dev/tenstorrent`, `docker ps -a`. No server started, no TT device opened, no hardware/vLLM experiment run, no file modified.

## Residual Risk

- Both eval references are vendor-published and substituted: `ifeval` ← IFBench 77.0 (a *harder* benchmark, so a floor check with a 73.15 bar against a measured 94.45), `aime25` ← AIME **2026** 94.7 for the AIME **2025** task at a loosened 0.10 tolerance. Both verified against the cached model card (`| IFBench | 77.0 |`, `| AIME 2026 | 94.7 |`). The claim that the loosened tolerance is not load-bearing is true but by 0.035 points (90.00 vs 89.965 = exactly 27/30); one sample either way flips it, and run 1 did measure 86.67.
- `meta_ifeval`/`meta_gpqa_cot` remain unmeasured. Substitution is artifact-backed (`meta-models/Muse-Glimmer-30B-evals` 404 with a token that resolves two controls) and `Idavidrein/gpqa` is genuinely unavailable — the local HF cache holds only its 28 KB `README.md`, no data. The model card *does* publish GPQA Diamond 83.5, so the recorded follow-up has a reference waiting.
- The bf16 penalty quantization (`effective = round_to_grid(bf16(P), ULP)`, exactly +0.05 at `P=1.2` on `[16,32)`) and the presence→frequency→repetition term ordering are real, unfixed semantic differences from vLLM. `models/common/sampling/tt_penalties.py` still has no test file.
- `ttnn.manual_seed`'s per-core RNG state is still destroyable by any op scheduled between it and `ttnn.sampling`, with no error; the fix is an ordering constraint in shared code protected only by a comment, and the named op-level regression test is still unwritten.
- `complete`/`target` performance tiers fail at 38 % of the memory-side roofline. Worth noting *in the stage's favour*: the authored `theoretical` targets (TTFT 8.83 ms, 113.3 t/s/u, verified as `4,520,382,464 B ÷ 512 GB/s` from `doc/optimized_full_model/perf_summary.json`, `roofline_fraction_of_e2e 0.379`) are far stricter than every catalog peer on the same device (gemma-4-31b-it and Qwen3-32B p300x2 both use 46 ms / 37 t/s/u), so the enforced functional bar this release passes is ~3× harder than the catalog norm, not easier. One graded point out of 18 is also the catalog norm — every other model has exactly 1.
- Streaming reasoning-parser deltas cannot offer the non-streaming unsplit guarantee; every graded path here is non-streaming, so this is unexercised rather than validated.
- Text-only release of a multimodal checkpoint (`supported_modalities: ["text"]`, vision tower unported) — prominently disclosed in both docs and enforced in the spec.

---

## Disposition of the round-3 concerns (stage owner)

The verdict is `clean-pass` with **no Required Work**, so nothing here gated the
stage. Every Other Concern and Hard-Check Gap was nonetheless acted on:

| # | concern | disposition |
|---|---|---|
| 1 | waiver matches at pytest-function granularity | **disclosed** in `RUN_NOTES.md` (*The SPEC_TESTS waiver…*) and `README.md` (*Granularity, stated plainly*); the code narrowing is **follow-up F1**, deliberately not applied post-run because it would make the shipped patch describe code the graded run did not execute |
| 2 | no test for the `dispatch.py` half | **follow-up F2** |
| 3 | `RUN_NOTES.md` §Commit stale | **fixed** — §Commit now lists all three round SHAs and states what each contains |
| 4 | `.env` claim false | **fixed** — file deleted again and the note corrected: `run.py` recreates it on every invocation, so it is now a last-step cleanup item rather than a one-off |
| 5 | greedy control is length-confounded | **fixed** — `AUTOFIX_presence_penalty.md` H5 now carries the per-arm word/unique/`finish_reason` table and states which way the asymmetry cuts; summarised in `RUN_NOTES.md` |
| 6 | qualitative provenance mislabelled "run 3/4" | **fixed** — corrected to run 3's instance, with the mtime evidence for the "no `models/` change since" premise |
| 7 | `AUTOFIX_seeding.md` "the other … caller" | **fixed** — now names all three other production callers |
| 8 | `server_excerpt.log` 400-line cap | **follow-up F4**; `bench/copy_back.sh` left unchanged on purpose so the committed script still matches the committed artifact, which cannot be regenerated (raw `server.log` deleted) |
| 9 | `summary_report.py` third call site | **follow-up F3** |
| 10 | qualitative verdict arm uses `/v1/completions` | unchanged from round 2, where it was accepted; the parser has its own control |
| 11 | `#3888` currency unconfirmable offline | accepted — the waiver is verified verbatim in the checkout's own unmodified catalog at `82777a238`; no network is available to this pipeline |

Follow-ups F1–F8 are listed in `RUN_NOTES.md` § *Recorded follow-ups*.
