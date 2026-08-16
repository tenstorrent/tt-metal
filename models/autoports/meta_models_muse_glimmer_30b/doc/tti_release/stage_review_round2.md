# Stage Review — round 2

Stage 11 — TTI release (tt-inference-server model release workflow)
Target: `meta-models/Muse-Glimmer-30B`, autoport `models/autoports/meta_models_muse_glimmer_30b`
Branch `agentic-research/hous/muse-glimmer-30b`, live worktree, uncommitted.
Reviewer mode: independent subagent, read-only (no server, no device, no hardware run).
Round-1 report under review for closure: `doc/tti_release/stage_review.md`.

Verdict: clean-pass

All six round-1 P2 findings are closed on evidence, not on assertion. The
release was genuinely rerun: the graded report is
`report_..._2026-08-16_04-04-24.md`, the run spec is
`runtime_model_spec_2026-08-16_02-39-22_...eqbjPKm1.json`, and every headline
number in `README.md` and `RUN_NOTES.md` re-derives correctly from the new
artifacts. No stale reference to the superseded run survives in any document —
the only occurrences of `01-10-53` / `23-50-19` in the tree are inside the
retained older run log, where they belong, and the single occurrence of `94.64`
is a correctly-labelled historical entry in the four-run reproducibility list.
The remaining items below are disclosure-level and do not change any technical
conclusion.

## Round-1 closure ledger

- **P2-1 `ifeval` truncation — CLOSED, verified.** `max_gen_toks` is 32768 in
  the graded run (`evals/results_2026-08-16T02-57-28.201205.json`,
  `generation_kwargs.max_gen_toks`) and in the live TTI eval config. I
  recomputed truncation from the uncopied samples for all four ifeval runs
  myself: `19-47-39` 0 truncated but **6 empty** (pre-fix parser, 94.09),
  `21-59-11` 3 truncated (95.38), `00-04-08` 3 truncated (94.64),
  **`02-57-28` 1 truncated (94.4547 → 94.45)**. So 3/541 → 1/541 is real. The
  residual is doc_id 162 / key 1880, 261,038 chars, `prompt_level_strict_acc`
  False — recorded with exactly those identifiers in
  `evals/ifeval_sample_health.json` and in both docs. I recomputed both health
  files from the samples: they reproduce the graded scores exactly
  (ifeval 511/541 = 0.944547, aime25 27/30 = 0.9), the source file names match
  the graded `results_*.json`, and `aime25` shows 0 empty / 0 truncated.
  The "scores False, which is what a run of `c`s would have scored as a reply
  too" claim is verifiable and true: the doc's `instruction_id_list` is
  `[keywords:letter_frequency, length_constraints:number_words]` and the strict
  per-instruction result is `[True, False]` — it failed on word count, which a
  261 kB reply would also fail.
- **P2-2 `ifeval` scoring prose — CLOSED.** `RUN_NOTES.md:169-174` now says
  `prompt_level_strict_acc` alone via `score_task_single_key`, and the eval
  config lists one `result_keys` entry with a comment explaining why. **The gate
  was not weakened:** `score_task_single_key` always graded `result_keys[0]`,
  which was already `prompt_level_strict_acc`, so the graded metric is
  unchanged; and of IFEval's four metrics in the results JSON
  (`prompt_level_strict 94.45`, `inst_level_strict 96.28`,
  `prompt_level_loose 96.12`, `inst_level_loose 97.48`) the retained one is the
  **lowest and strictest**. Dropping the second key made the config honest and
  strictly could not have loosened it.
- **P2-3 missing seeding log — CLOSED.** `logs/tti_release_20260816T014529Z.log`
  exists, its first timestamped line is `2026-08-15 21:45:29` (= the cited UTC
  stamp), and it is the only log in the tree containing `Determinism Failed`.
  The misleading `prefix_run2` name is gone.
- **P2-4 unlabelled pre-fix probe — CLOSED.** `smoke/conformance_probe.json` now
  carries `_what` ("PRE-FIX arm … Do not read this file as the shipped
  behaviour"), `_arm` and `_superseded_by`.
- **P2-5 "no packaged implementation in the run spec" — CLOSED.** Both
  `README.md:35` and `RUN_NOTES.md:49` now name the `docker_image` field
  explicitly. I re-ran the string search over both run-spec JSONs:
  `models/tt_transformers` 0, `models/demos` 0, `tt_vllm_plugin` 0, and the only
  non-autoport string is
  `"docker_image": "ghcr.io/tenstorrent/tt-media-inference-server:0.20.0-7db0eca"`.
- **P2-6 text-only release of a multimodal checkpoint — CLOSED.**
  `README.md` *Limitations* item 1 and a dedicated `RUN_NOTES.md` section, both
  naming `MuseGlimmerForConditionalGeneration`, the vision tower, and
  `supported_modalities: ["text"]` (which I confirmed is set in both run specs).

Round-1 *Other Concerns*: presence-penalty residuals (a) and (b) are now stated
beside the claim (`README.md:168-184`); all nine seeding probe scripts are in
`seeding/` and are **byte-identical to the `/tmp` originals** (I diffed all
nine); `logs/meta_evals_dataset_404.json` exists with a valid-token control on
`meta-llama/Llama-3.1-8B-Instruct-evals` (200) and on the model itself (200);
the ifeval row is labelled a floor check in the README status table; the
`aime25` tolerance is stated as not load-bearing (bar 89.965 at the default
0.05, reported 90.00 — re-derived from
`(score/reference) >= 1 - tolerance` at `eval_config.py:129`); `copy_back.sh`
now selects benchmark JSON by mtime between the run's own spec and its report
and asserts `n == 18`; the spec/server drift concern is refuted by the launch
log, which I read directly — `server_excerpt.log:15` shows
`additional_config: {'tt': {'sample_on_device_mode': 'all', …}}` and line 76
`TTModelRunner: … sample_on_device_mode=all`.

## What I re-derived independently this round

- **Graded numbers.** `ifeval prompt_level_strict_acc = 0.944547134935305` →
  94.45 (README, RUN_NOTES, report all agree); `aime25 exact_match = 0.9` →
  90.00 = 27/30; report row TTFT **72.1 ms** / TPOT 23.0 / tput_decode 42.7,
  functional tier PASS at TTFT ≤ 88.3 and tput_user ≥ 11.33 (ratio 3.833 ×
  11.33 = 43.4, which is the README's "43.4 t/s/u"). Conformance 21/22 — I
  counted the 22 parametrization rows in the report. Roofline 4,520,382,464 B ÷
  512 GB/s = 8.829 ms matches `doc/optimized_full_model/perf_summary.json`
  `roofline_inputs.per_device_bytes_per_token`, and the tier arithmetic
  (88.3 / 17.66 / 8.83, 11.33 / 56.65 / 113.3) is exact.
- **18/18 sweep points, 0 failed requests.** Every copied benchmark JSON has
  `"failed": 0` and `completed == num_prompts`, and all 18 launch stamps
  (03:35:08 – 03:54:32) fall inside the graded run window (02:39:22 launch →
  04:04:24 report).
- **Implementation evaluated is the autoport, in the NEW artifacts.**
  `runtime_model_spec…eqbjPKm1.json` → `impl.code_path =
  models/autoports/meta_models_muse_glimmer_30b`, `code_link` → the autoport at
  `7db0eca`, report metadata `model_impl = muse-glimmer-30b-autoport`, and the
  server log shows
  `models.autoports.meta_models_muse_glimmer_30b.tt.generator_vllm:initialize_vllm_model`
  plus 52/52 layers from the autoport's `tt/model.py`, at `max_model_len=131072`.
  The server instance that produced the graded run started 02:35:29, before the
  02:39:22 launch.
- **Non-zero exit honestly reported.** The wrapper log ends
  `Workflow release had 1 failed task(s)` / `rc=1`, and
  `report_data_*.json` lists exactly two blockers, both the same spec-test row.
  Benchmarks PASS (1/18, 17 NA), Evals PASS (2/2), Spec Tests FAIL (0/1).
  `limit_samples_mode: None` in the run log confirms the unrestricted set.
  Wall clock 02:39:22 → 04:04:24 = 85.0 min, matching the RUN_NOTES breakdown
  (ifeval 1083.96 s, aime25 2258.02 s, spec tests 368.19 s from their own JSON).
- **The waived row.** The report's failure message
  `0.12376237623762376 >= (0.15079365079365079 * 0.9)` → ratio 0.8207399687,
  byte-identical to all three trials in `presence_penalty_repeats.json`, which
  were taken on the *previous* server instance — so the row reproduced exactly
  across instances here, which strengthens rather than weakens the waiver.
- **Shared-code diff is still only the seeding fix.** `git diff` in tt-metal
  touches exactly `models/common/sampling/tt_sampling.py` (the moved
  `ttnn.manual_seed` call + comment) and the autoport `.gitignore` (one
  `server*.log` rule). I also checked every production caller of
  `ttnn.manual_seed` in the tree — `models/common/modules/sampling/sampling_1d.py:406`
  and both `llama32_1b_quasar` copies — and all three already seed with no
  intervening op, so no other call site needs the same fix.
- **TTI local edits.** `git diff` in the scratch checkout is **byte-identical**
  to `tti_local_edits/tt_inference_server_local_edits.patch` (regenerated after
  the rerun), the checkout has **no untracked files**, HEAD is `82777a238`,
  `VERSION` 0.20.0. No TTI test file is touched.
- **Qualitative.** Re-verified from raw text, not from the summary: the release
  chat completions are **character-identical** to the datatype-sweep stage's
  standalone-model completions on all six prompts once the single
  `<|message|>` token is removed (381/559/628/600/561/703 chars, no differing
  character on any prompt). Degenerate gate `exit_code 0`, "No degenerate output
  detected". `non_aligned_probe.json` 9/9.
- **Unit tests.** `pytest models/autoports/.../tests/test_reasoning_parser.py -q`
  → **13 passed** (host-only, no device opened).
- **Cleanup.** No tmux server, no `vllm`/`EngineCore`/`run.py` process, no
  tt-inference-server container (only unrelated 8-day-old exited sandbox
  containers), `/dev/tenstorrent/{0,1,2,3}` unused, no `.env` in the TTI
  checkout, no file over 2 MB anywhere in `doc/tti_release` (2.8 MB total), no
  token/secret pattern in any tracked file, and all 100 evidence files are
  git-addable (none silently ignored).

## Required Work

- None.

## Other Concerns

- **`RUN_NOTES.md:509` says the TTI patch is "351 lines"; it is 366.** The
  patch was regenerated at 04:07 after the eval-config edit and grew. Stale by
  one number, introduced by the rerun. One-word fix on the next touch.
- **The four-run `ifeval` reproducibility series mixes three harness
  configurations.** `RUN_NOTES.md:176-178` presents 94.09 / 95.38 / 94.64 /
  94.45 as "reproducibility across the four release runs … on the graded
  metric". The numbers are each correct — I recomputed all four — but run 1
  (94.09) ran under the *pre-fix* reasoning parser, which returned
  `content=None` on a truncated turn, and it has **6 empty responses** scored
  False; runs 2–3 ran at `max_gen_toks=8192` with 3 truncations each; run 4 ran
  at 32768. Read as a single noise band it slightly overstates comparability.
  One clause naming the three configurations would fix it.
- **The residual ifeval truncation has an in-tree control the docs do not
  cite.** The same document (doc_id 162 / key 1880) scored **True** in both
  8192-budget runs, with 527- and 721-character replies. That is the evidence
  that the 259,858-character run of `c` in the graded run is a
  temperature-1.0 sampling excursion on an adversarial letter-frequency
  constraint rather than a deterministic model defect — a stronger statement
  than the "not a budget problem" the docs make, and it is free.
- **`AUTOFIX_seeding.md` still cites the probe scripts by their `/tmp` paths**
  (`/tmp/rand_probe3.py`, `/tmp/seed_repro2.py`, …) even though all nine are now
  copied into `seeding/`. The copies are byte-identical, so the evidence is
  reproducible; only the citations point at a directory a reader will not have.
- **`AUTOFIX_seeding.md` calls `models/common/modules/sampling/sampling_1d.py`
  "the other in-tree caller" of `ttnn.manual_seed`.** There are three other
  production callers (that one plus two `llama32_1b_quasar` copies). I checked
  all three and the conclusion — none needs the fix — holds; the wording is what
  is imprecise.
- **`presence_penalty_repeats.json` is cited as "the shipped instance"** in
  `README.md:177-181` but was measured at 01:18, on the instance before the
  graded rerun. Harmless here because the graded report reproduces the same
  ratio to 16 digits, which is worth saying instead.

## Hard-Check Gaps

- **`logs/server_excerpt.log`'s pattern section is truncated at its 400-line
  cap, and the cap is consumed by routine `GPU KV cache usage` logger lines.**
  `copy_back.sh` greps for
  `reasoning|autoports|generator_vllm|KV cache|…|DEGRADED|…` and takes
  `head -400`; the section hits exactly 400 lines and its last entry is
  `03:41:19`, so any `DEGRADED`, host-fallback or device-health line between
  03:41 and the 04:04 tail is not represented. The raw `server.log` is deleted,
  so this is unrecoverable for this run. The one `DEGRADED PATH
  untraced_eager_decode` line that is present is at 02:39:03, during warmup and
  before trace capture at 02:39:03.451, so it is not serving evidence — but the
  excerpt cannot prove there were no later ones. Dropping `KV cache` from the
  grep, or grepping the anomaly patterns separately from the informational
  ones, would fix this for future stages.
- **The `$qualitative-check` verdict arm posts pinned token ids to
  `/v1/completions`,** so it deliberately does not exercise the reasoning parser
  or the chat endpoint that every graded eval used. That is the right choice for
  like-for-like comparison against earlier stages and the parser has its own
  control (`smoke/reasoning_parsed.json` vs `reasoning_control_unparsed.json`,
  identical 618 completion tokens), but no artifact exercises the chat path
  end-to-end as a *quality* verdict; the eval sample-health files are now the
  closest thing.
- **The qualitative artifacts date from the 23:35 server instance, not the
  02:35 instance that produced the graded report.** No model or serving code
  changed between them (only `max_gen_toks` in the TTI eval config), and the
  outputs are character-identical to the standalone model, so the evidence
  carries — but the release's qualitative row is, strictly, from a sibling
  instance.
- `qualitative_vllm_vs_datatype_sweep_chat.json` still records only
  `first_divergence` and 160-character heads, so the README's
  "character-identical … modulo one stripped special token" remains checkable
  only by going to the raw files (I did; it is true on all six prompts).

## Anomaly Ledger

- Observed anomaly: one graded `ifeval` turn (doc_id 162, key 1880) is 261,038
  characters, of which 259,858 are the literal character `c`, emitted inside the
  analysis channel until the 32768-token cap.
  Evidence: `samples_ifeval_2026-08-16T02-57-28.201205.jsonl`;
  `evals/ifeval_sample_health.json` `truncated_docs`.
  Affected path: model sampling at temperature 1.0 on an adversarial
  letter-frequency constraint × the parser's return-unsplit-on-truncation rule.
  Control or comparison: the **same document scored True** in both 8192-budget
  runs (527 and 721-character replies), and the strict per-instruction result
  for the degenerate turn is `[True, False]` — it satisfied the letter-frequency
  instruction and failed the word count, exactly as the docs claim a `c`-run
  would. 1 of 541, conservative direction, bounded at 0.18 points.
  Likely subsystem: sampling excursion, not a stage-introduced defect; the
  budget cap is what bounds it rather than causes it.
  Investigation performed: this review, plus the stage's per-document health
  artifact and its recorded follow-up to point the degenerate gate at eval
  samples.
  Resolution: controlled — disclosed, bounded and reproducibly scored; the
  control above should be cited in the docs (Other Concerns).

- Observed anomaly: `test_penalties[presence_penalty-1.2-repeat_trap]` fails and
  the release exits 1.
  Evidence: report `Detailed Test Results`
  (`0.12376237623762376 >= 0.15079365079365079 * 0.9`);
  `presence_penalty_repeats.json` 3/3 identical at ratio 0.8207399687.
  Affected path: `models/common/sampling/tt_penalties.py` + bf16 logit grid.
  Control or comparison: vLLM's own fp32 host sampler fails the same assertion
  more often than the device (1/4 vs 2/4; 0.3585 vs 0.9725 on the RNG-free
  greedy trial); presence-in-bf16 reproduces the device's greedy tokens 160/160
  where every rival rule is refuted; grid-aligned penalties 0.5/1.25/2.0 are
  byte-identical over 1024 greedy tokens.
  Likely subsystem: bf16 quantization of the penalty onto the logit ULP.
  Investigation performed: `AUTOFIX_presence_penalty.md`, H1–H6, six artifacts;
  round-1 acceptance re-checked here against the new report, which reproduces
  the identical ratio.
  Resolution: controlled — `issue-waived` upheld; both residuals (no issue URL,
  control shares the autoport's logits) are now stated beside the claim.

- Observed anomaly: run 1's `ifeval` had 6 empty responses.
  Evidence: `samples_ifeval_2026-08-15T19-47-39.652030.jsonl`, 6 zero-length
  responses, all scored False.
  Affected path: the pre-fix reasoning parser returning `content=None` on a turn
  truncated inside the analysis channel.
  Control or comparison: the shipped parser returns such turns unsplit; runs
  2–4 have 0 empty responses, and
  `tests/test_reasoning_parser.py::test_content_is_always_a_string` pins it
  (13 passed here).
  Likely subsystem: API-layer parser, fixed within the stage.
  Resolution: fixed — but the run-1 score is quoted in the reproducibility line
  without that context (Other Concerns).

- Observed anomaly: `DEGRADED PATH untraced_eager_decode` in the server log.
  Evidence: `logs/server_excerpt.log:167`, 02:39:03.008.
  Affected path: decode warmup before trace capture.
  Control or comparison: trace capture logs at 02:39:03.451 and decode warmup
  re-reports `trace=captured`; no further occurrence in the excerpt.
  Resolution: controlled — warmup-time only; see the excerpt-coverage gap above.

## Scope Inspected

- Goal/skill paths: `.agents/skills/{stage-review,tti-release,qualitative-check,autofix,tt-device-usage}/SKILL.md`
  (device-usage indirectly — no reset or ARC event occurred).
- Artifact paths: all 100 files under `doc/tti_release/` — `README.md`,
  `RUN_NOTES.md`, `stage_review.md` (round 1), the new
  `report/report_*_2026-08-16_04-04-24.{md,json}`, both `run_spec/*.json`, both
  `evals/results_*.json`, both `evals/*_sample_health.json`, all 18
  `benchmarks/*.json`, all 16 `logs/*`, `qualitative/`, `qualitative_runner/`,
  `smoke/`, `seeding/` (all 9 scripts + 2 traces + evidence), `presence_penalty/`
  and the five loose `presence_penalty_*.json`, `non_aligned_probe.json`, both
  `AUTOFIX_*.md`, `tti_local_edits/…patch`, `bench/` (all 8 scripts), `server/`.
  Cross-stage: `doc/optimized_full_model/perf_summary.json`,
  `doc/datatype_sweep/qualitative/qualitative_tt_chat.json`.
  Outside the tree (read-only): the TTI checkout
  (`git status`/`git diff`, `eval_config.py`, `eval_utils.py`,
  `benchmark_config.py`, `workflow_venvs.py`,
  `model_performance_reference.json`), and the uncopied
  `samples_{ifeval,aime25}_*.jsonl` for all four ifeval and four aime25 runs.
- Code paths: `models/common/sampling/tt_sampling.py` (diff),
  `models/common/modules/sampling/sampling_1d.py`,
  `models/experimental/llama32_1b_quasar/sampling/tt_sampling.py`,
  `models/autoports/meta_models_muse_glimmer_30b/{tt/reasoning_parser.py,tests/test_reasoning_parser.py,.gitignore}`.
- Commands run: `git status/diff/log/rev-parse` in both repos; `diff` of the
  copied patch against the live TTI diff; `diff` of all nine `/tmp` seeding
  probes against their `seeding/` copies;
  `pytest .../tests/test_reasoning_parser.py -q` (13 passed, host-only);
  read-only Python over the copied JSON, the benchmark JSON, and the uncopied
  `samples_*.jsonl`; `tmux ls`, `ps`, `docker ps -a`, `ls /dev/tenstorrent`,
  `find -size +2M`, secret grep. No server started, no TT device opened, no
  hardware or vLLM experiment run.

## Residual Risk

- `meta_gpqa_cot` still has no substitute measurement of GPQA itself. The
  `meta_*` substitution rationale is now artifact-backed
  (`logs/meta_evals_dataset_404.json` plus `workflow_venvs.py:444`
  `evals_dataset = f"{_model_name}-evals"`, which I re-read), and
  `Idavidrein/gpqa` remains gated for this account. Open follow-up, as recorded.
- Both accuracy references are vendor-published and cross-benchmark
  (IFBench→IFEval) or cross-year (AIME 2026→AIME 2025). The rows establish "not
  broken", not "matches reference"; the ifeval bar is 73.15 against a measured
  94.45, the aime25 bar is 89.965 at the default tolerance against a measured
  90.00.
- The bf16 penalty quantization and the presence→frequency→repetition term
  ordering remain real, unfixed semantic differences from vLLM;
  `models/common/sampling/tt_penalties.py` still has no test file.
- `ttnn.manual_seed`'s per-core RNG state is still destroyable by any op
  scheduled between it and `ttnn.sampling`, with no error. The fix is an
  ordering constraint in shared code protected only by a comment; the named
  op-level regression test is still not written.
- Streaming responses cannot offer the non-streaming unsplit guarantee. Every
  graded path here is non-streaming, so this is unexercised rather than
  validated.
- `complete` and `target` performance tiers fail at 38 % of the memory-side
  roofline — expected at `FUNCTIONAL`, but the released number is a
  first-bring-up number.
- The graded `ifeval` 94.45 is a floor by at most 0.18 points, and one further
  sampling excursion of the doc-162 kind on a different prompt would move it by
  the same amount; the score's run-to-run spread across the four runs is
  ~1.3 points, larger than the truncation effect.
