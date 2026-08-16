# Stage Review

Stage 11 — TTI release (tt-inference-server model release workflow)
Target: `meta-models/Muse-Glimmer-30B`, autoport `models/autoports/meta_models_muse_glimmer_30b`
Branch `agentic-research/hous/muse-glimmer-30b`, live worktree, uncommitted.
Reviewer mode: independent subagent, read-only (no server, no device, no hardware run).

Verdict: more-work-needed

No P1. The technical core of this stage holds up under independent re-derivation:
the evaluated implementation really is the autoport, the context contract is
preserved end to end, the seeding root cause and fix are well proven, the
reasoning parser provably does not change generation, and the presence-penalty
waiver is measurement, not prose. The required work below is six specific,
cheap corrections — one graded-row measurement defect that the stage fixed for
`aime25` but never checked for `ifeval`, and five places where the handoff
states something its own artifacts do not support.

What I re-derived myself (not taken from the READMEs):

* the release chat completions are **character-identical** to the
  datatype-sweep stage's standalone-model completions on all 6 prompts once the
  single `<\|message\|>` token is removed (381/559/628/600/561/703 chars, first
  differing character `None` on every prompt). The README claim is true, and
  stronger than what the copied comparison JSON alone shows;
* the graded `ifeval` score is `prompt_level_strict_acc`, **not** a mean
  (`reference_config/evals/eval_utils.py:7` — `score_task_single_key` takes
  `result_keys[0]`), 0.9463955637707948 → 94.6395…;
* the final `aime25` run: 30 samples, **0 empty**, 27 correct, every response
  contains `\boxed`, response length 313–1287 chars → the parser did hand
  lm-eval the visible channel only;
* the 32768-budget `aime25` run really did have **4 empty responses**, and the
  98304 run has 0 — the budget raise is evidence-backed;
* `tt-inference-server` is `82777a238`, `VERSION` 0.20.0, and
  `git merge-base --is-ancestor 6e396b4 HEAD` succeeds (#4345 present);
* the copied `tti_local_edits/…patch` is **byte-identical** to the live
  `git diff` of the TTI checkout, and the checkout has no untracked files —
  no undisclosed local edit, and no test file touched;
* `(131072, 128)` is dropped by TTI's own build-time filter
  (`reference_config/benchmarking/benchmark_config.py:161`, `isl + osl >
  max_context`), not by this stage;
* the 13 reasoning-parser unit tests pass here (`13 passed in 5.59s`);
* vLLM 0.24.0 constructs the parser **per request**
  (`entrypoints/openai/chat_completion/serving.py:262`), so the parser's
  `_emitted_reasoning`/`_emitted_content` streaming counters cannot leak across
  requests;
* cleanup is real: no tmux server, no `vllm`/`EngineCore` process, no
  container, `/dev/tenstorrent/{0,1,2,3}` free, no `.env` in the TTI checkout,
  no secret or >2 MB file in the evidence tree.

## Required Work

- P2: `ifeval` was graded with 3 of 541 turns truncated inside the analysis
  channel — the exact defect the stage found and fixed for `aime25`, never
  checked for `ifeval`
  Evidence: `samples_ifeval_2026-08-16T00-04-08.193939.jsonl` (in the TTI cache,
  not copied) contains 3 responses of 27,749 / 28,070 / 29,607 characters that
  begin `" to=selfWrite a short proposal…"` — i.e. `max_gen_toks=8192` was
  exhausted mid-analysis, `reached_visible_channel()` returned False, and
  `tt/reasoning_parser.py:243-247` handed the whole analysis back as `content`.
  ifeval scored the analysis for those three (doc keys 1300, 2273, 3048; 2 of 3
  scored `prompt_level_strict_acc = False`). The immediately preceding run
  (`21-59-11`) has the same 3. The eval config comment justifying 8192 says
  "the analysis channel alone runs to ~600 tokens on a four-sentence
  instruction" — measured here it runs past 8192 on ~0.6 % of ifeval prompts.
  Why this matters: the same under-budget symptom was treated as a harness
  defect for `aime25` and fixed by raising the budget to 98304; leaving it in
  place for `ifeval` means the graded 94.64 is not a clean measurement of the
  model's instruction following, and the reported number is a floor rather than
  a result. The direction is conservative (it depresses the score), so this is
  not a hidden failure — it is an inconsistently applied fix on a graded row.
  Required next step: either raise `ifeval`'s `max_gen_toks` the way `aime25`'s
  was raised and rerun `--workflow evals` (the run spec, server and report
  wiring are unchanged, so this does not need the full release), or record the
  measured truncation count, the affected doc keys and the scoring effect in
  `RUN_NOTES.md`/`README.md` and label 94.64 explicitly as a floor.

- P2: `RUN_NOTES.md` misstates how the headline `ifeval` number is computed
  Evidence: `RUN_NOTES.md:167` — "`ifeval` is scored as the mean of
  `prompt_level_strict_acc` (94.64) and `inst_level_strict_acc` (96.28)". The
  mean of those two is 95.46; the graded 94.64 is `prompt_level_strict_acc`
  alone, because `score_task_single_key`
  (`reference_config/evals/eval_utils.py:6-17`) uses `result_keys[0]` and
  ignores the rest, while the config
  (`tti_local_edits/…patch`, ifeval `score_func_kwargs`) lists two keys — which
  reads as if a mean was intended (`score_task_keys_mean` exists in the same
  file). The "reproducibility across the three runs" line
  (`RUN_NOTES.md:171-172`, 94.82 / 96.07 / 95.46) quotes composites, not the
  graded metric; recomputed from the three sample files the graded values are
  **94.09 / 95.38 / 94.64**, and the two earlier runs' results JSON were not
  copied, so the line cannot be checked from the evidence tree.
  Why this matters: this is the release handoff's description of the number the
  acceptance gate actually used.
  Required next step: correct the sentence and the reproducibility line to the
  graded metric, and either drop the second `result_keys` entry or switch the
  config to `score_task_keys_mean` so the config states what it does.

- P2: `AUTOFIX_seeding.md` cites an evidence file that is not in the tree
  Evidence: `AUTOFIX_seeding.md:17` cites
  `doc/tti_release/logs/tti_release_20260816T014529Z.log` for the original
  failure. That file does not exist. The run it names is present under the
  misleading name `logs/tti_release_prefix_run2_prefix.log` (its first line is
  `2026-08-15 21:45:29` = `20260816T014529Z`, `workflow: release`,
  `prefix_cache` not used), and it is the only copied log containing
  `Determinism Failed`.
  Why this matters: it is the primary citation for the failure that justified a
  change to shared code used by every model in `models/common/sampling/`.
  Required next step: rename the log back to the cited name (or fix the
  citation), and drop the "prefix" label, which describes a different workflow.

- P2: `smoke/conformance_probe.json` is unlabelled pre-fix evidence that
  contradicts the shipped parser
  Evidence: the file (mtime 20:43, before the truncation fix) records
  `"content": null` for `echo_max32_default_reasoning`,
  `echo_max32_reasoning_low` and `stop_seq_count_to_5`. The shipped parser
  returns those turns unsplit — `README.md:132-137` states "`content` is a
  string for every response this server can produce", and
  `tests/test_reasoning_parser.py::test_content_is_always_a_string` pins it.
  Unlike every other artifact in the tree, this file has no `_what` key saying
  which arm it is.
  Why this matters: a reader checking the parser's central guarantee against the
  smoke evidence finds the opposite of the claim.
  Required next step: add a `_what` line marking it as the pre-fix arm (it is
  useful as such — it is the evidence that `content=None` broke four
  conformance rows), or re-take it against the shipped parser.

- P2: the "no packaged implementation appears in the run spec" claim is not
  literally true
  Evidence: `README.md:35` and `RUN_NOTES.md:49` assert that
  `models/tt_transformers`, `models/demos`, `tt_vllm_plugin` and
  `tt-media-server` are absent from the copied run spec "(string search over the
  whole JSON)". `tt_transformers`, `models/demos` and `tt_vllm_plugin` are
  indeed absent (I re-ran the search), but both `run_spec/*.json` contain
  `"docker_image": "ghcr.io/tenstorrent/tt-media-inference-server:0.20.0-7db0eca"`.
  The claim only passes because the searched string was hyphenated differently
  from the string that is present.
  Why this matters: this is the implementation-path proof, the one claim in the
  stage that must be exact. The finding is cosmetic in substance — Docker was
  never used and `impl.code_path` is the autoport — but the sentence as written
  is contradicted by the artifact it cites.
  Required next step: restate it as "the only non-autoport reference is the
  unused catalog-default `docker_image` field; `impl.code_path`, `code_link` and
  `model_impl` all name the autoport".

- P2: the release ships the text stack of a multimodal checkpoint, and the
  handoff does not say so
  Evidence: the HF config
  (`…/snapshots/f84ecc3…/config.json`) is
  `MuseGlimmerForConditionalGeneration` with `vision_config`, `image_token_id`
  and `video_token_id`; `doc/functional_decoder/README.md:531-539` flagged the
  vision tower as out of scope then but "waiting for later stages", and warned
  that "a full-model or serving stage that ignores … the vision tower will be
  wrong". The release spec sets `supported_modalities: [text]` (which is what
  suppresses image benchmarks), and the only explanation is a comment inside
  `tti_local_edits/…/llm.yaml`. `README.md`'s *Limitations and follow-ups* and
  `RUN_NOTES.md` do not mention it at all.
  Why this matters: this is the customer-facing release stage, and the largest
  capability caveat of the released artifact is the one caveat not listed beside
  the six that are.
  Required next step: add it to `README.md` *Limitations* and `RUN_NOTES.md` —
  text-only release of a multimodal checkpoint, image/video input unsupported,
  `supported_modalities: [text]` set for that reason, vision tower not ported by
  any stage. No porting work is being asked for here; only the disclosure.

## Other Concerns

- **The presence-penalty waiver has no issue URL, and its control shares the
  autoport's logits.** `$tti-release` allows `issue-waived` on either a linked
  issue/release note *or* proof the target is invalid for reasons unrelated to
  the autoport, and the stage takes the second route with four measurements —
  160/160 rule reproduction, the closed-form bf16 tie, 3/3 grid-aligned
  penalties byte-identical over 3072 greedy tokens, and vLLM's own fp32 host
  sampler failing the same assertion 3/4 against the device's 2/4. That is
  substantively stronger than a link. Two residuals: (a) there is no issue URL
  or release-note reference a downstream reader can follow, and (b) the "zero
  Tenstorrent code in the sampling path" control still runs on TT-produced
  logits, so it isolates the penalty implementation but is not a canonical-model
  control (no GPU on this host — `libcuda.so.1` missing — so none was possible).
  I accept the waiver; recommend recording (b) explicitly beside the claim.
- **The failing row is not stable across server instances.**
  `AUTOFIX_seeding.md:168-173` records a post-fix run where the whole file was
  `22 passed`, presence-penalty included; `presence_penalty/conformance_after.log`
  and the release run are `21 passed, 1 failed`. Within the shipped instance it
  is exactly deterministic (`presence_penalty_repeats.json`: 3/3 identical
  texts, ratio 0.8207 three times, matching the report's failure message
  verbatim), and H5 shows the row is a coin flip across seeds. The cross-instance
  flip is disclosed only inside `AUTOFIX_seeding.md`; `README.md`'s "The one
  failing row" reads as if it always fails. One sentence would close it.
- **The shared-code seeding fix is guarded only by a comment.**
  `models/common/sampling/tt_sampling.py` now depends on `ttnn.manual_seed`
  being the last op before `ttnn.sampling`, with a silent failure mode; the
  stage names the exact regression test it would add
  (`tests/ttnn/unit_tests/operations/reduce/test_manual_seed.py`, a
  `typecast → bfloat16` between the two ops) and declines it as out of scope.
  The fix itself is minimal and correct — I read the diff: a pure move, no
  behaviour change for unseeded requests beyond drawing from the RNG state that
  was actually installed, and the other in-tree caller
  (`models/experimental/llama32_1b_quasar/sampling/tt_sampling.py:762`) has no
  intervening op, so it needs no change. Worth adding the test on the next
  touch of this file.
- **The op-level seeding probes live in `/tmp`.** `seeding/evidence.json` cites
  `/tmp/rand_probe{,2,3,4,5}.py`, `/tmp/seed_probe_host.py`, `/tmp/seed_trials.py`,
  `/tmp/seed_repro{,2}.py`. The numbers are recorded, the scripts are not
  copied (unlike `presence_penalty/`, which copies all four). The bisect that
  names `typecast → bfloat16` is therefore not reproducible from the evidence
  tree.
- **No artifact for the `-evals` 404.** The justification for substituting
  `meta_ifeval`/`meta_gpqa_cot` rests on `meta-models/Muse-Glimmer-30B-evals`
  returning 404; unlike the GPQA blocker there is no log. I confirmed the
  structural half locally: `workflows/workflow_venvs.py:444` builds
  `evals_dataset = f"{hf_model_repo}-evals"`, and every `meta_*` entry in
  `eval_config.py` is a `meta-llama/*` repo. The substitution is factual, not
  convenient; only the probe output is missing.
- **The `ifeval` accuracy gate is very weak as configured.** The check is
  one-sided (`score / reference >= 1 - tolerance`,
  `reference_config/evals/eval_config.py:130`), and the reference is IFBench
  77.0 — a different, harder benchmark — so anything ≥ 73.15 passes for a model
  that scores 94.6. That is disclosed and argued (conservative floor, no
  measured GPU reference possible here), and `aime25` at 0.9504 vs a 0.95 bar is
  a genuinely tight gate, so the suite as a whole is not toothless. Recording
  that the ifeval row is a floor-check rather than an equivalence-check in the
  README status table would be honest.
- **`aime25`'s loosened tolerance is not what makes it pass.** At the default
  0.05 the bar is 89.965 and the reported 90.0 passes (0.95037 ≥ 0.95); the
  0.10 tolerance is what would have covered the 86.67 run. So the tolerance is
  not tuned-to-pass for the reported result, but it is load-bearing for
  run-to-run robustness, and the reported run clears the default bar by
  0.0004. Worth stating plainly rather than leaving the reader to compute it.
- **Spec/server config drift.** The run spec's `override_tt_config` carries
  `sample_on_device_mode: all`, which `bench/serve_release.sh`'s `--tt-config`
  does not set. Harmless here (TTI never launched a server), but the spec is
  what a future `--docker-server`/`--local-server` run would use, so it would
  not reproduce the configuration that was validated.

## Hard-Check Gaps

- Nothing in the copied tree lets a reader check per-sample eval health
  (finish reason, response length, empty responses). Both of this stage's eval
  findings — the `aime25` budget and the `ifeval` truncation above — are only
  visible in the uncopied `samples_*.jsonl`. A tiny per-doc summary
  (doc id, response length, finish reason, score — no text) would be a few KB
  and would have caught the ifeval case during the stage.
- `qualitative_vllm_vs_datatype_sweep_chat.json` records only token-list first
  divergence and 160-character heads, so the README's "character-identical …
  modulo one stripped special token" is not checkable from the artifact. It is
  true — I verified all six prompts in full — but the harness should record the
  stripped-token comparison it is claiming.
- `qualitative/qualitative_prompt_format.json` is emitted by the shared runner
  with `"stage": "vllm_integration"` and `"arm": "vllm-serving"` baked in, so
  the stage-11 copy self-labels as a different stage. Cosmetic, but it makes the
  prompt-format record look copied rather than re-run (it was re-run: mtime
  23:35, against this stage's server).
- `bench/copy_back.sh` selects benchmark JSON by "18 most recent" rather than by
  the run stamp it already computed. It happened to be right here (all 18 files
  fall inside 00:41:48–01:01:03 of the graded run), but a partial later sweep
  would silently mix runs.

## Anomaly Ledger

- Observed anomaly: `test_penalties[presence_penalty-1.2-repeat_trap]` fails, and
  the release exits 1.
  Evidence: report `Detailed Test Results`; `presence_penalty_repeats.json`
  (0.1508 → 0.1238, ratio 0.8207, 3/3 identical).
  Affected path: `models/common/sampling/tt_penalties.py` + bf16 logit grid.
  Control or comparison: vLLM's own fp32 host sampler on the same model and
  prompt fails the same assertion 3/4 vs the device's 2/4, and 0.3585 vs 0.9725
  on the RNG-free greedy trial; five candidate rules scored against the device's
  own tokens leave only vLLM's rule-in-bf16 standing at 160/160.
  Likely subsystem: sampling penalties (bf16 quantization of the penalty onto
  the logit ULP: effective 1.25 for a requested 1.2).
  Investigation performed: `AUTOFIX_presence_penalty.md`, H1–H6, six artifacts.
  Resolution: controlled — `issue-waived` accepted, with the two residuals noted
  under *Other Concerns*.

- Observed anomaly: 3 of 541 `ifeval` responses are 27–30 kB of analysis-channel
  text, headers included.
  Evidence: `samples_ifeval_2026-08-16T00-04-08…jsonl`, doc keys 1300/2273/3048.
  Affected path: eval `max_gen_toks=8192` × `tt/reasoning_parser.py`'s
  return-unsplit-on-truncation rule.
  Control or comparison: the `aime25` equivalent was found and fixed by raising
  the budget (4 empty at 32768 → 0 at 98304, verified in the sample files).
  Likely subsystem: eval harness configuration, not the model.
  Investigation performed: this review.
  Resolution: more-work-needed — see Required Work #1.

- Observed anomaly: TT and the HF control diverge early on all six qualitative
  prompts, and on p1 TT answers directly (`to=user`) where HF opens an analysis
  channel (`to=self`).
  Evidence: `qualitative/qualitative_comparison_chat.json`
  (`first_divergence_from_hf` 1–2, `exact_match: false` on all six).
  Affected path: whole-model numerics.
  Control or comparison: TT output is **character-identical** to the
  datatype-sweep stage's standalone TT model on all six prompts (verified
  here), and the degeneracy metrics are comparable to HF's
  (`trigram_loop` 0.047–0.15 TT vs 0.047–0.117 HF, `adjacent_dup` 0.0,
  `replacement_char_fraction` 0.0).
  Likely subsystem: bf16 accumulation, pre-existing and characterised by
  earlier stages.
  Resolution: controlled — not stage-introduced, no regression.

- Observed anomaly: `qualitative_runner` p4 greedy has
  `trigram_loop_fraction = 0.5`, the highest value in the degenerate check.
  Evidence: `qualitative/degenerate_check.json`.
  Affected path: raw-completion stress arm.
  Control or comparison: the text is
  `".\n\nBonjour, comment allez-vous aujourd'hui?"` — a 6-word correct
  translation; the metric is noise at that length. The earlier stage's 3-of-12
  verbatim-loop caveat does not reproduce here (0 of 12).
  Resolution: controlled — not a defect.

- Observed anomaly: server logs
  `Auto-initialization of reasoning token IDs failed`.
  Evidence: `logs/server_excerpt.log:26`.
  Affected path: `reasoning_config.enabled`.
  Control or comparison: the stage traced the flag to a single consumer
  (`thinking_token_budget`), which no release path uses.
  Resolution: controlled — classified, no correctness impact.

## Scope Inspected

- Goal/skill paths: `.agents/skills/{stage-review,tti-release,qualitative-check,autofix}/SKILL.md`
  (`tt-device-usage` covered indirectly — no reset/ARC event occurred in this stage).
- Artifact paths: all of `doc/tti_release/` — `README.md`, `RUN_NOTES.md`,
  `report/report_*.md` + `report_data_*.json`, `run_spec/` (both files),
  `evals/results_*.json`, all 18 `benchmarks/*.json`, `logs/` (release run log,
  smoke, evalsmoke, two earlier release logs, server excerpt, qualitative and
  degenerate logs, gpqa log), `qualitative/`, `qualitative_runner/`, `smoke/`,
  `seeding/`, `presence_penalty/` + the five loose `presence_penalty_*.json`,
  `non_aligned_probe.json`, `AUTOFIX_seeding.md`, `AUTOFIX_presence_penalty.md`,
  `tti_local_edits/…patch`, `bench/` (all 7 scripts), `server/`.
  Cross-stage: `doc/context_contract.json`, `doc/optimized_vllm/README.md`,
  `doc/functional_decoder/README.md`, `doc/datatype_sweep/qualitative/`,
  `doc/vllm_integration/bench/qualitative_vllm.py`.
  Outside the tree (read-only): the TTI checkout at
  `/home/ttuser/dev/muse-glimmer/tti-release/muse-glimmer-30b/tt-inference-server`
  (`git status`/`git diff`, `eval_utils.py`, `eval_config.py`,
  `llm_eval_tests.py`, `benchmark_config.py`, `workflow_venvs.py`,
  `test_module/`), the uncopied `samples_{ifeval,aime25}_*.jsonl` for all three
  runs, the HF `config.json`, and the installed vLLM 0.24.0 serving code.
- Code paths: `models/common/sampling/tt_sampling.py` (diff + surrounding
  `__call__`), `models/autoports/meta_models_muse_glimmer_30b/tt/reasoning_parser.py`,
  `models/autoports/meta_models_muse_glimmer_30b/tests/test_reasoning_parser.py`,
  `models/autoports/meta_models_muse_glimmer_30b/.gitignore`,
  `models/experimental/llama32_1b_quasar/sampling/tt_sampling.py` (the other
  `manual_seed` caller).
- Commands run: `git status/diff/log` in both repos; `diff` of the copied patch
  against the live TTI diff; `git merge-base --is-ancestor 6e396b4 HEAD`;
  `pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_reasoning_parser.py -q`
  (13 passed, host-only, no device); read-only Python over the copied JSON and
  the uncopied `samples_*.jsonl`; `tmux ls`, `ps`, `docker ps`,
  `ls /dev/tenstorrent` for leftovers. No server started, no TT device opened,
  no hardware or vLLM experiment run.

## Residual Risk

- `meta_gpqa_cot` has no substitute measurement of GPQA itself; `aime25` covers
  the reasoning dimension but the GPQA row stays open until the HF account is
  granted `Idavidrein/gpqa`. Recorded as a follow-up by the stage; agreed.
- Both accuracy references are vendor-published, and one is from a different
  benchmark (IFBench→IFEval) and one from a different contest year
  (AIME 2026→AIME 2025). No GPU control is possible on this host. The accuracy
  rows therefore establish "not broken", not "matches reference".
- The bf16 penalty quantization (`effective = round_to_grid(bf16(P), ULP)`,
  +0.05 at P=1.2) is a real, unfixed semantic difference from vLLM. Bounded at
  half a ULP and only observable at exact ties, but it is a difference a
  customer could hit.
- Penalty term ordering differs from vLLM (device presence→frequency→repetition
  vs vLLM repetition→frequency→presence). Unobservable in this suite; becomes
  observable the first time `repetition_penalty != 1.0` is combined with a
  presence or frequency penalty. `models/common/sampling/tt_penalties.py` still
  has no test file.
- `ttnn.manual_seed`'s per-core RNG state remains destroyable by any op
  scheduled between it and `ttnn.sampling`, with no error. The fix is an
  ordering constraint in shared code protected by a comment only.
- Streaming responses cannot offer the non-streaming unsplit guarantee; a turn
  truncated inside analysis streams as reasoning with no content. Every graded
  path here is non-streaming, so this is unexercised rather than validated.
- `complete` and `target` performance tiers fail at 38 % of the memory-side
  roofline. Expected at `FUNCTIONAL`, and the roofline is compute-free so the
  tiers are strict rather than lenient — but the released number is a
  first-bring-up number.
