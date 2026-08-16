# Muse-Glimmer-30B — tt-inference-server release

The customer-facing release workflow for the generated autoport
`models/autoports/meta_models_muse_glimmer_30b`, run as a **client** of the
autoport's own vLLM server on a 4-die Blackhole P300_X2 at the full advertised
131072-token context. No Docker, no stock `tt-transformers`, no reduced context.

**Read this first.** The release passes on accuracy and on its enforced
performance tier, and the interesting part of the stage is what making the
release *runnable* uncovered. Wiring the model into tt-inference-server's own
conformance and eval suites — which nothing before this stage had done — exposed
two defects that every earlier gate had missed, both in code shared across
autoports rather than in this model's own files:

* **concurrent seeded decode was not reproducible**, because `ttnn.manual_seed`
  installs per-core RNG *registers* and a `typecast` kernel between it and
  `ttnn.sampling` wiped them on the cores it touched. Fixed by one moved call.
  This also retires the "seeded reproducibility at batch > 1" limitation that the
  vLLM-integration and optimized-vLLM stages both recorded as unresolved;
* **the OpenAI API was returning the model's private analysis channel as its
  answer**, because Muse Glimmer is a channelled model and no reasoning parser
  was configured. Fixed by adding one.

One conformance row still fails and is `issue-waived` with a control showing
vLLM's own float32 reference fails it more often than this autoport does.

Full detail, commands, versions and cleanup in [`RUN_NOTES.md`](RUN_NOTES.md).

## Status

| gate | result |
|---|---|
| `run.py --workflow release`, unrestricted (no `--limit-samples-mode`) | **exit 1**, caused solely by the one waived conformance row below |
| Release readiness | **`release-readiness-pass`** — full-set accuracy, every required row passing or issue-waived |
| Implementation evaluated | **`models/autoports/meta_models_muse_glimmer_30b`** — proven by the copied run spec's `impl.code_path`, its `code_link` and the report's `model_impl`. `models/tt_transformers`, `models/demos` and `tt_vllm_plugin` appear nowhere in either. The only non-autoport string in the run spec is the catalog-synthesised `docker_image` field, which is unread here: no Docker was used and no image was pulled |
| `ifeval` (541/541, full set) | **94.45** (`prompt_level_strict_acc`) vs a 77.0 reference, ratio 1.227 — ✅ PASS. A **floor check**, not an equivalence check: the reference is IFBench, a harder benchmark, so the bar is 73.15. The score is itself a floor, understated by at most 0.18 points — see *Eval measurement health* |
| `aime25` (30/30, full set) | **90.00** vs a 94.7 reference, ratio 0.950 — ✅ PASS. A tight gate: it clears even the *default* 0.05 tolerance (bar 89.965) by 0.035, so the configured 0.10 is not what makes it pass |
| Benchmark target, functional tier | ✅ PASS — TTFT 72.1 ms vs ≤ 88.3, decode 43.4 t/s/u vs ≥ 11.33 |
| Benchmark target, complete + target tiers | ❌ FAIL, informational at `FUNCTIONAL` status by design — 38 % of the memory-side roofline |
| Benchmark sweep, ISL 127 → 65535 | **18/18 points completed, 0 failed requests** |
| API parameter conformance | **21 of 22** parametrizations pass |
| `test_penalties[presence_penalty-1.2-repeat_trap]` | ❌ FAIL — **issue-waived**, see below |
| Served context vs `doc/context_contract.json` | **131072 = 131072**, no reduction anywhere |
| Non-aligned prompt lengths | **9/9** dedicated probe, plus 18/18 sweep points at odd ISL |
| `$qualitative-check` shared suite | **pass** — coherent on all 6 prompts, `replacement_char_fraction` 0.0000, and character-identical to the standalone model modulo one stripped special token |
| Degenerate-output gate, `--scope all` | **pass**, exit 0 |
| Hardware | no resets, no hangs, no ARC/ERISC/Ethernet events, no `tt-triage` needed |

## What had to be built to make the release evaluate *this* model

A built-in TTI model name selects a stock implementation, so five things were
added to the (scratch, uncommitted) tt-inference-server checkout — full diff in
[`tti_local_edits/`](tti_local_edits/):

1. a `muse_glimmer_30b_autoport` `ImplSpec` whose `code_path` is the autoport;
2. a catalog entry for the model on `P300X2`. This is not optional decoration:
   `EVAL_CONFIGS` is built by iterating `MODEL_SPECS`, so a runtime spec the
   catalog has never seen gets **no eval tasks at all**. It also carries
   `max_tokens_all_users_override: 1050624` — the KV pool the autoport actually
   allocates, read off the running server — without which the pool is inferred as
   `max_context` and benchmark concurrency is understated 8×;
3. an `EvalConfig` with `ifeval` + `aime25`;
4. P300X2 performance targets;
5. registration in the spec-test suites. Without this the release logged
   *"No spec test suites match model='Muse-Glimmer-30B' device='p300x2' —
   skipping spec_tests"* and produced a report with **zero API conformance
   coverage** while still exiting 0.

The spec handed to `run.py` is *derived* from (2) by
[`bench/export_runtime_spec.py`](bench/export_runtime_spec.py), which asserts the
autoport code path and the context contract before writing, so the catalog entry
and the runtime JSON cannot drift apart. Its embedded `cli_args` are set for the
external-server topology (`docker_server=false`, `local_server=false`,
`service_port=8000`, `workflow=release`) rather than relying on command-line
flags to override a loaded JSON.

No TTI test was edited, relaxed or skipped.

## The two bugs this stage found

### Concurrent seeded decode was not reproducible

`test_non_uniform_seeding` sends 32 concurrent requests, 16 with `seed=0`, and
requires those 16 to be byte-identical. One or two diverged every run — always at
the same decode step, always to the same alternative continuation, but in a
request position that moved run to run.

`ttnn.manual_seed` installs the per-token RNG state as a **register on each
core**; `ttnn.sampling` advances it. Nothing carries it in a tensor. The shared
sampler called `manual_seed`, then ran ~17 elementwise ops
(`_adjust_values_for_tiebreak`), then `ttnn.sampling`. One of those ops,
`ttnn.typecast(int32 → bfloat16)`, destroys the RNG state on the cores it runs
on — so users mapped to those cores drew a different random number from an
identical seed. The affected batch slots were exactly `{0, 11, 22}`.

The fix moves `ttnn.manual_seed` to be the last op before `ttnn.sampling`
(`models/common/sampling/tt_sampling.py`). Evidence, including an op-level bisect
that names `typecast → bfloat16` and `exp` as breaking while `add`, `abs`, `max`,
`eq`, `typecast → int32` and `untilize` are clean, is in
[`AUTOFIX_seeding.md`](AUTOFIX_seeding.md).

Afterwards: `test_non_uniform_seeding` passes 3/3, the 32-way reproduction gives
1 distinct output, and the shared plugin suites
`test_seeding_and_variety.py` + `test_request_isolation.py` go from 6–7 failures
to **29 passed**. `doc/vllm_integration/README.md` *Limitations 1* and
`doc/optimized_vllm/README.md` *Sampling suite* describe that as an unresolved
limitation; **those two documents are superseded on this point.**

The kernel behaviour is *avoided*, not fixed: any op inserted between the seed
call and `ttnn.sampling` will silently re-break seeding with no error. That is
recorded as the upstream ask.

### The API returned the analysis channel as the answer

Muse Glimmer's chat template ends the assistant prompt at `<|start|>assistant`
and the model writes its own channel header, so a turn is `to=self` (analysis)
then `to=user` (the reply). With no reasoning parser, vLLM hands both back
concatenated as `message.content` and leaves `reasoning_content` null.

That is not a cosmetic problem. Asked for a four-sentence summary *in all
lowercase*, the unparsed response is 618 tokens that open
`" to=selfWrite a 4 sentence summary..."` and are full of capitals; the reply the
model actually produced obeys the instruction perfectly. An instruction-following
eval reading `content` scores the analysis.

[`tt/reasoning_parser.py`](../../tt/reasoning_parser.py) routes the `self`
channel to `reasoning_content`. It is API-layer text routing only — same
sampling, same generator, same tokens on device — and the control in
[`smoke/`](smoke/) shows the same greedy request producing the **identical 618
completion tokens** either way, with `reasoning_content + content` reconstructing
the unparsed string exactly, minus the two channel headers.

It also never removes information: a turn cut off inside the analysis channel is
returned *unsplit*, so `content` is a string for every response this server can
produce. Returning `content=None` there — what vLLM's `<think>`-style parsers do
— broke four conformance rows with `TypeError: argument of type 'NoneType' is not
iterable` before that was fixed. 13 host-only unit tests pin the behaviour.

## The one failing row

`test_penalties[presence_penalty-1.2-repeat_trap]` asserts
`unique_ratio(penalty) >= unique_ratio(base) * 0.90` on "Write a very repetitive
story.". Measured deterministically: 0.1508 → 0.1238, ratio **0.8207**, FAIL.

Classified **`issue-waived`**, on measurement rather than disclosure — four
independent results, all in [`AUTOFIX_presence_penalty.md`](AUTOFIX_presence_penalty.md):

1. the device computes vLLM's exact rule on vLLM's exact token set — presence-in-bf16
   reproduces the device's greedy tokens **160/160**, while by-count contradicts it
   at step 30, prompt∪output at step 88, and no-penalty at step 9;
2. without a penalty the device and vLLM's host sampler are **byte-identical over
   1172 greedy characters**; with `presence_penalty=1.2` they agree to character
   725 and then differ because `bf16(1.2) = 1.203125` lands on a 0.125-spaced
   logit grid as exactly 1.25, creating a genuine tie;
3. the falsifiable prediction that follows — grid-exact penalties must match
   exactly — holds for **0.5, 1.25 and 2.0 over 1024 greedy tokens, 3 of 3**;
4. and the assertion is not a property of the implementation at all: run against
   **vLLM's own float32 host sampler**, with zero Tenstorrent code in the sampling
   path, the reference fails the same assertion *more often* than the device
   (1 pass/4 vs 2 pass/4), and in the greedy trial with no RNG anywhere the
   reference scores 0.3585 (FAIL) against the device's 0.9725 (PASS).

`frequency_penalty` and `repetition_penalty` pass on all three of the suite's
prompts. There is no upstream issue URL because filing one is outside this
stage's authority; (4) is a control against the canonical implementation, which
is what the waiver rests on.

Two things about that control are worth stating plainly rather than leaving a
reader to notice them:

* **it isolates the penalty, not the model.** vLLM's host sampler is reference
  fp32 code with no Tenstorrent arithmetic in it, but it consumes logits this
  autoport produced. A true canonical-model control would need the checkpoint on
  a GPU, and this host has none (`libcuda.so.1` is absent), so none was possible.
  The control settles "is the *penalty* implementation at fault"; it does not
  settle "would the reference *model* answer this prompt the same way".
* **the row is deterministic within a server instance but not across them.** In
  the shipped instance it fails identically three times out of three
  (`presence_penalty_repeats.json`, ratio 0.8207 each time, matching the report's
  failure message verbatim). One earlier post-fix instance ran the whole
  conformance file at `22 passed`, presence-penalty included
  (`AUTOFIX_seeding.md`). The row is a coin flip across seeds and instances, not
  a standing failure — which is itself consistent with a heuristic rather than a
  defect.

## Eval measurement health

A turn that exhausts its generation budget inside the model's analysis channel
has no reply in it, and the reasoning parser returns it unsplit — so the harness
grades reasoning. That is a measurement defect, and this stage hit it twice:

* **`aime25`**: at 32768 tokens, 4 of 30 problems returned nothing at all.
  Replayed with headroom the same problems terminate on their own (doc 9 at
  23589 tokens, with the correct answer), so the budget was raised to 98304 —
  96K of the 131072 context, following the gpt-oss-120b `aime25` entry in the
  same file, which uses 120K. The graded run has **0 empty responses**.
* **`ifeval`**: at 8192 tokens, 3 of 541 turns were graded on their analysis.
  Raising the budget to 32768 — the same fix, for the same reason — reduced that
  to **1 of 541**. That last one is not a budget problem: asked for *"a short
  article … 200 words or less … make sure the letter c appears at least 60
  times"*, the model's analysis degenerates into a literal run of `c`
  characters, 261,038 of them, until the cap. It scores `False` on the
  word-count instruction (the letter-frequency one passed), which is what a run
  of `c`s would have scored as a reply too. It is a temperature-1.0 excursion,
  not a deterministic defect: **the same document answered normally in both
  8192-budget runs**, 527- and 721-character replies, strict `True` each time.

So the graded `ifeval` 94.45 is a **floor**, understated by at most 1/541 = 0.18
points, in the conservative direction. `evals/ifeval_sample_health.json` and
`evals/aime25_sample_health.json` record, per document, the response length, the
score and whether the turn reached the visible channel — a few KB, no model text
— so this is checkable from the evidence tree rather than only from the
uncopied `samples_*.jsonl` where both defects were originally hiding.

**Follow-up:** the shared degenerate-output gate scans qualitative artifacts, not
eval samples. Pointing it at these two files would catch this class
automatically instead of relying on someone reading the samples.

## Evals: why `ifeval` and `aime25`

`meta_ifeval` and `meta_gpqa_cot`, the usual mandatory text-LLM gates, are
Llama-family-only — llama-cookbook builds their datasets from
`<hf_model_repo>-evals`, and `meta-models/Muse-Glimmer-30B-evals` returns 404
with a valid token while `meta-llama/Llama-3.1-8B-Instruct-evals` returns 200.
Their prompts are also pre-rendered in Llama 3's chat format, so pointing them at
this tokenizer would break `$qualitative-check`'s prompt-format rule.

`gpqa_diamond_cot_zeroshot` was the first choice for the reasoning gate and could
not run either: every lm-eval GPQA task reads the **gated** `Idavidrein/gpqa`,
which this host's HF account has not been granted
([`logs/gpqa_dataset_gated.log`](logs/gpqa_dataset_gated.log)). Accepting a
third-party dataset licence on the user's account is outside this stage's
authority. **Follow-up: grant GPQA access and add the row.**

So the gates are `ifeval` (instruction following) and `aime25` (zero-shot
chain-of-thought), the model-appropriate equivalents that every non-Llama entry
in TTI's eval catalog uses. Both references are **vendor-published model-card
scores, not measured GPU references** — no Tenstorrent control run exists for
this checkpoint — and the reasoning for each substitution and tolerance is
recorded inline in the eval config and in `RUN_NOTES.md`.

`aime25`'s generation budget was raised from 32768 to 98304 tokens on evidence,
not on preference: at 32768, four of the thirty problems ran past the cap and
returned **no visible channel at all**, scoring 0 on a truncated turn rather than
on an answer. Replayed with headroom the same problems terminate on their own —
doc 9 finishes at 23589 tokens with the correct answer
([`smoke/aime25_budget_probe.json`](smoke/aime25_budget_probe.json)) — and the
final run has zero empty responses. 96K of the 131072 context follows the
gpt-oss-120b `aime25` entry in the same file, which uses 120K of the same.

## Artifacts

| what | path |
|---|---|
| final release report | `report/report_id_muse-glimmer-30b-autoport_Muse-Glimmer-30B_p300x2_2026-08-16_04-04-24.md` |
| release report data | `report/report_data_..._2026-08-16_04-04-24.json` |
| run spec TTI wrote (implementation-path proof) | `run_spec/runtime_model_spec_2026-08-16_02-39-22_*.json` |
| spec handed to `run.py` | `run_spec/muse_glimmer_30b_autoport_release.json` |
| eval results, per task | `evals/results_*.json` |
| benchmark JSON, 18 sweep points | `benchmarks/` (per-token `itls` and per-request `generated_texts` trimmed; every metric kept, drop recorded in-file) |
| run log | `logs/run_*_release_*.log` |
| server excerpt | `logs/server_excerpt.log` (raw `server.log` is gitignored) |
| `$qualitative-check` suite | `qualitative/`, `qualitative_runner/` |
| prompt-format decisions | `qualitative/prompt_format_tti_release.json`, `qualitative/qualitative_prompt_format.json` |
| reasoning-parser control | `smoke/reasoning_control_unparsed.json`, `smoke/reasoning_parsed.json` |
| non-aligned prompt lengths | `non_aligned_probe.json` |
| per-document eval health (length, score, whether the turn reached the visible channel) | `evals/ifeval_sample_health.json`, `evals/aime25_sample_health.json` |
| why `meta_*` evals cannot run | `logs/meta_evals_dataset_404.json`, `logs/gpqa_dataset_gated.log` |
| seeding bug | `AUTOFIX_seeding.md`, `seeding/` (evidence JSON, two instrumented seed traces, and all nine op-level / server-level probe scripts) |
| presence-penalty row | `AUTOFIX_presence_penalty.md`, `presence_penalty/`, `presence_penalty_row.json`, `presence_penalty_repeats.json`, `presence_penalty_greedy_equivalence.json`, `presence_penalty_host_control.json`, `presence_penalty_control.json` |
| tt-inference-server local edits | `tti_local_edits/tt_inference_server_local_edits.patch` |
| commands, versions, cleanup, classifications | `RUN_NOTES.md` |

## Limitations and follow-ups

1. **This is a text-only release of a multimodal checkpoint.**
   `meta-models/Muse-Glimmer-30B` is `MuseGlimmerForConditionalGeneration` — its
   HF config carries a `vision_config`, an `image_token_id` and a
   `video_token_id`, and the model card advertises interleaved text-and-image
   input through a ~1.8B ViT-G/14 perception encoder. **The autoport implements
   the text stack only; no stage of this bring-up ported the vision tower.** The
   release spec sets `supported_modalities: ["text"]`, which is what stops TTI
   generating the image benchmark sweep. Image and video input are unsupported
   and every number in this report is a text-only number.
2. **`Idavidrein/gpqa` is gated** for this account, so there is no GPQA row.
   `aime25` covers the same reasoning dimension; add GPQA once access is granted.
3. **Both eval references are vendor-published, not measured.** No Tenstorrent
   GPU control run exists for this checkpoint and a 30B HF reference run was not
   feasible here. The rows are labelled as such in the report.
4. **`ttnn.manual_seed`'s per-core state has no way to say "this must survive".**
   The seeding fix is an ordering constraint; a future op inserted between the
   seed call and `ttnn.sampling` will silently re-break it. Upstream ask, plus an
   op-level regression test.
5. **Penalty term ordering differs from vLLM's** — device applies presence →
   frequency → repetition, vLLM applies repetition → frequency → presence.
   Unobservable in this suite (frequency 0 and repetition 1.0 are bit-exact
   identities) but it will matter when `repetition_penalty != 1.0` is combined
   with a presence or frequency penalty. `models/common/sampling/tt_penalties.py`
   has no test file today.
6. **Streaming reasoning-parser deltas cannot offer the non-streaming
   guarantee**: a turn cut off inside the analysis channel streams as reasoning
   with no content. Every eval, benchmark and conformance path here runs
   non-streaming.
7. **Complete/target performance tiers fail**, at 38 % of the memory-side
   roofline. That is a first-bring-up result and the `FUNCTIONAL` status says so;
   it is not a regression against anything this stage measured.
