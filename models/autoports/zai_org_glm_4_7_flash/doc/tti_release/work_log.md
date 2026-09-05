# TTI release stage work log: zai-org/GLM-4.7-Flash autoport

Stage 11, `$tti-release` + `$tt-device-usage`, one Blackhole chip (1x1 mesh,
board series p300c, host `tt-quietbox`), 2026-09-04. Results, classifications
and commands live in `RUN_NOTES.md`; this log records the order of work, the
wrong turns, and the decisions.

## TR-000 Release integration: making TTI evaluate the autoport, not tt-transformers

The checkout at `/home/stisi/tt-inference-server` (VERSION 0.20.0,
`v0.10.0-1141-gbc296ab54`) has no GLM-4.7-Flash entry and no notion of an
autoport implementation. Its `--model` argument is a fixed choice list built from
the catalog, and `EVAL_CONFIGS` is the intersection of the eval list with
`MODEL_SPECS`, so a runtime spec JSON alone would not have produced eval
configs. The wiring therefore went into the dev catalog:

* a new `autoport_glm47_flash` `ImplSpec` with
  `code_path = models/autoports/zai_org_glm_4_7_flash`. This is the load-bearing
  piece: `get_model_id()` keys on `impl_name`, and reusing `tt_transformers_impl`
  would have produced a report that scored a different implementation of the
  same checkpoint;
* a dev spec for `zai-org/GLM-4.7-Flash` on `P150` at `max_context: 202752`,
  `max_concurrency: 32`;
* eval, benchmark-target and spec-test entries (TR-002, TR-003, TR-004).

`--tt-device p150` is the right device even though the boards report as p300c:
the autoport opens a 1x1 mesh on one chip, and TTI's `P150` is its
single-Blackhole-chip device. `--dev-mode` is mandatory. Committed to
tt-inference-server as `66b9f5a398a5b48c09efcf67c8713a4371402345`.

**Wrong turn, recorded because it costs 20 minutes of silence:** `run.py --help`
documents `--server-url` "together with `--service-port` when not using
`--docker-server` or `--local-server`", which is exactly this topology. Passing
it sets `remote_server=True`, and `ServerConnection.url_with_port` drops the port
for remote connections, so the benchmark client polls port 80 and waits out its
1200 s health timeout with no log line. Omitting the flag resolves the same
localhost default with the port attached and works. Left as a recorded TTI sharp
edge rather than patched, since the no-flag path is correct.

## TR-001 Prefill OOM above about 40k tokens through the serving path

The first release run died at benchmark sweep point 21 (`isl=65536`), taking the
server, the last three sweep points and the whole `spec_tests` child with it.
Root cause was an unreserved whole-prompt prefill activation pair in
`get_max_tokens_all_users`; fixed with `$autofix` and verified on hardware up to
a 202751-token prompt. Full report in `AUTOFIX_prefill_dram.md`, evidence in
`autofix_prefill_dram.json`, the before side in
`logs/prefill_oom_before_fix.log`, and the persistent-allocation change recorded
in `../context_contract.json` under `tti_release`.

This is the stage's most important finding: before it, the advertised 202752
context was not deliverable through vLLM serving even though the model harness
had proven a 202751-token prefill, because the harness never had a 471k-token KV
pool resident at the same time.

## TR-002 Eval task selection for a thinking model with no Meta eval datasets

`meta_ifeval` and `meta_gpqa_cot`, which the stage goal names as the mandatory
text-LLM gates, cannot run for this checkpoint. `WorkflowVenvType.EVALS_META`
builds its datasets by running llama-cookbook's `prepare_meta_eval.py` against
`f"{hf_model_repo}-evals"`, and Meta publishes those joined parquet datasets only
for the Llama families. There is no `zai-org/GLM-4.7-Flash-evals`. Only 3 of the
catalog's models use the `meta_*` tasks; every non-Llama model uses `ifeval` and
`gpqa_diamond_cot_zeroshot`, which are the direct standard equivalents
(instruction following, zero-shot CoT reasoning), and those are what this model
now runs.

Both tasks run on the chat API on purpose. GLM-4.7-Flash is thinking-by-default:
`add_generation_prompt` emits `<|assistant|><think>`, so a raw completion returns
the reasoning trace and any exact-match or format scorer sees the trace instead
of the answer. The release server therefore runs with `--reasoning-parser glm47`,
which puts the trace in `reasoning_content` and leaves the post-`</think>` answer
in `message.content`, which is what lm-eval scores. The parser is API-server
post-processing; measured decode TPOT is unchanged.

Output budgets are the other half of that decision. A first release attempt at
`max_gen_toks=32768` left 2 of 10 GPQA items with an empty answer, both still
inside their reasoning block at the cap, and lm-eval scored them wrong: 60.0%.
Doubling to 65536 removed the truncation entirely (0 of 10 empty) and the score
moved to 70.0%. The model card's own recipe is 131072 output tokens, which is not
runnable here: a single straggler would hold an eval wave for over an hour at
~34 t/s/u on one chip.

## TR-003 Benchmark targets

`build_benchmark_config` derives the whole 23-point sweep from the spec, so no
benchmark table entry was needed. Perf targets are a separate file, and every
model in the catalog defines targets for exactly one shape
(128/128/concurrency-1). GLM-4.7-Flash now has that entry: the `theoretical`
block is the autoport's own committed DRAM-bandwidth roofline (3.9 ms/token,
256.4 t/s/u, from `doc/optimized_vllm/perf_summary.json`), and the `measured`
block is its recorded serving headline, so the `target` tier acts as a regression
check against the optimized-vLLM result rather than against a roofline no
bring-up reaches. Result and the one caveat (the TTFT half of the target was
measured through a different client) are in `RUN_NOTES.md`.

## TR-004 Spec tests: 14 failed / 8 passed, then 22 passed / 0 failed

`--workflow spec_tests` first reported "No spec test suites match" because the
model was not registered in `test_module/server_tests_config.json` /
`test_module/test_suites/llm.json`. Registered for
`VLLMParamConformanceTest` on p150.

Run as-is the suite was 14 failed / 8 passed, and every failure had the same
shape: the suite chooses 32 to 64 token budgets because it tests API parameter
semantics, the model spends them inside its reasoning block, `message.content`
comes back `None` behind the reasoning parser, and assertions crash on `None`
(`TypeError`, `AttributeError: 'NoneType' object has no attribute 'lower'`)
rather than evaluating the parameter under test.

That this is structural rather than an autoport defect was proven directly
before any code was written: the coherence prompt at `max_tokens=32` returns
`finish_reason=length, content=None` with thinking on, and returns
`"The quick brown fox jumps over the lazy dog."` in 11 completion tokens with
`chat_template_kwargs.enable_thinking=false`. So the fix is a per-model request
default, added as a `targets.request_defaults` channel in the suite config plus
a `--extra-request-json` option on the shared pytest fixtures, merged only into
keys a test did not set. Blast radius is limited to models that configure it.
Result: 22 passed / 0 failed, including `test_non_uniform_seeding` at 32
concurrent seeded requests.

## TR-005 Two ifeval items that come back empty at concurrency 16

Not length truncation. Replayed alone through `/v1/completions` with the chat
template applied and the same seed, both closed their reasoning block at 7440 and
1482 completion tokens; replayed through `/v1/chat/completions` at concurrency 2
with the same seed, both returned non-empty content at the identical token
counts. The only difference from the eval run is concurrency. Consistent with
upstream `tenstorrent/tt-metal#55408`, which the optimized-vLLM stage established
across five measurements and could not narrow. This stage did not narrow it
either; it adds one datapoint. Artifacts: `evals/empty_answers_8192.json`,
`evals/empty_answers_chat_lowconc.json`.

## TR-006 Greedy non-termination on an open-ended creative prompt

`run_vllm_server --stages qualitative` cannot run against a server with a
reasoning parser: it hardcodes `max_tokens=256` and raises
`Chat completion returned no text content`. The shared six-prompt suite was
therefore run with the same prompts, prompt mode and greedy/sampled pair at a
budget the model can finish in (`qualitative/`).

Eleven of twelve completions are coherent. The twelfth, the greedy arm of the
story-completion prompt, returns nothing at 4096 and again at 16384 output
tokens. Characterised rather than dismissed: the raw greedy trace has
`adjacent_duplication` 0.0000 (so not the stale token/position feedback class the
degenerate-output checker exists to catch) but `trigram_loop_fraction` 0.7955 and
a single trigram repeated 29 times, i.e. phrase-level looping while re-drafting
the same paragraph. The same prompt under the model card's recommended sampling
settings terminates in 1713 tokens with a 0.0846 loop fraction. Full ledger entry
in `RUN_NOTES.md`; the missing piece is an HF reference control, which needs a
GPU host.

## Order of work

1. Read `$tti-release`, `$tt-device-usage`, `$stage-review`, `$autofix`,
   `$qualitative-check`; surveyed the checkout's own `run.py --help`.
2. Device health, then started the autoport server (TR-000 wiring in parallel).
3. Health, `/v1/models`, one OpenAI-compatible chat request, non-aligned
   prompt-length probe.
4. Tiny no-Docker TTI benchmark smoke with `--disable-trace-capture`: exit 0,
   `completed 8 / failed 0`, run spec verified.
5. Standalone `spec_tests` (TR-004) and a smoke-limited `evals` run to de-risk
   the two harness paths before spending hours.
6. Release run 1 with `--limit-samples-mode ci-nightly`: died at sweep point 21
   (TR-001).
7. `$autofix` on TR-001, verified on hardware to a 202751-token prompt.
8. Empty-answer probes (TR-005) before restarting the long run.
9. Release run 2: exit 0, acceptance PASS, all 23 sweep points completed.
10. Qualitative suite and greedy-trace characterisation (TR-006).
11. Copy-back, `RUN_NOTES.md`, context-contract `tti_release` section, gates,
    `$stage-review`, commits.

## Commits

* tt-inference-server `66b9f5a398a5b48c09efcf67c8713a4371402345`: release wiring
  (TR-000, TR-002, TR-003, TR-004).
* tt-metal: see the SHA recorded at the end of this file after the stage commit.
