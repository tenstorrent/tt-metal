# AutoFix Report

## Starting Evidence

- Source diagnosis: `doc/tti_release/AUTODEBUG.md`.
- Failure log: `doc/tti_release/logs/tti_release_ci_nightly.log` reported that
  Qwen3.6-27B had no standard eval tasks, so release ran zero standard evals.
- Failing command: `python run.py --model Qwen3.6-27B
  --runtime-model-spec-json .../autoport_release_spec.json --tt-device p300x2
  --workflow release --service-port 8000 --server-url http://127.0.0.1
  --no-auth --skip-system-sw-validation --limit-samples-mode ci-nightly`.

## Hypothesis Experiments

- Hypothesis: Qwen3.6-27B is under-onboarded in the eval catalog: its only
  active task is agentic, so the standard release selector returns no tasks.
  Experiment: inspected `reference_config/evals/eval_config.py`,
  `llm_module/eval_configs.py`, and `workflow_module/workflows.py` at TTI commit
  `e26e723bf0266cde85f674e381fbee10068ae0ec`.
  Result: the Qwen entry contained only `terminal_bench_2` with
  `EVALS_AGENTIC`; standard selection admits only common/Meta/vision tasks and
  release converts an empty task result to a successful no-op.
  Verdict: verified.
  Fix: added active `EVALS_META` `meta_ifeval` and `meta_gpqa_cot` tasks to the
  existing Qwen config. Both use the established preformatted-prompt contract
  (`include_path="work_dir"`, `apply_chat_template=False`) and explicit
  `CI_NIGHTLY: 0.05` limits. No runtime-spec, dispatch, server, or context
  change was made.
  Verification: focused command-construction test proves both tasks survive
  selection, use `http://127.0.0.1:8000/v1/completions`, omit a second chat
  template application, propagate `--limit 0.05`, and inherit
  `max_length=262144`.

- Hypothesis: an authoritative existing Qwen-specific score can safely supply
  the mandatory accuracy thresholds.
  Experiment: searched all local TTI history/branches and current configs for
  Qwen3.6 `meta_ifeval` / `meta_gpqa_cot` references, then compared the Qwen
  model-card metrics with the exact TTI task recipes.
  Result: no matching TTI or CI-subset references exist. The model card has no
  IFEval score and its GPQA Diamond score does not identify the TTI
  `meta_gpqa_cot` strict-match recipe, so it is not a defensible threshold.
  Verdict: refuted.
  Fix: no Llama/Meta score or mismatched model-card score was copied. The task
  rows are emitted and measurable, but their accuracy checks remain `N/A`
  until matched Qwen GPU/control baselines establish full-set or CI-subset
  references.

## Verification

```text
pytest -q tests/llm_module/test_eval_command.py \
  tests/workflows/test_workflow_dispatch_routing.py \
  tests/workflow_module/test_release_workflow.py
126 passed in 0.19s

git diff --check
PASS
```

No TT hardware or live server was contacted.

## Final Status

- Eval wiring failure: fixed with source/unit evidence.
- Mandatory score thresholds: still unresolved. A matched Qwen GPU/control run
  of the same Meta tasks and CI-nightly subsets is required before these rows
  can be accuracy-pass-classified; absent references intentionally produce
  `N/A`, which is not a release-readiness pass.
- Remaining risk: the next release run is now expected to execute both tasks,
  but runtime correctness and scores require the stage owner's external-server
  rerun.

## Runtime Follow-up: Public Qwen-Compatible Tasks

- Hypothesis: the Llama-cookbook `EVALS_META` tasks can prepare their datasets
  for Qwen while retaining the mandatory release row names.
  Experiment: reran the release client; see
  `logs/tti_release_ci_nightly_fixed.log:610-707`.
  Result: `prepare_meta_eval.py` rejects Qwen because the proprietary dataset
  validator accepts only Llama 3.1/3.2 collections. The optional preparation
  failure was ignored, after which lm-eval could not register either Meta task.
  Verdict: runtime-refuted. The `EVALS_META` change was removed.

- Hypothesis: public lm-eval IFEval and GPQA CoT tasks can provide the same
  mandatory release row contracts without using or spoofing Meta datasets.
  Experiment: inspected the installed `EVALS_COMMON` task YAMLs and loaded its
  `TaskManager` without contacting a model endpoint. Both `ifeval` and
  `gpqa_diamond_cot_zeroshot` registered successfully. The former uses
  `google/IFEval`; the latter uses `Idavidrein/gpqa`, a step-by-step boxed-answer
  prompt, and `exact_match,flexible-extract`.
  Verdict: verified.
  Fix: added optional `EvalTask.eval_task_name` to separate the public lm-eval
  registry name from the stable TTI report row. Qwen now runs public `ifeval`
  and `gpqa_diamond_cot_zeroshot` through `EVALS_COMMON`/the OpenAI chat API,
  while emitted report rows remain `meta_ifeval` and `meta_gpqa_cot`. The server
  applies Qwen's native chat template exactly once. CI limits and 262144 context
  remain unchanged. Aliased blocks also record the public `eval_task_name` in
  both targets and data so report JSON proves which harness task supplied each
  mandatory row.

Focused registration smoke:

```text
.workflow_venvs/.venv_evals_common/bin/python - <<'PY'
from lm_eval.tasks import TaskManager
manager = TaskManager()
for name in ('ifeval', 'gpqa_diamond_cot_zeroshot'):
    assert name in manager.all_tasks
    print('REGISTERED', name)
PY
REGISTERED ifeval
REGISTERED gpqa_diamond_cot_zeroshot
```

Updated source-only verification:

```text
pytest -q tests/llm_module/test_eval_command.py \
  tests/test_module/llm_tests/test_llm_eval_tests.py \
  tests/workflows/test_workflow_dispatch_routing.py \
  tests/workflow_module/test_release_workflow.py
149 passed, 1 warning in 0.26s
```

Exact rerun guidance: reuse the prior release command and external server after
the TTI client checkout includes this patch. The expected eval commands now use
`--tasks ifeval` and `--tasks gpqa_diamond_cot_zeroshot`,
`--model local-chat-completions`, `/v1/chat/completions`, `--limit 0.05`, and
`max_length=262144`. Confirm the report targets remain `meta_ifeval` and
`meta_gpqa_cot`. Accuracy references remain the previously documented,
separate unresolved control requirement.

## Runtime Follow-up: Non-aligned Long Prefill

- Fresh diagnosis: `AUTODEBUG_PREFILL_ALIGNMENT.md`.
- Hypothesis: the fixed 32,768-token stack stride is incompatible with vLLM's
  effective hybrid-cache page size.
  Evidence: `readiness_vllm/server.log:20-21` records vLLM enlarging and
  equalizing the attention page to 800 tokens. The valid rendered prompt has
  32,780 tokens; its second model stack chunk starts at 32,768, while
  `32768 % 800 == 768`. The decoder correctly rejects that cache write.
  Experiment: a host-only planner probe selected the largest stride no greater
  than 32,768 divisible by both the bound KV page and the 32-token linear scan
  quantum. It returns 32,768 for pages 64/1024 and 32,000 for page 800. Boundary
  simulation proved every nonzero start is aligned for prompts around 32,000,
  32,768, 32,780, 33,600, and 64,000.
  Verdict: verified.
  Fix: `_prefill_forward_streaming` now uses the aligned runtime stride rather
  than the class constant. `bind_kv_cache` also verifies every externally bound
  full-attention cache agrees with the model's effective page size. The decoder
  guard remains in place. Logical prompts, context, page tables, and output
  slicing were not padded, shortened, or aligned by the harness.

Verification:

```text
pytest -q \
  models/autoports/qwen_qwen3_6_27b/tests/test_vllm_adapter_contract.py \
  models/autoports/qwen_qwen3_6_27b/tests/test_linear_prefill_long_state.py
14 passed, 2 warnings in 11.39s
```

Exact server verification uses the same optimized-vLLM launch and then:

```text
/home/mvasiljevic/tt-metal/python_env/bin/vllm bench serve \
  --backend openai-chat --model Qwen/Qwen3.6-27B \
  --base-url http://localhost:8000 --endpoint /v1/chat/completions \
  --dataset-name random --random-input-len 32768 --random-output-len 128 \
  --num-prompts 1 --max-concurrency 1 --ignore-eos --temperature 0.0
```

Runtime pass criteria: rendered length remains 32,780, all 128 requested output
tokens complete, no page-boundary exception occurs, and a subsequent short
health/API request succeeds. Hardware rerun remains required to close this
model-path fix beyond the source/unit proof.

## Runtime Follow-up: Public Chat Payload Construction

- Hypothesis: `apply_chat_template=False` lets the chat server perform the only
  Qwen formatting while lm-eval supplies a valid message payload.
  Experiment: `logs/tti_release_ci_nightly_final2.log:108-230` shows both public
  tasks register and select their intended CI counts, then
  `LocalChatCompletion._create_payload` rejects the task string because it
  requires `list[dict]` messages.
  Verdict: runtime-refuted.

- Hypothesis: in pinned lm-eval's non-tokenized `local-chat-completions` path,
  `--apply_chat_template` structures task text into messages without rendering
  the model's token template client-side.
  Experiment: inspected pinned `api_models.py`: `apply_chat_template` returns a
  `JsonChatStr` containing JSON message dictionaries when
  `tokenized_requests=False`; `create_message` decodes that wrapper to
  `list[dict]`. `openai_completions.py::_create_payload` places that list in the
  OpenAI `messages` field. A client-free payload probe instantiated this exact
  pinned adapter path and produced:

  ```text
  CHAT_PAYLOAD_OK [{'role': 'user', 'content': 'Reply briefly.', 'type': 'text'}]
  ```

  The payload had `messages`, no rendered `prompt`; therefore the OpenAI server
  remains the only component that renders Qwen's token template.
  Verdict: verified.
  Fix: set `apply_chat_template=True` on both Qwen public-task aliases. The
  generated commands now include `--apply_chat_template`; task, CI limit,
  external URL, and 262144 context wiring are unchanged.

Focused verification:

```text
pytest -q tests/llm_module/test_eval_command.py \
  tests/test_module/llm_tests/test_llm_eval_tests.py
27 passed, 1 warning in 0.16s
```

Exact rerun: use the same final2 release command/spec and healthy external
server. Confirm both eval commands contain `--model local-chat-completions`,
`--apply_chat_template`, `/v1/chat/completions`, `--limit 0.05`, and
`max_length=262144`; neither task may raise the `messages must be list[dict]`
assertion. The copied result blocks must retain stable rows `meta_ifeval` and
`meta_gpqa_cot` plus public `eval_task_name` provenance.

## Runtime Follow-up: Benchmark Selection and Agentic Readiness

- Hypothesis: the custom runtime spec cannot generate the standard benchmark
  matrix because its model ID is absent from the built-in catalog.
  Experiment: loaded `autoport_release_spec.json` through `ModelSpec.from_json`
  and called the current benchmark builder with the inherited environment
  varied.
  Result: runtime-spec resolution is correct. With
  `ONLY_BENCHMARK_TARGETS` unset it produces the ten standard text shapes:
  `(128,128)`, `(128,1024)`, and OSL 128 for ISL 1024, 2048, 4096, 8192,
  16384, 32768, 65536, and 131072, all concurrency 1. With any non-empty value
  (including `0`) the builder intentionally suppresses generated sweeps; the
  release spec has no `perf_reference`, so zero configurations remain.
  Verdict: runtime-spec hypothesis refuted; inherited target-only environment
  verified as the cause.
  Fix: command/environment correction only. Run release with
  `env -u ONLY_BENCHMARK_TARGETS`. Do not copy the 8/8 smoke reference or invent
  targets: the built-in Qwen3.6 spec also declares no performance references
  and is explicitly experimental. A focused regression proves a runtime-style
  autoport spec with empty references and CI-nightly selects the exact ten-shape
  standard matrix and preserves the 262144 context.

- Hypothesis: the agentic child was not part of Stage 11, or its hang was
  authentication-related.
  Experiment: inspected release child selection and no-auth setup.
  Result: Qwen config contains `terminal_bench_2`, so release automatically
  appends the agentic child. The no-auth placeholder `OPENAI_API_KEY=EMPTY` is
  valid. The actual hang was readiness constructing
  `RemoteOpenAIController(base_url=ctx.server_url)`: the host-only URL dropped
  service port 8000 and polled port 80.
  Verdict: URL hypothesis verified.
  Fix: agentic readiness now uses canonical `ctx.base_url`, matching standard
  eval/benchmark routing. Tests cover host-only URL plus service port and an
  explicitly ported URL without duplication.

Focused verification:

```text
pytest -q tests/llm_module/test_benchmark_configs.py \
  tests/test_module/llm_tests/test_agentic_eval_tests.py
66 passed in 0.20s
```

Exact rerun prefix:

```text
env -u ONLY_BENCHMARK_TARGETS python run.py \
  --model Qwen3.6-27B \
  --runtime-model-spec-json /home/mvasiljevic/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/tti_release/autoport_release_spec.json \
  --tt-device p300x2 --workflow release --service-port 8000 \
  --server-url http://127.0.0.1 --no-auth \
  --skip-system-sw-validation --limit-samples-mode ci-nightly
```

Confirm the run spec still preserves external/no-Docker mode and 262144
context; benchmarks emit all ten shapes unchanged; agentic readiness polls
`http://127.0.0.1:8000/v1/models`. Terminal-Bench remains a configured release
child, not an accidental auth failure.

## Final4 Follow-up: Terminal-Bench Docker Infrastructure

- Evidence: final4 produced both mandatory eval blocks and all ten standard
  benchmark blocks with zero request failures. The sole blocker is the emitted
  `terminal_bench_2` failure block after Harbor reported `Docker is not
  installed or not on PATH` in the reservation container.
- Official mechanism audit: task-scoped `known_issues` can mask an emitted
  `EVALS` block during acceptance/report regeneration, and final4 preserved a
  suitable failed block. It does not itself stop Harbor from running. More
  importantly, this release contract requires a current linked issue proving a
  row-specific waiver; no current issue was found for reservation-container
  Docker absence. The existing Qwen agentic integration issue #3359 is closed
  and records reference results, not this infrastructure limitation. Applying
  it as a Docker waiver would be unsupported.
- Exclusion audit: current release automatically appends agentic when Qwen's
  configured `terminal_bench_2` is present, including its five-task CI-nightly
  subset. There is no official skip/exclusion field that both avoids execution
  and emits issue-backed SKIP provenance. A locally invented skip flag or task
  deletion would create missing coverage and is not a defensible clean pass.
- Report reuse: the saved final4 blocks can be regenerated cheaply with amended
  `known_issues`; no eval or benchmark rerun is technically necessary. Do this
  only after a current issue explicitly approves the Docker-infrastructure
  waiver. Without that issue, regeneration would produce a mechanically
  passing but invalid report.

Defensible fallback: run only the Harbor/agentic child from the physical
loudbox host against the existing reservation server, then regenerate the
combined report from final4 blocks plus the successful agentic block. Before
launching Harbor:

1. Advertise a non-loopback reservation-server address; `127.0.0.1` from the
   physical host or a Harbor task container is wrong. The reservation container
   currently reports `172.17.0.2`, but resolve/verify it from the physical host
   rather than assuming it is stable.
2. From the physical host, curl `<reservation-address>:8000/v1/models` and one
   tiny no-auth chat request.
3. From a disposable Docker container on the same network Harbor will use,
   repeat `/v1/models`. Do not proceed until both probes pass.
4. Run the existing five CI-nightly task names using the generated final4
   Harbor config, changing only `api_base` from loopback to the verified
   host-reachable address. Harbor is client-side and must not mount or access TT
   devices.
5. Preserve/copy its small result into final4's `agentic/` evidence, rebuild the
   `terminal_bench_2` block, and regenerate reports only. Do not rerun the
   completed mandatory evals or ten benchmarks.

Context guardrail: the current Terminal-Bench config advertises 256K input plus
80K output, which exceeds the real 262144 total context. Before the fallback,
the agentic adapter must ensure each actual prompt plus requested completion is
at most 262144. Only mathematically invalid requests may be reduced/rejected;
the model context must not be capped.

### Terminal-Bench token-budget repair

The invalid request budget is fixed at its configuration boundary. Completion
capacity remains 80K and the model context remains the full 262144 tokens;
maximum input is now 176K, so `176 * 1024 + 80 * 1024 == 262144` exactly.
`llm_kwargs.max_tokens` remains identical to `model_info.max_output_tokens`.
No benchmark/eval request was aligned or shortened below this mathematical
constraint.

```text
pytest -q tests/test_module/llm_tests/test_agentic_eval_tests.py
35 passed in 0.18s

git diff --check -- reference_config/evals/eval_config.py \
  tests/test_module/llm_tests/test_agentic_eval_tests.py
PASS
```
