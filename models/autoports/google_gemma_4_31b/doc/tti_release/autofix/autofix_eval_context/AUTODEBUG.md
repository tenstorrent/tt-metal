# AutoDebug: Gemma 4 release eval context contract

## Verdict

The observed `Using max length 2048 - 1` is caused by TTI omitting lm-eval's
`max_length` model argument. The smallest correct context fix is to derive and
pass `max_length=113280` from
`model_spec.device_model_spec.max_context` for both mandatory commands.

`max_length=113280` is correct; **do not pass 113281**. lm-eval 0.4.4 stores
`self.max_length = max_length - 1` because its loglikelihood request always asks
the server for one additional output token. Thus 113279 input tokens plus that
one generated token exactly fits the 113280 server envelope. Passing 113281
would permit a mathematically invalid 113280-input-plus-one-output request.

For these two `generate_until` tasks, the 2048 default does **not** currently
truncate prompts. It is nevertheless wrong release metadata and a latent limit
for loglikelihood workloads, so it must be fixed before rerunning release.

There is a second, accuracy-critical discrepancy: the allegedly raw Gemma base
completion eval is feeding Llama-3.1-Instruct-formatted prompts containing
literal Llama control-token spellings. That contradicts the base/no-chat
contract and should be fixed in the same repair before a release score is
trusted.

## Evidence-ranked findings

### 1. High confidence: TTI drops the model spec's 113280 context when building both lm-eval commands

- `tt-inference-server-v2/llm_module/eval_command.py:229-239` reads
  `device_model_spec.max_context`, but uses it only for the heuristic output
  clamp.
- `eval_command.py:241-254` builds optional model args without adding
  `max_length`.
- `eval_command.py:280-282,326-344` serializes only `task.model_kwargs` and
  those optional args into `--model_args`.
- `evals/eval_config.py:3714-3755` defines both base Gemma tasks without a
  `max_length` model kwarg.
- The exact release log at `tti_release_pass.log:68` therefore contains no
  `max_length`; at line 77 lm-eval reports `Using max length 2048 - 1`.
- Installed lm-eval 0.4.4 declares `max_length=2048` at
  `.workflow_venvs/.venv_evals_meta/lib/python3.10/site-packages/lm_eval/models/api_models.py:54-80`
  and stores `max_length - 1` at lines 107-109.
- A local constructor probe proved the exact mapping: 2048 -> 2047,
  113280 -> 113279, and 113281 -> 113280.

This finding explains the log exactly. The intervention boundary is the TTI
command builder, not the model/server.

### 2. High confidence: generation prompts are not silently truncated, with either tokenized_requests polarity

- In lm-eval's `TemplateAPI.generate_until`, lines 526-603 of
  `api_models.py`, `self.max_length` is never consulted.
- With `tokenized_requests=True` (the current command's default), lines 537-540
  tokenize the full request. `tok_encode` has `truncation=False` by default at
  lines 272-291 and no `left_truncate_len` is supplied.
- With `tokenized_requests=False`, lines 541-559 skip tokenization and send the
  original context strings directly. There is no length slice or truncation.
- The constructor's `truncate=False` default is stored at lines 67 and 104 but
  is not used by the generation path.
- `self.max_length` does affect loglikelihood and rolling-loglikelihood paths:
  `api_models.py:406-423` slices loglikelihood input and lines 605-620 size
  rolling windows. This is why the omitted context still needs repair even
  though the current generation prompts were intact.

The OpenAI-compatible server remains the exact admission boundary for
generation. It should accept a request when
`prompt_token_count + requested_max_tokens <= 113280` and reject it otherwise.
No arbitrary alignment or prompt-reserve clamp is needed.

### 3. High confidence: the task YAML output budgets are retained, intentional task semantics, and safely fit the current prompts

- Prepared `work_dir_gemma-4-31B/ifeval/ifeval.yaml:10-14` requests at most
  1280 generated tokens.
- Prepared `work_dir_gemma-4-31B/gpqa_cot/gpqa_0shot_cot.yaml:16-20`
  requests at most 2048 generated tokens.
- TTI's CLI contributes only `stream=False,seed=42`. lm-eval merges these into
  the YAML dictionary rather than replacing it (`evaluator.py:170-177,246-250`
  and `task.py:630-643`), so the YAML `max_gen_toks` values survive.
- `LocalCompletionsAPI._create_payload` converts `max_gen_toks` to the OpenAI
  `max_tokens` request field (`openai_completions.py:22-46`).
- Offline measurement with the exact cached Gemma tokenizer, including the BOS
  used by the command, found:

  | Task | Samples | Maximum prompt | YAML output budget | Maximum total |
  |---|---:|---:|---:|---:|
  | `meta_ifeval` | 541 | 387 | 1280 | 1667 |
  | `meta_gpqa_cot` | 448 | 2551 | 2048 | 4599 |

  Both totals are far below 113280.

These completion budgets bound each benchmark's intended answer, not the
model's context window. Raising either one to 113280 would make every non-empty
prompt mathematically invalid and would change the benchmark. They should not
be removed or inflated merely to make the command display the context contract.

### 4. High confidence, accuracy-critical: prepared prompts violate the claimed raw Gemma base prompt contract

- `workflows/workflow_venvs.py:27-35` maps `google/gemma-4-31B` preparation to
  `meta-llama/Llama-3.1-8B-Instruct-evals`.
- The cookbook preparation code renames that dataset's
  `input_final_prompts` to IFEval's `prompt`
  (`prepare_meta_eval.py:35-76`). The prepared IFEval YAML then uses that
  column directly (`ifeval.yaml:8`).
- Prepared GPQA's `doc_to_text` returns `input_final_prompts[0]` directly
  (`work_dir_gemma-4-31B/gpqa_cot/utils.py:8-17`).
- Inspection of every prepared sample found the literal markers
  `<|start_header_id|>`, `<|end_header_id|>`, and `<|eot_id|>` in all 541
  IFEval prompts and all 448 GPQA prompts.
- The exact Gemma tokenizer does not recognize `<|start_header_id|>` as one
  control token; it tokenizes the spelling into eight ordinary pieces.
- This contradicts `evals/eval_config.py:3710-3713`, which says these rows use
  the raw base completion contract and no chat template.

`--apply_chat_template` is correctly absent, but that does not undo formatting
already embedded in dataset strings. Scores from these prompts are not valid
evidence for the claimed raw Gemma base path.

### 5. High confidence, release-gating: both mandatory accuracy rows are ungradable and nonblocking

- `evals/eval_config.py:3714-3755` gives both mandatory base-model tasks an
  `EvalTaskScore`, but both `published_score` and `gpu_reference_score` are
  absent. Avoiding unrelated `-it` thresholds is correct, but the current
  release gate does not fail closed.
- `_score_one` in
  `tt-inference-server-v2/test_module/llm_tests/llm_eval_tests.py:146-170`
  assigns `accuracy_check=NA` whenever both references are absent, regardless
  of the measured score.
- `_check_evals` in
  `tt-inference-server-v2/report_module/acceptance_criteria.py:331-359` counts
  that state as `NA` but creates no blocker. Global acceptance at lines 71-83
  is simply the absence of blockers.
- Waivers are consulted only after a blocker message exists (lines 341-352), so
  this path neither blocks nor records an explicit issue waiver.

Consequently, both mandatory rows could score arbitrarily poorly and the
release could still be accepted. The honest remedy is a same-prompt Gemma-base
HF/GPU reference for each repaired raw-completion task. Until those references
exist, mandatory-no-baseline must be an explicit blocker that can only be
demoted through a task-specific `known_issues` waiver. Do not borrow `-it`
thresholds.

### 6. Medium confidence, currently non-triggering: `_clamp_max_gen_toks` violates exact per-request admission semantics

- `eval_command.py:29-59` reserves a fixed 1024 prompt tokens and can reduce a
  task's configured output budget before any actual prompt is known.
- Such a clamp can reject/change valid short-prompt requests and can still fail
  for prompts longer than the assumed reserve. It is not the mathematical
  condition `actual_prompt + actual_requested_completion <= max_context`.
- It does not trigger for the two current Gemma tasks because their
  `EvalTask.gen_kwargs` contains only `stream=False`; their output budgets come
  later from YAML.

This is not the cause of the current 2048 log, but it should not be extended as
the context fix. Prefer exact per-instance validation or the server's exact
request validation.

### 7. High confidence: the full release did not apply a hidden sample cap

- `tti_release_pass.log` records `limit_samples_mode: None` and
  `eval_samples: None`.
- The exact command at line 68 has neither `--limit` nor `--samples`.
- The log registers all 541 IFEval samples.
- `eval_command.py:385-391` appends those flags only when runtime configuration
  explicitly resolves them.

## Smallest safe repair

1. In `build_eval_command`, for text API evals, append
   `max_length=<device_model_spec.max_context>` when the task has not explicitly
   supplied `model_kwargs["max_length"]`. For this release both commands must
   contain exactly `max_length=113280`.
2. Do not add one to that value. Do not enable `truncate`. Do not align prompt
   lengths and do not synthesize a fixed prompt reserve.
3. Preserve the IFEval 1280 and GPQA 2048 task output budgets. Reject an
   instance only when its actual tokenized prompt plus that instance's requested
   completion exceeds 113280; otherwise send it unchanged. Relying on vLLM's
   exact server-side admission is valid for this API path.
4. Fix Gemma-specific Meta preparation so `doc_to_text` receives a genuinely
   base-compatible raw prompt, not Llama `input_final_prompts`. At minimum,
   IFEval should source the original raw instruction column and GPQA should use
   an explicitly reviewed raw question/choice completion prompt. Rebuild or
   invalidate the cached `work_dir_gemma-4-31B` artifacts after changing this;
   the current cache validator checks filenames/task IDs but not prompt format.

## Required focused tests

1. Extend
   `tests/test_workflow_venvs_meta.py::test_gemma_eval_commands_keep_base_completion_identity_and_full_dataset`
   to assert each command's `--model_args` contains `max_length=113280`, no
   `truncate=True`, no `--limit`, and no `--samples`.
2. Add command-builder tests showing a task-specific explicit `max_length`
   remains intentional if override precedence is retained, and that missing
   device context does not emit `max_length=None`.
3. Add a local lm-eval adapter regression for both `tokenized_requests=True`
   and `False`: a prompt longer than 2048 but valid under 113280 reaches
   `_create_payload` unchanged; a total exactly 113280 is accepted by the test
   admission stub and 113281 is rejected. No live server or hardware is needed.
4. Add Gemma preparation/cache-validation tests that inspect all prepared
   IFEval and GPQA prompts and fail on Llama sentinel strings. Assert the prompt
   source is the reviewed raw/base construction, not `input_final_prompts`.
5. Add an offline exact-tokenizer boundary test over prepared mandatory tasks:
   every request must satisfy `prompt_tokens + max_gen_toks <= 113280`; the test
   must fail rather than truncate or align any violating sample.
6. Add an acceptance regression proving a mandatory eval with no baseline is a
   blocker unless an explicit task-specific known-issue waiver exists. Add
   same-prompt Gemma-base reference scores before claiming an accuracy pass.

## Remaining uncertainty

No live request was sent and no hardware was touched. Code inspection proves
the client does not silently truncate current generation prompts and the
offline tokenizer sweep proves all currently prepared requests fit. A final
no-Docker rerun must still demonstrate that the emitted commands contain
`max_length=113280`, that rebuilt prompts contain no Llama wrappers, and that
the external autoport server accepts the unchanged requests.
