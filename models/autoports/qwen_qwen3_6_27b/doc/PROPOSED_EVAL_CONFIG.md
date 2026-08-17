# Proposed `EvalConfig` change to onboard Qwen3.6-27B properly

Written 2026-08-17. Every value below is justified either by a measurement recorded in
this directory or by the pattern the `vvukoman/add-8-models-to-release-flow` branch uses
for this model's closest sibling, `Qwen/Qwen3.8-27B`.

## The problem being fixed

Upstream `reference_config/evals/eval_config.py` gives `Qwen/Qwen3.6-27B` exactly one
active task, `terminal_bench_2` (`EVALS_AGENTIC`). Standard release selection admits only
`EVALS_COMMON` / `EVALS_META` / `EVALS_VISION`, so it selects nothing and the workflow
records the empty result as a **successful no-op**. See
`doc/tti_release/AUTODEBUG.md` and `doc/RELEASE_CONFIG_DIVERGENCE.md`.

The local checkout's AUTOFIX added `meta_gpqa_cot` mapped to
`gpqa_diamond_cot_zeroshot`. **That mapping should not be upstreamed**: that task sets no
`max_gen_toks` (so lm-eval uses 256), `until: ["</s>"]` (not a Qwen stop token), and
greedy `do_sample: false / temperature: 0.0`, which is measured to send this model into a
repetition loop on hard items — 16,384 tokens at 50.06% duplicate 12-grams, one 12-gram
repeated 1,241 times.

## The proposed tasks

```python
    EvalConfig(
        hf_model_repo="Qwen/Qwen3.6-27B",
        tasks=[
            # ---------------------------------------------------------------- GPQA
            EvalTask(
                # R1-style zero-shot reasoning GPQA Diamond, matching how the sibling
                # Qwen3.8-27B entry is onboarded. Chosen over gpqa_diamond_cot_zeroshot
                # for four reasons, each measured on TT silicon:
                #   * cot_zeroshot sets NO max_gen_toks, so lm-eval falls back to 256
                #     tokens, which cannot escape this model's <think> block;
                #   * cot_zeroshot sets greedy decoding, which loops on hard items
                #     (50.06% duplicate 12-grams over 16,384 tokens);
                #   * cot_zeroshot's `until` is ["</s>"], not a Qwen stop token;
                #   * cot_zeroshot's strict-match regex looks for "The answer is" while
                #     its own prompt asks for \boxed{}, so that arm is always 0.00.
                # r1_gpqa_diamond fixes all four: 32768 budget in the YAML, correct
                # <|im_end|> stop list, non-greedy sampling, and its own extractor
                # scoring exact_match,none.
                task_name="r1_gpqa_diamond",
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                # Chat endpoint so the SERVER applies the chat template, which is what
                # carries thinking mode. Client-side templating on /v1/completions would
                # bypass it. (Same rationale as the Qwen3.8-27B entry.)
                use_chat_api=True,
                model_kwargs={"max_length": 262144},   # the P300X2 spec's max_context
                gen_kwargs={
                    # stream=false is REQUIRED: lm-eval's local-chat-completions
                    # streaming parser raises KeyError 'message' on every response.
                    "stream": "false",
                    # 32*1024, not the sibling's 80*1024. Measured on this hardware:
                    # decode is ~56 ms/token, so 80*1024 is up to ~76 min PER DOCUMENT
                    # and ~12.7 h for the 10-document CI subset, whereas 32768 is
                    # ~31 min worst case. Non-greedy convergence on a real Diamond item
                    # measured ~1.8k tokens, so this budget should rarely bind. Raise it
                    # for a published-number run on faster silicon.
                    "max_gen_toks": 32 * 1024,
                    "until": [],
                    # This model's own generation_config.json: do_sample true,
                    # temperature 1.0, top_k 20, top_p 0.95. Greedy is what breaks it.
                    "do_sample": "true",
                    "temperature": 1.0,
                    "top_k": 20,
                    "top_p": 0.95,
                },
                score=EvalTaskScore(
                    # TODO: fill from the Qwen/Qwen3.6-27B model card's GPQA Diamond
                    # number. Deliberately NOT guessed here -- the local weight snapshot
                    # is weights-only with no README, and model_performance_reference.json
                    # has no entry for this model, so there is no defensible local source.
                    # AUTOFIX.md already refused to copy a mismatched score for the same
                    # reason. Until it is filled, resolve_eval_reference() has neither a
                    # published nor a GPU baseline.
                    published_score=None,
                    published_score_ref=None,
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        # r1_gpqa_diamond uses its own process_results_gpqa and emits an
                        # unfiltered metric, so the key has the ,none suffix -- NOT
                        # exact_match,flexible-extract as the cot_zeroshot entry used.
                        "result_keys": ["exact_match,none"],
                        "unit": "percent",
                    },
                ),
                # Reasoning eval at low batch: documents run effectively sequentially and
                # dominate CI runtime. Same values as the sibling entry.
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.05,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            # ------------------------------------------------------------- IFEval
            EvalTask(
                task_name="meta_ifeval",
                eval_task_name="ifeval",
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                use_chat_api=True,
                gen_kwargs={
                    "stream": "false",
                    # ifeval.yaml sets max_gen_toks 1280 itself, which is a THINKING-mode
                    # problem rather than a budget problem: measured samples ran 759 words
                    # median with only 9/28 ending in terminal punctuation, and loose_acc
                    # equalled strict_acc exactly (15/43 instructions), meaning none of
                    # lm-eval's eight formatting variants rescued a single instruction.
                    # IFEval's checks inspect response SHAPE ("all lowercase", "wrap in
                    # quotes"), so grading them against a reasoning chain is meaningless.
                    # Either the server must run with reasoning_parser=qwen3 so content
                    # holds only the answer, or thinking must be disabled for this task.
                    "max_gen_toks": 4096,
                    "do_sample": "true",
                    "temperature": 1.0,
                    "top_k": 20,
                    "top_p": 0.95,
                },
                score=EvalTaskScore(
                    published_score=None,      # TODO: the Qwen card publishes no IFEval
                    published_score_ref=None,  # score; AUTOFIX.md records this.
                    score_func=score_task_keys_mean,
                    score_func_kwargs={
                        "result_keys": [
                            "prompt_level_strict_acc,none",
                            "inst_level_strict_acc,none",
                            "prompt_level_loose_acc,none",
                            "inst_level_loose_acc,none",
                        ],
                        "unit": "percent",
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.05,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            # terminal_bench_2 unchanged (agentic; needs Docker, which the model
            # container lacks -- satisfiable from the host, see doc/HANDOFF-style notes).
        ],
    ),
```

## Two things this does not fix

1. **The silent no-op itself.** Even with these tasks added, a future model that is
   onboarded with only agentic tasks will still have its empty standard selection recorded
   as success. That is a defect in `llm_module/eval_configs.py:get_llm_eval_tasks` +
   `workflow_module/workflows.py`, not in any model's entry, and it is what allowed this to
   go unnoticed. Worth fixing separately: an empty standard selection for a model that
   declares standard evals should be an error, not a no-op.

2. **The `max_num_seqs` mismatch.** The release spec serves at `max_concurrency: 32`, and
   decode cost on this port follows the allocated batch rather than the active rows, so a
   single user there sees ~270 ms/token against the ~56 ms the headline figures quote. See
   `doc/SERVING_BATCH_LATENCY.md`. That is a serving-claim question, independent of evals.

## Verification available locally

`tests/run_r1_gpqa.sh` runs exactly this task with exactly these `gen_kwargs` (at the
32768 budget) against the autoport with `--reasoning_parser qwen3`, so the proposed entry
can be validated before it is upstreamed.
