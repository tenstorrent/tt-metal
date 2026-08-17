# The release flow will test this model in a materially different configuration

Checked 2026-08-17 against `tt-inference-server` branch
`vvukoman/add-8-models-to-release-flow` (tip `60f80c4b`, one commit ahead of `main`)
and its `workflows/model_specs/dev/llm.yaml` entry for `Qwen/Qwen3.6-27B`.

That branch adds five *new* models (`google/gemma-4-26B-A4B-it`, `Qwen/Qwen3.8-27B`,
`Qwen/Qwen3.6-35B-A3B`, `meta-models/Muse-Glimmer-30B`,
`google/diffusiongemma-26B-A4B-it`). **Qwen3.6-27B is already in both the dev and prod
specs**, so what matters here is its existing release entry, not the diff.

## How `vllm_args` reach the server

`workflows/model_spec.py:420-430` merges the spec's `vllm_args` over defaults and
passes them as server flags:

```python
default_vllm_args = {
    "block_size": "64",
    "max_model_len": str(self.max_context),
    "max_num_seqs": str(max_concurrency),
    "max_num_batched_tokens": str(self.max_context),
    "max-log-len": "32",
    "seed": "9472",
    "additional_config": json.dumps({"tt": self.override_tt_config}),
}
merged_vllm_args = {**default_vllm_args, **self.vllm_args}
```

So everything in the spec's `vllm_args` block becomes a real CLI flag.

## The divergences

| setting | stage-11 run (and my re-runs) | release spec (P300X2) |
|---|---|---|
| `impl` | `qwen36_autoport` | **`qwen36_blackhole`** |
| `reasoning_parser` | **absent** | **`qwen3`** |
| `tool_call_parser` | absent | `qwen3_coder` |
| `enable_auto_tool_choice` | absent | `true` |
| `max_num_seqs` / `max_concurrency` | **1** | **32** |
| `sample_on_device_mode` | unset (model default) | **`decode_only`** |
| `fabric_config` | `FABRIC_1D_RING` | **`FABRIC_1D`** |
| `trace_region_size` | 200,000,000 | **1,073,741,824** |
| `l1_small_size` | unset | **24576** |
| `max_tokens_all_users_override` | unset | **525312** |
| `seed` | 42 (my re-runs) | **9472** |
| env vars | `{}` | `TT_QWEN35_TEXT_VER=qwen36_blackhole`, `MESH_DEVICE="(1, 4)"`, `TT_MESH_GRAPH_DESC_PATH=...p300_x2_mesh_graph_descriptor.textproto`, `QWEN36_MAX_TOKENS_ALL_USERS=525312` |

The stage-11 spec is unambiguous about the parser: its `device_model_spec.vllm_args`
is exactly

```json
{"model": "Qwen/Qwen3.6-27B", "block_size": "64", "max_model_len": "262144",
 "max_num_seqs": "1", "max_num_batched_tokens": "262144"}
```

with `reasoning_parser_name: "qwen3"` appearing only under `metadata`, which is
informational and never becomes a flag. My own server startup dump confirms the
consequence: `reasoning_parser=''`.

## Why the reasoning parser is the important one

Without a reasoning parser, vLLM returns the **entire** `<think>` block in
`choices[0].message.content`. lm-eval grades `content`. So every recorded number on
this branch was computed over reasoning text with the answer, if any, at the end.

With `reasoning_parser: qwen3`, vLLM splits the response: the think block goes to
`reasoning_content` and **`content` holds only what follows `</think>`**. That changes
grading directly:

- **GPQA** (`exact_match,flexible-extract`, via the patched `boxed_choice` filter):
  the filter would see a short clean answer rather than a long chain. It finds
  `\boxed{}` either way, so the *extraction* is not the issue — but a truncated
  response now yields **empty** content rather than a partial chain, which is a
  cleaner failure and a different one.
- **IFEval** (mean of the four prompt/inst x strict/loose keys): its instruction
  checks inspect response shape — "respond in all lowercase", "wrap in quotes",
  "include keyword X". Grading those against a reasoning chain instead of the answer
  is close to meaningless. The recorded 15/43 instruction-level score was measured
  that way. **With the parser, this number could move substantially, in either
  direction, for reasons that have nothing to do with the port.**

So the IFEval and GPQA figures recorded on this branch are not the numbers the
release flow will produce, and the gap is a configuration difference rather than a
model change.

## The second important one: the release serves at batch 32

`max_concurrency: 32` means the release server runs `--max_num_seqs 32`. Per
`SERVING_BATCH_LATENCY.md`, decode cost on this port follows the **allocated** batch,
not the active rows: one active request on a 32-slot server costs ~270 ms/token
against ~56 ms at `max_num_seqs=1`.

Consequently the headline single-user figures on this branch — `TPOT 61.893 ms`,
`ITL P50/P99 55.840/56.850 ms`, decode `16.157 t/s/u`, all measured at
`max_num_seqs=1` — **are not what the release configuration delivers**. At
`max_num_seqs=32` a single user should see roughly `3.7 t/s/u`. That is worth
resolving before any single-user latency claim is attached to the release.

## The third: the release runs a different code tree

`impl: qwen36_blackhole` with `TT_QWEN35_TEXT_VER: qwen36_blackhole` points at
`models/demos/blackhole/qwen36`. That directory **exists on tt-metal `origin/main`**
(`f3cfc53ef81`) but **not at this branch's base pin**, and this branch's work lives at
`models/autoports/qwen_qwen3_6_27b` under `impl_id: qwen36_autoport`.

So findings on this branch transfer to the release only insofar as the two trees share
code. Anything specific to the autoport — the precision policy in
`doc/datatype_sweep/selected_precision_config.json`, `LINEAR_PREFILL_CHUNK_SIZE` and
the new `QWEN36_LINEAR_PREFILL_CHUNK_SIZE` hook, the prefill scan implementation whose
op mix is analysed in `PREFILL_CHUNK_LEVER.md` — must be re-checked against
`models/demos/blackhole/qwen36` before being claimed of the release. I have not done
that comparison; the demo tree is not present at this pin.

This is the single most important caveat on everything else in this directory.

## What to test locally, in order

1. **Reasoning parser, everything else unchanged.** Add `--reasoning_parser qwen3` to
   the known-good server (`FABRIC_1D_RING`, 200 MB trace, `max_num_seqs 1`) and re-run
   the Diamond probe plus IFEval. Isolates the grading effect of the parser from every
   other difference. Lowest risk: server-side only, no device-config change.
2. **Release `override_tt_config`.** Switch to `FABRIC_1D`, `trace_region_size`
   1 GB, `l1_small_size` 24576, `sample_on_device_mode decode_only`. Each of these can
   plausibly fail on the autoport: the port's own multichip documentation justifies
   `FABRIC_1D_RING`, and `sample_on_device_mode` interacts with the TP4 vocabulary
   shard — the sibling gemma-4 entry in the same spec carries the comment "Required on
   the TP mesh so token ids >= 65536 are reachable (host sampling only sees device 0's
   vocab shard)", and this model's EOS ids are **248,044 and 248,046**, far above that
   threshold.
3. **`max_num_seqs 32`.** Confirms the batch-32 latency penalty in the release
   configuration and, incidentally, makes long reasoning evals faster in wall clock by
   overlapping documents.

---

## Code-verified prediction: the release configuration scores 0.00 on GPQA

Established 2026-08-17 from source and measurement, before running the release
configuration. Six links, each checked:

1. **Thinking is ON in the release.** The chat template's default branch emits an open
   `<think>` (see `doc/tti_release/NON_TERMINATION.md`), and the Qwen3.6-27B release
   entry sets **no** `default-chat-template-kwargs`. The sibling gemma-4 entry in the
   same file *does* (`'{"enable_thinking": true}'`), so the knob exists and was simply
   not set here.

2. **The release enables the reasoning parser.** `vllm_args: reasoning_parser: qwen3`,
   and `workflows/model_spec.py:429` merges `vllm_args` straight onto the server
   command line.

3. **The release grants no generation budget.** `EvalTask.gen_kwargs` defaults to
   `{"stream": "False"}` (`reference_config/evals/eval_config.py:209`) and the
   Qwen3.6-27B tasks do not override it. `gpqa_diamond_cot_zeroshot`'s own YAML sets no
   `max_gen_toks` either, so lm-eval falls back to its API default of **256**. This is
   exactly what stage 11 recorded: `gen_kwargs = {'stream': False, 'seed': 42}`.

4. **256 tokens cannot reach `</think>` on a real Diamond item.** Measured: the same
   Diamond row 0 consumed the entire 32,768-token budget in thinking mode without
   closing.

5. **The parser then returns no content.** `vllm/reasoning/qwen3_reasoning_parser.py`,
   `extract_reasoning`:

   ```python
   # Thinking enabled but no </think>: output was truncated.
   # Everything generated so far is reasoning.
   return model_output, None
   ```

   Its docstring states the same: "Otherwise (thinking enabled, default), a missing
   `</think>` means the output was truncated and everything is reasoning: returns
   `(model_output, None)`."

6. **lm-eval grades `content`.** Empty content yields `[invalid]` from the
   `boxed_choice` filter, so `exact_match,flexible-extract` — the single key the GPQA
   score function reads — is **0.00**.

So the release configuration is predicted to score **0.00**, *below* the 0.30 the
autoport recorded without a parser, and for a purely configurational reason. Nothing
about the port changes between those two numbers.

### The same file already contains the fix pattern

Other reasoning models get an explicit budget for the very same task:

| model | task | `max_gen_toks` |
|---|---|---:|
| `zai-org/GLM-5.2` | **`gpqa_diamond_cot_zeroshot`** | **200 x 1024** |
| `moonshotai/Kimi-K2.6` | `r1_gpqa_diamond` | 256 x 1024 |
| `moonshotai/Kimi-K2.7-Code` | `r1_gpqa_diamond` | 256 x 1024 |
| **`Qwen/Qwen3.6-27B`** | **`gpqa_diamond_cot_zeroshot`** | **none** |

### But the budget fix alone is impractical on this hardware

At the measured 56 ms/token, a 200 x 1024 = 204,800-token budget is up to **3.2 h per
document** and about **32 h for the ten-document CI subset**. GLM-5.2 presumably runs
on hardware where that is affordable. Here it is not.

### What the measurements say the fix should be

Thinking OFF answers the hard item correctly and cheaply. Measured on this port,
`chat_template_kwargs: {"enable_thinking": false}`:

| item | tokens | finish_reason | answer |
|---|---:|---|---|
| "What is 2 + 2?" | **2** | stop | correct |
| easy physics MCQ | 472 | stop | `\boxed{A}` correct |
| **real Diamond row 0** | **1,849** | **stop** | **`\boxed{A}` correct** |

The same Diamond row 0 that does not converge in 32,768 tokens with thinking ON is
answered **correctly in 1,849 tokens** with thinking OFF.

So the recommended release configuration for this model, in order of preference:

1. **Set `default-chat-template-kwargs: '{"enable_thinking": false}'`** plus a modest
   `gen_kwargs.max_gen_toks` (4096 is ample given the 1,849-token measurement). Cheap,
   terminating, and it grades the model's actual answer. It must be labelled as a
   non-thinking condition, because it is not comparable to a published thinking-mode
   score.
2. **Or keep thinking ON and set a real budget**, accepting the wall-clock cost and
   the risk that some items still will not converge. On this hardware that is a
   multi-day CI job, so it belongs on a published-number run rather than in nightly CI.

Doing neither leaves the release reporting 0.00 for a model that answers the same
questions correctly.

### One subtlety still to verify by running

`Qwen3ReasoningParser.__init__` reads the flag once, at construction:

```python
chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
self.thinking_enabled = chat_kwargs.get("enable_thinking", True)
```

That is *constructor* state, not per-request state. A request that passes
`chat_template_kwargs={"enable_thinking": false}` may therefore still be parsed with
`thinking_enabled=True`, in which case output containing no `</think>` — which is what
the thinking-off template produces — would be classified entirely as reasoning and
**content would be empty even in the thinking-off arm**. The docstring says the serving
layer handles this via `prompt_is_reasoning_end`, but explicitly in the *streaming*
path.

The queued release probe tests exactly this: it runs with `--reasoning_parser qwen3`
active and includes a `thinking_enabled=false` case, reporting `content` and
`reasoning_content` lengths separately. If the thinking-off arm also returns empty
content in non-streaming mode, then recommendation 1 above needs
`default-chat-template-kwargs` at *server* level rather than per-request
`chat_template_kwargs` — which is precisely how the gemma-4 entry sets it.

---

## Which GPQA task: `r1_gpqa_diamond`, not `gpqa_diamond_cot_zeroshot`

Answered from the branch itself. **All four models the branch adds use
`task_name="r1_gpqa_diamond"`**, each with an explicit budget:

| model | task | `max_gen_toks` |
|---|---|---:|
| `Qwen/Qwen3.8-27B` | `r1_gpqa_diamond` | 80 x 1024 = 81,920 |
| `Qwen/Qwen3.6-35B-A3B` | `r1_gpqa_diamond` | 80 x 1024 = 81,920 |
| `meta-models/Muse-Glimmer-30B` | `r1_gpqa_diamond` | 24 x 1024 = 24,576 |
| `google/diffusiongemma-26B-A4B-it` | `r1_gpqa_diamond` | 32 x 1024 = 32,768 |
| **`Qwen/Qwen3.6-27B`** (already in main) | **`meta_gpqa_cot` -> `gpqa_diamond_cot_zeroshot`** | **none** |

So this model is still on the older pattern while its closest sibling,
`Qwen/Qwen3.8-27B`, is onboarded the new way.

### Why the task choice is decisive, not cosmetic

`lm_eval/tasks/r1_evals/gpqa_reasoning_diamond.yaml` against
`lm_eval/tasks/gpqa/cot_zeroshot/_gpqa_cot_zeroshot_yaml`:

| | `gpqa_diamond_cot_zeroshot` (what ran) | `r1_gpqa_diamond` (release) |
|---|---|---|
| `max_gen_toks` | **absent -> lm-eval API default 256** | **32768, in the YAML** |
| `until` | `["</s>"]` — **not a Qwen stop token** | `<\|im_end\|>`, `<\|endoftext\|>`, `<\|end_of_text\|>` |
| sampling | `do_sample: false`, `temperature: 0.0` | `temperature: 0.6, top_k: 40, top_p: 0.95` |
| extraction | `strict-match` (regex "The answer is", structurally unmatchable) + `flexible-extract` | own `process_results_gpqa` |
| graded key | `exact_match,flexible-extract` | `exact_match,none` |

Every one of those four differences hurts a thinking model, and together they account
for what I measured: a 256-token cap that cannot escape the `<think>` block, greedy
decoding on a model whose card specifies otherwise, and a stop list that does not
contain the token this model actually emits.

### The sibling entry's own notes are worth quoting

```
# R1-style zero-shot reasoning GPQA Diamond: the model emits reasoning then a
# final answer, and the task's own extractor scores exact_match,none. Do NOT
# switch to gpqa_diamond_generative_n_shot -- its 5-shot examples demonstrate
# bare "(C)" answers and suppress reasoning (that cost gemma-4 ~30 points).
```

and on the endpoint:

```
# Use the chat endpoint so the server applies the chat template (which is what
# carries thinking mode); client-side apply_chat_template on /v1/completions
# would bypass it.
```

That second note independently confirms the mechanism documented in
`NON_TERMINATION.md`: the chat template is what carries thinking mode.

### The sampling divergence I had not tested

The sibling's `gen_kwargs`:

```python
"stream": "false",          # REQUIRED: lm-eval's streaming parser raises KeyError 'message'
"max_gen_toks": 80 * 1024,
"until": [],
"do_sample": "true",
"temperature": 1.0,         # Qwen card, thinking mode
"top_k": 20,
"top_p": 0.95,
```

`temperature 1.0 / top_k 20 / top_p 0.95` is exactly this model's own
`generation_config.json`. **Every graded run on this branch, and every probe I ran
until now, used greedy `temperature 0.0`** — which is neither the task YAML's value
(0.6) nor the release's (1.0), and greedy decoding is a known non-convergence mode for
reasoning models.

That is a live candidate explanation for the 32,768-token runaway, and it had not been
tested. The updated release probe now includes both non-greedy profiles on Diamond
row 0 so the comparison is direct:

| arm | sampling | budget |
|---|---|---:|
| `greedy_diamond0_256` | greedy t=0 | 256 |
| `r1sampling_diamond0_32768` | t=0.6, top_k=40, top_p=0.95 | 32768 |
| `release_sampling_diamond0_32768` | t=1.0, top_k=20, top_p=0.95 | 32768 |
| `thinkoff_diamond0_4096` | greedy t=0, thinking OFF | 4096 |

If a non-greedy arm converges where greedy did not, the runaway was a
sampling-configuration artifact that the release `gen_kwargs` already avoids — and the
correct conclusion is that the port was fine and the *old* task entry was the defect.

### Local test now mirrors the release

`/tmp/run_r1_gpqa.sh` runs `--tasks r1_gpqa_diamond` with the release's own gen_kwargs
(`stream=false, do_sample=true, temperature=1.0, top_k=20, top_p=0.95`), `--seed 9472`,
`--block_size 64`, `--reasoning_parser qwen3`, and `apply_chat_template` (the EvalTask
default is `True`, so the release passes it too). Verified that `r1_gpqa_diamond`
resolves in this venv without `--include_path` — it is in the task index and loads its
198 documents.

Two device settings are deliberately left at the autoport's validated values —
`FABRIC_1D_RING` rather than the spec's `FABRIC_1D`, and a 200 MB rather than 1 GB
trace region — so that this run isolates the *task and grading* change. The device
settings get their own run.

### Recommendation for the release config

Onboard `Qwen/Qwen3.6-27B` the same way its sibling `Qwen/Qwen3.8-27B` is onboarded:
`r1_gpqa_diamond`, explicit `max_gen_toks`, model-card sampling, chat endpoint. Note
the wall-clock consequence on this hardware: at the measured 56 ms/token an 80 x 1024
budget is up to ~76 min per document, so the YAML's own 32,768 is the affordable
starting point for CI and the larger budget belongs to a published-number run.

---

## Correction: upstream, this model has NO standard eval task at all

Checked 2026-08-17 against `tt-inference-server` `origin/main` (`6d8c1aab`) and branch
`vvukoman/add-8-models-to-release-flow` (`60f80c4b`).

Earlier sections of this document describe the Qwen3.6-27B `EvalConfig` as containing
`meta_ifeval` -> `ifeval` and `meta_gpqa_cot` -> `gpqa_diamond_cot_zeroshot`. **That is
true only of the local checkout, not of upstream.**

| | standard eval tasks for `Qwen/Qwen3.6-27B` |
|---|---|
| local checkout `b9a18e8f` | `meta_ifeval` -> `ifeval`, `meta_gpqa_cot` -> `gpqa_diamond_cot_zeroshot`, `terminal_bench_2` |
| upstream `main` **and** the new branch | **`terminal_bench_2` only** (with `swe_bench_verified` commented out) |

Evidence:

- The branch's diff of `reference_config/evals/eval_config.py` has **zero deletions** and
  mentions `Qwen3.6-27B` only inside *comments* on the newly added entries, which refer to
  it as "the sibling Qwen3.6-27B config above". So the branch does not modify this model's
  entry.
- `git log origin/main..HEAD` in the local checkout shows `b9a18e8f` — *"Qwen3.6-27B: eval
  config, terminal-bench token budget, external-chat meta evals"* — and
  `git merge-base --is-ancestor HEAD origin/main` reports **not an ancestor**. The tree is
  clean, so the AUTOFIX change is committed locally and **never upstreamed**.
- `git log -S "meta_gpqa_cot"` attributes the addition to that same local commit.

### Consequence when the branch lands

The under-onboarding that `AUTODEBUG.md` diagnosed is **still live upstream**: the
standard release child admits only `EVALS_COMMON` / `EVALS_META` / `EVALS_VISION`, this
model's only active task is `EVALS_AGENTIC`, so standard selection returns `[]` — and the
workflow converts an empty task result into a **successful no-op**.

So when the commit lands, this model is evaluated on **neither** `r1_gpqa_diamond` **nor**
`gpqa_diamond_cot_zeroshot`. It runs only `terminal_bench_2`, which additionally needs
Docker that the model container does not have.

This also corrects the prediction recorded above that the release "scores 0.00 on GPQA".
That prediction was derived from the local checkout's configuration. Upstream there is no
GPQA task to score, which is a worse failure mode than 0.00 because a zero is visible in
a report and a silent empty selection is not.

### What the fix should be

Not to upstream the local AUTOFIX as written — its `meta_gpqa_cot` -> `cot_zeroshot`
mapping is the variant with the 256-token default, the `["</s>"]` stop list, and greedy
sampling, i.e. the configuration proven above to produce a repetition loop.

Instead, onboard this model the way the same branch onboards its sibling
`Qwen/Qwen3.8-27B`:

```python
EvalTask(
    task_name="r1_gpqa_diamond",
    workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
    use_chat_api=True,                      # server applies the chat template
    model_kwargs={"max_length": 262144},
    gen_kwargs={
        "stream": "false",                  # lm-eval streaming parser raises KeyError
        "max_gen_toks": 32 * 1024,          # 80*1024 is ~76 min/doc at 56 ms/token here
        "until": [],
        "do_sample": "true",
        "temperature": 1.0, "top_k": 20, "top_p": 0.95,   # this model's own card
    },
    score=EvalTaskScore(
        published_score=...,                # from the Qwen3.6-27B card
        score_func=score_task_single_key,
        score_func_kwargs={"result_keys": ["exact_match,none"], "unit": "percent"},
    ),
    limit_samples_map={EvalLimitMode.CI_NIGHTLY: 0.05, EvalLimitMode.SMOKE_TEST: 0.01},
)
```

plus an `ifeval` task if instruction-following is wanted — noting that IFEval's checks
inspect response *shape*, so it should be run either with a reasoning parser configured
or with thinking disabled, or it grades the think block rather than the answer.

The budget choice deserves a deliberate decision rather than copying 80*1024: at the
measured 56 ms/token that is up to ~76 minutes per document on this hardware, whereas the
task YAML's own 32,768 is ~31 minutes and the measured convergence for this item under
non-greedy sampling should be far below either.
