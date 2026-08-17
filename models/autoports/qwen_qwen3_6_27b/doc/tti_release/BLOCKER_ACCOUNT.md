# What actually blocked stage 11, separated into three different things

Operator account, 2026-08-17, written while the corrected re-runs execute. The
short version: "stage 11 blocked on eval quality" conflates a fixed
infrastructure defect, a set of evals that cannot pass or fail because they have
no thresholds, and one genuine remaining blocker that is not about quality at all.

## 1. The original defect: zero standard evals ran, and that counted as success

`AUTODEBUG.md` established it and `AUTOFIX.md` fixed it. The TTI eval catalog
entry for `Qwen/Qwen3.6-27B` contained only the agentic `terminal_bench_2`. The
standard release child admits only `EVALS_COMMON`, `EVALS_META` or
`EVALS_VISION`, so it selected **zero** tasks — and the workflow converts an
empty task result into a **successful no-op**.

That is the same defect class as the `EXPERIMENTAL` eval-enforcement hole found
on the Gemma port: a release can report success while measuring nothing. Worth
noting as a pattern, because it is silent in both cases.

Fixed by adding active `EVALS_META` `meta_ifeval` and `meta_gpqa_cot` entries.

## 2. The standard evals have no thresholds, so they cannot fail

`AUTOFIX.md` is explicit, and this is the part that gets lost:

> "no Llama/Meta score or mismatched model-card score was copied. The task rows
> are emitted and measurable, but their accuracy checks remain **`N/A`** until
> matched Qwen GPU/control baselines establish full-set or CI-subset references."

It also records *why* no threshold could be borrowed: the Qwen model card has no
IFEval score at all, and its GPQA Diamond figure does not identify the TTI
strict-match recipe, so using it would not be defensible.

So the low numbers below are **not** a failed gate. There is no gate. They are
unanchored measurements, and the missing input is a matched control baseline.

## 3. The recorded numbers are grading artifacts, not model quality

Both `release_final3_cache` and `release_final4_cache` ran the generic lm-eval
tasks and recorded byte-identical scores:

| task | metric | recorded |
|---|---|---:|
| `gpqa_diamond_cot_zeroshot` | `exact_match,flexible-extract` | **0.30** |
| `gpqa_diamond_cot_zeroshot` | `exact_match,strict-match` | **0.00** |
| `ifeval` | `inst_level_strict_acc` | **0.3488** |
| `ifeval` | `prompt_level_strict_acc` | **0.1786** |

### GPQA flexible-extract 0.30 is chance, under a 256-token cap

`lm_eval/tasks/gpqa/cot_zeroshot/_gpqa_cot_zeroshot_yaml` sets no `max_gen_toks`,
so generation fell to the API backend's default **256**. Four choices means
guessing scores 0.25; 0.30 on ten documents is one document above chance.

The retained per-sample outputs confirm the mechanism rather than infer it:
median **120 words**, **0/20** ending in terminal punctuation, **0/20**
containing `boxed`, **16/20** filtered to `[invalid]`, and tails severed
mid-expression (`$\Gamma_2 \approx \frac{\hbar}{`, `1. 1,`).

### GPQA strict-match 0.00 is structural and cannot be fixed by budget

The prompt in that same YAML instructs:

> "Please reason step by step, and put your final answer (only the letter A, B,
> C, or D) within `\boxed{}`."

while the `strict-match` filter is:

```
regex_pattern: "(?<=The answer is )(.*)(?=.)"
```

Nothing in the prompt ever elicits the string "The answer is". A perfectly
compliant model that does exactly what it was asked scores **0.00** on this
filter. This is a defect in the task definition, not in the port, and no
generation budget changes it. The same 0.0-in-both-arms pattern was seen on the
Gemma port.

`flexible-extract` is the only meaningful arm here, and in this venv it is a
**locally patched** `boxed_choice` filter — its own comment records replacing
`multi_choice_regex`, which used to take the last `(X)` in the response and so
grabbed chemistry stereodescriptors like (E)/(R)/(H) instead of the choice. That
patched filter reads `\boxed{}` correctly, which is exactly why the 0/20
`boxed`-free truncated responses produced `[invalid]`.

### IFEval was measured under truncation too

`lm_eval/tasks/ifeval/*.yaml` sets `max_gen_toks: 1280`, and the retained samples
show median **759 words** with only **9/28** ending in terminal punctuation.
Also worth flagging: `loose_acc` equals `strict_acc` **exactly** at both prompt
and instruction level, in both runs — loose is normally the more forgiving
metric, so identical values are worth understanding rather than accepting.

## 4. The one genuine remaining blocker

`terminal_bench_2` — the only task the catalog originally had — is agentic and
needs Docker, which the model container does not have. It is satisfiable from the
host (serve from the container, run the Harbor client outside it, point
`api_base` at the container bridge address), but that is an infrastructure
decision, not a model or accuracy question.

## What the corrected re-runs are for

Not to pass a gate — there isn't one. They establish the port's number at a
**correct** generation budget, which is the defensible TT side of the reference
`AUTOFIX.md` says is missing. Queue:

1. `gpqa_diamond_cot_zeroshot` at `max_gen_toks=32768` — the decisive test
2. `ifeval` at 4096 — above its declared 1280
3. `ifeval` at 1280 — control, in the stage's own `max_num_seqs=1` config

Expected, recorded before seeing results so it can be checked: `flexible-extract`
should rise well above 0.30 if truncation was the cause, and `strict-match`
should stay at **0.00** regardless, because that filter cannot match this
prompt's requested format. If `flexible-extract` does *not* rise, truncation was
not the whole story and the next suspect is the numerical policy.

## Pattern across this fleet

Third instance of a grading artifact presenting as a model defect: Falcon3-7B's
IFEval metric-variant mismatch (0.544 → parity), Qwen's GPQA 256-token cap, and
Qwen's IFEval 1280 truncation. In each case the port was measured, not broken.
The lesson worth carrying: check the task definition and the retained samples
before attributing a low score to the implementation.

---

## Resolved: why IFEval `loose` equals `strict` exactly

Flagged above as "worth understanding rather than accepting". It is benign, and it
corroborates the truncation diagnosis rather than complicating it.

`lm_eval/tasks/ifeval/utils.py:test_instruction_following_loose` evaluates **eight**
variants of each response and passes an instruction if *any* of them satisfies it:

```
response, revised_response (asterisks removed),
response_remove_first, response_remove_last, response_remove_both,
revised_response_remove_first, revised_response_remove_last, revised_response_remove_both
```

So `loose >= strict` holds by construction, and `loose > strict` exactly when a
failure was caused by presentation — a markdown wrapper, a preamble line, a
trailing sign-off.

The recorded values decode to whole counts:

| metric | recorded | as a fraction |
|---|---:|---:|
| `inst_level_*_acc` | 0.3488372093023256 | **15/43** |
| `prompt_level_*_acc` | 0.17857142857142858 | **5/28** |

Identical for strict and loose, in both `release_final3` and `release_final4`.
So across 28 prompts and 43 instructions, **none of the eight formatting variants
rescued a single instruction.**

That is what truncation looks like. Stripping asterisks or dropping a first line
cannot rescue a response that stopped mid-sentence — and the retained samples show
median 759 words with only 9/28 ending in terminal punctuation. Had the failures
been formatting artifacts, loose would have exceeded strict.

Read the other way, this is mildly reassuring about the port: the model is not
producing subtly malformatted output. It is producing output that ends too early.
Whether a larger budget actually recovers the score is what the queued
`ifeval` @ 4096 run tests; the 1280 re-run is only a reproducibility control,
since 1280 is what the task already granted.

---

## Correction: what the committed eval config actually says

Read directly from `reference_config/evals/eval_config.py:1265-1320` at
tt-inference-server. Three points, one of which corrects this document and one of
which corrects `AUTOFIX.md`.

### 1. The `meta_*` names are labels; the mapping is explicit

```python
EvalTask(task_name="meta_ifeval",   eval_task_name="ifeval",                     ...)
EvalTask(task_name="meta_gpqa_cot", eval_task_name="gpqa_diamond_cot_zeroshot",  ...)
```

`task_name` is TTI's internal label; `eval_task_name` is the lm-eval task actually
executed. So the release runs executing `ifeval` and `gpqa_diamond_cot_zeroshot`
were doing **exactly what the config asks**. There is no task-variant mismatch here,
and anyone reading the recorded results should not go looking for one. (This is
unlike the Falcon3-7B case, which was a genuine metric-variant mismatch.)

### 2. `AUTOFIX.md`'s description of its own fix does not match what landed

`AUTOFIX.md` states the tasks were added as `EVALS_META` using "the established
preformatted-prompt contract (`include_path="work_dir"`,
`apply_chat_template=False`)". The committed entries say otherwise:

```python
workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
use_chat_api=True,
apply_chat_template=True,
```

`EVALS_COMMON`, not `EVALS_META`; and `apply_chat_template=True`, not `False`.

This matters, and not pedantically. `apply_chat_template=True` is **precisely the
path that turns thinking mode on**: lm-eval renders the chat template itself with no
`enable_thinking` argument, so the generation prompt ends with an open `<think>`.
Had the preformatted-prompt contract `AUTOFIX.md` described actually landed, the
prompt text would have been fixed in the task file and inspectable, and this
behaviour would have been visible rather than inherited.

So the drift between the report and the code is the reason the thinking-mode
default went unnoticed. Trust the code here, not the report.

### 3. `strict-match` is not the graded key, so its structural 0.00 is harmless

The GPQA score function reads exactly one key:

```python
score_func=score_task_single_key,
score_func_kwargs={"result_keys": ["exact_match,flexible-extract"]},
```

So the structurally-unmatchable `strict-match` regex documented above — the prompt
asks for `\boxed{}` while the filter looks for "The answer is" — **never affected
scoring**. It remains a real defect in the upstream task definition and it is still
worth reporting upstream, but it is not part of this port's story and should not be
counted as a blocker.

IFEval, by contrast, averages all four keys:

```python
score_func=score_task_keys_mean,
score_func_kwargs={"result_keys": [
    "prompt_level_strict_acc,none", "inst_level_strict_acc,none",
    "prompt_level_loose_acc,none",  "inst_level_loose_acc,none"]}
```

which is why the loose/strict identity documented above matters for its score: with
loose equal to strict, the mean is just the strict pair, i.e. (0.1786 + 0.3488)/2.

### 4. No thresholds, confirmed in code

```python
published_score=None,
published_score_ref=None,
```

for both tasks. This confirms from the code what `AUTOFIX.md` asserted in prose:
these evals emit numbers and cannot pass or fail.
