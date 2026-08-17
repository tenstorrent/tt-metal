# The model does not stop: GPQA consumed all 32,768 tokens on document 1

Measured 2026-08-17. This supersedes the "GPQA was truncated at 256" account as a
*complete* explanation: the 256-token cap was real, but removing it does not fix
the score, because generation does not terminate.

## The measurement

Re-ran `gpqa_diamond_cot_zeroshot` at `max_gen_toks=32768` in the stage's own
serving configuration (`max_num_seqs=1`, `max_num_batched_tokens=262144`,
confirmed by the server allocating 1,727,200 KV tokens in 2,159 page-800 blocks,
byte-identical to the stage-10 record).

Document 1: **1,887.95 s**, at a server-reported **17.8 tokens/s** — i.e.
**~33,600 tokens, the entire 32,768 budget**, without stopping. lm-eval projected
**4:43:11** for the remaining nine.

The 17.8 tok/s figure is itself a useful cross-check: it equals the recorded
56 ms/token ITL, confirming decode speed is as documented and that the
`max_num_seqs=1` config took effect.

I stopped the run after document 1. Nine more documents at ~31 min each would have
cost ~5 hours to confirm what document 1 already establishes — every document will
hit the cap — while the more valuable question is *why* it does not stop.

## What this means for the earlier account

`BLOCKER_ACCOUNT.md` attributed GPQA's 0.30 to the implicit 256-token cap, with
retained samples showing median 120 words, 0/20 containing `boxed`, and tails cut
mid-expression. All of that remains true and correctly diagnosed. What was wrong
was the implied conclusion that a larger budget would recover the score. It does
not: at 256 tokens the response is cut early in a chain, and at 32,768 it is cut
late in the same chain. `flexible-extract` needs a `\boxed{}` that never arrives
either way.

So the recorded prediction — "flexible-extract should rise well above 0.30 if
truncation was the cause" — is answered **no**, and the fallback branch of that
prediction now applies: truncation was not the whole story.

## Hypotheses, and what has been ruled out by inspection

Three candidate causes:

- **A. stop tokens are generated but not honoured.**
- **B. degenerate repetition under greedy decoding.**
- **C. genuinely long reasoning that does not fit 32,768.**

Ruled out by reading the code and config, so that the probe does not re-litigate
them:

1. **The EOS ids are configured.** The snapshot's `generation_config.json`
   declares `eos_token_id: [248046, 248044]`. vLLM honours a list.
2. **The port is right not to handle EOS itself.** There are **zero** references to
   `eos`, `im_end`, `stop_token` or `finish_reason` anywhere in
   `models/autoports/qwen_qwen3_6_27b/tt/`. That is correct: vLLM's engine owns
   stop-token detection; the model returns logits. Absence here is not a defect.
3. **Padded vocabulary columns cannot win the selection.** This was my leading
   hypothesis and it is wrong. Both pad sites in
   `models/common/sampling/tt_sampling.py` mask correctly — the chunked path pads
   with `torch.finfo(torch.bfloat16).min` and the single-shot path with
   `-sys.float_info.max`, and the index tensor pads with `-1`. The chunk-base
   restoration `ttnn.add(chunk_relative_indices, i * split_width, dtype=uint16)`
   also stays in range: the largest valid local ID is 62,079 against a uint16
   ceiling of 65,535. `mask_invalid_vocab` defaults to `False`, but it is a
   belt-and-braces mask that is not needed given the pad values.

So A is unlikely at the config level, though it remains possible at the
integration level, and the remaining live candidates are B and C.

## The gap this exposes, which is the real finding

**No test on this branch checks that generation terminates naturally.**

Every generation test fixes the token count in advance: `--gen-len 100`,
`--decode-tokens 128`, "100 HF and 100 TT tokens, coherent". The readiness and
correctness evidence is all *teacher-forced or fixed-length*: prefill PCC at
S=33/S=161, decode PCC at fixed positions, replayed decode bit-exactness, top-1/
top-5 agreement over 100 tokens. Every one of those can pass while the model never
emits EOS, because none of them ever waits for it.

That is why an 11-stage pipeline with strong correctness evidence shipped a model
whose first real open-ended generation ran to a 32,768-token cap. The missing check
is trivial to state: *given a prompt with an obviously short answer, does
generation stop, and at how many tokens?*

This is a pipeline gap, not only a model bug. It belongs with the other
grading/enforcement defects already recorded: the eval-catalog under-onboarding
that made zero standard evals a successful no-op, and the structurally unmatchable
`strict-match` regex.

## The probe

`/tmp/probe_stop.py` distinguishes the remaining hypotheses in minutes rather than
hours, from one server start:

| probe | budget | what it decides |
|---|---:|---|
| "What is 2 + 2? Answer with just the number." | 64 | if this hits the cap, stop tokens are not honoured (A) |
| "Name the capital of France in one word." | 64 | same, second sample |
| GPQA-style physics multiple choice | 4096 | 12-gram repetition rate separates a greedy loop (B) from long reasoning (C) |

It records `finish_reason`, `completion_tokens` against the cap, whether the text
ends in terminal punctuation, whether it contains `boxed`, and the most-repeated
12-gram with its count.

---

## The cause: this is a thinking model, and the template leaves `<think>` open

Found by reading the tokenizer's chat template. This explains the 256-token result
exactly and predicts the 32,768-token result.

### The template

`tokenizer_config.json`'s chat template contains `enable_thinking`, and its
generation-prompt tail is:

```jinja
{%- if enable_thinking is defined and enable_thinking is false %}
{{- '<think>\n\n</think>\n\n' }}      {# closed empty block: answer directly #}
{%- else %}
{{- '<think>\n' }}                    {# DEFAULT: an OPEN think tag #}
```

So unless `enable_thinking=false` is passed explicitly, **every prompt ends with an
unclosed `<think>`**, and the model is expected to reason, emit `</think>`, and only
then produce the answer.

lm-eval's `--apply_chat_template` renders the template with no `enable_thinking`
argument. So the default branch applied to **every graded response on this branch**.

### Why this explains the observations exactly

The retained 256-token samples showed: median 120 words, **0/20 containing
`boxed`**, 16/20 filtered to `[invalid]`, tails severed mid-expression
(`$\Gamma_2 \approx \frac{\hbar}{`). That is precisely what a reasoning chain cut at
256 tokens looks like — the entire budget was spent inside the `<think>` block, so
the `\boxed{}` the prompt asked for was never reached. The extractor was not
failing; there was nothing yet to extract.

It also predicts the 32,768 result: document 1 consumed the whole budget, meaning
`</think>` had still not arrived. Whether that is a very long chain or a loop is
what the two-arm probe measures.

### A second, independent configuration gap

The server starts with `reasoning_parser=''` (visible in its own startup config
dump). With no reasoning parser, the whole think block is returned in `content`
rather than split into `reasoning_content`. For the `boxed_choice` filter this is
survivable — it will find `\boxed{}` wherever it sits — but it means the graded
"answer" text is mostly reasoning, and any metric that inspects response shape,
length, or formatting is measuring the think block. IFEval's instruction checks are
exactly such metrics, which is worth remembering when reading its 15/43.

### Two legitimate configurations; the pipeline chose neither

- **Thinking ON with an adequate budget.** This is how a reasoning model's GPQA
  number is normally produced, and is the comparable condition for a model-card
  reference. It requires the model to actually close `</think>`, which is exactly
  what is in doubt.
- **Thinking OFF (`chat_template_kwargs: {"enable_thinking": false}`).** The
  template pre-closes an empty block, the model answers directly, and a modest
  budget suffices. Terminating and cheap, but a weaker and different benchmark
  condition, so it is not interchangeable with a published thinking-mode score.

What actually happened was neither: thinking mode was inherited by default and
combined with lm-eval's implicit 256-token cap. Nothing in the pipeline chose that,
and nothing detected it.

### Why no stage caught it

Because, as recorded in this document, **no test on this branch waits for
generation to end.** Fixed-length tests (`--gen-len 100`, `--decode-tokens 128`)
cannot distinguish a model that is mid-reasoning from one that has answered, and
teacher-forced top-1 agreement is measured against HF running in the same thinking
mode — so it agrees, correctly, about reasoning tokens. Every piece of correctness
evidence on this branch is compatible with a model that never closes `</think>`.

### Pending measurement

`tests/thinking_mode_probe.py` runs two arms per prompt, everything else identical:

| arm | budget | decides |
|---|---:|---|
| trivial, thinking OFF | 128 | do stop tokens work at all? |
| trivial, thinking ON | 4096 | does a trivial question still trigger long reasoning? |
| GPQA-style, thinking OFF | 1024 | does it answer, with `\boxed{}`, and terminate? |
| GPQA-style, thinking ON | 8192 | where does `</think>` land, and is there repetition? |

Reads: **thinking OFF terminates** → stop handling is sound and this is a
serving-configuration defect, fixable in the eval invocation. **thinking OFF also
runs to the cap** → the defect is in the port, not the configuration, and the
`</think>` character index plus the 12-gram repetition rate distinguish a
degenerate loop from genuinely long reasoning.

---

## Correction: the model DOES stop. Withdrawing this document's headline claim

Measured 2026-08-17, immediately after the sections above were written. The title
claim — "the model does not stop" — is **too strong and is withdrawn as a general
statement.** What the probe found:

### Probe results, thinking mode ON (the default, i.e. the graded condition)

| prompt | budget | result |
|---|---:|---|
| "What is 2 + 2? Answer with just the number." | 64 | hit cap, `finish_reason=length` |
| "Name the capital of France in one word." | 64 | hit cap, `finish_reason=length` |
| easy physics multiple choice, same `doc_to_text` form | 4096 | **stopped at 1362 tokens** |

The easy physics item is decisive:

```
finish_reason      : stop
completion_tokens  : 1362 / 4096
ends terminal punct: True
contains 'boxed'   : True
12-gram repetition : 0.0%
tail: "...The calculated ground state energy matches option (A).\n\n\boxed{A}"
```

It reasoned, converged, emitted `\boxed{A}` — **the correct answer** — and stopped on
its own. So:

- **stop tokens are honoured.** vLLM saw an EOS and ended the request.
- **the model terminates** on an item of exactly the graded form.
- **there is no degenerate loop** here: 0% 12-gram repetition, coherent prose.

The two trivial prompts hitting a 64-token cap are fully explained by thinking mode:
their output is coherent reasoning preamble ("Thinking Process: 1. **Identify the
user's core question:** ...") at 0% repetition. A 64-token budget simply cannot hold
a reasoning preamble plus an answer. That observation licenses no conclusion about
stop-token handling, and the probe's own verdict logic — which printed "stop tokens
are NOT honoured" — was confounded and has been fixed.

### What survives, as a much narrower question

Still true and still unexplained: **real GPQA Diamond document 1 consumed the entire
32,768-token budget** (1,887.95 s at 17.8 tok/s). That is no longer "the model cannot
stop". It is:

> Why does a real Diamond item run past 32,768 tokens when an easy item of the same
> prompt form converges in 1,362?

Three candidates, in the order I would test them:

1. **Hard items produce genuinely non-convergent reasoning.** Diamond is
   PhD-level; a thinking model can spiral without reaching `</think>`. A model
   property, not a port defect — but it makes thinking-mode GPQA unusable at any
   budget the harness would plausibly grant.
2. **Degenerate looping specific to hard items.** Distinguished from (1) by the
   12-gram repetition rate, which was 0% on the easy item.
3. **Something in the lm-eval path my direct API call does not reproduce.** The
   probe posts `messages` and lets the server render the template once; lm-eval with
   `--apply_chat_template` renders client-side and then hands the result to the same
   chat endpoint. The eval config's own comment asserts "the OpenAI server still
   performs the only token rendering", but that is an assertion to verify, not a
   fact I have checked. A doubly-applied template would produce a malformed prompt —
   plausibly two nested `<think>` openings — which is exactly the kind of thing that
   would prevent convergence.

### The extended probe now queued

It uses the **actual Diamond rows**, built with the task's verbatim `doc_to_text`.
Row 0 is almost certainly the graded document 1: it is the two-lifetimes
energy-resolution physics item ("Two quantum states with energies E1 and E2 have a
lifetime of 10^-9 sec and 10^-8 sec"), and the retained 256-token sample was severed
mid-expression at `$\Gamma_2 \approx \frac{\hbar}{` — the linewidth calculation for
precisely that question.

| case | budget | decides |
|---|---:|---|
| trivial, thinking OFF | 128 | does the OFF arm terminate at all |
| easy physics, thinking OFF | 1024 | OFF arm on a known-good item |
| **Diamond row 0, thinking OFF** | 4096 | can it answer the hard item at all |
| **Diamond row 0, thinking ON** | 16384 | reproduce the runaway, measure repetition |
| Diamond row 1 (organic chem), thinking ON | 16384 | is it domain-specific |

It records `</think>`'s character index, the extracted boxed letter against the gold
answer, and the repetition rate, and checkpoints its JSON after every case so a
long-running arm cannot lose the earlier results.

### Why the correction matters beyond bookkeeping

The withdrawn claim would have pointed the next reader at the sampler, the EOS
configuration, or the port's decode path — and I had already spent time ruling
those out. The evidence says the port's generation and stop handling are working.
The remaining problem is about how hard items interact with thinking mode and with
the harness, which is a different investigation with different owners.

---

## Hypothesis 3 eliminated: lm-eval does not double-apply the chat template

Checked in source rather than assumed, because the eval config only *asserted* it.

`lm_eval/models/api_models.py:442`:

```python
def apply_chat_template(self, chat_history, add_generation_prompt=True):
    if self.tokenizer_backend == "huggingface" and self.tokenized_requests:
        return self.tokenizer.apply_chat_template(          # would render to a string
            chat_history, tokenize=False,
            add_generation_prompt=add_generation_prompt, ...)
    elif self.tokenizer_backend == "remote" and self.tokenized_requests:
        return chat_history
    else:
        # bit of a hack. We'll load back before sending to the API
        return JsonChatStr(json.dumps(
            [{**item, "type": "text"} for item in chat_history], ensure_ascii=False))
```

The invocation used `tokenizer_backend=huggingface`, so the first branch depends on
`tokenized_requests`. The eval log settles it:

```
[models.api_models:955] Tokenized requests are disabled. Context + generation length is not checked.
```

`tokenized_requests` is **False**, so the `else` branch runs: the chat history is
passed through as message dicts and the **server performs the only template
rendering**. There is no client-side render, therefore no nesting of `<think>`
openings, and the eval config's comment — "the OpenAI server still performs the only
token rendering" — is correct.

This matters for interpretation: my direct-API probe posts `messages` to the same
chat endpoint, so it renders **identically** to the graded path. The probe is a valid
reproduction, and the difference between the easy item converging at 1,362 tokens and
Diamond document 1 exceeding 32,768 is therefore a property of **the question**, not
of the harness.

Remaining hypotheses, now two:

1. genuinely non-convergent reasoning on PhD-level items;
2. degenerate looping specific to hard items.

Both are measured by the queued Diamond-row probe, and they are separated by the
12-gram repetition rate — 0% on the easy item that converged.

Incidental note from the same log line: with tokenised requests disabled, lm-eval
does **not** check that prompt plus generation fits the context window. Harmless at
this model's 262,144-token context, but worth knowing for a model where it would not
be.

---

## Practical consequence: thinking-mode GPQA is barely runnable at this decode speed

Independent of whether the reasoning converges, the arithmetic is worth stating,
because it connects the accuracy investigation to the performance work.

At the measured **56 ms/token** (17.8 tok/s, matching the recorded ITL) and a
32,768-token budget:

| scope | sequential wall clock |
|---|---:|
| one document at the cap | **~31 min** (measured: 1,887.95 s) |
| the 5% CI subset, 10 documents | **~5.2 h** |
| the full Diamond set, 198 documents | **~102 h** |

So even the CI subset costs a fifth of a day per attempt, and the full set is not
runnable. Any iteration on thinking-mode GPQA — a precision change, a policy change,
a re-measurement — pays that cost each time. That is a serious obstacle to the very
reference baseline `AUTOFIX.md` says is required.

### Two ways to make it tractable, both already evidenced on this branch

1. **Run the eval concurrently.** The evals ran at `num_concurrent=1` against a
   `max_num_seqs=1` server, so exactly one slot was ever busy. The recorded burst
   profile shows `ITL P50 244.0 ms` at `max_num_seqs=32` with 32 requests in flight —
   i.e. **7.6 ms/token/user**. Ten documents served concurrently would cost roughly
   `32,768 x 244 ms ~= 2.2 h` for the whole subset instead of 5.2 h sequentially,
   because the ten generations overlap. This needs no model change, only
   `--max_num_seqs 32` on the server and `num_concurrent=10` in the invocation.

   Caveat that must be respected: this changes the serving configuration, so it is
   valid for *measuring accuracy* but the resulting latency numbers are not the
   single-user ones. Keep the two purposes separate.

2. **Disable thinking for the graded run.** `chat_template_kwargs:
   {"enable_thinking": false}` collapses the budget requirement to hundreds of tokens.
   This produces a different and weaker benchmark condition, not comparable to a
   published thinking-mode score, so it is a legitimate configuration only if it is
   labelled as such.

### Why an HF reference was not attempted here

`AUTOFIX.md` identifies the missing input as "matched Qwen GPU/control baselines". A
CPU reference on this host is not a sensible substitute: 249 GB total RAM but only
**61 GB available** with the device server resident, against ~54 GB for 27B in bf16,
on 16 cores. Even one document would take hours while competing for memory with the
TT server, and it would still not be the GPU reference the config wants. Recorded as
a follow-up for a GPU host rather than attempted badly here.

The comparison that reference would settle is specific and worth stating so it can be
run cheaply when a GPU is available: **does HF, in thinking mode, converge on Diamond
row 0 within 32,768 tokens?** If it does and this port does not, the divergence is
port-side and the numerical policy is implicated. If neither converges, it is a model
property and the benchmark configuration is what needs to change.
