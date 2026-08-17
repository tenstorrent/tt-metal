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
