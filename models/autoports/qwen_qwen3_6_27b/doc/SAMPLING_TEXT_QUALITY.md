# r1_gpqa_diamond result: 0.60, and a text-quality defect the score hides

Measured 2026-08-17. Run: `tests/run_r1_gpqa.sh 0.05 32768` against the autoport,
`--reasoning_parser qwen3` active, release sampling.

## The headline number

```
r1_gpqa_diamond   exact_match,none   0.60   stderr 0.163   n-samples effective 10
effective generation_kwargs: do_sample True, max_gen_toks 32768, temperature 1.0,
                             top_k 20, top_p 0.95, n 1, stream False,
                             until ['<|end_of_text|>', '<|endoftext|>', '<|im_end|>']
```

Against the same port on the wrong task variant — `gpqa_diamond_cot_zeroshot` at
lm-eval's implicit 256-token cap with greedy decoding — which recorded
**`exact_match,flexible-extract` 0.30**, one document above four-choice chance.

So changing only the task configuration doubled the score, from chance to 0.60, with
no model change. Reference for scale: the model card publishes **87.8** for GPQA
Diamond, and with no GPU baseline the accuracy check's bar is `87.8 * 0.95 = 83.41%`.
With 10 documents the stderr is **0.163**, so 0.60 has a very wide interval and the
sibling entry's own note applies: under `--ci-mode` the subset is compared against the
full-set number and the CI subset is harder than average, so low readings are expected.

## The score is not the important finding

`</think>` appears in **10/10** responses, which confirms the prediction recorded
before the run (the chat template's multi-turn branch splits on `'</think>'`, so the
model must produce it). Good. But the per-sample text is badly degraded.

| doc | score | words | tail |
|---:|---:|---:|---|
| 0 | 0 | 286 | `### **Step1: CalculateCalcul The core challenge  </think>` |
| 1 | **1** | 4195 | `**R**: **(CH)** **(C) 12} **  **Let:** The**( : **(CH3)**` |
| 2 | 1 | 6457 | `lation yields $\approx -0.67$. Closest option is **(A) -0.**` |
| 3 | 0 | **6** | `</think>    Let's start the thinking process` |
| 4 | 1 | 3816 | `</think>  The correct answer is \boxed{(A)}` |
| 5 | 0 | 468 | `The correct answer must be]( </think>  ** ### ** ###  ###` |
| 6 | **1** | 218 | `**Option **( ** </think>  </think>  Let` |
| 7 | **1** | 13559 | `** Let's Let's ** **: ** Let's Let's Let's ** ** Let's` |
| 8 | 0 | 392 | `(O) triisopropyl is a D** **( **O****  **   # D. **O****` |
| 9 | **1** | 4274 | `* **  ** Let  **. **.**. ** Let's  R  # a  **  ###  . ** **.` |

Only **doc 4** is well formed end to end: `...</think>  The correct answer is
\boxed{(A)}`. Doc 2 is close. Of the six documents that scored 1, **four contain no
`boxed` at all** (1, 6, 7, 9) — so `process_results_gpqa` extracted a letter from
degraded text. With four choices, that is substantially luck. **The 0.60 should not be
read as 60% of questions genuinely answered.**

Doc 3's entire response is 45 characters:

```
\n</think>\n\n\n\nLet's start the thinking process
```

The model closed the think block on its first token, emitted a stub, and stopped.

## The degradation is progressive, and starts clean

Doc 0 in full shows the shape. It opens correctly:

> "To determine which energy difference would allow the two quantum states to be
> "clearly resolved" (or distinguished), we must look at the relationship between
> lifetime and energy uncertainty."

then degrades:

> "According quantum(Heisen Uncertainty Principle)" ... "Lifetime of state state
> state 1" ... "$\Delta E \approx \10^{-9}$s andes" ... "smallerther than the
> naturallinewidth" ... "To \"clearlyy\" distingdistinguish" ... "the peaksopt the
> spectrum"

Fluent at the start, corrupted by the end. Note the *kind* of corruption: duplicated
fragments (`state state state`, `CalculateCalcul`, `distingdistinguish`,
`Let's Let's Let's`), missing spaces (`naturallinewidth`, `peaksopt`, `smallerther`),
and mangled LaTeX (`\10^{-9}`, `\t10^{-16}`).

## Two candidate mechanisms; I am not asserting which

**A. Wrong logits, exposed by sampling.** Greedy decoding takes an argmax and is
robust to small logit errors; sampling at `temperature 1.0, top_k 20, top_p 0.95`
draws from the distribution, so errors in the non-top-1 tail become wrong token
choices. This port's recorded top-1 agreement with HF is 97%, i.e. 3% of tokens
already differ under greedy, and nothing has ever measured the *distribution* beyond
top-1/top-5. This would explain why greedy output was fluent (but looped) while
sampled output is garbled.

**B. Incremental detokenization / text assembly.** The specific corruption —
duplicated substrings, missing spaces at token joins, merged words — is more
characteristic of a text-assembly bug than of semantically wrong token choices.
Choosing a wrong token normally yields a *well-formed but wrong* word, not
`naturallinewidth`.

Evidence pulls both ways, which is why I am recording both. Doc 3 (immediate
`</think>` then stop) and the greedy repetition loop measured earlier
(`110^{-64}` x 1241) are token-level behaviours that B cannot explain; the
space-and-merge pattern is one that A explains poorly.

### The discriminating test

Compare, for one prompt, the **token ids** the server returns against a reference
decode, rather than the detokenized string:

1. request with `logprobs` / token ids enabled and capture the raw id sequence;
2. detokenize that id sequence offline with the HF tokenizer in one shot;
3. compare against the streamed/assembled `content`.

If the one-shot detokenization of the same ids is clean, the defect is **B**
(assembly). If the ids themselves decode to garbage, the defect is **A** (logits or
sampling), and the next step is comparing the sampled distribution against HF for a
fixed prefix.

This has not been run. It is cheap — one request — and it is the right next
experiment.

## Why this matters for the release

The release `gen_kwargs` for this family set exactly this sampling
(`do_sample=true, temperature=1.0, top_k=20, top_p=0.95`). So the release will
exercise the configuration that produces this text, and it will report a
letter-extraction score that looks plausible while the underlying generations are
degraded. A reviewer reading only `exact_match,none` would not see it.

That also reframes the greedy finding. Greedy is not simply "the wrong setting": it is
the setting under which this port produces *fluent* text, and its failure mode
(looping on hard items) is more visible than the sampled failure mode (quiet
degradation). Both are unacceptable for a release; they are different defects and the
sampled one had not been measured until now.

## What remains sound

Nothing about the task-configuration analysis is retracted. `cot_zeroshot` is still
the wrong variant for four independent measured reasons, and `r1_gpqa_diamond` is
still the right one. The 0.30 -> 0.60 improvement is real as a *scored* result. What
is new is that the port has a text-quality problem under sampling that the score does
not reveal, and that problem is now the most serious open item on this port —
more serious than the prefill performance work, because it affects output correctness
in the configuration the release will use.
