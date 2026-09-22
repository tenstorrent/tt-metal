# Full-model qualitative review — draft, more work needed

Date: 2026-09-12. Independent CPU/read-only assessment under the qualitative-check skill. No TTNN imports, model execution, hardware access, or implementation changes by this reviewer.

**The 256-token run supports coherent completed answers for p1 and p4 and finds no mechanical degeneration. It does not support a clean full-suite qualitative pass.** Prompts p0, p2 and p3 are capped on both implementations; p5 completes on HF but is capped before the TT function is complete. Matched extensions are required to distinguish insufficient generation budget from a TT convergence problem. The coordinator has agreed to matched 1024-token HF/TT runs for p0, p2, p3 and p5; those results are pending this draft.

## Control and prompt integrity

Reviewed artifacts: `tt_qualitative.json` and `hf_qualitative_256.json`. Both identify `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, Qwen2Tokenizer, chat mode with a chat template, greedy generation and a maximum of 256 new tokens. The shared source is `models/common/readiness_check/vllm_prompts.txt`, SHA256 `2ad452c15d8442d3fe641eb968bbd84e3bfe9acb01ddc540f451983c0da3cbbd`; the on-disk source matches that hash.

A host comparison verified identical prompt strings, rendered chat strings and exact prompt token IDs for all six HF/TT pairs. Each rendered prompt contains the same xhigh reasoning system message and ends in an assistant `<think>` prefix. Thus reasoning prose and the subsequent `</think>` marker are expected in these raw decoded artifacts. Both HF and TT also retain the terminating `<|im_end|>` token in completed text. These matching control markers do not establish unexpected serving/UI leakage. Repeating the user's request in the reasoning preamble is also present on HF; p2 repeating the story fragment in its final answer is natural for its completion request.

The HF runner uses a left-padded batch of six; the TT runner submits each exact unpadded prompt through the full model and common split sampler. Both artifacts trim their displayed text/token arrays through the first configured EOS. TT generation itself executes the fixed requested length, so its performance counters can report 256 generated tokens even where the saved first-EOS completion is shorter. The counts below are saved completion lengths, not performance counters.

## Per-prompt findings

`think end` is the zero-based generated-token index of `</think>`; an absent marker means no final-answer section was reached within the cap.

| Prompt | TT / HF tokens | TT / HF think end | Concrete comparison | Current verdict |
| --- | --- | --- | --- | --- |
| p0: machine-learning haiku | 256 / 256, neither EOS | absent / absent | HF settles on the candidate “Data flows in light / models learn from patterns deep / answers emerge now” and rechecks its syllables. TT has valid first two candidate lines but keeps trying four-syllable last lines, including “Patterns appear,” “A pattern forms,” and “A pattern grows.” | No delivered final haiku on either. TT's slower convergence is a real visible difference requiring extension; the present trace does not prove an infinite loop or stale feedback. |
| p1: supervised vs unsupervised learning | 148 / 133, both EOS | 31 / 31 | TT explains learning from labeled examples versus finding groups without answers, using cats and photo grouping. HF gives the same distinction with cats/dogs and customer groups. | Coherent, complete, comparable answers. Different wording/examples are acceptable. |
| p2: inventor story | 256 / 256, neither EOS | 185 / 61 | TT starts a coherent brass-key story, then stops at “The people of the kingdom were amazed.” HF develops a brass key/music-box story and is cut off at “El”. TT spends 124 more tokens before closing its reasoning section, leaving much less story at this cap. | Both stories incomplete. TT prose is coherent where visible, but completion and sustained quality remain unverified. |
| p3: three laws of thermodynamics | 256 / 256, neither EOS | 187 / 184 | Both discuss the usual first/second/third-law interpretation in reasoning, then begin a correctly framed first-law answer with `Delta U = Q - W`. Both stop during the first-law explanation, before a delivered second/third-law answer. | Comparable cap truncation. The reasoning outlines are not a completed answer to the user's three-law request. |
| p4: French translation | 145 / 113, both EOS | 134 / 102 | TT: “Bonjour, comment allez-vous aujourd'hui ?” HF gives the same polite French translation with a typographic apostrophe. | Coherent, complete, correct-language translation. The English reasoning preamble is explained by the matched control; the final translation is French. |
| p5: Fibonacci function | 256 / 237; TT capped, HF EOS | 163 / 81 | HF supplies a complete iterative function returning the first n Fibonacci numbers and the correct example `[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]`. TT agrees on first-n semantics, emits checks for n<=0 and n==1, and reaches `fib.append(1)` before the cap; its loop and final return are absent. | Material incompleteness at the current service budget. TT consumes 82 more tokens before closing reasoning. This supports a budget/convergence investigation, not a claim that the completed TT algorithm is wrong. |

The p0 last-line alternatives are different strings with coherent syllable discussion. They merit a targeted convergence check, but do not display the nearly-every-word duplication that identifies the shared checker's stale-input failure signature. Conversely, the fact that HF also reasons about syllables does not excuse TT forever cycling without producing a haiku. The extension must establish what happens next.

For p5, TT has only 92 tokens after `</think>` within the 256-token cap, compared with HF's 155 through its EOS. Its unfinished function cannot be presented as a working final answer. Do not claim a syntax failure merely from the open code fence: the observable defect is missing requested computation/return, caused here by a capped output whose continuation has not yet been examined.

## Shared mechanical-degeneration check

The reviewer imported `models/common/readiness_check/check_degenerate_output.py` directly by file path and called its `check_completion` function once for each of the six saved TT and six saved HF completions, passing both text and token IDs. With nonempty text, the checker computes word-based metrics. No Torch, TTNN, model or device module was imported.

`qualitative_degeneracy.json` records 12 measurements, exit code 0, and no critical or advisory findings. Maximum adjacent-word duplication is 0.0129 for TT and 0.0161 for HF; maximum trigram-loop fraction is 0.125 for TT and 0.1169 for HF. For p0 specifically, adjacent duplication is zero on both, and trigram-loop fractions are 0.0667 TT versus 0.0647 HF. These are below the checker's thresholds.

This checker assesses mechanical duplication, looping and near-empty text. It does not assess syllable accuracy, convergence to a final answer, completeness of code, factual quality, or whether a generation limit is sufficient. Its clean result cannot convert this draft's incomplete qualitative verdict into a pass.

## Required matched extension

Run p0, p2, p3 and p5 with a 1024-token cap on both HF and the full TT model, preserving the exact rendered messages, prompt IDs/token IDs, model revision and greedy sampling settings. Preserve the current six-prompt 256-token artifacts separately; the current TT helper otherwise overwrites `tt_qualitative.json`. Completed p1/p4 do not need another run just to raise the cap. A 512-token probe could answer some questions, but 1024 provides one larger controlled window for all four presently censored prompts. A cap of 1024 is an experiment, not a promise that every story must terminate by then.

Check each extension through its first EOS, record its first closing-think/EOS indices, and verify whether its initial 256-token prefix matches the corresponding old run. Prefix stability helps attribute a changed verdict to the larger budget; if batching or numerical changes alter the prefix, retain that distinction rather than calling it an exact continuation.

- p0 must progress from candidate checking to a delivered haiku; inspect the actual final three lines. If HF finishes while TT repeatedly cycles through near-identical failed syllable choices, classify that as unresolved TT quality regression and investigate with the output evidence.
- p2 should sustain coherent entities/events and provide an ending if it terminates. If both remain coherent but capped, report the limited continuation evidence honestly; do not claim story completion.
- p3 needs the delivered first, second and third laws, with consistent first-law sign convention and appropriate conditions for the third law.
- p5 needs a complete function with a loop/recurrence and final return for general n, plus sensible n=0/1 handling. Compare correctness of the completed output, not token identity with HF.

Rerun the shared checker on the extended outputs and update this review with concrete outcomes. If TT remains materially less complete than HF at the matched larger budget, keep the stage qualitative verdict open. No implementation change is justified solely by early free-running token divergence in these 256-token samples.

## Reviewed artifact identities

```text
tt_qualitative.json       bbb345afcba7797e5e1c735ed4c034a4e3e9b58e72d9422070aaed9c3f34ae98
hf_qualitative_256.json   ea77402b6404022c1bed0fe041aca374409dd7326d387a378f0d99d10f5a671b
check_degenerate_output.py 2629d86ae7be926a21e4839d151d21037e0851e98266f1b9a100e7cf14541db4
```

This is a draft assessment of these exact saved outputs. The coordinator's hardware exit status establishes successful execution of that run; it is not substituted for the pending qualitative completion checks.
