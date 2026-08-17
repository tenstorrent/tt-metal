# What this release actually establishes, and what it does not

Operator analysis, 2026-08-17. Written from committed artifacts on this branch.

## Status: all stages complete, `release-readiness-ci-subset-pass`

This is the only one of the three ports run through this pipeline that reached a
passing release. Nothing blocks it. What follows is about the *strength* of that
pass, because two things are weaker than the headline suggests.

## Is the PCC good? Yes

| gate | value |
|---|---:|
| AIME24 prefill top-1 / top-5 / top-100 | 92 % / 100 % / 100 % |
| AIME24 teacher-forcing top-1 / top-5 / top-100 | 93 % / 100 % / 100 % |
| multichip layer PCC | 0.999851 – 0.99999994 |
| max context, 32,767 prompt + decode | pass, all 1,024 pages and 224 K/V tensors checked |
| sampled key / value cache PCC vs HF | 0.996441 / 0.995348 |

Decode reached **110.4 t/s/u** standalone and serving overhead is closed. The
port is faithful on every gate it was measured against.

## Weakness 1: the passing eval rows rest on one document

Both mandatory evals passed against **`CI_NIGHTLY` subsets**, and the margins are
single documents:

| eval | TT | HF control | subset | ratio |
|---|---:|---:|---|---:|
| `meta_ifeval` | 21.43 % (6/28) | 17.86 % (5/28) | 28 of 541 | 1.2 |
| `meta_gpqa_cot` | 60 % (6/10) | 50 % (5/10) | 10 of 198 | 1.2 |

A ratio of 1.2 computed from 6/28 versus 5/28 is **one document of difference**,
with stderr ±7.4 and ±16.3. That is enough to refute a catastrophic regression
and nowhere near enough to certify parity. The stage was right to claim
`ci-subset-pass` rather than full-set readiness. A full-set CPU control is ~45 h
(the 28-sample IFEval control alone took 2 h 15 m); a GPU would make it cheap.

The IFEval row is also the one that taught this fleet a lesson worth keeping:
its first grading compared TT's `prompt_level_strict` against a model-card 34.3
whose variant the card never states, producing an apparent **0.544 ratio and a
quality FAIL**. The HF control's `inst_level_loose` came out at 34.88 — nearly
exactly the card's number — proving the card publishes an instruction-level
figure and the port was at parity all along. See
`doc/tti_release/IFEVAL_VARIANT_ANALYSIS.md`.

## Weakness 2: quality was never measured at length

| what | length |
|---|---:|
| longest correctness-measured generation | **100 tokens** (`--gen-len 100`, `--max-new-tokens 100`) |
| reference-checked prompt | ~155–161 tokens |
| max context validated (capacity) | 32,767 + decode |
| served `max_model_len` | 32,768 |

The 32,767-token evidence is a **capacity** result — the prefill fits, all pages
and K/V tensors are populated, decode advances. It is not a sustained
autoregressive generation with output checked against a reference. So on this
branch, quality is established at ~100 generated tokens and capacity at 32,768,
with nothing in between.

**Why to care here, not just in principle:** the sibling Gemma-4-26B port on
`mvasiljevic/fmf/google-gemma-4-26b-a4b-it` has *better* layer PCC than this one,
passes at 100- and 1,280-token generations, and then **fails `meta_gpqa_cot` 4/10
against an HF control's 10/10 at a 32,768-token generation budget**, with the
divergence narrowed to its long-decode numerical path. This branch's evidence
would not have detected that failure mode: its own GPQA control ran at
`max_gen_toks` far below the regime, on 10 documents.

Falcon3-7B is a dense GQA model with a 32 K context, so it carries less of the
risk that hurt Gemma (no 128-expert routing, no 1024-token sliding window to wrap
32 times). But "not obviously exposed" is not "measured".

## Recommended, in priority order

1. **A long-generation teacher-forcing check.** Generate 4,096 tokens against the
   HF reference and record first-divergence index and PCC at 256/512/1024/2048/
   4096. The existing teacher-forcing path already reports per-position top-k
   agreement; it only needs to run longer than 100 tokens. No GPU needed.
2. **Full-set eval references** if this port is to be quoted as release-ready
   rather than CI-subset-ready — ~45 h on CPU, cheap on a GPU.
3. **Model-scope two shared-infra edits** before this branch approaches main: the
   `--plugin-config` → `--additional-config` rename (correct, but now duplicated
   on three branches and belonging upstream) and the Falcon-specific 37-token
   non-aligned check hard-coded into the shared qualitative path with
   `"status": "pass"` written unconditionally and no error handling.
