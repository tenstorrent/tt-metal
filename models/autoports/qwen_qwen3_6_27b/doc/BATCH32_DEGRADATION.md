# Isolation experiments: the batch-32 decode path degrades long generations

Measured 2026-08-18. Two arms, each differing from the known-good hand-rolled config in
exactly one way (`FABRIC_1D_RING`, 200 MB trace, `reasoning_parser qwen3`, thinking
disabled, temperature 0 held constant throughout).

## Result: a clean 2x2

Same prompt (real GPQA Diamond row 0), same sampling, same everything except two release
variables:

| | `sample_on_device_mode: all` | `sample_on_device_mode: decode_only` |
|---|---|---|
| **`max_num_seqs = 1`** | **CLEAN** — 1849 tok, `\boxed{A}` correct, 0% rep *(hand-rolled baseline)* | **CLEAN** — 1849 tok, `\boxed{A}` correct, 0% rep *(arm A)* |
| **`max_num_seqs = 32`** | **BROKEN** — 4096 tok (cap), 12.6% rep, no answer *(arm B)* | **BROKEN** — completed docs 1-196 words, corrupt from char 1 *(CI-faithful)* |

**`sample_on_device_mode` is exonerated.** Arm A (`max_num_seqs=1` + `decode_only`)
reproduced the baseline exactly — 1849 completion tokens, the correct boxed answer, zero
12-gram repetition, and a clean opening:

> "To determine the minimum energy difference required to clearly distinguish (resolve)
> two quantum states, we use the Heisenberg Uncertainty Principle."

**`max_num_seqs = 32` is the discriminator.** It is broken under *both* sampling modes;
batch 1 is clean under both.

## Scope: only the long/hard input

Every arm answered the short prompts correctly:

| prompt | arm A (B1, decode_only) | arm B (B32, all) |
|---|---|---|
| "What is 2 + 2?" | 2 tok, `4`, stop | 2 tok, `4`, stop |
| easy oscillator MCQ | 472 tok, `\boxed{A}` correct | 484 tok, `\boxed{A}` correct |
| **Diamond row 0** | **1849 tok, `\boxed{A}` correct** | **4096 tok (cap), 12.6% rep, no answer** |

So the defect needs length — or difficulty — to appear. A 472-token generation is fine at
batch 32; an 1800+ token one is not. That is consistent with state that drifts or is
mismanaged as a sequence progresses, rather than with a static misconfiguration.

## A metric bug of mine, corrected

The probe's `baseline_words` values (2 / 472 / 1849) were taken from the earlier probe's
**completion-token** counts, not word counts. Every "ratio ~0.5" and every
`WORD-COUNT COLLAPSE` flag in the raw output is therefore spurious — it is just the normal
words-per-token ratio. Comparing tokens to tokens, arm A matches the baseline to the
token (1849 vs 1849, 472 vs 472, 2 vs 2). The flags that matter are the repetition rate,
the boxed answer, and `finish_reason`, all of which are unaffected by the mistake.

## What this means

The release spec serves at `max_concurrency: 32`. On this port that configuration:

1. costs ~270 ms/token instead of ~56 ms, because decode cost follows the allocated batch
   rather than the active rows (`SERVING_BATCH_LATENCY.md`);
2. **and degrades long generations** — repetition, failure to converge, and in the
   CI-faithful run outright corruption from the first token.

(1) was already documented as a latency and eval-runtime problem. (2) is new and more
serious: it means the batch-32 path is not merely slow but produces materially worse
output on exactly the long-reasoning workloads this model is for. Both of the reported
scores — the 0.10 from the CI-faithful run and its true 0/10 — are consequences of running
at batch 32, not of the model's ability. The hand-rolled 0.60 at batch 1 is the closer
estimate of what this port can do, and even that is depressed by the sampled-text
degradation documented separately.

Worth restating: this is **at** vLLM pin `03fa3af2e` ("Fixed state slot bugs (#466)"),
whose parent carries `#468 "release a preempted request's device state slot"`. Both
upstream state-slot fixes are present. Whatever remains is either not covered by them or
lives in the autoport's own batch handling.

## Why the autoport's own code is now the leading suspect

`SERVING_BATCH_LATENCY.md` established that this port captures **one** decode trace at a
**fixed** batch size and replays it regardless of how many rows are live: 31 idle rows are
paid for in full. If that single traced graph also carries per-slot state — and the
gated-delta layers do carry recurrent state per slot — then a single-request workload at
batch 32 exercises 1 live row against 31 dead ones inside a graph built for 32. Long
generations give any per-slot bookkeeping error time to accumulate, which matches the
observed length dependence.

That makes `tests/full_model_mixed_slots.py` and `tests/greedy_sampler_active_rows.py`
the right places to look next: both exist on this branch, and neither is run at batch 32
with a single active row over a multi-thousand-token generation.

## The next experiment

Narrow it inside the batch-32 path, cheapest first:

1. **`max_num_seqs 2`, 4, 8** on Diamond row 0. If degradation scales with allocated batch
   rather than switching on, that points at per-slot state volume; if it appears only at
   large batch, at a specific geometry.
2. **Batch 32 with 32 concurrent identical requests.** If output is clean when all rows
   are live but broken when one row is live, the defect is in inactive-row handling, which
   is a much narrower target.
3. Only then the token-id-vs-string test from `SAMPLING_TEXT_QUALITY.md`, to separate
   sampling from detokenization within whichever regime is broken.

Experiment 2 is the highest-information one and directly tests the active-row hypothesis.

---

## Does CI test batch 32 the way that would catch this? No.

Question asked directly, answered from the sweep matrix rather than by inference. CI
exercises `max_num_seqs 32` in two places, and neither can detect the degradation
documented above.

### 1. The benchmark phase: batch 32 with all rows live, but only short outputs

The p300x2 text sweep for this model, enumerated from
`reference_config/benchmarking/benchmark_config.py` via `get_benchmark_config()`:

| isl | **osl** | max_conc | n | has perf targets |
|---:|---:|---:|---:|:---:|
| 128 | 128 | 1 | 8 | **yes** |
| 128 | 128 | 1 | 8 | no |
| 128 | 128 | **32** | 256 | no |
| 128 | **1024** | 1 | 4 | no |
| 128 | **1024** | **32** | 128 | no |
| 1024 | 128 | 1 / 32 | 4 / 128 | no |
| 2048 | 128 | 1 / 32 | 4 / 128 | no |
| 4096 | 128 | 1 / 32 | 4 / 128 | no |
| 8192 | 128 | 1 / 32 | 2 / 64 | no |
| 16384 | 128 | 1 / 31 | 2 / 62 | no |
| 32768 | 128 | 1 / 15 | 1 / 15 | no |
| 65536 | 128 | 1 / 8 | 1 / 8 | no |
| 131072 | 128 | 1 / 4 | 1 / 4 | no |

**Output length is pinned at 128 for every point except two.** The sweep scales *input*
aggressively — to 131,072 tokens — while holding generation at 128 tokens. Measured on
this port, 128-484 token generations are **clean at batch 32** (the isolation probe's
trivial and easy prompts were correct in every arm). So these points exercise the batch-32
path in precisely the regime where it works.

The single longest-output point is `osl=1024` at `max_conc=32`. Degradation appeared at
~1,800+ tokens, so even that point sits below the threshold.

And decisively: **the benchmark sweep measures throughput and latency, never output
correctness.** Nothing in it inspects the generated text, so even a point that did
degrade would pass.

### 2. The eval phase: long outputs at batch 32, but most rows idle and only the answer graded

The eval serves at `max_num_seqs 32` with `num_concurrent=32`, but `CI_NIGHTLY: 0.05`
yields **10 documents**. So at most 10 of 32 rows are ever live — the configuration in
which the isolation probe measured corruption. But the eval grades only the extracted
letter, so degradation surfaces as a low score, which reads as a model-quality problem
rather than a batch-path defect. That is exactly what happened: the CI-faithful run
reported 0.10 (truly 0/10) and nothing in the output pointed at batch 32.

### The empty cell

|  | short output (<=1024 tok) | **long output (>1800 tok)** |
|---|---|---|
| **batch 1** | benchmark points, clean | eval at conc 1, clean (measured 0.60) |
| **batch 32** | benchmark points, clean and *throughput-only* | **never checked for correctness** |

The combination that breaks — long generation at batch 32 — is never validated for
correctness anywhere in the release flow. The eval reaches that cell but cannot attribute
what it finds; the benchmarks never reach it.

### Two further gaps in the same table

1. **Only one sweep point carries perf targets**: `isl=128 osl=128 max_concurrency=1`. Its
   own comment reads: *"ASSUMED, NOT VALIDATED. No t3k entry exists for this model, so
   numbers are inferred from the closest same-size/type model in this file: 'Qwen3-32B' t3k
   (ttft 62.0 / tput_user 41.0)"*, and it flags both caveats itself — a cross-architecture
   transplant (8 Wormhole chips vs 4 Blackhole) and that *"Qwen3.6 is a hybrid
   Gated-DeltaNet/Gated-Attention model whose decode cost profile differs from dense
   Qwen3-32B."* So the only gated perf number is an unvalidated guess.

2. **That gated point assumes `max_concurrency: 1`** — *"'tput' is set equal to 'tput_user'
   because max_concurrency is 1"* — while the serving spec sets **32**. The one perf target
   that gates the release is therefore defined in a configuration the release does not
   serve in, and `SERVING_BATCH_LATENCY.md` shows the two differ by ~4.8x per token.

### What would close it

One sweep point: **`osl >= 4096` at `max_conc = 32`**, with an output-quality assertion
rather than a throughput target. Cheap to add to the same matrix, and it is the only cell
that distinguishes "slow at batch 32" from "wrong at batch 32".
