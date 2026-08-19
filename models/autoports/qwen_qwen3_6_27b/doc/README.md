# Qwen3.6-27B autoport — why the CI checks are not passing

Written 2026-08-19. Start here; the detailed documents are linked from each section.

## The bottom line

**At batch 1 with the correct task, this port answers GPQA Diamond at 0.60 (6/10).** It is not
a broken model. Six of seven CI checks nevertheless fail or cannot be graded, and the causes
sort into four groups with different owners:

| # | Cause | Kind | Checks it breaks |
|---|---|---|---|
| **A** | Prefill and decode work is sized to the **allocated** batch, not the live rows | **port defect, needs code** | graded benchmark point, benchmark sweep, 5/10 eval timeouts, 9/22 spec-test timeouts |
| **B** | Generated text degrades under non-greedy sampling | **port defect, needs code** | eval quality at every batch size |
| **C** | The model was not onboarded, and is graded on the wrong task variant | **configuration, no code** | evals ran at all / at the wrong budget; spec tests never ran |
| **D** | Harness timeouts and assumptions that no port on this hardware can meet | **upstream harness** | eval timeouts, spec-test timeouts and crashes |

**A single choice — serving at `max_concurrency: 32` — accounts for the graded benchmark point,
the unfinishable sweep, the eval's five timeouts and half the spec-test failures.** It is
separable from the port's correctness at batch 1.

---

## A. Work is sized to the allocated batch, not the live rows

The port captures **one** decode trace and one prefill path at a **fixed** batch size and replays
them regardless of how many rows are live. Thirty-one idle rows are paid for in full.

Measured on TTI's own benchmark workflow ([`SERVING_BATCH_LATENCY.md`](SERVING_BATCH_LATENCY.md)):

| | batch 1 | batch 32 | ratio |
|---|---:|---:|---:|
| TTFT, 128-token prompt | 3,784 ms | **105,354 ms** | **27.8×** |
| decode per token (ITL) | 55.8 ms | 243.8 ms | 4.4× |

ITL is constant to **0.23%** across 1 vs 32 live rows and 128 vs 1024 output tokens, so the cost
is genuinely a function of the allocated batch and nothing else. Consequences:

- **graded benchmark point**: target `ttft_ms 62.0`, measured **105,354 ms**;
- **benchmark sweep**: cannot finish — the sweep scales input to 131,072 tokens, projecting ~30 h
  for one prefill against a 6 h workflow timeout;
- **eval**: 105 s of prefill leaves ~6,900 tokens inside lm-eval's 1800 s default, so 5 of 10
  documents time out;
- **spec tests**: no case needing a real generation can answer inside its 30 s read timeout, so 9
  of 22 fail.

**Fix**: size prefill and decode to the live row count, or capture traces at several batch sizes
and dispatch on active rows. Prefill is the bigger prize (27.8× vs 4.4×). Task #15.

## B. Generated text degrades under non-greedy sampling

The one defect that affects **output correctness**, and the most serious open item
([`SAMPLING_TEXT_QUALITY.md`](SAMPLING_TEXT_QUALITY.md),
[`BATCH32_DEGRADATION.md`](BATCH32_DEGRADATION.md)).

Under the release's own sampling (`temperature 1.0, top_k 20, top_p 0.95`), generations start
fluent and decay: `state state state`, `naturallinewidth`, `clearlyy distingdistinguish`, and
answer options losing their exponents. It is worse at batch 32 (documents of 1–196 words,
corrupt from the first token) but **present at batch 1 too**, so it is not merely a batch
artifact. A clean 2×2 pins batch size, not device sampling mode, as the aggravating factor:

| | `sample=all` | `sample=decode_only` |
|---|---|---|
| `max_num_seqs=1` | clean, 1849 tok, correct | clean, 1849 tok, correct |
| `max_num_seqs=32` | broken, 4096 cap, 12.6% rep | broken, 1–196 words |

Two hypotheses remain open — wrong logits exposed by sampling, or incremental detokenization —
with the discriminating test written and unrun: capture raw token ids, detokenize offline in one
shot, compare against the assembled `content`. Task #17.

Under greedy decoding the text is fluent but hard items loop instead (one 12-gram repeated
**1,241 times**), so neither sampling mode is currently sound.

## C. The model was not onboarded, and is graded on the wrong task

Three instances of the same pattern ([`RELEASE_CONFIG_DIVERGENCE.md`](RELEASE_CONFIG_DIVERGENCE.md),
[`CI_COVERAGE_GAP_ANALYSIS.md`](CI_COVERAGE_GAP_ANALYSIS.md)):

1. **No standard evals upstream.** `origin/main` and the release branch give this model only the
   agentic `terminal_bench_2`, so standard selection returns `[]` — and the workflow records that
   empty selection as a **successful no-op**. Fixed by a data-only commit adding
   `r1_gpqa_diamond` (tt-inference-server `fa86cb64`, +84 lines).
2. **No spec-test registration.** The model appeared in no matrix in
   `test_module/test_suites/llm.json`, so the workflow selected zero tests and exited
   "No blocks accumulated". Fixed data-only (tt-inference-server `fd621c1e`, +13/−1).
3. **The wrong GPQA variant.** `gpqa_diamond_cot_zeroshot` sets no `max_gen_toks` (lm-eval falls
   back to 256), uses greedy decoding, has `until: ["</s>"]` which is not a Qwen stop token, and a
   `strict-match` regex looking for "The answer is" while its own prompt asks for `\boxed{}`.
   `r1_gpqa_diamond` fixes all four. Measured effect: **0.30 → 0.60**, no model change.

Plus a fourth, configurational rather than missing: **thinking mode plus a reasoning parser with
no budget**. The chat template leaves `<think>` open by default; with `reasoning_parser: qwen3`, a
generation that never reaches `</think>` returns `content: null`. That produces `[invalid]` GPQA
extractions and the nine `AttributeError: 'NoneType'` spec-test crashes.

## D. Harness assumptions no port on this hardware can meet

Upstream defects, each affecting every model, not just this one:

- **lm-eval's default request timeout is 1800 s and TTI never overrides it** — no `timeout=` in
  the `model_args` it builds, and none anywhere in `eval_command.py`.
- **The conformance suite's read timeout is 30 s**, unreachable at batch 32.
- **`spec_tests --local-server` races the server by ~2 seconds** — it launches the server and
  tests it immediately. `base_test.py:446` defines `wait_for_server_ready`; this path never calls
  it.
- **An empty eval selection is recorded as success**, which is how this model shipped with zero
  standard evals measured.
- **The conformance suite crashes on `content is None`** rather than skipping, which any reasoning
  model with a reasoning parser will trigger.
- **CI never checks output correctness at batch 32 with long generations.** The benchmark sweep
  pins `osl` at 128 for every point but two and measures only throughput; the eval reaches long
  outputs but grades a single letter. That is the empty cell where defect **B** lives.

## What each check needs in order to pass

| CI check | current | what it needs |
|---|---|---|
| Graded benchmark point | ✗ 105,354 ms vs 62 ms | **A**; and a target that describes the served configuration — the existing one is self-described "ASSUMED, NOT VALIDATED", transplanted from Qwen3-32B on t3k, and assumes `max_concurrency 1` |
| Benchmark sweep | 4 of 20 points | **A** |
| Benchmark at concurrency 32 | ✓ clean (256 reqs, 0 errors) | already passes — it only exercises short outputs |
| Layer PCC + AIME24 | ✓ | — |
| Eval — GPQA | 0.60 at b1 / 0-0.10 at b32 | **A**, **B**, **C**, **D** |
| Spec tests | 4/22, 0 genuine failures | **A** and **D**; no model change implicated |
| Full release workflow | ✗ on evals | the above |

## Reading order for the detail

1. [`SERVING_BATCH_LATENCY.md`](SERVING_BATCH_LATENCY.md) — cause **A**, with the measured sweep.
2. [`SAMPLING_TEXT_QUALITY.md`](SAMPLING_TEXT_QUALITY.md) and
   [`BATCH32_DEGRADATION.md`](BATCH32_DEGRADATION.md) — cause **B** and the 2×2 isolation.
3. [`tti_release/BLOCKER_ACCOUNT.md`](tti_release/BLOCKER_ACCOUNT.md) and
   [`tti_release/NON_TERMINATION.md`](tti_release/NON_TERMINATION.md) — cause **C**, including the
   greedy-loop measurement and the thinking-mode template.
4. [`RELEASE_CONFIG_DIVERGENCE.md`](RELEASE_CONFIG_DIVERGENCE.md) and
   [`PROPOSED_EVAL_CONFIG.md`](PROPOSED_EVAL_CONFIG.md) — what CI actually runs, and the config to
   upstream.
5. [`CI_FAITHFUL_RUN.md`](CI_FAITHFUL_RUN.md) and
   [`CI_COVERAGE_GAP_ANALYSIS.md`](CI_COVERAGE_GAP_ANALYSIS.md) — the end-to-end release run and
   the spec-test verdict.
6. [`PREFILL_CHUNK_LEVER.md`](PREFILL_CHUNK_LEVER.md) and
   [`TTFT_MEASUREMENT_DEFECT.md`](TTFT_MEASUREMENT_DEFECT.md) — prefill implementation detail, and
   why earlier TTFT numbers cannot be compared.
7. [`OPERATIONAL_NOTES.md`](OPERATIONAL_NOTES.md) — instrumentation traps, including three of mine
   that produced confidently wrong results. Read before trusting any tooling here.

## Two standing caveats on everything above

- **CI as pinned would test `models/demos/blackhole/qwen36`, not this autoport.** The prod spec
  pins `tt_metal_commit: de59f8a` / `vllm_commit: 03fa3af`, and upstream vLLM maps Qwen to the
  demo. Every measurement here required redirecting that registry to the autoport. Findings
  transfer only insofar as the two trees share code, and that comparison has not been made.
- **Generated reports state spec-declared pins, not what ran.** A report from these runs reads
  `tt_metal_commit: de59f8a` while the actual tree is this branch. Do not use one as evidence of
  what was verified.
