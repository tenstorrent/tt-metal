# The r1_gpqa_diamond 40 has a cause: the recurrent state is BFP8_B

Measured 2026-09-10/11 on p300x2 (TP4), against QB2 CI run 34360801790 and an
exact local reproduction of it.

`doc/SAMPLING_TEXT_QUALITY.md` recorded that this port's generations are
"badly degraded" under release sampling, offered two candidate mechanisms, and
named a discriminating test it had not run. `c8cc2ad` then measured the eval
again after the decode work and recorded `40 -> 40`, attributing it to "the
port pre-existing quality gap ... consistent with doc/SAMPLING_TEXT_QUALITY.md".

That gap now has a cause, and it is one line of configuration.

## The finding

`doc/datatype_sweep/selected_precision_config.json` sets

```
"linear_recurrent_state_dtype": "BFP8_B"
```

48 of this model's 64 layers are gated-delta linear attention. Their recurrence

```
S_t = S_{t-1} * decay + k_t (x) v_t
```

re-quantizes `S` to 8-bit block float on **every decode step**, so quantization
error is re-injected each step and random-walks upward with step count. Output
is clean early and degrades progressively as the generation gets longer.

Setting that one field to `BF16` removes the corruption.

## The A/B, at the current tip

Single request, concurrency 1, same prompt, same sampling
(`temperature 1.0, top_k 20, top_p 0.95`), `linear_kda_conv` base policy,
`5ee0500`. Metric is tokenizer round-trip instability per 250-token window:
the fraction of a window's sampled ids that do not survive
`encode(decode(ids))`. It needs no hand-written patterns, and a well-formed
generation scores 0.

| tokens | stock (BFP8_B) | BF16 state |
| ---: | ---: | ---: |
| 0 - 3500 | 0.0% | 0.0% |
| 3500 - 3750 | 1.6% | 0.0% |
| 3750 - 4250 | 1.2% | 0.0% |
| 4250 - 4500 | **3.6%** | 0.0% |
| 4500 - 4750 | 1.6% | 0.0% |
| 4750 - 5000 | **4.4%** | 0.0% |
| 5000 - 6000 | 1.1% / n/a | 0.0% |

The stock run also emitted `</think>` **three** times (correct: one). Its tail:

> `...The single and doublemutant phenotypes are unspecified. Since\nStep 2:
> Determine the answer. In\nCannot determine.` ... `But(0)`

The BF16 run is fluent to 6000 tokens, which is 900 tokens further than the
stock run reached at all.

### Cost: ~2% of decode, not 26%

| | ITL, 300 tokens, concurrency 1 |
| --- | ---: |
| stock BFP8_B | 88.1 ms (10.54 t/s/u) |
| BF16 state | 90.0 ms (10.31 t/s/u) |

The 88.1 ms confirms `87d1f58`'s 88.37 ms independently, at
`max_model_len 262144` rather than its 4096.

**This cost is new.** The same A/B at `38153c48` (before `44e1aefa` and
`8148d6dfa`) cost **26%**: 4.08 -> 2.9 t/s/u. Writing the state once instead of
twice and giving the state matmuls a batched-reuse program took the state
traffic off the critical path, so the precision of the state is now nearly free.
Anyone who rejected BF16 state on cost before those two commits should re-price
it.

## What this corrects in SAMPLING_TEXT_QUALITY.md

That doc offered two mechanisms and said evidence pulled both ways. Both are
wrong, and its discriminating test is what shows it. The test: take the token
ids the server sampled, detokenize them offline in one shot, compare.

- **B, incremental detokenization / text assembly: dead.** The corruption is
  present in a one-shot offline HF detokenization of the raw ids, with vLLM's
  assembly nowhere in the path.
- **A, wrong logits exposed by sampling: dead as written.** Identical sampling
  is clean for 3500 tokens in the same server, then degrades. A per-token
  distribution error would corrupt uniformly from token 1.

The doc reasoned that greedy was fluent while sampled was garbled, so sampling
must be the discriminator. The actual discriminator is **token position**; its
greedy runs looped early (`110^{-64}` x 1241) and never reached the degraded
region, so the comparison was confounded.

Its description of the corruption is exactly right and was the strongest clue:
"fluent at the start, corrupted by the end", `naturallinewidth`, `peaksopt`,
`distingdistinguish` -- merges and duplicated fragments, growing with position.

## Ruled out by experiment, each negative

| hypothesis | test | result |
| --- | --- | --- |
| concurrency / batched decode | concurrency 1, 4, 10 | clean at all three |
| client abort + slot reuse | 6 of 10 dropped mid-flight, survivors checked | 0/4 survivors corrupt |
| prompt content | the 10 real eval prompts replayed | corruption tracks length, not prompt |
| text assembly | offline one-shot detokenization of the ids | corruption present in the ids |

The abort test was worth running because vllm-tt-plugin carries three slot-state
fixes (#454, #466, #468); it is not that.

## Why the eval scores 40, in full

Two independent faults. Fixing either alone cannot pass.

**1. The client timeout.** The lm-eval fork (`tstescoTT@321e3bb`) hardcodes
`ClientTimeout(total=1800)` and `stop_after_attempt(3)`
(`api_models.py:277,811`). CI and the local repro agree exactly:

```
             CI 34360801790   local repro
t=1800s      6 time out       6
t=3600s      4                4
t=5400s      3                3
sentinels    3                3
score        40               40      (ratio 0.4484)
eval wall    5416.1 s         5415.8 s
```

A timed-out request returns the literal string `__INFERENCE_ERROR__:
TimeoutError()`, which carries no boxed letter, so `process_results_gpqa`
scores it 0. Three of ten are therefore hard zeros and the ceiling is 70%
against a bar of `89.2 * 0.95 = 84.74%`. Each retry regenerates from scratch and
`stream=false` forfeits partial output.

`reference_config/evals/eval_config.py:2094` asks for `max_gen_toks = 80*1024`.
At the pre-`44e1aefa` decode rate only ~4900-7000 tokens fitted in 1800 s.
`model_kwargs={"timeout": ...}` reaches `--model_args` via
`llm_module/eval_command.py:303`; `workflows/requirements_target_pack.py:330`
already defaults to 3600 and `reference_config` does not.

Lowering eval concurrency does **not** help. Measured ITL at `38153c48` was
236.5 ms at concurrency 1, 4 **and** 10 -- decode cost follows `max_num_seqs=32`,
not the active rows. That confirms `doc/SERVING_BATCH_LATENCY.md` with a
per-token instrument where it previously had only a derived lower bound.

**2. The corruption above.** Raising the timeout lets documents run *longer*,
further into the degraded region.

### This also explains `mean_seconds_per_task`

`c8cc2ad` recorded as unexplained that `mean_seconds_per_task` was 541.5 against
541.6 despite 1.6x the tokens. It is the eval's wall clock divided by the ten
documents, and the wall clock is pinned by the timeout ladder, not by the model:
three attempts x 1800 s = 5400 s, plus startup. `5415.8 / 10 = 541.58`;
`5416.1 / 10 = 541.61`. As long as at least one document exhausts all three
attempts, the number is a constant and says nothing about decode speed. It will
only move once no document times out.

### A CI bug hides all of this

The `workflow_logs` artifact upload fails with

```
The artifact name is not valid: workflow_logs_evals_Qwen/Qwen3.8-27B_bh-qb-ge_default.
Contains the following character:  Forward slash /
```

so `samples_*.jsonl` -- the only place the corrupted text is visible -- is never
uploaded. The report artifact step already sanitizes the model name; that same
fix is needed here. Until it lands, every CI eval for a model whose name
contains `/` reports a score with no way to see what produced it.

## Recommendation

Set `linear_recurrent_state_dtype` to `BF16` in
`doc/datatype_sweep/selected_precision_config.json`. It costs ~2% of decode at
the current tip and is the difference between coherent and incoherent output
past ~3500 tokens.

Not done here on purpose: this doc is the evidence, and flipping the shipped
precision config is a release decision that wants its own PCC and eval run
behind it. The eval to run after flipping is the CI one with the timeout raised,
so that the score reflects the model rather than the timeout ladder.

## Confidence and what is thin

- The A/B is **one run per arm** at the tip, and sampling at temperature 1.0
  makes the two arms different token sequences. The stock side is corroborated
  many times over (this run, the CI eval's samples, a 10-prompt replay at 4000
  tokens, the earlier `38153c48` profile, and SAMPLING_TEXT_QUALITY.md's
  independent 2026-08-17 observations). The BF16 side is **one 6000-token run at
  the tip plus one at `38153c48` plus a 10-prompt replay at `38153c48`**, all
  clean. More repeats per arm would firm up the onset point, which is the least
  certain number here (~2750 tokens at `38153c48`, ~3500 at the tip -- whether
  that shift is the `linear_kda_conv` policy or run-to-run variation is
  untested).
- The mechanism -- per-step re-quantization accumulating -- is inference from
  the shape of the curve plus the dtype, not a measurement of the state error
  itself. Instrumenting `S`'s divergence from a BF16 reference per step would
  settle it directly and has not been done.
- `kv_cache_dtype: BFP8_B` is the other position-dependent candidate and was
  **not** independently cleared: BF16 KV does not fit (`allocate_kv_cache` wants
  885 MB, 91 MB free), so the two could not be separated by holding the state
  fixed. The recurrent-state change alone being sufficient is what argues
  against KV being involved.

## Reproducing

Harnesses in `harnesses/`. They talk to a running server on :8000 and need the
weights path for the tokenizer.

| script | what it answers |
| --- | --- |
| `length_profile.py` | corruption vs token position, one request |
| `rt_one.py` | round-trip instability profile of a saved run |
| `discriminate.py` | ids vs server string, the SAMPLING_TEXT_QUALITY test |
| `discriminate_conc.py` | the same at N concurrent |
| `abort_test.py` | do client aborts corrupt the survivors |
| `replay_eval.py` | replay the real eval prompts from a samples jsonl |
| `decode_probe.py` | ITL / t-s-u / TTFT at N concurrent |

Two traps that cost time here:

- **`QWEN36_PRECISION_CONFIG` is inert under vLLM.** `generator_vllm.py:138`
  passes `precision_config=DEFAULT_PRECISION_CONFIG` explicitly, and
  `load_precision_config` only consults the environment when its argument is
  `None`. The serving wrapper still logs
  `setting env var: QWEN36_PRECISION_CONFIG=...`, so it looks applied and is
  not. To A/B precision on the serving path you must overwrite
  `doc/datatype_sweep/selected_precision_config.json` itself. Worth fixing:
  passing `None` there would make the documented knob work.
- The P300X2 spec's `EXTRA_MODELS_DIR` and `TT_MESH_GRAPH_DESC_PATH` are
  container-relative (`../../tt-metal/...`). Under `--local-server` they resolve
  against `<tt-inference-server>/vllm-tt-metal/src`, so without a `tt-metal`
  symlink at that repo's root the mesh descriptor `TT_FATAL`s -- and before
  that, the autoport bundle is skipped with a *warning*
  (`is not a directory; ignoring`) and the **demo** `models.demos.blackhole.qwen36`
  is served instead. Always confirm
  `Registered TT model ... (from EXTRA_MODELS_DIR/qwen36_autoport)` before
  trusting a local number.
