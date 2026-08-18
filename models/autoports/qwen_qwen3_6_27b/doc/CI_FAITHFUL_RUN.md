# CI-faithful run: the release configuration scores 0.10, and cannot finish the eval

Measured 2026-08-18 by running tt-inference-server's **own** runner, not a hand-rolled
script:

```
run.py --model Qwen3.6-27B --workflow release --tt-device p300x2 --local-server
       --service-port 8000 --no-auth --skip-system-sw-validation
       --limit-samples-mode ci-nightly --ci-mode
       --tt-metal-home ~/tt-metal --tt-metal-python-venv-dir ~/tt-metal/python_env
       --vllm-dir ~/vllm --override-tt-config '{"trace_region_size": 200000000}'
```

with `tt-inference-server` at `fa86cb64` (= `origin/vv-8models` + one data-only commit)
and `vllm` at the prod spec's own pin `03fa3af2e` plus a single-line registry redirect.

## The result

| run | config | `exact_match,none` |
|---|---|---:|
| hand-rolled | `max_num_seqs 1`, `num_concurrent 1`, `timeout 14400` | **0.60** |
| **CI-faithful** | **spec: `max_num_seqs 32`, `num_concurrent 32`, no timeout** | **0.10** |

Same model, same weights, same task, same `gen_kwargs`, same reasoning parser. The
sixfold difference is entirely configuration.

```
RuntimeError: Evaluation completed with 5 failed prompt(s). Check samples_*.jsonl for
__INFERENCE_ERROR__ and __PARTIAL_OUTPUT__ entries.
```

The samples contain `__INFERENCE_ERROR__: TimeoutError()`. **Five of ten documents timed
out** and scored zero.

## Why, with every link measured

1. **lm-eval's default request timeout is 1800 s.**
   `lm_eval/models/api_models.py:251` — `timeout: int = 1800`.
2. **tt-inference-server never overrides it.** The `model_args` it built contain no
   `timeout=` and no `max_retries=`:
   ```
   model=Qwen/Qwen3.6-27B,base_url=...,tokenizer_backend=huggingface,
   max_length=262144,num_concurrent=32
   ```
   There is no `timeout` anywhere in `llm_module/eval_command.py`.
3. **The release serves at `max_concurrency: 32`,** so decode costs **~270 ms/token**
   rather than ~56 ms. This is the mechanism recorded in `SERVING_BATCH_LATENCY.md`:
   decode cost follows the *allocated* batch, not the active rows.
4. **So a request can emit at most ~6,600 tokens** within 1800 s.
5. **Reasoning documents need 19k-32k tokens** — measured in the hand-rolled run, where
   per-document generation ranged ~4.3k to the full 32,768 cap.
6. Documents past ~6,600 tokens therefore die with `TimeoutError()` and score 0.

A telling coincidence: at batch 1 (~56 ms/token) 1800 s covers ~32,100 tokens, almost
exactly the 32,768 budget. **The default timeout is well matched to this budget only at
batch 1.** At batch 32 it cannot complete a single real reasoning document.

## What this means for the release

As configured, the release flow would report **0.10** for a model whose own card
publishes 87.8, and the number would be dominated by client timeouts rather than by
model quality. The accuracy check target was registered correctly —

```
kind=evals id=Qwen3.6-27B_P300X2 targets={'task_name': 'r1_gpqa_diamond',
  'tolerance': 0.05, 'published_score': 87.8, 'published_score_ref': ...}
```

— so the bar is 83.41% and the run reads 0.10, i.e. a catastrophic-looking failure with a
purely mechanical cause. Worth noting the workflow did **not** stop: the eval is
`check=False`, so it logged `⛔ ... return code: 1`, recorded the block, and proceeded to
the benchmark phase. Informational at EXPERIMENTAL, exactly as the sibling entry's
comments predict.

## Three fixes, in order of preference

1. **Pass an explicit `timeout` scaled to the budget and the serving batch.** For
   `max_gen_toks 32768` at `max_concurrency 32` that is ~2.5 h, so `timeout=14400`. This
   belongs in `eval_command.py` as a function of `gen_kwargs.max_gen_toks` and
   `device_model_spec.max_concurrency`, not as a per-model constant, because every
   reasoning model on a high-concurrency spec has the same problem.
2. **Serve reasoning evals at low concurrency.** At `max_num_seqs 1` the default 1800 s
   almost exactly covers a 32,768-token budget. This also runs *faster* in wall clock
   here: the hand-rolled sequential run took 2.3 h and completed all ten documents, while
   the CI-faithful concurrent run took **6,334.9 s (1 h 46 m)** and lost half of them.
3. **Reduce `max_gen_toks`** so documents fit the timeout. At batch 32 that means ~6,600
   tokens, which is far too small for this model's reasoning and would simply move the
   failure from timeouts to truncation.

## A prediction of mine that this falsified

I estimated the CI configuration would finish in ~85 min against 2.3 h for the sequential
run, on the grounds that ten concurrent documents would amortise the batch-32 step cost.
That was wrong. With only 10 of 32 slots occupied, each document still advances at
~270 ms/token, so concurrency does not recover the 4.8x per-token penalty. I had applied
the allocated-batch finding to latency but not to my own throughput estimate. Measured:
**1 h 46 m for the eval phase, with 5 of 10 documents lost**, versus 2 h 18 m and 10/10
sequentially.

## Two spec problems found on the way

Both are in the release spec rather than the model, and both cost a full startup cycle
(~15 min) to discover:

1. **`TT_MESH_GRAPH_DESC_PATH` is a relative path.**
   `"../../tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto"`
   resolves against the server's cwd `<tti>/vllm-tt-metal/src`, i.e. to
   `<tti>/tt-metal/...`. It therefore only works when tt-metal is nested inside the
   tt-inference-server tree. Otherwise fabric init dies with
   `TT_FATAL: Custom mesh graph descriptor file not found`, pointing at tt-metal rather
   than at the spec. Reproduced CI's layout with a symlink rather than editing the spec.

2. **`trace_region_size: 1073741824` exhausts DRAM with this implementation.**
   ```
   Out of Memory: Not enough space to allocate 476544000 B DRAM buffer across 8 banks,
     each bank needs 59568000 B, bank size 3198599552 B
     (allocated: 3189618496 B, free: 8981056 B, largest free block: 4394752 B)
   ```
   99.7% of each 3.2 GB bank already in use. Lowering only that key to 200 MB via
   `--override-tt-config` let the server start; `model_spec.py:873` merges the CLI value
   over the spec, so `FABRIC_1D`, `l1_small_size 24576` and
   `sample_on_device_mode decode_only` were all preserved and verified in the server log.
   Whether the 1 GB region fits the *demo* implementation is untested here, so this may be
   an autoport footprint difference rather than a spec error.

## Remaining deviations from true CI

- `tt_metal_commit`: the prod spec pins `de59f8a`, not fetchable into this shallow clone.
- `impl`: the registry is redirected to the autoport because
  `models/demos/blackhole/qwen36` does not exist at this tt-metal pin. **CI as pinned
  would test the demo, not this port.** This remains the largest caveat on every finding
  in this directory.
- `trace_region_size` lowered as above.

## Text quality under `decode_only`

Not yet analysed. The release config uses `sample_on_device_mode: decode_only` where the
hand-rolled run used `all`, and this is the discriminator recorded in
`SAMPLING_TEXT_QUALITY.md` between the logits/sampling and detokenization hypotheses.
The five documents that did complete are in this run's `samples_*.jsonl` and should be
compared against the hand-rolled ones before either hypothesis is accepted.
