# Serving performance — next steps for prefill and scheduling

Muse-Glimmer-30B decodes well under load and prefills badly under load. This
document records the measurement, names the cause in code, and ranks the work.

Measured 2026-08-31 on P300x2 (4 Blackhole dies, 1x4 `FABRIC_1D_RING`) against
tt-metal `0dd37ce6ee3`, through the vLLM OpenAI server at `max_num_seqs=32`,
`max_model_len=131072`. Raw data: `logs/batchsweep/`, summary
`logs/batch_sweep_summary.json`.

## 1. What the sweep found

**Decode batching is close to free.** At ISL 1,024 / OSL 512, concurrency 1 to 32
costs 4% of per-user speed and returns 22.7x aggregate throughput.

| conc | TPOT ms | t/s/u | out tok/s |
| ---: | ---: | ---: | ---: |
| 1 | 23.84 | 41.94 | 41.5 |
| 32 | 24.88 | 40.19 | 942.0 |

**Prefill does not batch at all.** Concurrent requests prefill one after another.
The arithmetic is exact — at ISL 4,096, concurrency 32:

```
predicted  32 x 458.1 ms (batch-1 TTFT)  = 14.66 s
measured   median TTFT                   = 14.87 s   (1.4% error)
```

**The serial queue then corrupts decode.** Decode steps interleave with the
still-draining prefill queue, so TPOT under load is contention, not decode rate:

| ISL | conc | TPOT ms, batch 1 | TPOT ms, at conc | inflation |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 32 | 24.97 | 24.88 | 1.0x |
| 16,384 | 32 | 30.33 | 59.05 | 1.9x |
| 32,768 | 31 | 35.24 | 125.37 | 3.6x |
| 130,560 | 8 | 64.79 | 169.17 | 2.6x |

Aggregate throughput falls from 1,154 tok/s (ISL 128) to 17.7 tok/s (ISL 130,560).
Decode is not the limiter anywhere in that range. Prefill scheduling is.

## 2. Where it comes from, in code

1. `vllm-tt-plugin/src/vllm_tt_plugin/platform.py:57` —
   `_CHUNKED_PREFILL_MODEL_TYPES = {"gemma4", "gemma4_unified"}`. This is an
   allowlist. `model_type=muse_glimmer` is absent, so line 79 sets
   `enable_chunked_prefill = False`. The restriction is a plugin policy, not a
   property of this model.
2. `platform.py:98` forces `long_prefill_token_threshold = 0`, because the base
   scheduler would otherwise split a prefill the model cannot resume.
3. `tt/generator_vllm.py:808-815` rejects any resumed prefill:
   `NotImplementedError`, "serving prefill starts every request at position 0 ...
   this port does not expose the layer stack's continuation prefill through the
   serving path."
4. The capability exists one level down. `tt/model.py:718` already chunks prefill
   internally at 8,192 tokens, and `generator.prefill_forward` documents
   mixed-length batched prefill into distinct cache slots.
5. The adapter already forwards the plural forms — `prompt_lens=lens` and
   `user_ids=empty_slots` (`generator_vllm.py:820`, `:823`). Nothing in the port
   forces one request per prefill call.

The gap is between the scheduler and the serving adapter, not in the layer stack.

## 3. Ranked work

### P1 — Chunked prefill

Split a long prefill across scheduler steps so other requests decode in between.
This addresses both the TTFT queue and the TPOT inflation.

Work:
1. Implement continuation prefill in `generator_vllm.prefill_forward`. Accept
   `start_pos > 0`. Remove the guard at `generator_vllm.py:808-815`.
2. Add `muse_glimmer` to `_CHUNKED_PREFILL_MODEL_TYPES` in the plugin.
3. Restore a nonzero `long_prefill_token_threshold` for this model type.

Risk — this is the hard part, and it is specific to this model:

* 39 of the 52 layers are sliding-window (`sliding_window=2048`, `model.py:358`,
  `:748`). A chunk boundary must carry the correct K/V tail into the next chunk,
  or the window silently truncates. A truncated window does not raise; it returns
  fluent, wrong text. This is the single most likely way to ship a regression here.
* `paged_fill_cache` requires a multi-token prefill to start on a 64-token page
  boundary. Chunk sizes must be multiples of 64.
* Verify with PCC against the reference, not with throughput. A wrong window
  produces plausible text.

Expected: TPOT at ISL 16,384 / conc 32 returns from 59.05 ms toward the batch-1
30.33 ms; TTFT becomes fair-shared instead of FIFO.

### P2 — Batched prefill for short prompts

Prefill several short requests in one call.

The port already supports this (section 2, items 4 and 5). The limiter appears to
be scheduler-side: only one prefill reaches the runner per step.

**Confirm before building.** The present evidence is timing, not instrumentation.
Count the requests per prefill step in `model_runner.py` around line 2068 and
confirm the count is 1. If the scheduler already groups them, the cause is
elsewhere and this item is void.

Expected if confirmed: at ISL 128 / conc 32, TTFT 2.16 s toward roughly one
batched prefill. Low risk, no new numerics.

### P3 — Re-measure prefill tracing

`MUSE_GLIMMER_VLLM_PREFILL_TRACE=0` is the shipped default, chosen in the
optimized-vLLM stage. Re-measure with tracing on now that prefill is known to
dominate serving. Cheap: an env var and one sweep.

### P4 — Scheduler knobs, no code change

Deployments that care about TTFT more than aggregate throughput can lower
`--max-num-seqs`. Table B shows the trade directly. Document per-workload
profiles rather than one default.

## 4. Operating guidance until P1 lands

* Interactive and agentic coding — long ISL, short OSL. Cap concurrency well
  below 32. TTFT is set by the queue ahead of the request.
* Batch and offline — short ISL. Run at 32. Throughput peaks at 1,154 tok/s.
* Do not quote a concurrent TPOT above ISL 16,384 as a decode figure. It is
  contention. Quote the batch-1 column.
