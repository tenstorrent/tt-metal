# Decode cost is set by the allocated batch, not the active rows

Operator analysis, 2026-08-17. Found while re-running the stage-11 evals; it
bears on the release's single-user serving claim.

## The three data points

| source | max_num_seqs | active requests | per-token |
|---|---:|---:|---:|
| stage 10 `optimized_vllm` (recorded) | 1 | 1 | **ITL P50 55.840 ms** |
| stage 09 `vllm_integration` burst (recorded) | 32 | 32 | **ITL P50 244.0 ms** |
| this re-run (measured 2026-08-17) | 32 | **1** | **~270 ms** |

The third row is the new one. An lm-eval `ifeval` re-run at `num_concurrent=1`
against a server started with `--max_num_seqs 32` advanced at 345.4 s per
document with `max_gen_toks=1280`. Most documents run to the cap — the audit of
the original run's retained samples found median 759 words and only 9/28 ending
in terminal punctuation — so ≈1280 generated tokens per document gives
345.4/1280 ≈ **270 ms/token**. This is derived from the observed iteration rate
rather than a per-token instrument, and if documents generated fewer than 1280
tokens the true per-token cost is *higher*, so treat it as a lower bound on the
penalty.

## What it means

One active request on a 32-slot server costs **~270 ms/token**, essentially the
same as 32 active requests on that server (**244 ms**), and **4.4–4.8× more than
one request on a 1-slot server (55.8 ms)**.

So the decode step's cost is determined by the batch the model was *built and
traced* at, not by how many rows are actually live. Thirty-one idle rows are paid
for in full. At B32 that is excellent throughput (244 ms for 32 tokens ≈ 7.6
ms/token/user) and poor latency; at B1 it is the reverse. There is no
intermediate behaviour, because there is one captured decode trace at one batch
size.

## Consequence for the release claim

The headline single-user numbers on this port — `TTFT ... TPOT 61.893 ms; ITL
P50/P99 55.840/56.850 ms; TPOT-derived decode 16.157 t/s/u` — were all measured
with `max_num_seqs=1`, i.e. on a server that can serve exactly one request. The
stage-11 release `runtime_model_spec` also records `vllm_args.max_num_seqs = "1"`.

That is a legitimate single-user configuration and the numbers are real. But a
32-slot deployment's single-user latency is **~3.7 t/s/u, not 16.157**, and
nothing on the branch records that. The headline figure is achievable only on a
server provisioned for one user at a time.

## The lever: batch-adaptive decode

Capture decode traces at several batch sizes (for example B1/B4/B8/B32) and
dispatch on the live active-row count instead of always replaying the B32 trace.
Single-user latency on a 32-slot server would approach the B1 number, up to
~4.8× better in the low-occupancy regime that a single-user claim describes.

Why this is plausible rather than speculative on this port:

- The active-row concept already exists here. `tt/generator.py` tracks
  `_slots_requiring_prefill`, supports `reset_slots` to invalidate selected
  request slots "without disturbing live peers", and there are dedicated tests
  `tests/greedy_sampler_active_rows.py` and `tests/full_model_mixed_slots.py`.
  What is missing is using the active count to select a *smaller traced decode*,
  not the bookkeeping to know it.
- Trace capture is already parameterised by prompt length and batch through
  `_capture_token_out_trace`, and the port already runs with
  `trace_region_size` of 200–300 MB.

Costs and risks to measure, not assume:

- **Trace region.** N traces cost roughly N× the decode trace's region. Whether
  four sizes fit alongside the 262,144-token cache is an envelope question, and
  the datatype-sweep notes already record that full-stack residency reduced
  proven B32 context in one configuration.
- **Capture time.** Each additional size adds one capture to server startup,
  which is already ~16 minutes here.
- **Switching cost.** Changing batch mid-stream means re-seeding the token-out
  trace and, for the GatedDeltaNet layers, carrying the recurrent state across a
  differently shaped replay. The 48 linear layers make this the substantive
  engineering, not the sampler.
- **Numerics.** Batch is a matmul dimension, so B1 and B32 replays need not be
  bit-identical. The branch records decode PCC 0.999906 at both B1 and B32, so
  both are individually sound, but a mid-stream switch would need its own
  equivalence check.

## Practical note for anyone re-running the stage evals

Match the stage: `--max_num_seqs 1 --max_num_batched_tokens 262144`. Using 32
neither reproduces the recorded configuration nor merely costs a little time — it
runs the whole eval ~4.8× slower, which turns a `max_gen_toks=32768` task into a
multi-hour-per-document proposition.

---

## Prefill also pays the full allocated batch — and 28x worse than decode

Measured 2026-08-18 by running tt-inference-server's **benchmarks** workflow, which is where
CI exercises this configuration. Sweep point 1 is `isl=128 osl=128 max_conc=1 n=8`, served by
the release spec's `max_concurrency: 32` server:

```
Mean TTFT (ms):                          105353.92
Median TTFT (ms):                        105326.86
P99 TTFT (ms):                           105540.48
Mean TPOT (ms):                             251.16
Mean ITL (ms):                              249.20
Median ITL (ms):                            243.63
Output token throughput (tok/s):              0.93
Mean E2EL (ms):                          137251.81
```

Server batch confirmed from its own log: `GPU KV cache size: 1,752,000 tokens`, which matches
the figure stage 09 recorded for "the capacity profile at max-num-seqs 32".

### The scaling

| | batch 1 | batch 32 | ratio |
|---|---:|---:|---:|
| TTFT, 128-token prompt | 3,784 ms | **105,354 ms** | **27.8x** |
| per-token decode (ITL) | 55.8 ms | 243.6 ms | 4.4x |

The allocated batch is 32x larger. Decode degrades 4.4x; **prefill degrades 27.8x, i.e.
essentially linearly with the allocated batch.** `SERVING_BATCH_LATENCY.md` established that
decode cost follows the allocated batch rather than the active rows. This extends that finding
to prefill, where the effect is far larger — a 128-token prompt takes **105 seconds** to
prefill on a 32-slot server with one active request.

E2EL confirms the decomposition: 105 s TTFT + 128 tokens x 250 ms = 137 s, against a measured
mean E2EL of 137.25 s.

### It completes the timeout explanation

`CI_FAITHFUL_RUN.md` attributed the eval's five timeouts to decode being ~4.8x slower at batch
32. That was only part of it. With 105 s consumed by prefill before the first token, the
1800 s budget leaves `(1800 - 105) / 0.25 = 6,780` tokens — which matches the ~6,600 estimated
there from decode alone, but for a different reason than stated. Prefill is the larger term
per request.

### The only gated perf number misses by 1,700x

The single sweep point carrying perf targets is exactly this one, and its target is
`ttft_ms: 62.0` with `tput_user: 41.0`. Measured: **105,354 ms TTFT** and **0.93 tok/s output
throughput**. That is a ~1,700x miss on TTFT and a ~44x miss on throughput.

Two things to hold together here:

1. The target is self-described as *"ASSUMED, NOT VALIDATED"*, transplanted from Qwen3-32B on
   t3k, and — crucially — **it assumes `max_concurrency: 1`** (*"'tput' is set equal to
   'tput_user' because max_concurrency is 1"*) while the spec serves at 32. So the target and
   the measurement are not describing the same configuration. A 62 ms TTFT target is not
   reachable by *any* configuration of this port; the closest measured value is 3,784 ms at
   batch 1.
2. Nevertheless the benchmark phase **does** surface this, and loudly. That is worth stating
   because it qualifies the coverage claim above: CI cannot catch the *text degradation* at
   batch 32, but it would catch the *performance collapse* — if the target is enforced.
   At `EXPERIMENTAL` status evals and benchmarks are informational, so today it would be
   logged and not blocked.

### What this changes about the recommendations

The batch-adaptive decode lever recorded in `SERVING_BATCH_LATENCY.md` was scoped to decode.
It should be scoped to **prefill and decode**, and prefill is now the bigger prize: 27.8x
against 4.4x. A single traced graph built for the maximum batch is being replayed for one
active row in both phases.

It also means the `max_concurrency: 32` choice in the spec is not merely a latency trade for
throughput. At one active request it costs 105 s before the first token, which is not a
serving configuration any user would accept, and it is the configuration the release ships.
Either the spec should serve a lower batch, or the port needs to size prefill and decode work
to the live rows.
