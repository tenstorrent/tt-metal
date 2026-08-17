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
