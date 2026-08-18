# Stage 09 — optimized vLLM serving — Qwen3-Coder-30B-A3B-Instruct

**Single-user serving decode was already within 3 % of the standalone traced
floor and stayed there. What this stage actually moved is the 32-slot server,
and what it mostly produced is evidence: the standalone batch-32 control stage 08
never took, and the discovery that vLLM's async decode split was already on —
including in stage 08's own numbers, which said it was inert.**

> **Corrected 2026-08-18, after review.** This document first published the async
> split as worth **1.754 ms/token, 8.9 %**. That was wrong: the figure came from
> the *mean* TPOT of a leg carrying a single ~179 ms event, and the honest,
> re-measured, median-based number is **0.438 ms/token, 2.16 %**. Re-measuring it
> also produced the better finding — that the ~159 ms is one fixed per-request
> cost that async scheduling merely **relabels** between TTFT and the first
> inter-token latency, not a recurring win. Separately, `sync_decode_reads` was
> mis-implemented, and inactive-row expert gating did **not** survive request
> turnover; both are fixed and now have controls. The 32-slot gating result
> (3.796 → 4.346 t/s/u) and the 0.588 ms serving-overhead headline are unaffected.
> See "Why the first A/B had to be redone", "The win did not survive request
> churn", and work log §11.

## Headline — primary single-user serving

Workload: **128-token input, 128-token output, 1 prompt, `--max-concurrency 1`,
`max_num_seqs=1`, `ignore_eos`, greedy (`--temperature 0.0`)**, complete 48-layer
model, 1x4 mesh, `max_model_len` 262144, `sample_on_device_mode: all`, decode
trace on, `--generation-config vllm`. Same runner, same flags, same server
command before and after; the only difference is
`QWEN3_DECODE_ACTIVE_ROW_GATING`, so the two rows are the same binary.

| Metric | Before | After | Δ |
|---|---|---|---|
| **TTFT** median / P99 | 309.810 ms | **299.142 ms** | −10.668 ms |
| **Decode t/s/u** (`1000 / 19.801`) | 50.484 | **50.503** | +0.018 |
| TPOT mean / P99 | 19.808 ms | 19.801 ms | −0.007 ms |
| ITL median / P99 | 19.805 / 20.348 ms | 19.809 / 20.744 ms | +0.004 ms |
| Aggregate output throughput | 45.298 tok/s | 45.485 tok/s | +0.187 tok/s |
| End-to-end latency | 2825.441 ms | 2813.862 ms | −11.579 ms |
| Completed | 1/1, 128 output tokens | 1/1, 128 output tokens | — |

Artifacts: [`bench/single_user_before_vllm_benchmark.json`](bench/single_user_before_vllm_benchmark.json)
and [`bench/single_user_after_vllm_benchmark.json`](bench/single_user_after_vllm_benchmark.json)
(raw `*_vllm_result.json` beside each).

**This row is deliberately flat, and that is the result.** Stage 09's decode
change is gated off at `max_batch_size == 1` — `_decode_active_mask` returns
`None`, so the captured graph is byte-for-byte the one stage 08 shipped — and
everything else this stage touched is measurement. The 0.007 ms of TPOT and
10.668 ms of TTFT are run-to-run noise in both directions.

**Against the standalone floor.** Stage 07 shipped standalone traced token-out at
**19.213 ms / 52.049 t/s/u**. Serving decode is **19.801 ms**: vLLM adds
**0.588 ms per token, 3.1 %**, over a path that includes the same 48 layers, the
same LM head, the same on-device split sampling and the same `tt_out_tok`
feedback. There is no remaining serving-side term of that size to remove — see
"Where the 0.588 ms goes", which now has an A/B behind it rather than a list.

## The finding that changes stage 08's story: async decode is on by default

Stage 08 reported that `--async-scheduling` is "accepted but inert" because
`TTScheduler` is not an `AsyncScheduler`, and that "the shipped command leaves it
off". **Both halves are wrong, and the shipped command was never running with it
off.**

1. `vllm-tt-plugin/src/vllm_tt_plugin/scheduler.py:31` is literally `class TTScheduler(AsyncScheduler)`.
   The vLLM log line stage 08 quoted — "*If* you have subclassed Scheduler
   instead of AsyncScheduler, you will see degraded performance" — is emitted
   unconditionally for **any** custom `scheduler_cls`
   (`vllm/config/scheduler.py::get_scheduler_cls`). It is a warning about a
   hypothetical, read as a statement of fact.
2. vLLM 0.24.0 **enables async scheduling by default**:
   `scheduler_config.async_scheduling` defaults to `None`, and
   `vllm/config/vllm.py:964-1004` turns it on unless something is incompatible.
   The plugin only turns it off for a model that does not declare
   `supports_async_decode` (`platform.py:955-968`). So `supports_async_decode`
   is not a nice-to-have flag here — it is what keeps async scheduling on.
   `Asynchronous scheduling is enabled.` appears in this stage's server logs and
   in **stage 08's retained `readiness_vllm/server.log`** alike.

So it was never measured. This stage measured it, by turning it off — and then,
after review, **measured it again properly**, because the first attempt got the
size of the effect badly wrong. Both measurements are retained; the second one is
the result.

### Why the first A/B had to be redone

The first pass compared the two legs' **mean** TPOT and reported the async split
as worth **1.754 ms/token, 8.9 %**. That figure was an artifact of a single
number, and the artifact said so if read carefully:
`bench/single_user_no_async_vllm_result.json` reports `mean_itl 21.5553`,
`median_itl 20.2449`, `p99_itl 21.0446`, `std_itl 14.1009`. **`p99` below `mean`
is impossible without a large excess above the 99th percentile.** Solving those
moments for 127 ITLs gives 126 tokens at ~20.30 ms plus one at **~179.8 ms**. The
async leg's `std` over the identical workload is **0.4025** — the leg that
decided the conclusion carried **35x** the dispersion of the leg it was compared
against, and that was never classified. Worse, the summary JSON kept only
moments, so this had to be *solved for* rather than read.

The re-measurement therefore (a) passes `--save-detailed` through to
`vllm bench serve` so **every per-token ITL is retained**, (b) repeats each leg
**three times** at 128 output tokens, and (c) adds a **512-token** output run so
no single event can carry a mean.

### What the re-measurement found

The ~180 ms event is real, and the inference above was accurate to a fraction of
a millisecond. But it is **not an outlier** — it is deterministic, it happens
**once per request**, and it always lands at **ITL index 0**, the gap between the
first and second output tokens:

| leg | run | n | mean ITL | **median ITL** | std | stalls (>2x median) |
|---|---|---|---|---|---|---|
| async **on** | 128 r1 | 127 | 19.8057 | 19.7991 | 0.196 | none |
| async **on** | 128 r2 | 127 | 19.8069 | 19.8036 | 0.168 | none |
| async **on** | 128 r3 | 127 | 19.8100 | 19.7993 | 0.739 | none |
| async **on** | 512 r1 | 511 | 19.8990 | 19.8918 | 0.350 | none |
| async **off** | 128 r1 | 127 | 21.5784 | 20.2574 | 14.047 | **1 @ index 0, 179.178 ms** |
| async **off** | 128 r2 | 127 | 21.5775 | 20.2371 | 13.978 | **1 @ index 0, 178.405 ms** |
| async **off** | 128 r3 | 127 | 21.5078 | 20.2145 | 14.087 | **1 @ index 0, 179.621 ms** |
| async **off** | 512 r1 | 511 | 20.7033 | 20.3205 | 7.094 | **1 @ index 0, 180.491 ms** |

[`bench/async_ab/`](bench/async_ab/) (eight runs, per-token ITLs retained),
[`bench/async_ab_summary.json`](bench/async_ab_summary.json),
[`probes/async_ab_summary.py`](probes/async_ab_summary.py),
[`logs/async_ab_summary.log`](logs/async_ab_summary.log), server logs
[`logs/server_remeasure_async_on.log.gz`](logs/server_remeasure_async_on.log.gz)
and [`logs/server_remeasure_async_off.log.gz`](logs/server_remeasure_async_off.log.gz).

**That stall is the TTFT difference, moved.** This is the whole finding, and it
is arithmetic, not interpretation:

| | async **on** | async **off** |
|---|---|---|
| TTFT | 300.796 ms | 141.592 ms |
| ITL[0] (median of 3) | 20.000 ms | 179.178 ms |
| **TTFT + ITL[0]** | **320.796 ms** | **320.770 ms** |

The two totals agree to **0.026 ms**. There is one fixed ~159 ms per-request cost
— the eager prefill's decode-trace capture — and async scheduling does not
remove it, add to it, or overlap it away. It only decides **which bucket it is
billed to**: TTFT with async on, the first inter-token latency with async off.

This also disposes of the explanation the first pass offered. It said async
"defers the first output frame by a scheduler step"; a scheduler step here is
19.8 ms, which cannot account for 153 ms — it is out by a factor of eight. The
correct statement is about *ordering*, not deferral: with async scheduling
**off**, vLLM releases the first output frame **before** the decode trace has
been captured, so the capture falls between token 1 and token 2 and is billed to
ITL[0]; with it **on**, the frame is released after, and the capture is billed to
TTFT. Neither mode avoids the capture.

### The corrected A/B

| 128/128/1, `max_num_seqs=1` | async **on** (default, shipped) | async **off** | Δ |
|---|---|---|---|
| **Steady-state ITL** (median of 3 runs) | **19.7993 ms** | 20.2371 ms | **0.4378 ms/token, 2.16 %** |
| Decode t/s/u (from steady-state ITL) | **50.507** | 49.414 | **+2.21 %** |
| Mean TPOT *(carries the one-off stall)* | 19.8075 ms | 21.5545 ms | 1.7470 ms — **do not use** |
| TTFT | 300.796 ms | **141.592 ms** | −159.204 ms |
| TTFT + ITL[0] | 320.796 ms | 320.770 ms | 0.026 ms |
| **End-to-end, 128 tokens** | **2816.667 ms** | 2881.932 ms | **65.265 ms, 2.26 %** |
| **End-to-end, 512 tokens** | **10468.317 ms** | 10726.216 ms | **257.899 ms, 2.40 %** |
| `read_decode_output(async_read=True)` taken | **yes** (logged) | **no** (0 occurrences) | |

**The honest per-token gain is ~0.44 ms, 2.2 % — not 1.754 ms, 8.9 %.** The
corrected figure matches what this README's own ITL-median row said all along
(20.2449 − 19.8087 = 0.436), which the first pass never reconciled with its
mean-derived number. The e2e framing, which is immune to which bucket the one-off
lands in, agrees independently at 2.26 % and 2.40 %.

**Every downstream figure changes with it:**

| Figure | first published | corrected |
|---|---|---|
| async split worth | 1.754 ms/token | **0.438 ms/token** |
| as a share of decode t/s/u | 8.9 % | **2.21 %** |
| vLLM per-token host cost over the 19.213 ms standalone floor | 2.342 ms | **1.024 ms** |
| serving overhead **with** the split | 0.588 ms | **0.586 ms** (the same, within noise) |
| share of host cost the split hides | 75 % | **42.7 %** |
| async-off TTFT advantage | 153.4 ms | **159.2 ms**, but see below |

The serving-overhead headline is the one figure that survives, because it was
never derived from the defective leg: it is the async-**on** steady state against
the standalone floor, and both of those were always sound. 0.588 ms came from the
original run's mean TPOT and 0.586 ms from the re-measured median — the same
number twice.

### Recommendation, restated

**It does not invert — but the trade the first pass described does not exist.**

Async **on** is better on *every* total-latency axis: 2.2 % faster steady-state
decode, 65 ms faster end-to-end at 128 output tokens, and 258 ms at 512. The
one-off cost is the same to within **0.03 ms** (320.796 against 320.770 ms) --
async scheduling does not add or remove it, it only moves which bucket bills it.
There is no output length at
which async off produces a complete response sooner.

What async **off** genuinely buys is a **first token 159 ms sooner** — and then a
**179 ms gap before the second one**. For a streaming UI judged on time-to-first-token
that may still be the right trade; for anything judged on time-to-complete-response,
or on tokens per second, it is a strictly worse setting. The first pass framed this
as "153.4 ms of TTFT for 1.754 ms/token", which overstated the price by ~4x and
concealed that the 153.4 ms is not saved at all, only deferred by one token.

**Shipped: vLLM's default, async on.**

## The 32-slot collapse: the control, and what it says

Stage 08's biggest open item was that a `max_num_seqs=32` server serves **one**
user at 3.796 t/s/u against 50.560 at `max_num_seqs=1`, and attributed it to MoE
decode batch scaling in `tt/model.py` rather than to the adapter — from
measurements taken entirely inside vLLM. There was no standalone batch-32 decode
anywhere in stages 01–08.

[`probes/batch_decode_control.py`](probes/batch_decode_control.py) is that
control: the generator at `max_batch_size=32`, *k* rows prefilled, the other
`32-k` carrying the inactive sentinel `current_pos = -1`, timed around
`ttnn.execute_trace`. No vLLM, no HTTP, no profiler.

| live rows of 32 | `token_out` **before** | `token_out` **after** |
|---|---|---|
| 1 | 262.562 ms | **229.202 ms** |
| 2 | 262.606 ms | 230.396 ms |
| 4 | 262.619 ms | 232.949 ms |
| 8 | 262.810 ms | 238.089 ms |
| 16 | 262.583 ms | 248.295 ms |
| 32 | 268.556 ms | 268.737 ms |

[`probes/batch_decode_control_before.json`](probes/batch_decode_control_before.json),
[`probes/batch_decode_control_after.json`](probes/batch_decode_control_after.json).

**Before, the curve is flat.** One live user costs the same as sixteen, and
thirty-two costs 2.3 % more than one. So stage 08's phrase "MoE decode batch
scaling" was describing the right op and the wrong quantity: the cost did not
scale with the *work*, it was simply always paid.

**The adapter is exonerated quantitatively.** Standalone at 32 slots with one
live row is 262.562 ms; the same 128/128/1 workload *served* is 263.445 ms. All
of vLLM — request handling, scheduling, sampling translation, page-table
refresh, readback — is **0.883 ms, 0.34 %**. After the change the pair is
229.202 / 230.079 ms, **0.877 ms** — the same absolute overhead, which is what an
orchestration cost should do.

**The two sides are not configured identically, and the difference should be
visible rather than buried in a subtraction.** The standalone control runs
`context 4096`, which gives the generator `pages_per_user 128`; the served side
runs `max_model_len 262144` with a `[1, 8192]` page table. So this is an
*estimate* of adapter overhead, not a controlled difference: the served step
carries a 64x wider page table through its per-token `torch.equal` and a
correspondingly larger rotary table. Both of those are host-side and small
against a 230 ms step — which is the reason the subtraction still lands at a
sane sub-millisecond figure — but the number should be read as "under a
millisecond, at two different context configurations", not as a measurement of
the same binary twice.

### What was taken: inactive-row expert gating

An inactive decode row still embeds a token, still runs attention, and still
routes to a full top-8 of experts. Those `(row, expert)` pairs land in
`ttnn.sparse_matmul`'s sparsity tensor and cost real weight reads and real math
for a row whose output vLLM discards. `Qwen3CoderModel._decode_active_mask`
builds a `[1,1,batch,1]` mask **on device, inside the traced graph**, from the
`current_pos` trace input, and `decoder_layer_decode_multichip` multiplies the
routing vector by it — so an inactive row contributes exactly zero to the
sparsity.

Why on device rather than as another trace input: `current_pos` is already
persistent, the traced graph advances it with
`ttnn.plus_one(..., skip_negative_entries=True)`, and that flag leaves an
inactive row at `-1` through any number of replays. A mask derived from it inside
the same graph is correct on every replay with no refresh and no way to go stale.
A host-supplied mask would be one more thing that has to be right.

**Measured, on the served workload** (128/128/1, one active user, `max_num_seqs=32`):

| | Before | After | Δ |
|---|---|---|---|
| TPOT mean | 263.445 ms | **230.079 ms** | **−33.366 ms, −12.7 %** |
| **Decode t/s/u** | 3.796 | **4.346** | **+14.5 %** |
| ITL median / P99 | 263.322 / 266.188 ms | 230.067 / 232.315 ms | −33.255 ms |
| TTFT | 352.492 ms | 352.130 ms | −0.362 ms |
| Aggregate output | 3.786 tok/s | 4.328 tok/s | +14.3 % |

[`bench/maxnumseqs32_before_vllm_benchmark.json`](bench/maxnumseqs32_before_vllm_benchmark.json),
[`bench/maxnumseqs32_after_vllm_benchmark.json`](bench/maxnumseqs32_after_vllm_benchmark.json).

At **full** occupancy it is exactly break-even, which is the correct behaviour
and is measured rather than assumed: standalone 268.556 → 268.737 ms at 32 live
rows, and the CI burst below moves by 0.1 %.

**It changes no token.** [`probes/inactive_row_gating_probe.py`](probes/inactive_row_gating_probe.py)
runs four real text prompts through the same generator twice, gated and ungated,
and compares greedy token sequences:

| Check | Result |
|---|---|
| `live_rows_token_identical` | 4 rows x 24 tokens, **identical token for token** |
| `outputs_are_varied` | 16–21 distinct tokens per row, all four rows different — so the equality above has teeth |
| `mask_matches_positions` | the mask read off the device equals `current_pos >= 0` for the live trace's own `current_pos` |
| `mask_survives_replays` | after 23 replays advanced every live row, the mask is still exactly the original active set |

[`probes/inactive_row_gating_probe.json`](probes/inactive_row_gating_probe.json).

### The win did not survive request churn — a defect found in review, and fixed

Everything measured above is **single-request, single-install**: one request
admitted into a 32-slot server and decoded to completion. A real server recycles
slots, and every recycle re-installs host state into the live decode trace
through `Qwen3CoderForCausalLM._merge_scheduler_view`. The version this stage
first shipped ended with

```python
merged_positions = torch.where(continuing, device_positions, torch.clamp(host_positions, min=0))
```

An inactive row arrives from the plugin as `-1` (`model_runner.py:969-970` pads
decode positions with `-1` "to indicate no position") and is never `continuing`,
so that `clamp` installed it as **0** — and since `_decode_active_mask` derives
the mask from `current_pos >= 0`, position 0 reads as **live**. The gating became
a no-op for every unoccupied slot.

It escaped every measurement because `_merge_scheduler_view` returns
`host_positions` untouched when there is no decode device state yet, which is
the case on the **first** install. The clamp is only reached from the second
install onward — i.e. exactly on turnover, which nothing in this stage measured.

[`probes/churn_occupancy_control.py`](probes/churn_occupancy_control.py) is the
missing control: the real adapter over a real generator, 4 of 32 slots occupied,
then three slot recycles that each admit a new request onto fresh physical
blocks. `--legacy-clamp` reinstates the shipped expression, so both legs are one
artifact.

| | inactive rows still at `-1` | `token_out` |
|---|---|---|
| **shipped (`--legacy-clamp`)** initial install | 28/28 | 232.171 ms |
| — after recycle 1 | **0/28** | **264.791 ms** |
| — after recycle 2 | **0/28** | 264.730 ms |
| — after recycle 3 | **0/28** | 264.703 ms |
| **fixed** initial install | 28/28 | 232.147 ms |
| — after recycle 1 | **28/28** | **232.192 ms** |
| — after recycle 2 | **28/28** | 232.171 ms |
| — after recycle 3 | **28/28** | 232.164 ms |

**The entire gating win evaporated on the first slot recycle**: +32.619 ms drift,
from 232.171 ms back to 264.791 ms, which is within 4 ms of the 268.737 ms
full-occupancy cost. After the fix the drift is **0.045 ms** and the curve is
flat across recycles. The inactive rows read back at position `1`, not `0`, in
the legacy leg — installed at 0, then advanced by the traced
`ttnn.plus_one(..., skip_negative_entries=True)`, which correctly declines to
skip a row that is no longer negative. The sentinel was the only thing holding
the mask up.

The fix preserves the sentinel rather than clamping it, and normalises any
negative to exactly `-1`:

```python
host_positions_kept = torch.where(host_positions < 0, torch.full_like(host_positions, -1), host_positions)
merged_positions = torch.where(continuing, device_positions, host_positions_kept)
```

[`probes/churn_occupancy_control_legacy.json`](probes/churn_occupancy_control_legacy.json)
(exit 1, as it should),
[`probes/churn_occupancy_control_fixed.json`](probes/churn_occupancy_control_fixed.json)
(exit 0), [`logs/churn_occupancy_control.log`](logs/churn_occupancy_control.log).

**The published serving numbers are unaffected**, because they were all taken at
a single install — the 3.796 → 4.346 t/s/u result stands. What the fix changes is
that the win now *holds* on a server that churns, which is the only kind that
matters. Token identity was re-verified after the change (below).

**The chat-templated suite was re-collected too.** This checkpoint declares a
chat template and the shared runner sends raw `/v1/completions`, so for an
instruct model the chat form is the canonical one — and
`readiness_vllm/vllm_qualitative_chat_outputs.json` was still stage 08's.
Re-collected against this stage's 32-slot server
([`probes/collect_chat_qualitative.py`](probes/collect_chat_qualitative.py) →
[`logs/vllm_qualitative_chat_outputs.json`](logs/vllm_qualitative_chat_outputs.json)):
two of the six greedy completions are **byte-identical** to stage 08's and the
other four are stage 08's completion **continued** — this collection used a
256-token budget where stage 08's used a smaller one, so every one of the six is
the same greedy token sequence for as far as stage 08 generated. Not a byte
differs before stage 08's stopping point.

End to end, all **six greedy qualitative completions are byte-identical to
stage 08's** ([`logs/vllm_qualitative_outputs.json`](logs/vllm_qualitative_outputs.json)
against `../../readiness_vllm/vllm_qualitative_outputs.json`).

### What is left, and why it was not taken

The after-curve is `≈ 227.9 + 1.28 x live_rows`. So of the 268.737 ms a full
32-slot decode step costs, **~227.9 ms is fixed** in the number of *configured*
rows and only **~40 ms** is attributable to the users. At `max_num_seqs=1` the
whole served step is 19.801 ms. That fixed term is the decode graph being 32 rows wide:
`ttnn.sparse_matmul` with `nnz=None` still visits every `(row, expert)` slot to
read its validity flag even when it does no math there (its own docstring says
so), the replicated router runs a 128-wide `topk` per row on one core, and paged
SDPA runs 32 users' windows.

Removing that needs a **variable-width decode graph** — capturing decode at
several row counts and compacting the live rows into the narrowest one, with the
page table, positions, tokens, sampling parameters and outputs remapped around
it. That is a real redesign with real correctness surface (the sampler addresses
32 fixed slots; penalties are staged per slot), it was not attempted here, and it
is recorded as the next cut with the curve above sizing it. **The honest summary
for an operator today is that `max_num_seqs` is a latency decision, not just a
capacity one.**

## Secondary — CI serving burst (vLLM-nightly shape)

Workload: **100-token input, 100-token output, 32 prompts, no explicit
`--max-concurrency`, `max_num_seqs=32`, `ignore_eos`, greedy**. Same model, mesh
and TT config.

| Metric | Before | After |
|---|---|---|
| **Aggregate output throughput** | 105.589 tok/s | **105.471 tok/s** |
| TTFT median / P99 | 4452.051 / 4453.450 ms | 4472.942 / 4474.009 ms |
| TPOT mean / P99 | 261.196 / 262.503 ms | 261.330 / 262.647 ms |
| ITL median / P99 | 261.134 / 263.947 ms | 261.224 / 263.738 ms |
| TPOT-derived per-user decode | 3.829 t/s/u | 3.827 t/s/u |
| Completed | 32/32, 3200 output tokens | 32/32, 3200 output tokens |

[`bench/maxnumseqs32_before_vllm_ci_serving_benchmark.json`](bench/maxnumseqs32_before_vllm_ci_serving_benchmark.json),
[`bench/maxnumseqs32_after_vllm_ci_serving_benchmark.json`](bench/maxnumseqs32_after_vllm_ci_serving_benchmark.json).

**This is not the headline decode number.** Every request is admitted in a burst,
so each TTFT queues behind 31 other prefills. It is here as capacity and
nightly-parity evidence, and its job this stage is to show the gating change is
**neutral at full occupancy**: 0.1 % on aggregate throughput, 0.05 % on TPOT.
That is the same result the standalone control predicts at 32 live rows, from a
completely different harness.

## The penalised path — re-examined, not taken

Stage 08 left an "incremental operand update" as the next cut, blocked on a
"same request still in this slot" key the adapter does not receive. The brief
asked whether vLLM's request ids or slot mapping can supply one.

**The key exists.** The adapter already derives exactly that continuity for
`_merge_scheduler_view`: a slot is the same request iff its device position is
continuous with the scheduler's *and* its page-table row is unchanged. Row *r*'s
`prompt_tokens` being unchanged with `output_tokens` extended by one is a second,
independent key, and both are per-step host comparisons over a few hundred ints.

**It still does not pay, and stage 08's own measurement is why.** Stage 08 re-timed
its staging at a serving-sized 256-token history and got 1.5674 / 3.7624 ms
against the correctness batch's 1.5351 / 3.3894 ms, noting that it "barely
moves, because the staging is dominated by the fixed 9.7 MB operand, not by the
history length". An incremental update removes history re-derivation — the part
that barely moves — and leaves the full-width upload, which is the part that
costs. The cut that would reach the upload is an on-device scatter of the changed
columns, a different and larger piece of work than the one that was blocked.

So this stage **does not adopt it**, and records that the stated blocker was not
the real one. The penalised-path figures from stage 08 stand unchanged
(44.049 t/s/u with `repetition_penalty`, 40.079 with all three, against 50.321
unpenalised); all four non-presence `test_tt_penalties` still pass in this
stage's sampling run.

## Where the 0.588 ms goes

Serving decode is 19.801 ms against a 19.213 ms standalone traced token-out.
Stage 08 listed the adapter's per-token host work and asserted there was nothing
left; this stage has an A/B instead. With the async split disabled the same path
costs **20.237 ms in steady state**, so **vLLM's per-token host cost is 1.024 ms
and the async split hides 0.438 ms — 42.7 % — of it**. (An earlier version of
this section put those at 2.342 and 1.754 ms, from the async-off leg's *mean*
TPOT, which carries a one-off ~179 ms first-ITL stall; see "Why the first A/B had
to be redone".) What remains inside the 0.588 ms is the deferred
readback of the 128-byte sampled-token tensor, its event synchronisation, and the
plugin's own per-step marshalling (`submit_decode` rebuilds `TTSamplingParams`
from `.tolist()` on every field every step, in the plugin, which this stage does
not modify).

On the adapter's side the steady-state per token is: two
`ttnn.execute_trace(..., blocking=False)` calls, one `torch.equal` over the
`[1, 8192]` int32 page table, one 128-byte readback, and a `set_sampling_params`
that returns early on an unchanged snapshot. The mechanical evidence is
[`probes/adapter_contract_probe_after.json`](probes/adapter_contract_probe_after.json)
— stage 08's own 13-check probe, re-run unchanged against this stage's code at
`max_num_seqs=4` so the **gated** graph is the one under test: **13/13 pass, 0
failed**, with `token_host_copies +0`, `position_host_copies +0`,
`rotary_position_host_copies +0`, `page_table_host_copies +0` for an unchanged
table and `+1` for a changed one, `caller_token_readbacks +8` over 8 tokens, and
`captures/releases/warmups +0` in steady state.

## Gates

| Gate | Result |
|---|---|
| Primary single-user benchmark | 1/1 completed, 128 output tokens, both legs |
| CI serving burst | 32/32 completed, 3200 output tokens, both legs |
| Sampling suite (`--sampling-profile full`, `--tt-max-num-seqs 32`) | **57 / 15 / 1** and **56 / 16 / 1** on the shipped code; **58 / 14 / 1** with the merge fix reverted. Same two failure classes throughout. **Not fully resolved — see below.** |
| Qualitative | 6 prompts, greedy byte-identical to stage 08; `check_degenerate_output.py --scope all` → `No degenerate output detected.`, exit 0 |
| Model suite (`-m "not models_performance_bare_metal"`) | **158 passed, 16 deselected**, exit 0 — the stage bar exactly ([`logs/stage09_model_suite.log`](logs/stage09_model_suite.log)) |
| Stage gate `09-optimized-vllm.check.sh` | both halves exit 0, [`logs/stage09_gate_09-optimized-vllm.check.log`](logs/stage09_gate_09-optimized-vllm.check.log) |
| Non-aligned prompt lengths | 37 / 131 / 333 / 1025 / 4097 token ids + one natural-language prompt, all served with `usage.prompt_tokens` equal to the request |
| Turnover control (`probes/churn_occupancy_control.py`) | fixed leg **exit 0**, drift 0.045 ms across three slot recycles; `--legacy-clamp` leg **exit 1**, drift +32.619 ms — the defect, exhibited |
| Adapter contract, re-run after the counter fix | **13/13, 0 failed**; `async_decode_reads` 19 + `sync_decode_reads` 1 == 20 decode steps, an exact partition |
| Token identity after the merge fix | `inactive_row_gating_probe.py` **4/4 pass**, including `live_rows_token_identical` |
| Async A/B, re-measured | 8 runs with per-token ITLs retained; stall present once in all 4 async-off runs, absent from all 4 async-on |
| Chat-format qualitative | 6 prompts re-collected on this stage's 32-slot server; every greedy completion is stage 08's **exactly or as a strict prefix** ([`logs/vllm_qualitative_chat_outputs.json`](logs/vllm_qualitative_chat_outputs.json)) |
| Context contract | 262144, unchanged and unreduced |

**The failure *classification* is preserved; the failure *count* is not, and that
is an open item rather than a clean pass.**

Four runs of the full sampling profile, all at `--tt-max-num-seqs 32`:

| Run | code | result | failures beyond the stage-09 baseline set |
|---|---|---|---|
| stage-09 archived ([`logs/sampling_tests_prefix_stage09_original.log`](logs/sampling_tests_prefix_stage09_original.log)) | before the review fixes | **58 / 14 / 1** | — |
| isolation ([`logs/sampling_tests_isolation_no_merge_fix.log`](logs/sampling_tests_isolation_no_merge_fix.log)) | shipped code with **only** the `_merge_scheduler_view` sentinel fix reverted | **58 / 14 / 1** | none; one *swap* within the class (`test_specific_seed_reproducible[999]` → `[42]`) |
| shipped run 1 ([`logs/sampling_tests.log`](logs/sampling_tests.log)) | shipped | **57 / 15 / 1** | `test_specific_seed_reproducible[42]` |
| shipped run 2 ([`logs/sampling_tests_shipped_run2.log`](logs/sampling_tests_shipped_run2.log)) | shipped | **56 / 16 / 1** | `+ test_topk[19]` |

**What is solid.** Every failure in every run is in the same two classes stage 08
shipped: `test_seeding_and_variety.py` seeding/variety assertions, plus the two
`TestPresencePenalty` tests. **No test outside those classes ever failed**, and
the deterministic path is provably untouched — `live_rows_token_identical` 4/4,
the six completions-format qualitative outputs byte-identical to stage 08, the
chat-format greedy outputs stage 08's exactly or as strict prefixes, model suite
158 passed, adapter contract 13/13.

**What is not resolved.** Reverting *only* the merge-sentinel fix returned the
count to 14 in a single run, and both runs with it landed at 15 and 16. Two runs
either side is not enough to establish causation, and the two extra failures
point in **contradictory directions** — `test_specific_seed_reproducible` fails
when two runs *differ*, `test_topk` fails when two runs *do not vary enough* — so
a simple "the fix made the device more/less deterministic" story does not hold.
The honest reading is that this is the documented coin-flip class (per-request
seeds are still not honoured, limitation 4), with the merge fix as an
unexcluded contributor. Counted **per class**, every run is inside the 12–14
seeding/RNG band stage 08 recorded — 12, 12, 13 and 14 — with the presence class
exactly 2 throughout and nothing outside the two classes failing in any run. An
earlier revision of this section said the shipped runs land "1–2 above the band";
that was wrong, and wrong against this stage's own interest. For scale: stage 08's
**committed** run was itself 56/16/1, and its failure set differs from stage 09's
worst run by a single `test_topk` parametrization (`[15]` against `[19]`) — two
parametrizations of the same test.

**This was not traded away.** The merge fix is retained because the defect it
repairs is a measured 32.619 ms regression on any server that recycles a slot,
against at most two flaky assertions in a class that is already known-failing for
an unrelated reason. But it should be settled with a repeat-count study (5–10
runs per arm) rather than left at four runs, and it is recorded as an open item.

The baseline 14 failures are **12 seeding/RNG** (`test_seeding`,
`test_same_seeds_reproduce_across_batches`, 3x `test_specific_seed_reproducible`,
4x `test_uniform_seed_deterministic`, `test_batch1_no_seed_varied`,
`test_temperature_varied_between_batches`,
`test_request_isolation::test_mixed_params_batch`) and **2 presence-penalty**
(`TestPresencePenalty::test_different_presence_penalties`,
`::test_presence_penalty_mixed_batch`) — the same two classes stage 08 shipped,
which stage 08 showed fail against vLLM's own reference sampler on this
checkpoint. Stage 08 recorded that the seeding class fluctuates between 12 and 14
run to run without any code change, because a test whose assertion is "two runs
differ" is a coin flip against a fixed device RNG buffer. **No test outside these
two classes moved from passing to failing in any of the four runs.**

## Invariants, checked rather than assumed

**Context contract: 262144, unreduced.** Every server in this stage ran
`--max-model-len 262144`; `doc/context_contract.json` still records
`current_supported_context: 262144` and `capability_reduction: false`, and
nothing in this stage's diff touches context, cache geometry, trace buffers or
any persistent allocation whose size depends on context. The 32-slot server log
echoes `GPU KV cache size: 263,168 tokens` — which is also the first *quoted*
confirmation of the 8224 x 32 figure stage 08 had to derive, because that run's
log was overwritten. [`logs/server_bench_after.log.gz`](logs/server_bench_after.log.gz).

**Non-aligned prompt lengths, re-verified after the change.** Not inherited:
[`probes/non_aligned_prompt_lengths.py`](probes/non_aligned_prompt_lengths.py)
was run against the live optimized server.

| Prompt tokens | ÷8 | ÷32 | ÷64 | ÷128 | ÷1024 | `usage.prompt_tokens` | Result |
|---|---|---|---|---|---|---|---|
| 37 | no | no | no | no | no | 37 | 12 tokens out |
| 131 | no | no | no | no | no | 131 | 12 tokens out |
| 333 | no | no | no | no | no | 333 | 12 tokens out |
| 1025 | no | no | no | no | no | 1025 | 12 tokens out |
| 4097 | no | no | no | no | no | 4097 | 12 tokens out |
| 14 (natural text) | no | no | no | no | no | 14 | 48 tokens out |

[`probes/non_aligned_prompt_lengths.json`](probes/non_aligned_prompt_lengths.json).
Nothing was capped or truncated. The change cannot have affected this in any
case — it touches only the decode routing vector — but the brief asks for it
re-verified, so it is.

**Sampling stays on device.** `sample_on_device_mode: all` throughout; the
adapter-contract probe shows the traced sampler is still
`_WatcherCleanSampling1D` from `tt/model.py`, `token_host_copies` is 0 in steady
state, and the only readback on the token path is the 128-byte sampled-token
tensor. No host argmax, no full-logits readback, no eager sampling on the
measured path.

**Batch capability preserved.** `max_num_seqs` 1 and 32 both measured; the 32-slot
server ran the full sampling suite, the qualitative suite and both benchmark
profiles.

## Runtime fallback and process-cleanup audit

**Fallbacks on the measured path: none.** `serving_audit()` and
`Qwen3CoderModel.runtime_fallback_audit()` are captured in
[`probes/adapter_contract_probe_after.json`](probes/adapter_contract_probe_after.json).
The new decode ops are `ttnn.to_layout` / `ttnn.typecast` / `ttnn.gez` /
`ttnn.transpose` on a `[1,1,1,32]` tensor once per step plus one broadcast
`ttnn.mul` per layer, all **inside** the captured trace: no `torch`, no
`from_torch`/`to_torch`, no host round trip, no reshard, no extra readback and no
extra synchronisation. The mask adds **zero** host work per token — it is derived
from a tensor the trace already owns.

Two audit counters were added so the async claim is countable rather than
argued: `async_decode_reads` (incremented in `read_decode_output`, with a
one-time log line) and `sync_decode_reads` (incremented when
`process_decode_output_host` is handed a **device-resident** tensor, i.e. the
plugin's synchronous path).

**`sync_decode_reads` was mis-implemented when this stage first shipped, and the
probe published the contradiction.** The counter sat behind
`if not isinstance(tt_out, torch.Tensor)`, which is **dead**: the `torch.Tensor`
case returns two lines earlier, so the guard was true on every step. And the
async path's `read_decode_output` returns `tt_out.cpu(blocking=False)` — a ttnn
**host** tensor, not a `torch.Tensor` — so async reads were counted as
synchronous ones too. The old
`probes/adapter_contract_probe_after.json` showed it plainly:
`device_sampled_decode_steps: 20`, `async_decode_reads: 19`,
`sync_decode_reads: 20` — 39 reads across 20 steps.

The discriminator is now device residency
(`ttnn.is_tensor_storage_on_device(tt_out)`), which is the actual question being
asked. Re-running the same probe unchanged gives
`device_sampled_decode_steps: 20`, `async_decode_reads: 19`,
`sync_decode_reads: 1` — **19 + 1 = 20**, an exact partition of the decode steps,
with the single synchronous read being the one step the probe drives through the
plugin's synchronous path deliberately. 13/13 contract checks still pass
([`logs/stage09_review_probes.log`](logs/stage09_review_probes.log)).

**Process cleanup.** Every server was shut down with `pkill -f
readiness_check.run_vllm_server`, then `pkill -9 -f VLLM::EngineCore` and
`pkill -9 -f vllm.entrypoints`, then verified with `ps aux`; two servers needed an
explicit `kill -9` on the surviving pids, which is recorded in the work log.
Final state: **0 vLLM, EngineCore or entrypoints processes, and nothing holding
`/dev/tenstorrent/*`**. No device reset was needed in this stage. No Tracy,
`tt-perf-report`, `TT_METAL_DEVICE_PROFILER` or `ttnn.ReadDeviceProfiler` run was
made, against a live server or otherwise.

**The plugin checkout is untouched.** `/home/raahem/vllm-tt-plugin` is
byte-identical to `bc4af2d`.

## Server commands

Common to all:

```bash
source python_env/bin/activate
export EXTRA_MODELS_DIR=$PWD/models/autoports/qwen_qwen3_coder_30b_a3b_instruct/vllm_bundle
```

Shipped configuration (the "after" rows above):

```bash
python -m models.common.readiness_check.run_vllm_server \
  --model-dir <output-dir> \
  --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --mesh-device P300x2 --max-num-seqs 1 --max-model-len 262144 \
  --block-size 32 --port 8100 --stages serve \
  --tt-config '{"trace_region_size": 50331648, "fabric_config": "FABRIC_1D_RING"}' \
  --additional-server-args "--generation-config vllm"
# then, against that live server:
python -m models.common.readiness_check.run_vllm_server --stages benchmark \
  --model-dir <output-dir> --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --max-num-seqs 1 --server-url http://localhost:8100 --no-benchmark-ci-serving
```

`--max-num-seqs 32` for the CI burst, sampling and qualitative stages.
`QWEN3_DECODE_ACTIVE_ROW_GATING=0` in front of the launch reproduces the
stage-08 decode graph and is how every "before" row was taken.
`--additional-server-args "--generation-config vllm --no-async-scheduling"` is the
async-off leg.

**Retaining per-token ITLs.** The A/B re-measurement adds
`--additional-benchmark-args="--save-detailed"` to the benchmark stage, which the
runner passes straight through to `vllm bench serve`. Without it the runner
deletes the `itls` list before saving and only the moments survive — which is why
the original stall had to be inferred from `p99 < mean` instead of read. Note the
`=` form: argparse consumes a bare `--save-detailed` as a flag of its own.

```bash
python -m models.common.readiness_check.run_vllm_server --stages benchmark \
  --model-dir <scratch> --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --max-num-seqs 1 --server-url http://localhost:8100 --no-benchmark-ci-serving \
  --benchmark-prompt-len 128 --benchmark-output-len 128 \
  --benchmark-num-requests 1 --benchmark-concurrency 1 \
  --additional-benchmark-args="--save-detailed"
```

**`--model-dir` is a scratch directory on purpose.** The runner writes
`server.log`, the benchmark JSON and the qualitative JSON under
`<model-dir>/readiness_vllm/`, and stage 08's committed README cites exactly
those files. Pointing the runner at a scratch tree and copying what this stage
needs into `doc/optimized_vllm/` keeps the stage-08 evidence intact.

## Limitations and open items

1. **Per-user decode at `max_num_seqs=32` is still ~11.6x slower than at 1**
   (230.079 ms against 19.801 ms). ~227.9 ms of that is fixed in the *configured*
   row count, not the live one — see the control curve. The remaining cut is a
   variable-width decode graph; not attempted here.
2. **Async scheduling is not the TTFT/decode trade this stage first published.**
   The measured steady-state difference is **0.438 ms/token (2.16 %)**, not
   1.754 ms, and the ~159 ms of TTFT that async scheduling costs is **not saved**
   by turning it off — it reappears in full as a ~179 ms first inter-token
   latency (`TTFT + ITL[0]` is 320.796 ms async-on against 320.770 ms async-off).
   Async on is faster end-to-end at every output length measured (2.26 % at 128
   tokens, 2.40 % at 512). vLLM's default (on) is shipped; `--no-async-scheduling`
   is worth it only if time-to-*first*-token is the metric and the 179 ms gap
   before the second token is acceptable.
3. **A penalised request still decodes 14–26 % slower** (stage 08 figures,
   unchanged). The blocker stage 08 named turned out not to be the real one; the
   real one is the fixed full-width operand upload.
4. **Per-request seeds are still not honoured** (14–16 sampling failures across
   four runs; the classes are unchanged from stage 08 and, counted per class,
   every run is inside the 12–14 seeding/RNG band stage 08 recorded — stage 08's
   own committed run was 56/16/1 — with the `_merge_scheduler_view` sentinel fix
   an unexcluded contributor to which end of the band a run lands on — see the
   sampling gate discussion; needs a 5–10 run
   repeat-count study per arm to settle). `Sampling1D.decode_forward` takes a `seeds=` tensor;
   wiring it into the traced decode input set was not in this stage's scope.
5. **The token sampled by prefill is still not penalised**, **`top_k` is still
   clamped to 32**, **prefix caching is still off**, **prefill is still eager**,
   and **data parallel > 1 is still rejected** — all unchanged from stage 08 with
   the same reasons.
6. **No profiler evidence, by rule.** `perf_summary.json` carries `null` for
   device time and roofline with the reason named. Device-op context comes from
   the non-serving profiles in `doc/optimized_full_model/`.

## Artifacts

Under `doc/optimized_vllm/`:

| File | What |
|---|---|
| `README.md`, `work_log.md` | this, and the chronological account |
| `perf_summary.json` | the `$optimize` performance-accounting block, with the no-profiler reason |
| `bench/single_user_{before,after,no_async}_vllm_*.json` | primary 128/128/1 at `max_num_seqs=1`: the before/after pair and the async-off leg |
| `bench/maxnumseqs32_{before,after}_vllm_benchmark.json` | 128/128/1 at `max_num_seqs=32` — the slot-count figure |
| `bench/maxnumseqs32_{before,after}_vllm_ci_serving_benchmark.json` | CI serving burst 100/100/32 |
| `probes/batch_decode_control.py`, `batch_decode_control_{before,after}.json` | the standalone batch-32 control and its occupancy sweep |
| `probes/inactive_row_gating_probe.py`, `.json` | token equality against the ungated graph, plus the mask mechanics |
| `probes/non_aligned_prompt_lengths.py`, `.json` | non-aligned prompt lengths against the live optimized server |
| `probes/adapter_contract_probe_after.json` | stage 08's 13 contract checks, re-run on this stage's code with gating active |
| `bench/async_ab/` | the re-measured async A/B: 8 runs (3x128 + 1x512 per leg) with **per-token ITLs retained** |
| `bench/async_ab_summary.json`, `probes/async_ab_summary.py` | the A/B analysis: per-run dispersion, stall classification, and all three framings |
| `probes/churn_occupancy_control.py`, `churn_occupancy_control_{legacy,fixed}.json` | partial occupancy with request turnover; the `--legacy-clamp` leg exhibits the defect this stage shipped |
| `probes/check_published_figures.py` | re-derives every figure in this file from the artifacts above, prints the numbers it does **not** cover, and refuses a mean-based delta drawn from a leg with anomalous dispersion |
| `logs/sampling_tests.log` | the full TT plugin sampling suite, 58 / 14 / 1 |
| `logs/vllm_qualitative_outputs.json` | greedy + sampled completions from the optimized 32-slot server |
| `logs/server_*.log.gz` | four of the five servers this stage launched. **The single-user *before* leg (`max_num_seqs=1`, gating off) has no retained log** — it was not captured at the time and the server is gone; its benchmark JSONs are retained and are what the headline table cites. |
| `logs/stage09_model_suite.log` | the model test suite |
| `logs/stage09_gate_09-optimized-vllm.check.log` | both halves of the stage gate |
| `logs/check_published_figures.log` | the figure-verification gate's own output, including the coverage boundary |
| `logs/async_ab_summary.log`, `logs/server_remeasure_async_{on,off}.log.gz` | the A/B re-measurement's analysis output and its two server logs |
| `logs/churn_occupancy_control.log`, `logs/stage09_review_probes.log` | the turnover control, and the post-fix contract + token-identity re-runs |
| `logs/vllm_qualitative_chat_outputs.json` | the **chat-templated** qualitative pass, re-collected on this stage's server |
| `logs/batch_decode_control_{before,after}.log`, `logs/inactive_row_gating_probe.log` | probe run logs |

Source changed by this stage:

| File | What |
|---|---|
| `tt/model.py` | `_decode_active_mask`, `active_row_gating`, passing the mask into every decode layer |
| `tt/multichip_decoder.py` | `decoder_layer_decode_multichip(..., active_mask=)` — one broadcast multiply on the routing vector |
| `tt/generator.py` | `active_row_gating` folded into `_decode_graph_key` |
| `tt/generator_vllm.py` | `async_decode_reads` / `sync_decode_reads` audit counters and the one-time async-split log line |
