# Optimized full model — Qwen3-Coder-30B-A3B-Instruct on 4 Blackhole dies

Stage 06. The stage-05 full model — embeddings, 48 multichip decoder layers,
final norm, column-parallel LM head and on-device traced sampling — profiled at
all 48 layers, optimized on three levers, and re-measured. **No dtype, fidelity,
KV-cache, activation-memory or CCL policy changed**; the three adopted changes
are a program config on an op the model already called, and two rewrites of the
greedy sampler's reduction.

## Headline

Workload **prompt 128 / generate 128 / batch 1**, 48 layers, 1x4 P300_X2,
`FABRIC_1D_RING`. Baseline is the committed stage-05 tree
(`../full_model/probes/perf_full_model.json`); "stage 06" is
`probes/perf_full_model_p128_argmaxrows.json`. Both are 128 timed reps, median,
written by `probes/perf_full_model.py`.

| | stage 05 | **stage 06** | |
|---|---|---|---|
| **TTFT** (prompt 128, warmed, prefill → sampled token in hand) | 126.70 ms | **125.43 ms** | −1.26 ms |
| **traced decode, token-out** (model trace + sampler + on-device feedback) | 22.079 ms — 45.29 t/s/u | **19.693 ms — 50.78 t/s/u** | **1.12x** |
| traced decode, **logits only** (model trace alone, no sampling, no feedback) | 20.211 ms — 49.48 t/s/u | **19.567 ms — 51.11 t/s/u** | 1.03x |
| traced decode, token-out **+ per-token host readback** | 22.748 ms — 43.96 t/s/u | **19.710 ms — 50.74 t/s/u** | 1.15x |
| **teacher-forcing decode** (`run_teacher_forcing`, a *different* measurement) | 38.50 t/s/u | **42.25 t/s/u** | 1.10x |
| greedy sampler, eager, same logits | 1.125 ms | **0.928 ms** | 1.21x |
| split top-k sampler, eager — **the control** | 6.155 ms | 6.155 ms | unchanged |
| cold TTFT, first pass through a fresh mesh | 1115.16 ms | 221.49 ms | a compile-cache figure, not a model figure |

**Teacher-forcing decode and token-out decode are different numbers and both are
reported.** `run_teacher_forcing` uploads a forced token from the host and reads
the prediction back on *every* step; it is a correctness gate that happens to
print a rate. Token-out is the serving steady state — the sampled token becomes
the next decode input on device and the host replays two traces and does nothing
else. They moved by 1.10x and 1.12x respectively, and neither is a substitute for
the other.

The **split sampler row is the control**: nothing this stage did touches the
top-k/top-p path, and it does not move. The whole of the sampler gain lands on
the greedy path, which is where the two sampler changes are.

### Decode is now nearly flat in context, and that is the result

Three prompt lengths, same probe, same tree, one lever at a time
(`probes/perf_full_model_p{128,1024,4096}_{before,after,argmaxrows}.json`).
"before" is stage 05 plus the distributed argmax; "after" adds the paged-SDPA
program config; "shipped" adds the live-row slice.

| prompt | before | + paged SDPA | **shipped** | t/s/u shipped |
|---|---|---|---|---|
| 128 | 21.4776 ms | 20.1460 | **19.6925** | **50.78** |
| 1024 | 26.1432 | 20.4268 | **19.9787** | **50.05** |
| 4096 | 42.0623 | 20.9608 | **20.5050** | **48.77** |
| **4096 / 128** | **1.96x** | 1.04x | **1.04x** | |

Before this stage, decode cost grew 1.96x between a 128-token context and a
4096-token one; now it grows 1.04x. That is the property that makes the
advertised 262144-token context *usable* rather than merely allocatable, and it
is the reason `../context_contract.json` was rewritten this stage.

TTFT is unchanged at every length (125.88 → 125.43 at 128, 887.86 → 887.99 at
1024, 3592.58 → 3593.05 at 4096), which is the expected signature of adopting
decode and not prefill.

### One caveat on the baseline comparison, stated rather than folded in

The stage-05 baseline run allocated a **4096**-position KV cache and the stage-06
runs allocate **8192** (`context` in each JSON). That is not a like-for-like
allocation, and it is worth saying why it does not matter: SDPA-decode's cost is
independent of the *allocated* depth, measured directly at
`probes/sdpa_depth_probe.json` — 1024 / 4096 / 16384 / 65536-deep caches read
28.86 / 30.17 / 28.60 / 28.70 us at the same `cur_pos`. What the cost is linear
in is `cur_pos`, which both runs sweep identically (128 → 256). The `_before`
column of the table above is measured at `context` 8192 on the same tree as the
shipped column, so the *stage-06* deltas are like-for-like throughout.

## What changed, in three lines

| # | change | where | worth |
|---|---|---|---|
| 1 | **Distributed argmax.** Greedy stops all-gathering 151936 logit columns; each die reduces its own 37984 and the mesh all-gathers four candidate values and four indices. | `tt/model.py`, `_WatcherCleanSampling1D._sample_argmax` | 1.82x on the sampler standalone; token-out 22.079 → 21.478 |
| 2 | **A program config on paged SDPA-decode.** `SDPAProgramConfig(q_chunk_size=32, k_chunk_size=min(256, cache_depth), max_cores_per_head_batch=16)`, memoised on the compute grid. The paged path had none; the contiguous path already did. | `tt/multichip_decoder.py`, `_sdpa_program_config` | token-out 21.478 → 20.146 at ctx128, **2.01x at ctx4096** |
| 3 | **Reduce the live user rows, not the padding.** The sampler's logit tile is logically 32 rows because `ttnn.sampling` addresses 32 slots; at batch 1, 31 of them are zero-logit padding. The reduction now runs at `max_batch_size` rows and pads back. | `tt/model.py`, `_WatcherCleanSampling1D._sample_argmax` | 631.6 → 250.8 us standalone, **2.52x** on the whole sampler; token-out 20.146 → 19.693 |

Two more levers were **closed with evidence rather than with an attempt** — the
LM head's DRAM-sharded program config and the MoE reduce-scatter skew. Both are
in the rejection ledger below with their arithmetic.

Everything the previous stages established is untouched: TP=4 attention, EP=4
experts at `bfloat4_b`/LoFi, `bfloat8_b` DRAM-sharded projections, replicated
width-sharded residual norms, `bfloat16` paged KV at block 32, `Topology.Ring`
with 2 links prefill and 1 decode, caller-owned persistent collective buffers,
and the replicated `[1,1,B,2048]` inter-layer residual with no collective between
layers. `Topology::Linear` and `num_workers_per_link=1` were not introduced.

## Where the time goes now: a fresh 48-layer decode profile

`probes/profile_full_model_48.py` under `tt-perf-report`'s profiler, on the
**shipped** tree. Artifacts: `ops_perf_full_model_48layer_decode.csv.gz` (the
window), `tt_perf_report_full_model_48layer_decode.txt.gz` (its report),
`rank_full_model_48layer_decode.txt` (the region split),
`probes/profile_summary_decode.json` (the same figures as JSON fields, which is
what `probes/check_published_figures.py` reads).

The three `*_part1_preadoption.*` files beside them are the **superseded**
profile of the pre-adoption tree, kept because
`profile_48layer_work_log.md` — the lever analysis — quotes it throughout. Every
figure in *this* file comes from the unsuffixed, final-state artifacts.

That naming rule is now applied to the performance artifacts too:
`probes/perf_full_model_part1_preadoption.{csv,json}` is the part-1 run and
`probes/perf_full_model.{csv,json}`, the unsuffixed name a `--tag`-less re-run
would write, is the shipped one. **No file whose name reads as "the result"
holds a superseded result.**

### The window is one decode iteration, and that is checked on all four devices

Stage 03 published a window that straddled two decode iterations and invalidated
eight figures. Since then every window here is boundary-checked rather than
eyeballed. `probes/window_full_model_48.py` starts the window at the first of the
last three `EmbeddingsDeviceOperation` rows per device — `decode_hidden` opens
with exactly three embedding gathers per token (the token lookup plus the cos and
sin rows) and nothing else in the graph uses the op — and then asserts **ten
independent tallies per device, on all four devices, forty in total, every one
exact** (`logs/window_full_model_48_final.log`):

| count | op | why |
|---|---|---|
| 96 | `ReduceScatterMinimalAsync` | 2 all-reduces × 48 layers |
| 96 | `AllGatherAsync` | the same 2 all-reduces × 48 |
| 48 | `SdpaDecode` | 1 attention per layer |
| 96 | `SparseMatmul` | the expert pair × 48 |
| 96 | `PagedUpdateCache` | K and V × 48 |
| 3 | `Embeddings` | the boundary itself |
| 1 | `ArgMax` | the distributed argmax's per-die reduction |
| 2 / 2 / 1 | `AllBroadcast` / `Concat` / `Gather` | the two composite 4-wide gathers and the `ttnn.gather` of the local maximum |

**Two of stage 05's window constants are stale and would have failed here.**
Stage 05 expected `2 * layers + 1` all-gathers, the extra one being the old
sampler's single full-vocabulary `AllGatherAsync`. That op no longer exists, and
the distributed argmax's replacement gathers are **4-wide** — at a gather dim of
4 (padded to a 32 tile) `ttnn.all_gather` takes its *composite* path, which
decomposes into `AllBroadcast` + `UntilizeWithUnpadding` + `Permute` + `Concat` +
`TilizeWithValPadding` and is not `AllGatherAsync` at all. The windower was
rewritten around that; it is a finding as much as a fix, and it is the
unexploited lever named under Limitations.

The window is **14048 rows, 3512 ops per device, 4 devices**. Two independent
cross-checks say it is exactly one iteration and not one-and-a-bit:

* summed `DEVICE KERNEL DURATION` per device is **18889.5 / 18888.5 / 18888.3 /
  18886.8 us** — a spread of 2.7 us, **0.014%**. A window one iteration short or
  long would be off by ~19 ms, not by 3 us;
* it lands on an independently measured `token_out` of 19.6925 ms with 0.803 ms
  (4.08%) left over for dispatch and op-to-op gap across 3512 dispatches — 0.23 us
  each.

`--sync-host-device` is **not** used, unlike stage 05's 2-layer capture: stage
05's own report carries the warning that it "inflates every collective", and at
48 layers there are 96 collectives in the window whose absolute cost is the
question.

### The split

Headline split of the 18889.5 us iteration (device 0,
`probes/profile_summary_decode.json` `regions_us`):

| region | us | share |
|---|---|---|
| terminal-pre (token embedding + the two rope cos/sin gathers) | 53.0 | 0.28% |
| **the 48-layer stack** | **18470.0** | **97.78%** |
| terminal-post (final norm, LM head, sampler, feedback) | 366.5 | 1.94% |

**In-model per-layer cost: 384.791 us**, and it is the same on every die to
within 0.14 us (384.791 / 384.732 / 384.785 / 384.652). Top of the per-layer
ranking:

| us/layer | %iter | n | cores | op |
|---|---|---|---|---|
| 41.215 | 10.47% | 48 | 48 | `SparseMatmul 1x1x32x2048 @ 1x32x2048x1536` (gate/up) |
| 40.678 | 10.34% | 96 | 6 | `ReduceScatterMinimalAsync 1x1x32x2048` (×2) |
| 39.513 | 10.04% | 48 | 64 | `SparseMatmul 1x32x32x768 @ 1x32x768x2048` (down) |
| 26.357 | 6.70% | 48 | **1** | `TopK 1x1x32x128` (router top-8 over 128 experts) |
| 23.534 | 5.98% | 96 | 6 | `AllGatherAsync 1x1x32x512` (×2) |
| 16.399 + 16.269 + 9.124 | 10.62% | 144 | 48–110 | `ReshapeView` ×3 — expert-tile compaction |
| 14.048 | 3.57% | 96 | 110 | `Unary 1x32x32x1536` (silu, ×2) |
| 13.154 | 3.34% | 96 | 8 | `LayerNorm 1x1x32x2048` (residual norms, ×2) |
| 9.726 / 9.010 / 6.607 | 6.44% | 144 | 8–80 | QKV / o_proj / router matmuls |
| **8.952** | **2.27%** | 48 | 110 | **`SdpaDecode 1x1x32x128`** |

Terminal-post, the whole of it:

| us | %iter | cores | op |
|---|---|---|---|
| **226.130** | **1.20%** | 108 | `Matmul 32 x 2048 x 37984` — **the LM head**, DRAM-bound at 66.4%, 340 GB/s |
| 41.06 | 0.22% | 1–4 | the two composite 4-wide gathers, summed across their 16 rows |
| 16.943 | 0.09% | 108 | the sampler's pre-argmax `Untilize` over 37984 |
| 10.350 | 0.05% | 110 | `ArgMax 1x1x1x37984` — the per-die reduction |
| 6.583 | 0.03% | 8 | `model.norm` |
| 2.075 / 3.094 | 0.03% | 1 | the `Gather` of the local maximum and the live-row `Slice` |

`probes/profile_summary_decode.json` also carries the two aggregates the next
sections use: **`lm_head_us` 226.130** and **`sampler_us` 126.207** (every op
after the LM head matmul).

### Two ranks changed, and both are the changes this stage made

| op | pre-adoption profile | shipped profile | change |
|---|---|---|---|
| `SdpaDecode` | 20.704 us/layer | **8.952** | **−56.8%** |
| the sampler's `ArgMax` | 366.098 us | **10.350** | **−97.2%** |
| terminal-post, whole | 821.7 us | **366.5** | −55.4% |
| the two composite 4-wide gathers | ~141 us | **41.06** | −70.9% |
| in-model per layer | 396.904 us | **384.791** | −12.113 |
| whole iteration | 19926.5 us | **18889.5** | −1037.0 |

Two of those rows check each other and are worth reading together. The per-layer
saving is **12.113 us** and the `SdpaDecode` saving alone is **11.752 us** —
**97% of it**, with 0.361 us of run-to-run noise across everything else in a
72-op layer. Nothing else in the layer moved, which is what a program config on
one op should look like and what a numerics change would not.

**What the profile does *not* capture, and it is in the conservative direction.**
The profile is one iteration at `cur_pos ≈ 131`; token-out is a median over 128
tokens from position 128 to 256, and SDPA-decode's default cost is linear in
`cur_pos` while the configured cost is nearly flat
(`probes/sdpa_sweep_confirm_bf16.json`: default 23.72 → 37.54 us between 127 and
255, configured 19.00 → 17.48). So the profiled 48 × 11.752 us = 0.564 ms
**understates** the 1.332 ms token-out actually gained from that lever. The
profile is a snapshot of the cheapest position the workload visits.

The sampler lever, by contrast, is position-independent and the two agree
closely: the profile says terminal-post fell by **455.2 us** and token-out fell
by **453.5 us** (20.1460 → 19.6925).

### The ArgMax is no longer the story, and the LM head is

Before this stage the terminal path's largest op was a **366 us** `ArgMax` — the
second-largest op in the whole iteration. It is now **10.35 us**, and the LM head
matmul at **226.13 us** is 61.7% of terminal-post and the only item in it worth a
paragraph. Its `tt-perf-report` advice is unchanged and still unexpressible; see
the ledger.

The **composite 4-wide gather is now the second item** in the terminal block at
41.06 us over 16 rows (`AllBroadcast` 12.707 + 7.264, four
`UntilizeWithUnpadding` each, two `Concat`, two `Permute`, two
`TilizeWithValPadding`). It was ~141 us before the live-row slice, because it was
gathering 32-row tensors to move four numbers. It is 0.22% of the iteration and
it was not pursued; see Limitations.

## Prefill, profiled for the first time

Stage 05 shipped with prefill unprofiled and **disclosed it as a gap**; this
stage closes it. `probes/profile_full_model_48_prefill.py` runs a warm-up
48-layer prefill of a 128-token prompt — the length the published TTFT is
measured at — and then **two** measured ones, and
`probes/window_full_model_48_prefill.py` publishes the last of the two and
checks it row for row against the one before it. The comparison is between two
passes, not three. Artifacts: `ops_perf_full_model_48layer_prefill_s128.csv.gz`,
`tt_perf_report_full_model_48layer_prefill_s128.txt.gz`,
`probes/profile_summary_prefill.json`, `logs/window_full_model_48_prefill.log`.

**The prefill boundary is checked differently, and more strictly.** Prefill is
eager, not traced, and its rotary reads a slice of the precomputed tables rather
than gathering per position, so there is no three-embedding marker to anchor on.
Instead the window runs from the last of the (exactly one per pass)
`EmbeddingsDeviceOperation` rows to the end of the file, and the pass before it
must be the **identical sequence of op codes, row for row** — 4606 ops per
device, 18424 rows in the published window, matched position by position on all
four. On top of that, **fourteen
per-device tallies, 56 in total, all exact**: 96 reduce-scatters, 96 all-gathers,
48 `SDPAOperation`, 384 `SparseMatmul` (`2 × 48 × ceil(128/32)` — prefill's MoE
walks the sequence in 32-row blocks), 96 `PagedFillCache`, 96 `RotaryEmbedding`,
48 `TopK`, 48 `NlpCreateHeads`, 48 `NLPConcatHeads`, 50 `Concat` (one per layer
plus the sampler's two), and one each of `Embeddings`, `ArgMax`, `Gather`, plus
two `AllBroadcast`.

Per-device kernel time is **122920.8 / 122960.5 / 122943.7 / 122979.8 us**, a
0.048% spread, against an independently measured warmed TTFT of **125.431 ms** —
so **98.0% of TTFT is device kernel time** and 2.5 ms is everything else the host
does for a request.

| share | us | us/layer | op |
|---|---|---|---|
| **39.56%** | 48626.7 | 1013.06 | `SparseMatmul` gate/up, 192 calls (4 sequence blocks × 48) |
| **21.88%** | 26900.0 | 560.42 | `SparseMatmul` down, 192 calls |
| 6.12% | 7526.3 | 156.80 | `Unary 128x1x32x128` |
| 3.76% | 4625.7 | 96.37 | `BinaryNg 1x32x32x2048` |
| 3.13% | 3845.9 | — | **all four collectives together** |
| 1.04% | 1279.3 | 26.65 | `TopK` |
| **0.58%** | **712.0** | **14.83** | **`SDPA`** |
| 0.18% | 223.6 | — | the LM head |

Two things follow, and one of them settles an open question.

1. **Prefill is the expert matmuls and almost nothing else** — the
   `SparseMatmul` pair is **61.44%** of it. The collectives are 3.13%. Anything
   spent tuning prefill that is not the expert path is rounding error.
2. **The unadopted prefill SDPA program config is worth at most 0.58% here, and
   at S=128 it is a loss.** The measured in-model prefill SDPA is 14.83 us/layer;
   the standalone sweep prices the default at 23.92 us and the best chunking at
   25.72 us *at this length* (`probes/sdpa_prefill_confirm.json`). So the lever
   this stage built, measured and declined is now bounded from the model side as
   well as from the accuracy side. Its 6.3–6.8x lives at S ≥ 4096 and nothing in
   the current gate set goes there.

## Against the layer-stack lower bound

The goal asks for the bound recomputed from the *optimized* per-layer latency.
Doing that requires fixing the basis, because stage 05's bound used the wrong
one.

**Stage 05's bound was `48 × 0.4286 ms = 20.573 ms`**, and it concluded the model
was marginally *under* its own lower bound — which should have been the tell.
0.4286 ms (`../optimized_multichip_decoder/perf_decode.csv`, ctx128) is a **wall**
figure for a traced model containing one layer, so it carries one iteration's
worth of dispatch and host cost. Multiplying it by 48 charges 48 layers for an
overhead paid once. The device-kernel content of that same stage-04 layer is
**362.83 us** (`../optimized_multichip_decoder/window_decode.txt`).

Recomputed on the shipped profile:

| | ms/token | source |
|---|---|---|
| 48 × the stage-04 *isolated* layer's kernel time (362.83 us) | 17.416 | `../optimized_multichip_decoder/window_decode.txt` |
| **48 × the optimized *in-model* per-layer kernel time (384.791 us)** | **18.470** | `profile_summary_decode.json` `regions_us.layer_stack` |
| + terminal-pre (0.053) + terminal-post (0.367) | **18.889** | same file |
| **measured token-out decode** | **19.693** | `perf_full_model_p128_argmaxrows.json` |
| **gap: dispatch and op-to-op across 3512 ops** | **0.803 (4.08%)** | difference of the two |
| the superseded stage-05 bound, for comparison | 20.573 | 48 × 0.4286 |

The 18.889 row is the window's own total, not the sum of the three rounded
rows above it — adding those gives 18.890, and the 0.001 is this table's display
rounding and nothing else.

**The gap between token-out and (layer-stack bound + terminal work) is 4.08%.**
The goal flags anything above 10–15% as needing action; at 0.23 us per dispatch
over 3512 dispatches there is nothing to act on. The model is no longer "under
its bound" — that was an artifact of the wrong multiplier — and it is now 4.1%
above a bound computed on the right one.

The in-model layer is **+6.1%** on the isolated stage-04 layer (384.791 against
362.83), down from +9.4% before this stage; the difference is that
`SdpaDecode` no longer pays for decoding at a real position.

## Accuracy and readiness gates

All re-run on the shipped tree. Reference is `../../readiness_aime24_chat.refpt`
(AIME24, HF chat template, 158 prompt tokens, gen_len 100, top-100), generated by
stage 05 and unchanged. Bar: top-5 ≥ 0.98, top-100 = 1.00.

| gate | stage 05 | **stage 06** | artifact |
|---|---|---|---|
| `run_prefill_check` | 0.980 / 1.000 / 1.000 | **0.980 / 1.000 / 1.000** | `logs/run_prefill_check_argmaxrows.log` |
| `run_teacher_forcing` | 0.990 / 1.000 / 1.000 | **0.990 / 1.000 / 1.000** | `logs/run_teacher_forcing_argmaxrows.log` |
| `pytest tests/ -m "not models_performance_bare_metal" -q` | 145 passed | **146 passed**, 16 deselected | `logs/pytest_argmax_rows.log` |
| the same under `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1` | 145 passed, 0 asserts | **146 passed, zero tripped asserts** | `logs/watcher_argmaxrows.log.gz` |
| `check_degenerate_output --scope autoregressive` | clean | **"No degenerate output detected"** | `logs/check_degenerate_argmaxrows.log` |
| `check_degenerate_output --scope vllm` (the six-prompt suite) | clean, **inherited** | **"No degenerate output detected"**, **re-run** | `logs/check_degenerate_vllm_argmaxrows.log` |

**Not one accuracy figure moved.** Both bars are met with the maximum possible
margin — every one of the 100 reference positions is inside the model's top-5, on
both paths — and that is the same statement stage 05 made.

The watcher artifact is the **whole pytest run** under the watcher, not a final
dump: it carries its own `146 passed` tally and zero occurrences of "tripped an
assert" across 487 s.

### The free-running run

`run_autoregressive`, 128 greedy tokens from the same 59-token story prompt, HF
and TT side by side (`../../readiness_autoregressive/`,
`logs/run_autoregressive_argmaxrows.log`):

> **TT**: *" a peculiar shimmer in the air, like heat waves rising from summer
> stones. As she approached, the shimmer grew stronger, and suddenly, a portal
> opened before her eyes, revealing a world of impossible beauty. Elena stepped
> through the portal and found herself in a land where the sky was purple and the
> trees grew in spirals. …"*

Read, not just scored: fluent, on-prompt, keeps the character's name, holds tense
and viewpoint for the whole 128 tokens, no repetition and no collapse. The
degeneracy gate measures `num_tokens` 109, `adjacent_duplication` **0.0**,
`trigram_loop_fraction` **0.0275**, and reports **"No degenerate output
detected"**.

It diverges from HF at the **first generated token** — common prefix zero — and
**2 of 128** tokens match, at indices 1 and 94. HF's first token is `2494`, TT's
is `264`. That is the expected behaviour of free-running greedy decode over a 30B
MoE at `bfloat4_b` expert weights, for the reason stage 05 gives: the router
picks 8 of 128 experts on an argmax over logits that differ in the fifth decimal,
so one flipped selection sends the two continuations down different valid
branches. The agreement measurement is the teacher-forced one above, and it is
100/100 within top-5. Stage 05 measured 4 matching tokens; **this stage measures
2, and that is a real change** — the SDPA chunk order is not bit-identical (PCC
0.9994 in-model, `probes/sdpa_hf_pcc_at_depth.json`), so the free-running branch
point moved. The number that gates accuracy did not move.

### The six-prompt qualitative suite, re-run on the shipped sampler

Stage 05 ran this suite and stage 06 originally **inherited** its log, which was
wrong: the sampler's reduction changed twice this stage, so the inherited
evidence described a sampling path that no longer exists. It is re-run.
`probes/qualitative_probe.py` — the stage-05 script, copied into this directory
so it writes here and never into `../full_model/` — puts the repository's shared
suite `models/common/readiness_check/vllm_prompts.txt` (**six** prompts: a haiku,
an explanation, a story continuation, a factual list, a translation, and a Python
function) through the real 48-layer model on the real mesh, greedy **and** sampled
for each, twelve completions in all. Artifacts:
`logs/qualitative_check_argmaxrows.log` (the completions, to be read),
`probes/vllm_qualitative_outputs_argmaxrows.json` (the same as the `--scope vllm`
schema), and `logs/check_degenerate_vllm_argmaxrows.log` (the score).

`check_degenerate_output.py --scope vllm` reports **"No degenerate output
detected"** across all twelve. `adjacent_duplication` is **0.0** on **eight of
the twelve** and between 0.0161 and 0.0309 on the other four; the only high
`trigram_loop_fraction` values are 0.3 on the haiku and 0.5 on the translation,
both of which are
*complete, correct and six-to-ten words long* — a short answer has few trigrams,
which is a property of the metric and not of the model.

Read, not just scored. All twelve are coherent, on-task, in the right language,
and end where a complete answer ends or at the 128-token cap mid-structure. The
translation is exactly right, including the French spacing convention before the
question mark:

> `"Bonjour, comment allez-vous aujourd'hui ?"`

The haiku scans as a haiku:

> *Data streams flow— / Neural networks dream in patterns, / Wisdom emerges.*

Both code-shaped legs produce **valid Python** — correct base cases, correct
recurrence, a docstring with the right complexity claim — and are cut off by the
token cap partway through a second implementation, not by a collapse:

> ```python
> def fibonacci_recursive(n):
>     """
>     Calculate the nth Fibonacci number using recursion.
>     Time complexity: O(2^n) - Very slow for large n
>     """
>     if n <= 0:
>         return 0
>     elif n == 1:
>         return 1
>     else:
>         return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)
> ```

No repetition loop, no language drift, no degeneration into markup, and the
factual answers are right (ΔU = Q − W; entropy of an isolated system
non-decreasing; supervised = labelled, unsupervised = pattern-finding).

**One thing to state rather than bury: on four of the six prompts the sampled
completion is byte-identical to the greedy one**, so the sampled leg collapses
onto argmax there. That is not new and not this stage's doing — stage 05's
inherited log has the same collapse on **five** of six, and the cause is the
suite's sampling parameters (`top_k=20, top_p=0.9, temperature=0.7`) against a
model confident enough that the top-1 token survives them. The split path is
exercised end-to-end either way; what the suite does *not* give is six
independent samples of it.

Against stage 05's completions, four of the six greedy answers changed and two
are byte-identical. The changes are ordinary lexical branches, both readings
correct — "a teacher who **gives** you the right answers" became "a teacher who
**tells** you the right answers", "remains constant in ideal **reversible
processes**" became "remains constant in ideal **cases**" — which is the same
`bfloat4_b` expert-routing branch-point story as the free-running run above, at
sentence scale.

## The rejection ledger

Every candidate tried and rejected, with the number that rejected it. Rejections
with measurements are a deliverable here, not an appendix.

| candidate | measured | why not |
|---|---|---|
| **LM head DRAM-sharded program config** (`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, which `tt-perf-report` still recommends) | **236.428 us on device 3** at **66.4%** of DRAM bandwidth ⇒ a *perfect* rewrite saves **79.44 us**, **0.40%** of token-out | **Structural for the shard this model ships; a design choice for the alternative.** `N = 37984` over 8 DRAM banks is 4748 columns each, and 4748 is not a multiple of 32 — `tensor_layout.cpp:168` refuses. In tiles `37984/32 = 1187`, and **1187 is prime**, so no *uniform* width shard exists across 8 banks at any core count: that half is structural. The padded route is not — it is declined, for two reasons that are choices and are stated as such. See the ledger note below. |
| **Permuting experts across dies to un-skew the MoE reduce-scatter** | achievable saving **0.024–0.112 ms/iteration**, **0.12–0.57%** of token-out, and the top of that is a **floor** — measured, not derived: balanced expert-to-die assignments fitted on 128 decode tokens of one or two prompts and scored on a prompt the fit never saw, over every one of the nine such directions three prompts allow. The closest single figure to what would ship — fitted on two prompts, scored on an unseen third — is a mean of **0.085 ms/iteration** | **This row originally said "0 ms" and that was wrong.** Per-expert routing hotness is strong (the top 8 experts take 47.5–57.4% of a layer's selections against 6.2% under uniform routing), so the exchangeability argument that would give exactly 0 does not hold, and the hotness transfers between prompts well enough that pooling more of them makes the transfer *better*, not worse. The rejection does not rest on the gain being negligible; it rests on a fraction of a percent being a bad trade against a bit-identity obligation on the expert weights *and* `_expert_window_matrix`. Rewritten in full below the table. `probes/moe_skew_analysis_final.json`, `probes/moe_routing_across_tokens{,_prompt2,_prompt3}.json`, `probes/moe_routing_cross_prompt.json` |
| `keepdim=False` on the argmax (32 semaphore barriers → 1) | 309.3 vs 371.1 us on the op; **251.0 vs 250.8 us** on the full reduction once the rows are sliced | Real on its own — the 31 saved barriers are worth 62 us — but it buys **nothing** on top of the row slice, and costs a `[1,1,B] → [1,1,B,1]` reshape the rest of the reduction needs. Subsumed. Recorded because it *is* the right lever for a future `max_batch_size=32` deployment, where the row slice is a no-op. |
| `sub_core_grids` on the argmax, 8 / 16 / 32 cores | 2735.5 / 1429.5 / 792.2 us against 371.1 at the default 110 | Monotonically worse. `ttnn.argmax`'s multicore path is a scalar `bfloat16_greater` loop on a data-movement RISC, so the cost is scalar work and fewer cores is strictly more time. |
| `sub_core_grids` on the argmax, **64 cores** | **hung the device**, board reset required | Not a perf result — an upstream `uint32_t` underflow. See the bug list. |
| `ttnn.topk(k=1)` on the ROW_MAJOR tensor | does not build | `topk_device_operation.cpp:166` requires TILE layout and the multicore argmax requires ROW_MAJOR. Mutually exclusive at this shape. |
| `ttnn.topk(k=32)` on the TILE tensor | 6047.3 us | 16x the shipped argmax. |
| `argmax` on the TILE tensor (skip the untilize) | 23252 us | Single-core path. |
| TILE-layout slice to 1 row, then untilize | 71.5 us vs 75.1 for the full untilize | 3.6 us: a TILE slice on a non-tile-aligned height still touches all 1187 tiles. The untilize stays, and the slice happens in ROW_MAJOR after it at ~0.5 us. |
| a two-stage argmax via a tile-aligned reshape (the part-1 recommendation) | not attempted | `37984 = 2^5 × 1187` with 1187 prime, so the only tile-aligned reshape is `[1,1,37984,32]`, whose 32-wide reduction dimension gives the multicore factory 2 cores. And the premise was wrong: the op is scalar-compare-bound, not bandwidth-bound, so "get within 5x of bandwidth" is not a target that exists. |
| SDPA **`k_chunk_size=512`** (faster than 256 by 0.3–4% at `cur_pos` ≥ 4095) | **PCC −0.10 to +0.06** against HF in `test_multichip_decode_batch` | Memory-unsafe in-model: `k_chunk_size` must not exceed the cache's **per-user allocated depth**, and exceeding it does not raise — it reads past the buffer. It only fails when another test has run first in the same process, which is the tell. **No standalone probe reproduces it and two were written to try** (`probes/sdpa_shallow_cache_probe.py`, `probes/sdpa_kchunk_rule_probe.py`, both PCC 0.9997 at k512). `_sdpa_k_chunk` clamps to the cache depth; at the shipped context it never binds. |
| SDPA `q512/k512` on prefill | fails to build at **every** length including 128 (`program.cpp:1722`) | A resource limit, not an alignment rule. |
| **the prefill SDPA program config** (6.3–6.8x on the op at S ≥ 4096) | `run_teacher_forcing` top-1 **0.990 → 0.980**, bisected to the prefill leg specifically; TTFT 3448.79 ms baseline vs 3445.31 ms configured at the 158-token gate prompt — noise; and now, in-model, prefill SDPA is **0.58%** of a prefill | Left **wired and unadopted**. See Limitations. |
| `max_top_k` below 32 on the split path (carried from stage 05) | 6.268 ms at 16 vs 6.151 at 32; at 8 the composite gather **did not return** in 20 minutes and the mesh needed a reset | 32 is exactly one tile wide; below it `ttnn.all_gather` logs "Using slower composite all_gather". |
| `output_tile=Tile([1,32])` on the expert matmul (carried from stage 02) | 1.07x, still unreachable | `sparse_matmul` now **rejects the override outright** at the shipped 32×32 `in0` tile (`matmul_utilities.cpp:231`); giving `in0` a 1×32 tile makes the matmul run and then `reshape`/`sum`/`untilize` all raise. Blocks the 53.6 us/layer of expert-tail layout churn. |

### The MoE-skew rejection is withdrawn: the lever is not zero

The first draft of the row above retired a ~10.5 us/layer lever permanently on
three arguments. Two of them do not hold, the third was never stated, and the
measurement that decides it had not been taken. All four points are worth
recording because the *shape* of the error is the interesting part: a rejection
is a deliverable here, and a rejection has to be as sound as an adoption.

**1. The "already ahead of an arbitrarily chosen partition" argument was noise.**
It said the shipped contiguous windows come in 0.69 us/layer under the expected
maximum for an arbitrarily chosen partition, so a permutation would lose in
expectation. The
measured mean per-die maximum is **3.438** against **3.538** expected under
uniform routing — but the standard deviation of that maximum across the 48
layers is 0.85, so the standard error of the mean is 0.12 and the difference is
**z = −0.82, p = 0.41**. That is one sample not distinguishable from the
expectation, read as though it were a result. **Deleted**, from the README and
from `probes/moe_skew_analysis.py`.

**2. The chi-square dropped its most anti-uniform cell.** The test had
`if exp < 1.0: continue`, and for k=6 the expectation is 0.00385 × 192 = 0.738 —
so that bin was skipped. Its **observed count is 3**, and
`P(X ≥ 3 | λ = 0.738) = 0.039`. The single cell most in tension with uniformity
was the one being filtered out. Counts of k ≥ 5 are now **pooled** into one bin
rather than dropped, and the statistic is published with its degrees of freedom
and its p-value rather than as a bare number: **chi-square 4.06 on 6 bins,
df 5, p 0.54**.

**3. The sample is one token, and the counts are not independent.** The test
treated 192 die-layer counts as 192 independent `Binomial(8, 1/4)` draws. They
are not: within each layer the four counts sum to the router's top-8 by
construction, and all 48 layers come from a **single decode token**. The
effective sample is 48 layers of one token. Every p-value above is a *failure to
reject* uniformity and none of them is evidence for it.

**4. The per-die marginal was never computed, and it is the axis a permutation
targets.** Summed over all 48 layers the four dies fire **81 / 111 / 102 / 90**
times against 96 expected — die 1 fires 37% more often than die 0. On its own
that is chi-square **5.44**, df **3**, p **0.14**: again not significant at
n=1 token, and again not evidence of uniformity.

**What the sound argument actually is, and what it rests on.** It is not about
this sample's mean at all. If routing is **exchangeable over experts** — no
expert systematically hotter than another — then every relabelling of experts to
dies induces the *identical* distribution of per-die counts, so the expected
saving from a permutation is exactly **0 by construction**, whatever any single
profile happens to show. That is the argument, it was buried, and it rests
entirely on exchangeability. Persistent per-expert hotness is the one thing that
breaks it, and it is invisible in a one-token profile.

**So it was measured.** `probes/moe_routing_across_tokens_probe.py` runs the real
48-layer model on the real mesh for **128** free-running decode tokens, wrapping
`ttnn.topk` to record the router's top-8 at every layer of every token — 6144
samples per prompt instead of 48 — and it was run on **three** prompts (one
technical, one narrative, one expository), because routing is content-dependent.

* **Hotness persists across tokens, strongly.** Per layer, the eight
  most-selected experts take **47.5% / 57.4% / 53.1%** of all selections on the
  three prompts. Independent uniform routing would give 6.2%. The router is
  **not** exchangeable over experts, so the exchangeability argument above does
  not apply and "0 ms by construction" is not available.
* **Per-die selections** over 128 tokens × 48 layers are
  **12390 / 11968 / 11691 / 13103**, **11327 / 12412 / 12933 / 12480** and
  **12910 / 11532 / 11664 / 13046** against 12288 expected. Which die is busiest
  changes with the prompt.
* **Within a prompt, a permutation is worth about 1%.** Fitting a balanced
  expert-to-die assignment by swap-descent on the **first 64 tokens** and scoring
  it on the **last 64, which the search never saw**, gives 0.5247 / 0.5856 /
  0.5908 experts/layer — **0.173 / 0.193 / 0.194 ms/iteration**. In sample all
  three are ~2.43 against ~2.90 held out, and that gap is exactly the overfitting
  the split exists to expose.
* **Across prompts it survives, at about a fifth of that.** A shipped layout is
  fixed at weight-load time, so the gain that counts is one measured on routing
  the fit never sees. Over the six single-prompt → single-prompt directions
  three prompts allow, that is **0.024–0.111 ms/iteration, mean 0.057**
  (`probes/moe_routing_cross_prompt.json`).

  **This stage first published "0.024–0.028" and that was the bottom of the
  range being read as the range.** With two prompts only two of the six
  directions exist, and they turned out to be the two *smallest*. The other four
  are 0.033, 0.073, 0.075 and 0.111 — the largest is **4.6x** the figure that
  was published and the mean over six is **2.2x** it.
* **And fitting on one prompt is the worst case, not the representative one.**
  Pooling the fit over two prompts and scoring on the genuinely unseen third
  gives **0.058 / 0.085 / 0.112 ms/iteration, mean 0.085** — **0.43%** of
  token-out — better than the best single-prompt fit in **all three** cases. So
  the range above is a **floor**, and pooling more prompts would be expected to
  raise it.

  This is also why the sentence this stage used to justify the single-prompt
  design — *"the only gain that counts is one that survives being fitted on one
  prompt and scored on another"* — **was too strong, and it is withdrawn.**
  Fit-on-one is a *lower bound* on what a fixed layout can do, not the definition
  of it: a layout that would actually ship would be fitted on a corpus, and every
  measurement here says a corpus fit transfers better. Fit-on-one is still worth
  reporting; it is just the floor and not the number.
* **The shared structure is there independently of the search.** Prompt pairs
  agree on **2.33 of 8** of a layer's hottest experts on average against **0.50
  of 8** under independent routing, and their full per-expert selection counts
  have a Spearman rank correlation of **0.535** (for the original two prompts
  alone, 1.46/8 and 0.433 — the round-2 review measured 0.427 for the same pair
  with its own tie handling). That corroborates the fit from a direction a
  stochastic swap-descent cannot bias: shared hotness is real, which is why the
  lever is not zero, and it is partial, which is why the lever is small.

**Where that leaves the rejection.** The candidate is still declined, but on a
different and now-measured basis:

| | ms/iteration | % of token-out |
|---|---|---|
| what the first draft claimed was achievable | 0 | 0% |
| a permutation refitted per prompt, held out across tokens | 0.173–0.194 | 0.88–0.98% |
| a fixed permutation fitted on **one** prompt, scored on another (6 directions) | 0.024–0.111 | 0.12–0.56% |
| fitted on two prompts, scored on an unseen third (3 held-out prompts) | 0.058–0.112 | 0.29–0.57% |
| **everything measured at n=3, which is what is published** | **0.024–0.112** | **0.12–0.57%** |

**0.024–0.112 ms/iteration, 0.12–0.57% of token-out**, is the published figure,
and the high end of it is a floor rather than a bound: pooling improved transfer
in every case tested, and n=3 is a small n.

The top of that range — 0.112 ms/iteration — against a numerical-identity obligation (permute the expert
weights **and** `_expert_window_matrix` together, and prove the result is
bit-identical) is the trade, and it is declined. **The rejection no longer rests
on the gain being negligible** — it rests on a fraction of a percent not being
worth a bit-identity proof on the weight layout. What would reopen it properly is
more prompts (the pooled figure is still rising at n=3) and a routing sample from
the readiness prompts rather than from free-running generation. The idle the skew
actually costs is **0.473 ms/iteration** against a **0.506 ms/iteration** floor
under perfectly uniform routing; that inequality is still true and still
arithmetic, and what it does *not* mean — and was made to mean — is that nothing
is recoverable.

### Two notes on the ledger, where the first draft of it overstated its case

**The LM head is closed on two different grounds and only one of them is
structural.** The *unpadded uniform width shard* is genuinely impossible:
`37984/32 = 1187` tiles, 1187 is prime, so there is no equal split across 8 DRAM
banks at any core count and `tensor_layout.cpp:168` refuses the unequal one. That
is arithmetic. The *padded* route — pad the vocabulary to `N = 38912 = 1216 × 32`,
which does shard 8 ways — is a different matter, and the honest record is that
`probes/lm_head_dram_sharded_probe.py`'s padded arm **did not fail on tile
alignment**. It raised an undiagnosed `TT_THROW @ program.cpp:1722` and the
archived log truncates there, so what that arm establishes is "did not build,
reason unknown", not "cannot exist". The padded route is **declined by design**,
for two reasons, both choices:

* this model's LM head is specified on `vocab_padding: 0` (`151936 = 4 × 37984`
  exactly), and the audit asserts it;
* padding the vocabulary would break the **distributed argmax**, which is this
  stage's largest sampler win. The reduction is exact only because each die's
  local argmax ranges over real vocabulary — padded columns would have to be
  masked to `-inf` on every die on every token, or they would win ties and
  return token ids that do not exist. Adding that mask to save at most 79.44 us
  of a 19.693 ms token-out is a bad trade, and declining it is the point.

So: structural for the shard that ships, a design decision for the alternative,
and either way **under 0.4% of token-out**.

**The 79.44 us headroom is device 3's, and that is stated because it has to be.**
`tt-perf-report` merges the four devices and prints the slowest one's row; the
LM-head row it prints is **device 3, 236.428 us, 66.4%**. The terminal-post table
above is device 0, whose same matmul is **226.130 us** (per-device: 226.130 /
225.932 / 225.253 / 236.428, `profile_summary_decode.json`
`lm_head_us_all_devices`). The first draft of this ledger row multiplied device
0's kernel time by device 3's utilisation, which is not a quantity. Both
self-consistent readings are small — 79.44 us on device 3, 69.14 us on device 0
at its own 69.42% (utilisation × duration is the same byte count on every die) —
and the ledger quotes device 3's, the larger of the two,
because the closure argument should be made against the most favourable number
the lever could have.

## Runtime fallback audit

`Qwen3CoderModel.runtime_fallback_audit()` on the real 48-layer model, plus the
two properties this stage changed that the audit dict does not yet name.
Captured verbatim at `probes/runtime_fallback_audit.json` /
`logs/runtime_fallback_audit.log`; every field of the audit dict itself is
asserted by `test_runtime_fallback_audit_is_clean` on every run.

* **layer, unchanged from stage 04**: `dram_sharded_taken` True, per-die qkv
  `(2048, 1280)`, wo `(1024, 2048)`, `gate_up_in0_block_w` 16,
  `down_in0_block_w` 12, expert intermediates **L1**, local heads `(8, 1)`,
  local experts 32, `norm_shard_feeds_qkv_directly` True,
  `decode_ccl_buffers_persistent` True;
* **wrapper, unchanged from stage 05**: embedding
  `replicated_bf16_no_collective`; residual contract `replicated [1,1,B,2048]
  bf16 TILE DRAM, no inter-layer collective`; LM head column-parallel, local
  vocab 37984, `bfloat8_b` (`DataType.BFLOAT8_B`), **`vocab_padding` 0**; decode rope
  `rotary_embedding_hf(is_decode_mode=True)` with the position advanced by
  `ttnn.plus_one` **inside the trace**; KV cache `bfloat16`, paged, block 32;
  `Topology.Ring`; 2 links prefill, 1 decode;
* **boundaries**: `host_logit_readback_on_token_out_path` **False**,
  `host_argmax_on_token_out_path` **False**. Logits never reach the host on the
  measured path, and neither sampling strategy all-gathers them any more — greedy
  gathers four candidates, split gathers 32;
* **sampling**, this stage's wording:
  `Sampling1D force-argmax, distributed: per-die untilize/argmax/gather →
  all-gather 4 candidates → masked-min, traced, writes tt_out_tok`. The class is
  `_WatcherCleanSampling1D`, a subclass of the shared `Sampling1D` with two
  overridden methods and no edit to `models/common/`;
* **new in stage 06, and read off the modules rather than the audit dict**:
  paged SDPA-decode at `q_chunk_size` 32, `k_chunk_size` **256** (unclamped at
  the shipped cache depth), `max_cores_per_head_batch` 16, program configs
  memoised on the compute grid; prefill SDPA at the op default
  (`sdpa_program_config=None`, verified by reading the source of
  `decoder_layer_prefill_multichip`); sampler `_dist_active_rows` = 1 =
  `max_batch_size`, `_dist_local_vocab` 37984, distributed path taken.

**Cache ownership** is explicit both ways and tested:
`prefill_forward`/`decode_forward` use the caller's `kv_cache` and `page_table`
verbatim (`test_caller_owned_cache_is_used_verbatim`) and `generate` allocates
and owns its own. `reset()` zeroes the cache in place with
`ttnn.fill(..., output_tensor=)` so tensor identities and DRAM addresses survive
for trace replay (`test_reset_zeroes_the_cache`,
`test_reset_makes_generation_reproducible`).

**Host sampling** (`sampling_mode="host"`) remains an explicit compatibility mode
and is never on the measured path.

**Steady state does no host work**, measured rather than asserted: across two
consecutive steady-state traced tokens, of the thirteen counters the generator
keeps, **only `replays` moves** (`runtime_fallback_audit.json`
`steady_state_two_tokens.only_replays_moved`). No token, position, rotary,
page-table or sampling-parameter host copy, no synchronisation, no cache reset —
the same property `test_steady_state_decode_does_no_host_work` asserts, recorded
here on the shipped tree with the SDPA and sampler changes in place.

## Three upstream bugs

All three are real defects in shared code, all three have a reproducer in this
tree, and none of them gates this stage.

**1. `all_gather_async` trips a BRISC `ASSERT` with `Topology::Linear` +
`num_workers_per_link=1`.** Found at stage 05, still open.
`ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp:119`,
on a 4-device Blackhole 1D-ring fabric. Either parameter alone is clean; the pair
trips on the first call, at the sampler's 37984-wide shape *and* at the layer's
512-wide one, so it is a property of the op's parameters and not of this model's
tensors. Reproducer: `../full_model/probes/ccl_watcher_ab.py --leg
linear_workers1`, ~20 lines, no model and no `Sampling1D`; full A/B matrix at
`../full_model/watcher_ab.log`.

**2. `models/common/modules/sampling/sampling_1d.py:294-346` steers every
sub-T3K mesh into exactly that combination.** `_get_argmax_all_gather_config`
forces `Linear` for any mesh under 8 devices while the fallback call hardcodes
`num_workers_per_link=1`; the Ring branch that was written to dodge a different
trace-capture problem is unreachable on the meshes it was written for
(`default_topology(mesh)` returns `Topology.Linear` for a 1x4 Blackhole ring —
the probe prints it). Fixing this upstream fixes every sub-T3K caller. This model
works around it locally in `_WatcherCleanSampling1D._argmax_all_gather`, which
passes the same op the same `dim` and semaphores with **no tuning knobs pinned at
all** — and that is both the watcher fix and a 1.65x speed-up of the greedy
sampler, because pinning one worker per link is a throughput cap.

**3. `ttnn.argmax` with `sub_core_grids` hangs the device on a `uint32_t`
underflow.** Found this stage. `argmax_multi_core_program_factory.cpp` computes

```
red_dim_units_last1 = red_dim_units1 - (ideal_red_dim_units - red_dim_units)
```

in `uint32_t`. With two core ranges whose total padded work exceeds the tensor by
more than one core's share, the subtraction goes negative: at 64 cores on this
13-wide grid it is `608 - 928`, which wraps to **4294966976**, and the reader
kernel then issues an unbounded NOC read and never returns. 8 / 16 / 32 cores are
safe only because their remainder happens to be smaller than one core's share.
Reproduced once — the board needed a `tt-smi -r` — and then removed from the
sweep rather than re-run; the guard and the arithmetic are documented in
`probes/argmax_outer_dim_probe.py`. **Not on the model's path**: the model does
not pass `sub_core_grids`.

## Named limitations

1. **The prefill SDPA program config is wired and not adopted.**
   `attention_prefill` gained an `sdpa_program_config=` seam defaulting to
   `None`, and `decoder_layer_prefill_multichip` passes `None` explicitly. The
   config is built and measured (`_sdpa_prefill_program_config`) and is worth
   6.3–6.8x on the op at S ≥ 4096, but it costs `run_teacher_forcing` top-1
   **0.990 → 0.980** and buys nothing at the 158-token prompt that gate uses —
   below the S ≈ 384 crossover. The prefill profile above bounds it further: SDPA
   is **0.58%** of a 128-token prefill. What it needs before adoption is a
   readiness reference with a multi-thousand-token prompt, so the regime where it
   pays is the regime the accuracy gate covers. Until then, **long prefills are
   slower than they need to be and this is the known reason.**
2. **The composite-path all-gather lever was not pursued.** The distributed
   argmax's two 4-wide gathers fall off `AllGatherAsync` onto
   `AllBroadcast` + `UntilizeWithUnpadding`×8 + `Permute`×2 + `Concat`×2 +
   `TilizeWithValPadding`×2, **41.06 us**, 0.22% of the iteration and 33% of the
   sampler — to move eight numbers. Packing the value and the index into one
   8-wide gather, or padding to a width the async path accepts, is the obvious
   shape. It was left alone because it is the exact path that hung the mesh for
   twenty minutes at stage 05 (`max_top_k=8`), so it needs a watcher A/B first,
   and 0.22% did not justify the risk inside this stage.
3. **The `Tile([1,32])` expert-tail blocker is unchanged and has hardened.** The
   three `ReshapeView`s plus the tilize/untilize around the router are **53.6
   us/layer, 2.57 ms, 13.6% of the iteration** — larger than either
   `SparseMatmul` alone — and the 1.07x that would remove them is still
   unreachable upstream. This is the largest identified-and-blocked item in the
   model.
4. **The profile is a single decode position.** Every op-level microsecond above
   is `cur_pos ≈ 131`. The published token-out figures span positions 128–256 and
   the context sweep goes to 4096, but there is **no op-level profile at 4096 or
   beyond**, so the claim "decode is flat in context" rests on the end-to-end
   sweep (which measures it directly) and not on a profile at depth.
5. **Nothing was profiled at batch > 1.** All figures are batch 1. The live-row
   slice is a no-op at batch 32 by construction, and `keepdim=False` — measured
   and rejected here — is the lever that would apply there instead. Batch 32 was
   not run. The slice is however now *tested* above batch 1:
   `test_distributed_argmax_is_exact_at_batch_above_one` drives the sampler at
   `max_batch_size=4` with random and with all-negative logits and asserts every
   live row against the host argmax of the same bf16 values, every padding row
   at exactly 0, and the caller's `tt_out_tok` object preserved. Stage 06 made
   that reduction batch-dependent and every other device-sampling test in the
   suite uses the batch-1 fixture, so the branch was uncovered where it was
   newest.
6. **The qualitative suite's sampled leg mostly does not sample.** On four of
   the six prompts the sampled completion is byte-identical to the greedy one at
   the suite's `top_k=20, top_p=0.9, temperature=0.7`. The split path is
   exercised end to end regardless, and stage 05's run collapsed on five of six,
   so this is a property of the shared suite rather than of this stage — but the
   suite gives fewer independent samples of the top-k/top-p path than its shape
   suggests, and a hotter leg would be better evidence.
7. **Prefill still does not chunk**, batch is still capped at **32** by
   `nlp_create_qkv_heads_decode_device_operation.cpp:51`, `max_top_k` is still
   pinned at 32, and the RoPE tables are still sized lazily — all unchanged from
   stage 05 and documented there.
8. **The expert-permutation lever is reopened at 0.12–0.57% and not taken.** The
   ledger's "0 ms" is withdrawn; what a fixed layout could actually buy is
   **0.024–0.112 ms/iteration** across every fit-and-score direction measured at
   n=3, on routing the fit never saw. It is small, but it is not zero and it is
   no longer closed by argument. The top of the range is a **floor**, not a
   bound: pooling the fit over more prompts transferred better in every case
   tried, so the number is still rising at n=3. The stage first published
   0.024–0.028 from an n=2 sample, which turned out to be the two smallest of the
   six directions n=3 makes measurable — the mean over six is 2.2x that figure
   and the largest is 4.6x it. **n=3 is still a small n and the same failure mode
   applies to it**; the range should be read as what three prompts could
   establish, not as a bound.
9. **The routing sample is from free-running greedy generation, not from the
   readiness prompts.** `moe_routing_across_tokens_probe.py` decodes eagerly from
   its own prompt; the distribution a served workload visits may differ.
10. **"Shotgun" is a declared property, not a measured breadth, and by measured
    breadth the mutation coverage is thinner than the green run implies.** The
    tester's gate is that no assertion is covered *only* by one of the four
    mutations that corrupt a whole document. That is a real gate and it passes,
    but it is not the same claim as "every assertion has a narrow mutation
    behind it". Measured, the four declared shotguns trip **61, 66, 242 and
    265** assertions, while `profile_decode` (82), `profile_decode_small` (69)
    and `perf_shipped` (50) are all *wider* than two of them and all count as
    targeted. **92 of 557** assertions have no mutation narrower than 21
    assertions covering them. Classifying by measured breadth instead — any
    mutation above some width counts as a shotgun — would close this properly,
    and it is not done here: at a threshold of 21 it would put those 92
    assertions in the shotgun-only bucket and the gate would fail, and the fix
    is not a reclassification but ~92 new narrow mutations. The tester now
    **measures** breadth, prints the distribution of each assertion's narrowest
    coverage, and prints this number on every run, so the gap is visible rather
    than implied. Closing it is stage-07 work.

## Two prose corrections, and the naming that invited one of them

Both were found as discrepancies while writing this file and both are now fixed.
They are recorded rather than quietly dropped, because the *shape* of each is the
interesting part.

**1. Two `tt/model.py` docstrings carried superseded figures, and one of them
mixed two accountings.** `sample_greedy_argmax` quoted the part-1 sampler figures
("6.8x … 0.901 ms against 6.155 ms") and a token-out of 21.461 ms as though it
were the shipped one. It now quotes **0.928 ms against 6.155 (6.6x)** and
token-out **19.693 ms / 50.78 t/s/u**, and attributes each of the two sampler
levers to its *own* like-for-like delta rather than presenting either as the
whole.

The `_WatcherCleanSampling1D._sample_argmax` docstring was worse than stale. It
priced the baseline path at `AllGatherAsync 889 us + ArgMax 859 us` from
`../full_model/tt_perf_report_full_model_decode.txt` and called that sum
"essentially all of the 1.87 ms of non-layer work inside a 22.079 ms token-out
decode step". Two different accountings: those rows are per-op device-kernel time
summed over the op's own cores (2 and 110), taken from a **2-layer** window that
charges the terminal path against two layers instead of 48; the 1.87 ms is
wall-clock. The near-agreement was a coincidence between incommensurable
measurements. **The claim is withdrawn in the docstring** and the terminal block
is now priced from the 48-layer profile — 366.5 us of an 18889.5 us iteration,
1.94%, of which the sampler is 126.2 us — with the 2-layer figures kept, labelled
as shares of *that* window (27.5% and 26.5%), and not summed against anything.

Prose only; no behaviour changed.

**2. `probes/perf_full_model.{csv,json}` used to be the part-1 measurement.** The
unsuffixed name is what `probes/perf_full_model.py` writes with no `--tag`, so it
is the file a reader or a re-run reaches by default — and it held token-out
21.4609 ms, a superseded figure, purely because a docstring cited that path. This
project has already been bitten once this stage by a superseded artifact under a
canonical name (the pre-adoption 48-layer profile). So the canonical name now
holds the **shipped** measurement, byte-identical to
`perf_full_model_p128_argmaxrows.{csv,json}`, and the part-1 file is
`perf_full_model_part1_preadoption.{csv,json}` — the same suffix the superseded
profile artifacts already use. `check_published_figures.py` asserts both halves:
that the unsuffixed file matches the shipped run on every published row, and that
the `_part1_preadoption` file is still distinguishable from it.

## Discrepancies found and not fixed

* **The published profile windows' *cut point* is not independently
  re-derivable.** Each window is boundary-checked from the inside — ten exact op
  tallies per device on all four — but the raw ~139 MB profiler capture it was
  cut out of is not archived, so nothing in this tree says how many rows were
  discarded or what they were. `probes/window_full_model_48.py` now takes
  `--manifest` and writes exactly that (the raw file's size and SHA-256, the
  per-device row counts before and after, the cut index per device, and a digest
  of the discarded rows), and the verification recipe passes it. The **shipped**
  windows predate it and cannot be given one retroactively: the surviving
  `/tmp` capture from this stage is a *different* run (3519 ops per device
  against the published window's 3512, and 10 `Permute` rows against 2), so
  generating a manifest from it would describe a capture that is not the
  published one. A manifest ships from the next capture forward.
* **`runtime_fallback_audit()` does not name the two properties stage 06
  changed** — the paged SDPA program config and the sampler's live-row count. Both
  are measured-path properties and both belong in an audit whose purpose is to
  make the measured path inspectable. Adding fields would change a dict
  `test_runtime_fallback_audit_is_clean` pins field by field, so it was recorded
  here and in `probes/runtime_fallback_audit.json` instead.

## Capacity

`probes/footprint_probe.py --context 262144` builds the **real** shipped model —
real weights, real embedding, real LM head, the real paged KV cache at the full
advertised context, the real RoPE tables — captures both decode traces and runs a
token through it. `probes/footprint_262144.json`:

| | GB/die |
|---|---|
| weights + `embed_tokens` + `lm_head` + RoPE tables | 5.311 |
| paged KV cache, 262144 tokens, batch 1 | 6.443 |
| captured traces + persistent collective buffers | 0.006 |
| **total** | **11.760** |
| free | **22.119** |
| DRAM per die reported by the allocator | 33.879 |

**No capability reduction.** The advertised context is 262144, the model holds
and runs it, and `../context_contract.json` is updated with these numbers and
with what stage 06 changes about them — see that file's `stage06_note`.

## Verification

```bash
source python_env/bin/activate
D=models/autoports/qwen_qwen3_coder_30b_a3b_instruct

# accuracy and readiness (each opens its own mesh; one device job at a time)
for RUNNER in run_prefill_check run_teacher_forcing; do
  python -m models.common.readiness_check.$RUNNER \
    --model-dir $D --reference $D/readiness_aime24_chat.refpt \
    --mesh-device P300X2 --fabric-config FABRIC_1D_RING --trace-region-size 300000000
done
python -m models.common.readiness_check.run_autoregressive \
  --model-dir $D --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --mesh-device P300X2 --fabric-config FABRIC_1D_RING \
  --trace-region-size 300000000 --max-new-tokens 128
python models/common/readiness_check/check_degenerate_output.py \
  --model-dir $D --missing-artifacts critical --scope autoregressive

# the six-prompt qualitative suite, on the shipped sampler (run the stage-06 copy)
python $D/doc/optimized_full_model/probes/qualitative_probe.py --layers 48 --gen-len 128 \
  > $D/doc/optimized_full_model/logs/qualitative_check_argmaxrows.log 2>&1
python models/common/readiness_check/check_degenerate_output.py \
  --model-dir $D --missing-artifacts critical --scope vllm

# the whole suite, then the same under the watcher (never with the profiler)
pytest $D/tests/ -m "not models_performance_bare_metal" -q
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
  pytest $D/tests/ -m "not models_performance_bare_metal" -q

# performance: the three prompt lengths at the top of this file
for P in 128 1024 4096; do
  python $D/doc/optimized_full_model/probes/perf_full_model.py \
    --layers 48 --prompt-len $P --gen-len 128 --context 8192 --tag _p${P}_argmaxrows
done

# the 48-layer decode profile, its verified window, and its report
python -m tracy -v -r -p --op-support-count 32000 -o /tmp/prof_fm48_dec \
  $D/doc/optimized_full_model/probes/profile_full_model_48.py
python $D/doc/optimized_full_model/probes/window_full_model_48.py \
  /tmp/prof_fm48_dec/reports/*/ops_perf_results_*.csv \
  --out /tmp/fm48_decode_window.csv --layers 48 \
  --manifest $D/doc/optimized_full_model/probes/window_manifest_decode.json
tt-perf-report /tmp/fm48_decode_window.csv --no-color
python $D/doc/optimized_full_model/probes/rank_full_model_48.py /tmp/fm48_decode_window.csv
python $D/doc/optimized_full_model/probes/profile_summary.py /tmp/fm48_decode_window.csv \
  --out $D/doc/optimized_full_model/probes/profile_summary_decode.json

# the 48-layer prefill profile, whose window is checked row for row
python -m tracy -v -r -p --op-support-count 32000 -o /tmp/prof_fm48_pf \
  $D/doc/optimized_full_model/probes/profile_full_model_48_prefill.py
python $D/doc/optimized_full_model/probes/window_full_model_48_prefill.py \
  /tmp/prof_fm48_pf/reports/*/ops_perf_results*.csv \
  --out /tmp/fm48_prefill_window.csv --layers 48 --seq-len 128
tt-perf-report /tmp/fm48_prefill_window.csv --no-color

# capacity and the runtime audit
python $D/doc/optimized_full_model/probes/footprint_probe.py --context 262144
python $D/doc/optimized_full_model/probes/runtime_fallback_audit_probe.py --layers 48

# the MoE routing sample: 128 decode tokens on each of three prompts, then the
# cross-prompt fit that decides the expert-permutation lever (device, serialised)
python $D/doc/optimized_full_model/probes/moe_routing_across_tokens_probe.py --tokens 128 --layers 48
python $D/doc/optimized_full_model/probes/moe_routing_across_tokens_probe.py --tokens 128 --layers 48 \
  --tag _prompt2 --prompt "Once upon a time in a small village at the edge of a great forest, there lived a clockmaker who had never once been late. Continue the story for several paragraphs, and keep the tone gentle."
python $D/doc/optimized_full_model/probes/moe_routing_across_tokens_probe.py --tokens 128 --layers 48 \
  --tag _prompt3 --prompt "Summarise the causes of the decline of the Western Roman Empire, listing the main economic, military and political factors, and briefly assess which historians consider most important."  # this one was run by the round-2 reviewer, not here; its capture is archived unmodified
python $D/doc/optimized_full_model/probes/moe_routing_across_tokens_probe.py --cross \
  $D/doc/optimized_full_model/probes/moe_routing_across_tokens_raw.json.gz \
  $D/doc/optimized_full_model/probes/moe_routing_across_tokens_prompt2_raw.json.gz \
  $D/doc/optimized_full_model/probes/moe_routing_across_tokens_prompt3_raw.json.gz

# the two archiving steps: gzip the tt-perf-report transcripts, and cut the
# windowed CSVs down to the columns the analysis consumes. Both exist only
# because the repo's check-large-files hook rejects anything over 500 KB; the
# full-width CSV and the uncompressed transcript are what the commands above
# produce, and they are what to keep if you are not committing them.
# (gzip -c > x.gz && rm x, not gzip in place: the mutation tester's scratch
# trees hard-link these files and gzip refuses to rewrite a linked inode.)
for R in decode decode_part1_preadoption prefill_s128; do
  F=$D/doc/optimized_full_model/tt_perf_report_full_model_48layer_$R.txt
  gzip -c $F > $F.gz && rm $F
done
python $D/doc/optimized_full_model/probes/reduce_profile_csv.py --audit
python $D/doc/optimized_full_model/probes/reduce_profile_csv.py \
  $D/doc/optimized_full_model/ops_perf_full_model_48layer_decode.csv.gz \
  $D/doc/optimized_full_model/ops_perf_full_model_48layer_decode_part1_preadoption.csv.gz \
  $D/doc/optimized_full_model/ops_perf_full_model_48layer_prefill_s128.csv.gz

# the analyses that need no device
python $D/doc/optimized_full_model/probes/moe_skew_analysis.py \
  --csv $D/doc/optimized_full_model/ops_perf_full_model_48layer_decode.csv.gz \
  --out $D/doc/optimized_full_model/probes/moe_skew_analysis_final.json

# and the check that no figure in these documents has drifted from its artifact,
# then the check that every one of its assertions can actually fail
python $D/doc/optimized_full_model/probes/check_published_figures.py
python $D/doc/optimized_full_model/probes/mutation_test_checker.py
```

## The checker, and the proof that it can fail

`probes/check_published_figures.py` re-derives **557 figures** in this file and
in `work_log.md` from the artifact each one cites — a CSV row, a JSON field, a
probe log line — and compares the document's string against the *computed*
rounding. Ratios are recomputed from their two artifact operands and the quoted
string must be the rounding of the result. It runs on the host in under a second
and it currently reports `all 557 published figures match their artifacts`.

That is the stage-04 pattern under the rule the stage-05 review sharpened: **a
checker that reads the document it is checking checks nothing.** That review
found one assertion whose second clause was already proven by the assertion
above it — so it could not fail — and finding it took a human reading 678 lines.

This stage does it mechanically. `probes/mutation_test_checker.py` hard-links the
model directory into a scratch tree, applies **292 mutations** one at a time —
corrupt a document's digits, corrupt its words, scale an artifact's numbers,
rename an op label, flip a boolean in the contract, change the *measured* side of
a boundary tally, rename a file the documents promise exists, reverse a ranking,
put a `failed` line in a pytest log — and re-runs the checker against each.
`logs/mutation_test_checker.log` is that run: **557 assertions, 557 made to
fail by at least one mutation**, none left over.

### The stage-05 version of this credited coverage it had not earned

The stage-06 review found the tester's crediting broken, and the honest numbers
are worth putting side by side, because "473 assertions, 473 made to fail" was
not a true statement about the previous tree:

| | claimed, before | the first run with crediting fixed | now |
|---|---|---|---|
| assertions | 473 | 503 | **557** |
| made to fail by **some** mutation | 473 of 473 | 493 of 503 | **557 of 557** |
| made to fail by a **targeted** mutation | not measured | 457 of 503 | **557 of 557** |
| mutations that changed no assertion's outcome | not reported | 14 of 193 | **0 of 292** |

The middle column is what the fixes below were driven by; it is the first honest
measurement this tester ever produced, and it is 40 assertions short of the
coverage the previous tree claimed. The assertion count then rose from 503 to
557 as the gaps that measurement exposed were closed — most recently the
multi-token routing artifacts, which the documents quoted and nothing checked.

**It keyed mutation credit by the check's formatted name.** Many of those names
embed the artifact value under test — `README quotes the degeneracy metric 109`.
Mutating that artifact to 77 produced a `FAIL` line reading `README quotes the
degeneracy metric 77`, a name that appears nowhere in the clean run, and the
intersection `failed & set(clean)` dropped it on the floor. For every check of
that shape the tester proved only that the *document* side could fail, never the
*artifact* side, and still reported full coverage. Credit is now keyed by a
**stable id** — the source line of the `check()` call plus an ordinal for calls
inside a loop — which cannot vary with any artifact value. The name is still
printed; it is no longer the key.

**Two further gaps are now reported rather than passed over.**

* **Coverage that comes only from a shotgun.** Four of the mutations corrupt
  every digit or every word of a whole document. "This assertion failed under
  `readme_digits`" is nearly no evidence about that assertion — something else
  failed first for an unrelated reason. The tester now separates targeted
  coverage from shotgun coverage and **fails** if any assertion has only the
  latter. Fixing that meant giving the README's three self-accounting figures,
  the four audited audit fields and the nine artifact names the documents
  promise a mutation each.

  This section used to say those four "trip 200+ assertions at once"; review
  prompted measuring it, and the shipped run reports **61, 66, 242 and 265**
  (the measured set moves as assertions are added — it read 29/34/206/230 when
  a review first measured it, and 39/44/236/260 a pass later). It was true of
  two of the four and badly wrong about the other two, so the tester now
  **measures** breadth and prints it rather than asserting a constant, and this
  document quotes what it measured. See limitation 10 for what that measurement
  then exposed about the classification itself.
* **Mutations that break nothing.** A mutation with no effect on any assertion
  proves nothing and inflates the count this section quotes. The tester now
  **fails** on one, and the ones that had no effect were repaired or dropped.

Several assertions turned out to be unfailable once the crediting was honest,
and they are fixed rather than noted:

* `all(...)` over a one-element default list sliced `[1:]` — deleting the probe
  leg it checks caused **zero** failures;
* a check that the README lists the matching token indices, satisfied by any
  small integers occurring anywhere in a 55 KB document;
* a check on how many sampled legs collapsed onto their greedy leg, which passed
  whether the count was four or six;
* `README states the audited collective_topology`, which was an unanchored
  case-folded search for `"ring"` — 19 hits, including "docstring" and
  "gathering". The audit's values are now rendered into a required string
  **derived from the value** and looked for inside the README's own audit
  section;
* a regex parsed out of the README, discarded, and compared against a hardcoded
  literal `4`. Stage 05's agreement figure is now read from stage 05's own gate
  log.

### The vacuous-search class, and the lint that closes it

Three rounds of review each found the same defect and each fixed only the
instances it happened to notice: an assertion that a figure *appears somewhere*
in a 55 KB document, satisfied by text that has nothing to do with the figure.
`composite_gather_rows` = 16 was satisfied by the word `bfloat16`; the
chi-square's `df` = 5 by a markdown list marker; the SDPA `k_chunk_size` = 256
by `SHA-256`; the prefill `SparseMatmul` count of 384 by the unrelated
"S ≈ 384 crossover". Two were worse than coincidental: the decode tally of 40
was satisfied by *this section's own sentence* about being "40 assertions short
of the coverage the previous tree claimed", and the degeneracy metric 109 by the
README's prose quoting the check's own name. Both colliders move whenever an
assertion is added, so those checks rode on the same self-referential feedback
path that made the tester oscillate.

Finding them by reading is what had already failed three times, so the fourth
fix is mechanical. `appears()` now **lints its own needle** and the checker
exits non-zero on a violation, separately from any assertion failure:

* a numeric needle under four characters may not be searched over a whole
  document at all — no opt-out, because no promise about today's document keeps
  a short number from colliding tomorrow. It must be **anchored**: the phrase
  that is supposed to carry the figure is built from the parsed artifact value
  and matched against the whitespace-flattened document;
* a needle the document carries more than once must be anchored the same way or
  declared `restated=RESTATED`, which records that the recurrence was looked at
  and is the same figure quoted twice.

The lint flagged **88 whole-document searches at 70 call sites** on the run that
introduced it. Twenty were anchored — the seven proven vacuous above, four more
short needles the same rule catches, and the figures restated at several sites
where one site is authoritative (the pytest tally, the per-die vocabulary, the
uniform-routing share). The remaining 68 are distinctive decimals a table and a
paragraph both quote; each now carries `restated=RESTATED` at its call site,
which is an explicit, greppable decision rather than an omission. What it does
**not** claim is that those 68 are strong: corrupting one of their sites alone
still leaves the check green, and that residual weakness is named here rather
than left implicit.

### The archive's provenance

`--bootstrap` runs the mutation tester against a clean tree that is not green,
so a bootstrap log's measured breadths are inflated and the run says so in a
banner ending `THIS LOG IS NOT THE ARCHIVE`. Nothing read that banner: pasting
it, plus a clean-tree line reporting failures, onto
`logs/mutation_test_checker.log` passed every assertion the checker made. The
archive must now open with `clean tree: <this run's assertion count> checks, 0
failing` and carry no trace of the bootstrap path, and the laundering is a
mutation of its own so the assertion has to be able to fail.

### What this establishes, stated precisely

Not that the 557 assertions are the *right* ones, and — this is the part the
first draft of this section got wrong — not that "none of them is a tautology
over the checker's own literals" or that "none is a search a corrupted document
could still satisfy". Both of those claims were false when written, and the
tester as it then stood could not have detected either.

What the tester establishes is this: **for every one of the 557 assertions,
there exists at least one mutation of a document, an artifact, a log or a source
file in this tree that flips that assertion from PASS to FAIL, and for every one
of them at least one such mutation is targeted rather than a document-wide
corruption.** That is a statement about failability under a specific, listed set
of 292 perturbations. It is not a proof that no unfailable assertion remains —
an assertion could still be satisfiable by a corruption nobody wrote a mutation
for — and the honest reading of a green run is "this set of 292 corruptions is
detected", not "this document cannot lie".

## Artifacts

**Two of these are archived smaller than they were captured, and this section
says so rather than leaving a reader to assume otherwise.** The repo's
`check-large-files` pre-commit hook rejects any file over 500 KB, and six
artifacts here were over it.

* **The three `tt-perf-report` transcripts are archived gzipped**, as
  `tt_perf_report_full_model_48layer_*.txt.gz`. Nothing is lost — `zcat` gives
  back the byte-identical transcript, and `probes/check_published_figures.py`
  reads them through gzip. They were 804, 1332 and 800 KB and are 32, 56 and
  32 KB.
* **The three windowed ops CSVs are archived with the columns the analysis
  consumes, not all of Tracy's 128.** Every row of each verified window is
  present, so every boundary tally still holds; what is cut is the width, to 35
  columns, and the files go from 2728, 2732 and 2308 KB to 250, 251 and 364 KB.
  `probes/reduce_profile_csv.py` is the
  reduction, archived so it is reproducible; `python reduce_profile_csv.py
  --audit` re-derives the consumed column set by scanning every consumer rather
  than trusting a list, and its docstring says which columns are dropped and
  why. Each consumer was re-run against the reduced file and produced output
  **byte-identical** to the full-width run: `probes/profile_summary_{decode,
  prefill}.json`, `rank_full_model_48layer_decode{,_part1_preadoption}.txt`,
  `probes/moe_skew_analysis{,_final}.json` and the decode window's boundary
  tallies.
* **`ATTRIBUTES` is the one column a reader might miss.** It is 280 KB
  compressed on the decode window on its own — more than the whole budget —
  nothing under `probes/` reads it, and no figure in these documents is derived
  from it, so it is dropped. The consequence is that **`tt-perf-report` cannot
  be re-run from the archived CSVs**: it reads about forty columns including
  that one. That is why its output is archived whole rather than regenerated,
  and the full 128-column capture is regenerable from the profiling commands in
  the reproduction block above.

| file | what it is |
|---|---|
| `work_log.md` | what happened while doing this, including what broke and what was wrong |
| `profile_48layer_work_log.md` | the lever analysis in full — parts 1, 2 and 3, written as they happened. Its op-level figures are the **pre-adoption** ones and it says so |
| `ops_perf_full_model_48layer_decode.csv.gz` | **the shipped** verified one-iteration decode window, 14048 rows — every row, 35 of Tracy's 128 columns, see the note above |
| `probes/reduce_profile_csv.py` | the column reduction that keeps the three windows under the repo's 500 KB limit, and the `--audit` that re-derives the consumed column set from the consumers |
| `tt_perf_report_full_model_48layer_decode.txt.gz` | its `tt-perf-report` |
| `rank_full_model_48layer_decode.txt` | its region split and per-layer ranking |
| `probes/profile_summary_decode.json` | the same figures as JSON fields — what the checker reads |
| `ops_perf_full_model_48layer_decode_part1_preadoption.csv.gz`, `tt_perf_report_full_model_48layer_decode_part1_preadoption.txt.gz`, `rank_full_model_48layer_decode_part1_preadoption.txt` | the **superseded** pre-adoption profile, kept because `profile_48layer_work_log.md` quotes it |
| `ops_perf_full_model_48layer_prefill_s128.csv.gz`, `tt_perf_report_full_model_48layer_prefill_s128.txt.gz`, `probes/profile_summary_prefill.json` | the 48-layer prefill profile — new this stage |
| `logs/window_full_model_48_final.log`, `logs/window_full_model_48_prefill.log` | the boundary checks, 40 and 56 tallies, all four devices |
| `probes/perf_full_model_p{128,1024,4096}_{before,after,argmaxrows}.{csv,json}` | the three-lever, three-length performance sweep |
| `probes/perf_full_model.{csv,json}` | the canonical name, holding the **shipped** ctx128 run — the same bytes as `_p128_argmaxrows` |
| `probes/perf_full_model_part1_preadoption.{csv,json}` | the **superseded** part-1 run (token-out 21.4609 ms), kept because `profile_48layer_work_log.md` uses it as its reference point |
| `probes/qualitative_probe.py`, `logs/qualitative_check_argmaxrows.log`, `probes/vllm_qualitative_outputs_argmaxrows.json`, `logs/check_degenerate_vllm_argmaxrows.log` | the six-prompt qualitative suite, re-run on the shipped sampler: the script, the completions, the schema and the score |
| `logs/run_prefill_check_argmaxrows.log`, `logs/run_teacher_forcing_argmaxrows.log`, `logs/run_autoregressive_argmaxrows.log`, `logs/check_degenerate_argmaxrows.log` | the readiness gates on the shipped tree |
| `logs/pytest_argmax_rows.log`, `logs/watcher_argmaxrows.log.gz` | 146 passed, and 146 passed with zero tripped asserts under the watcher |
| `probes/runtime_fallback_audit.json`, `logs/runtime_fallback_audit.log` | the audit for the measured path |
| `probes/footprint_262144.json`, `.log` | the capacity table behind the contract |
| `probes/moe_skew_analysis_final.json` | the MoE skew analysis re-run on the shipped profile — one decode token |
| `probes/moe_routing_across_tokens_probe.py`, `probes/moe_routing_across_tokens{,_prompt2,_prompt3}.json`, `probes/moe_routing_across_tokens{,_prompt2,_prompt3}_raw.json.gz`, `probes/moe_routing_cross_prompt.json` | the router's top-8 at every layer of 128 decode tokens on each of three prompts, the raw routing behind it, and the cross-prompt and pooled permutation fits that reopen and then re-close the expert-permutation lever. **The third prompt's capture is the round-2 reviewer's, not this stage's** — the reviewer ran the probe unmodified on a prompt of its own choosing to test whether the two-prompt figure was representative, found it was not, and handed the capture over; it is archived byte-identical with a `provenance` key naming its origin and the sha256 of its raw routing. The reviewer also reproduced this stage's two published cross-prompt directions exactly (+0.0241, +0.0276) and independently replicated the within-prompt result (`mean_top8_share` 0.531 against 0.475 and 0.574; held-out-across-tokens +0.194 ms against 0.173 and 0.193) |
| `probes/sdpa_*.{py,json}` | the SDPA lever: depth, `cur_pos`, program-config sweeps at both dtypes, the in-model PCC at depth, the prefill sweep, and the two probes that *failed* to reproduce the k512 corruption |
| `probes/argmax_outer_dim_probe.{py,json}`, `probes/distributed_argmax_probe.{py,json}` | the sampler levers and their tie-breaking checks |
| `probes/lm_head_dram_sharded_probe.py`, `probes/tile_1x32_*.py` | the two structurally-blocked levers |
| `probes/check_published_figures.py` | re-derives 557 figures in these documents from the artifact each cites; host-only, under a second |
| `probes/mutation_test_checker.py`, `logs/mutation_test_checker.log` | the proof that all 557 of those assertions can actually fail — 292 mutations, every assertion broken by at least one targeted mutation |
| `probes/*.py` | every probe, runnable |
