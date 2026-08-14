# Optimized multichip decoder — Qwen3-Coder-30B-A3B-Instruct on 4 Blackhole dies

Stage 04. An in-place `$optimize` pass over the stage-03 multichip decoder
(`tt/multichip_decoder.py`, committed at `56366307f44`). The parallelisation is
unchanged — attention TP=4, experts EP=4, router and both RMSNorms and the
residual replicated, `FABRIC_1D_RING`, two RS+AG all-reduces per layer — and so
is the inter-layer contract. What changed is *where the activations live inside
the layer*, plus the decode collectives' link count.

`FABRIC_1D_RING` still has two links available and prefill still uses both;
decode now uses one, which is lever 4 below.

**Which files stage 04 edits.** `tt/multichip_decoder.py` is the target and
carries every shipped change. Three others are touched and none of them changes
a shipped number: `tt/optimized_decoder.py` gains one optional `rope=` parameter
that defaults to the op it already called — a seam, so that limitation 4's
rotary alternative could be measured without disturbing the 1x1 baseline this
document compares against; `tt/functional_decoder.py` and `tt/weight_mapping.py`
gain the helpers that alternative needs. All three additions are inert on the
shipped path, and `test_meta_rope_weights_match_hf` covers the last two.

The audit is `topology_audit.md`; what happened while doing it is `work_log.md`.

## What changed, in three lines

1. **Both residual RMSNorms are width-sharded across 8 cores** instead of
   running on one. 19.82 → 4.92 µs each standalone, and **more accurate** than
   the call they replace (max error against a torch fp64 reference 6.711e-02 →
   1.686e-02), because the shipped call passed no compute config and so
   accumulated the sum of squares in bf16.
2. **The router projection reads its activation from that L1 shard** instead of
   DRAM-interleaved. **24.62 → 5.85 µs** standalone, **output bit-identical**
   (`max|diff|` exactly 0.0), which is what lets the four dies keep agreeing on
   the top-8. 5.85 is the **8-core** leg of `probes/norm_router_probe.log`, and
   8 is what ships: the shard the norm produces is `_NORM_SHARD_CORES = 8` wide.
   The same sweep's 4-core leg reads 4.30 µs — **that leg is not the shipped
   configuration** and no figure here is priced against it. In the layer the
   projection is profile row 182, **6.241 µs**.
3. **The two collectives use caller-owned persistent buffers**, so nothing in
   the forward path allocates inside the trace. 0.2%.
4. **Decode collectives use one ethernet link, not two.** Stage 03 measured this
   at 0.6% and called it noise; against the stage-04 layer it is **1.22%**,
   output bit-identical. A decode collective moves 128 KB per die and is
   latency-bound, so the second link buys no bandwidth and costs the split and
   merge. Prefill keeps both links, where they are worth 1.84× at 2 MB.
   `probes/links_probe.py` now **alternates the leg order** across six passes,
   because review found the probe could no longer distinguish its own legs; see
   "What did not work" below for why that control matters here.

The first norm's output shard is deliberately the memory config
`attention_decode_optimized` already reshards its input into, so it crosses into
the qkv projection with no conversion at all.

## Results

Warmed prefill and warmed **traced** decode, one decoder layer, batch 1.

| | 1x1 | 1x4 stage 03 | 1x4 **stage 04** | vs 1x1 | vs stage 03 |
|---|---|---|---|---|---|
| prefill S=128 | 73.01 µs/tok | 24.57 | **25.13** | 2.91× | 0.98× |
| prefill S=512 | 69.05 | 19.51 | **19.43** | 3.55× | 1.00× |
| prefill S=1024 | 68.85 | 18.38 | **18.39** | 3.74× | 1.00× |
| prefill S=2048 | 69.28 | 18.26 | **18.02** | **3.85×** | 1.01× |
| traced decode ctx128 | 0.5638 ms | 0.4767 | **0.4286** | **1.315×** | **1.112×** |
| traced decode ctx1k | 0.6611 | 0.5722 | **0.5254** | 1.258× | 1.089× |
| traced decode ctx4k | 0.9912 | 0.9124 | **0.8667** | 1.144× | 1.053× |

Every cell is a cell of a CSV: the 1x1 column is `perf_baseline_1x1_*.csv` in
this directory (stage 04's own re-measurement of the untouched
`optimized_decoder.py`), the stage-04 column is `perf_*.csv` in this directory,
the stage-03 column is `../multichip_decoder/perf_*.csv` **frozen and never
regenerated**, and both ratio columns are computed from those cells by
`probes/summarize_perf.py` into `perf_summary.json`. No ratio here is typed.

**Decode is the result: 1.112× at ctx128 over stage 03, and 1.315× over one
die** where stage 03 reached 1.18×. **Prefill is unchanged**, which is what the
code says it should be — nothing stage 04 touched is on the prefill path
(`decode_residual_norm` asserts `S <= 32`, `_decode_ccl_buffers` returns `None`
above one tile, and `_links` gives prefill both links) — and the S=128 cell's
2.2% is inside its own measurement spread: stage 03's five iterations there span
3.119–3.637 ms and stage 04's span 3.176–3.606.

On device time, in the profile:

| | stage 03 | stage 04 | speedup |
|---|---|---|---|
| decode layer, device 0 | 414.661 µs | **362.828** | **1.143×** |

The device-time pair is two windows of two profiles in this directory, printed
by `probes/window.py`. `probes/layer_levers.py` additionally measures the
before/after pair **in one process on one mesh**: **0.4700 → 0.4282 ms**,
1.098×, the same answer the two CSVs give (0.4767 → 0.4286, 1.112×) from a
different session.

That leg is a copy of the committed stage-03 layer **body**, not of stage 03's
whole collective path: it calls `MC.all_reduce`, which is stage 04's — it
already carries the persistent buffers and the one-link decode. So the 1.098×
is the gain of the *norms and the router projection alone*, and it **understates
the pass**: it credits stage 03 with two of stage 04's four changes. The CSV
pair, which compares two whole committed trees, is the honest ratio and is the
larger of the two, which is the direction that bias predicts.

## The decode layer, term by term

Device 0 — **the slowest of the four dies**, published for that reason. Eleven
(stage 03) and twelve (stage 04) contiguous row ranges; **both columns sum
exactly to their totals**, so this is a decomposition and not a selection.

| block | rows (04) | stage 03 | stage 04 |
|---|---|---|---|
| `input_layernorm` | 154–155 | 20.081 | **6.663** |
| attention (projections + body) | 156–175 | 61.020 | 60.400 |
| all-reduce after `wo` | 176–178 | 36.319 | **33.063** |
| residual add | 179 | 1.878 | 1.969 |
| `post_attention_layernorm` | 180–181 | 20.127 | **6.663** |
| router block | 182–201, 203 | 90.243 | **71.412** |
| `normed` shard→interleaved for the experts | 202 | — | 0.876 |
| expert `sparse_matmul` pair | 205, 213 | 82.653 | 82.718 |
| expert reshape/eltwise tail | 204, 206–212, 214–217 | 70.021 | 69.573 |
| all-reduce after the experts | 218–220 | 30.446 | **27.581** |
| residual add | 221 | 1.873 | 1.910 |
| **total** | 154–221 | **414.661** | **362.828** |

The two all-reduce rows each include a `CloneOperation` (1.260 and 1.276 µs)
that copies the persistent all-gather buffer out, so the −6.12 µs across them is
net of that cost rather than before it.

Windows: stage 03 is `../multichip_decoder/ops_perf_multichip_decode.csv.gz`,
device 0, rows 134–197 (64 ops); stage 04 is
`ops_perf_optimized_multichip_decode.csv.gz` in this directory, device 0, rows
154–221 (68 ops). Both are the **second and last** decode iteration
`probes/profile_layer.py` runs. `probes/window.py` re-derives both windows from
the CSV — end at the first `BinaryNg` at or after the layer's second
`AllGatherAsync`, start at the first `InterleavedToSharded` after the *previous*
layer's end — and refuses to print unless the window contains exactly two
`ReduceScatterMinimalAsync` and two `AllGatherAsync` and begins and ends on
those ops. That invariant is what cost stage 03 a review when it was only
eyeballed.

The corroborating iteration is rows 82–153, and it is **72** ops rather than 68:
the four extra (rows 104–107, two `TilizeWithValPadding` and two `Typecast`,
47.456 µs) are the one-off upload of the persistent collective buffers for the
decode shape — the two `ttnn.from_torch(torch.zeros(...))` calls in
`_decode_ccl_buffers`, fp32 row-major in, bf16 `TILE` out, hence a tilize and a
typecast each. They land on the **first** decode call and on no later one.

The priming prefill has already allocated a set of its own, at rows 36–39, and
those are plain `Tilize` rather than `TilizeWithValPadding` because its 32
logical rows are already tile-aligned where decode's `batch` rows are not — the
op code is the evidence that the two sets are distinct. That the prefill gets a
set at all is not general: `_decode_ccl_buffers` returns `None` above one tile,
and this priming prefill is at `PROMPT = 32`, exactly the boundary the `S <= 32`
test admits. A priming prefill at S = 512 would allocate nothing and the four
ops would be the only set in the profile. **Four and not eight** because the
layer's two decode all-reduces have the same key and share one set (see
`work_log.md` §5); it is the prefill/decode pair that must not, and does not.

Without them iteration 1 reads **367.115
µs** against iteration 2's 362.828, 1.18% apart, which is the check that the
window boundaries are right. Those four ops are also why the layer must be run
once eagerly before `begin_trace_capture`; every harness in this stage does, and
the one probe that did not hung the mesh (`work_log.md` §6).

Per device, the same window: **362.828 (device 0)**, 349.795, 357.491, 338.760.

## The inter-layer residual layout contract

Written down here so full-model bringup preserves it rather than rediscovering
it, as the goal requires.

> **A decoder layer takes and returns a replicated `[1, 1, B, 2048]` bfloat16
> tensor, `TILE_LAYOUT`, `DRAM_MEMORY_CONFIG`, identical on all four dies.**
> There is **no** all-gather, reshard, layout conversion or all-reduce between
> layers: `out = layer_n(x)` is exactly the `x` of `layer_{n+1}`. All four
> collectives (2 RS + 2 AG, i.e. two all-reduces) are *inside* a layer, and so
> are all four of stage 04's new L1 shards — the two norm outputs and the two
> reshards that bracket them are created and consumed between the layer's own
> boundaries.

`topology_audit.md` derives why this is not merely inherited: both consumers of
the residual (`wqkv`, and the router + `gate_up` pair) are column-parallel over
the whole 2048-wide hidden, so a hidden-sharded residual needs RS-after-`wo`,
AG-before-the-norm, RS-after-`down`, AG-before-the-next-`wqkv` — **the same 2 RS
+ 2 AG**, since an all-reduce *is* an RS followed by an AG — and a distributed
RMSNorm on top of that adds two more collectives. Decode collectives are
latency-bound (this profile's `AllGatherAsync` costs 12.932 and 11.287 µs on a
32 KB payload), so what a decode layer pays for is the *count*, and the count is
identical. There is no lower-movement family here to adopt.

## Correctness

Every gate stage 03 shipped, at the same thresholds, on the stage-04 path:
**112 passed, 0 failed** (`pytest_full.log`, the whole 4-die suite with the perf
tests deselected). Stage 03's 111 plus one: `test_meta_rope_weights_match_hf`,
added with the rotary investigation in limitation 4 and host-only. Every figure below is a line of `pcc_log.txt`.

| | stage 03 | **stage 04** |
|---|---|---|
| prefill S = 32, 33, 100, 128, 257, 512 vs single-chip TTNN | 0.99962 – 0.99987 | **0.999620 – 0.999868** |
| prefill vs HF (S = 33, 128) | 0.99896, 0.99909 | **0.998960, 0.999088** |
| decode vs single-chip, contiguous / paged | 0.99997 | **0.999935 / 0.999941** |
| decode batch 8 contiguous, per user | 0.99996 – 0.99998 | **0.999893 – 0.999965** |
| decode, 4 consecutive steps vs HF | 0.99878 – 0.99931 | **0.998739 – 0.999393** |
| decode batch 1 / 2 / 8 / 32, per user vs HF | 0.99335 – 0.99959 | **0.993216 – 0.999566** |
| two stacked layers vs two stacked single-chip layers | 0.99945 | **0.999446** |
| layer under maximally unbalanced routing | 0.99994 | **0.999941** |
| router windows stitched vs global dense routing | **exactly 0.0** | **exactly 0.0** |
| per-die KV heads stitched vs single-chip cache | 1.0 | **1.0** |

**The one movement is `decode vs single-chip`, 0.99997 → 0.99994, and it is the
right sign.** The comparison is against `optimized_decoder.py`, whose
**numerics** stage 04 left unchanged — its only edit is the optional `rope=`
seam limitation 4 needed, which defaults to the shipped op, so the 1x1 column of
the results table is measured on an unchanged code path — and which still runs
the one-core interleaved RMSNorm; the
multichip norm is now *closer to fp64* than that reference is (§1 of
`work_log.md`), so the two paths differ slightly more than they used to while
the multichip one moved towards the true answer.

**Against HF, which is the arbiter, three of the four decode steps are very
slightly lower and one is higher** — the claim "nothing regressed" would be
false, so here are the four, from `pcc_log.txt` against
`../multichip_decoder/pcc_log.txt`:

| decode step vs HF | stage 03 | stage 04 | Δ |
|---|---|---|---|
| step 0 (pos 32) | 0.9992853 | 0.9992450 | **−4.0e-05** |
| step 1 (pos 33) | 0.9987817 | 0.9987388 | **−4.3e-05** |
| step 2 (pos 34) | 0.9993126 | 0.9993928 | +8.0e-05 |
| step 3 (pos 35) | 0.9989592 | 0.9989421 | **−1.7e-05** |

Every delta is at the fifth decimal, in both directions, against a gate at
0.99. That is not "nothing regressed"; it is **no movement outside the noise of
a bf16 path whose norm changed**, which is a weaker and true claim, and it is
the one this stage makes. The two prefill-vs-HF figures are unchanged to seven
digits (S=33, 0.9989603 both stages).

Also re-asserted, unchanged: `ttnn.topk` bit-identical across the four dies over
8 cases including an all-zero input; a die holding none of the global top-8
contributing exactly 0.0; trace replay bit-identical to eager; 20 consecutive
decode steps bit-identical; three prefill runs bit-identical on all four dies;
and a clean runtime fallback audit at batch 1 and 32.

Two assertions are **new in stage 04**, both guarding a way this pass could be
silently wrong:

* `test_decode_output_layout_matches_input` — the layer's output logical shape,
  dtype, layout and memory config must equal its input's. This is the
  inter-layer residual contract stated as a test, and it is what catches a
  persistent collective buffer imposing its own logical shape on the result;
  the defect it was written for is in `work_log.md` §5.
* `test_no_runtime_fallbacks` now also asserts
  `norm_shard_feeds_qkv_directly` — that the sharded norm's output shard is
  still bit-for-bit the one the DRAM-sharded qkv projection reads. If it stops
  being, TTNN inserts a reshard and the layer gets slower with no error.

## What did not work, with the number

Each measured on the whole warmed traced layer at ctx 128, median of 100, in two
interleaved passes so a leg's spread against itself (0.05%) is on the page next
to the differences between legs.

| lever | traced decode | verdict |
|---|---|---|
| stage-04 default | 0.4348 / 0.4346 ms | reference for the rows below |
| **threshold routing tail** — removes the `untilize → scatter → tilize` round trip stage 02 called unremovable, 17.007 µs of profile (rows 190–197) | 0.4382 / 0.4382, and 0.4355 against a 0.4282 default in the final run | **0.8–1.7% worse** in every run. Output bit-identical on all four dies |
| **first reduce-scatter fed from L1** — it costs 18.871 µs where the second, fed from L1, costs 15.018 for the same shape | 0.4403 / 0.4399 | **1.2% worse** |
| same, combined with persistent buffers | 0.4389 / 0.4397 | worse |
| **logits staged in L1 before `topk`** | 0.4380 / 0.4383 | **0.8% worse** |
| RS and AG staged in L1 | 0.4368 against 0.4378 early in the stage; 0.4377 against a 0.4282 default in the final run | **worse in both, and the second is confounded.** `layer_levers.py`'s leg read `ctx.num_links` (2) where the shipped `all_reduce` reads `_links` (1 at decode), so it paid for the second ethernet link as well as for L1 — the 2.2% is an upper bound on the L1 penalty, not a measurement of it. The early pair is link-matched and already says "worse", the direction is the same in both, and no run has ever had it ahead, so the rejection stands. The confound is real and the probe is fixed (`MC._links`), but this row is **the one verdict on the page resting on a leg with two variables** |

The one lever in this family that *did* pay is `num_links=1` for decode, and it
is in "what changed" above rather than here. **Review reopened it, and it
survived a control it had not previously faced.**

`_links` had stopped honouring an explicit `num_links=2`, so `links_probe.py`
was measuring one link against one link and the published 1.2% could not be
reproduced. Repairing `_links` and re-running returned the same gap — which is
*suspicious* rather than reassuring, since the broken probe had produced that
gap between two identical configurations. Either the lever is real, or the leg
that runs first in each pass is simply slower and the figure was always an
artifact. The probe now alternates the leg order across six passes to separate
the two:

| | position A | position B |
|---|---|---|
| `num_links=2` | 0.4342 / 0.4341 / 0.4340 | 0.4341 / 0.4337 / 0.4339 |
| `num_links=1` | 0.4290 / 0.4288 / 0.4286 | 0.4291 / 0.4283 / 0.4287 |

Each configuration reads the same at **both** positions, so the gap follows the
link count and not the running order: **0.42875 against 0.43400 ms, 1.22%**,
output bit-identical on all four dies in all twelve legs. The lever is real.
The old log simply predates `_links` — it was taken while `all_reduce` read
`ctx.num_links` directly, which did distinguish the legs — so the *figure* was
right all along and only its reproducibility had been lost by the adoption
itself.

It is now the default, which is why the `+ num_links=1` leg of the final
`probes/layer_levers.py` run reads 0.4286 against the default's 0.4282 — the
same configuration measured twice.

Standalone, at the shipped shapes:

| lever | measured | verdict |
|---|---|---|
| **`matmul_reduce_scatter_async` on the `wo` → RS edge** — stage 03's one named untried lever | fused 30.85–30.91 µs against unfused 18.82 + 10.73 = **29.55** | **rejected**, and the real cost is larger: the fused op takes a 2D matmul config, so `wo` would give up the DRAM-sharded config that runs it at 8.228 µs in the layer (profile row 174) and pay 18.82 instead |
| **DRAM-sharded router weight**, N padded 128→256, bf16 and bfloat8_b | 7.34–7.40 µs against the shipped 8-core L1 leg's **5.85** (plus a 0.45–0.51 µs sharded→interleaved of its own), and `max|diff|` 5–7e-02 against the reference logits | **rejected on both speed and exactness** — 26–33% slower than what ships, and it is the only router spelling swept that is not bit-identical |
| `topk` with `sorted=False` | 33.78 vs 33.81 µs | no |
| `topk` on a bf16 input | 31.81 vs 33.81 | 6%, and forbidden — routing must select in fp32 logit space |
| sharded norm at 16 cores | 4.14 vs 4.26 µs, with i2s+s2i 0.75/0.76 vs 0.51/0.53 | net worse. **Both figures are `probes/norm_router_probe.log`'s**: `norm_accuracy_probe.py` — the probe that adds the fp64 column — *crashed* at its 16-core leg with `Illegal kernel placement for writer_unary_sharded, Kernels cannot be placed on dispatch cores!`, because it lays its cores out as a rectangle that reaches a dispatch row at 16. So there is **no accuracy figure at 16 cores**; the rejection rests on speed alone, which is enough only because 16 cores is *slower* end to end. `work_log.md` §1 |
| sharded norm at 4 cores | 7.53 µs | worse than 8 |

`decode_levers.py` constructs its `MeshContext` positionally, so under the
repaired `_links` a decode collective reads `decode_num_links` and ignores that
argument — the probe would no longer reproduce these legs at 2 links if re-run,
which is a further reason they are carried rather than re-measured.

Carried from stage 03, unchanged and not re-measured: AG-of-partials (0.4801 vs
0.4760), `Topology.Linear` (0.4836 vs 0.4766), bfloat8_b collective payload
(0.4854), expert intermediates in DRAM (0.5128), capacity-padded `nnz`, dense
all-expert decode. `num_links` is the one stage-03 lever that was re-opened, and
it changed answer.

## Named limitations

1. **`TopK` is 26.356 µs on one core** — 7.3% of the stage-04 layer, and now the
   largest single op outside the expert matmuls — with a `FillPad` in front of
   it (4.190 µs) that is `ttnn.topk`'s own, because the decode logits are
   logically one row inside a 32-row tile. A 128-wide top-k over a single row
   has nothing to spread. Three spellings were swept and all are worse or
   forbidden.
2. **Dynamic `nnz` is 1.47× exact `nnz`**, ~26 µs of every decode layer, and
   there is no legal exact value under EP. Unchanged from stage 03.
3. **Collectives are 60.644 µs, 16.7% of the layer** (rows 176–178 and 218–220,
   including the two `CloneOperation`s, 1.260 and 1.276 µs, that copy the
   persistent all-gather buffer out). Down 6.12 µs from stage 03's 66.765.
   Every fusion, placement, dtype, link-count and buffer lever available has
   been measured; see the tables above.
4. **28.478 µs of the attention body still runs on single cores** (two per-head
   `LayerNorm`, two `RotaryEmbedding`, `NLPCreateQKVHeads`, two
   `PagedUpdateCache`, four reshards) — rows 158–167 and 169. These are
   single-die costs in `optimized_decoder.py`'s scope rather than multichip
   ones, and finding A's fix does not transfer: they are 128 wide, so a *width
   shard* has almost nothing to spread.

   **`rotary_embedding_llama` was named but not measured. It is measured now,
   it is 3.05× faster, and it is still rejected — for a reason that has nothing
   to do with its speed.** An earlier draft dismissed it by the width-shard
   argument above; review rejected that, correctly, because the named
   alternative is a different op rather than a width shard of the same one.

   `probes/rope_probe.py`, at the shipped per-die decode shape `[1, 1, 32, 128]`
   bf16, trace slope:

   | | measured | cores |
   |---|---|---|
   | `ttnn.experimental.rotary_embedding` (shipped, HF order, DRAM) | 3.84 µs | 1 |
   | `ttnn.experimental.rotary_embedding_llama` (`is_decode_mode=True`, Meta order, L1) | **1.26 µs** | 1 |
   | the `interleaved → height-sharded [32,128]` it needs in front | 0.20 µs | 1 |

   **3.05×, `max|diff|` exactly 0.000e+00, PCC 1.0000000** against the shipped
   op after permuting the result back to HF channel order. Rows 163 and 164 are
   4.699 + 4.659 = 9.358 µs, so the lever looked worth ≈4.9–6.3 µs, **1.4–1.7%
   of the layer** — more than the one-ethernet-link lever this stage adopted.

   Note what the 3.05× is *not*: the llama decode factory shards over **batch**,
   not heads, so at batch 1 it runs on one core exactly like the op it replaces.
   None of the gain is parallelism. It is the activation living in L1 and a
   kernel that multiplies by a resident 32×32 matrix instead of gathering a
   cos/sin row out of DRAM — finding B's lever, on a different op. The
   width-shard argument was true and simply never bore on the question.

   **So it was built and wired in.** The permutation it needs — Q and K row
   blocks of `wqkv`, *and* Qwen3's per-head `q_norm`/`k_norm`, which are applied
   between the head split and RoPE — is offline, weight-side and exact;
   `test_meta_rope_weights_match_hf` asserts the whole convention on the host,
   and `weight_mapping.permute_wqkv_to_meta` leaves V and `wo` untouched. The
   position gather is hoisted out of the forward path onto the first eager call,
   which is legitimate because `token_index` is a Python int for the shipped op
   too — neither spelling can advance the rotary position inside a replayed
   trace.

   **It then failed `test_multichip_decode_vs_single_chip` at PCC 0.876, and
   `probes/rope_layer_probe.py` says why:**

   | KV cache | PCC, Meta decode vs HF decode |
   |---|---|
   | fresh | **0.9999697** |
   | primed by a prefill | **0.1932974** |

   RoPE runs *before* K is written, so **the cache inherits the rotary's channel
   convention**. Prefill is untouched by this lever and writes HF-ordered keys; a
   Meta-ordered decode Q then scores against them and the dot products are
   meaningless. The op-level probe looked clean only because its cache was fresh
   and every other key was zero.

   **The lever is therefore not decode-local.** Adopting it means adopting the
   llama rotary in **prefill** as well, permuting prefill's interleaved `wqkv`
   copy, and changing the KV cache's channel convention — which
   `test_per_die_kv_heads_stitched` compares against a single-chip cache, and
   which the single-chip baseline in this document's results table does not
   share. That is a whole-layer change to `optimized_decoder.py`'s RoPE
   convention, not the in-place decode pass this stage is, and it would
   invalidate the prefill numbers this stage reports as unchanged.

   A second, independent cost, worth recording because it bounds the prize:
   `ATTENTION_WEIGHT_DTYPE` is `bfloat8_b`, whose 16-element blocks share an
   exponent, so the channel permutation **regroups the blocks and requantizes**.
   Even on a fresh cache the two paths are not bit-identical in the layer —
   attention out `max|diff|` 1.221e-04, K cache 3.125e-01 after permuting back.
   "Bit-identical" is a property of the op at fixed input, not of the layer at
   permuted weights.

   **Left in place, runnable, and off by default**: `functional_decoder.apply_rope_llama`,
   `multichip_decoder._meta_rope` and `upload_multichip_weights(meta_rope=True)`
   build the whole alternative, and the two probes re-measure both halves. The
   shipped upload pays no DRAM for it. This is the same treatment
   `router_forward_threshold` gets and for the same reason: the measurement is
   the useful part, and the next stage — which can change prefill — should
   inherit it rather than rediscover it.

5. Everything stage 03 named — SDPA-decode's `max_cores_per_head_batch=64`
   workaround at 1 KV head, the watcher not fitting on active ethernet cores,
   batch capped at 32, the expert M-padding compaction — carries over unchanged.

## Verification

```bash
source python_env/bin/activate
D=models/autoports/qwen_qwen3_coder_30b_a3b_instruct/doc/optimized_multichip_decoder

# correctness on the 4-die mesh
pytest models/autoports/qwen_qwen3_coder_30b_a3b_instruct/tests/test_multichip_decoder.py -q

# whole suite, no watcher, perf deselected
pytest models/autoports/qwen_qwen3_coder_30b_a3b_instruct/tests/ -q -m "not models_performance_bare_metal"

# watcher-clean. DISABLE_ETH is required: the watcher's active-eth program does
# not fit alongside the fabric router. Never combine with the perf tests.
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
  pytest models/autoports/qwen_qwen3_coder_30b_a3b_instruct/tests/ -q \
  -m "not models_performance_bare_metal"

# performance (rewrites the four CSVs in this directory)
pytest models/autoports/qwen_qwen3_coder_30b_a3b_instruct/tests/test_perf.py -q -k optimized_multichip
python $D/probes/summarize_perf.py

# op-level profile, and the window every op-level figure above comes from
python -m tracy -v -r -p --sync-host-device -o /tmp/prof_omc_dec $D/probes/profile_layer.py decode
python -m tracy -v -r -p --sync-host-device -o /tmp/prof_omc_pf  $D/probes/profile_layer.py prefill
tt-perf-report /tmp/prof_omc_dec/reports/*/ops_perf_results_*.csv
python $D/probes/window.py $D/ops_perf_optimized_multichip_decode.csv.gz decode

# the probes behind every number above
python $D/probes/norm_router_probe.py      # the norm and router sweeps
python $D/probes/norm_accuracy_probe.py    # core count and compute config, against fp64
python $D/probes/layer_levers.py           # stage 03 vs stage 04, one process
python $D/probes/layer_levers2.py          # threshold tail, persistent buffers, fused matmul-RS
python $D/probes/layer_levers3.py          # the stage-04 re-audit's levers
python $D/probes/mmrs_probe.py             # matmul_reduce_scatter_async, standalone
python $D/probes/links_probe.py            # one ethernet link vs two, for decode
python $D/probes/rope_probe.py             # rotary_embedding_llama vs the shipped rotary
python $D/probes/rope_layer_probe.py       # ...and why it cannot be adopted for decode alone

# and the check that no figure in these documents has drifted from its artifact
python $D/probes/check_published_figures.py

# context contract (unchanged by this stage)
python .agents/scripts/check_context_contract.py \
  --model-dir models/autoports/qwen_qwen3_coder_30b_a3b_instruct \
  --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct --require-contract
```

## Note on the raw op profiles

Stored **gzipped**, as stage 03's are, and for the same reason: a 4-die profile
carries four device columns and exceeds the repo's 500 KB per-file limit. Every
row citation refers to the uncompressed file. `probes/window.py` reads either.

## Artifacts in this directory

| file | what it is |
|---|---|
| `topology_audit.md` | the operation-topology audit, before and after, with every family compared |
| `work_log.md` | what happened while doing it, including what broke |
| `perf_baseline_1x1_prefill.csv`, `perf_baseline_1x1_decode.csv` | the single-chip baseline, re-measured by this stage |
| `perf_prefill.csv`, `perf_decode.csv` | the stage-04 multichip measurements |
| `perf_summary.json` | speedups and efficiencies computed from those four **and** from stage 03's frozen pair, by `probes/summarize_perf.py` |
| `pcc_log.txt` | every PCC quoted above, filtered out of `pytest_full.log` (the 112-test run). All 112 of its PCC-bearing lines were re-checked line by line against the regenerated `pytest_full.log` after stage 04's review edits and **every one reproduces to the last digit** (in fact all 139 lines of the file do) — which is the evidence that the `_links` refactor and the `rope=` seam changed no number |
| `pytest_full.log`, `pytest_perf.log`, `pytest_watcher.log` | the three test runs, 112 / 8 / 112 passed |
| `pytest_decode_retry.log` | the 38-test decode re-run that confirmed the persistent-buffer fix in isolation (`work_log.md` §5) |
| `ops_perf_optimized_multichip_decode.csv.gz`, `tt_perf_report_optimized_multichip_decode.txt` | the decode profile and its `tt-perf-report` |
| `ops_perf_optimized_multichip_prefill_s512.csv.gz`, `tt_perf_report_optimized_multichip_prefill_s512.txt` | the prefill profile |
| `window_decode.txt` | `probes/window.py`'s output: the published window, op by op, with its invariants checked |
| `watcher.log` | the watcher-clean run: 112 passed, and **zero** lines matching `error|assert|corrupt` in 152,839 bytes (149.2 KB) of waypoints. A previous revision called this "152 KB", which was the byte count read as kilobytes |
| `probes/rope_probe.log`, `probes/rope_layer_probe.log` | the rotary lever: 3.05× and bit-identical standalone, PCC 0.193 in the layer against a primed cache |
| `probes/*.log` | every sweep quoted above, as its script printed it |
| `probes/` | every script, runnable |
