# Operation-topology audit — stage 04, the multichip decode path

Required by `$optimize` before any knob tuning, and re-run after every change
that landed, because in this model every round has promoted a new top op that
the previous audit dismissed (`../optimized_decoder/topology_audit.md`, finding
G; `../multichip_decoder/work_log.md` §13).

The audited artifact is the **stage-03 multichip decode profile**,
`../multichip_decoder/ops_perf_multichip_decode.csv.gz`, device 0, rows 134–197
sorted by `HOST START TS` — 64 ops, 414.661 µs of `DEVICE KERNEL DURATION`. The
after-artifact is `ops_perf_optimized_multichip_decode.csv.gz` in this
directory, device 0, rows 154–221 — 68 ops, 362.828 µs. Both windows and both
op lists are printed by `probes/window.py`, which re-derives the boundaries
from the CSV and asserts the two structural invariants (exactly 2
`ReduceScatterMinimalAsync` + 2 `AllGatherAsync`; the window starts at the
first norm op and ends at the residual `BinaryNg`) before printing anything.

## Prefill, audited and deliberately untouched

`ops_perf_optimized_multichip_prefill_s512.csv.gz` against
`../multichip_decoder/ops_perf_multichip_prefill_s512.csv.gz`, device 0, the
second of the two passes `profile_layer.py prefill` runs: **rows 281–541, 261
ops, 9238.11 µs** against stage 03's rows 275–535, 261 ops, 9221.63 µs. Same op
count, same op sequence, +0.18%.

That is the intended result. Every stage-04 change is gated on `S <= 32`:
`decode_residual_norm` asserts it, `_decode_ccl_buffers` returns `None` above
it, and `_links` gives anything above it both ethernet links. The profile
confirms it rather than the code merely claiming it — the four `LayerNorm` ops
in the prefill window total **60.719 µs** against stage 03's **60.745**, i.e.
the prefill norms are the same one-core interleaved kernel they always were.

Prefill's collectives read 230.13 µs (2.49%) against 181.86 (1.97%), and the
whole difference is **one op**: the second reduce-scatter, 108.14 µs against
68.94, at an unchanged shape, dtype, memory config and link count. The other
three are within 4 µs of their stage-03 values. This is run-to-run variance on
a code path stage 04 does not touch, and the warmed medians in the CSVs are the
figure to trust: 18.02 µs/token at S=2048 against stage 03's 18.26.

Prefill is a deliberate non-target. It is already at 3.85× of one die and 96%
parallel efficiency, its collectives are 2.5% of it, and the two levers that
carried decode do not exist there — a 512-row norm is not one core's worth of
work, and a bandwidth-bound collective wants both links.

## The decomposition, before and after

Eleven (stage 03) and twelve (stage 04) contiguous row ranges. **Both columns
sum exactly to their window total**, so this is a decomposition and not a
selection — the check that surfaced two hand-typed figures in stage 03.

| block | rows (04) | stage 03 | stage 04 | Δ |
|---|---|---|---|---|
| `input_layernorm` | 154–155 | 20.081 | **6.663** | **−13.42** |
| attention (projections + body) | 156–175 | 61.020 | 60.400 | −0.62 |
| all-reduce after `wo` | 176–178 | 36.319 | **33.063** | **−3.26** |
| residual add | 179 | 1.878 | 1.969 | +0.09 |
| `post_attention_layernorm` | 180–181 | 20.127 | **6.663** | **−13.46** |
| router block | 182–201, 203 | 90.243 | **71.412** | **−18.83** |
| normed shard→interleaved for the experts | 202 | — | 0.876 | +0.88 |
| expert `sparse_matmul` pair | 205, 213 | 82.653 | 82.718 | +0.07 |
| expert reshape/eltwise tail | 204, 206–212, 214–217 | 70.021 | 69.573 | −0.45 |
| all-reduce after the experts | 218–220 | 30.446 | **27.581** | **−2.87** |
| residual add | 221 | 1.873 | 1.910 | +0.04 |
| **total** | 154–221 | **414.661** | **362.828** | **−51.83** |

The two all-reduce rows include the `CloneOperation` (1.260 and 1.276 µs) that
copies the persistent all-gather buffer out, so the −6.12 µs across them is net
of that cost, not before it.

Device 0 is published because it is the slowest of the four dies and a
synchronized mesh advances at its slowest: the same stage-04 window reads
349.795 on device 1, 357.491 on device 2 and 338.760 on device 3.

## Findings and what was done

| # | finding | evidence | action | result |
|---|---|---|---|---|
| **A** | **Both residual RMSNorms run on one core.** `CORE COUNT` is 1 on rows 134 and 159 of the stage-03 profile, 20.081 and 20.127 µs for a 2048-wide bf16 row over one 32-row tile — 128 KB in 20 µs, i.e. 6.5 GB/s, one core's share of L1 bandwidth. Stage 02 never gave `ttnn.rms_norm` a program config, so both stages inherited the interleaved single-core factory. | rows 134, 159; `probes/norm_router_probe.py`, `probes/norm_accuracy_probe.py` | Width-shard the activation over 8 cores (one per DRAM bank, `[32, 256]`) and pass `LayerNormShardedMultiCoreProgramConfig` plus a HiFi4/fp32-accumulate compute config. The shard spec is deliberately `_width_sharded_l1(2048)`, the memory config the DRAM-sharded qkv projection already reshards into, so the first norm's output crosses into attention with no conversion at all. | **19.82 → 4.92 µs** standalone, and *more* accurate: max error against a torch fp64 reference falls from 6.711e-02 to 1.686e-02, because the shipped call accumulated the sum of squares in bf16. In the layer, 40.208 → 13.326 µs including both reshards. |
| **B** | **The router projection reads its 0.5 MB weight from DRAM-interleaved on 4 cores.** 24.916 µs for `[1,1,32,2048] × [2048,128]` is 21 GB/s across 4 cores. N = 128 is 4 tiles, so 4 cores is the ceiling on core count — but not on bandwidth. | row 160; `probes/norm_router_probe.py` | Feed it the width-sharded L1 activation the norm now produces. Same weight, same dtype, same fp32 output. | **24.62 → 5.85 µs** standalone at the **shipped 8-core** shard, **max\|diff\| exactly 0.0** against the interleaved spelling. In the layer, row 182 reads 6.241 µs. (The sweep's 4-core leg reads 4.30, but 4 cores is not what ships and nothing is priced against it: the norm emits an 8-core shard and 8-core norm + 8-core matmul, 4.92 + 5.85 = 10.77, beats 4-core + 4-core's 7.53 + 4.30 = 11.83.) |
| **C** | **Repeated same-input matmuls.** The router projection and `gate_up` read the same `post_attention_layernorm` output; `wqkv` is already one packed projection and `gate_up` is already packed (stage 02 finding B). | rows 160/182 (03), 182/205 (04) | Nothing to pack: the two consumers have different N, different weights, different dtypes (bf16 fp32-out vs bfloat4_b), and one is sparse. What *is* shared is the activation, and after finding B they share it in **one** L1 shard rather than each reading DRAM. | no separate change |
| **D** | **Material collectives: 66.765 µs, 16.1% of the stage-03 layer.** Four ops, two all-reduces. | rows 156–157, 195–196 | see the collective family below | **60.644 µs, 16.7%** of the smaller stage-04 layer (rows 176–178, 218–220, including the two clones the persistent buffers cost) — 6.12 µs less in absolute terms, from persistent buffers and one link |
| **E** | **Reshard / layout conversions.** Stage 03's layer holds 4 `InterleavedToSharded`, 5 `ShardedToInterleaved`, 4 `UntilizeWithUnpadding` and 1 `TilizeWithValPadding` — 14 ops that only move data. | op-code tally over rows 134–197 | Removed one (`attention_decode_optimized`'s qkv reshard, now satisfied by the norm's output); added two (the norms' input reshards, 0.888 + 0.874) and one (`normed` back to interleaved for `sparse_matmul`, 0.876). Attacked the tilize/untilize round trip in the router directly — see the rejection below. | net +4 ops, −51.8 µs |
| **F** | **Fused matmul-CCL paths.** `wo` → reduce-scatter is the only shape-eligible edge; the other three neighbour a norm or a residual add. | `../multichip_decoder/README.md` limitation 2 | Built and measured, standalone and in the layer. | **rejected, see below** |
| **G** | **Lower-movement residual layout.** | `mesh_plan.md` §4.1 | Re-derived against the stage-04 profile rather than inherited. | **no such family exists here** — see below |
| **H** | **After A and B, `TopK` is the largest single op in the layer outside the expert matmuls**: 26.356 µs on **one core**, 7.3% of the layer, with a 4.190 µs `FillPad` in front of it. | rows 183–184 | Swept: `sorted=False` 33.78 vs `sorted=True` 33.81 µs standalone; bf16 input 31.81 (and forbidden — routing must select in fp32 logit space); logits staged in L1 measured **0.4380/0.4383 ms on the layer against 0.4346/0.4348**, i.e. worse. | **no lever found.** Named limitation; a 128-wide `topk` over a single row is one core by construction and the `FillPad` is `ttnn.topk`'s own, because the logits are logically 1 row inside a 32-row tile |

## Coherent families

### Residual layout — replicated stays, and this is a derivation, not an inheritance

The contract is explicit that a lower-movement family may not be rejected by
measuring it with an immediate restore to the old contract. The reason nothing
was measured here is stronger than that: **there is no lower-movement variant to
measure.**

Both consumers of the residual stream — `wqkv` (column-parallel over the full
2048) and the router + `gate_up` pair (likewise) — need the whole hidden vector.
So:

| contract | collectives per layer | primitive hops |
|---|---|---|
| **replicated residual** (shipped) | AR after `wo`, AR after `down` | 2 RS + 2 AG |
| hidden-sharded residual (512/die) | RS after `wo`; AG before the norm/router; RS after `down`; AG before the next `wqkv` | 2 RS + 2 AG |
| hidden-sharded + distributed RMSNorm | as above, **plus** a stats all-reduce per norm | 2 RS + 2 AG + 2 |

An all-reduce *is* a reduce-scatter followed by an all-gather, so the sharded
contract recreates exactly the traffic it is supposed to remove, and the
distributed-norm variant adds two collectives on top. Decode collectives are
**latency-bound, not byte-bound** — the measured floor is ~11 µs and this
profile's `AllGatherAsync` on a `[1,1,32,512]` payload costs 12.932 and 11.287
µs — so what a decode layer pays for is the *number* of collectives, which is
identical, and the sharded variant's only distinguishing feature is the extra
pair the distributed norm needs.

The one restructuring that would genuinely move less would be making attention
row-parallel in K so it could consume a 512-wide shard directly; that turns the
qkv projection's output into a partial that needs its own all-reduce over a
1280-wide tensor, i.e. strictly more collective, and it gives up the
DRAM-sharded decode projection stage 02 measured at 1.11×.

**The inter-layer residual layout contract is therefore unchanged and is
written down in `README.md` for full-model bringup**: replicated
`[1, 1, B, 2048]`, bf16, TILE, DRAM-interleaved, in and out, with **no**
gather, reshard or all-reduce between layers. Everything stage 04 changed is
*inside* the layer: the two norms' L1 width-shards and the router projection's
activation shard are created and consumed between the layer's own boundaries.

### Collective placement, dtype, persistent buffers

| lever | measured | verdict |
|---|---|---|
| RS+AG vs AG-of-partials | stage 03, 0.4760 vs 0.4801 ms | RS+AG, carried |
| `Topology.Ring` vs `Linear` | stage 03, 0.4766 vs 0.4836 | Ring, carried |
| ~~`num_links` 2 vs 1~~ | ~~0.4365 vs 0.4378 (stage-04 layer, with the threshold tail)~~ | **SUPERSEDED** by the "one ethernet link for decode collectives" row below — do not read this row as a verdict. It was measured **against a layer that still carried the threshold routing tail**, i.e. a slower layer with an extra ~17 µs of single-core work, so the collectives were a smaller share of it and the gap read 0.3%. Re-measured against the layer that actually ships — the tail rejected, the norms and the router projection landed, the collectives on persistent buffers — the same lever is **1.22%** and reads identically at both leg positions over six order-alternating passes. Same lever, smaller denominator, different answer |
| bfloat8_b collective payload | stage 03, 0.4854 vs 0.4766 | rejected; decode is latency-bound so halving the bytes buys nothing and costs precision |
| RS and AG staged in L1 | 0.4368 vs 0.4378 | inside noise; not adopted |
| **first RS fed from L1** — its input is DRAM-interleaved and it costs 18.871 µs where the second RS, fed from L1-interleaved, costs 15.018 for the same shape | **0.4403 / 0.4399 against 0.4348 / 0.4346** | **rejected, 8.2 µs of asymmetry left on the table, and 1.2% worse if taken.** The asymmetry is real and repeats in the stage-03 profile (20.413 vs 16.322), but it is not caused by the input's buffer type: making it L1 costs more than it saves |
| **persistent RS + AG buffers** | **0.4343 / 0.4337** (`layer_levers3.py`) and **0.4335 / 0.4333** (`layer_levers2.py`) against 0.4348 / 0.4346 | **adopted**, ~0.2%, consistent over four paired measurements against a leg-vs-itself spread of 0.05% |
| **one ethernet link for decode collectives** | **0.42875 against 0.43400 ms**, six passes with the leg order alternating, output bit-identical in all twelve legs (`probes/links_probe.py`) | **adopted, 1.22%.** Stage 03 measured the same lever at 0.6% and called it noise; against the stage-04 layer it is ~7–10× the leg-against-itself spread (0.5–0.8 µs). Re-opened by review — `_links` had stopped honouring an explicit `num_links=2`, so the probe could not tell its legs apart — and the figure survived both the repair **and** an order control that each configuration reads identically at both positions. Prefill keeps two links |
| **`matmul_reduce_scatter_async` on the `wo` → RS edge** | standalone at the shipped shapes: fused 30.85–30.91 µs against an unfused 2D matmul + RS of 18.82 + 10.73 = **29.55** | **rejected.** Fused is 4.4% slower than the unfused pair *before* accounting for the real cost: the fused op takes a 2D `MatmulMultiCoreReuseMultiCast` program config, so `wo` gives up the DRAM-sharded config that runs it at 8.228 µs in the layer and pays 18.82 instead. It is also numerically different (max\|diff\| 2.734e-02 against the unfused pair). The first attempt raised `matmul_reduce_scatter_async.cpp:36 mesh_device != nullptr` inside the layer; that was *not* taken as the rejection — `probes/mmrs_probe.py` rebuilt the edge standalone, where it runs, and priced it |

### Precision / fidelity

| lever | measured | verdict |
|---|---|---|
| sharded norm compute config | default 4.26 µs / 3.586e-02 error, HiFi4+fp32acc 4.92 µs / 1.686e-02 | **HiFi4 + fp32 accumulate**, 0.66 µs for 2.1× the accuracy — and still 4× faster and 4× more accurate than the shipped interleaved call |
| DRAM-sharded router weight, N padded 128→256, bf16 and bfloat8_b | 7.34–7.40 µs, plus a 0.45–0.51 µs sharded→interleaved of its own, against the **shipped** L1-sharded-activation matmul's **5.85**; and max\|diff\| 5–7e-02 against the reference logits | **rejected on both counts** — 26–33% slower than what ships, and the only router spelling swept that is not bit-identical |
| expert dtypes, fidelity, `fp32_dest_acc_en` | stage 02 | carried unchanged |

### Rejected: removing the router's ROW_MAJOR round trip

Stage 02 recorded the `untilize → scatter → tilize` round trip in the routing
tail as **not removable**, because `ttnn.scatter` takes only ROW_MAJOR while
every consumer of the dense vector needs TILE. That reasoning is wrong, and
`router_forward_threshold` in `tt/multichip_decoder.py` is the counter-example:
`topk(sorted=True)` puts the 8th-largest logit in column 7, so the dense vector
is `exp(logits − max) · (logits ≥ that)` computed over all 128 columns and
never leaves TILE. It deletes rows 190–197 — a `zeros_like`, two typecasts, three untilizes, a
scatter and a tilize, **17.007 µs** of profile (17.144 µs at the equivalent
stage-03 rows 168–175) — and widens the softmax's `sub` and `exp` from 8
columns to 128.

It is **0.8% slower**: 0.4382 / 0.4382 ms against the then-shipped 0.4348 /
0.4346 in two interleaved passes, and 1.7% slower against the final default in
the last run (0.4355 against 0.4282). The eltwise it adds back is 128 fp32 columns wide
with a broadcast operand where the ops it removes were 8 columns or one row of
ROW_MAJOR data, and that costs more than the layout conversions save. The
output is **bit-identical** (`max|diff| 0.000e+00` on all four dies), which
also settles the tie question the construction raises — with fp32 logits over
K = 2048 no two logits tie at rank 8.

Kept in the module as a measured, correct alternative rather than deleted,
because the arithmetic it establishes is the useful part.

## Where the time goes now

Stage-04 decode layer, 362.828 µs, by op code over rows 154–221:

| op code | µs | share |
|---|---|---|
| `SparseMatmulDeviceOperation` | 82.718 | 22.80% |
| `ReshapeViewDeviceOperation` | 44.752 | 12.33% |
| `ReduceScatterMinimalAsyncDeviceOperation` | 33.889 | 9.34% |
| `MatmulDeviceOperation` (qkv, wo, router, ones, window) | 27.487 | 7.58% |
| `TopKDeviceOperation` | 26.356 | 7.26% |
| `AllGatherAsyncDeviceOperation` | 24.219 | 6.68% |
| `UnaryDeviceOperation` | 19.697 | 5.43% |
| `LayerNormDeviceOperation` (2 residual + 2 per-head) | 19.299 | 5.32% |
| `BinaryNgDeviceOperation` | 17.204 | 4.74% |
| `SdpaDecodeDeviceOperation` | 9.816 | 2.71% |
| everything else | 57.391 | 15.82% |

**The replicated fraction is 84.738 µs — 23.35%** (rows 154–155 and 180–181, the
two norms, 13.326; rows 182–201 and 203, the router, 71.412), down from 110.451
µs and 26.63% of the larger stage-03 layer. Of what remains, `TopK` + its
`FillPad` is 30.546 µs, a single core and an op-internal pad fill, and the
router's arithmetic tail is another 34.6 µs of ops that are each 1–6 µs of
launch floor.

## What the next round would look at, with the measurement

Named rather than silently dropped. All three are *single-die* costs that the
mesh does not touch, i.e. they belong to `optimized_decoder.py`'s scope rather
than to this pass, and all three are visible in the stage-04 window:

* **Single-core ops inside attention**: 28.478 µs of rows 156–175 runs on one
  core, of which 26.289 is seven substantive ops — two per-head `LayerNorm`,
  two `RotaryEmbedding`, `NLPCreateQKVHeads` and two `PagedUpdateCache` — and
  the other 2.189 is four single-core reshards. Finding A's fix does not
  transfer — these are 128-wide, so a width shard has almost nothing to spread.

  **`rotary_embedding_llama` is now measured rather than named**, because
  review would not accept the argument above as a verdict on a *different op*.
  It is **3.84 → 1.26 µs, 3.05×, bit-identical** standalone
  (`probes/rope_probe.py`), which against rows 163–164's 9.358 µs would be
  1.4–1.7% of the layer — and it is **rejected**, on a cache-convention
  conflict rather than on speed: RoPE runs before K is written, so the KV cache
  carries the rotary's channel convention, and prefill (untouched by this
  lever) writes HF-ordered keys. Meta decode against a prefill-primed cache
  reads **PCC 0.1932974** where a fresh cache reads 0.9999697
  (`probes/rope_layer_probe.py`). Adopting it is a whole-layer change —
  prefill's rotary and weights, and the KV cache contract — not a decode-local
  one. Built, wired, measured and backed out; left runnable behind
  `upload_multichip_weights(meta_rope=True)`. See `README.md` limitation 4.
* **The expert M-padding compaction**, 41.313 µs in three `ReshapeView` (rows
  206, 211, 214). Both removals are measured and rejected in
  `../optimized_decoder/work_log.md`: staying rank-6 is 6% slower, and
  `output_tile=Tile([1,32])` is 1.07× faster but silently corrupts every
  downstream consumer.
* **Dynamic `nnz`**, ~26 µs of the 82.718 µs expert pair, blocked on TTNN
  offering an upper-bounded `nnz` (`../multichip_decoder/README.md`,
  limitation 1).
