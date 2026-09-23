# Qwen3.5-2B prefill on the 32-chip Blackhole Galaxy: measured roofline and where the time goes

Companion to `SP_PREFILL_README.md`, which covers the SP x TP wavefront itself. This file
records what the hardware actually delivers, why the under-2 ms target is out of reach, and the
per-op cost ranking that any further optimisation should be driven by.

Everything below is **measured on this box**, traced (not eager), best-of-N. Reproduce with the
scripts in `tests/perf/roofline/`. Numbers dated 2026-09-22.

## Status: what is a result and what is not

Read this first. Over the course of this work several estimates were presented alongside
measurements, and they were not always kept apart. This section is the line between them.

**Exactly one performance number in this document is a result:**

> **37.43 ms** prefill wavefront (min 37.41), SP=4 x TP=8, all 32 chips, **PCC 0.9993**, argmax equal,
> 2026-09-23, **defaults only -- no env overrides**. Two verified changes stacked:
> Aniruddha's round-4 opts (42.78 -> 39.80) and the replicated residual (39.80 -> 37.43).
>
> Prior best **42.79 ms**, commit `b9356ef71e3` (reproduced 2026-09-23 on the reset device at 42.78 ms).

Everything else is one of:

| kind | examples | trust |
|---|---|---|
| **measured** (Tracy / traced microbenchmarks) | per-op costs, collective costs, peak TFLOPS, MLP floor per split | yes -- these are the "now" columns |
| **estimated** (formula or judgment) | the 8-11 ms budget's "target" column, every pipeline-parallel number | no -- not verified, do not quote as results |
| **verified, default ON** | the `QWEN36_REPL_RESIDUAL` path | **39.80 -> 37.43 ms**, PCC 0.9993, argmax equal (2026-09-23). Collectives 6/layer -> 4/layer. |
| **verified, default ON** | the `QWEN36_ATUPE_OPTS` port of Aniruddha's round-4 GDN/SDPA/L1 changes | **42.78 -> 39.80 ms**, PCC 0.9994, argmax equal (2026-09-23); per-feature attribution pending |

**Verified savings so far: 5.35 ms (42.78 -> 37.43), from two changes:**

| change | wavefront | PCC | commit |
|---|---|---|---|
| baseline | 42.78 ms | 0.9991 | `b9356ef71e3` |
| + Aniruddha's round-4 opts (`QWEN36_ATUPE_OPTS`) | 39.80 ms (**-2.98**) | 0.9994 | `34a28258232` |
| + **replicated residual** (`QWEN36_REPL_RESIDUAL`) | **37.43 ms** (**-2.37**) | 0.9993 | this commit |

Everything else in the ~11 ms plan below is still an estimated cut against a measured cost.

## The 42.79 ms result: exact configuration and run details

| | |
|---|---|
| model / input | Qwen3.5-2B, ISL 4096, **prefill only** (TTFT) |
| chips | **all 32** Blackhole Galaxy = 4 SP spans x 8 TP |
| sequence parallel | SP=4: 4096 tokens -> 4 spans of 1024; layer-wise wavefront, die d -> d+1 over **MeshSockets** carrying GDN state + KV prefix |
| tensor parallel | TP=8 inside each span: weights sharded; all 6 collectives/layer live here |
| fabric | `FABRIC_1D`, `Topology.Linear` (Ring overridden -- Galaxy rows are lines, not rings) |
| env | `SP_DIES=4 SP_TP=8 QWEN36_NO_AGMM=1` |
| precision | bf4 gate/up, bf8 down, **fp32** GDN out-proj reduce-scatter (bf16 partials gave PCC 0.69) |
| measurement | traced replay, warm, min-of-runs; **wavefront = last die's finish time** |
| correctness | PCC **0.9991** vs a TP=4 oracle, argmax equal |
| commit | `b9356ef71e3` on `arg/qwen36_2b_sp` |

Per-die finish: `d0=38.51  d1=39.51  d2=40.71  d3=42.79 ms`. So **38.5 ms is one die's own
work and 4.3 ms is wavefront tail.** Any optimisation has to attack the 38.5, not the 4.3.

Starting point was **180 ms** (demo TTFT), a 4.2x reduction. The last two steps were turning
AGMM off (44.19 -> 42.79 ms; the fusion was a pessimisation) and Linear topology. The other
32-chip split, SP=8 x TP=4 on the 2D mesh, measured 43.58 ms.

**Superseded 2026-09-23:** 39.80 ms with `QWEN36_ATUPE_OPTS` on (now default); same config
otherwise, PCC 0.9994. See the port section.

**What 42.79 ms excludes:** logits readback, the prefill -> decode handoff, all 8 decode
steps, and host dispatch. It is the device wavefront only. See the next section.

## Prefill vs decode: where SP and TP are used, and the handoff

SP and TP are split **by phase**, not both used in both:

| phase | parallelism | chips | code |
|---|---|---|---|
| **prefill (TTFT)** | **SP x TP** -- SP=4 spans, TP=8 inside each | 32 | `tt/sp_prefill.py` |
| **decode** | **TP only** -- TP=4 | 4 | `model.decode_tp()` |

SP is prefill-only: with one token per step there is no sequence to split. Prefill uses both;
the 15.1 ms of collectives comes from the inner TP=8, not from SP.

**The handoff is a host round-trip, and it is unmeasured.** `tt/sp_handoff.py` pulls the KV
cache and GDN state (left unsharded on the last die by SP prefill) to host as torch tensors,
reshapes them into the per-device shards the TP=4 decode model expects, and uploads them
back. Its own docstring: *"v1 goes through host torch tensors (no on-device transfer)."*

For the OSL=8 target this means:

    real e2e  =  42.79 ms prefill  +  host handoff (unmeasured)  +  8 decode steps (unmeasured)

Neither of the last two terms appears in any number in this document. The 63 / 68 ms figures
in `SP_PREFILL_README.md` are from the earlier 4-die QuietBox configuration, not this one. Any
e2e claim needs the decode path and handoff measured first; an on-device handoff would remove
the host copy entirely.

## Headline: 2 ms is not reachable; ~6.5 ms is the floor

| | per layer | 24 layers |
|---|---|---|
| 2 ms target | 83 us | 2.00 ms |
| measured compute floor (best split, MLP only) | 180 us | **4.31 ms** |
| measured compute floor (all ops, MLP ~= 66% of layer FLOPs) | ~270 us | **~6.5 ms** |
| **current, verified** | ~1600 us | **42.79 ms** |

The floor above already assumes zero collectives, zero data movement and perfect overlap. 2 ms
would need ~90% MFU on shapes that measure 17-28%.

### Why: the model is too small to fill the machine

One Blackhole chip's peak, swept over shapes (`bench_peak.py`):

| shape (M x K x N) | TFLOPS |
|---|---|
| 2048 x 4096 x 4096 | **193** |
| 1024 x 2048 x 6144 | 143 |
| 2048 x 2048 x 2048 | 116 |
| 512 x 2048 x 6144 | 86 |

193 TFLOPS is the practical ceiling, **not** the ~300 on the spec sheet. Qwen3.5-2B split 32
ways never presents a shape that big: 2B params / 32 chips = 64M params per chip, and 4096
tokens / 8 spans = 512 tokens. Per-chip MLP wall-clock for one layer, for every split of 32
chips (`bench_split.py`):

| split | tok/chip | MLP/layer | MFU | x24 layers |
|---|---|---|---|---|
| SP4 x TP8 | 1024 | 211 us | 24% | 5.05 ms |
| **SP8 x TP4** | 512 | **180 us** | **28%** | **4.31 ms** |
| SP16 x TP2 | 256 | 229 us | 22% | 5.50 ms |
| SP32 x TP1 | 128 | 306 us | 16% | 7.35 ms |

**Adding chips to a layer makes it worse.** 32-way token split runs at 16% MFU, 8-way at 28%:
more parallelism shrinks each matmul below what fills 120 Tensix cores. This is the single most
important fact for anyone planning a "use all 32 chips on one layer" scheme.

### L1 residency buys nothing (this refutes the SRAM / weight-reload thesis)

Same shape, weights in L1 vs DRAM, traced (`bench_l1_vs_dram.py`):

```
weights DRAM:  289.7us   33.4 TFLOPS
weights L1:    290.0us   33.3 TFLOPS
```

0.1% apart. bfloat4_b vs bfloat8_b is likewise a wash. **These matmuls are utilisation-bound,
not memory-bound**, so keeping the whole 2B model in SRAM, and the tt-blaze reload pipeline as a
*latency* lever, do not apply. (Reload solves kernel-binary staging, which is not what we pay
for here.) One layer's weights are 48 MB against 189 MB of L1, so residency is *possible* -- it
just does not help.

## Where the 42.79 ms actually goes

Device-kernel time by op, from a Tracy capture of the SP x TP run, **excluding
Send/RecvDirectAsync** (those are wavefront *wait*, not work -- they read as 96% of raw total
and will mislead you):

| op | share of work | per call | cores used |
|---|---|---|---|
| **AllGather + ReduceScatter** | **45%** | 79 / 155 us | 10-12 |
| Matmul | 15% | 41 us | 77 |
| ChunkGdnScan + ChunkGdnPrep | 11% | 146 / 67 us | **8** / 64 |
| SDPA | 9% | 494 us | 120 |
| Slice / BinaryNg / Tilize / Untilize / Concat | 10% | 4-15 us | 113-120 |

We spend **3x more on collectives than on matmul.** Per layer that is 4 all-gathers + 2
reduce-scatters, in two repeating groups of three. Every one is in a shared framework file,
not in this model's code:

| # | collective | where | dtype | cost |
|---|---|---|---|---|
| 1 | AllGather -- **norm stats** | `models/common/rmsnorm.py:234`, inside `attention_norm` | bf16 | 79 us |
| 2 | AllGather -- **norm output** | `models/tt_transformers/tt/distributed_norm.py:171`, `attention_norm` | bf16 | 79 us |
| 3 | ReduceScatter -- **attn / GDN out-proj** | `tt/gdn/tp.py:795` | **fp32** (GDN) / bf16 (FA) | 196 / 130 us |
| 4 | AllGather -- **norm stats** | `rmsnorm.py:234`, inside `ffn_norm` | bf16 | 79 us |
| 5 | AllGather -- **norm output** | `distributed_norm.py:171`, `ffn_norm` | bf16 | 79 us |
| 6 | ReduceScatter -- **MLP down-proj** | `tt/mlp.py:368` | bf16 | 130 us |

The accounting closes against the profile, which is how this mapping is known rather than
guessed:

    ReduceScatter, per iteration:
      18 GDN layers x 196.4us (fp32)  = 3535us
       6  FA layers x 130.2us (bf16)  =  781us
      24 MLP layers x 130.2us (bf16)  = 3125us
                                total = 7441us     profile: 7440us
    AllGather:  4/layer x 24 x 78.8us = 7565us     profile: 7630us
                                                   -------
                                                   15.1 ms

The op counts confirm it independently: the capture holds exactly **1152 fp32 reduce-scatters**
(= 18 GDN layers x 64 iterations) and **1920 bf16** (= [6 FA + 24 MLP] x 64). That is how the
fp32 one is known to be the GDN out-proj specifically.

**Four of the six exist only because the residual is hidden-fractured.** The reduce-scatter at
#3 and #6 leaves the residual sharded on the hidden dim; that fracture forces
`is_distributed_norm() -> True`, which costs a stats gather (#1, #4) and then an output gather
(#2, #5) to rebuild the full hidden dim -- which the next column-parallel matmul wanted all
along. We scatter, then immediately gather back. None of this is in the SP wavefront: the
sockets carry GDN state and KV prefix between spans and cost almost nothing.

### Collectives are a fixed ~120 us each, NOT bandwidth

Isolated, traced, 8 chips, Linear topology (`bench_ccl.py`):

| collective | bytes | time |
|---|---|---|
| all_gather bf16 (1024x256) | 0.52 MB | 123.0 us |
| reduce_scatter bf16 (1024x2048) | **4.19 MB** | **117.5 us** |
| all_gather fp32 | 1.05 MB | 207.6 us |
| reduce_scatter fp32 | 8.39 MB | 201.3 us |
| all_gather, tiny (1024x32) | 0.07 MB | 47.8 us |
| **fused ttnn.all_reduce** (1024x2048) | 4.19 MB | **223.3 us** |

8x the payload for the same time. There is a large fixed cost and only a weak per-byte term, so
**collective COUNT is the lever, not message size or tensor layout.** Note the isolated 123 us
exceeds the in-model average of 78.8 us, so the in-model figure is already partly overlapped --
the 79 us is not idle skew waiting to be scheduled away.

fp32 costs ~1.7x bf16. 1152 of the reduce-scatters (one per GDN layer) run in fp32. That is
deliberate, not an oversight: bf16 partial sums on the GDN out-proj took PCC to ~0.69. See
`ccl_cast()` in `tp_common.py`.

## Change in flight: replicated residual (`QWEN36_REPL_RESIDUAL=1`)

**Status: implemented; A/B produced no numbers -- but that is now attributed to a wedged
device, not to this code.** The four-run A/B (baseline TTFT, baseline PCC, repl TTFT, repl PCC)
printed neither a wavefront time nor a PASSED/FAILED line. Each run had `timeout 1500`; a later
baseline with full stdout captured showed exactly what that looks like: the process hangs
silently at `Loading 24 transformer layers` (weight upload, before any forward pass) and is
killed at 25 min with no error. All six modules import cleanly, so it was not an import break.
See *Hang diagnosis* under the port section for the evidence and recovery. **Resolved 2026-09-23: verified at 37.43 ms, PCC 0.9993, now the default.** Three bugs, all mine:

1. **The embedding is hidden-fractured.** `model.embd` returns `[1, T, dim/tp]`, and under the
   fractured layout every layer keeps it that way. Once the out-projections return replicated
   tensors the very first residual add broadcasts `dim/tp` against `dim` -> `Invalid subtile
   broadcast type`. Fix: `tp_common.replicate_residual` all-gathers once after the embedding,
   per die, for the whole model.
2. **DistributedNorm was still wrapping every norm.** Its gather-then-norm branch fires when
   `is_distributed_norm()` is False *and* `is_multichip` is True -- so turning the distributed
   norm off silently turned on a DIFFERENT gather, one per norm, and double-gathered a
   replicated stream. Fix: skip the wrapper entirely under this path (`layer.py`, `model.py`);
   the bare `RMSNorm` already holds the full replicated gamma.
3. **The full-attention out-proj was never converted.** `gdn/tp.py` (x3) and `mlp.py` were
   switched to `residual_all_reduce`, but `attention/tp.py` had three more `tt_all_reduce` sites
   of its own. GDN and MLP layers returned replicated tensors while FA layers still returned a
   1/8 shard -- the debug trace showed exactly that: `x [1,1,1024,2048]` vs
   `attn [1,1,1024,256]` at the first `fa=True` layer. Fix: all 7 sites go through one helper.

The **fused** `ttnn.all_reduce` (one collective instead of two) still hangs in-model on the
layer-0 GDN out-proj, and that is NOT explained: it works in isolation in both bf16 (26 ms) and
**fp32 (34 ms)** on the same `(1,8)` submesh at `num_links=2`, so the fp32 hypothesis is refuted.
The shipped default is the **composite** (reduce-scatter in the input dtype -> typecast the final
shard to bf16 -> all-gather), which is two collectives per half-layer instead of the old three.
`QWEN36_REPL_MODE=fused` keeps the one-op path for whoever gets to the bottom of it; a third
collective per layer is still available there.

Today each half-layer pays three collectives:

```
reduce_scatter  ->  norm stats all_gather  ->  norm output all_gather
(residual left fractured on hidden)   (forced DistributedNorm)   (re-materialise full hidden)
```

The fracture is what forces the distributed norm, and the norm's output gather rebuilds exactly
the full hidden dim the next column-parallel matmul wanted anyway. Classic Megatron TP instead
keeps the residual **replicated**: one fused all-reduce, both norm gathers local.

Arithmetic from the table above, per half-layer: `117 + 48 + 123 = 288 us` today vs `223 us`
fused. Using in-model rather than isolated rates the saving is ~355 us/layer, i.e. **~8.5 ms of
a hoped-for 42.8 -> ~34 ms**. The activation-memory cost of replicating is the reason the
fractured layout exists at all, and it does not bind on a 2B model with 189 MB of L1.

Implemented as `tp_common.residual_all_reduce()` behind one predicate,
`tp_common.repl_residual_enabled()`, which also forces `is_distributed_norm() -> False` and
`agmm_disabled() -> True`. These three gates MUST move together; they have drifted apart twice
before and the failure mode is a silent K mismatch in the next matmul.

One subtlety worth knowing: `tt_all_reduce` **ignores** `cluster_axis` on a flat mesh (it has a
`1 in mesh.shape` branch), which is why every call site passes `cluster_axis=0` even on a (1,N)
submesh where axis 0 has extent 1. `ttnn.all_reduce` does *not* ignore it and would silently
reduce over a single device, returning the unreduced partial. `residual_all_reduce` passes
`cluster_axis=None` on a flat mesh.

## Things that do not work, and why

- **All 32 chips on one layer, one layer resident at a time (tt-blaze "partial unroll" shape).**
  Killed by the split table: 32-way is 16% MFU vs 28% at 8-way. Fewer, bigger shards win.
- **Replicating MLP weights and splitting tokens to avoid the MLP all-reduce.** The replacement
  token-dim all_gather costs 123.8 us -- the same fixed cost as the collective it removes --
  while the MLP compute goes 211 -> 306 us. Net loss.
- **Keeping the model in L1 / tt-blaze runtime binary reload as a latency lever.** See above:
  L1 vs DRAM is 0.1%.
- **More ethernet links.** `TT_CCL_LINKS=3` and `=4` both fail with
  `fabric.cpp:184 link_idx < candidate_eth_chans.size()`. Only 2 channels exist on this mesh.
  The `tt-multichip-ccl-review` skill's "Galaxy = 4 links" is Wormhole TG, not BH Galaxy.
- **Wider core grids for collectives.** Grid 72 -> 108 cores changed nothing; they are not
  core-bound.
- **AGMM (`all_gather_minimal_matmul_async`).** A pessimisation here: 44.19 ms with vs 42.79 ms
  without. Keep `QWEN36_NO_AGMM=1`.
- **SP=16.** The mesh caps at 8 columns. SP16 x TP2 also measured 51.76 ms e2e -- the 16-hop
  wavefront costs more than the cheaper collectives save.

## Pipeline parallel: a paper sweep, never built, priced with the wrong cost model

**Nothing in this section was built or measured.** It is a formula swept over stage count.

Each row is pipeline parallel with `S` stages, the leftover chips doing tensor parallel
*inside* each stage so `S x TP = 32`, minimised over the microbatch count `M`:

    total = (M + S - 1) * (24 / S) * T_layer(4096 / M)
    T_layer(span) = 0.82 ms + 1.67 us * span

| stages | chips | best M | tok/microbatch | "optimum" | collectives inside a stage? | is the 0.82 ms fair? |
|---|---|---|---|---|---|---|
| 4 | 4 x TP8 = 32 | 5 | 819 | 105.0 ms | yes, 6/layer | roughly -- same TP as measured |
| 8 | 8 x TP4 = 32 | 8 | 512 | 75.4 ms | yes, 6/layer, fewer hops | somewhat too high |
| 16 | 16 x TP2 = 32 | 11 | 372 | 56.2 ms | yes, 6/layer, 1 hop | too high |
| **24** | **24 x TP1 = 24 (8 idle)** | 14 | 293 | 48.4 ms | **none** | **wrong** |

**The flaw:** the only measured input, `T_layer`, was fitted on **SP=4 x TP=8 -- the running
system** -- so the 15.1 ms of collectives is baked into the 0.82 ms fixed term. The TP=1 row is
the one with *zero* collectives, and it is the one priced with a cost model that includes them.
Its 48.4 ms is therefore an overestimate by an unknown amount; the TP=1 per-layer cost has never
been measured (`n_local_kv_heads` is unset unless `sequence_parallel=True`, which is what blocked
it).

**Why TP=1 has zero collectives, verified in code:** at `num_devices=1`,
`is_distributed_norm()` returns False from its first line (`if not self.is_multichip`), so the
norm gathers at `rmsnorm.py:234`, `distributed_norm.py:137` and `:171` are all skipped, and
`tt_all_reduce` hits `if mesh_shape == [1, 1]: return input_tensor` and no-ops. Nothing is split,
so there is nothing to reassemble; and with no partial sums the fp32 constraint on the GDN
out-proj disappears too. GDN state and KV also never move: stage l owns layer l and its
recurrent state feeds the next microbatch on the same chip. The only inter-chip traffic is one
point-to-point socket send of the activation per stage boundary per microbatch
(`[293 x 2048]` bf16 = 1.2 MB, ~14 us measured for `RecvDirectAsync`).

Each stage must be its own **1-device submesh** for this to hold. Hand the model a 24-chip mesh
and `is_multichip` is True and all six collectives fire.

**Why PP still does not look like the route:** zero collectives does not mean fast. The
`(M + S - 1) / M` factor is 37/14 = **2.6x wasted pipeline slots**, because 4096 tokens cannot
fill a 24-deep pipeline. Even a corrected `T_layer` in the 0.6-0.9 ms range lands ~22-33 ms,
which straddles the current 42.79 ms rather than beating it clearly. The one conclusion the
table supports is directional: more stages is better *because* it drives TP toward 1 and deletes
collectives. The absolute numbers should not be quoted.

The honest use of this analysis is as evidence for attacking collectives **inside** the current
SP x TP design -- PP's zero-collective property without its pipeline-occupancy tax.

## Aniruddha's measurements (atupe/qwen35-sp-prefill, 4-die QuietBox, TP=1)

Reported 2026-09-23, his config is SP=4 x **TP=1** on 4 dies -- not ours:

| config | GDN layers | FA layers | note |
|---|---|---|---|
| SP=4 (TP=1) | **65%** of TTFT | 21% | pipeline fill (GDN state + KV to the last chip, sockets) **~4 ms** |
| TP=1 | **74%** of TTFT | 18% | |

Two things carry over to our SP=4 x TP=8 numbers. GDN dominating is consistent with our
profile (GDN scan + prep + its data-movement chain is ~18% of *work* here, but that is after
collectives take 45%; at TP=1 there are no collectives, so GDN's share rises). And his ~4 ms
socket fill matches our measured wavefront tail (d3 - d0 = 4.28 ms) almost exactly -- so that
tail is the state/KV handoff, and it is the same size at TP=1 and TP=8.

## Porting atupe/qwen35-sp-prefill round 4 to SP=4 x TP=8 (2026-09-23)

Branch `atupe/qwen35-sp-prefill` (3 commits, **no common ancestor** with ours -- a squashed
snapshot of a different base, so nothing cherry-picks; ported at file level). His round-4
commit `c58a31a46d0` changes four model files:

| change | file | his measurement (T=1024, tp=1) |
|---|---|---|
| **KDA fused conv1d+SiLU+QKV-split** replacing concat/conv1d/silu/slice | `gdn/tp.py` | **643 -> 316 us**, PCC 0.99999 |
| SDPA q_chunk=128 / k_chunk=256 on an 8x8 grid | `attention/tp.py` | **1198 -> 703 us** on the 4096-key die, PCC unchanged |
| L1-resident SiLU(z), gate multiply, conv row-major intermediates | `gdn/tp.py` | removes ~86 + 36 + 19.5 us of L1->DRAM |
| L1-resident SiLU(gate)*up | `mlp.py` | removes 106 us L1->DRAM + 94 us DRAM->L1 |
| on-device argmax + optional async socket FIFO | `sp_prefill.py` | **not ported** (see below) |

### As written, all four perf changes are inert on our config

Every one carries the gate `self.mesh.get_num_devices() == 1`. His branch is the 4-die
QuietBox at TP=1. On the Galaxy each span's submesh is `create_submeshes(MeshShape(1, tp))`
(`sp_prefill.py:209`), so at TP=8 `get_num_devices()` is 8 and **every gate is False**. Dropping
his files onto our branch and re-running would have produced exactly 42.79 ms and told us
nothing. That is the first finding of this exercise and the reason the port needed edits rather
than a copy.

Nothing in the optimisations themselves is TP=1-specific: `tw["conv_taps"]` is already
per-device sharded by `shard_small` (that *is* the per-device depthwise conv the op wants), and
every L1-resident intermediate is 8x smaller at TP=8 than in his measurements, so L1 is less
contended, not more.

### What was done

- The four changes were applied with the device-count gate replaced by one predicate,
  `tp_common.atupe_opts_enabled()` (`QWEN36_ATUPE_OPTS`). **Default off until verified**; the
  A/B sets `=1`. Unset is byte-identical to before the port. His per-feature opt-outs
  (`QWEN36_SP_KDA_CONV=0`, `QWEN36_SP_L1_RES=0`, `QWEN36_SP_SDPA_LEGACY=1`) still work under it,
  so each change can be isolated.
- **Chunk size had to change.** The KDA op validates `chunk % 32 == 0`, `chunk <= C`, and
  `C % chunk == 0`. His `channel_chunk_size=512` is at C=6144; our per-device width at TP=8 is
  **C=768** (2B GDN: 16 k-heads + 16 v-heads x 128 -> 6144 / 8), and 512 does not divide it.
  `tp_common.kda_channel_chunk(C)` picks the largest tile-aligned divisor <= 512: **384 at
  C=768**, and reproduces his 512 at C=6144. `QWEN36_SP_KDA_CHUNK` overrides for sweeps,
  validated against the same rules.
- The op is **already in our build** (`ttnn.experimental.kda.qkv_causal_conv1d_silu`, from the
  DeepSeek KDA work upstream), so no C++ port or rebuild. His 215 differing C++ files are
  base-tree drift and were not touched.
- `sp_prefill.py` was **skipped**: the on-device argmax moves the *total* TTFT stamp and changes
  `prefill_traced` / `_tail` return contracts, but does not touch the wavefront time we compare
  on; the socket-FIFO experiment is opt-in and off. Neither bears on this A/B.
- Every anchor was asserted to occur exactly once before any write, all four files parse, all
  six qwen36 modules import cleanly, and the `Nk/Dk/Nv/Dv`, `_L1`, `new_state`, `os` names his
  code relies on are bound in the enclosing scopes at each insertion point.

### The import test also clears a suspect for the empty replicated-residual A/B

The leading theory for that A/B printing nothing (including its baseline) was a circular import
from the module-level `tpc` import added to `gdn/tp.py`. All six modules import without error,
so that is not it. The baseline re-run with full stdout captured (see below) is what will show
the real cause.

### A/B protocol and results

- **baseline**: `SP_DIES=4 SP_TP=8 QWEN36_NO_AGMM=1`, `QWEN36_ATUPE_OPTS` unset. The process was
  started **before** the port was written, so it runs the pre-port code (full log:
  `base_head.log`).
- **ported**: same env plus `QWEN36_ATUPE_OPTS=1`, launched after the baseline finished
  (`atupe_head.log`). Guarded: not launched if the baseline exited by timeout/kill (124/143),
  the pattern that wedges the device.

| run | wavefront | per-die finish | PCC | status |
|---|---|---|---|---|
| baseline (HEAD, opts unset) | **42.78 ms** (min 42.74) | | | **PASS** in 34 s -- reproduces 42.79; wedge diagnosis confirmed, refactor cleared |
| **ported, all three on** | **39.80 ms** (min 39.77) | d0=35.89 d1=36.78 d2=37.91 d3=39.82 | **0.9994**, argmax 11 = 11 | **PASS** -- both tests, 22 s + 26 s. **-2.98 ms (-7.0%)**; per-die own work 38.51 -> 35.89 |

The first ported attempt failed in 12 s with `TypeError: qkv_causal_conv1d_silu(): incompatible
function arguments`: our nanobind binding is a newer revision of the op than Aniruddha's tree and
requires two extra keyword-only tensors his call omitted -- `actual_start` (a replicated UINT32
row-major `[1]` scalar; `[0]` for zero-offset, allocated **once before any trace capture** because
the trace bakes in its address) and `predecessor_carry` ("for local execution, alias history", so
`cs_rm`). Fixed and re-run; the row above is the fixed run.

**Per-feature attribution** (TTFT only, each feature turned off in turn, healthy device, 2026-09-23):

| variant | wavefront | d0 | vs all-on 39.80 | => that feature is worth |
|---|---|---|---|---|
| all three on (reference) | **39.80 ms** | 35.89 | -- | -- |
| **legacy SDPA** (KDA + L1 on) | 41.44 ms | 37.25 | +1.64 | **SDPA q128/k256 on 8x8: -1.64 ms** |
| **no KDA conv** (L1 + SDPA on) | 40.90 ms | 36.95 | +1.10 | **KDA fused conv: -1.10 ms** |
| **no L1 residency** (KDA + SDPA on) | 40.26 ms | 36.35 | +0.46 | **L1-resident eltwise: -0.46 ms** |
| baseline, all off | 42.78 ms | 38.51 | +2.98 | |

Individual contributions sum to 3.20 ms against 2.98 ms measured together -- close to additive,
~0.2 ms of overlap. The SDPA change, not the GDN conv, is the largest single term at TP=8: six FA
layers x ~273 us each. Note this is the *opposite* ranking from Aniruddha's TP=1 numbers (KDA
conv 643 -> 316 us was his headline), because at TP=8 each chip's conv is 8x narrower while the
per-head SDPA work is unchanged.

**`channel_chunk_size` for the KDA op at C=768** (all three on):

| chunk | wavefront | d0 |
|---|---|---|
| 192 | 39.84 ms | 35.90 |
| **384 (default: largest tile-aligned divisor <= 512)** | **39.80 ms** | 35.89 |
| 768 (single chunk) | 40.08 ms | 36.21 |

384 and 192 tie within noise; 768 is +0.28 ms. The default stands; chunk size is second-order here.

Results to be filled in when the runs land; per-feature isolation runs follow only if the
combined run moves the number.

### Hang diagnosis: the first A/B attempt timed out, and it was the device, not the code

The first baseline of this A/B ran for 25 min and was killed by its timeout (`BASE_EXIT=124`)
having printed no result. Its log ends at `Loading 24 transformer layers` (15:16:32) and is
silent for the next 25 minutes: it hung during **weight upload**, before any forward pass,
trace capture, or collective. That is the same signature as the four blank replicated-residual
runs the day before.

Why the code is the unlikely culprit:

- The baseline process was started *before* the port was written, so it ran pre-port code.
- The earlier refactor (`dc36369ff8f`) changes nothing at model-construction time that touches
  the device; and all six qwen36 modules import cleanly.
- The worker was at ~300% CPU the whole time -- busy on the host, blocked on the device.

Why the device is the likely culprit:

- On 2026-09-22 `bench_allreduce.py` hung on a second submesh and was killed by `timeout` mid
  collective (exit 124). **Every mesh run after that hung** -- the four A/B runs and this
  baseline, five in a row, ~2 hours -- and nothing had worked on the mesh in between.
- The "HEALTH OK" probe run after that kill opened a **single device** (`open_device(0)`) and
  did one add. That does not exercise the other 31 chips or their dispatch queues, and it said
  OK while the mesh was wedged. This was the blind spot.

Recovery (2026-09-23): `tt-smi -r` -- all 32 PCI devices, rc=0. It prints *"CPLD FW v1.16 or
higher is required ... please continue to use tt-smi -glx_reset instead"*; it did not fail,
and `-glx_reset` must never be used on this box regardless. Then a **full-mesh probe**:
`open_mesh_device(MeshShape(4, 8))`, replicate a tensor to all 32, `add`, `synchronize`, read
back -- passed (sum 65536 / 65536). That is the probe to run after any killed or timed-out mesh
job; the single-device one is not sufficient.

The A/B was then restarted on the verified device as one sequential script
(baseline -> ported -> PCC), each step gated on the previous one producing a wavefront number
and on the device being free. **If the baseline hangs again on a clean device, the HEAD default
path is the culprit and the next step is to bisect against `b9356ef71e3`.** If it passes, the
wedge diagnosis is confirmed.

**Confirmed 2026-09-23 15:47:** on the reset, probe-verified device the same baseline completed in
**34 seconds** at 42.78 ms. The earlier "20-minute" runs were never slow -- they were hung from
the start.

A second, self-inflicted bug found on the way: the original chain waited on
`ps | grep '[p]ytest'` returning 0 as its "device free" check. That pattern matches any bash
whose *script text* contains the word pytest -- i.e. the chain's own wrapper -- so it could
never clear, and the chain sat deadlocked on itself for 35 min. The replacement checks for
open `/dev/tenstorrent` file descriptors, which is the actual busy signal and is what revealed
the device was free.

## E2E with overheads, stated explicitly (2026-09-23)

Every number previously quoted was the **device wavefront**. This section adds what sits on
top, because those overheads are not optional parts of the flow.

### Prefill, ISL 4096, SP=4 x TP=8 on 32 chips

| component | ms | in the wavefront figure? |
|---|---|---|
| die-0 own work (24 layers) | 33.66 | yes |
| **socket wavefront tail** (3 SP hops: GDN state + KV prefix to the last die) | **3.89** | **yes** |
| = **wavefront** | **37.55** | |
| first-token readback (on-device argmax -> 1 token) | 0.70 | no -- added below |
| = **true TTFT** | **38.25** | |
| full [1,1,32,vocab] logits readback | *not counted* | PCC/debug only |

**The socket overhead is already inside our number** and always has been -- it is the
`d3 - d0` spread. Per-die finish: `d0=33.66  d1=34.57  d2=35.62  d3=37.56`. It is **3.89 ms on
32 chips**, matching Aniruddha's 4-5 ms at 4 dies, because it is set by **SP depth (3 hops),
not chip count** -- SP=4 either way. The hops are not equal (0.91 / 1.05 / 1.94 ms): the last
is the largest because die 3 receives three spans' worth of KV prefix. At SP=8 it would be 7
hops, which is why SP=8 x TP=4 measured worse (43.58 ms) despite a better MLP floor.

### The readback overhead that was hiding: 26 ms

The TTFT test reports both `wavefront` and `total`, and until now only the first was quoted:

    before (host argmax):   wavefront 37.43 ms | total 63.58 ms   <- +26.15 ms
    after  (device argmax): wavefront 37.55 ms | total 38.25 ms   <- +0.70 ms

The full logits are `[1, 1, 32, vocab]` gathered across the vocab-sharded mesh -- ~9.7 MB --
and were being read to host purely to call `torch.argmax`. Porting the on-device argmax from
`atupe/qwen35-sp-prefill` (which this port had originally skipped as "not affecting the
wavefront" -- true, but it dominated true TTFT) cuts **25.33 ms**. `prefill_traced` now asserts
the device argmax equals the host one.

### Prefill -> decode handoff: NOT measured, and structurally large

`sp_handoff.py` migrates state to the decode model **through host torch tensors** ("v1 goes
through host torch tensors (no on-device transfer)"). Sized from the config at ISL 4096:

| | MB |
|---|---|
| full-attention KV (6 layers x 2 x 2 heads x 4096 x 256, bf16) | 50.3 |
| GDN recurrent state (18 layers, fp32) | 18.9 |
| GDN conv carry (3 rows) | 0.7 |
| **total across the host round-trip** | **69.9** |

~70 MB down and back. At a ~2 GB/s effective PCIe round-trip that is **~35 ms**, which would
exceed the entire prefill. This is an estimate from bytes, **not a measurement** -- but it is
the single largest unquantified item in the flow and an on-device handoff would remove it
outright. Mohamed is right to call it a big overhead to avoid.

### Decode

TPOT 6.99 ms/token is **device time only** at TP=8 (see the per-layer section). It excludes
sampling and any host interaction per step. Median per-layer gives 6.99; total device work in
the window / steps gives **7.69 ms**, so quote 6.99 as median with 7.69 as the worst case.

### MEASURED e2e: prefill + handoff + 8 decode (2026-09-23)

`tests/perf/perf_e2e_prefill_decode.py` -- one device session, all three phases on one clock.

| phase | ms | share |
|---|---|---|
| prefill, true TTFT | **38.14** | 10% |
| handoff: export device->host | 178.42 | |
| handoff: inject host->device | 96.10 | |
| **handoff total** | **274.52** | **69%** |
| **8 decode tokens** (10.54 ms/token, traced) | **84.29** | 21% |
| **E2E TOTAL** | **396.95 ms** | |

(A further ~57 ms of full-logits readback happens inside `prefill_traced` for PCC/debug and is
excluded; production reads only the on-device-argmax token.)

#### Decode must be TRACED -- 116.79 -> 10.54 ms/token

The first e2e run called `model.decode_tp()` in a Python loop and measured **116.79 ms/token**
against 6.99 ms/token of device time. `decode_tp` is eager: ~24 layers of ops dispatched from
host every step. That is not how decode is meant to be driven.

The reference is `demo/text_demo.py`, and our model already implements the tt_transformers
Generator interface for it (`prepare_inputs_decode` / `ttnn_decode_forward` /
`process_output_decode`). The pattern:

1. `prepare_inputs_decode(...)` once -> **persistent** device input buffers (`dev`).
2. Throwaway eager pass to compile, then `begin_trace_capture` around `ttnn_decode_forward`,
   with a **per-shard `ttnn.argmax` folded into the same trace**, then `end_trace_capture`.
3. Per step: `copy_host_to_device` of only tokens/pos/rope into the persistent buffers (the
   page table is constant and its address is baked into the trace), one `execute_trace`, and a
   readback of the tiny argmax tensor rather than the full vocab.

That is **11x**: 84.29 ms for 8 tokens instead of 934.35. Per-step is now
`9.84 9.69 9.84 9.91 9.80 9.81 10.14 15.26` ms. The remaining ~3.5 ms/token over the 6.99 ms
device figure is the host input update, the sync and the token readback.

#### The handoff is now 69% of e2e, and it should not exist at all

274.52 ms to move ~70 MB out to host and back. It is not bandwidth-bound: it is a per-layer
Python loop of `ttnn.to_torch` -> host reshard -> `ttnn.from_torch`, 24 layers x several
tensors, each its own round-trip. (An earlier estimate of ~35 ms from bytes/bandwidth was
therefore ~8x too optimistic -- the wrong model of the cost entirely.)

**In `demo/text_demo.py` there is no handoff.** Prefill and decode share one model and one
paged KV cache, so decode simply continues. Ours needs a handoff only because SP prefill runs
on four separate submesh models and decode on a fifth. But **the last SP die already holds the
whole 4096-token KV and the final GDN state** -- the data never needs to leave the mesh.

The blocker is not the data, it is the model config: `sp.models[-1]` is built
`sequence_parallel=True`, and under the replicated residual its norms hold full-width gamma
while `decode_tp` feeds hidden-fractured activations (`Gamma's last padded dim needs to equal
tile width`). Make the last span's model decode-compatible -- or reshard on device instead of
through host -- and e2e goes to roughly **38 + small + 84 = ~125 ms**.

## Per-layer cost, prefill and decode (measured 2026-09-23)

Tracy device profile, filtered to the `start`..`stop` signpost window and bucketed by the
per-layer signposts inside it, using the CSV's `DEVICE ID` column. Per-layer figure is the
**slowest device** in that layer; median over layer instances.

**Method note.** An earlier pass reported ~470 ms/die and blamed profiler inflation. That was
wrong: the capture contained the warm-up pass *and* the measured pass (48 signposts for 24
layers) and was never filtered to `start`..`stop`. Filtered correctly the profile reconciles
with the traced wall clock to within 6%, so **the profiler does not inflate** -- always filter
to the signpost window first. Reference pattern: `arg/pplx-embed-upstream`,
`models/demos/blackhole/pplx_embed_0_6b/tests/perf/new_perf_bs1_isl512.py`.

### Prefill -- SP=4 x TP=8, 32 chips, span 1024

| | Now (measured) | Then (projected) |
|---|---|---|
| GDN layer (x18) | **1.245 ms** | ~0.90 |
| FA layer (x6) | **1.524 ms** | ~1.25 |
| 18xGDN + 6xFA | 31.55 ms | 23.7 |
| wavefront tail (3 SP hops) | 3.80 ms | ~3.0 |
| **TTFT wavefront** | **37.43 ms** | **~27 ms** |

`18*1.245 + 6*1.524 = 31.55` vs die-0's measured own work of 33.63 ms -- 94% attributed, the
remainder being gaps between signposted regions.

| GDN layer 1.245 ms | ms | | FA layer 1.524 ms | ms |
|---|---|---|---|---|
| ReduceScatter | 0.316 | | SDPA | 0.408 |
| AllGather | 0.233 | | AllGather | 0.245 |
| Matmul | 0.154 | | ReduceScatter | 0.241 |
| **ChunkGdnScan** (8/120 cores) | 0.146 | | Matmul | 0.136 |
| LayerNorm | 0.075 | | LayerNorm | 0.079 |
| ChunkGdnPrep | 0.067 | | BinaryNg | 0.059 |
| BinaryNg | 0.057 | | Slice | 0.031 |
| QkvCausalConv1dSilu (KDA) | 0.043 | | PagedFillCache | 0.016 |
| **collectives subtotal** | **0.549 (45%)** | | **collectives subtotal** | **0.486 (38%)** |

### Decode -- TP=8, device time only

| | Now (measured) | Then (projected) |
|---|---|---|
| GDN layer (x18) | **0.300 ms** | ~0.24 |
| FA layer (x6) | **0.263 ms** | ~0.21 |
| **TPOT** | **6.99 ms** | **~5.6 ms** |

192 layer instances in the window = 24 layers x 8 steps, so the bucketing is exact. GDN decode
op mix (ms/layer, per-device): Matmul 0.054, ReduceScatter 0.050, AllGather 0.050, BinaryNg
0.042, ReshapeView 0.025, LayerNorm 0.022 -- collectives 0.099, **33%**. Decode is
dispatch/latency-bound: the matmuls are single-tile.

**Caveats.** TPOT is device time only: it excludes the host round-trip in `sp_handoff.py` and
any sampling. It was measured at TP=8 on 8 chips; the production decode path in
`model.decode_tp` is TP=4.

### Where the "Then" numbers come from

Not a target -- a sum of specific, identified cuts:

| cut | basis | GDN prefill | FA prefill |
|---|---|---|---|
| collectives 4/layer -> 3 (fused `ttnn.all_reduce`) + narrower TP | fused op measured 223 us isolated vs 240 for RS+AG; hangs in-model, unexplained | -0.22 | -0.20 |
| `ChunkGdnScan` 8 -> 64+ cores | 8 of 120 cores is the worst occupancy in the model | -0.10 | -- |
| fuse GDN plumbing (prep / BinaryNg / Slice) | ~35k op invocations per capture | -0.04 | -- |
| SDPA further tuning | already -1.64 ms from Aniruddha's config; little left | -- | -0.05 |

Confidence: the `ChunkGdnScan` cut is the best-founded (pure occupancy). The collectives cut is
the least -- it assumes the fused all-reduce is made to work in-model AND that a narrower TP
reduces per-collective cost, and the TP=4 collective benchmark has never completed. If only the
fused op lands and TP stays at 8, TTFT is ~31 ms rather than 27.

**~27 ms is a defensible projection; the ~11 ms floor in the next section needs collectives
essentially eliminated, which at TP=8 there is no measured path to.**

## Path to ~11 ms: measured costs, estimated cuts, zero verified so far

Per-iteration device time from the Tracy capture (capture totals / 64 iterations). The **now**
column is measured. The **target** column is a judgment about how far each item can be cut;
none of it has been demonstrated.

| item | now (measured) | target (estimate) | how | confidence |
|---|---|---|---|---|
| **collectives** | **15.1 ms** | 1.5 | 6 -> 2 per layer via replicated residual (coded, broken); then narrower TP for fewer hops | medium |
| **GDN plumbing** | **7.2 ms** | 2.0 | `ChunkGdnScan` 8 -> 64+ cores; fuse the Slice/Binary/Tilize/Untilize/Concat chain (~35k op invocations per capture) | medium |
| matmul | 4.9 ms | 4.0 | wider N per chip, program configs | high |
| SDPA | 3.0 ms | 2.0 | already on 120 cores | high |
| wavefront tail | 4.3 ms | 1.0 | fewer SP hops once collectives are cheap | medium |
| norms | 0.8 ms | 0.3 | local norms, free with replicated residual | high |
| misc (unary, socket recv) | 0.9 ms | 0.5 | -- | high |
| **total** | **~38 ms** | **~11.3 ms** | | |

**Verified against this table so far (2026-09-23): -5.35 ms total, 42.78 -> 37.43.** SDPA
**-1.64 ms** (the row's target reached), GDN plumbing **-1.10 ms** (KDA fused conv), eltwise DRAM
round-trips **-0.46 ms** (L1 residency), and **collectives -2.37 ms** (replicated residual, 6 -> 4
per layer). The collectives row still has the most left in it: 4 per layer at ~120 us each is
~11.5 ms, against a 1.5 ms target that needs both a narrower TP and the fused single-op
all-reduce working in-model.

So **~11 ms is reachable on measured line items if every estimated cut lands; 8 ms is not**
without also winning on matmul MFU. Commit to ~11-13 ms; treat 8 ms as stretch.

Sanity check that this is not wishful: 11.2 TFLOP / 10 ms / 32 chips = **35 TFLOPS per chip**,
and the actual per-chip shapes already measure 34-80 TFLOPS. We currently get **8.2 TFLOPS per
chip effective**. The whole 4.3x gap is overhead, not shape efficiency -- which is why the plan
is almost entirely "delete overhead", not "make matmuls faster".

Collectives and GDN plumbing are 22 of the 38 ms; everything else is rounding. The order is
forced:

1. **Debug the replicated-residual regression.** Coded, largest single item, currently broken.
2. **`ChunkGdnScan` occupancy.** 8 of 120 cores, 2.6 ms -- the worst utilisation in the model.
3. **Fuse the GDN data-movement chain.** 4.5 ms.
4. **Re-run the SP x TP sweep afterwards.** The optimum will move once collectives are cheap;
   SP8 x TP4 already has the better MLP floor (4.31 vs 5.05 ms) and fewer hops.
5. **Fabric packet size.** The CCL layer logs `Fabric packet size 4352 B is suboptimal for
   transporting 2048 B pages. Configure 8192 B` -- 2 tiles per packet where 4 fit. No env knob
   exists (`TT_METAL_FABRIC_*` has none for packet size); needs a fabric-builder change. Untried.
6. **Conv2d flat cost.** 0.173 ms at span 512 vs 0.174 ms at span 1024 -- ~4.2 ms across 24
   layers of pure overhead. A tap-based rewrite is blocked on 1-3 row shifts not being
   tile-aligned in TILE layout.

None of this touches the decode path or the prefill -> decode handoff, which sit **on top of**
whatever prefill number results and are unmeasured.

## Reproducing

```bash
cd tt-metal && source python_env/bin/activate
export PYTHONPATH=$PWD LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 HF_MODEL=Qwen/Qwen3.5-2B
unset TT_METAL_CACHE
R=models/demos/blackhole/qwen36/tests/perf/roofline

python $R/bench_peak.py        # per-chip peak matmul TFLOPS
python $R/bench_split.py       # MLP wall-clock floor for each SP x TP split of 32 chips
python $R/bench_l1_vs_dram.py  # L1 vs DRAM weight residency A/B
python $R/bench_ccl.py         # collective cost vs bytes and dtype
python $R/bench_allreduce.py   # fused all_reduce vs reduce_scatter + all_gather
```

Gotchas that cost real time here:

- Set `ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)` **before** opening the mesh, or every
  collective dies with "Trying to get un-initialized fabric context".
- Open the full `MeshShape(4, 8)` and carve with `create_submeshes` (plural). Partial opens
  fail the fabric handshake.
- On a `(1, N)` submesh the live cluster axis is **1**, not 0.
- Creating a second submesh in the same process after using the first **hangs**
  (`bench_ccl_hops.py` is the known-bad example; run one TP width per process).
- Do not `pkill` a run mid-fabric-op; it can wedge the device. `tt-smi -r` recovers it.
  **Never `tt-smi -glx_reset`** -- it bricks this box.
