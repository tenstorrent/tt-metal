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

> **42.79 ms** prefill wavefront, SP=4 x TP=8, all 32 chips, PCC 0.9991, commit `b9356ef71e3`.

Everything else is one of:

| kind | examples | trust |
|---|---|---|
| **measured** (Tracy / traced microbenchmarks) | per-op costs, collective costs, peak TFLOPS, MLP floor per split | yes -- these are the "now" columns |
| **estimated** (formula or judgment) | the 8-11 ms budget's "target" column, every pipeline-parallel number | no -- not verified, do not quote as results |
| **implemented, unverified** | the `QWEN36_REPL_RESIDUAL` path | its A/B produced no output, including the baseline control; treat as broken |
| **implemented, A/B in flight** | the `QWEN36_ATUPE_OPTS` port of Aniruddha's round-4 GDN/SDPA/L1 changes | default OFF until verified; see the port section |

**Verified savings so far: zero.** The 8-11 ms plan below is a budget of estimated cuts against
measured costs; none of the cuts has been demonstrated.

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

**Status: implemented; A/B ran and produced NO usable output -- including the baseline
control. Treat as broken until debugged.** The four-run A/B (baseline TTFT, baseline PCC, repl
TTFT, repl PCC) completed with exit 0 but none of the runs printed a wavefront time, a PCC, or
even a pytest PASSED/FAILED line. Because the *baseline* also printed nothing, the likely cause
is that the refactor broke the default path too (e.g. the `tpc` import or `residual_all_reduce`
signature), not the new mode. Diagnosis was started and interrupted; the next step is a single
baseline run with full stdout captured. Off by default, so `b9356ef71e3` behaviour is unaffected
only if the default path is in fact intact -- verify this first.

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
| baseline (pre-port HEAD) | *pending* | | | running |
| ported, all three on | *pending* | | *pending* | chained behind baseline |

Results to be filled in when the runs land; per-feature isolation runs follow only if the
combined run moves the number.

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
