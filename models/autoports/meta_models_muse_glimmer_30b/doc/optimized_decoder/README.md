# Optimized decoder — `meta-models/Muse-Glimmer-30B`

A faster decoder layer for the [fused decoder](../fused_decoder/README.md): same
public contract, same paged prefill/decode semantics, same 131072-token
capability, **2.51–2.56x traced decode** and **1.40–1.63x prefill**, bought with
weight precision, DRAM-sharded decode matmuls, a sharded decode activation
layout, and a prefill kernel the previous pass had ruled out on a single API
error.

| item | value |
| --- | --- |
| implementation | `models/autoports/meta_models_muse_glimmer_30b/tt/optimized_decoder.py` |
| tests | `models/autoports/meta_models_muse_glimmer_30b/tests/test_optimized_decoder.py` (129 tests, 129 passed) |
| baseline | `models/autoports/meta_models_muse_glimmer_30b/tt/fused_decoder.py` |
| device | 1 x Blackhole (`ttnn.MeshShape(1, 1)`, 11x10 compute grid, 8 DRAM banks) |
| precision policy | `attn-bfp8-mlp-bfp4-kv-bfp8-lofi` — BF16 activations/residuals/norms, BFP8 attention weights, BFP4 MLP weights, BFP8 KV cache, LoFi projections |
| weight footprint | 967.8 MB -> **314.8 MB** per layer (one DRAM width-sharded tensor per projection, shared by prefill and decode) |
| acceptance bar | PCC >= 0.995 vs HF on the **released checkpoint** (the functional stage's bar); a documented 0.99 bar on the synthetic harness, see [Two PCC bars](#two-pcc-bars) |
| capability | unchanged: 131072 tokens, batch to 32, non-aligned lengths — `../context_contract.json` |


## Result

Warmed and signposted. `device` is the `Device Time` column of `tt-perf-report`
over a Tracy window (decode divided by its 8 trace replays); `e2e` is warmed host
wall time from `bench/layer_ab.py` (min of 3 rounds) in a **separate run with no
profiler attached**. `baseline` is the fused decoder in the same harness on the
same host.

| kind | window | fused e2e | optimized e2e | speedup | optimized device |
| --- | --- | --- | --- | --- | --- |
| sliding | traced decode @ 2048 | 2.7345 | **1.0909 ms/token** | **2.51x** | 1.072 |
| sliding | traced decode @ 131071 | 2.7340 | **1.0891 ms/token** | **2.51x** | 1.071 |
| full | traced decode @ 2048 | 2.7089 | **1.0601 ms/token** | **2.56x** | 1.049 |
| full | traced decode @ 131071 | 3.2048 | **1.2657 ms/token** | **2.53x** | 1.255 |
| sliding | prefill, 128 tokens | 3.49 | **2.18 ms** | **1.60x** | 2.140 |
| sliding | prefill, 256 tokens | 3.93 | **2.48 ms** | **1.58x** | — |
| sliding | prefill, 512 tokens | 5.18 | **3.31 ms** | **1.56x** | — |
| sliding | prefill, 1024 tokens | 7.44 | **5.32 ms** | **1.40x** | — |
| sliding | prefill, 8192 tokens | 64.05 | **44.30 ms** | **1.45x** | 37.762 |
| full | prefill, 128 tokens | 3.47 | **2.13 ms** | **1.63x** | 2.096 |
| full | prefill, 8192 tokens | 64.65 | **43.40 ms** | **1.49x** | 36.606 |

Device-time prefill against the fused stage's own committed Tracy tables:
8192 tokens 49.318 -> **37.762 ms** (sliding) and 47.975 -> **36.606** (full);
16384 tokens 104.789 -> **81.874** and 111.037 -> **88.133**.

**Decode op count is unchanged** — 44 ops/token on `sliding`, 34 on `full`,
exactly the fused stage's numbers. This stage does not rewrite the decode
topology; it changes what the ops *are*. The 8192-token prefill chunk gains two
ops, the `ttnn.typecast` calls that cast the K/V fill tensors to the BFP8 cache
dtype because `paged_fill_cache` does no conversion of its own; the 128-row window
gains eight more, the conversion pair bracketing each of the four sharded prefill
norms, which are inside the measured candidate: the norms themselves are 4.1x faster
(135.8 -> 33.0 μs) and the whole 128-row window is 1.19x (2549 -> 2140 μs device).

The cost is accuracy, and it is bounded and measured rather than assumed. Against
the same HF reference on the released checkpoint the optimized layer loses
1.7e-3 to 3.0e-3 of PCC relative to the fused one; the worst of all 38
real-weight checks is **0.995079** against the 0.995 bar. See
[Correctness](#correctness) — that thin margin is real and is called out as
limitation 1 rather than smoothed over.


## What changed

Five levers, in descending order of what they were worth.

### 1. Precision policy — 3.07x less weight traffic

The fused decoder ended at the BF16 weight-streaming roofline: 93 % of its
2.710 ms decode step was six matmuls moving 967,835,648 B at 383 GB/s. The only
ways to move that are fewer bytes and a better matmul; this is the first.

| tensor group | dtype | why |
| --- | --- | --- |
| attention weights (`wqkv`, attention gate, `o_proj`) | BFP8 | BFP4 measured 0.977/0.980 prefill PCC on **real** weights — rejected, see below |
| MLP gate / up / down | **BFP4** | the mandatory `$optimize` trial; won on real-weight evidence |
| KV cache | BFP8 | worth 9.9 % at 131071, but only once the SDPA chunking moved with it |
| activations, residual, norms | BF16 | BFP8 is blocked by an exact op contract |
| decode + prefill projections | LoFi | HiFi2 is 69 % slower in decode for +2.4e-4 PCC |

Every projection weight is one **DRAM width-sharded** tensor shared by prefill and
decode. `logs/weight_layout_probe.log` established that both the DRAM-sharded
decode matmul and `minimal_matmul` accept that layout, so there is no second copy
and the per-layer weight footprint *drops* 3.07x.

### 2. DRAM-sharded decode matmuls — 383 -> 429-433 GB/s on the attention rows

All six decode projections are explicit
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` dispatches with a
per-`(role, dtype)` swept `in0_block_w`. The attention rows now run at **84 % of
peak DRAM** and `tt-perf-report` marks them `✅ Optimized`. `in0_block_w` is the
field that matters most on this part and the L1 ceiling is dtype-scaled, which is
why the table is keyed on `(role, dtype)` rather than role alone: `in0_block_w=26`
is fastest for `wqkv` at BFP4 and *illegal* at BFP8.

### 3. A two-grid decode activation layout

The decode step carries two width-sharded L1 layouts, because one grid cannot
serve both sub-blocks (`$optimize` OPT-011):

* **boundary grid, 16 cores** — every `hidden_size`/4608/4096-wide tensor. 16
  divides 208, 144 and 128 tiles exactly, so nothing is shard-padded. It wins the
  whole layer (1.0916 vs 1.1228 ms at 8 cores) *even though* the fused stage
  measured its norm as the slower one, because its 13-K-tile shard is what makes
  `in0_block_w = 13` legal for `wqkv` and the attention gate.
* **MLP working grid, 26 cores** — two reshards at ~2 μs each, against the 57 μs
  that keeping `in0_block_w = 8` on gate/up is worth.

### 4. Prefill: three kernels by row count

This is the part the first pass got wrong, and it is written up in full in
[A rejection that was not earned](#a-rejection-that-was-not-earned).

| rows | kernel | why |
| --- | --- | --- |
| exactly 32 (one M tile) | DRAM-sharded matmul | 3.8x faster than `minimal_matmul` (0.0575 vs 0.2168 ms on `wqkv`) |
| 64 – 1024 (per-role) | **`ttnn.linear` with an explicit 2D-multicast config** | 1.3–2.0x faster than `minimal_matmul` |
| above the crossover | `minimal_matmul` with swept blocking | 80 cores lose to 110 once the matmul is compute-bound |

Independently of the matmul, the four hidden-size norms run **width-sharded in L1 on
an exact 8x2 rectangle below 256 rows** rather than DRAM interleaved, because an
interleaved `ttnn.rms_norm` parallelises over tile rows and a 128-row prefill gives it
only four. That is worth 135.8 -> 33.0 μs per norm and takes the whole 128-token
prefill from 2.57 to **2.18 ms**; see [the write-up](#the-sharded-prefill-rmsnorm--rejected-then-shipped).

### 5. Decode SDPA — the chunking, not the grid

`q_chunk_size = k_chunk_size = 0` (the op chooses) over the whole compute grid,
replacing the fused stage's inherited `q=32 / k=64`. On the `full` (NoPE) layer at
131071, where the SDPA reads the entire cache, that row goes **529 -> 235 μs**.
The core grid is worth 0.5 % of it; the chunking is worth the rest — and it is
also what makes the BFP8 KV cache worth anything at all (1.4041 -> 1.2658
ms/token with the op's chunking, against a dead heat at the fixed one). A
reduced-cache trial measured only under the inherited config would have concluded
BFP8 KV is useless.


## A rejection that was not earned

The first pass of this stage rejected `ttnn.linear` for prefill on one error
message:

```
MatmulMultiCoreProgramConfig: Input B memory layout must be INTERLEAVED
```

That is the *auto-selected fallback* program config talking, not a statement
about the op, and `$optimize` is explicit that a first API error is not a
rejection. Reading `matmul_device_operation.cpp` instead of the error:

* `validate_matmul_mcast2d_config` (`:1368`) accepts a `WIDTH_SHARDED`
  `input_tensor_b` **in DRAM** (`:1541-1553`);
* the extra "per_core_N must equal the in1 shard width" clause that would have
  made it useless here is gated on `buffer_type() != DRAM` (`:1525`);
* the only width-shard clause that does apply is that the in1 shard grid is one
  row tall, which the 8-DRAM-bank weight already is.

So an **explicit** 2D-multicast program config reads exactly the tensor this stage
already ships. Measured against `minimal_matmul` with its swept blocking, per
projection shape at the shipped dtype (`logs/prefill_mcast_probe.log`, 182
measurements; `logs/prefill_mcast_probe_bigrows.log`, 339 more):

| role | 64 r | 128 r | 256 r | 512 r | 1024 r | 2048 r | 8192 r |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `wqkv` | 1.63x | 1.64x | 1.61x | 1.48x | 1.75x | 0.95x | 0.68x |
| `attn_gate` | 1.43x | 1.47x | 1.44x | 1.31x | 1.74x | 0.93x | 0.76x |
| `o_proj` | 1.49x | 1.46x | 1.40x | 1.30x | 1.46x | 0.93x | 0.68x |
| `mlp_gate`/`up` | 2.00x | 1.95x | 1.82x | 1.54x | 0.87x | 0.75x | 0.67x |
| `mlp_down` | 1.59x | 1.57x | 1.53x | 1.49x | 0.88x | 0.99x | 0.77x |

Summed over the six dispatches at 128 rows: **3.060 ms of `minimal_matmul`
against 1.762 ms**. The fused stage's BF16 `ttnn.linear` was 2.67 ms there, so
short prefill goes from *15 % slower than the decoder it replaces* to **1.37x
faster** at the whole-layer level — and to **1.60x** once the sharded prefill norm
lands on top. The first pass had disclosed that regression as a limitation; it was
not a limitation, it was an unearned rejection.

The `>= 2048` columns are why the table hands large rows back rather than assuming
the new kernel is uniformly better. Those candidates are not merely untested:
`out_block_h`/`out_block_w` bounding makes them *legal* (they otherwise overflow
L1), and they are still 5–33 % slower, because by then the matmul is compute-bound
and the 2D-multicast path is pinned to 8 core columns against `minimal_matmul`'s
110 cores.


## Three TTNN findings

The first two miscomputes — the op validates, launches, and returns wrong numbers — so both
are pinned by a test rather than by a comment. The third silently ignores a
layout request, and explains a constraint the rest of the report leans on.

### 1. 2D-multicast with a width-sharded DRAM weight miscomputes when `grid_x != dram_banks`

`bench/prefill_mcast_probe.py --repro` -> `logs/mcast_gx_bug_repro.log`:

```
REPRO dram banks = 8; K=6656 N=4608 rows=128, in1 bfloat8_b TILE
REPRO in1=width-sharded DRAM   grid=(8, 4) per_core_N= 18  non-finite=       0  pcc=0.999744
REPRO in1=width-sharded DRAM   grid=(9, 4) per_core_N= 16  non-finite=       2  pcc=nan
REPRO in1=width-sharded DRAM   grid=(11, 4) per_core_N= 14  non-finite=       3  pcc=nan
REPRO in1=interleaved DRAM     grid=(8, 4) per_core_N= 18  non-finite=       0  pcc=0.999739
REPRO in1=interleaved DRAM     grid=(9, 4) per_core_N= 16  non-finite=       0  pcc=0.999739
REPRO in1=interleaved DRAM     grid=(11, 4) per_core_N= 14  non-finite=       0  pcc=0.999739
```

The same grids are correct with a DRAM-**interleaved** in1, which isolates it to
the width-sharded in1 reader assigning core column `j` to weight shard `j` and
running off the end of the shard set. Nothing in the validator rejects it. Every
candidate in the sweep is therefore gated on a finite-output + PCC check *before*
its latency is reported, and `test_prefill_mcast_table_is_legal` asserts the layer
can never build such a config.

### 2. Sharded `ttnn.rms_norm` miscomputes when its program grid exceeds its shard grid, at `block_h > 1`

Found while evaluating the sharded-prefill-norm candidate below.
`bench/sharded_norm_grid_probe.py` -> `logs/sharded_norm_grid_probe.log`:

| rows | `block_h` | cores | program grid | non-finite | max abs diff |
| --- | --- | --- | --- | --- | --- |
| 32 | 1 | 4, 8 | == shard | 0 | 0.03182 |
| 32 | 1 | 13, 16, 26, 52 | != shard | 0 | 0.03182 |
| 128 | 4 | 8 | == shard | 0 | 0.04022 |
| 128 | 4 | 13 | 11x2 = 22 | 13,222 | `inf` |
| 128 | 4 | 16 | 11x2 = 22 | **75,155** | `inf` |
| 128 | 4 | 26 | 11x3 = 33 | 77,465 | `inf` |
| 128 | 4 | 52 | 11x5 = 55 | 0 | **1.93936** |
| 256 | 8 | 26 | 11x3 = 33 | 152,494 | `inf` |
| 256 | 8 | 52 | 11x5 = 55 | 0 | 2.08197 |

The `block_h = 1` row is the important one for this model: **the shipped decode
norm is correct**, even though it puts a 16-core shard under an `11x2 = 22`-core
program grid, and that probe is what proves it rather than assuming it. At
`block_h > 1` the mismatch corrupts, sometimes to `inf` and sometimes to finite
garbage, with no exception either way.


### 3. The DRAM-sharded matmul ignores the output shard grid it is given

Not a miscompute, but it silently invalidates a layout assumption. Asking the decode
DRAM-sharded matmul for a 16-core `8x2` rectangular output returns
`{[0-0 - 10-0], [0-1 - 4-1]}` — the row-major prefix of the compute grid, bounding
box `11x2`. The op writes on its own storage-core layout regardless of the
`memory_config` it is handed.

That is *why* the decode norms run under a program grid wider than their shard: every
decode boundary tensor comes out of that matmul, so the prefix is not a choice this
layer makes. It is safe there and only there, because a decode step is one tile row
and `block_h = 1` is the case finding 2 measures correct for every core count — and
`_decode_norm_configs` now raises rather than construct the combination above it.
Prefill norms start from a tensor this layer shards itself, which is why they can
have the rectangle.


## A host upload removed from the measured prefill path

`ttnn.zeros(..., device=...)` is not a device op:
`ttnn::creation_detail::full_impl`
(`ttnn/cpp/ttnn/operations/creation/creation.cpp:51-73`) allocates a host
`std::vector`, fills it, and uploads it. The inherited sliding prefill built its zero
Q filler that way at **every internal chunk boundary**, and at the real shape —
`[1, 32 heads, 2048 window, 128]` BF16 = 16,777,216 B — it showed up as a
**2015.9 μs** op-to-op gap in the warmed two-chunk sliding window, against 33.8 μs
total for the `full` window running the same Python chunk loop without a filler.
16.78 MB / 2.0159 ms = 8.3 GB/s, a host PCIe write rate.

The filler is a constant, so **one** window-length buffer is built lazily and kept:
`32 heads x 2048 window x 128 head_dim x 2 B = 16,777,216 B` per sliding layer,
recorded in the capability contract under
`implementation.extra_persistent_buffers`. A continuation tail shorter than the
window slices that buffer rather than allocating its own — deliberately, because
`tail_len` is `min(window, start_pos)` and therefore caller-controlled: a cache keyed
on it would grow one entry per tile-aligned offset, up to 64 entries and ~545 MB per
layer, and no test in this suite reaches enough distinct offsets to notice.

The worst op-to-op gap in that window is **0.610 μs** and the window's total went
2051 -> **35.7 μs**. At full context a sliding prefill has 15 internal boundaries, so
this was of order 30 ms per layer of pure host upload.

It also names a blind spot in the host-fallback audit:
`test_no_host_fallback_in_forward` patches Python entry points, so a C++-side host
buffer creation passes it silently. The op-to-op gap evidence above is what caught
it, which is why limitation 4 now points at gap evidence rather than at the audit.


## Correctness

129 tests, 129 passed, 244 asserted HF-vs-TTNN PCC checks
(`logs/full_test_run.log`, `test_results.xml`, `logs/pcc_summary.txt`).

### Two PCC bars

| population | checks | bar | worst | role |
| --- | --- | --- | --- | --- |
| released bf16 checkpoint | 38 | **0.995** | **0.995079** | the acceptance gate; the precision policy is selected on it |
| i.i.d.-Gaussian synthetic | 206 | 0.99 | 0.990467 | the inherited harness, on a documented looser bar |

The synthetic harness draws each tensor from a Gaussian with the real tensor's
mean and std, and under BFP4 MLP weights it lands **2.6x further from the HF
reference** than the real checkpoint does. `bench/bfp_block_range_probe.py`
refutes the three obvious mechanisms (`logs/bfp_block_range_probe.log`):

| hypothesis | measurement | verdict |
| --- | --- | --- |
| i.i.d. samples widen the 16-element BFP block's dynamic range | `max\|w\|/mean\|w\|`: real 2.638-2.742, synth 2.631-2.633 | refuted, within 4 % |
| the synthetic weights quantise worse | BFP4 round-trip max rel. error: real 0.82-6.52, synth 0.39-0.80 | refuted, **wrong direction** |
| a BFP4 projection is less accurate on synthetic weights | output PCC vs an FP32-weight matmul: real 0.99296-0.99356, synth 0.99344-0.99390 | refuted, real marginally worse |

So BFP4 represents the two weight sets equally well *per projection*, and the gap
is an interaction inside the layer that this stage did not isolate. `$optimize`
OPT-012 is what makes the response "widen the real-weight coverage until it covers
every disputed condition", not "ship the slower policy": the real-weight surface
runs six prefill lengths x two kinds (including `seq_len=1`, the non-aligned
2049/4097/8193, and multi-chunk 12345), eight consecutive decode steps off the
BFP8 cache, traced replay at batch 8, and a bounded head-to-head against the fused
decoder. The looser synthetic bar is documented, **not** an expected-failure
marker, and the two slower fallbacks are reported below with their numbers.

### Optimized-vs-fused, same reference and inputs

`test_optimized_vs_fused_accuracy`, `seq_len = 4097`:

| population | prefill | decode | bound |
| --- | --- | --- | --- |
| real, sliding | 0.999597 -> 0.997858 (+1.740e-3) | 0.998916 -> 0.996090 (+2.826e-3) | 4.0e-3 |
| real, full | 0.999660 -> 0.997116 (+2.544e-3) | 0.998870 -> 0.995874 (+2.996e-3) | 4.0e-3 |
| synthetic, sliding | 0.999509 -> 0.993613 (+5.896e-3) | 0.998783 -> 0.992348 (+6.435e-3) | 1.0e-2 |
| synthetic, full | 0.999470 -> 0.992155 (+7.316e-3) | 0.998823 -> 0.991684 (+7.139e-3) | 1.0e-2 |

These bounds are diagnostic, not the gate — the gate is the absolute 0.995
real-weight bar. They are the measured worst delta plus ~35 % margin, so a real
drift trips them while re-measuring the same code does not. The 2.5x gap between
the two bounds *is* the OPT-012 argument in one constant.

### The optimized path is asserted, not inferred

PCC cannot tell an optimized dispatch from a fallback — the fused decoder's
`ttnn.linear` would pass every PCC test in this file while giving up 383 GB/s
against 430. So the kernels and dtypes are checked directly:

* `test_decode_uses_dram_sharded_matmuls` — all six decode projections are
  DRAM-sharded program configs on width-sharded L1 activations and width-sharded
  DRAM weights, with the swept `in0_block_w`, `per_core_M == 1`, and the policy's
  weight dtype (`$optimize` OPT-013);
* `test_weight_dtype_policy_reaches_the_tensors` — the *tensors* are the policy,
  and prefill and decode share one copy;
* `test_prefill_uses_the_expected_dense_kernel` — all three prefill kernels at 32,
  128, 2048 and 12345 rows, including the 2D-multicast grid-width rule;
* `test_prefill_mcast_table_is_legal` / `test_decode_geometry_table_is_legal` —
  the geometry tables host-side, including the `per_core_M <= 4` L1 bound.

Independently, the dtype policy is visible in the committed perf tables: every
dominant decode row reads `LoFi BF16 x BFP8 => BF16` or `LoFi BF16 x BFP4 =>
BF16`.

### Everything inherited

Determinism (bit-identical over 3 repeats), the host-fallback audit (clean: no
`from_torch`/`to_torch`/`as_tensor` or 13 torch entry points in a full prefill and
decode), 64-step stress, full 131072 context for both kinds, non-aligned lengths
`{1, 100, 2049, 4097, 8193, 12345, 130073}`, caller-chunked continuation prefill
including a sub-window tail, batch 4/13/32, the non-zero cache slot, and an FP32
HF control all still pass against the optimized path.

**Watcher clean.** `TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0
TT_METAL_WATCHER_NOINLINE=1` over 30 node ids covering every structurally
distinct optimized path: zero `Watcher detected` / tripped / sanitize /
`TT_ASSERT` / `DEBUG_ASSERT` / out-of-bounds / fault / Error lines across 23,720
log lines and 44 dumps (`watcher/watcher.log.gz`, `logs/watcher_run.log`). Run
separately from every profiler capture.


## Performance accounting

Roofline, device time and end-to-end from the same configuration.

| workload | bytes/token | roofline @ ~512 GB/s | device | e2e | roofline fraction |
| --- | --- | --- | --- | --- | --- |
| sliding @ 2048 | 315.9 MB | 0.617 ms | 1.072 ms | 1.0909 ms | 58 % |
| full @ 131071 | 386.1 MB | 0.754 ms | 1.255 ms | 1.2657 ms | 60 % |

`bytes/token` is 314,802,176 B of weights (BFP8 attention + BFP4 MLP) plus the KV
the SDPA actually reads — 1.1 MB inside the 2048-token sliding window, 71.3 MB for
the full cache at 131071.

**e2e minus device is 11–19 μs, at worst 1.7 %.** There is no material host term
left in the decode loop; it is a real traced replay. That subtraction is across two
runs, and deliberately so: the profiler inflates dispatch gaps, so the profiled
decode window reports 56 μs/replay of op-to-op gap — more than the 19 μs the
unprofiled end-to-end measurement leaves room for. The device figure is the sum of
op durations from the Tracy window; the end-to-end figure is wall time from a
profiler-free run.

**The roofline gap is one thing.** Per replay at 2048 (sliding), device time
splits as six matmuls 883.3 μs (82.4 %), `BinaryNg` 58.2 μs, six norms 40.6 μs,
`SdpaDecode` 29.7 μs, and ~60 μs of head creation, RoPE gather, paged update and
reshards. The matmuls average 356 GB/s, but that average hides the whole story:

| rows | dtype | DRAM | of peak | FLOPs % | `tt-perf-report` |
| --- | --- | --- | --- | --- | --- |
| `32 x 6656 x 4608` (wqkv) | BFP8 | 432.2 GB/s | 84.4 % | 41.7 % | ✅ Optimized |
| `32 x 6656 x 4096` (gate) | BFP8 | 429.4 GB/s | 83.9 % | 41.4 % | ✅ Optimized |
| `32 x 4096 x 6656` (o_proj) | BFP8 | 432.0 GB/s | 84.4 % | 41.7 % | ✅ Optimized |
| `32 x 6656 x 19968` (mlp gate) | BFP4 | 289 GB/s | 56.4 % | 55.7 % | SLOW |
| `32 x 6656 x 19968` (mlp up) | BFP4 | 286 GB/s | 55.9 % | 55.2 % | SLOW |
| `32 x 19968 x 6656` (mlp down) | BFP4 | 297 GB/s | 58.0 % | 57.3 % | SLOW |

The BFP8 rows are at the bandwidth limit. The BFP4 rows are not, and their higher
`FLOPs %` says why: at half the bytes per element the same worker cores must
unpack twice as many elements per byte, so the row has crossed from
bandwidth-bound to unpack-bound. **That worker count is not a program-config
field** — `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:240` sets
`num_worker_cores = num_dram_banks`, where `num_dram_banks` comes from
`get_optimal_dram_bank_to_logical_worker_assignment` (`matmul_utilities.cpp:381`),
i.e. from the device's DRAM-reader assignment. Named limitation 2.

Two different counts appear in this report and they are not the same quantity, so
to be explicit: **8** is `mesh.dram_grid_size().x`, the shardable DRAM grid width —
it sets how many shards the weight is split into and is the value the prefill
2D-multicast grid must match. **12** is what the perf rows show for the decode
matmul's compute cores, which comes from the reader assignment above, not from
`dram_grid_size()` and not from anything this layer passes. `num_dram_channels()`
is not exposed through the Python API on this build, so the 12 here is the measured
row plus the source path, not a derived figure.

The fused stage sat at 93 % of *its* roofline only because it moved 3.07x more
bytes; cutting weight traffic necessarily raises the fixed-cost share. In absolute
terms the step is 2.5x faster.


## What `tt-perf-report` advised, and what happened

Advice was left enabled in every committed table.

| advice | rows | action | evidence |
| --- | --- | --- | --- |
| "No output subblock size found" | the 3 BFP4 decode MLP rows | **not actionable** — `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` has no `out_subblock_h/w` fields at all (only `in0_block_w`, `per_core_M`, `per_core_N`, `fused_activation`). Reported as a `tt-perf-report` improvement candidate | — |
| "Use HiFi2 or HiFi4 with BF16 activations for improved accuracy" | every matmul row | **rejected, measured** — HiFi2 is 69 % slower in decode and 35 % slower in prefill for +2.4e-4 PCC | `logs/layer_ab_precision*.log` |
| "If possible place input 0 in L1" | all six 128-row prefill rows | **tried and rejected** — legal (`BLOCK_SHARDED` in0), and 0.92-0.97x, because the activation is ~2 % of the bytes the matmul moves | `logs/short_prefill_layout_probe.log` |
| "No program_config specified, try one to override `in0_block_w` and `out_subblock_h/w`" | the `MinimalMatmul` prefill rows | **false negative** — a `MinimalMatmulConfig` *is* passed on every one of these; the ops CSV does not record it. `tt-perf-report` improvement candidate | `logs/mm_block_sweep_*.log`, ~1400 measurements |
| "High Op-to-Op Gap ... running with tracing could save 9 μs" | `SdpaDecode`, 7 μs | **inapplicable** — the window *is* a traced replay; the advice assumes it is not. `tt-perf-report` improvement candidate | `tracy/*/decode_perf_report.txt` |
| "High Op-to-Op Gap ... could save 29 μs (1.3 %)" (`sliding`) / "36 μs (1.7 %)" (`full`) | the first `LayerNorm` and `wqkv` matmul of the **128-row** prefill window: 24.157 + 17.042 μs on `sliding`, 28.095 + 20.247 μs on `full` | **accepted as disclosed, not fixed here** — 41.2 μs (`sliding`) / 48.3 μs (`full`) on the two ops right after the signpost, of a 61.9 / 68.4 μs window total, i.e. first-dispatch cost that a traced prefill would remove. Prefill tracing belongs to the stage that owns the generator loop (limitation 3); the largest remaining gap in either window is 1.477 μs | `tracy/{sliding,full}/prefill_128_perf_report.txt` |

Three of the five are candidate improvements to `tt-perf-report` itself rather
than to this model.


## Measured and rejected

Every row here was built and measured, not reasoned away.

### Packed same-input projections

| candidate | matmul level | whole layer | verdict |
| --- | --- | --- | --- |
| packed QKV + attention gate (OPT-001) | best packed 0.1274 vs 0.1304 ms split — but only on a 13-core grid the layer cannot use; on the boundary grid 0.1345 vs 0.1326 | 1.1298 / 1.1073 vs 1.1228 / 1.0961 | **rejected, 0.6 % slower** |
| packed MLP gate + up (OPT-010) | 0.4851 vs 0.4600 ms for two separate dispatches, *before* the split | 1.1517 / 1.1248 vs 1.1228 / 1.0961 | **rejected, 2.6 % slower** |

Both fail by the same mechanism, and it is the one OPT-010 names: the doubled
output width forces `in0_block_w` down (from 13 to 2 for QKV+gate; capped at 2 at
every legal core count for gate+up), and on this part `in0_block_w` is the field
that matters most.

### Precision candidates (real checkpoint, whole layer)

All five candidates from one run on the **released checkpoint**
(`bench/layer_ab.py --candidates fused,b16_all_bfp8,gateup_bfp4,mlp_bfp4,all_bfp4
--real-weights`, `logs/layer_ab_real_final.log`; every row carries the `AB[real]`
prefix, so no synthetic number is mixed in):

| policy | decode ms/token (sliding / full) | prefill 8192 | prefill PCC | decode PCC | verdict |
| --- | --- | --- | --- | --- | --- |
| fused baseline (BF16) | 2.7346 / 2.7092 | 63.84 / 64.12 | 0.999566 / 0.999607 | 0.999601 / 0.999447 | baseline |
| all BFP8 | 1.2653 / 1.2348 | 45.83 / 44.86 | 0.999239 / 0.999349 | 0.999000 / 0.998798 | 16 % slower than shipped |
| gate/up BFP4, down BFP8 | 1.1487 / 1.1180 | 44.72 / 43.32 | 0.998328 / 0.998082 | 0.998172 / 0.997624 | 5.3 % slower than shipped |
| **attn BFP8, MLP BFP4 (shipped)** | **1.0908 / 1.0601** | **43.68 / 43.10** | **0.997536 / 0.997197** | **0.997157 / 0.996804** | **kept** |
| all BFP4 incl. attention (OPT-007) | 1.0572 / 1.0262 | 43.91 / 42.15 | 0.977175 / 0.979843 | 0.984938 / 0.977697 | **rejected: 0.977 on real weights** |

The frontier is monotone and the shipped point is the fastest that clears the bar.
The BFP4-attention candidate is OPT-007's mandatory attention-weight trial, run on
**real** weights and rejected on measured model-visible accuracy — 0.977/0.980
prefill PCC is a 2 %+ loss, not a margin question.

### The sharded prefill RMSNorm — rejected, then shipped

The first write-up of this stage rejected it, and the rejection was wrong for an
instructive reason: the probe that produced the constraint had hard-coded the shard
core set to a row-major prefix of the 11-wide device grid, so a 16-core shard was
always built as `11 + 5` under an `11x2` program grid — the one shape that
corrupts. `core_range_set(16, ...)` on an `8x10` grid is an exact `8x2` rectangle,
16 divides the 208 hidden-size tiles, and the intersection the report called empty
was not.

Re-measured with exact rectangles (`logs/sharded_norm_grid_probe_rect.log`), norm
plus both conversions:

| rows | `block_h` | 16 c (`8x2`) | 8 c (`8x1`) | interleaved | result |
| --- | --- | --- | --- | --- | --- |
| 32 | 1 | correct | correct | — | `max\|diff\| = 0.03182` |
| 128 | 4 | **33.0 μs** | 44.1 μs | 135.8 μs | **4.1x**, `max\|diff\| = 0.04022` |
| 256 | 8 | **57.6 μs** | L1 | ~136 μs | 2.4x, `max\|diff\| = 0.03786` |
| 512 | 16 | CB overflow | L1 | 135.9 μs | no legal point |

So it ships for `rows <= 256` on 16 cores
(`PREFILL_NORM_SHARD_MAX_ROWS`/`PREFILL_NORM_SHARD_CORES`), and whole-layer that is
worth 2.57 -> **2.18 ms** at 128 tokens and 2.48 ms at 256 — the difference between
1.37x and **1.60x** against the fused decoder. Above 256 rows the shard stops
fitting L1 *and* the interleaved norm has enough tile rows to fill the grid anyway,
so the band boundary is not a compromise.

Two guards, because the failure mode is silent rather than loud:
`test_prefill_norm_is_sharded_and_rectangular` asserts the program grid covers
exactly the shard, and `test_decode_norm_refuses_the_silently_corrupting_shape`
asserts the decode path raises rather than build the unsafe combination.

### Folding the activations into the matmul

`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` has a `fused_activation`
field and the factory honours non-RELU activations
(`..._dram_sharded_program_factory.cpp:349`), so SiLU can ride on the gate matmul
and sigmoid on the attention-gate matmul, leaving plain multiplies behind. Built
and measured, whole-layer traced decode:

Both candidates in one run, `bench/layer_ab.py --candidates mlp_bfp4,fused_act`
(`logs/layer_ab_fused_activation.log`):

| candidate | sliding | full |
| --- | --- | --- |
| **activation on the `ttnn.mul` (shipped)** | **1.0908** | **1.0602** |
| `fused_act` — folded into the matmul | 1.1393 (+4.4 %) | 1.1082 (+4.5 %) |

Output PCC is identical to six decimals either way in the same run (0.993759 prefill
/ 0.993506 decode for both rows), so this is purely scheduling: the matmul's `SFPU_ACTIVATION` runs on its
**12** worker cores — fixed to the DRAM bank count — interleaved with the unpack it
is already bottlenecked on, while a separate `ttnn.mul` gets the MLP's 26-core or
the boundary's 16-core shard. Fewer, larger ops is the usual direction; here the op
it would merge into has the fewest cores. Kept as a one-flag comparison
(`DECODE_FUSED_ACTIVATION`).

The related question — the two activation-folded multiplies cost more in *absolute*
terms than on the fused stage's 110 DRAM-interleaved cores (SwiGLU 14.23 -> 40.47 μs,
attention gate 5.96 -> 14.28 μs, per-replay means from the two stages' committed
decode windows) — has the same answer from the other side
(`logs/decode_elementwise_probe.log`):

| candidate | SwiGLU, SiLU on `a` (26 c) | attention gate, sigmoid on `b` (16 c) |
| --- | --- | --- |
| **folded multiply (shipped)** | **41.72 μs** | **29.21 μs** |
| plain multiply, activation removed | 29.80 μs | 28.77 μs |
| separate unary then multiply | 49.29 μs | 45.83 μs |
| reshard to 52 c, multiply, reshard back | 79.11 μs | — |
| reshard to 104 c, multiply, reshard back | 109.09 μs | — |

Each case uses the unary and the operand the layer actually folds — SiLU on `a` for
SwiGLU, sigmoid on **`b`** for the attention gate — checked against a matching
float64 reference. Timing is host wall clock around untraced dispatches, so all rows
at one shape carry the same launch floor and only the differences between them are
meaningful; the attention-gate rows sit near 29 μs where that op's committed device
row is 14 μs.

On that basis: the transcendental is not the cost. Removing it moves the SwiGLU row
by ~12 μs and the attention-gate row by **0.44 μs**; the rest is the multiply itself
on a narrow shard. And widening the shard is far worse than it looks worth, because
resharding a 19968-wide tensor twice costs more than the multiply saves. The
elementwise ops are at their floor given the layout contract, and the +34 μs they
cost is bought back many times over by the 2.5x the same contract buys the matmuls.

### Left alone, with the exact contract that blocks each

* **`paged_fused_update_cache`** — needs K and V on disjoint cores;
  `nlp_create_qkv_heads_decode` emits them on Q's grid, and its
  `overlap_qk_coregrid=False` mode needs a QKV shard width dividing `head_dim`.
  This stage's QKV output is width-sharded at 4608/16 = 288 elements, which does
  not divide 128; reaching 128 needs 36 output cores and 36 does not divide the
  208 K-tiles the matmul's activation shard requires. The two `paged_update_cache`
  dispatches are 7.17 μs of a 1072 μs step (0.67 %).
* **BFP8 activations** — blocked by an exact op contract:
  `nlp_create_qkv_heads_decode` accepts FLOAT32 or BFLOAT16 only
  (`nlp_create_qkv_heads_decode_device_operation.cpp:41`).
* **The prefill SDPA chunk (256)** — swept by the fusing stage over 90
  measurements across 9 lengths and both kinds; nothing this stage does changes
  that constraint set.
* **The two per-head QK norms on one core** — `ttnn.rms_norm` rejects
  height-sharded inputs (`layernorm_device_operation.cpp:166`); 0.7 % of a step.


## Artifacts

```bash
D=models/autoports/meta_models_muse_glimmer_30b/doc/optimized_decoder
# correctness (the acceptance gate)
python -m pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_optimized_decoder.py -q
# device-time profiles, advice enabled, one job at a time, no watcher
bash $D/bench/run_tracy.sh
# watcher, in a separate run
bash $D/bench/run_watcher.sh
# the capability contract, regenerated from the committed junit + suite log
python $D/bench/refresh_context_contract.py --check
python $D/bench/summarize_pcc.py --check
# the mechanically-sourced figures in README.md and context_contract.json,
# re-derived from the CSVs/junit/logs
python $D/bench/check_reported_figures.py --check
```

That last one exists because four consecutive review rounds of this stage found the
same defect class and nothing else: a number in a report that no longer matched the
CSV it came from. `refresh_context_contract.py` regenerates `tests.*` and the PCC
blocks but not the `performance` block or any prose, and `summarize_pcc.py` only
covers PCC — so the figures it now guards had no mechanical guard at all.

It covers `README.md` and `context_contract.json`: every per-window device time and
op count, the decode op-group breakdown, the worst prefill op-to-op gap, the test
and PCC-population counts, both `fused_activation` A/B rows, the precision frontier
against `logs/layer_ab_real_final.log`, the per-role DRAM/FLOPs columns, and the
watcher node-id count. It does **not** parse `work_log.md`, which is chronological
and deliberately keeps superseded snapshots; those are labelled as superseded
rather than checked.

`tracy/{sliding,full}/` holds ten `*_perf_report.txt` tables, `*_perf_report.csv`,
and the ops CSVs (gzipped where they exceed the repo's 500 KB hook limit) for
prefill at 128 / 8192 / 16384 tokens and traced decode at 2048 / 131071, for both
layer kinds. All ten captures are free of dropped profiler markers, and all ten were
re-taken against the final code rather than carried over from an earlier candidate.

| probe | question it answers |
| --- | --- |
| `bench/weight_layout_probe.py` | which weight layouts each matmul accepts |
| `bench/decode_matmul_sweep.py` | dtype x fidelity x (cores, `in0_block_w`) per decode projection |
| `bench/minimal_matmul_block_sweep.py` | `MinimalMatmulConfig` per shape, dtype and row count |
| `bench/prefill_mcast_probe.py` | 2D-multicast vs `minimal_matmul` per row count; `--repro` for TTNN bug 1 |
| `bench/short_prefill_layout_probe.py` | the two open `tt-perf-report` advice items at 128 rows |
| `bench/sharded_norm_grid_probe.py` | TTNN finding 2, whether the shipped decode norm is affected, and (`--rect`) the exact-rectangle band the prefill norm ships on |
| `bench/decode_elementwise_probe.py` | where the decode SwiGLU / attention-gate multiplies spend their time |
| `bench/check_reported_figures.py` | re-derives the mechanically-sourced figures in `README.md` and `context_contract.json` from the committed CSVs/junit/logs |
| `bench/bfp_block_range_probe.py` | why synthetic and real weights disagree under BFP4 |
| `bench/layer_ab.py` | whole-layer candidate ranking: precision, geometry, SDPA, packing |
| `bench/variants.py` | the rejected packed-projection and old-SDPA layers |


## Capability contract

Unchanged: 131072 tokens, batch to 32, arbitrary non-aligned logical lengths. The
optimized layer is 3.07x smaller in weights (967.8 -> 314.8 MB) and 1.88x smaller
in KV cache (134.2 -> 71.3 MB per layer at full context), so nothing is closer to
a limit than it was. `../context_contract.json` carries the byte budget, every
measured PCC, and the performance block, and
`bench/refresh_context_contract.py --check` fails if any of it goes stale against
the committed junit and suite log.


## The `$optimize` checklist

Items that do not apply to a single-chip decoder layer (multi-device topology
families, collectives, fused CCL+matmul, persistent CCL buffers, MoE
`sparse_matmul`, LM head and sampling, vLLM serving, `$qualitative-check`) are
marked **n/a** with the reason; this model is a dense decoder on a 1x1 mesh with no
LM head in scope.

| item | status | evidence |
| --- | --- | --- |
| Decoder path fully traced, no host fallbacks | ✅ | `test_no_host_fallback_in_forward`; e2e within 11-19 μs of device time |
| Decode activations width-sharded in L1 across norm / attention / residual / MLP / output | ✅ | `test_decode_layout_contract`; 16-core boundary + 26-core MLP grid |
| Prefill activations DRAM interleaved; 2D matmul program configs for large prefill matmuls | ✅ | interleaved throughout; 2D-multicast is the *measured* winner to 1024 rows and `minimal_matmul`'s M/K/N-blocked 110-core kernel beats an 8-column 2D-multicast above it |
| Operation-topology audit recorded | ✅ | [work_log](work_log.md) §2, both phases, with action per row |
| Multi-device topology families measured as coherent families | **n/a** | single chip, 1x1 mesh; no collectives in the graph |
| Lower-movement residual candidates measured without an old-contract restore | **n/a** | no collectives; the residual never leaves L1 in decode |
| Best-candidate comparison against the strongest prior correct baseline | ✅ | [precision candidates](#precision-candidates-real-checkpoint-whole-layer), packing table, geometry tables; fused decoder re-measured in the same harness |
| Final default reproduced the selected best candidate | ✅ | `logs/layer_ab_final_{2048,131071}.log`, re-run after every knob was frozen |
| Final dtype/fidelity policy verified in measured runtime rows | ✅ | every dominant row in `tracy/*/`*`_perf_report.txt` reads `LoFi BF16 x BFP8` or `LoFi BF16 x BFP4` (OPT-013) |
| SDPA / optimized composite ops instead of hand-built attention | ✅ | `paged_scaled_dot_product_attention_decode`, `SDPAOperation`, `nlp_create_qkv_heads*`, `rotary_embedding_hf`, `paged_fill_cache`/`paged_update_cache` |
| Fused or packed repeated same-input projections, or measured evidence to keep them separate | ✅ | both packed candidates built and measured slower ([table](#packed-same-input-projections)) |
| Explicit `memory_config`, `program_config`, `compute_kernel_config` on important ops | ✅ | all six decode matmuls, all prefill matmuls, all sharded norms (decode *and* short prefill), decode SDPA |
| Elementwise / activation placement measured, not assumed | ✅ | `logs/decode_elementwise_probe.log` plus the `fused_activation` whole-layer A/B; five candidate placements, shipped one wins |
| Runtime data movement: host terms driven out of the measured path | ✅ | the 16.78 MB per-chunk host upload removed; worst prefill op-to-op gap 2015.9 -> 0.610 μs |
| Per-role program-config sweep for dominant matmuls, incl. larger `in0_block_w` | ✅ | `logs/decode_matmul_geometry_bfp{4,8}*.log.gz`; `mlp_down` swept to `in0_block_w=24`, `wqkv`/`attn_gate` to 13 (26 is illegal at BFP8) |
| Decode compute fidelity swept as a perf knob, not assumed from dtype | ✅ | LoFi vs HiFi2 on the identical dtype policy: HiFi2 69 % slower in decode |
| Attention projection dtype/fidelity swept separately from MLP | ✅ | OPT-007 BFP4-attention trial on real weights: 0.977/0.980 prefill PCC, rejected |
| BFP4/LoFi trials for MLP gate/up **and** down | ✅ | both shipped at BFP4; the down-BFP8 fallback is reported with its numbers |
| Shard specs / core grids divide tensor dims cleanly, grids as large as legal | ✅ | 16 divides 208/144/128 tiles exactly; 26 chosen for the MLP because 13 fails L1 in the layer |
| DRAM-sharded decode matmuls | ✅ | all six, `✅ Optimized` on the three BFP8 rows |
| Collective topology minimised | **n/a** | none |
| Fused matmul-CCL ops | **n/a** | none |
| Persistent/preallocated decode CCL buffers | **n/a** | none |
| MoE routed active-expert path | **n/a** | dense MLP, not MoE |
| LM head / sampling terminal path | **n/a** | this stage owns one decoder layer; no LM head or sampler |
| LM head DRAM-sharded | **n/a** | as above |
| Reduced precision/fidelity experiments on real weights and activations | ✅ | five-policy frontier on the released checkpoint |
| Performance accounting reconciled (roofline, device, e2e) | ✅ | [Performance accounting](#performance-accounting); gaps named, not hand-waved |
| Batch capability preserved; larger batch tested to 32 | ✅ | `test_batched_prefill_decode_pcc[4/13/32]`, ragged per-user lengths and positions |
| Functional checks still pass; PCC at the bar; paged KV + traced replay correct | ✅ | 129/129, worst real-weight 0.995079 |
| Stress / repeated-run coverage | ✅ | 64-step soak on both kinds in the suite; both kinds' soak and both kinds' 256-row sharded-norm prefill also run under watcher |
| Row-count-dependent branches asserted at their boundaries | ✅ | `test_prefill_pcc_across_the_norm_shard_band` (224/256/288/320, both kinds, both weight populations); `test_prefill_mcast_table_is_legal` bounds `per_core_M` at each band's worst case |
| Watcher clean, separate from profiler runs | ✅ | 30 node ids, zero detections |
| `tt-perf-report` with advice enabled, applicable advice tried | ✅ | ten tables; [advice ledger](#what-tt-perf-report-advised-and-what-happened) |


## Limitations and known issues

1. **The real-weight PCC margin is thin.** The worst of the 38 real-checkpoint
   checks is 0.995079 against the 0.995 bar — 7.9e-5 of headroom, on
   `decode[sliding] step=6 pos=3006`. It is not flaky (the layer is asserted
   bit-deterministic and the reference is fixed, so the number reproduces
   exactly), but it means BFP4 MLP weights have spent essentially the whole
   per-layer accuracy budget this stage was given. The next-slower policy —
   gate/up BFP4 with **down** at BFP8 — restores ~2.7e-3 of margin for 5.3 % of
   decode, and is the first thing to reconsider if a stacked model needs headroom.
   `$optimize` OPT-012 is why it was not chosen here: the faster policy passes the
   real-weight gate, and a policy is not rejected for passing narrowly.
2. **BFP4 decode matmuls reach 56-58 % of peak DRAM, against 84 % for BFP8.**
   They are unpack-bound on a worker set whose size is fixed to the DRAM bank
   count by `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:240`
   (`num_worker_cores = num_dram_banks`) and cannot be widened from the program
   config. They are 63 % of the decode step, so this is the single largest
   remaining decode lever and it needs a TTNN change, not a model change.
3. **Prefill is not traced**, so it keeps a host gap (38.0 ms device against
   44.2 ms e2e at 8192 tokens). This stage owns one decoder layer, not a generator
   loop; tracing prefill belongs to the stage that owns the loop.
4. **Untraced prefill host gaps are now bounded but not zero.** The one large,
   removable term — a 16.78 MB host upload per internal sliding chunk boundary — is
   gone (see [above](#a-host-upload-removed-from-the-measured-prefill-path)); the
   two-chunk sliding window's total op-to-op gap is 35.7 μs, worst single gap 0.610 μs.
   What remains is the 6.5 ms between device time and wall time on an 8192-token
   prefill, which is untraced dispatch and Python chunk-loop overhead. Note that the
   host-fallback audit cannot see host-side buffer creation inside a C++ op, so
   op-to-op gap evidence, not that audit, is what bounds this.
5. **The synthetic-weight PCC gap is measured but not explained.** BFP4 costs the
   i.i.d.-Gaussian harness 2.6x more than the real checkpoint; three candidate
   mechanisms are refuted with data and the real cause is an interaction inside the
   layer that this stage did not isolate.
6. **`minimal_matmul` is not bit-identical to `ttnn.linear`**, and this stage now
   dispatches both — the 2D-multicast rows carry ~1.1e-4 less PCC than the
   `minimal_matmul` rows at the same row count. Inherited from the fused stage;
   all PCC evidence here is against the HF reference, and nothing claims
   bit-equality with any earlier stage.
7. **Two silent-miscompute shapes are one refactor away** — a 2D-multicast grid
   wider than the DRAM bank count, and a sharded-norm program grid wider than its
   shard above one tile row. Both are pinned by tests
   (`test_prefill_mcast_table_is_legal`,
   `test_decode_norm_refuses_the_silently_corrupting_shape`) rather than by
   convention, because neither raises on its own. They are TTNN bugs, not model
   bugs, and the repros are committed.
