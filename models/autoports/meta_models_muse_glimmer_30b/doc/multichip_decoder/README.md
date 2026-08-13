# Multichip decoder — `meta-models/Muse-Glimmer-30B`

A tensor-parallel decoder layer for the four Blackhole dies of this host's
`P300_X2` mesh, built on the single-chip [optimized decoder](../optimized_decoder/README.md).
Same public contract, same paged semantics, same 131072-token capability,
**2.39–2.49x traced decode** and **2.30–2.33x prefill** end to end (2.40–2.50x
and 2.04–2.07x of device time), with the per-device weight footprint down 3.88x
and the per-device KV cache down 2x.

| item | value |
| --- | --- |
| mesh open | `l1_small_size=6144` and an 8192 B fabric packet payload — both load-bearing, see [TTNN behaviour 1](#1-ccl-semaphores-fragment-l1-unless-the-mesh-has-an-l1_small-region) and [Fabric packet size](#fabric-packet-size) |
| implementation | `models/autoports/meta_models_muse_glimmer_30b/tt/multichip_decoder.py` |
| tests | `tests/test_multichip_decoder.py` + `tests/test_multichip_vs_single_chip.py` |
| baseline | `models/autoports/meta_models_muse_glimmer_30b/tt/optimized_decoder.py`, re-measured on this host in the same harness |
| device | 4 x Blackhole, `ClusterType::P300_X2` (2 x P300 boards), `ttnn.MeshShape(1, 4)`, `FabricConfig.FABRIC_1D_RING`, `ttnn.Topology.Ring`, 2 links |
| parallelism | TP=4 over all four dies; column-parallel QKV/gate/MLP-in, row-parallel `o_proj`/`mlp_down`, replicated residual, 2 reductions per layer |
| precision | inherited `attn-bfp8-mlp-bfp4-kv-bfp8-lofi`, plus a **BFP8 prefill collective payload** |
| per-device weights | 314.8 MB -> **81.0 MB** (3.88x) |
| per-device KV cache | 71.3 MB -> **35.7 MB** per layer at 131072 (2x) |
| capability | unchanged: 131072 tokens, batch to 32, non-aligned lengths — [`../context_contract.json`](../context_contract.json) |

## Result

Warmed, `bench/layer_ab.py`, min of 3 rounds, no profiler attached; `single` is
the `OptimizedDecoder` on a 1x1 mesh measured by the same harness on the same
host (`logs/layer_ab_single_baseline.log`), and the multichip row is the **final
shipped code** re-measured after the last change (`logs/layer_ab_final.log`).

| window | 1 chip | 4 chips | speedup | efficiency |
| --- | --- | --- | --- | --- |
| traced decode, sliding @2048 | 1.0908 | **0.4572 ms/token** | **2.39x** | 60 % |
| traced decode, full @2048 | 1.0602 | **0.4260 ms/token** | **2.49x** | 62 % |
| prefill 8192, sliding | 44.12 | **19.14 ms** | **2.30x** | 58 % |
| prefill 8192, full | 43.59 | **18.68 ms** | **2.33x** | 58 % |

Traced decode is reproducible to four decimals across rounds and runs (three
rounds inside a run agree to 1e-4, and the five A/B runs of the shipped config
span 0.4572-0.4590 on `sliding` across the configuration changes). Warmed prefill is noisier: repeat runs of the *same* build
span about 1.6-3.5 % (18.72 / 19.08 / 19.14 / 19.34 / 19.39 ms on `sliding`), so the prefill
speedups above are quoted from the final run rather than from the best one.

Device time from the committed Tracy tables (`tracy/`), decode divided by its 8
trace replays, against the optimized stage's own committed tables:

| window | 1 chip | 4 chips | speedup |
| --- | --- | --- | --- |
| decode sliding @2048 | 1072 μs | **441.5 μs** | 2.43x |
| decode full @2048 | 1049 | **419.0** | 2.50x |
| decode sliding @131071 | 1071 | **440.5** | 2.43x |
| decode full @131071 | 1255 | **522.7** | 2.40x |
| prefill 128, sliding / full | 2140 / 2096 | **839.8 / 814.6** | 2.55x / 2.57x |
| prefill 8192, sliding / full | 37762 / 36606 | **18212 / 17925** | 2.07x / 2.04x |

Efficiency is 60–62 % of linear on decode and 58 % on prefill, and the reason is
one number in each regime: **the part of a decoder layer that does not shrink
with tensor parallelism.** Section [Performance accounting](#performance-accounting)
takes both apart op by op.

## The plan

### Mesh

Four dies, every die with exactly two Ethernet neighbours (`{2:4}` degree
histogram from auto-discovery) — a ring. Two mesh views open on it, and the
choice is not cosmetic: `get_boundary_mode` (`ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:98-123`)
demotes `Ring` to `Linear` for **any** mesh axis of extent 2, so on a `2x2` view
every collective in this layer is a line. Measured at the decode payload
(`logs/ccl_tuning_probe.log`), a line reduce-scatter is 39.88 μs against 33.55
for the ring; independently, the repo's own bandwidth *gate* for this SKU
(`tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py:37-72`,
a different op) sets the line floor at half the ring floor. So: **`1x4` + `FABRIC_1D_RING`**, set
before `open_mesh_device` — without it every CCL op fails at
`control_plane.cpp:2222`, which is setup evidence, not hardware evidence.

### Tensors

Per layer, per device, at the shipped precision policy. `column` = fractured
over the output dim, `row` = over the input dim.

| tensor | full `[K, N]` | scheme | per device | tiles | padding |
| --- | --- | --- | --- | --- | --- |
| `wqkv` | 6656 x 4608 | column | 6656 x **1280** | 40 wide | KV heads replicated 2x |
| `attn_gate` | 6656 x 4096 | column | 6656 x 1024 | 32 | none |
| `o_proj` | 4096 x 6656 | row | 1024 x 6656 | K=32 | none |
| `mlp_gate` | 6656 x 19968 | column | 6656 x **5120** | 160 | 4992 -> 5120 |
| `mlp_up` | 6656 x 19968 | column | 6656 x **5120** | 160 | 4992 -> 5120 |
| `mlp_down` | 19968 x 6656 | row | **5120** x 6656 | K=160 | 4992 -> 5120 |
| 4 x RMSNorm | 6656 | replicated | 6656 | 208 | none |
| K/V cache | blocks x 2 x 64 x 128 | by KV head | blocks x **1** x 64 x 128 | — | KV replicated 2x |
| page table, positions | batch x blocks | replicated | identical | — | none |
| RoPE tables | 131072 x 128 | replicated | identical | — | none |
| sliding prefill Q filler | 32 x 2048 x 128 | by head | **8** x 2048 x 128 | — | 16.8 -> 4.2 MB |

Per-device weight bytes: 9,052,160 (`wqkv`) + 7,241,728 (`attn_gate`) +
7,241,728 (`o_proj`) + 3 x 19,169,280 (MLP) = **81,043,456 B**, against
314,802,176 on one chip. That is 3.88x, not 4x, and the 3.0 % difference is
exactly the two padded rows: the duplicated K/V columns (3.6 MB across the mesh)
and the MLP zero padding (5.8 MB).

**KV replication.** The model has 32 query heads and `num_key_value_heads = 2`,
so 2 KV heads cannot be split four ways. Each device takes 8 contiguous query
heads and the one KV head those heads read under GQA (group size 16): devices
0,1 -> KV head 0, devices 2,3 -> KV head 1. This is the pattern
`models/demos/gemma4/tt/attention/weights.py:55-95` uses; the assignment rule is
`kv_head_of_device(d) = (d * local_heads) * n_kv // n_heads`, asserted as
`[0,0,1,1]` by `test_plan_matches_the_hardware` and checked against the actual
weight bytes on each device by `test_qkv_weight_is_gqa_assigned`.

The consequence is that per-device KV cache **halves** rather than quarters. The
alternative — splitting the KV *sequence* across each device pair and merging the
two SDPA partials with their log-sum-exps — would quarter it, and is rejected
here as scope: it needs a partial-softmax merge TTNN does not expose for the
paged decode op, and the SDPA is 4.7 % of the decode step at 2048 and 23.0 % at
131071 (`tracy/full/decode_131071_perf_report.txt`), so the ceiling on it is
smaller than the risk. Recorded, not hidden.

**MLP intermediate padding.** 19968/4 = 4992 = 156 tiles is not a multiple of the
8 DRAM banks a width-sharded weight shards over (156/8 = 19.5), and 156 shares
only {1,2,4} with the 208 hidden-size tiles. Padding each device's slice to 5120
(160 tiles) with zeros costs 2.6 % of the MLP weight bytes and buys an exact
one-shard-per-bank weight *and* a single 16-core L1 grid for the whole decode
step — the single-chip layer needed a second 26-core grid and two reshards per
token. `silu(0) * 0 = 0`, and `test_mlp_padding_is_inert` asserts the padded
columns and rows really are zero on every device.

### The one grid

Every width-sharded L1 tensor in the decode step lives on **16 cores**. The
candidate set is bounded by a hard rule rather than by taste: a matmul's *input*
shard must not be padded, because the padded columns would be summed into the
reduction. So the count has to divide every width that is an `in0` — 208
(hidden), 32 (the gated attention output, for `o_proj`) and 160 (the padded MLP
intermediate, for `mlp_down`) — which admits only {1, 2, 4, 8, 16}. 16 is the
largest, and it is the measured winner (`logs/layer_ab_geometry_final.log`):

| grid | sliding | full |
| --- | --- | --- |
| **16 everywhere (shipped)** | **0.4669** | **0.4360** |
| 8 everywhere | 0.5012 | 0.4702 |
| 8 boundary + 16 MLP, 2 reshards | 0.4885 | 0.4579 |
| 16 boundary + 8 MLP, 2 reshards | 0.4866 | 0.4562 |
| 4 everywhere | L1 clash | L1 clash |

The one width 16 does not divide is the 40-tile QKV projection *output* (2.5
tiles per core, written as 3). It is never an `in0` — it goes straight through
`sharded_to_interleaved` into `nlp_create_qkv_heads_decode` — so the padding is
inert, and `test_qkv_output_shard_is_padded_not_wrong` pins that it stays on an
output. The 7 % the wider grid buys the rest of the step is worth more than 20 %
of one projection's output tiles.

`in0_block_w` per role is then the largest legal value, and legality moved with
every per-device K:

| role | K | K-tiles/core | legal | shipped | cost of the next one down |
| --- | --- | --- | --- | --- | --- |
| `wqkv`, `attn_gate` | 6656 | 13 | 1, 13 | **13** | 53 % (`qkv_bw1`) |
| `o_proj` | **1024** | 2 | 1, 2 | **2** | 2.4 % (`oproj_bw1`) |
| `mlp_gate`, `mlp_up` | 6656 | 13 | 1, 13 | **13** | 39 % (`gu_bw1`) |
| `mlp_down` | **5120** | 10 | 1, 2, 5, 10 | **10** | 0.9 % / 4.4 % (`down_bw5` / `down_bw2`) |

### Decode SDPA cores per (batch, head)

A knob that only exists once the KV heads are fractured. `SdpaDecode` gives each
(batch, KV head) pair `max_cores_per_head_batch` cores and reduces across them in
a binary tree bounded to 6 rounds (`sdpa_decode_program_factory.cpp:239-245`).
With **one** local KV head per device the default 16 leaves most of the grid
idle. Measured (`logs/layer_ab_sdpa_131071.log`, `logs/layer_ab_sdpa_2048.log`):

| cap | full @131071 | full @2048 | sliding @131071 | sliding @2048 |
| --- | --- | --- | --- | --- |
| 16 (the default) | 0.6077 | — | 0.4640 | — |
| **32 (shipped)** | **0.5384** | **0.4352** | 0.4650 | **0.4663** |
| 64 | 0.5408 | 0.4359 | 0.4658 | 0.4670 |
| fixed q=32/k=64 | 0.5627 | — | 0.4675 | — |

Leaving the default alone would have cost 12 % of the long-context decode step.

## Collectives

Both row-parallel projections produce a full-width partial sum. The layer keeps
the **residual stream replicated** and reduces once per sublayer — two reductions
per layer, no conversion at the layer boundary, so a stacked model hands one
layer's output straight to the next.

### The contract families, measured through the next consuming op

`bench/topology_probe.py` runs four boundary contracts as complete chains —
`row-parallel matmul -> collective(s) -> RMSNorm -> residual add -> the next
column-parallel matmul` — each ending on the contract it started on, so a winner
is stackable. Not "reduce-scatter followed by an immediate all-gather back to the
old contract", which would measure the wrong thing.

| candidate | CCL bytes/device | 32 rows | 8192 rows |
| --- | --- | --- | --- |
| `replicated` (all-reduce) | 1.50 S | 317.24 μs | 8598.18 μs |
| `replicated_bfp8` | 0.75 S | 312.87 | 7406.71 |
| `fractured` (reduce-scatter residual) | 1.50 S | **284.30** | 7356.10 |
| `gather_heads` (column `o_proj`) | 1.21 S | 345.08 | 7156.12 |
| `gather_heads_fractured` | 1.21 S | 312.79 | **6153.76** |

`S = rows * 6656 * 2 B`. Two things this table says plainly:

1. **A fractured residual saves no bytes on a ring.** An all-reduce already *is* a
   reduce-scatter plus an all-gather. What it changes is the width the norm and
   the residual add run at, and the number of dispatches.
2. **The fractured family wins this probe**, which is measured in the
   DRAM-interleaved regime — i.e. the prefill regime.

Both are why the decision needed decode-regime evidence rather than this table
alone. The probe's regime *is* the prefill regime, and in it the ranking survives
tuning: re-run with the shipped reducer on **both** arms
(`logs/topology_probe_decode32_tuned.log`), `fractured` is 282.81 μs against
`replicated`'s 315.36 — the same 10 % it showed before the collective tuning.

The decode regime is different in exactly one way, and it is the way that
decides: its RMSNorms are already **width-sharded in L1**. So the three terms the
fractured contract trades were measured at the shipped memory configs, traced
(`bench/fractured_decode_probe.py`, `logs/fractured_decode_probe.log`; every row
is a one-op trace, so they share a replay floor and only the differences are
meaningful):

| term | replicated pays | fractured pays |
| --- | --- | --- |
| hidden-size RMSNorm | 15.46 μs (6656 on 16 cores) | 10.83 / 12.88 / 15.55 / 19.24 μs (1664 on 4 / 13 / 26 / 52) |
| residual add | 8.44 μs | 8.08–8.44 μs — latency-bound, no saving |
| distributed-norm stats all-gather | — | **10.42 μs** |
| `rms_norm_pre_all_gather` | — | **27.01 μs** |

The best case for the fractured contract is 15.46 − 10.83 = **4.63 μs** saved per
norm, against **10.42 μs** for the stats gather it must add — a net loss of
5.8 μs per distributed norm before the `pre_all_gather` and before the extra
all-gather of the residual. Two of the four hidden norms would be distributed, so
the contract loses ≥11.6 μs per decode step, on a 444 μs step. The interleaved
probe reaches the opposite answer because an interleaved full-width norm is the
expensive thing there, and this layer's decode norms are not interleaved.

(An earlier version of this section rejected the contract on an L1 argument — that
a fractured residual forces a ≤4-core grid — and that argument was **wrong**: the
matmul inputs are the *gathered* tensors and stay on 16 cores, while the fractured
tensors can sit on any count dividing their 52 tiles. All four counts are in the
table above; none of them fails L1. The rejection stands on the measurements, not
on that.)

`gather_heads` is a genuine trade rather than a blocker, and it is rejected on
measured whole-layer grounds: `o_proj`'s parallelism is a single load-time choice
shared by prefill and decode, so buying prefill's 17 % costs decode 8.8 % (345.08
against 317.24 at 32 rows) and adds a third collective to every decode step. A
decoder layer is judged on decode, so the row-parallel `o_proj` stays and the
prefill payload dtype below recovers most of the prefill gap instead.

The one place the fractured family is **not** refuted is prefill, where the
regime is exactly the probe's and there is no L1 grid to satisfy. That is
[limitation 1](#limitations-and-known-issues), with its measured size.

### Payload dtype, split by mode

`ttnn.all_reduce` and `ttnn.reduce_scatter` accept BFLOAT8_B and up-cast
internally for the reduction, and asking the row-parallel matmul for that dtype
directly costs **no extra op** — no typecast either side, and the residual add
that consumes the reduced tensor takes its dtype from the BF16 residual, so the
layer's output contract is unchanged (`test_layer_output_dtype_is_the_activation_dtype`).

| mode | BF16 payload | BFP8 payload | real-weight PCC cost |
| --- | --- | --- | --- |
| prefill 8192 (sliding / full) | 21.16 / 20.88 ms | **18.82 / 18.35 ms** | 1.1e-4 / 1.2e-4 |
| traced decode (sliding / full) | 0.4661 / 0.4351 | 0.4585 / 0.4275 | 0.8e-4 / 1.8e-4 |

Prefill takes it; decode does not — and the decode half was settled by running
the candidate, not by arguing from the prefill number. `$optimize` OPT-012
forbids rejecting a faster reduced-precision candidate on synthetic evidence, so
the BFP8 decode payload was measured twice on the **released checkpoint**:

| evidence | worst real-weight check | against |
| --- | --- | --- |
| `logs/real_weight_ccl_dtype_gate.log` — 8-step decode off a real 3000-token prefill, both kinds | 0.995354 | 0.995440 for BF16 |
| `logs/real_weight_decode_bfp8_experiment.log` — the suite's *whole* real-weight surface with the payload flipped | **0.9950028** on `decode[sliding] step=6 pos=3006` | 0.995105 for BF16 |

So it **passes**, by 2.8e-6, where the shipped BF16 payload passes by 1.05e-4. It
spends 97 % of the layer's remaining accuracy budget to buy 1.6 % of the decode
step. This layer is a *stacking* baseline — the full model composes 52 of them,
and `test_two_layers_stack` measures how the error composes (0.9936 per layer
becomes 0.972 for two) — and a margin of three parts per million is not a margin
the next stage can build on. Prefill is the opposite case on every count: ~2.1e-3
of headroom, 1.2e-4 of cost, 11 % of the window.

### Reducer form and worker count

A ring all-reduce is a reduce-scatter plus an all-gather, so both forms move
identical bytes and the only question is one fused dispatch against two
(`logs/layer_ab_ccl_mode.log`, `logs/layer_ab_ccl_workers.log`):

| reducer | traced decode | prefill 8192 |
| --- | --- | --- |
| `all_reduce` both modes | 0.4661 / 0.4354 | 18.79 / 18.86 ms |
| `rs_ag` both modes | 0.4617 / 0.4311 | 21.03 / 20.59 ms |
| **`rs_ag` decode, `all_reduce` prefill (shipped)** | **0.4589 / 0.4284** | **18.72 / 18.43 ms** |

(All three rows are from the same pair of A/B runs, so they compare like with
like; the shipped row's *final* re-measurement after the last code change is the
0.4589 / 0.4283 and 19.39 / 19.13 ms in [Result](#result).)

And the single largest non-matmul win in the stage is one integer,
`num_workers_per_link=1` on the decode reduce-scatter
(`logs/ccl_tuning_probe.log`, traced, at the real decode shape):

| candidate | DRAM | L1 sharded |
| --- | --- | --- |
| default | 33.55 μs | 33.78 μs |
| **`num_workers_per_link=1`** | **20.93** | **21.19** |
| `num_workers_per_link=2` / `4` | 34.30 / 34.55 | 34.59 / 34.62 |
| `num_links=1` | 27.82 | 28.11 |
| `Topology.Linear` | 39.88 | 40.09 |
| `ttnn.all_reduce` | 46.38 | 50.97 |
| `ttnn.all_gather` (not tunable) | 16.39 | 17.22 |

`chunks_per_sync` (2/5/10/20) and `num_buffers_per_channel` (2/4/8) move it under
1 %; `use_l1_small_for_semaphores=True` cannot allocate on this part. At 40 KB
the collective is pure fixed cost — 2.7 GB/s against the 90–120 GB/s the same
fabric reaches at the 8192-row payload — so extra workers only add setup and sync. Whole-layer:
0.4618/0.4311 at the default against **0.4589/0.4284** at one worker, in the same
run.

## Performance accounting

Per-device weight traffic is 81,043,456 B, so at this part's ~512 GB/s the decode
floor is 158 μs against 441.5 μs of device time — 36 %, where the single-chip
layer sat at 58 % of its own (larger) roofline. The whole difference is the part
of the layer that does not shrink with TP, and the committed profile says exactly
how much (`tracy/sliding/decode_2048_perf_report.csv`, per replay):

| group | μs | share | scales with TP? |
| --- | --- | --- | --- |
| 6 matmuls | 254.6 | 57.7 % | yes (weights/4) |
| 2 reductions (`ReduceScatter` + `AllGather`) | 53.2 | 12.0 % | no — new cost |
| 4 hidden-size RMSNorms | 33.7 | 7.6 % | **no** — replicated residual |
| 4 elementwise (`BinaryNg`) | 26.0 | 5.9 % | partly |
| `SdpaDecode` | 20.9 | 4.7 % | half (KV replicated 2x) |
| 2 per-head QK norms | 7.5 | 1.7 % | yes |
| RoPE gather + tilize + transpose | 22.6 | 5.1 % | yes |
| paged update, head create/concat, resharding | ~23 | 5.2 % | mostly |

Prefill is the same story with a different balance
(`tracy/sliding/prefill_8192_perf_report.csv`, per 8192-token chunk):

| group | μs | share |
| --- | --- | --- |
| 6 matmuls | 5751 | 31.6 % |
| 6 RMSNorms | 3460.4 | 19.0 % |
| prefill SDPA | 3070.9 | 16.9 % |
| 2 reductions | 3447.8 | 18.9 % |
| 4 elementwise | 1959.8 | 10.8 % |

The **norms are the largest non-matmul term in both regimes and they do not
shrink at all**, because a replicated residual means all four devices compute the
same 6656-wide normalisation. That is the price of the residual contract, it is
paid once per norm per device, and it is what [limitation 1](#limitations-and-known-issues)
is about.

`tt-perf-report`'s per-row verdicts on the decode matmuls, with the same shape of
result as the single-chip stage — the BFP8 rows are at the bandwidth limit, the
BFP4 rows are unpack-bound:

| row | dtype | DRAM | of peak | verdict |
| --- | --- | --- | --- | --- |
| `32 x 6656 x 1280` (`wqkv`) | BFP8 | 384 GB/s | 75.0 % | ✅ DRAM |
| `32 x 6656 x 1024` (`attn_gate`) | BFP8 | 350 GB/s | 68.4 % | ✅ DRAM |
| `32 x 1024 x 6656` (`o_proj`) | BFP8 | 319 | 62.4 % | SLOW |
| `32 x 6656 x 5120` (gate, up) | BFP4 | 264–268 | ~52 % | SLOW |
| `32 x 5120 x 6656` (`mlp_down`) | BFP4 | 267 | 52.2 % | SLOW |

`o_proj` is the one row that moved against the single-chip layer (84 % -> 62 %):
its per-device K is 1024, which allows only `in0_block_w <= 2`, so it has a
quarter of the K-blocking the other attention rows get. Both alternatives were
measured — `in0_block_w=1` is 2.4 % slower whole-layer, and a wider grid is not
legal because 32 tiles must divide the core count.

## Correctness

Every PCC number in the main test module compares TTNN against the HF reference,
and that comparison carries this model's precision policy as a floor of ~1e-2 —
wide enough to hide a tensor-parallel fault worth 1e-3. So the parallelisation is
pinned by a **separate** comparison, in its own module
`tests/test_multichip_vs_single_chip.py`: one process opens a 1x1 mesh, runs the
single-chip `OptimizedDecoder`, copies its outputs to host and closes it, then
opens the 1x4 mesh and runs the multichip layer on identical weights, inputs,
page table and positions. Prefill and every decode step are asserted at **0.999**,
where the shared precision policy cancels out of both sides. Measured
(`logs/vs_single_chip_run.log`):

| case | prefill | 4 decode steps | worst |
| --- | --- | --- | --- |
| sliding, 2049 tokens, batch 1 | 0.999839 | 0.999851–0.999860 | **0.999839** |
| full, 2049 tokens, batch 1 | 0.999807 | 0.999816–0.999825 | **0.999807** |
| sliding, 12345 tokens, batch 4 | 0.999721 | 0.999843–0.999844 | **0.999721** |
| full, 12345 tokens, batch 4 | 0.999609 | 0.999183–0.999202 | **0.999183** |

The second pair is the one that matters for coverage: 12345 tokens is two
internal prefill chunks (so the per-device single-KV-head sliding tail is carried
across a chunk boundary), batch 4 gives four cache slots and ragged decode
positions, and 12345 is divisible by neither the tile, the page block nor the
chunk. 20 checks, worst 0.999183.

That module exists separately because the obvious construction does not work on
this build: carving a `1x1` **submesh** out of the open `1x4` mesh succeeds, and
work on the submesh succeeds, but every subsequent collective on the *parent*
mesh hangs — and the hang wedges the fabric until `tt-smi -r`, so a later fresh
process hangs in its first `all_reduce` too. The minimal repro is in that
module's docstring and the recovery is in [the work log](work_log.md).

See [the suite log](logs/full_test_run.log), [`test_results.xml`](test_results.xml)
and [`logs/pcc_summary.txt`](logs/pcc_summary.txt) for the full surface:

* both layer kinds throughout;
* nine prefill lengths `{1, 100, 128, 2048, 2049, 4097, 8192, 8193, 12345}` —
  six of them not divisible by the tile size, and eight not equal to the
  8192-token internal chunk, which is what keeps the multichip path from exposing an
  aligned-only public contract;
* caller-chunked continuation prefill including a sub-window sliding tail, at
  splits `(4096, 3000)`, `(1024, 1024)` and `(64, 100)`;
* batch 4, 13 and 32 with ragged per-user lengths and positions, and a non-zero
  cache slot;
* the full 131072 context and the non-aligned 130073, prefill and decode;
* eight-step decode off the paged cache, a 64-step soak, traced replay,
  determinism over three repeats, an FP32 HF control, and the released
  checkpoint at six lengths on both kinds;
* the structural assertions only this stage can make — the mesh plan, the
  per-device weight and cache shapes, the GQA KV assignment, the inert MLP
  padding, the DRAM-sharded decode dispatches, exactly two reductions per
  sublayer on `Topology.Ring`, the padded QKV output, bit-identical replicas, and
  each device's cache holding the KV head its query heads read.

**104 tests, 104 passed, 290 asserted PCC checks** ([`logs/pcc_summary.txt`](logs/pcc_summary.txt)):

| population | checks | bar | worst |
| --- | --- | --- | --- |
| multichip vs single-chip TTNN | 20 | 0.999 | **0.999183** |
| released bf16 checkpoint | 30 | 0.995 | **0.995105** |
| i.i.d.-Gaussian synthetic, single layer | 238 | 0.99 | 0.990516 |
| two chained layers | 2 | 0.96 | 0.967946 |

The two-layer row has its own bar because two layers compose two layers' error: a
single layer is ~0.9936 on this harness and the chain measures 0.972 against the
same HF math, so composing the single-layer bar would assert something
arithmetically false. What that test is for is the layout contract — the tensor
that comes out of layer *n*'s `prefill_forward` is the tensor that goes into
layer *n+1*'s, with the same dtype, memory config and shape, and no conversion.

**Watcher clean, with a teardown fault that is not.** `TT_METAL_WATCHER=10
TT_METAL_WATCHER_APPEND=0 TT_METAL_WATCHER_NOINLINE=1` over 31 node ids covering
every structurally distinct multichip path: **31 passed**, and zero
`Watcher detected` / tripped / sanitize / `TT_ASSERT` / `DEBUG_ASSERT` /
out-of-bounds / fault / Error lines across 28,788 log lines and 50 dumps
([`logs/watcher_run.log`](logs/watcher_run.log), `watcher/watcher.log.gz`). Run
separately from every profiler capture.

The process then **aborts at teardown**, after every test has reported:

```
Device 0: Virtual core 29-25, Port status: 0x1, Retrain count: 0x0, Rx link up: 0x1, ...
TT_THROW: Device 0: Timed out while waiting for active ethernet core 29-25 to
become active again. Try resetting the board.
... Aborted (core dumped)          # pytest exit code 134
```

It is disclosed rather than filed under the grep counts, because it is a device-
health event on this stage's core path and the last version of this document did
not say so. What is known about it, from two occurrences:

* it fires in `RiscFirmwareInitializer::teardown` -> `MetalContext::~MetalContext`,
  i.e. closing the **1x4 `FABRIC_1D_RING` mesh**. No submesh and no second mesh
  are involved, so it is not the 1x1-after-1x4 interaction that
  `test_multichip_vs_single_chip.py` documents;
* it does **not** self-recover: both times, the *next* process to open the mesh
  failed at startup with the same signature, and `tt-smi -r` cleared it both
  times (work log §9.2). The reset is bounded and no second reset was needed;
* it has only ever been seen on the **watcher** path. The same 31 node ids run
  inside the 100-test acceptance suite with no such line, and neither the A/B, the
  eight Tracy captures, nor the comparison module has produced one.

`bench/run_watcher.sh` now captures and prints the pytest exit code instead of
letting `set -e` swallow it, and `bench/run_evidence_chain.sh` records it and
continues, so the artifact is reproducible from the committed scripts. **The
full-model stage should expect this**: it opens and closes this mesh in every
process, and if it hits the same teardown timeout it needs the same bounded
reset.

## Two TTNN behaviours this layer has to live with

Both were found by running the layer, both are pinned by a test rather than a
comment, and neither is a model bug.

### 1. CCL semaphores fragment L1 unless the mesh has an `L1_SMALL` region

Every collective creates a global semaphore, and
`all_gather_multicast_factory.cpp:36-43` allocates it in `L1_SMALL` **only if the
mesh has one** — otherwise in the main L1 pool, with the warning *"Allocating
semaphores in L1, which may fragment L1 and reduce headroom for subsequent op
allocations."* The semaphore belongs to the cached program, so it is never freed
and it sits at the top of L1. Measured with `ttnn.get_memory_view`:

| point | allocated/bank | largest contiguous free/bank |
| --- | --- | --- |
| fresh mesh | 0 | 1,461,376 |
| after a 256-row prefill | 384 | 1,460,992 |
| after one decode | 704 | **1,434,048** |

27 KB of the top of L1 disappears after one decode, and the *next* 256-row
prefill — whose sharded RMSNorm wants two 213 KB L1 tensors — then fails with
*"Statically allocated circular buffers in program 20 clash with L1 buffers"*.
So `open_multichip_mesh` opens with `l1_small_size = 6144` **and** the decode
reduce-scatter passes `use_l1_small_for_semaphores=True` (only `all_gather` reads
the bank size on its own). With both, the residue is a flat 256 B/bank and the
largest contiguous block never moves, and latency is unchanged.

The size is bounded on both sides and the ceiling is the decode step's own
budget: its circular buffers end at 1,137,536 B and its live L1 tensors take
316,544 B from the top, so it uses 1,454,080 of the 1,461,376 B pool and has
**7,296 B** to give away. The whole ladder is measured, not just the first value
that worked:

| bytes | CCL programs | result |
| --- | --- | --- |
| 32768 | 128 | the *first* 256-row prefill fails — the inherited sharded prefill norm wants two 213 KB L1 tensors and sits at that edge |
| 8192 | 32 | 896 B over the decode budget: *"L1 buffer allocated at 1136640 and static circular buffer region ends at 1137536"* |
| 7168 | 28 | passes, 128 B from the ceiling |
| **6144** | **24** | **ships**: passes, 1,152 B of margin |
| 4096 | 16 | passes; the region itself becomes the constraint |
| 2048 | 8 | the region fills mid-suite |

**How many CCL programs that is, and what frees them.** Measured directly: six
distinct `reduce_scatter` shapes take 1,536 B of the region — **256 B each** — and
`mesh.clear_program_cache()` returns all of it. So a mesh holds **24 distinct CCL
programs** at a time, and the 25th gets *"Out of Memory: Not enough space to
allocate 1760 B L1_SMALL buffer across 110 banks"*.

That is not a constraint this layer can hit: a stacked model dispatches two CCL
shapes per layer kind for decode and one per prefill chunk size, and every layer
of the same kind reuses the same program. It *is* a constraint on anything that
sweeps shapes — the correctness suite bounds itself with a fixture that clears the
cache when the region runs low, and the batched test draws its 32 ragged per-user
lengths from 8 distinct values rather than 32 for the same reason. **The
full-model stage should know this number**: a serving process that prefills
arbitrary prompt lengths without padding them into a bounded set of chunk shapes
will exhaust the region and has to clear the program cache periodically.

### 2. The DRAM-sharded matmul picks its own output core count

40 QKV tiles over 16 cores is 2.5, so `per_core_N` rounds to 3 — and the op
writes on its own storage-core layout rather than the requested grid, so the
output comes back on `ceil(40/3) = **14**` cores of 3 tiles. 42 tiles for 40. The
layer is unaffected (the tensor's logical width is still 1280 and the next op is
`sharded_to_interleaved`), but the *contract* is not what `memory_config` says,
so `test_qkv_output_shard_is_padded_not_wrong` asserts the real one.

## Measured and rejected

| candidate | verdict | evidence |
| --- | --- | --- |
| `2x2` mesh view | rejected: every collective demoted to `Linear`, 19 % slower reduce-scatter | `logs/ccl_tuning_probe.log`, `ccl_common.cpp:98-123` |
| fractured (reduce-scatter) residual, decode | rejected: needs a <=4-core L1 grid, which fails L1; and 2 extra stats gathers (~27 μs) against ~14 μs saved | `logs/layer_ab_geometry_final.log`, `logs/topology_probe_decode32.log`, `tracy/sliding/decode_2048_perf_report.csv` |
| `gather_heads` (column-parallel `o_proj`) | rejected: −8.8 % decode for +17 % prefill, and `o_proj`'s layout is one choice for both | `logs/topology_probe_*.log` |
| BFP8 **decode** collective payload | rejected: **passes** the 0.995 real-weight bar by 2.8e-6 (against 1.05e-4 for BF16) to buy 1.6 % of decode | `logs/real_weight_ccl_dtype_gate.log`, `logs/real_weight_decode_bfp8_experiment.log` |
| `rs_ag` reducer for prefill | rejected: 12 % slower | `logs/layer_ab_ccl_workers.log` |
| `all_reduce` reducer for decode | rejected: 0.9 % slower | `logs/layer_ab_ccl_mode.log` |
| 8-core and 4-core decode grids | rejected: 7.3 % slower / fails L1 | `logs/layer_ab_geometry_final.log` |
| separate MLP working grid (the single-chip shape) | rejected: 4.4–4.8 % slower | `logs/layer_ab_geometry_mlpgrid.log` |
| folding the activations into the matmul | rejected: 2.0–2.6 % slower, as on one chip | `logs/layer_ab_geometry_final.log` |
| `max_cores_per_head_batch` 16 / 64 | rejected: 12 % slower at long context / 0.4 % slower | `logs/layer_ab_sdpa_*.log` |
| `chunks_per_sync`, `num_buffers_per_channel`, `num_links` | rejected: under 1 %, or slower | `logs/ccl_tuning_probe.log` |
| KV sequence-split across the device pair | not attempted: needs a partial-softmax merge the paged decode op does not expose; ceiling is the SDPA's 4.7 % / 23.0 % | — |

## The `$multichip` checklist

| item | status | evidence |
| --- | --- | --- |
| single-chip baseline named and re-measured | ✅ | `OptimizedDecoder`, `logs/layer_ab_single_baseline.log` |
| target mesh, topology and strategy with calculated evidence | ✅ | [The plan](#the-plan) |
| tensor/config/shard table with per-device shapes and padding | ✅ | [Tensors](#tensors), [The one grid](#the-one-grid) |
| context contract updated for the mesh | ✅ | [`../context_contract.json`](../context_contract.json) |
| non-aligned logical lengths still work | ✅ | nine prefill lengths, 130073, continuation splits |
| multi-chip prefill and decode PCC vs the single-chip TTNN baseline | ✅ | `test_multichip_matches_single_chip`: both kinds x {2049 batch 1, 12345 batch 4}, worst 0.999183 at a 0.999 bar |
| stacked-decoder input/output contract | ✅ | `test_two_layers_stack`: layer *n*'s output is layer *n+1*'s input, same dtype/memory config/shape, no conversion |
| paged KV behaviour on the target mesh | ✅ | `test_kv_cache_holds_the_expected_head`, batched non-zero slot, full context |
| warmed trace replay for decode | ✅ | `test_traced_decode_pcc`, all decode perf windows |
| determinism / stress | ✅ | 3 repeats bit-identical, 64-step soak |
| runtime fallback audit clean | ✅ | `test_no_host_fallback_in_forward` |
| watcher clean, separate from profiling | ✅ | 31 nodes, 0 detections, `logs/watcher_run.log` |
| hardware recovery recorded | ✅ | [work log §9.2](work_log.md), `triage/` |
| baseline and multichip latency, speedup, efficiency | ✅ | [Result](#result) |
| `tt-perf-report` tables with the communication/DRAM/compute findings | ✅ | `tracy/`, [Performance accounting](#performance-accounting) |
| row-parallel boundary topology table | ✅ | [The contract families](#the-contract-families-measured-through-the-next-consuming-op) |
| lower-movement residual contracts measured through the next consuming op | ✅ | `bench/topology_probe.py` |
| reusable modules considered | ✅ | see [below](#reusable-modules) |
| MoE routed active-expert path | **n/a** | dense SwiGLU MLP; `config.json` has no expert fields |
| 2D mesh plan | **n/a** | 4 dies; the 2D view is measured and rejected above |
| expert replication / EP axis | **n/a** | as above |

### Reusable modules

`models/common/modules/attention/attention_1d.py` and `mlp_1d.py` implement the
same 1D TP contract this layer needs, and both were read before this path was
written. Neither can be used directly, and the mismatches are exact rather than
vague:

* `attention_1d.py:1674-1685` raises when `n_kv_heads % num_devices != 0`, which
  is this model at 2 KV heads on 4 devices. The GQA-assigned replication that
  fixes it is in `models/demos/gemma4/tt/attention/weights.py:55-95`, and that is
  the pattern reproduced here;
* neither module has the model's per-head **attention output gate** (a second
  6656 x 4096 projection multiplied into the SDPA output), its scale-less per-head
  QK norms, its four-norm sandwich, or its sliding/full layer kinds;
* the fused `all_gather_matmul_async` path in `attention_1d.py:1272-1296` is
  gated on 8 devices and Ring, so it does not engage at TP=4 on this part.

What was reproduced from them is the contract, not the code: local head
ownership, GQA-assigned KV replication, setup-time weight fracturing through mesh
mappers, column-then-row parallel matmul pairs, one reduction per sublayer, and a
residual layout that is a first-class decision rather than a leftover.

## Limitations and known issues

1. **The four hidden-size RMSNorms do not shrink with TP**, and they are 7.6 % of
   the decode step and **19.0 % of the prefill layer**. A replicated residual
   means every device computes the same 6656-wide normalisation. The fractured
   residual that would quarter it is refuted for decode with two independent
   pieces of evidence (above), but **not** for prefill, where the regime is
   DRAM-interleaved and `bench/topology_probe.py` measures the fractured family
   1.2–1.4x faster on the boundary chain at 8192 rows. Making prefill fractured
   while decode stays replicated is expressible — the two modes are separate call
   paths — and is the single largest remaining prefill lever, worth an estimated
   11 % of the prefill layer. It is not taken here because it introduces a second
   residual contract for the full-model stage to carry, and that is a decision
   the stage that owns the layer stack should make with its own numbers.
2. **`o_proj` runs at 62 % of peak DRAM against 76 % for the other attention
   rows.** Its per-device K is 1024 = 2 tiles per core, so `in0_block_w` cannot
   exceed 2. Both legal values were measured; a wider grid is not legal.
3. **The BFP4 MLP rows are unpack-bound at 52 % of peak**, inherited unchanged
   from the single-chip stage along with its cause
   (`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:240` fixes the
   worker count to the DRAM bank count). They are 45 % of the decode step, so
   this is still the largest single lever and it still needs a TTNN change.
4. **Per-device KV cache halves rather than quarters**, because the model has 2 KV
   heads and the mesh has 4 devices. This is the model's GQA ratio, not a mesh
   limit, and the sequence-split alternative is recorded above.
5. **Prefill is not traced**, so it keeps a host gap: 18.21 ms device against
   19.14 ms end-to-end at 8192 tokens (4.9 %). Inherited; tracing prefill belongs
   to the stage that owns the generator loop.
6. **The real-weight PCC margin is inherited and thin.** This stage adds no
   measurable loss of its own — multichip-vs-single-chip PCC is asserted at 0.999
   — and its own worst real-weight check, 0.995105, is marginally *better* than
   the single-chip stage's 0.995079. But 1.05e-4 of headroom against a 0.995 bar
   is what rules out the BFP8 decode collective payload, which measures 0.9950028
   on the same surface.
7. **A mesh holds 24 distinct CCL programs at a time** (6 KB of `L1_SMALL`, 256 B
   per program, released only by `clear_program_cache()`). A stacked model needs
   two per layer kind for decode and one per prefill chunk size, so it cannot
   reach that; a process that prefills arbitrary lengths without bounding them
   into a set of chunk shapes can, and the full-model stage should know the
   number. See [TTNN behaviour 1](#1-ccl-semaphores-fragment-l1-unless-the-mesh-has-an-l1_small-region).

## Artifacts

```bash
D=models/autoports/meta_models_muse_glimmer_30b/doc/multichip_decoder
# correctness (the acceptance gate) -- two pytest invocations, see the note below
bash $D/bench/run_suites.sh
# every device job the evidence needs, in order, one at a time:
# the shipped A/B, the single-chip baseline, the eight Tracy windows, the watcher
# run, and the BFP8-decode-payload gate
bash $D/bench/run_evidence_chain.sh
# individually, if you want one of them:
bash $D/bench/run_tracy.sh          # profiles, no watcher in this run
bash $D/bench/run_watcher.sh        # watcher, no profiler in this run
python $D/bench/layer_ab.py --mesh 1x4 --candidates tp4
python $D/bench/layer_ab.py --mesh 1x1 --candidates single
# the capability contract and the PCC summary, regenerated from the committed junit
python $D/bench/refresh_context_contract.py --check
python $D/bench/summarize_pcc.py --check
# every mechanically-sourced number in README.md and context_contract.json,
# re-derived from the committed CSVs/logs
python $D/bench/check_reported_figures.py
```

| probe | question it answers |
| --- | --- |
| `bench/topology_probe.py` | which residual/collective contract wins, measured through the next consuming op |
| `bench/ccl_tuning_probe.py` | op variant, topology, worker count and sync hyperparameters at the decode payload |
| `bench/ccl_probe.py` | the fabric's bandwidth curve from 32 to 8192 rows |
| `bench/layer_ab.py` | whole-layer candidate ranking: grid, `in0_block_w`, SDPA cap, CCL dtype/mode/workers, and the single-chip baseline |
| `bench/perf_windows.py` | the signposted Tracy windows |
| `bench/smoke.py` | first-run bring-up: plan, per-device shapes, PCC, replica identity |
| `bench/ccl_dtype_gate.py` | the BFP8 decode payload against the 0.995 real-weight bar it would have to clear |
| `bench/check_reported_figures.py` | re-derives every mechanically-sourced figure in this README and the contract from the committed CSVs and logs |
