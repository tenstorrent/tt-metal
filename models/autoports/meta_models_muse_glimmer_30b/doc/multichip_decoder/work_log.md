# Multichip decoder — work log

Chronological. Superseded snapshots are labelled rather than deleted; the
authoritative numbers are in [`README.md`](README.md), which quotes only the
final logs.

Model: `meta-models/Muse-Glimmer-30B`. Stage input: the completed single-chip
[optimized decoder](../optimized_decoder/README.md)
(`tt/optimized_decoder.py`, 2.51–2.56x traced decode over the fused stage).

## 1. Hardware survey

```
$ /home/ttuser/.tenstorrent-venv/bin/tt-smi -ls --local
4 x Blackhole, board type p300c, PCI 0000:01..04, all MMIO
$ python -c "import ttnn; print(ttnn.GetNumAvailableDevices(), ttnn.GetNumPCIeDevices())"
4 4
```

`tt_metal/llrt/tt_cluster.cpp:195-201` classifies "P300 board, 4 chips" as
`ClusterType::P300_X2` (two P300 boards, four Blackhole dies). Its mesh-graph
descriptor `tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto`
declares `device_topology { dims: [2, 2] }` with `channels { count: 2 }`, i.e. two
Ethernet links per side. In this checkout the control plane actually comes up on
auto-discovery, which reports

```
Fabric | Logical multi-mesh adjacency: ... intra-mesh degree histograms mesh0 {2:4}
```

— every die has exactly two Ethernet neighbours, i.e. the four dies form a
**ring**. Both `ttnn.MeshShape(2, 2)` and `ttnn.MeshShape(1, 4)` open
(`logs/` — probe run inline); the `1x4` view is a Hamiltonian path around that
ring, so its two ends are also physically adjacent and `ttnn.Topology.Ring` is
legal on it.

Compute grid per die is unchanged from the single-chip stage: 11x10 worker cores,
8 DRAM banks — including with the fabric enabled, which is worth checking because
enabling fabric can move the dispatch core axis.

## 2. Mesh shape: 1x4, not 2x2

`get_boundary_mode` (`ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:98-123`) demotes
`Ring` to `Linear` for **any** mesh axis of extent 2, with or without
`cluster_axis`, because on a two-device axis the ordinary and wrap neighbours are
the same device. So on a `2x2` view every collective in this layer — all of which
span all four devices — is a line, not a ring.

That is not a theoretical difference. Measured at the decode payload,
`logs/ccl_tuning_probe.log`:

| topology | `reduce_scatter` [1,1,32,6656] BF16 |
| --- | --- |
| `Ring` (1x4 + `FABRIC_1D_RING`) | **33.55 μs** |
| `Linear` | 39.88 μs |

and at the prefill payload the repo's own bandwidth gate for this exact SKU
(`tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py:37-72`)
sets the qualification floor at 90 GB/s for a ring against 45 GB/s for a line,
with the comment that a lost wraparound path falls to 64–67 GB/s.

So: **`ttnn.MeshShape(1, 4)` with `FABRIC_1D_RING`**, set before
`open_mesh_device`. Without the fabric config every CCL op fails at
`control_plane.cpp:2222` (`fabric_context_ != nullptr`) — setup evidence, not
hardware evidence (`logs/ccl_probe_1x4_ring.log` was first captured that way and
re-run with the fabric set).

## 3. The parallelisation plan, before any code

Model shapes: `hidden_size` 6656, `intermediate_size` 19968,
`num_attention_heads` 32, `num_key_value_heads` **2**, `head_dim` 128, plus the
model's extra per-head attention output gate (6656 -> 4096). Dense MLP, **not**
MoE — so the MoE/expert rows of `$multichip` are n/a here and are marked as such
in the README's checklist rather than silently skipped.

### 3.1 Tensor table

| tensor | full `[K, N]` | scheme | per device | padding |
| --- | --- | --- | --- | --- |
| `wqkv` | 6656 x 4608 | column | 6656 x 1280 | KV heads replicated 2x |
| `attn_gate` | 6656 x 4096 | column | 6656 x 1024 | none |
| `o_proj` | 4096 x 6656 | row | 1024 x 6656 | none |
| `mlp_gate` | 6656 x 19968 | column | 6656 x 5120 | 4992 -> 5120 |
| `mlp_up` | 6656 x 19968 | column | 6656 x 5120 | 4992 -> 5120 |
| `mlp_down` | 19968 x 6656 | row | 5120 x 6656 | 4992 -> 5120 |
| 4 x RMSNorm | 6656 | replicated | 6656 | none |
| K/V cache | blocks x 2 x 64 x 128 | by KV head | blocks x 1 x 64 x 128 | KV replicated 2x |
| page table | batch x blocks | replicated | batch x blocks | none |
| RoPE tables | 131072 x 128 | replicated | 131072 x 128 | none |

Two rows are not a plain division:

**KV replication.** 2 KV heads cannot be split over 4 devices. Each device takes
8 contiguous query heads and the one KV head those heads read under GQA (group
size 32/2 = 16): devices 0,1 -> KV head 0; devices 2,3 -> KV head 1. This is the
`kv_replicated` pattern of `models/demos/gemma4/tt/attention/weights.py:55-95`;
the identical assignment rule is in `models/demos/blackhole/qwen36/tt/tp_common.py:554-564`.
Consequence: the fused QKV weight grows 4608 -> 4x1280 = 5120 columns, and the
per-device KV cache **halves** rather than quarters.

**MLP intermediate padding.** 19968/4 = 4992 = 156 tiles, which is not a multiple
of the 8 DRAM banks a width-sharded weight shards over (156/8 = 19.5). Padding to
5120 (160 tiles) with zero columns costs 2.6 % of the MLP weight bytes and gives
both an exact one-shard-per-bank weight and a core count the hidden size shares.
`silu(0) * 0 = 0` and the matching zero rows of `mlp_down` make it inert.

### 3.2 Collective volume, per layer, before measuring

With `S = rows * 6656 * 2 B` and a 4-device ring (a collective moves `(P-1)/P`
of its payload per device; an all-reduce is a reduce-scatter plus an all-gather,
so twice that):

| contract | attention sublayer | MLP sublayer | total | CCL ops |
| --- | --- | --- | --- | --- |
| replicated residual, all-reduce | 1.50 S | 1.50 S | **3.00 S** | 2 |
| fractured residual, RS + AG | 0.75 + 0.75 S | 0.75 + 0.75 S | 3.00 S | 4 + 2 stats |
| gather heads, column `o_proj` | 0.46 + 0.75 S | 1.50 S | 2.71 S | 3 |
| gather heads + fractured | 0.46 + 0.75 S | 0.75 + 0.75 S | 2.71 S | 4 + 2 stats |

The fractured residual does **not** save bytes on a ring — an all-reduce already
*is* a reduce-scatter plus an all-gather. What it changes is where the norm and
the residual add run (on 1664 elements instead of 6656) and how many dispatches
there are. Only the gather-heads family moves fewer bytes, and only in the
attention sublayer, because the gathered tensor there is 4096 wide rather than
6656.

## 4. Boundary-contract measurements (`bench/topology_probe.py`)

Each candidate runs the whole chain `row-parallel matmul -> collective(s) ->
RMSNorm -> residual add -> the next column-parallel matmul`, and each ends on the
contract it started on, so the comparison is stack-compatible rather than
"reduce-scatter followed by an immediate all-gather back to the old contract".

`logs/topology_probe_decode32.log` (traced, 32 rows) and
`logs/topology_probe_prefill8192.log` (warmed, 8192 rows), **as they read at the
time**. Both logs have since been regenerated at the shipped configuration and
the 8192 column moved by up to 8 % (§14.3, §15.2); the current numbers are in the
README's contract-families table. This table is kept as the capture the decision
below was actually made on.

<!-- superseded: regenerated at the shipped packet and per-payload worker count; see README "The contract families" and §15.2 -->

| candidate | 32 rows | 8192 rows | CCL bytes |
| --- | --- | --- | --- |
| `replicated` | 317.24 μs | 8598.18 μs | 1.50 S |
| `replicated_bfp8` | 312.87 μs | 7406.71 μs | 0.75 S |
| `fractured` | **284.30 μs** | 7356.10 μs | 1.50 S |
| `gather_heads` | 345.08 μs | 7156.12 μs | 1.21 S |
| `gather_heads_fractured` | 312.79 μs | **6153.76 μs** | 1.21 S |

Read literally, the fractured family wins in this probe. It is measured in the
**DRAM-interleaved** regime, which is exactly the prefill regime and is *not* the
decode regime — the shipped decode step keeps every activation width-sharded in
L1 and dispatches DRAM-sharded matmuls. Two things follow, and both are decided
against the fractured contract by evidence collected later:

1. ~~**It cannot be expressed on the decode L1 grid.**~~ **Withdrawn** — this
   argument was wrong and is retracted in §7.4, §14.1 and the README. It claimed
   a fractured residual forces a ≤4-core grid that fails L1; in fact the matmul
   inputs are the *gathered* tensors and stay on 16 cores, and all four core
   counts dividing the fractured width fit. (A real 4-core constraint does exist,
   but it is on the *distributed norm*, not the matmuls, and it was not found
   until §14.1.)
2. ~~**The extra dispatches cost more than the saved work.**~~ **Withdrawn** —
   the arithmetic here mixed a floor-inclusive collective cost with a
   floor-cancelling norm difference. §14.1 replaces it with a floor-calibrated,
   path-vs-path measurement: 8.11 μs for the shipped full-width norm against
   14.90 μs for the distributed one it would force, i.e. +13.57 μs per decode
   step. The rejection stands; this reasoning for it does not.

The gather-heads family is a real trade and is rejected on a measured
whole-layer basis, not on bytes: `o_proj`'s parallelism is a single load-time
choice shared by prefill and decode, so buying prefill costs decode, and it adds
a third collective to the decode step. (The percentages this paragraph originally
carried came from the superseded capture above; at the shipped configuration it
is +13.8 % prefill for −9.1 % decode — see the README's rejected-candidates
table.)
Decode is the metric a stacked decoder layer is judged on, so the row-parallel
`o_proj` stays and the prefill payload dtype (section 7) recovers most of the
prefill gap instead.

## 5. Bring-up

`bench/smoke.py` — first run passed on both layer kinds without a debugging
detour (`logs/smoke_v1.log`, `logs/smoke_v1_full.log`):

```
PLAN tp=4 local_heads=8 local_kv=1 kv_replicated=True local_qkv_width=1280 local_intermediate=5120
PREFILL[sliding] seq_len=2049 0.9936738334644831 -> PASS      (single chip: 0.993759 at 8192)
PREFILL replica device=1/2/3 bit-identical=True
DECODE[sliding] pos=2049 0.9918319710852616 -> PASS
```

The implementation is a subclass of `OptimizedDecoder`: `from_state_dict` is
replaced (mesh mappers, local head counts, padded MLP), and the two row-parallel
projections gain their collective inside `_decode_projection` /
`_prefill_projection`. Everything else — the chunked prefill loop, the
sliding-window tail hand-off, paged fill/update, RoPE, the head ops, the sharded
prefill norm — is inherited unchanged, because putting the **local** head counts
into `MuseGlimmerLayerConfig` makes every inherited forward compute the
per-device shape without a second head-count concept in the runtime path.

## 6. Decode geometry sweep

All in `logs/layer_ab_geometry_final.log` unless noted; traced decode ms/token,
min of 3 rounds, sliding / full, one knob changed per row against `tp4`.

The core-count candidate set is bounded by a hard rule: a matmul's **input**
shard must not be padded, so the count has to divide every width that is an
`in0` — 208 (hidden), 32 (gated attention) and 160 (padded MLP intermediate).
That admits only {1, 2, 4, 8, 16}. Rows from `logs/layer_ab_geometry_final.log`
unless the note says otherwise:

| candidate | sliding | full | note |
| --- | --- | --- | --- |
| **`tp4` (16 cores everywhere)** | **0.4669** | **0.4360** | shipped grid |
| `grid8` | 0.5012 | 0.4702 | 7.3 % slower |
| `grid8_mlp16` | 0.4885 | 0.4579 | separate MLP grid + 2 reshards (`logs/layer_ab_geometry_mlpgrid.log`, `logs/layer_ab_geometry2.log`, `logs/layer_ab_geometry3.log`) |
| `grid16_mlp8` | 0.4866 | 0.4562 | |
| `grid4` | L1 clash | L1 clash | CB overflow, program.cpp:1779 |
| `qkv_bw1` | 0.7158 | 0.6845 | `in0_block_w` 13 -> 1 |
| `gu_bw1` | 0.6467 | 0.6154 | |
| `down_bw5` | 0.4708 | 0.4400 | 10 -> 5 |
| `down_bw2` | 0.4885 | 0.4576 | |
| `oproj_bw1` | 0.4781 | 0.4473 | 2 -> 1 |
| `fused_act` | 0.4763 | 0.4450 | activations folded into the matmul; loses, as on one chip |

16 cores wins despite being the one grid that leaves the 40-tile QKV projection
output padded (2.5 tiles per core, written as 3). That output is never a matmul
`in0` — it goes through `sharded_to_interleaved` into
`nlp_create_qkv_heads_decode` — so the padding is inert, and the 7 % the wider
grid buys the rest of the step is worth more than 20 % of one projection's
output tiles.

### 6.1 SDPA cores per (batch, head)

New knob on this stage. `SdpaDecode` gives each (batch, KV head) pair
`max_cores_per_head_batch` cores and reduces across them in a binary tree bounded
to 6 rounds (`sdpa_decode_program_factory.cpp:239-245`). With **one** local KV
head the default 16 leaves most of the grid idle, and 64 is still legal.
`logs/layer_ab_sdpa_131071.log` and `logs/layer_ab_sdpa_2048.log`:

| cap | full @131071 | full @2048 | sliding @131071 | sliding @2048 |
| --- | --- | --- | --- | --- |
| 16 | 0.6077 | — | 0.4640 | — |
| **32** | **0.5384** | **0.4352** | 0.4650 | **0.4663** |
| 64 | 0.5408 | 0.4359 | 0.4658 | 0.4670 |
| fixed q=32/k=64, cap 64 | 0.5627 | — | 0.4675 | — |
| 8x8 grid, cap 64 | 0.5503 | — | 0.4639 | — |

32 is best or tied in three of four cells and the 12 % gap at cap 16 on the full
layer is the load-bearing part: it is the *default*, so leaving it alone would
have cost 12 % of the long-context decode step.

## 7. Collective tuning

### 7.1 Payload dtype, split by mode

`ttnn.all_reduce` and `ttnn.reduce_scatter` take BFLOAT8_B and up-cast internally
for the reduction, and asking the row-parallel matmul for that dtype directly
costs **no extra op**. Measured on the released checkpoint
(`logs/layer_ab_real_ccl.log`) and the synthetic harness
(`logs/layer_ab_geometry_final.log`):

| mode | BF16 payload | BFP8 payload | real-weight PCC cost |
| --- | --- | --- | --- |
| prefill 8192 | 21.16 / 20.88 ms | **18.82 / 18.35 ms** | 1.1e-4 / 1.2e-4 |
| traced decode | 0.4661 / 0.4351 | 0.4585 / 0.4275 | 0.8e-4 / 1.8e-4 |

Prefill takes it; decode does not, and the decode half was settled by running the
candidate on the released checkpoint twice rather than by arguing from the
prefill number ($optimize OPT-012):

| evidence | worst real-weight check | BF16 for comparison |
| --- | --- | --- |
| `logs/real_weight_ccl_dtype_gate.log`, 8-step decode off a real 3000-token prefill, both kinds | 0.995354 | 0.995440 |
| `logs/real_weight_decode_bfp8_experiment.log`, the suite's whole real-weight surface with the payload flipped (`MG_MULTICHIP_DECODE_CCL_DTYPE=bfloat8_b`, reproducible from committed code) | **0.9950028** (`decode[sliding] step=6 pos=3006`) | 0.995105 (`logs/full_test_run.log`) |

It passes, by 2.8e-6, where BF16 passes by 1.05e-4 — 97 % of the layer's
remaining accuracy budget for 1.6 % of the decode step, on a layer whose whole
purpose is to be stacked 52 times and whose two-layer chain already measures
0.972. Rejected on that trade, with both numbers, not on a synthetic result.

### 7.2 Reducer form, split by mode

A ring all-reduce is a reduce-scatter plus an all-gather, so the two forms move
identical bytes and the question is one *host* dispatch against two — on device
both forms run the same pair, since `ttnn.all_reduce` decomposes into
`reduce_scatter_minimal_async` + `all_gather_async`
(`ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.cpp`), which is why the two
measure so nearly alike at a bandwidth-bound payload.

**This table is superseded.** Its prefill column was taken before the worker
count was split per payload, so the `rs_ag` row measures a prefill payload at the
decode-tuned single worker; §14.2 re-measures that directly as an 11.4 % / 12.5 %
worker-count effect, and the reducer form itself is worth 0.24 %. The current
reducer table is in the README (`logs/layer_ab_reducer_final.log`).

<!-- superseded: prefill column predates DEFAULT_PREFILL_CCL_RS_WORKERS=4; see 14.2 and README "Reducer form and worker count" -->

| reducer | traced decode | prefill 8192 |
| --- | --- | --- |
| `all_reduce` both modes | 0.4661 / 0.4354 | **18.79 / 18.86 ms** |
| `rs_ag` both modes | 0.4617 / 0.4311 | 21.03 / 20.59 ms |
| **shipped: `rs_ag` decode, `all_reduce` prefill** | **0.4589 / 0.4284** | **18.72 / 18.43 ms** |

### 7.3 `num_workers_per_link`

The largest non-matmul win in the stage, and it is one integer.
`logs/ccl_tuning_probe.log`, traced, at the real decode shape, both memory
configs the layer could hand it:

| candidate | DRAM | L1 sharded |
| --- | --- | --- |
| `reduce_scatter` default | 33.55 μs | 33.78 μs |
| **`num_workers_per_link=1`** | **20.93** | **21.19** |
| `num_workers_per_link=2` | 34.30 | 34.59 |
| `num_workers_per_link=4` | 34.55 | 34.62 |
| `num_links=1` | 27.82 | 28.11 |
| `Topology.Linear` | 39.88 | 40.09 |
| `ttnn.all_reduce` | 46.38 | 50.97 |
| `ttnn.all_gather` (not tunable) | 16.39 | 17.22 |

`chunks_per_sync` (2/5/10/20) and `num_buffers_per_channel` (2/4/8) move it by
under 1 %. The `use_l1_small_for_semaphores=True` row of that sweep reads *"Out
of Memory: Not enough space to allocate 1760 B L1_SMALL buffer across 110
banks"*, which the first version of this entry recorded as the flag being
unusable on this part. **That was wrong** (a round-1 finding, propagated here
only in round 3): the probe reaches that case as roughly its 19th distinct CCL
program and had exhausted its *own* `L1_SMALL` region — see §9.1b. The shipped
decode reduce-scatter passes the flag, and it works.

At 40 KB the collective is pure fixed cost — 2.7 GB/s against the **120.6 GB/s**
the same fabric reaches on the 8192-row BF16 all-gather at the shipped packet
size (`logs/fabric_packet_probe.log`, 81,788,928 B in 678.41 µs) — so extra
workers only add setup and sync.

Whole-layer effect (`logs/layer_ab_ccl_workers.log`): 0.4618/0.4311 at the
default against **0.4589/0.4284** at one worker.

## 7.4 Two knobs the first sweep missed, and one wrong rejection

Found by the stage review and settled by measurement
(`logs/fabric_packet_probe.log`, `logs/fractured_decode_probe.log`,
`logs/topology_probe_decode32_tuned.log`).

**`num_workers_per_link` is per payload.** One worker wins the 40 KB decode
collective by 39 % and *loses* the 107 MB prefill one by 2.4x: on the shipped
BFP8 prefill payload `reduce_scatter` is 1814.9 μs at one worker and **759.9 μs**
at four. The shipped knob is now split (`DEFAULT_DECODE_CCL_RS_WORKERS = 1`,
`DEFAULT_PREFILL_CCL_RS_WORKERS = 4`). It also corrects a rejection: the prefill
reducer was recorded as "`rs_ag` 12 % slower", which was the decode-tuned worker
count applied to a prefill payload. At four workers the pair is 759.9 + 810.2 =
1570.1 μs against `all_reduce`'s 1563.7 — **the same to 0.4 %** — so prefill keeps
the fused op on dispatch count, not on a margin.

**Fabric packet size.** Every CCL dispatch logs *"Fabric packet size 4352 B is
suboptimal for transporting 2048 B pages. Configure 8192 B"*
(`ccl_common.cpp:39-71`), and the first sweep never tried it. The advice is about
the **page** size, so it flips with the payload dtype (BF16 page 2048 B, BFP8
page 1088 B) and this layer reduces decode in BF16 and prefill in BFP8:

| op / payload | 4352 B | 8192 B |
| --- | --- | --- |
| decode, BF16: RS + AG | 35.80 μs | **34.82** |
| prefill, BF16: all-reduce | 2197.8 | **1928.3** |
| prefill, BFP8 (shipped): all-reduce | **1563.7** | 1581.1 |

8192 ships (1.96 μs per token per layer of decode against 0.0042 μs per prefill
token), and the warning now fires on the prefill collectives recommending 4352 --
disclosed rather than silenced. 15232, the hardware maximum, adds nothing.

**The fractured-residual rejection was right for the wrong reason.** The first
version rejected it partly on an L1 argument -- that a fractured residual forces a
≤4-core grid, citing `grid4`'s CB clash. That is wrong: `grid4` is a
*replicated*-residual run whose per-core shards are 4x wider, and in the fractured
contract the matmul inputs are the *gathered* tensors and stay on 16 cores while
the fractured tensors sit on any count dividing their 52 tiles. Re-measured
properly, at the shipped sharded memory configs (`bench/fractured_decode_probe.py`):

| term | replicated | fractured |
| --- | --- | --- |
| hidden RMSNorm | 15.46 μs (16 cores) | 10.83 / 12.88 / 15.55 / 19.24 μs (4 / 13 / 26 / 52 cores) |
| residual add | 8.44 μs | 8.08-8.44 μs |
| stats all-gather | — | 10.42 μs |
| `rms_norm_pre_all_gather` | — | 27.01 μs |

Best case 4.63 μs saved per norm against 10.42 μs added, i.e. a net loss of
5.8 μs per distributed norm before the `pre_all_gather` and the extra residual
all-gather. The interleaved probe reaches the opposite answer because an
interleaved full-width norm is the expensive thing there; these norms are not
interleaved. The probe was also re-run with the shipped reducer on both arms
(`--tuned`, now the default) in case the tuning had confounded it: `fractured`
282.81 vs `replicated` 315.36, i.e. unchanged.

## 8. Profiles

Eight windows, `bench/run_tracy.sh` -> `tracy/{sliding,full}/`: prefill at 128 and
8192 tokens and traced decode at 2048 and 131071, for both layer kinds. All eight
are free of dropped profiler markers (`grep -c "markers were dropped"` is 0 in
every `logs/tracy_*.log`). Findings are in the README's
"Performance accounting"; the two that changed decisions:

* the two reductions are **53.2 μs per decode replay** (12.0 %) at a 40 KB
  payload, i.e. pure fixed cost, which is what makes an extra collective
  expensive and settles the residual-contract question;
* the four hidden-size RMSNorms are **33.7 μs** (7.6 %) of the `sliding` decode
  step, and all six RMSNorms are **3460.4 μs** (19.0 %) of the 8192-token
  `sliding` prefill layer, and they do not shrink at all under a replicated
  residual — the largest non-matmul cost in both regimes, and the subject of
  limitation 1.

(Both figures are re-derived from the committed CSVs by
`bench/check_reported_figures.py`; an earlier version of this entry said 55.9 μs
/ 12.6 % and 3433.0 μs / 19.2 %, which are in no capture.)

One capture had to be re-taken. `run_tracy.sh` originally copied the newest
`ops_perf_results_*.csv`, and Tracy's post-processing writes a *device-only* CSV
over the same path a moment later; that file has no `DEVICE ID` column, so
`tt-perf-report --csv` refused it. The `capture` helper now picks the newest CSV
that still has the column, and the `full` decode@131071 window was re-captured.

## 9. Correctness

### 9.1 Two bugs the suite found

**`minimal_matmul` subblocks, on every batched prefill.** The inherited prefill
path chose `(subblock_h, subblock_w)` as `(2, 4)` or `(4, 2)` from the output
width against the row count. That is safe on one chip and **not** here: tensor
parallelism narrowed every per-device output width, so a table entry with
`M_block = 2` that took the `(2, 4)` branch on one chip takes the `(4, 2)` branch
against a 1280-wide fractured `wqkv` output — and `minimal_matmul` requires
`M_block_size % subblock_h == 0`:

```
TT_FATAL: M_block_size (2) must be divisible by subblock_h (4)
```

It fired on all six `test_batched_prefill_decode_pcc` cases and both
`test_real_weights_traced_decode_and_batch` cases, because a batched prefill's
per-user rows exceed 1280. Fixed by `minimal_matmul_subblocks()`, which keeps the
inherited pair wherever it is legal — every measured table entry was swept with
it — and falls through to the largest legal pair otherwise.

**A submesh of the open mesh wedges the parent's collectives.** The first version
of the multichip-vs-single-chip comparison built the single-chip layer on
`mesh.create_submesh(ttnn.MeshShape(1, 1))`. The submesh works; the *parent* mesh
does not, afterwards. Minimal repro (`/tmp/submesh_repro.py`, reproduced in
`tests/test_multichip_vs_single_chip.py`'s docstring):

```
STAGE open 1x4 / opened / submesh ok devices=1
STAGE submesh from_torch ok / submesh matmul ok
SUB_RESULT torch.Size([1, 1, 32, 256])
STAGE parent from_torch ok
<hangs in ttnn.all_reduce, forever; a 120 s SIGALRM never fires because the wait
 is inside a C++ op that does not release the GIL>
```

The hang also wedges the fabric: a **fresh** process afterwards hangs in its first
`all_reduce` too. Recovery followed `$tt-device-usage` — see 9.2. The comparison
now runs two meshes sequentially in its own module instead, which is also the
more faithful arrangement (each layer runs in the regime its own stage measured).

### 9.1b Two more, from the long session

**CCL semaphores fragment L1, and the fix is one mesh-open argument.** Every CCL
dispatch creates a global semaphore, and
`all_gather_multicast_factory.cpp:36-43` puts it in `L1_SMALL` **only if the mesh
has one** -- otherwise in the main L1 pool, with the warning *"Allocating
semaphores in L1, which may fragment L1 and reduce headroom for subsequent op
allocations. Configure an L1_SMALL region to mitigate this."* The semaphore
belongs to the cached program, so it is never freed, and it lands at the top of
L1. Measured with `ttnn.get_memory_view(mesh, ttnn.BufferType.L1)`:

| point | allocated/bank | largest contiguous free/bank |
| --- | --- | --- |
| fresh mesh | 0 | 1,461,376 |
| after a 256-row prefill | 384 | 1,460,992 |
| after one decode | 704 | **1,434,048** |

27 KB of the top of L1 is gone after a single decode, and the *next* 256-row
prefill -- whose sharded RMSNorm wants two 213 KB L1 tensors -- fails:

```
Statically allocated circular buffers in program 20 clash with L1 buffers on
core range [0-0 - 7-1]. L1 buffer allocated at 1119552 and static circular
buffer region ends at 1137536
```

Three sizes were tried before the shipped one. `l1_small_size = 32768` makes the
*first* prefill fail instead (the 256-row sharded norm is within 32 KB of the L1
ceiling -- an inherited single-chip design point, `PREFILL_NORM_SHARD_MAX_ROWS`
was chosen at that edge). `l1_small_size = 2048` alone is not enough either,
because only `all_gather` reads the L1_SMALL bank size; `reduce_scatter` needs
`use_l1_small_for_semaphores=True` explicitly. With both at 2048 the residue is
flat and unfragmented -- but the region itself then fills partway through the
suite, because a semaphore is 16 B per bank and 2 KB holds only 128 of them:

```
TT_FATAL: Out of Memory: Not enough space to allocate 1760 B L1_SMALL buffer
across 110 banks, where each bank needs to store 16 B
```

at 354 cached programs, in `test_batched_prefill_decode_pcc[13-sliding]`.

`8192` is over the ceiling from the other side: the decode step's circular
buffers end at 1,137,536 B and its live L1 tensors take 316,544 B from the top, so
it uses 1,454,080 of the 1,461,376 B pool and has 7,296 B to give away -- at 8 KB
it is 896 B short (*"L1 buffer allocated at 1136640 and static circular buffer
region ends at 1137536"*, in the graph-audit tests).

**What ships is 6144, not the 4096 this section originally recorded.** 4096 was
the first value that worked, and it is what the measurements below were taken at.
The full ladder came later, when the fabric packet probe swept it (§7.4), and 6144
is the largest value that keeps a four-figure margin: **24** distinct CCL programs
and 1,152 B of decode margin, against 16 and 3,200 B at 4096. The README ladder,
`DEFAULT_L1_SMALL_SIZE` and `context_contract.json` all say 6144/24; review round 3
caught this section still saying 4096/16 — a value the README ladder describes as
"passes; the region itself becomes the constraint".

At the 4096 it was measured at: 256 semaphores, 3,200 B of decode margin, and:

| point | allocated/bank | largest contiguous free/bank |
| --- | --- | --- |
| fresh mesh | 0 | 1,459,328 |
| after prefill + decode, both kinds | 256 | 1,459,328 |

No whole-layer latency claim is attached to this change. An earlier version of
this line cited a before/after pair that appears in no committed log, and the runs
that bracket the change also moved the packet size and the per-payload worker
counts, so the artifacts cannot isolate it. What is verified is the shipped
configuration's final numbers, in `logs/layer_ab_final.log`.

**How much room 4 KB is, measured rather than assumed.** Six distinct
`reduce_scatter` shapes take 1,536 B of the region -- 256 B each -- and
`mesh.clear_program_cache()` returns all of it:

```
L1_SMALL[fresh]                    alloc=0    free=4096  progs=0
L1_SMALL[after 6 shapes]           alloc=1536 free=2560  progs=12
L1_SMALL[after clear_program_cache] alloc=0   free=4096  progs=0
```

So a mesh held **16 distinct CCL programs** at 4096 — 256 B each, which is the
arithmetic that gives **24** at the shipped 6144. The suite exceeded that
twice before it was bounded: once at 354 cached programs with a 2 KB region, and
once inside a single `test_batched_prefill_decode_pcc[32-*]`, whose 32 distinct
per-user prefill lengths are 32 distinct CCL programs and which no
between-tests clear can help. Both are properties of a *suite*, not of the layer
-- a stacked model reuses one program per layer kind -- so the fixes are in the
tests: a fixture that clears on either trigger (program count > 120, or under
1536 B of L1_SMALL free), and 8 distinct per-user lengths across the 32 users
instead of 32. The full-model stage inherits the number, and the README says so.

**The QKV projection's output lands on 14 cores, not 16.** 40 tiles over 16 cores
is 2.5, so `per_core_N` rounds up to 3 -- and the DRAM-sharded matmul writes on
its own storage-core layout rather than the requested grid (the single-chip
stage's TTNN finding 3), so it uses `ceil(40/3) = 14` cores of 3 tiles. 42 tiles
for 40. `test_qkv_output_shard_is_padded_not_wrong` now pins the core count as
well as the padding; the earlier version asserted 16 and was wrong about the op,
not about the layer.

### 9.2 Hardware recovery

| item | value |
| --- | --- |
| failure signature | `ttnn.all_reduce` never returns, on the parent mesh after a submesh was created; then in a fresh process too |
| exposing command | `python -m pytest tests/test_multichip_decoder.py -k "vs_single_chip and sliding"` (exit 124 at 420 s) |
| triage | `tools/tt-triage.py --llm-output` -> `triage/tt-triage.txt.gz`, `triage/triage-summary.txt`: every script `pass` except `check_binary_integrity` (stale kernel ELFs, expected on idle cores); `dump_op_mesh` shows **all four devices idle**, `dump_running_operations` `pass` — so nothing was in flight on the device and the wait was host-side |
| processes killed | the hung pytest, plus one second pytest of my own that had queued behind it (my error: two device jobs at once) |
| reset | `tt-smi -ls --local` (4 boards) -> `tt-smi -r` (exit 0, "Resetting all PCI devices: [0, 1, 2, 3]") -> `tt-smi -ls --local` (4 boards) |
| second reset needed | no |
| locks cleared | none needed |
| mesh smoke | `open_mesh_device(1x4)` + `all_reduce` -> `CCL_SMOKE_OK Shape([1, 1, 32, 256])` |
| classification | infrastructure recovery from a TTNN submesh/fabric interaction, not a model correctness or performance result |

Three more resets were needed later in the stage, all from the same Ethernet-core
signature and all bounded (one `tt-smi -r`, list before and after, CCL smoke to
confirm): after the two correctness modules were run in **one** pytest invocation
(the 1x1 mesh opens while the 1x4 session mesh is still owned), and after each of
the two watcher runs' teardown aborts (§9.4). None of them is a model result; all
of them are recorded because the full-model stage opens and closes this mesh in
every process and will meet the same signature.

### 9.3 The comparison the stage rests on

`tests/test_multichip_vs_single_chip.py`, `logs/vs_single_chip_run.log`:

| case | prefill | 4 decode steps | worst |
| --- | --- | --- | --- |
| sliding, 2049, batch 1 | 0.999839 | 0.999851-0.999860 | 0.999839 |
| full, 2049, batch 1 | 0.999807 | 0.999816-0.999825 | 0.999807 |
| sliding, 12345, batch 4 | 0.999721 | 0.999843-0.999844 | 0.999721 |
| full, 12345, batch 4 | 0.999609 | 0.999183-0.999202 | **0.999183** |

Bar 0.999. The second pair was added after the stage review pointed out that a
single batch-1, single-chunk case is narrower than the surface it stands in for:
12345 tokens is two internal prefill chunks (so the per-device single-KV-head
sliding tail crosses a chunk boundary), batch 4 gives four cache slots and ragged
decode positions, and the length divides by neither tile, page block nor chunk.

The rest of the surface is in `logs/full_test_run.log`, `test_results.xml` and
`logs/pcc_summary.txt`: **104 tests, 104 passed, 298 asserted PCC checks** -- 20
against the single-chip layer (worst 0.999183), 30 on the released checkpoint
(worst 0.995105, bar 0.995), 238 single-layer synthetic (worst 0.990516, bar
0.99) and 2 two-layer-chain (worst 0.967946, bar 0.96).

### 9.4 Watcher, and a teardown fault

31 node ids, `TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0
TT_METAL_WATCHER_NOINLINE=1`, in a run with no profiler attached: **31 passed**,
zero `Watcher detected` / tripped / sanitize / `TT_ASSERT` / `DEBUG_ASSERT` /
out-of-bounds / fault / Error lines across 28,788 log lines and 50 dumps
(`logs/watcher_run.log`, `watcher/watcher.log.gz`).

The process then aborts **at teardown**, after every test has reported, with
pytest exit 134:

```
TT_THROW: Device 0: Timed out while waiting for active ethernet core 29-25 to
become active again. Try resetting the board.
  --- RiscFirmwareInitializer::assert_active_ethernet_cores_to_reset
  --- tt::tt_metal::MetalContext::~MetalContext()
```

Seen twice (18:48 and 19:52), both times closing the 1x4 `FABRIC_1D_RING` mesh
with no submesh and no second mesh in the process -- so it is *not* the
1x1-after-1x4 interaction documented in `test_multichip_vs_single_chip.py`, which
an earlier version of this log wrongly conflated with it. It does not
self-recover: both times the next process to open the mesh failed at startup with
the same signature and `tt-smi -r` cleared it (one reset each). It has only been
seen on the watcher path -- the same node ids inside the 100-test acceptance
suite, the A/B runs, the eight Tracy captures and the comparison module have
never produced it.

`bench/run_watcher.sh` now runs without `-e`, captures the pytest exit code,
prints it next to the grep counts and exits with it; `bench/run_evidence_chain.sh`
records it and carries on to the last step. Before that fix the script could not
have produced its own artifact.

The comparison module is deliberately not in that run. It opens a 1x1 mesh, and
opening one shortly after a `FABRIC_1D_RING` 1x4 mesh has closed intermittently
times out on an Ethernet core (*"Timed out while waiting for active ethernet core
29-25 to become active again"*) and costs a `tt-smi -r`; it happened twice, once
in the watcher chain and once when the two correctness modules shared a pytest
invocation. Every op it dispatches is covered by the same layer on the same mesh
in the 31 nodes above.

## 12. Review round 1, and the artifact it could not reproduce

The first `$stage-review` returned `more-work-needed` on six required items; §7.3
and §7.4 above are the measurements that closed them (the fractured-residual
rejection rebuilt on decode-regime numbers, the fabric packet size measured per
payload dtype, the "cannot allocate on this part" claim corrected to "the probe
exhausted its own L1_SMALL region", the `l1_small` ladder recorded, per-payload
`num_workers_per_link`, and the BFP8 decode-payload rejection restated with both
margins). Two smaller notes were also fixed: `bench/run_tracy.sh` no longer claims
per-device rendering "via `--device-id`" -- a flag it never passes -- and instead
records what tt-perf-report actually does with a 4-device capture ("Detected data
from 4 devices. Merging device data...", one row per op instance, verified by
summing Device Time over a decode window and dividing by the 8 replays); and
`bench/run_evidence_chain.sh` gzips its ~700 KB Tracy console log, which the
repo's file-size hook rejects.

The last one was a reproducibility hole rather than a wording bug.
`logs/real_weight_decode_bfp8_experiment.log` -- the whole real-weight surface
re-run with the **decode** collective payload flipped to BFP8, which is the
measurement that rejects that payload -- had been produced by editing
`DEFAULT_DECODE_CCL_DTYPE` in the source and reverting it. Nothing committed could
regenerate it. `build_multichip` in the suite now reads
`MG_MULTICHIP_DECODE_CCL_DTYPE`, so the artifact regenerates from committed code:

    MG_MULTICHIP_DECODE_CCL_DTYPE=bfloat8_b python -m pytest \
      models/autoports/meta_models_muse_glimmer_30b/tests/test_multichip_decoder.py \
      -k real_weights -q

Re-running it that way reproduced the number the rejection rests on to every
printed digit -- 16 passed, worst real-weight PCC **0.9950028290443281** on
`decode[sliding] step=6 pos=3006`, against 0.995105 for the shipped BF16 payload.
The env var is test-only plumbing; it is not read by `multichip_decoder.py` and
changes no default.

## 13. Review round 2, and the five things it caught

The second `$stage-review` returned `more-work-needed`. Nothing it found was a
defect in `multichip_decoder.py`; all eight items were the same failure mode in
different places — **a correction applied at one site and not at the others**.
Round 1 withdrew three claims (the `use_l1_small_for_semaphores` "part" claim, the
fractured-residual L1 argument, and the "rs_ag prefill is 12 % slower" figure) and
each withdrawal reached the source comment but not the README summary table, or
the README prose but not the work log. The result was a document that contradicted
itself 300 lines apart — the README both stated that the semaphore flag cannot
allocate and that the shipped code passes it.

Fixed by propagating each correction to **every** site — except, as round 3 then
found, in this log, which was appended to rather than edited, so this section as
originally written claimed a §7.3 fix that had not been made. §14 records that.
The corrections were: the flag (README §7.3), the fractured rejection (the "Measured and rejected" row now cites
`logs/fractured_decode_probe.log` and the *tuned* topology log, and states the
≥11.6 μs net loss instead of the retracted L1 blocker), and the prefill reducer
(`_all_reduce`'s docstring, `DEFAULT_PREFILL_CCL_MODE`, and the rejected-candidates
row).

### 13.1 Three device runs, because three claims were under-measured

`bench/run_review2_chain.sh`, one job at a time:

1. **`fractured_decode_probe` and the tuned topology probe re-run.** Both predated
   the 8192 B packet change and logged the old 4352 B warning, so neither measured
   the shipped fabric. Re-run at 8192: norms 15.47 / 10.80–19.25, stats gather
   10.47, `pre_all_gather` 26.97 (was 15.46 / 10.83–19.24 / 10.42 / 27.01), and
   `fractured` 282.19 against `replicated` 315.15 (was 282.81 / 315.36). Every
   number moved by less than 0.5 % and the rejection is unchanged, but it now
   rests on the configuration that ships.

2. **One A/B invocation for all four reducer candidates**
   (`logs/layer_ab_reducer_final.log`), because the README's three-row reducer
   table had been stitched from three different runs. The result is more useful
   than the fix: rows 1 and 4 dispatch the *same* prefill collective and differ by
   0.57 ms on `sliding` (3.0 %), so whole-layer prefill cannot
   resolve a reducer choice at all — and the `rs_ag` prefill row is 2.8 % slower on
   `sliding` and 0.6 % *faster* on `full`. Decode is decided and repeatable: the
   pair wins 1.1 % on both kinds. The prefill claim now rests on the op-level
   0.24 % at the shipped packet size, and the README says so.

3. **The fabric packet size, measured whole-layer for the first time.** `layer_ab`
   gained `--packet-bytes` (a mesh-open argument, so it needs one process per
   value, not a CANDIDATE). 8192 against 4352: decode **0.4574 / 0.4259** against
   0.4589 / 0.4282, prefill **18.84 / 18.29** against 19.11 / 18.32. 8192 wins or
   ties all four windows. The isolated 1.1 % prefill-collective regression at 8192
   is real and does **not** survive to the layer. The runtime's advice is also
   provably unsatisfiable: at 8192 it asks for 4352 (1088 B BFP8 prefill pages), at
   4352 it asks for 8192 (2048 B BF16 decode pages). Both warnings are in the two
   logs. The README now has the `Fabric packet size` section it had been linking
   to, and "load-bearing" is restated as what it is: half a percent of decode, not
   a correctness constraint.

### 13.2 The gate that should have caught all of this

Every round-2 finding lived in the part of the README `check_reported_figures.py`
did not cover. It now covers three more classes:

- **the DRAM/%-of-peak table** — per-role means over the 8 replays, re-derived from
  the CSV. This immediately caught two stale entries: `wqkv` was 384 / 75.0 against
  a real 382.41 / 74.69, and `attn_gate`'s 68.4 was the *minimum* instance, not the
  mean (68.55);
- **every artifact the prose cites** — a ``logs/...`` path that does not exist is
  now a failure;
- **every in-document link** — which found the dangling `#fabric-packet-size`
  anchor the summary table had been pointing at since the packet decision was made.

Also classified, at the review's request: the `No output subblock size found`
advice on 32 of 48 decode matmul rows. Not actionable —
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` exposes only `in0_block_w`,
`per_core_M`, `per_core_N` and `fused_activation`, and the report's subblock
columns are empty for every matmul row in this stage *and* the single-chip one
(24 of 48 rows there). The count rises 24 → 32 only because `o_proj` joins the
SLOW set, which is the per-device K=1024 blocking limit already documented.

And one artifact disagreement: `context_contract.json` said the `l1_small` ladder
"2048/4096/6144/7168 all pass" while the README and the source both record 2048
filling the region mid-suite. The contract is the machine-read artifact, so it now
carries the whole ladder verbatim.

## 14. Review round 3: the log itself was the defect

Round 3 returned `more-work-needed` with six required items, and the diagnosis was
sharper than the list. Rounds 1 and 2 had both been closed by editing the README
and the source and **appending** to this log. Round 3 checked the log against them
and found that three withdrawn claims were still asserted here, that §9.1b still
documented `l1_small_size = 4096` and 16 CCL programs as shipped when the module
had shipped 6144 and 24 since the ladder was measured, and that §13 claimed a fix
to §7.3 that had never been made. An append-only log is not a log; it is a pile of
drafts, and the only automated gate read the README.

So the fix has two halves.

**The log is now edited in place.** §7.3 carries the retraction of the
`use_l1_small_for_semaphores` claim and the sourced 120.6 GB/s figure. §9.1b says
what ships (6144 / 24) and labels its own measurements as taken at the 4096 that
was current when they were made. The before/after latency pair it quoted for the
`l1_small` fix is gone: it appears in no committed log, and the runs that bracket
that change also moved the packet size and the worker counts, so nothing committed
can isolate it — saying so is better than quoting a number that cannot be found.
§12's references to two sections that do not exist (7.5 and 10) are corrected —
written without the section sigil here, because the gate below cannot tell a
reference from a mention of a broken one.

**The gate now reads this file.** `bench/check_reported_figures.py` gained three
checks over `work_log.md`: every ``logs/…`` path it cites must exist, every §N.N
cross-reference must resolve to a heading, and any *shipped* value it quotes for
`l1_small_size` or the fabric packet payload must equal the constant the module
defines. The last of those is the one that would have caught the 4096 the day it
went stale. It found two more problems as soon as it ran: a bad cross-reference in
§12 and this section's own forward reference before it was written.

### 14.1 The fractured-residual rejection, rebuilt properly

The substantive finding was that the decode-regime rejection did not survive its
own methodology. It summed floor-cancelling *differences* (two norms) with
floor-inclusive *absolutes* (a stats gather), and the replay floor is 3–10 µs —
larger than the effect being measured. It also priced only one term of the
distributed norm, and priced it on DRAM-interleaved inputs while claiming the
shipped sharded layout.

`bench/fractured_decode_probe.py` was rewritten:

- **the floor is calibrated.** Each op runs at 1, 2, 4 and 8 copies per trace; the
  per-op cost is the slope, the floor the intercept. Both print. (The full-width
  norm: 8.11 µs per op on a 7.36 µs floor — the old probe's single point read
  15.49, which is the sum.)
- **the distributed norm is priced whole, on the sharded layout**, as
  `rms_norm_pre_all_gather` → `all_gather` → `rms_norm_post_all_gather`. Getting
  there needed two TTNN facts the old probe had walked around: the pre-op requires
  the sharded program config (without one it raises *"std::get: wrong index for
  variant"*, which is why the old probe fell back to DRAM), and the post-op
  requires the gathered statistics to be **sharded** too (*"Stats must be
  sharded"*, `layernorm_device_operation.cpp:236`).

The rejection survives, with a bigger and better-founded margin: **8.11 µs** for
the shipped full-width norm against **14.90 µs** for the distributed path
(3.19 + 6.41 + 5.29), i.e. **+6.79 µs per distributed norm** and **+13.57 µs per
decode step** (3.1 % of 444 µs), against no saving on the residual add
(5.41 replicated vs 5.42 at every fractured core count) and identical collective
bytes.

And a constraint fell out that neither earlier version had found: of the four core
counts that divide the fractured width's 52 tiles, a *distributed* norm is legal
only at **4**. 13, 26 and 52 are each attempted and each raises *"Sharded
layernorm does not support a non-rectangular core grid for distributed norm"*. This is not the ≤4-core claim
round 1 withdrew — that one was about the matmuls and was wrong — but it is a real
one, in a different place, and the probe now records it as a measurement rather
than an assumption.

### 14.2 The reducer family, completed

Round 3 also caught that the "four candidates in one invocation" table had only
three distinct configurations: `ccl_mode="rs_ag"` and `prefill_ccl_mode="rs_ag"`
resolve to the same thing. Rather than delete the duplicate, it is now labelled as
what it is — a **same-config repeat inside one process**, which measures the
prefill spread with everything else held fixed (2.3 % on `sliding`) — and a
genuinely distinct candidate was added: `ccl_rs_ag_prefill_w1`, prefill `rs_ag` at
**one** worker.

That row explains the figure this stage has now corrected twice. At one worker,
prefill `rs_ag` is 21.04 / 20.65 ms against the shipped 18.89 / 18.36 — **11.4 % /
12.5 %** slower. So the original "12 % slower" was a real measurement of the
*worker count*, recorded as if it were the *reducer form*. Split per payload, the
reducer form on prefill is worth 0.24 % and the worker count is worth 12 %.

### 14.3 Everything re-run at the shipped configuration

`logs/topology_probe_prefill8192.log` was the last artifact still carrying the old
4352 B packet and the untuned reducer, and it is the sole evidence for limitation 1
and for the 8192-row column of the contract-families table. It was re-run at the
shipped packet and the tuned reducer — which round 4 then showed was still not the
shipped *configuration*, because the probe pinned `num_workers_per_link=1` at the
prefill payload. See §15.2; the numbers that round produced are superseded.

`bench/run_review2_chain.sh` runs all six device jobs in order. Round 3 noticed
that the committed *console transcript* of that script had been stitched from two
versions of it, so no transcript is committed at all now: each of the six steps
writes its own log, those are the artifacts, and every one of them is reproduced
by the committed script. A console transcript that can drift from the per-step
logs is a second source of truth for nothing.


## 15. Review round 4: the gate now reads the numbers, not just the paths

Round 4 returned `more-work-needed` on six items, and every one of them was the
same defect round 3 had already named — a correction landing in one document
while another kept the superseded figure. Round 3's own fix *caused* most of
them: it regenerated six logs and rewrote the README, and left the earlier
sections of this log quoting the numbers those logs used to contain. §4's entire
8192-row column, §7.2's reducer table, and a dozen figures in §7.4 and §13.1 had
become unfindable in any artifact.

### 15.1 The check that ends this class

`bench/check_reported_figures.py` gained the check the last three rounds needed.
For every block of README.md and work_log.md that cites a `logs/...` artifact,
every **measurement** in that block must appear in one of the artifacts it cites.
A measurement is three decimals or more (`0.4573`), or two decimals at magnitude
5 or more (`8598.18`, `27.01`) — which selects what a probe prints and skips what
a document computes: ratios, percentages and differences are in no log by
construction. Two refinements were needed to make it usable rather than noisy:

- a table inherits the citation of the paragraph that introduces it, since that
  is where a doc normally names its source. Without this the round-4 findings —
  a whole table gone stale under a regenerated log — stay invisible;
- a quoted value may be a **rounding** of a full-precision log value, because PCC
  prints 17 digits and the tables show six. The check accepts any log value that
  rounds to the quoted one at the quoted precision.

A block that is deliberately historical opts out with `<!-- superseded: … -->`,
which forces that decision to be written down instead of merely being true today.
§4's contract table and §7.2's reducer table now carry one, each with a pointer
to the current numbers.

Run against HEAD it found **40 violations in ten blocks**, including all six
round-4 items and three the review had not reached. Every one is now either
re-derived, re-cited, or explicitly marked superseded.

### 15.2 One finding was not bookkeeping

The 8192-row contract-families column was labelled "the shipped configuration"
after §14.3 re-ran it at the shipped packet — but `bench/topology_probe.py` still
pinned `num_workers_per_link=1` on every arm, and 1 is the *decode* value. At the
prefill payload that costs the reduce-scatter 2.4x (§7.4: 1814.9 μs against
759.9 at four workers), which inflates every arm that uses one — while the
`gather_heads*` arms, which use only `all_gather`, escape it entirely. The
comparison was not like-for-like, and the number handed to limitation 1 was
wrong.

The probe now takes `--rs-workers`, defaulted from the row count exactly as the
shipped layer does (`DEFAULT_DECODE_CCL_RS_WORKERS` at 32 rows,
`DEFAULT_PREFILL_CCL_RS_WORKERS` at 8192), and prints it in its header line so a
log says which it used. Re-run, the confound is visible and large:

| candidate | 8192 rows, `rs_w1` (round 3) | 8192 rows, `rs_w4` (shipped) |
| --- | --- | --- |
| `replicated` | 9002.41 μs | **8242.22 μs** (−8.4 %) |
| `fractured` | 7756.40 | 7212.19 (−7.0 %) |
| `gather_heads` | 6811.28 | 7108.30 (+4.4 %) |
| `gather_heads_fractured` | 5789.59 | 5791.07 (+0.0 %) |

The two arms with no reduce-scatter did not move; the two with one gained 7–8 %.
The ranking survives, but the margins limitation 1 hands to the full-model stage
change materially: the fractured family is **1.14x** faster on the prefill chain,
not 1.16x, and `gather_heads` buys **13.8 %** of prefill, not 24 %. Both are
corrected in the README.

### 15.3 The rest

- The distributed-norm legality claim was evidenced for 13 and 26 cores only. 52
  is now attempted too and also raises *"Sharded layernorm does not support a
  non-rectangular core grid for distributed norm"*, so "legal only at 4" is
  measured at every count rather than at two of three.
- `55.9 μs (12.6 %)` for the two decode reductions and `3433.0 μs (19.2 %)` for
  the prefill norms are in no capture; the CSVs give 53.2 μs (12.0 %) and, for
  all six norms, 3460.4 μs (19.0 %). The README's copies were already right.
- The watcher run has **25** dumps, not 50 — `grep -c Dump` counts each dump's
  start and completion lines. `bench/run_watcher.sh` made the same miscount and
  now divides.
- Three README percentages silently mixed capture windows: the SDPA's 4.7 % is
  `sliding`@2048 and the 23.2 % is `full`@131071; the BFP4 MLP rows are 43.4 % of
  the `sliding` decode step and 45.5 % of the `full` one; and the 19.0 % of
  prefill is all six RMSNorms, not the four hidden ones.
- `test_kv_cache_holds_the_expected_head` checked the **K** cache's head identity
  and gave V only shape and dtype. K and V are written by separate
  `paged_fill_cache` calls off separate slices of the fused QKV output, so a
  head-assignment mistake can land in one and not the other. It now unpicks both
  through the permuted page table (8 more asserted PCC checks: 298, not 290).
- `ttnn.all_reduce` is not one device program — it decomposes into
  `reduce_scatter_minimal_async` + `all_gather_async`. "One dispatch against two"
  is a statement about the *host*, and §7.2 now says so. It also explains why the
  two forms measure so nearly alike at a bandwidth-bound payload, which had been
  presented as a surprise.
