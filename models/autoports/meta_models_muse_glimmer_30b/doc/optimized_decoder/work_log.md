# Optimized decoder work log — `meta-models/Muse-Glimmer-30B`

This log is chronological. Sections 15, 16.6 and 17.6 record the state at the end
of each earlier review round and are kept for the audit trail; **section 18.3 is
the current state.**

Date: 2026-08-12. Host `tt-quietbox`, 4 x Blackhole visible, stage run on a
1x1 mesh. Repo `/home/ttuser/dev/muse-glimmer/tt-metal`, branch
`agentic-research/hous/muse-glimmer-30b`. Python env
`/home/ttuser/dev/muse-glimmer/muse-glimmer_pyenv` (transformers 5.15.0,
tt-perf-report 1.2.8).

Stage input: the completed fused decoder (`tt/fused_decoder.py`,
`doc/fused_decoder/`). Stage output: `tt/optimized_decoder.py`,
`tests/test_optimized_decoder.py`, this directory.

## 0. Device health

`tt-smi` is not installed on this host, so the bounded list/reset/list sequence
from `$tt-device-usage` is unavailable; the TTNN open/close mesh smoke was used
instead, before and after the stage:

```bash
which tt-smi                       # -> "tt-smi not found"
python -c "import ttnn; m=ttnn.open_mesh_device(ttnn.MeshShape(1,1), trace_region_size=0); \
           print('MESH_SMOKE_OK', m.arch(), m.compute_with_storage_grid_size(), \
                 m.dram_grid_size(), ttnn.get_num_devices()); \
           ttnn.close_mesh_device(m)"
# MESH_SMOKE_OK Arch.BLACKHOLE 11-10 8-1 4      (before and after the stage)
```

The `8-1` DRAM grid is not incidental: it is the number the prefill 2D-multicast
matmul's core-column count must equal, and section 11 is about what happens when
it does not.

No hangs, no ARC/ERISC/Ethernet faults, no resets and no `tt-triage` capture were
needed. Every device-facing command was run one at a time; the watcher run and
the Tracy runs are separate invocations.

## 1. The starting point and the two levers

`doc/fused_decoder/README.md` ends with an unusually specific hand-off
(limitations 1, 2 and 14): decode is at the **BF16 weight-streaming roofline**,
93 % of a 2.710 ms step is six matmuls moving 967,835,648 B of BF16 weights at
383 GB/s (75 % of this part's ~512 GB/s), and prefill is compute-bound on
`minimal_matmul`. "The remaining levers are weight dtype and matmul config."

So the whole stage is two questions, and everything else follows from the answers:

1. **How few bytes can the weights be?** -> precision policy, section 3.
2. **How close to peak can the decode matmul get?** -> DRAM-sharded matmul plus
   the activation layout it requires, sections 2 and 4.

### 1.1 The op-contract survey that shaped the design

Three probes, run before any code was written, because each answer removes or
creates whole branches of the design space
(`logs/weight_layout_probe.log`, `logs/short_prefill_probe.log`):

| question | answer | consequence |
| --- | --- | --- |
| does the DRAM-sharded matmul need a width-sharded **weight**? | yes — `input_tensor_b.memory_config().memory_layout() == WIDTH_SHARDED` (`matmul_device_operation.cpp:1312`) | the weight layout is not a free choice |
| can `minimal_matmul` read a width-sharded weight? | **yes**, and marginally faster than interleaved (1.5633 vs 1.5750 ms at 8192 rows) | **one** weight tensor serves prefill and decode; no second copy |
| can `ttnn.linear` read a width-sharded weight? | ~~**no**~~ — **only with the wrong program config.** `MatmulMultiCoreProgramConfig: Input B memory layout must be INTERLEAVED` (`:1233`) is the *auto-selected fallback* talking; the 2D-multicast validator accepts it (`:1541-1553`). **Corrected in section 11.** | the fused stage's "`ttnn.linear` below 3072 rows" branch was retired for one release of this stage and then came back as a 2D-multicast dispatch |
| can the DRAM-sharded matmul do more than one M tile? | **no** — `M == 1` (`:1287`) | it serves decode and a <=32-row prefill, nothing between |
| can it write an interleaved output? | **no** — `Output memory config must be sharded` (`:1268`) | the decode activation layout has to be sharded end to end |

The second row is the one that made the stage cheap. Had `minimal_matmul` refused
the width-sharded weight, the choice would have been between a second weight copy
(+314.8 MB per layer, x52 in a full model) and a much worse prefill; instead
prefill and decode share one tensor and the layer's weight footprint *drops* from
967.8 MB to 314.8 MB.

The third and fourth rows are the stage's two real costs, and both are recorded
as README limitations rather than smoothed over.

## 2. Operation-topology audit of the measured path

Required by `$optimize` before local knob tuning. Derived from the fused stage's
committed `tt-perf-report` tables plus the decode/prefill op sequence in
`tt/fused_decoder.py`, not from a comparison to any other model.

### 2.1 Decode (batch 1, context 2048), fused baseline

| # | op group | x | share | topology observation | action taken |
| --- | --- | --- | --- | --- | --- |
| 1 | `MatmulDeviceOperation` (6 dense projections) | 6 | 93.2 % | DRAM-bound at 383 GB/s on an auto-selected config; weights BF16 interleaved | **BFP8/BFP4 weights + explicit DRAM-sharded program config** (sections 3, 4) |
| 2 | `LayerNormDeviceOperation` | 6 | 2.2 % | already width-sharded L1 multi-core (the fusing stage's win) | grid moved 8 -> 16 cores to serve the matmuls; norms measured as part of the whole layer |
| 3 | `SdpaDecodeDeviceOperation` | 1 | 1.4 % @2048, **16.6 % @131071** | `q_chunk=32, k_chunk=64` inherited unexamined | **op-chosen chunking**, 19 % of the step at 131071 (section 5) |
| 4 | `BinaryNg` (2 residual adds, gating mul, SwiGLU mul) | 4 | 0.9 % | already sharded and activation-folded | unchanged |
| 5 | head split/concat, RoPE gather+apply, paged update, reshards | 27 | 2.4 % | mostly sub-0.1 % each, but not all: in the *optimized* step `NLPCreateQKVHeadsDecode` is 13 us (1.2 %) and the two `TilizeWithValPadding` inside `ttnn.embedding` are 6 us each (0.6 %). The group is 5.6 % of a 2.5x-faster step. | unchanged; the two `paged_update_cache` calls re-examined (section 6), the tilizes disclosed by the fused stage and inherited |

**Repeated same-input matmuls.** Two groups: `wqkv` + attention gate (both consume
the `input_layernorm` output) and MLP gate + up (both consume the pre-FF norm
output). Both were packed, measured across every legal geometry, and **rejected**
— section 7.

**Avoidable reshard/layout conversions.** The fused decode had 5
`InterleavedToSharded`/`ShardedToInterleaved`. The optimized decode has 6: entry
`i2s`, the QKV `s2i` to L1 interleaved for `nlp_create_qkv_heads_decode`, the
attention-output `i2s`, two MLP boundary reshards, and the exit `s2i`. The two MLP
reshards are new and are the subject of section 4.2; every other one is forced by
an op contract, listed in the README's audit table.

**Collectives.** None: single chip.

### 2.2 Prefill (8192 tokens), fused baseline

| # | op group | x | share | observation | action |
| --- | --- | --- | --- | --- | --- |
| 1 | `MinimalMatmulDeviceOperation` | 6 | 65.0 % | 228.5-255.5 TFLOPs, no row marked `SLOW`; compute-bound at BF16 | **BFP8/BFP4 weights** (1.5-1.7x) plus a per-shape, per-dtype, per-row-count block sweep (section 8) |
| 2 | `SDPAOperation` | 1 | 15.4 % | `q_chunk == k_chunk == 256`, already swept by the fusing stage over 90 measurements | unchanged; re-swept lengths would re-derive the same constant |
| 3 | `BinaryNg` | 4 | 8.6 % | already activation-folded | unchanged |
| 4 | `LayerNorm` | 6 | 7.9 % | 110 cores, DRAM-bandwidth bound | unchanged |
| 5 | `RotaryEmbeddingHf` | 2 | 1.6 % | already the dedicated op | unchanged |
| 6 | heads split/concat, paged fill, page-table slice | 5 | 1.5 % | | unchanged |

## 3. Precision policy, one tensor group at a time

All numbers `bench/layer_ab.py`, min of 2-3 rounds, traced decode at context 2048,
warmed prefill at 8192 tokens, PCC against the HF reference at seq_len 100 plus
one decode step off the prefilled cache. Both layer kinds.

### 3.1 The candidate table (real checkpoint weights)

`logs/layer_ab_real_final.log`, `logs/layer_ab_precision_real.log`:

All five candidates from **one** run, every row `AB[real]`
(`bench/layer_ab.py --candidates fused,b16_all_bfp8,gateup_bfp4,mlp_bfp4,all_bfp4
--real-weights`, `logs/layer_ab_real_final.log`). An earlier revision of this table
mixed populations -- its baseline PCC columns were the *synthetic* fused numbers --
which is why the whole frontier was re-measured in a single harness state:

| policy | decode ms/token (sliding / full) | prefill 8192 ms | prefill PCC | decode PCC | verdict |
| --- | --- | --- | --- | --- | --- |
| fused baseline (BF16) | 2.7346 / 2.7092 | 63.84 / 64.12 | 0.999566 / 0.999607 | 0.999601 / 0.999447 | baseline |
| all BFP8 | 1.2653 / 1.2348 | 45.83 / 44.86 | 0.999239 / 0.999349 | 0.999000 / 0.998798 | +16 % slower than shipped |
| gate/up BFP4, down BFP8 | 1.1487 / 1.1180 | 44.72 / 43.32 | 0.998328 / 0.998082 | 0.998172 / 0.997624 | +5.3 % slower than shipped |
| **attn BFP8, MLP BFP4 (shipped)** | **1.0908 / 1.0601** | **43.68 / 43.10** | **0.997536 / 0.997197** | **0.997157 / 0.996804** | **kept** |
| all BFP4 (incl. attention) | 1.0572 / 1.0262 | 43.91 / 42.15 | 0.977175 / 0.979843 | 0.984938 / 0.977697 | **rejected: below the 0.995 bar on real weights** |

So the frontier is monotone and the shipped point is the fastest one that clears
the functional acceptance bar. The BFP4-attention candidate is 3.1 % faster and
fails PCC by a wide margin (0.977 / 0.980 prefill) — that is OPT-007's mandatory
attention-weight trial, run on **real** weights and rejected on measured
model-visible accuracy, not on preference.

### 3.2 Synthetic weights disagree, and the three obvious reasons are all wrong

On `reference.synthetic_state_dict` (i.i.d. Gaussian with each real tensor's mean
and std) the same shipped policy measures 0.993904 / 0.992468 prefill and
0.993733 / 0.992232 decode — a **2.6x larger error** than on the checkpoint, and
below 0.995.

`bench/bfp_block_range_probe.py` tests the plausible mechanisms and refutes all
three (`logs/bfp_block_range_probe.log`):

| hypothesis | measurement | verdict |
| --- | --- | --- |
| i.i.d. samples widen the 16-element BFP block's dynamic range | `max|w| / mean|w|`: real 2.638-2.742, synth 2.631-2.633 | **refuted**, within 4 % |
| the synthetic weights quantise worse | on-device BFP4 round-trip max relative error: real 0.82-6.52, synth 0.39-0.80 | **refuted, wrong direction** — real is 1-8x worse (heavier tails) |
| a BFP4 projection is less accurate on synthetic weights | output PCC vs an FP32-weight matmul, same activation: real 0.99296-0.99356, synth 0.99344-0.99390 | **refuted**, within 1-9 %, real marginally worse |
| the layer output is less residual-dominated on synthetic weights | `||y-x|| / ||x||`: real 0.943 / 1.093, synth 1.042 / 1.183 | too small — a 10 % effect against a 2.6x one |

So BFP4 represents the two weight sets equally well *per projection*, and the gap
is an interaction inside the layer — most plausibly error cancellation across the
SwiGLU product and the down projection when the weights are structured — which
this stage did not isolate. The README says exactly that rather than inventing a
mechanism.

**The policy decision does not depend on the mechanism.** OPT-012's rule is that a
synthetic-distribution PCC cannot veto a policy that passes real-weight PCC under
the disputed conditions. So the response was to widen the real-weight coverage
until it covers every disputed condition, not to pick the slower policy:

* `test_real_weights_prefill_pcc` — 6 lengths x 2 kinds, including `seq_len=1`
  (the tile-only DRAM-sharded prefill branch), 2049/4097 (non-aligned) and 12345
  (multi-chunk);
* `test_real_weights_decode_pcc` — 8 consecutive decode steps off the BFP8 cache,
  because a cache-precision fault compounds and a single step would not see it;
* `test_real_weights_traced_decode_and_batch` — trace replay (the measured perf
  path) at batch 8;
* `test_optimized_vs_fused_accuracy[real-*]` — bounded regression against the
  fused decoder on the same reference.

All hold 0.995. The synthetic bar is 0.99 and documented; it is **not** an
expected-failure marker, and the two slower fallbacks are reported with numbers.

### 3.3 KV cache dtype (OPT-002), and why it nearly measured as worthless

BFP8 cache, tested against BF16 at both contexts. At 2048 it is a wash in both
directions (1.0907 vs 1.0915 sliding). At 131071 on the `full` layer, where the
decode SDPA reads the whole cache, the answer **depends on the SDPA config**:

| SDPA q/k chunk | BFP8 cache | BF16 cache | BFP8 worth |
| --- | --- | --- | --- |
| 32 / 64 (fused stage's) | 1.5584 | 1.5569 | **-0.1 %** |
| op-chosen | 1.2658 | 1.4041 | **+9.9 %** |

At the inherited fixed chunking the attention is latency-bound and halving the
cache bytes buys nothing; a reduced-cache trial measured only there would have
concluded BFP8 KV is useless. Both knobs had to move together
(`logs/layer_ab_sdpa.log`). PCC is unaffected either way (0.993733 BFP8 vs
0.993621 BF16 — BFP8 is nominally *better*, i.e. the difference is noise).

Prefill fill: `paged_fill_cache` does no dtype conversion, so the K/V fill tensors
are explicitly `ttnn.typecast`-ed to the cache dtype before the call (this is the
first stage where that line actually does work). Decode `paged_update_cache`
inputs stay BF16 and the op repacks.

### 3.4 Activation dtype

BFP8 activations were tried and are **blocked by an exact op contract**:
`nlp_create_qkv_heads_decode` accepts FLOAT32 or BFLOAT16 only
(`nlp_create_qkv_heads_decode_device_operation.cpp:41`, *"Unsupported data
format"*). Activations stay BF16, which is also what the fallback policy in
`$optimize` prescribes. Norms and the residual stream stay BF16 with the fused
stage's uplifted compute-kernel config, inherited unchanged.

### 3.5 Math fidelity, dtype held fixed

LoFi against HiFi2 and HiFi4 on the identical dtype policy, whole layer:

| fidelity | decode ms/token (sliding / full) | prefill 8192 ms | prefill PCC | decode PCC |
| --- | --- | --- | --- | --- |
| **LoFi (shipped)** | **1.0907 / 1.0598** | **44.57 / 43.12** | 0.993904 / 0.992468 | 0.993733 / 0.992232 |
| HiFi2 | 1.8425 / 1.8152 | 58.44 / 58.81 | 0.994144 / 0.992726 | 0.993880 / 0.992357 |

HiFi2 is **69 % slower in decode and 35 % slower in prefill** for +2.4e-4 of PCC.
This is the concrete case the skill warns about ("do not assume BFP8 implies HiFi2
is fastest"): the fused stage's BF16 path used HiFi2 because that is what
`ttnn.linear` picks for BF16, and carrying that forward onto BFP4/BFP8 weights
would have cost more than every other decision in this stage put together.

## 4. Decode matmul geometry

### 4.1 The isolated sweep

`bench/decode_matmul_sweep.py`, all five projection shapes at 32 rows, over every
legal `(cores, in0_block_w)` pair — `cores` must divide the K-tile count and
`in0_block_w` must divide `K / (32 * cores)` — at BFP4 and BFP8 separately, with
PCC against a float64 reference on the same weights so an op failure cannot be
mistaken for a precision result. Logs: `logs/decode_matmul_geometry_bfp{4,8}.log.gz`,
`logs/decode_matmul_dtype.log`.

Two results shaped everything:

* the interleaved auto-configured `ttnn.linear` the fused stage used reaches
  377 GB/s summed over the six projections; the explicit DRAM-sharded config
  reaches **492 GB/s at BFP8**, 96 % of the ~512 GB/s part;
* **`in0_block_w` is the dominant field, and larger is better right up to the L1
  wall.** `mlp_down` at BFP4 goes 0.5496 ms at `in0_block_w=1` -> 0.3228 at 2 ->
  0.2792 at 3 -> 0.2627 at 4 -> 0.2454 at 6 -> 0.2376 at 8 -> 0.2297 at 12 ->
  0.2284 at 13 -> 0.2265 at 16 -> **0.2242 at 24**, where 48 overflows L1. The
  core count barely matters by comparison (0.2242 at 13, 26 and 52 cores).

The L1 ceiling is dtype-scaled, which is why the shipped table is keyed on
`(role, dtype)`: `in0_block_w=26` is the fastest legal value for `wqkv` at BFP4
and *illegal* at BFP8 (1,782,400 B of static circular buffers against a 1,572,864 B
budget). A single table would have either left BFP4 8 % slower or crashed BFP8.

### 4.2 The isolated sweep is not the answer — the layer is

The isolated probe has ~1.5 MB of L1 free for circular buffers; a real decode step
has ~232 KB of live L1 tensors when the MLP runs (the residual, the carried
`hidden`, and two 19968-wide gate/up outputs). So the isolated winner for the MLP
— 13 cores, `in0_block_w=8`, 0.2300 ms — **fails in the layer at every
`in0_block_w` including 2**, with *"Statically allocated circular buffers ... clash
with L1 buffers"* (`program.cpp:1779`). 26 cores halves the per-core output to
49 KB and works. Whole-layer traced decode, sliding:

| MLP working grid | `in0_block_w` gate/up | decode ms/token | note |
| --- | --- | --- | --- |
| 13 | 8, 4, 2 | **all fail** | 98 KB/core output leaves no room for the CBs |
| 26 | 8 | **1.1227** | shipped (with the 8-core boundary of that round) |
| 26 | 4 | 1.1390 | |
| 52 | 4 | 1.1234 | |
| 16 | 13 | fail | |

and `mlp_down`, holding the rest fixed: `in0_block_w` 24 -> 1.1228, 12 -> 1.1287,
8 -> 1.1362, 4 -> 1.1617. Largest legal wins, as in the isolated sweep.

### 4.3 The boundary grid

The boundary grid carries every `hidden_size`-wide residual/norm tensor, the QKV
output, the attention output and gate, and the `o_proj` output. It must divide
`208`, `144` and `128` tiles exactly for none of them to be shard-padded, which
leaves 1, 2, 4, 8 and 16.

| boundary cores | decode ms/token (sliding / full) | note |
| --- | --- | --- |
| 4 | fail | `in0_block_w` too large for the per-core L1 at this width |
| 8 | 1.1228 / 1.0961 | fastest sharded RMSNorm per the fused stage's probe (22.8 us) |
| **16** | **1.0916 / 1.0652** | **shipped**, 2.8 % faster whole-layer |

16 cores wins even though the fused stage measured its *norm* as slower
(24.4 vs 22.8 us) and even though its 13-tile shard forces the norm's
`subblock_w` to 1: the 13-K-tile shard is exactly what lets `wqkv` and the
attention gate run at `in0_block_w=13`, and that is worth more than the norm
gives up. This is the whole-layer-versus-isolated-op rule paying off in the
opposite direction from section 4.2.

`o_proj`'s `in0_block_w` at 16 cores: 4 -> 1.0916 / 1.0652 (shipped), 8 -> 1.0928
/ 1.0660, 2 -> 1.1016 / 1.0745.

## 5. Decode SDPA

See section 3.3 for the table. Shipped: the device compute grid with
`q_chunk_size = k_chunk_size = 0`, i.e. the op chooses. The win is entirely the
chunking (1.2658 vs 1.5584 at 131071 on `full`, **19 % of the whole step**); the
core grid is worth 0.5 % (11x10 1.2658 vs 8x8 1.2720). Nothing is traded at
context 2048, where every candidate is within 0.5 %.

`exp_approx_mode=False` is inherited unchanged.

## 6. What was left alone, and why

* **`paged_fused_update_cache`.** The fused stage established the exact blocker:
  it needs K and V on disjoint cores, `nlp_create_qkv_heads_decode` emits them on
  Q's grid, and its `overlap_qk_coregrid=False` mode needs a width-sharded QKV
  whose shard width divides `head_dim` (32/64/128). This stage's QKV projection
  output *is* width-sharded — but at 4608/16 = 288 elements per shard, which does
  not divide 128. Reaching 128 would need 36 output cores, and 36 does not divide
  the 208 K-tiles the matmul's activation shard requires. The two
  `paged_update_cache` dispatches are 7.17 us of a 1072 us step (0.67 %), and the
  fused stage measured the fused variant as a wash even where it was legal.
  Recorded, not pursued.
* **The prefill SDPA chunk (256).** The fusing stage swept it over 90
  measurements across 9 lengths and both kinds, and the constraint set is
  unchanged by anything this stage does.
* **Multi-chunk prefill data movement.** Inherited from the functional stage,
  which owns the chunking contract. Still disclosed in the README.
* **The two per-head QK norms on one core.** `ttnn.rms_norm` rejects
  height-sharded inputs (`layernorm_device_operation.cpp:166`); 7.42 us, 0.69 % of a step
  and unchanged from the fused stage.

## 7. Packed same-input projections: measured and rejected

`bench/variants.py` implements both, and `bench/layer_ab.py` measures them under
the same harness as the shipped layer.

### 7.1 QKV + attention gate (OPT-001)

Both consume the `input_layernorm` output. The packed weight is
`[6656, 4608 + 4096]`; the split happens after the `sharded_to_interleaved` the
QKV head-creation op needs anyway, so it costs two slices and one reshard.

At the **matmul level** across every legal geometry
(`logs/decode_matmul_geometry_packed.log.gz`): best packed 0.1274 ms (13 cores,
`in0_block_w=4`) against 0.1304 for the best split pair — packed wins by 3 us in
isolation, but only on a 13-core grid that the layer cannot use (see 4.3, and
`o_proj` cannot run on 13 cores at all because 13 does not divide 128 tiles). On
the boundary grid the doubled output width pushes the largest legal
`in0_block_w` from 13 down to 2, and packed is 0.1345 against 0.1326.

**Whole layer**: 1.1298 / 1.1073 ms/token against the shipped 1.1228 / 1.0961 —
**0.6 % slower**. Rejected on measurement, at the strongest geometry that
compiles in the layer.

### 7.2 MLP gate + up (OPT-010)

Packed weight `[6656, 2 x 19968]`. The 39936-wide output caps `in0_block_w` at 2
at every legal core count, so the packed matmul is 0.4851 ms against 0.4600 for
the two separate dispatches *before* the split. Whole layer: 1.1517 / 1.1248
against 1.1228 / 1.0961 — **2.6 % slower**. Rejected.

Both rejections are the same mechanism, and it is the one OPT-010 names: a wider
output forces a worse `in0_block_w`, and on this part `in0_block_w` is the field
that matters most (section 4.1).

## 8. Prefill blocking

`ttnn.linear` is gone from prefill (the width-sharded weight is illegal for it),
so `minimal_matmul`'s own `M=K=N=8` default now has to cover every row count —
and it is weak at the short ones. `bench/minimal_matmul_block_sweep.py` swept
`M_block x K_block x N_block` per shape at 128 / 512 / 2048 / 4096 / 8192 rows,
at both dtypes: `logs/mm_block_sweep_bfp{4,8}.log`, ~1400 measurements.

Gains over the op default, best per cell: +2 % to +27 % on the attention
projections, +13 % to +17 % on gate/up, +2 % to +20 % on `mlp_down`. The pattern
is consistent across dtypes: **large `N_block` (16-32) is what matters**, and
`M_block` should track the row count (2 at 128-512 rows, 8 from 2048 up). Four
(shape, dtype, rows) cells are fastest on the op's own choice and get an explicit
`None`.

### 8.1 The short-prefill gap, and why it stopped being a limitation

`prefill_matmul_sweep.py` measured the honest cost of losing the `ttnn.linear`
branch. Summed over the six dispatches, against the fused stage's actual dispatch:

| rows | fused (`ttnn.linear` BF16) | optimized (op default) | with the swept blocking |
| --- | --- | --- | --- |
| 32 | 2.55 ms | 3.80 ms | **0.85 ms** (DRAM-sharded matmul) |
| 128 | 2.67 ms | 3.39 ms | ~2.98 ms |
| 512 | 3.93 ms | 3.98 ms | ~3.5 ms |
| 1024 | 5.71 ms | 4.49 ms | faster |
| 2048+ | 11.70 ms | 7.31 ms | faster still |

The first pass of this stage shipped that table and disclosed the 64-512 row band
as README limitation 4: ~10 % slower per projection than the fused decoder, with
"no way to remove it without a second interleaved weight copy."

**That was wrong, and section 11 is what replaced it.** The premise -- that
`ttnn.linear` cannot read a width-sharded weight -- came from one error message
emitted by the *auto-selected fallback* program config, not from the op. With an
explicit 2D-multicast config the same weight is legal, and the band that was 10 %
slower is now 1.3-2.0x faster per projection and 1.37x faster whole-layer. The
row above is kept because it is still the correct comparison for the
`minimal_matmul`-only dispatch, which is what the >= 2048-row band still uses.

## 9. Final default confirmation

Re-measured on the shipped default path after every knob was frozen, in the same
harness as the candidates, min of 3 rounds (`logs/layer_ab_final_2048.log`,
`logs/layer_ab_final_131071.log`, `logs/layer_ab_short_prefill.log`):

| kind | window | fused | optimized | speedup |
| --- | --- | --- | --- | --- |
| sliding | traced decode @ 2048 | 2.7345 | **1.0909** | 2.51x |
| sliding | traced decode @ 131071 | 2.7340 | **1.0891** | 2.51x |
| full | traced decode @ 2048 | 2.7089 | **1.0601** | 2.56x |
| full | traced decode @ 131071 | 3.2048 | **1.2657** | 2.53x |
| sliding | prefill 128 | 3.49 | **2.18** | 1.60x |
| sliding | prefill 256 | 3.93 | **2.48** | 1.58x |
| sliding | prefill 512 | 5.18 | **3.31** | 1.56x |
| sliding | prefill 1024 | 7.44 | **5.32** | 1.40x |
| sliding | prefill 8192 | 64.05 | **44.30** | 1.45x |
| full | prefill 128 | 3.47 | **2.13** | 1.63x |
| full | prefill 8192 | 64.65 | **43.40** | 1.49x |

No candidate in this stage beat these on traced decode while clearing the
real-weight PCC bar.

## 10. Evidence chain

```bash
D=models/autoports/meta_models_muse_glimmer_30b/doc/optimized_decoder
# correctness (the acceptance gate)
python -m pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_optimized_decoder.py -q
# device-time profiles, advice enabled, one job at a time
bash $D/bench/run_tracy.sh
# watcher, in a separate run
bash $D/bench/run_watcher.sh
# the capability contract
python $D/bench/refresh_context_contract.py
```

Probes, each regenerable:

| probe | question |
| --- | --- |
| `bench/weight_layout_probe.py` | which weight layouts each matmul accepts |
| `bench/decode_matmul_sweep.py` | dtype x fidelity x (cores, `in0_block_w`) per decode projection |
| `bench/prefill_matmul_sweep.py` | prefill kernel/dtype per row count, against the fused baseline |
| `bench/short_prefill_probe.py` | where the DRAM-sharded matmul stops being legal (`M == 1`) |
| `bench/minimal_matmul_block_sweep.py` | `MinimalMatmulConfig` per shape, dtype and row count |
| `bench/bfp_block_range_probe.py` | why synthetic and real weights disagree under BFP4 (three refutations) |
| `bench/layer_ab.py` | whole-layer candidate ranking: precision, geometry, SDPA, packing |
| `bench/variants.py` | the rejected packed-projection and old-SDPA layers |
| `bench/prefill_mcast_probe.py` | 2D-multicast vs `minimal_matmul` per shape and row count; `--repro` for the `grid_x != dram_banks` silent-`inf` bug (section 11, 12.1) |
| `bench/short_prefill_layout_probe.py` | the last two `tt-perf-report` advice items at 128 rows (section 13) |
| `bench/sharded_norm_grid_probe.py` | whether a sharded `rms_norm` is correct when its program grid exceeds its shard grid, whether the shipped decode norm is affected (section 12.2), and (`--rect`) the exact-rectangle band the prefill norm ships on (section 16.3) |
| `bench/decode_elementwise_probe.py` | where the decode SwiGLU / attention-gate multiplies spend their time (sections 16.2, 17.2) |
| `bench/layer_ab.py --candidates mlp_bfp4,fused_act` | the `fused_activation` whole-layer A/B (section 17.1) |


## 11. Correcting section 1.1: `ttnn.linear` *can* read the width-sharded weight

This is the largest single finding of the stage, and it is a correction to its own
earlier conclusion rather than a new idea.

Section 1.1 recorded "can `ttnn.linear` read a width-sharded weight? **no**", on
the strength of one error message:

```
MatmulMultiCoreProgramConfig: Input B memory layout must be INTERLEAVED
```

`$optimize` is explicit that a first API error is not a rejection, and this is a
textbook case of why: that message comes from the *auto-selected fallback* program
config. Reading `ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp`
instead of the error:

| line | what it says |
| --- | --- |
| `:1541-1553` | `validate_matmul_mcast2d_config` accepts `WIDTH_SHARDED` `input_tensor_b` **in DRAM** |
| `:1525` | the extra "`per_core_N` must equal the in1 shard width" clause is gated on `buffer_type() != DRAM`, so it does not bind this weight |
| `:1543-1550` | the one width-shard clause that does bind is that the in1 shard grid's bounding box is one row tall -- which the 8-DRAM-bank weight already is |

So an **explicit** 2D-multicast program config reads exactly the tensor this stage
already ships, with DRAM-interleaved activations and output, which is the prefill
contract. `bench/prefill_mcast_probe.py` measured it against `minimal_matmul` with
its swept blocking, per projection shape, at the shipped dtype: 182 measurements in
`logs/prefill_mcast_probe.log` plus 339 in
`logs/prefill_mcast_probe_bigrows.log`.

| role | 64 r | 128 r | 256 r | 512 r | 1024 r | 2048 r | 8192 r |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `wqkv` | 1.63x | 1.64x | 1.61x | 1.48x | 1.75x | 0.95x | 0.68x |
| `attn_gate` | 1.43x | 1.47x | 1.44x | 1.31x | 1.74x | 0.93x | 0.76x |
| `o_proj` | 1.49x | 1.46x | 1.40x | 1.30x | 1.46x | 0.93x | 0.68x |
| `mlp_gate`/`up` | 2.00x | 1.95x | 1.82x | 1.54x | 0.87x | 0.75x | 0.67x |
| `mlp_down` | 1.59x | 1.57x | 1.53x | 1.49x | 0.88x | 0.99x | 0.77x |

Whole-layer, against the fused decoder in the same harness
(`logs/layer_ab_short_prefill.log`): 128 rows 3.51 -> 2.57 ms, 512 rows
5.15 -> 3.34, 1024 rows 7.53 -> 5.30. The band the first pass disclosed as a 10 %
regression is now a 1.37-1.54x win.

The `>= 2048` columns are measured, not assumed: `out_block_h`/`out_block_w`
bounding is what makes those candidates legal at all (they otherwise overflow the
1.5 MB static circular-buffer budget), and they are still 5-33 % slower, because
by then the matmul is compute-bound and the 2D-multicast path is pinned to 8 core
columns against `minimal_matmul`'s 110 cores. So the shipped dispatch is three
kernels by row count, and `PREFILL_MCAST2D` hands the large rows back.

### 11.1 The band bound has to be an upper bound, and finding that out cost a run

The first version of `PREFILL_MCAST2D` was keyed the way
`PREFILL_MINIMAL_BLOCKS` is -- `(min_rows, spec)`, descending. That is wrong for
this table, because `grid_y` fixes `per_core_M = ceil(rows / 32 / grid_y)`, which
sizes the L1 output block. A lower-bound band applies the `grid_y` measured at
1024 rows to *any* larger row count.

It surfaced immediately and unambiguously: a batched prefill's per-user 2000-token
prompt pads to 2016 rows, took the 1024-row band's `grid_y = 8`, asked for
`per_core_M = 8`, and threw

```
Statically allocated circular buffers on core range [0-0 - 7-7] grow to 1966976 B
which is beyond max L1 size of 1572864 B
```

out of six otherwise-passing tests (`test_batched_prefill_decode_pcc[4/13/32-*]`
and `test_real_weights_traced_decode_and_batch[*]`). Rekeyed to ascending
`(max_rows, spec)`, `per_core_M` is at most 4 at every legal row count, and
`test_prefill_mcast_table_is_legal` asserts that at each band's worst-case row
count -- which is what makes an arbitrary logical prefill length *safe* rather than
merely untested.

## 12. Two silent TTNN miscomputes, both now pinned by a test

(A third TTNN finding -- the DRAM-sharded matmul ignoring the output shard grid it
is handed -- came out of the round-2 review; see section 16.4.)

Neither raises. Both were found by gating every probe candidate on a
finite-output + PCC check before believing its latency.

### 12.1 2D-multicast + width-sharded DRAM in1 requires `grid_x == dram_banks`

`bench/prefill_mcast_probe.py --repro` -> `logs/mcast_gx_bug_repro.log`. At
`grid_x` of 9 or 11 against 8 DRAM banks the op validates, launches, and returns
`inf`; the same grids are correct with a DRAM-*interleaved* in1, which isolates it
to the width-sharded in1 reader assigning core column `j` to weight shard `j` and
running past the end of the shard set. `test_prefill_mcast_table_is_legal` asserts
the layer can never build such a config, on the table *and* on the real dispatch.

### 12.2 Sharded `ttnn.rms_norm` requires program grid == shard grid above `block_h = 1`

`bench/sharded_norm_grid_probe.py` -> `logs/sharded_norm_grid_probe.log`. At
`block_h = 4` (128 prefill rows), 16 shards under an `11x2 = 22`-core program grid
returns 75,155 non-finite elements; 13 under 22 returns 13,222; 26 under 33 returns
77,465; 52 under 55 returns finite but wrong output (`max|diff| = 1.94`).

The reason this mattered enough to isolate: **the shipped decode norm uses exactly
that mismatched shape** -- a 16-core width shard under an `11x2` program grid. At
`block_h = 1`, which is what a 32-row decode step is, every core count agrees with
the exact-grid case to `max|diff| = 0.03182` (BF16 rounding against float64), so
decode is correct. The probe is what proves that rather than inferring it from a
passing PCC dominated by BFP4 error.

## 13. The last two `tt-perf-report` advice items

Both from the 128-row prefill window, measured by
`bench/short_prefill_layout_probe.py` -> `logs/short_prefill_layout_probe.log`.

**"If possible place input 0 in L1."** Legal -- the 2D-multicast validator accepts
a `BLOCK_SHARDED` in0 -- and **slower**: 0.92x on `wqkv`, 0.97x on `mlp_gate` and
`mlp_down` at 128 rows, because the activation is ~2 % of the bytes the matmul
moves (1.7 MB against 74.8 MB of BFP4 gate/up weight) and the
`interleaved_to_sharded` costs more than it saves. Rejected with before/after.

**The core-starved prefill RMSNorm** (not advice, but the largest non-matmul row in
that window: ~134 μs each on 4 cores, 21 % of the window). A width-sharded L1 norm
does it in 45.5 μs including both conversions, **2.98x**, at 8 cores and 128 rows.

This section originally rejected it, on the grounds that the only correct core
counts were `{1, 2, 4, 8}` and none of them was L1-legal above 128 rows. **That was
wrong and it is now shipped** -- the constraint came from the probe harness hard-coding
an 11-wide shard prefix, not from TTNN. 16 cores laid out as an exact `8x2`
rectangle is both legal and the fastest point. See section 16.3.

## 14. Test-suite corrections made this session

Two real defects in the delivered tests, both found by running the full suite
rather than a subset:

1. `test_optimized_vs_fused_accuracy` called `.split()` on the second element of
   `comp_pcc`'s return value. `models/common/utility_functions.py:568` returns
   `(bool, float)`, not `(bool, str)`, so the test raised
   `AttributeError: 'float' object has no attribute 'split'` on its first
   parametrisation. Fixed to use the float directly. The earlier run had stopped at
   this test, which is why its four cases and the 37 tests after it had never
   executed.
2. With those cases running, `FUSED_REGRESSION_TOL` was too tight in both
   populations: the synthetic `full` prefill delta is 7.316e-3 against a 7e-3
   bound, and the real `full` decode delta is 2.996e-3 against 3e-3 -- passing by
   4e-6. Both bounds are now the measured worst delta plus ~35 % margin
   (`synthetic` 1.0e-2, `real` 4.0e-3), with the measured values recorded in the
   constant's docstring, and the docstring says plainly that these are diagnostic
   bounds while the acceptance gate is the absolute 0.995 real-weight bar.

## 15. Final state (superseded; the current state is section 18.3)

* 110 tests, 110 passed; 228 asserted HF-vs-TTNN PCC checks; worst real-weight
  0.995079 against the 0.995 bar, worst synthetic 0.990467 against 0.99.
* Ten Tracy windows (prefill 128 / 8192 / 16384, traced decode 2048 / 131071, both
  layer kinds), all free of dropped markers, with advice enabled.
* Watcher clean over 26 node ids: zero detections across 36,660 lines, 68 dumps.
* `doc/context_contract.json` regenerated by
  `bench/refresh_context_contract.py`; `--check` and
  `bench/summarize_pcc.py --check` both clean.
* Device healthy before and after: `MESH_SMOKE_OK Arch.BLACKHOLE 11-10 8-1 4`. No
  hangs, no ARC/ERISC/Ethernet faults, no resets, no `tt-triage` capture needed,
  and `$autofix` was not required -- both failures this session (the `comp_pcc`
  return type and the L1 overflow from a lower-bound band) had unambiguous single
  causes visible in the first traceback.


## 16. Stage-review round 2: four findings, four different outcomes

An independent `$stage-review` returned `more-work-needed` on the state above. All
four required items were real; none resolved the way the review predicted, which is
the useful part.

### 16.1 P1 - a 16.78 MB host upload inside the measured prefill path. **Fixed.**

The reviewer traced the 2015.9 us op-to-op gap in the two-chunk sliding prefill
window to `ttnn.zeros(..., device=...)` at each internal chunk boundary, and was
right about the mechanism: `ttnn::creation_detail::full_impl`
(`ttnn/cpp/ttnn/operations/creation/creation.cpp:51-73`) fills a host `std::vector`
and uploads it. 16,777,216 B / 2.0159 ms = 8.3 GB/s is a PCIe write, and the `full`
window - same Python chunk loop, no filler needed - had 33.8 us total.

`OptimizedDecoder._prefill_sdpa_sliding` now builds the filler once per
`(tail_len, dtype)` and keeps it. Worst gap in that window: **0.645 us**; window
total 2051 -> **36 us**; device time 81.962 -> 81.796 ms. The reviewer was also
right that the README mis-explained it as caller-side Python loop overhead, and that
`test_no_host_fallback_in_forward` structurally cannot see a C++-side host buffer
creation. Both corrected.

### 16.2 P1 - the elementwise regression. **Investigated; three candidates, all lose.**

The reviewer correctly caught that the audit's "already sharded and
activation-folded / unchanged" row was stale: the two activation-folded multiplies
went 24.21 -> 58.2 us/step, because they moved from 110 DRAM-interleaved cores onto
16- and 26-core shards. The proposed fix - fold the activations into the matmuls via
`fused_activation` - is legal and was untried.

Tried, whole-layer: **4.4 % / 4.6 % slower** (1.1392 / 1.1085 against 1.0909 /
1.0601), with bit-identical output PCC. The matmul's `SFPU_ACTIVATION` runs on its
12 worker cores, fixed to the DRAM bank count, interleaved with the unpack it is
already bottlenecked on.

`bench/decode_elementwise_probe.py` then refuted the premise: the transcendental is
**not** the cost. Removing it entirely takes the SwiGLU row 41.76 -> 30.02 us and
the attention-gate row 29.70 -> 29.07 us - 0.6 us. The rest is the multiply on a
narrow shard, and widening the shard is much worse (reshard to 52 cores 80.44 us, to
104 cores 109.94 us) because moving a 19968-wide tensor twice costs more than the
multiply saves. The ops are at their floor under this layout contract, and the
+34 us is bought back many times by the 2.5x the same contract gives the matmuls.

### 16.3 P2 - the sharded prefill norm rejection. **Unearned; now shipped.**

The reviewer was right, and the reason is worth recording. Both probes hard-coded
the shard core set to a row-major prefix of the **11**-wide device grid, so a
16-core shard was always built as `11 + 5` under an `11x2` program grid - the
corrupting shape. On an `8x10` grid, 16 cores is an exact `8x2` rectangle and 16
divides the 208 hidden-size tiles, so the intersection the report called empty
(`{1, 2, 4, 8}`) was wrong: 16 belongs in it, and it is the fastest point.

Re-measured with exact rectangles (`logs/sharded_norm_grid_probe_rect.log`):

| rows | `block_h` | 16 c (`8x2`) | 8 c | interleaved |
| --- | --- | --- | --- | --- |
| 32 | 1 | correct | correct | -- |
| 128 | 4 | **33.0 us** | 44.1 us | 135.8 us |
| 256 | 8 | **57.6 us** | L1 | ~136 us |
| 512 | 16 | CB overflow | L1 | 135.9 us |

Shipped for `rows <= 256` on 16 cores. Whole-layer: 128-token prefill 2.57 ->
**2.18 ms**, 256-token 2.48 ms; device time for that window 2549 -> **2140 us**.
Against the fused decoder, 1.37x -> **1.60x**. Decode untouched (1.0909 / 1.0601
reproduced exactly).

### 16.4 P2 - the decode norm's program grid wider than its shard. **Forced by an op; now guarded.**

The reviewer proposed making the decode boundary shard a rectangle too. It cannot
be, and finding out why produced the stage's third TTNN finding: the DRAM-sharded
matmul **ignores the output shard grid it is handed**. Asking for a 16-core `8x2`
output returns `{[0-0 - 10-0], [0-1 - 4-1]}`, the row-major prefix of the compute
grid. Every decode boundary tensor comes out of that matmul, so the prefix is not
this layer's choice and the norms must consume it.

That makes the shipped shape correct but only at `block_h == 1`, which a decode step
always is. So the resolution is the reviewer's second option:
`_decode_norm_configs` raises rather than construct the unsafe combination, and
`test_decode_norm_refuses_the_silently_corrupting_shape` pins it. `core_range_set`
keeps the prefix with the op contract documented at the call site;
`rect_width_sharded_l1` exists for the prefill norms, which shard their own input.

### 16.5 Smaller review items

* the `DECODE_MATMUL` dtype-keying rationale overstated the case - the shipped BFP4
  and BFP8 `wqkv`/`attn_gate` entries are identical and the real justification is
  the MLP rows; prose corrected;
* `tested.continuation_prefill.command` pointed at `test_fused_decoder.py`;
  retargeted;
* stray `__pycache__` removed from the artifact tree (any probe run regenerates it; it is not committed);
* the e2e-minus-device claim now says explicitly that it is a cross-run subtraction
  and why (the profiler inflates dispatch gaps: 56 us/replay in the profiled window
  against the 19 us the unprofiled e2e allows).

### 16.6 Final state after round 2 (superseded; current state is 18.3)

113 tests, 113 passed. 228 asserted PCC checks, worst real-weight 0.995079, worst
synthetic 0.990467 - unchanged, because the row counts that changed are not the ones
that set those minima. Ten Tracy windows re-taken against the final code, zero
dropped markers. Watcher clean over 26 node ids: zero detections across 21,602 lines
and 40 dumps. `refresh_context_contract.py --check` and `summarize_pcc.py --check`
both clean. Device healthy after the stage: `MESH_SMOKE_OK Arch.BLACKHOLE 11-10 8-1 4`.


## 17. Stage-review round 3: four more findings, all evidence-quality

Round 3 returned `more-work-needed` on four items, and the useful thing about them
is that none was a wrong *decision* -- all four were evidence that did not support
the decision it was attached to.

### 17.1 P1 - the `fused_activation` A/B existed only as prose. **Fixed.**

The numbers rejecting the candidate were real measurements, but they had been taken
by hand-editing `DECODE_FUSED_ACTIVATION` and reading the terminal, so nothing in
the tree could reproduce them and `bench/layer_ab.py` had no path that could even
build the candidate. `KWARG_CANDIDATES` now carries `fused_act`, and both rows come
from one committed run (`logs/layer_ab_fused_activation.log`):

| candidate | sliding | full |
| --- | --- | --- |
| **`mlp_bfp4` (shipped)** | **1.0908** | **1.0602** |
| `fused_act` | 1.1393 (+4.4 %) | 1.1082 (+4.5 %) |

with prefill/decode PCC identical to six decimals on both rows, which is the
bit-identical claim the earlier prose made without a source.

### 17.2 P2 - the elementwise probe measured the wrong activation. **Fixed.**

`bench/decode_elementwise_probe.py` applied `input_tensor_a_activations=[SILU]` to
*both* cases. The shipped attention gate is `out * sigmoid(gate)` --
`input_tensor_b_activations=[SIGMOID]`, a different unary on the other operand -- so
the row used to argue "the transcendental costs 0.6 us there" was not that op. Each
case now uses the unary and operand the layer folds, with a matching float64
reference. The conclusion survives (sigmoid costs **0.44 us**, SiLU ~12 us), but it
now comes from the right op.

The reviewer also noted the probe is host-wall-clock around untraced dispatches, so
the attention-gate rows sit near a ~29 us launch floor where the committed device row
for that op is 14 us. That is now stated in the probe docstring and in the README:
differences between rows at one shape are the result, absolute values are not
comparable to a perf-report row.

### 17.3 P2 - the Q-filler cache was unbounded. **Fixed.**

The round-2 fix cached one filler per `(tail_len, dtype)` and the docstring claimed
"at most two live". `tail_len` is `min(window, start_pos)` -- caller-controlled -- so
a decoder reused across continuation prefills at different offsets could accumulate
one entry per tile-aligned offset: 64 entries, up to ~545 MB per sliding layer, and
no test in this suite reaches enough distinct offsets to see it.

Now there is exactly **one** buffer, at the full window length
(`32 * 2048 * 128 * 2 B = 16,777,216 B`), and a shorter tail slices it. That is
recorded in `doc/context_contract.json` under
`implementation.extra_persistent_buffers`, which the round-2 fix had not updated.

### 17.4 P2 - the new norm band had no test at its edge. **Fixed.**

The band is `rows <= 256` and 256 is where L1 is tightest (512 overflows the CB
budget by 33 %), but the inherited length list jumps 128 -> 2048, so neither the
worst case inside the band nor the first case outside it was ever dispatched.
`test_prefill_pcc_across_the_norm_shard_band` now covers 224 / 256 / 288 / 320 on
both layer kinds and **both** weight populations -- 16 cases, worst real-weight PCC
0.996907. 224 and 288 are deliberately not powers of two, since the branch keys on a
row count and a serving prompt produces arbitrary tile-aligned lengths.

### 17.5 Smaller items

* the new 128-row window's advice item -- 24 us + 15 us op-to-op gaps on the first
  two ops after the signpost, 1.3 % of the window -- is now in the advice ledger,
  classified as untraced first-dispatch cost (every later op in the same window is
  0-1 us) rather than left unlisted;
* the round-2 review independently confirmed from tt-metal source
  (`matmul_device_operation.cpp:2477-2483`, `num_cores_to_corerangeset(..., row_wise=true)`)
  that the DRAM-sharded matmul rebuilds the output shard spec and keeps only the
  caller's memory layout and buffer type, which is the claim section 16.4 rests on.

### 17.6 Final state after round 3 (superseded; current state is 18.3)

129 tests, 129 passed. 244 asserted PCC checks: 38 real-weight (worst 0.995079,
unchanged -- the new cases are all above it) and 206 synthetic (worst 0.990467).
Ten Tracy windows re-taken against the final code, zero dropped markers; the
two-chunk sliding prefill's worst op-to-op gap is 0.610 us. Watcher clean over 26
node ids: zero detections across 21,559 lines, 40 dumps.
`refresh_context_contract.py --check` and `summarize_pcc.py --check` both clean.


## 18. Stage-review round 4: stale figures, and one number that was two numbers

Round 3's four fixes all verified from artifacts, and the reviewer independently
re-derived every headline number. What it found instead was drift: figures the
round-3 fixes had themselves invalidated.

* **"30 real-weight checks" in three places.** The band test added exactly 8 real
  checks, taking the population to 38. `refresh_context_contract.py --check` passed
  because it regenerates `tests.*` but not prose, so the contract's own note carried
  a stale count. The note no longer carries a number at all -- it points at
  `tests.real_weight_checks`, which is regenerated -- and the two README sites say 38.
* **A checklist row still claiming 110/110** against a 129-test junit. Fixed.
* **Two transcription drifts**: the worst prefill op-to-op gap quoted as 0.645 us in
  one place and 0.610 in three others (0.610 is what the committed CSV says), and
  sliding decode @131071 device time quoted as 1.070 against 1.071 in the CSV and
  the contract. Both re-transcribed from the CSVs.
* **8 versus 12.** The report used "the DRAM bank count" for both the shardable DRAM
  grid width (`dram_grid_size().x = 8`, which sets the weight's shard count and the
  prefill 2D-multicast `grid_x`) and the decode matmul's worker-core count (the perf
  rows show 12, from `get_optimal_dram_bank_to_logical_worker_assignment`). They are
  different quantities and limitation 2 depends on the second one, so the report now
  says which is which and notes that `num_dram_channels()` is not exposed through
  the Python API on this build -- the 12 is the measured row plus the source path,
  not a derived figure. The *decision* was already safe: `logs/decode_matmul_dtype.log`
  shows the many-core DRAM-interleaved BFP4 alternative losing 0.3303 vs 0.2300 ms.

Also from that round: watcher coverage now includes the `full` stress soak and the
256/288-row sharded-norm cases (30 node ids, was 26); the band test now asserts
*which* norm path each row count took, not just its PCC, so 256 and 288 cannot both
silently fall back to interleaved; `build_optimized`'s cache key includes kwarg
values, not just names; the decode audit row's "all under 0.1 % individually" is
corrected with the two ops that are not; and `run_tracy.sh`'s header says ten
windows rather than eight.

### 18.2 Round 5: sourcing the last unsourced numbers, and closing the class

Round 4's review confirmed every engineering claim re-derives from artifacts and
found only documentation drift -- the fourth consecutive round to find that same
class and nothing else. So this round fixed the instances *and* the class.

Instances:

* **the precision-frontier table's fused baseline row** was the one row in that
  table with no committed source, and its PCC columns were the *synthetic* fused
  numbers sitting in a table headed "real checkpoint". The whole five-candidate
  frontier was re-measured on real weights in one run
  (`logs/layer_ab_real_final.log`, every row `AB[real]`), so the table is now one
  harness state and one population. The conclusion is unchanged: the shipped policy
  is the fastest point clearing the bar, and all-BFP4 still fails at 0.977/0.980.
* **the advice-ledger row for the 128-row window** quoted 3 of its 4 figures wrong
  and folded two windows into one set of numbers; it now cites `sliding` and `full`
  separately, and the `full` window's larger 36 us / 1.7 % item is disclosed.
* **four smaller figures** re-derived from the CSVs: six decode norms 42 -> 40.6 us,
  `paged_update_cache` 8 -> 7.17 us, the three BFP8 DRAM rows to their per-replay
  means, and the `DECODE_FUSED_ACTIVATION` docstring to the committed A/B log
  (1.0908/1.0602 vs 1.1393/1.1082) -- that docstring also claimed a *real-weight*
  PCC identity for a run that is synthetic, which is now stated correctly.
* **a `DECODE_MATMUL` docstring left over from when `BOUNDARY_CORES` was 8**, which
  claimed BFP8's MLP "needs no reshard at all" directly above a dict where every
  BFP8 MLP entry is the 26-core working shard. Replaced with what the table does and
  does not say: the attention `cores` column is pinned by the layer, not chosen by
  the sweep, and the isolated sweep's own winner for `wqkv`/BFP8 is 13 cores.

The class: `bench/check_reported_figures.py` re-derives 34 quoted figures from the
committed CSVs, junit and logs and fails on drift -- device times, op counts, the
decode op-group breakdown, the worst prefill gap, test and PCC-population counts,
both `fused_activation` A/B rows, and the watcher node-id count. Neither existing
`--check` covered any of them: `refresh_context_contract.py` regenerates `tests.*`
and the PCC blocks but not `performance` or prose, and `summarize_pcc.py` is
PCC-only. It is in the evidence chain in the README.

### 18.3 Final state

129 tests, 129 passed. 244 asserted PCC checks: 38 real-weight (worst 0.995079) and
206 synthetic (worst 0.990467). Ten Tracy windows, zero dropped markers. Watcher
clean over 30 node ids: zero detections across 23,720 lines, 44 dumps.
`refresh_context_contract.py --check` and `summarize_pcc.py --check` clean.
