# Fused decoder work log — `meta-models/Muse-Glimmer-30B`

Date: 2026-08-11. Host `tt-quietbox`, 4 x Blackhole visible, stage run on a
1x1 mesh. Repo `/home/ttuser/dev/muse-glimmer/tt-metal`, branch
`agentic-research/hous/muse-glimmer-30b`. Python env
`/home/ttuser/dev/muse-glimmer/muse-glimmer_pyenv` (transformers 5.15.0,
tt-perf-report 1.2.8).

Stage input: the completed functional decoder
(`tt/functional_decoder.py`, `doc/functional_decoder/`). Stage output:
`tt/fused_decoder.py`, `tests/test_fused_decoder.py`, this directory.

## 0. Device health

`tt-smi` is not installed on this host, so the bounded list/reset/list sequence
from `$tt-device-usage` is not available; the TTNN open/close mesh smoke was
used instead, before and after the stage:

```bash
python -c "import ttnn; m=ttnn.open_mesh_device(ttnn.MeshShape(1,1), trace_region_size=0); \
           print('MESH_SMOKE_OK', m.arch(), m.compute_with_storage_grid_size(), ttnn.get_num_devices()); \
           ttnn.close_mesh_device(m)"
# MESH_SMOKE_OK Arch.BLACKHOLE 11-10 4
```

No hangs, no ARC/ERISC/Ethernet faults, no resets and no `tt-triage` capture
were needed during this stage. Every device-facing command was run one at a
time; the watcher run and the Tracy runs are separate invocations.

## 1. Method

`$graph-fusing`, iterated: write out the op table from the previous stage's
`tt-perf-report`, classify every subgraph against the skill's three rewrite
kinds plus whatever the tt-metal op library actually offers, apply the most
promising candidate, PCC-check it on device, measure it, then go back to the
table. Screening used `bench/ab_latency.py` (wall-clock A/B of warmed prefill
and traced decode against the *functional* decoder in the same process); every
accepted change was then confirmed with a Tracy device-profiler capture and
`tt-perf-report`.

Op-library exploration (skill step 1) covered
`ttnn/cpp/ttnn/operations/**`, `ttnn/ttnn/operations/*.py`,
`models/common/modules/{attention,rmsnorm,rope,mlp}`,
`models/tt_transformers/tt/{attention,model_config,rope}.py`,
`models/tt_dit/utils/tensor.py` and `models/demos/gemma4`.

## 2. The starting op table (functional decoder, sliding kind)

From `doc/functional_decoder/tracy/sliding/{prefill,decode}_perf_report.txt`.

Prefill, 8192 tokens, 42 device ops, 101.23 ms:

| op(s) | device time | note |
| --- | --- | --- |
| 3 x `MatmulDeviceOperation` (MLP) | 66.27 ms | all `SLOW`, ~96 TFLOPs, DRAM 6 % |
| `SDPAOperation` | 11.84 ms | `q_chunk == k_chunk == 128` |
| `MatmulDeviceOperation` o_proj | 4.67 ms | `SLOW`, 96 TFLOPs |
| RoPE: 2 x (slice, slice, neg, concat, mul, mul, add) + 2 x tilize | 2.39 ms | spelled-out rotate-half |
| `MatmulDeviceOperation` wqkv / attn-gate | 4.53 ms | |
| 6 x `LayerNormDeviceOperation` | 4.11 ms | 110 cores |
| `UnaryDeviceOperation` silu / sigmoid | 2.70 ms | |
| 2 x `BinaryNgDeviceOperation` residual add | 1.37 ms | |
| everything else | 3.3 ms | heads split/concat, paged fill, gating mul |

Traced decode, batch 1, context 2048, 64 device ops, 3.163 ms/token:

| op(s) | device time | note |
| --- | --- | --- |
| 6 x `MatmulDeviceOperation` | 2.533 ms | DRAM-bound, 385 GB/s, 75 % |
| 4 x `LayerNormDeviceOperation` (hidden-size) | 0.440 ms | **1 core each** |
| `SdpaDecodeDeviceOperation` | 0.036 ms | |
| RoPE table gather: 2 x embedding, 2 x tilize-with-val-padding, 2 x transpose, 4 x untilize, 4 x repeat, 2 x tilize | 0.049 ms | 16 ops to broadcast cos/sin |
| RoPE apply: 2 x (slice, slice, neg, concat, mul, mul, add) | 0.022 ms | 14 ops |
| everything else | 0.083 ms | head split/concat, paged update, gating |

Two things jump out and drove the whole stage: **prefill is dominated by four
matmuls the auto-selected `ttnn.linear` program config runs at ~40 % of the
throughput the QKV matmul reaches on the same dtype**, and **decode spends 14 %
in four RMSNorms that land on a single core** because the interleaved LayerNorm
kernel parallelises over rows and a decode step is one tile-row.

## 3. Rewrites applied

Priority order per the skill: dedicated fused ops, then graph rewrites, then op
merging.

### 3.1 Dedicated fused ops

| # | rewrite | evidence |
| --- | --- | --- |
| A | RoPE `slice,slice,neg,concat,mul,mul,add` (x Q,K) -> `ttnn.experimental.rotary_embedding_hf` | in-graph 2.39 -> 0.78 ms; isolated 0.301 -> 0.114 ms at 1024x32x128, PCC 0.999996 vs a torch rotate-half reference (`logs/op_merge_probes.log`) |
| B | `ttnn.linear` -> `ttnn.experimental.minimal_matmul` for the dense projections at prefill row counts | `logs/minimal_matmul_sweep.log`, `logs/prefill_matmul_probe.log`: 1.11-2.36x on all five projections at the 8192-token internal chunk, and *better* PCC at the same math fidelity (see 3.4) |
| C | decode RMSNorm -> width-sharded multi-core `ttnn.rms_norm` program config | `logs/norm_shard_probe.log`: 134.4 -> 22.8 us/call wall (min of 3 x 200 calls, including both reshards); in the real graph 109.9-110.0 -> 12.3-13.4 us device per hidden-size norm |

**A. RoPE.** Three candidate ops exist. `ttnn.experimental.rotary_embedding`
takes cos/sin but no per-user position tensor in the form decode needs;
`ttnn.experimental.rotary_embedding_llama` (and its `_fused_qk` sibling) use the
**Meta odd/even-interleaved** convention — its `get_rot_transformation_mat` maps
`x -> [-x1, x0, -x3, x2, ...]`, so adopting it would need both the Q/K weight
columns and the cos/sin tables permuted host-side.
`ttnn.experimental.rotary_embedding_hf` implements the HuggingFace
`rotate_half` convention with `cat(freqs, freqs)` tables — i.e. exactly what
`MuseGlimmerTextAttention` spells out — so it drops in with no weight or table
change at all. Chosen. Prefill mode takes `[1, heads, seq, dim]` interleaved
with `[1, 1, seq, dim]` cos/sin; decode mode takes the height-sharded
`[1, batch, heads, dim]` Q/K straight from `nlp_create_qkv_heads_decode` with
`[1, batch, 1, dim]` height-sharded cos/sin.

`rotary_embedding_llama_fused_qk` was assessed as a follow-on (one kernel for Q
and K): it needs `overlap_qk_coregrid=False` plus `[1, 2*batch, 32, dim]`
cos/sin, so it costs two extra shard ops for the K core grid to save one rope
dispatch (~2 us at decode shapes) — and it still needs the Meta permutation.
Rejected.

**B. minimal_matmul.** `models/common/modules/attention/attention_1d.py` opts
into `ttnn.experimental.minimal_matmul` for long prefill ("~2x faster than
`ttnn.linear`"); the same holds here, and more strongly for the two widest
matmuls.  Measured per projection at 8192 rows, both sides on the **same**
HiFi2 / no-fp32-accumulate compute-kernel config that `ttnn.linear` picks by
default for BF16, so this is a pure kernel comparison and no precision decision
is smuggled into a topology change (`logs/prefill_matmul_probe.log`,
`logs/minimal_matmul_sweep.log`):

| projection | `ttnn.linear` | `minimal_matmul` | speedup | PCC vs FP32 |
| --- | --- | --- | --- | --- |
| wqkv 6656x4608 | 2.668 ms | 2.331 ms | 1.14x | 0.999843 -> 0.999947 |
| attn gate 6656x4096 | 2.273 ms | 2.045 ms | 1.11x | 0.999843 -> 0.999947 |
| o_proj 4096x6656 | 4.876 ms | 2.177 ms | 2.24x | 0.999898 -> 0.999954 |
| mlp gate/up 6656x19968 | 23.657 ms | 10.024 ms | 2.36x | 0.999843 -> 0.999947 |
| mlp down 19968x6656 | 22.617 ms | 9.634 ms | 2.35x | 0.999556 -> 0.999910 |

The ranking inverts at small M, and *where* it inverts is per-projection:
`ttnn.linear`'s auto-selected config is not monotone in M (on the MLP shapes it
costs 2.85 ms at 2048 rows, 11.66 at 4096 and 8.98 at 6144), so the crossover is
a band.  The sweep covers M in {32, 128, 256, 512, 1024, 1536, 2048, 3072, 4096,
6144, 8192} for all five projections, min of 3 rounds, both sides on the shipped
compute-kernel config:

| M (chunk rows) | `ttnn.linear` | `minimal_matmul` | delta | winner |
| --- | --- | --- | --- | --- |
| 32 | 2.55 ms | 3.92 ms | +1.37 ms | linear |
| 128 | 2.64 ms | 3.96 ms | +1.31 ms | linear |
| 256 | 2.97 ms | 3.94 ms | +0.98 ms | linear |
| 512 | 3.94 ms | 4.04 ms | +0.10 ms | linear |
| 1024 | 5.79 ms | 5.63 ms | -0.16 ms | **minimal** |
| 1536 | 8.03 ms | 8.41 ms | +0.38 ms | linear |
| 2048 | 10.48 ms | 10.80 ms | +0.32 ms | linear |
| 3072 | 15.81 ms | 14.99 ms | -0.82 ms | **minimal** |
| 4096 | 39.52 ms | 19.12 ms | -20.40 ms | **minimal** |
| 6144 | 31.96 ms | 27.58 ms | -4.38 ms | **minimal** |
| 8192 | 79.75 ms | 36.23 ms | -43.51 ms | **minimal** |

(per-chunk total = wqkv + attn_gate + o_proj + 2 x mlp_gate_up + mlp_down, the
six dense dispatches a prefill chunk issues.)

`MINIMAL_MATMUL_MIN_ROWS = 3072` is the first row count from which *every*
measured point favours the fused kernel, which makes the fused layer **never
slower than the functional baseline at any row count**: below 3072 it runs
exactly the baseline's kernel.

A **per-projection** threshold was the obvious alternative and was costed out of
the same sweep.  Each projection's first win point is 512 for wqkv / attn_gate /
o_proj / mlp_down and 3072 for mlp_gate_up, so a per-projection rule would give
each of them `minimal_matmul` from 512 up.  Against the baseline that trades:

| M | shipped (single 3072 threshold) | per-projection thresholds |
| --- | --- | --- |
| 512 | +0.00 ms | **-0.17 ms** |
| 1024 | +0.00 ms | **-0.28 ms** |
| 1536 | +0.00 ms | +0.21 ms |
| 2048 | +0.00 ms | +0.24 ms |
| >= 3072 | -0.82 to -43.51 ms | identical |

i.e. it buys 0.17-0.28 ms at 512-1024 rows and gives back 0.21-0.24 ms at
1536-2048 rows, and it forfeits the "never slower than the baseline" property.
None of the five projections is monotone in M, because `ttnn.linear`'s
auto-selected config is not, so no *threshold* rule can capture both win bands;
only a fitted band table could, on 11 sample points.  The single threshold is
kept.

Explicit `MatmulMultiCoreReuseMultiCastProgramConfig` tilings were tried first
(the classic 2D fix for a `SLOW` matmul row).  Seven rectangles — 8x{1,2,4,8}
and 11x{1,2,4}, i.e. every grid height that divides all three `K` values
(6656 / 4096 / 19968) over 32 — were built for each of the four projections;
**all 28 exceed the L1 circular-buffer budget** (`program.cpp:1722`).
`MinimalMatmulConfig` block sizes were then swept on the winning kernel
(`M_block x K_block x N_block` in tiles, over the full 11x10 grid).  This first
sweep concluded that no explicit config beats the op's own choice — the best of
its six shapes (`M8 K4 N8`) measured 0.6-11.8 % worse (2.376 vs 2.328 ms wqkv,
10.349 vs 10.283 mlp gate, 10.729 vs 9.594 mlp down) — but it varied only
`K_block in {2, 4}` tiles against a default of **8**, so it could not have
beaten the default on the K axis, and it was measured on host wall-clock, which
§3.3 L later shows cannot resolve gaps this size at all.

**That conclusion is superseded by §3.3 L**, which reopened the sweep over the
whole legal `K_block` range on device kernel time and found configs that do win
on two of the five shapes.  The shipped `_dense` passes `config=` for those two
and nothing for the other three.  Both first-pass sweeps are kept in
`logs/prefill_matmul_probe.log` as the record of how the question was first
asked.

`minimal_matmul` also exposes a pack-time `fused_activation`, so the
"activation on the matmul" rejection (§3.3 I/J) was re-tested on the kernel
prefill actually uses: it fuses, but costs 12.101 vs 10.283 ms on the MLP gate
shape and 2.688 vs 2.328 on wqkv.  Rejected again; the activation stays on the
binary that consumes the result.

**C. Sharded decode RMSNorm.** `ttnn.rms_norm` accepts a
`LayerNormShardedMultiCoreProgramConfig` when its input is width- or
block-sharded in L1.  The core count must divide `hidden_size / 32 = 208 =
2^4 x 13` tiles for the shard width to stay tile-aligned, so the legal counts
are 1, 2, 4, 8, 13, 16, 26, 52, 104, 208.

13 and its multiples have no rectangle on an 11-wide grid, but that is **not** a
blocker: `layernorm_device_operation.cpp:185-215` explicitly accepts a
*non-rectangular* `CoreRangeSet` when the whole height fits on one core
(`M == block_h * TILE_HEIGHT`, always true for a decode step) and the grid is a
shard-order prefix of its bounding box.  Both families were built and measured
(`bench/norm_shard_probe.py`, `logs/norm_shard_probe.log`, min of 3 rounds x
200 calls, wall time including the two reshards):

| grid | cores | block_w | subblock_w | us/call | PCC vs torch |
| --- | --- | --- | --- | --- | --- |
| interleaved | 1 | — | — | 134.4 | 0.9999983 |
| rect 2x1 | 2 | 104 | 4 | `TT_THROW program.cpp:1779` | — |
| rect 4x1 / 2x2 | 4 | 52 | 4 | 29.3 / 31.0 | 0.9999964 |
| **rect 4x2** | **8** | **26** | **2** | **22.8** | 0.9999965 |
| rect 8x1 / 2x4 | 8 | 26 | 2 | 23.0 / 23.4 | 0.9999964 |
| prefix bbox 11x2 | 13 | 16 | 4 | 25.3 | 0.9999964 |
| rect 8x2 / 4x4 / 2x8 | 16 | 13 | 1 | 24.4 / 25.4 / 26.4 | 0.9999965 |
| prefix bbox 11x3 | 26 | 8 | 4 | 28.0 | 0.9999964 |
| prefix bbox 11x5 | 52 | 4 | 4 | 37.7 | 0.9999964 |
| prefix bbox 11x10 | 104 | 2 | 2 | 57.6 | 0.9999964 |

The non-rectangular grids are legal and correct but *slower*, even at
`subblock_w = 4`: a decode step is a single tile-row, so past ~8 cores the
per-core reduction and the cross-core stats exchange cost more than the extra
width parallelism buys.  8 cores is the measured optimum and
`choose_decode_norm_grid()` reproduces it from `DECODE_NORM_TARGET_CORES`
rather than from a grid-shape argument.

### 3.2 Graph rewrites

| # | rewrite | effect |
| --- | --- | --- |
| D | decode residual stream stays **width-sharded in L1** for the whole layer | the two residual adds become sharded element-wise ops and all four hidden-size norms consume/produce the sharded layout, so `LayerNorm` goes 447 -> 59 us/step. It does *not* remove the reshards: a decode step still runs 8 `InterleavedToShardedDeviceOperation` + 6 `ShardedToInterleavedDeviceOperation`, six of them on the hidden-size stream (`residual`, `attn_out`, `mlp_out` in; `normed`, `mlp_in`, `out` out). They cost 19.3 us of a 2,710 us step (that count is the `sliding` graph; a `full` step is 6 + 6, without the two RoPE table reshards), and §4.1 records why the two matmul-side ones cannot be merged away |
| E | prefill RoPE tables stored **pre-tilized**; at `start_pos == 0` the persistent table is handed to the op directly | removes 2 `TilizeDeviceOperation` + 2 `SliceDeviceOperation` per chunk (`rotary_embedding_hf` only requires `cos_seq_len >= seq_len`) |
| F | decode RoPE tables gathered **straight into** the height-sharded `[1, batch, 1, dim]` decode layout | removes 4 `ttnn.repeat` broadcasts and their untilize/tilize round trips (16 ops -> 6) |
| G | decode Q stays height-sharded from `nlp_create_qkv_heads_decode` through RoPE into the SDPA kernel | removes a DRAM round trip for Q |
| H | decode QKV projection writes **directly to L1** | removes the `CopyDeviceOperation` the functional layer needed to stage the fused QKV for `nlp_create_qkv_heads_decode` (the Blackhole tt-metal #16667 workaround stays, it just no longer costs an op) |

### 3.3 Op merging

| # | rewrite | effect |
| --- | --- | --- |
| I | `silu(gate) * up` -> `ttnn.mul(gate, up, input_tensor_a_activations=[SILU])` | isolated 4.458 -> 2.539 ms at 8192x19968, identical PCC (`logs/op_merge_probes.log`) |
| J | `heads * sigmoid(gate_proj(h))` -> `ttnn.mul(heads, gate, input_tensor_b_activations=[SIGMOID])` | removes the prefill sigmoid `UnaryDeviceOperation` (347.6 us on `sliding`, 380.3 on `full` in the functional captures) |
| K | prefill SDPA `q_chunk == k_chunk` 128 -> **256**, at **both** call sites | in-memory op at 8192 tokens: 12.368 -> 8.155 ms (sliding), 12.253 -> 7.730 (full), PCC unchanged. Paged `chunked_scaled_dot_product_attention` (every `full` chunk after the first): 36.204 -> 22.831 ms at `chunk_start_idx=8192` and 109.992 -> 72.277 at 32768 — **1.59x**, PCC 0.99978 -> 0.99982 / 0.99965 -> 0.99975 |
| L | explicit `MinimalMatmulConfig` blocking on the two projection shapes that want one: `o_proj` -> `M16 K4 N8` (full 8192-row chunk only), MLP gate/up -> `M8 K4 N16` | device kernel time at 8192 rows: `o_proj` 2011.9 -> 1957.1 us (+2.80 %), MLP gate/up 9052.9 -> 8795.7 us (+2.92 %), and the MLP win holds on tail chunks too (+0.92 % at 4096, +1.49 % at 6144). Three dispatches per chunk take it, so about 570 us of an 8192-token chunk: the committed window is 49.32 ms, and adding the three measured per-shape deltas back gives 49.89 ms — that is arithmetic on this capture, not a second one |

**K** deserves a note.  The functional stage pinned `q_chunk == k_chunk`
because `q_chunk == 2 * k_chunk` silently mis-masks the sliding window
(functional-stage limitation 1 and its committed reproducer), and never swept
the *size*.  `bench/sdpa_chunk_sweep.py` sweeps 128 / 256 / 320 / 384 / 512 at
nine lengths — 1024, 2048, **2080**, 3008, 4096, **4128**, 6144, 8192, **8224**
(the three bold ones are exactly the lengths that expose the chunk bug) — for
both kinds, min of 3 rounds (`logs/sdpa_chunk_sweep.log`):

* 384 and 512 exceed the L1 circular-buffer budget (1.93 MB / 2.87 MB against
  1.57 MB) at every length.
* 256 beats 128 everywhere, by 1.04x at 2048 and 1.52x at 8192.
* Between 256 and 320, **256 wins at 7 of the 9 lengths on the `full` kind
  and 6 of 9 on `sliding`** (the ninth, 8224 sliding, is 8.636 vs 8.628 ms, a
  0.1 % loss), including the
  8192-token internal prefill chunk (8.155 vs 8.600 ms) and everything below
  3008.  320 wins only in a narrow band around 4k (4096: 2.426 vs 2.574;
  4128: 2.421 vs 2.588) and ties at 8224 (8.628 vs 8.636).
* PCC is 0.99992 for all three at every length, including 2080 / 4128 / 8224.

The retune has to reach **both** prefill SDPA call sites.  A `sliding` layer
uses the in-memory `scaled_dot_product_attention` for every chunk, but a `full`
layer only uses it for chunk 0 — every later chunk reads the whole prefix back
out of the paged cache with `chunked_scaled_dot_product_attention`, which builds
its own program config from a separate constant.  That is the dominant op of any
prefill longer than one internal chunk (at the advertised 131072 context a
`full` layer makes 15 such calls, the last of them against a 122880-token
prefix), and the functional layer hard-coded 128 there.  `FusedDecoder` now
overrides `_prefill_sdpa_full` to seed it from `PREFILL_SDPA_CHUNK`, keeping the
halving loop that the op's `chunk_start_idx % q_chunk_size == 0` rule needs for
caller-level continuations.  Measured on the real op at the offsets an
8192-chunked prefill produces (`bench/chunked_sdpa_sweep.py`,
`logs/chunked_sdpa_sweep.log`):

| `chunk_start_idx` | prefix | 128 | 256 | speedup | PCC 128 -> 256 |
| --- | --- | --- | --- | --- | --- |
| 8192 | 16384 | 36.204 ms | 22.831 ms | 1.59x | 0.999781 -> 0.999816 |
| 32768 | 40960 | 109.992 ms | 72.277 ms | 1.52x | 0.999647 -> 0.999746 |

512 does not fit L1 there either, and 320 is unusable at that site at all: it
must divide `chunk_start_idx`, which is a multiple of the 8192 prefill chunk.
End to end this is worth 1.60-1.62x wall-clock on a two-chunk (16384-token)
prefill against the functional baseline, and 2.00-2.05x of device time — see `logs/multichunk_prefill_ab.log` and the
`prefill_16384` Tracy windows.

256 is shipped as a single constant.  A length-dependent rule was considered and
rejected: the 320 band is two sample points wide, the win there is ~6 % of SDPA
(under 1 % of prefill), and prefill is chunked at 8192 so only a prompt whose
*final* chunk happens to land in that band would see it.  The claim in the code
and README states exactly this rather than "320 is slower".

**L** needed a different measuring instrument than everything else here.  The
op's own `determine_default_block_sizes` (`minimal_matmul_program_factory.cpp:22-42`)
hands the no-`config=` path `M=K=N=8` with `2x4` subblocks, and the first sweep
(`bench/prefill_matmul_probe.py`) had only tried `K_block in {2, 4}` — i.e. it
could not have beaten the default on the K axis, and every measured pair was
still improving as K grew.  Reopening it:

* `bench/prefill_matmul_kblock_probe.py` sweeps `K_block` from 4 up to the full
  K-tile count on all four shapes, divisors and non-divisors alike.  Everything
  at or above 20 tiles is a hard L1 stop: *"Statically allocated circular
  buffers on core range [0-0 - 10-9] grow to 1684352-8893312 B which is beyond
  max L1 size of 1572864 B"* (`program.cpp:1722`).  The same wall hits every
  `M16 x N16`, `M16 K8` and `N24` variant.
* That sweep reported wins of 1-3 %, which is **the same size as its own
  noise**.  `bench/prefill_matmul_kblock_confirm.py` A/B'd each candidate
  against the shipped default inside one interleaved loop and measured the
  default against *itself* as a control: the control reports -0.5 % to -10.8 %.
  Host wall-clock cannot resolve this question at all.
* `bench/prefill_matmul_kblock_device{,2,3,4,5}.py` therefore measure **device
  kernel duration** under Tracy — the metric the committed perf reports use —
  with 8 reps per group and the default re-measured between every candidate.
  Default groups reproduce to +-0.1 %, and the answer is stable across the two
  independent rounds (`o_proj` +2.88 % then +2.80 %).

The result is per-shape, and three of the five shapes want nothing: `wqkv`
(best candidate -2.6 %), the attention gate (-0.01 %, i.e. its best candidate is
the default kernel) and `mlp_down` (-0.1 %).  The attention gate is its own
dispatch worth 1,865 us of the 49,318 us `sliding` window — the same order as
`o_proj`, the shape a config *was* worth +2.80 % on — so it was swept over the
same K range rather than assumed (`bench/prefill_matmul_kblock_device4.py`).  It is also
per-*height*: `o_proj`'s config wins by 2.80 % at the full 8192-row chunk and
loses 6.0-6.3 % at 4096 and 6144 rows, so `MINIMAL_MATMUL_BLOCKS` carries a
minimum row count per entry and `o_proj` keeps the op default on tail chunks.
The MLP config wins at every height that reaches `minimal_matmul` at all and so
carries the crossover threshold itself — measured, not extrapolated, at all four
points: +2.61 % at 3072 (`..._device5`), +0.92 % at 4096, +1.49 % at 6144 and
+2.92 % at 8192.
`test_minimal_matmul_block_config` pins the table, the row gate and the
subblock rule, so dropping the blocking fails the suite instead of quietly
costing 2-3 %.

### 3.4 Accuracy side effect, and the fidelity policy

`minimal_matmul` is more accurate than `ttnn.linear` **at the same math
fidelity** (see the table in 3.2 B), so the fused prefill is measurably closer
to the HF reference than the functional one, not just faster.  From
`logs/full_test_run.log` (`test_fused_vs_functional_equivalence`'s accuracy
control, which runs both graphs on the same inputs and compares both to the
same HF reference), **all six prefill comparisons improve** (+3.3e-4 to
+1.2e-3), which is why the prefill tolerance in
`test_fused_vs_functional_equivalence` is exactly zero.  Five of the six decode
comparisons improve (the largest, `sliding` at 12345, by +1.3e-3) and one drifts
by **-4.6e-5**.  Decode does *not* change matmul kernel — a step is 32 rows,
below the `minimal_matmul` crossover — so apart from the norm-fidelity uplift
(§3.5) its rewrites only re-associate BF16
rounding rather than changing precision, and the result can land either side of
the baseline's.

The decode tolerance is **2e-4**, which bounds that observed drift by about
**4.3x**.  It was 5e-4 — a 1.15x margin against a -4.3e-4 drift — until §3.5's
last correction: the prefill per-head QK norms were still on `ttnn.rms_norm`'s
default config, and since those norms write the Q and K prefill stores in the
paged cache, giving them the same uplift as every other norm in the layer
shrank the worst decode drift 10x.  Even at 2e-4 the guard sits far inside the
headroom from the suite's worst decode PCC (0.998152) to the 0.995 acceptance
bar.  The suite's worst HF-vs-TTNN check moves up over the functional stage
either way, 0.997422 -> 0.998152.

The fidelity policy itself is *not* a fusing decision and was deliberately kept
where the baseline had it.  `minimal_matmul`'s own default compute-kernel config
is more accurate still (PCC 0.999994 vs 0.999947 against an FP32 reference) but
costs 2.3-2.5 ms on an 8192-token prefill (`logs/dense_compute_kernel_probe.log`:
86.10 vs 88.60 ms sliding, 85.28 vs 87.66 ms full, both including the host
upload of the 109 MB activation, so the delta is the signal, not the absolute).
Shipping it would mean this stage selecting a slower higher-precision path and
would make the before/after comparison a mix of topology and precision.  The
shipped `dense_compute_kernel_config` therefore pins HiFi2 /
`fp32_dest_acc_en=False` / `packer_l1_acc=True` — exactly what `ttnn.linear`
reports in the functional stage's perf tables — and the more-accurate default is
recorded here for the optimized-decoder stage, which owns precision policy.

### 3.5 The one fidelity change: the RMSNorm compute-kernel config

The paragraph above is true of the *matmuls*.  It is not true of the norms, and
that has to be stated plainly because the accuracy gain above is partly theirs.

`ttnn.rms_norm`'s default compute-kernel config is
`HiFi4 / math_approx_mode=True / fp32_dest_acc_en=False / packer_l1_acc=False`
(`rmsnorm.cpp:16-20`), and the functional layer got it by passing no config at
all.  Every RMSNorm in the fused layer instead runs
`HiFi4 / approx=False / fp32_dest_acc_en=True / packer_l1_acc=True`
(`norm_compute_kernel_config()`), i.e. the same math fidelity with the
approximate reciprocal-sqrt off and FP32 destination accumulation on, for a
6656-wide BF16 reduction.

Measured in isolation against a **float64** reference
(`bench/norm_fidelity_probe.py`, `logs/norm_fidelity_probe.log.gz`):

| shape | op default | uplifted | PCC vs f64 | max relative error |
| --- | --- | --- | --- | --- |
| prefill, 8192x6656 interleaved | 978.27 us | 991.78 us | 0.999928684 -> 0.999998467 | 6.5e-2 -> 4.2e-3 |
| decode, 32x6656 width-sharded 4x2 | 15.53 us | **14.92 us** | 0.999993993 -> 0.999998450 | 1.0e-2 -> 4.8e-3 |

It is *free* in decode — the sharded kernel is 3.9 % faster with FP32
accumulation on — and costs 13.5 us per prefill norm, so ~54 us across the four
hidden-size norms (13.5 us each, measured) plus of order 25 us across the two
much smaller per-head QK norms — the prefill `LayerNorm` total moved
~3,868 -> 3,890 us when those two were included, which is at the edge of the
run-to-run spread and of which only the post-fix capture is committed:
**0.11-0.16 %** of a 49,318 us prefill window, for a 15x smaller worst-case
error on the op that feeds every matmul in the layer.

What it is worth at the model level is measured too, because otherwise §3.4's
"+3.2e-4 at 100 tokens, where no matmul kernel changes" would be attributed to
the RoPE/norm/activation *topology* rewrites when most of it is this.
`logs/norm_fidelity_control.log` is the identical graph with both norm configs
set to `None`:

| control | shipped | norms on the op default |
| --- | --- | --- |
| prefill[sliding] 100 | +0.000369 | **-0.000008** |
| prefill[full] 100 | +0.000334 | **+0.000000** |
| prefill[sliding] 4097 | +0.000992 | +0.000603 |
| prefill[full] 12345 | +0.001190 | +0.000794 |

So at 100 tokens the topology rewrites alone are a wash, and the whole gain
there is the norm fidelity; at 4097+ the remaining +6e-4 to +8e-4 is
`minimal_matmul`'s like-for-like kernel accuracy.  Note the first row: with the
norms on the op default the zero-tolerance prefill assertion **fails** by 8e-6,
which is the sharpest statement of what this knob is doing.

The uplift has to reach *every* norm to be worth what it is worth, and one path
was missed until stage review round 12: the **prefill** per-head QK norms go
through the inherited `_per_head_rmsnorm`, which passed no config, so those two
sat on the op default while the docs said "every RMSNorm".  Overriding them
matters more than their 395 us of runtime suggests, because they write the Q and
K that prefill stores in the paged cache — a decode step then reads a more
accurate cache.  The worst decode accuracy control went from **-4.3e-4 to
-4.6e-5**, a 10x improvement, which is what let
`ACCURACY_REGRESSION_TOL["decode"]` tighten from 5e-4 to 2e-4.  The lesson is in
the test: `test_every_norm_takes_the_uplifted_config` now patches
`ttnn.rms_norm` and asserts the `compute_kernel_config` of all twelve dispatches
(six prefill, six decode), because the previous attribute-level version could
not tell the two states apart.

Why keep it rather than pin the baseline: it is free in decode, 0.16 % in
prefill, and it makes the layer's most reduction-sensitive op an order of
magnitude more accurate — and the alternative, pinning it, would trade real
accuracy for a tidier one-line story about the stage.  What matters is that it
is *disclosed and measured* rather than smuggled in, which is what
`test_norm_compute_kernel_config_is_the_documented_uplift` and
`test_every_norm_takes_the_uplifted_config` now enforce.

## 4. Rewrites assessed and rejected

The first group was **implemented and measured** on device — the candidate
classes live in `bench/variants.py` and the numbers in `logs/variant_sweep.log`
(min of 3 rounds; decode is reproducible to +/- 0.001 ms/token and prefill to
about +/- 2 %, both printed per round so a sub-1 % delta can be judged).  Every
candidate runs the **same** `_dense` dispatch and the same
`dense_compute_kernel_config` as the shipped path, so each comparison isolates
the topology change.  The
second group was rejected on an **exact op contract**, not on a measurement,
because the op cannot express this layer's math or this stage's topology at
all; each row names the contract.

### 4.1 Measured and rejected

| candidate | why it lost |
| --- | --- |
| **`ttnn.linear(..., activation="silu"/"sigmoid")`** (matmul pack-time activation) | Does not fuse on this build for these shapes: the profiler still shows a separate 2,128 us `UnaryDeviceOperation` alongside the activation-carrying matmul. Isolated on the same shape: 23.964 -> 26.461 ms (`logs/op_merge_probes.log`). Strictly worse than doing nothing. (The in-graph matmul row also reads 786 us slower, but that shape shows ~770 us of capture-to-capture spread in the functional baseline itself, so the rejection rests on the surviving unary op and the isolated measurement.) Evidence: `logs/rejected/prefill_perf_report_matmul_activation_{sliding,full}.txt`. Replaced by the binary input-activation form (I, J). |
| **`ttnn.experimental.paged_fused_update_cache`** | The op asserts its two update tensors are on disjoint cores (`paged_fused_update_cache_device_operation.cpp:341-348`). `nlp_create_qkv_heads_decode` emits V on Q's grid unconditionally and can only move **K** off it via `overlap_qk_coregrid=False` — which the frontend *drops* for an interleaved input (`nlp_create_qkv_heads_decode.cpp:23`: `input_tensor.is_sharded() ? overlap_qk_coregrid.value_or(true) : true`) and which the device op then constrains to a shard holding the full height on one core with `head_dim % shard_width == 0` (`..._device_operation.cpp:56-72`), i.e. to a **WIDTH_SHARDED** QKV (measured: with this layer's L1-interleaved QKV the flag changes nothing — identical Q/K/V grids at batch 1/4/32), a shard width dividing `head_dim=128` (36 cores for the 4608-wide QKV), and `num_cores >= 2*num_users` (not binding here: the op already hard-caps `num_users` at 32 and this grid has 110 cores). This layer's decode QKV is L1 *interleaved* (what the op needs after the #16667 workaround), so the only reachable form here is a manual V reshard — measured 2.737 vs 2.734 ms/token sliding (worse) and 2.705 vs 2.708 full (better) — ~0.1 % either way, sign-flipping between the two layer kinds, so the reshard costs what the saved dispatch is worth. A **per-kind** selection (ship it for `full` only) was considered, since the `full` win reproduces across every round of a tighter 5-round / 256-iteration A/B (`logs/kv_update_ab.log`: 2.704 vs 2.707) and across two independent chain runs. It was not taken: the fused op does not remove work — it replaces two `PagedUpdateCache` dispatches (3.58 us each) with one fused write plus a `to_memory_config` reshard (~1.4 us), so device-side it is a wash and the +-0.003 ms/token is dispatch/DRAM overlap, which is exactly why it flips sign between two graphs that differ elsewhere. Forking the paged cache write by layer kind — the variant hand-builds a disjoint core grid for V — for +0.11 % on one kind and -0.07 % on the other is not a trade worth making in the most correctness-sensitive part of the decoder. `logs/kv_coregrid_probe.log` has the grid dumps, the `must not overlap` rejection at every batch, and the WIDTH_SHARDED control where the disjoint grids *do* appear. |
| **Shared-LHS packing of `wqkv` + attention gate** | One matmul over `concat([wqkv, w_gate], -1)` plus two slices. Decode, which reproduces to +-0.001 ms/token, is a consistent loss on both kinds: 2.738 vs 2.734 (sliding) and 2.709 vs 2.708 (full). Prefill wall-clock agrees (65.78 vs 65.24 sliding, 65.26 vs 64.04 full) but is not what the decision rests on: that A/B has a +-2 % round spread, so it can only say "not faster". The slices cost what the dispatch saves, and decode matmuls are weight-bandwidth bound so packing moves no bytes. |
| **Shared-LHS packing of the MLP gate/up** | 2.757 vs 2.734 ms/token, 66.78 vs 65.24 ms prefill (sliding). Same reason, and the slices are on a 19968-wide tensor. |
| **`ttnn.swiglu` on a packed `[up \| gate]` projection** | A *composite*: two slices + swish + multiply, so it adds ops. 2.767 vs 2.734 ms/token, 68.38 vs 65.24 ms prefill (sliding). |
| **`minimal_matmul(..., fuse_swiglu=True)`** | Genuinely one kernel for gate+up+silu+mul, and faster: 24.682 vs 25.718 ms at 8192 rows. But it needs the gate/up weight in a tile-pair-interleaved layout the decode path cannot use — at 32 rows it is 2.593 ms vs the shipped decode MLP's 1.406 ms, an 84 % decode regression — so the layer would have to carry **both** layouts: +531 MB per layer, i.e. +27 GB over 52 layers on a 32 GB part. Rejected on capacity, with the 1.04 ms (2.1 % of prefill) cost recorded. `logs/op_merge_probes.log`. |
| **`minimal_matmul(..., fused_activation=SILU)`** | The pack-time activation retried on the kernel prefill actually uses. It does fuse, but costs 12.101 vs 10.283 ms on the MLP gate shape and 2.688 vs 2.328 on wqkv (`logs/prefill_matmul_probe.log`). |
| **Peer-merging the two decode RoPE cos/sin gathers** | They share one index tensor, so one `[max_seq, 2*head_dim]` packed table gathers both in one `ttnn.embedding` + one `transpose` instead of two of each. Built and measured (`bench/decode_rope_gather_probe.py`): the outputs are **bit-identical** (`torch.equal` on both halves) and it is a **wash** — 14.08 vs 13.92 us/call, because the two width slices that split the halves apart again cost 5.23 us against the 5.09 us saved on the embedding and transpose (11.69 -> 6.60 us across those two ops). The packed table is the same total bytes, so there is no memory argument either way. |
| **Explicit 2D matmul program configs** | Seven rectangles per projection (8x{1,2,4,8}, 11x{1,2,4} — every grid height dividing all three K values), 28 attempts, **all** rejected by the L1 circular-buffer budget at `program.cpp:1722` (`logs/prefill_matmul_probe.log`). |
| **Explicit `MinimalMatmulConfig` on `wqkv`, the attention gate and `mlp_down`** | All three are fastest on the op's own default (`M=K=N=8`). Best candidate is 2.6 % worse on `wqkv`, 0.01 % on the attention gate (whose best candidate *is* the default: `M8 K8 N8` re-measures to -0.01 %, and `K` 7/9/14 are 0.8-4.4 % worse while 18 and up are L1-blocked) and 0.1 % on `mlp_down`. The remaining two of the five shapes *do* take a config — see §3.3 L. |
| **Writing the decode `o_proj` / `mlp_down` output straight into the sharded residual** | Would remove the two `InterleavedToShardedDeviceOperation` that follow them. `ttnn.linear` accepts the width-sharded memory config but **ignores its grid**: it keeps its own 110-core program config and returns `{[0-0 - 10-8], [0-9 - 4-9]}` where the norm needs `{[0-0 - 3-1]}`, and the sharded LayerNorm then refuses it (`shard_spec_validation.cpp:46`: *"shard_spec.grid size 11x10 does not fit within program_config grid 4x2"*). Adapted and retried: forcing the matmul onto the norm's 4x2 grid with an explicit `MatmulMultiCoreReuseMultiCast1DProgramConfig` works once `in0_block_w` is small enough to fit L1 (the full-K value overflows at 7.19 MB / 34.6 MB against 1.57 MB), and it costs 222.1 us against the shipped pair's 149.4 on `o_proj` and 1071.5 against 700.9 on `mlp_down` — i.e. the shipped pair is 32.8 % / 34.6 % faster, and the 8-core matmul alone is about 1.5x the 110-core one it replaces — both figures already including the 2.48 / 2.47 us reshard the merge would have removed. Eight cores cannot replace 110. `logs/decode_sharded_out_probe*`. |
| **Non-rectangular 13/26/52/104-core sharded decode RMSNorm** | Legal (`layernorm_device_operation.cpp:185-215`) and correct, but slower than 8 cores at every count: 25.3 / 28.0 / 37.7 / 57.6 vs 22.8 us (`logs/norm_shard_probe.log`). |
| **SDPA `q_chunk == 320`** | Fits L1 and wins in a two-point band around 4k, loses at 7 of 9 swept lengths including the 8192-token internal chunk (8.600 vs 8.155 ms). See §3.3 K. |
| **`minimal_matmul`'s own (more accurate) default compute-kernel config** | slower on an 8192-token prefill by 2.3-2.5 ms in the in-graph probe and ~1.0 ms summed over the six dispatches in the isolated sweep, for PCC 0.999994 vs 0.999947; both far above the 0.995 bar and both better than the `ttnn.linear` baseline's 0.999843. Precision policy is the optimized-decoder stage's; see §3.4. |

### 4.2 Rejected on an exact op contract

| candidate | the contract that blocks it |
| --- | --- |
| **`ttnn.rms_norm(..., residual_input_tensor=...)`** | Computes `norm(x + residual)`. Muse-Glimmer is *post-norm*: `x = residual + post_norm(sublayer(x))`, so the op cannot express it. The one add-then-norm site that does match its shape — `hidden = add(residual, attn_normed)` then `pre_feedforward_layernorm(hidden)` — is still a no-win, because `hidden` itself is consumed again by the final residual add, so the separate `add` cannot be removed and the fused form would only add a second read of `residual`. |
| **`ttnn.experimental.rotary_embedding_llama` / `_fused_qk`** | Meta odd/even-interleaved convention (`get_rot_transformation_mat` maps `x -> [-x1, x0, -x3, x2, ...]`), so both the Q/K weight columns and the cos/sin tables would need permuting; `_fused_qk` additionally needs `overlap_qk_coregrid=False` plus `[1, 2*batch, 32, dim]` cos/sin, i.e. two extra shard ops to save one ~2 us dispatch. `rotary_embedding_hf` is the same math with no permutation. |
| **`ttnn.transformer.concatenate_heads`** vs `ttnn.experimental.nlp_concat_heads` | Same op behind two names — `concatenate_heads.cpp:45-47` is literally `ttnn::prim::nlp_concat_heads(...)` followed by a `squeeze`. No change. |
| **`ttnn.experimental.matmul_decode`** | A dedicated decode matmul — exactly the op class that is 93 % of the decode step — but it requires **both** operands `WIDTH_SHARDED` (`matmul_decode_device_operation.cpp:32-39`), i.e. the weights resident in L1. This layer streams 968 MB of BF16 weights per decode step from DRAM; L1 is 1.5 MB per core. Unreachable at this weight dtype and placement, and re-checkable in one line if a later stage shrinks the weights. |
| **`ttnn.transformer.split_query_key_value_and_split_heads`** | Assumes `num_kv_heads == num_heads`; this layer is GQA (32 Q / 2 KV). `nlp_create_qkv_heads` is the GQA-capable dedicated op and was already in use. |
| **`ttnn.fused_rms_minimal`, `ttnn.experimental.dit_fused_distributed_rmsnorm`, `rms_norm_pre/post_all_gather`** | Distributed norms: they require a multi-device mesh and a global semaphore. This is the single-chip stage. |
| **`ttnn.experimental.dit_rms_norm_unary_fused`** | Fuses `unary(rms_norm(x))`. No norm in this layer is followed by a unary — they are all followed by a matmul or an add. |
| **Passing the head-concat shard config straight to the decode SDPA** | Not blocked; simply too small to be worth the batch-shape special-casing — it would remove one `InterleavedToShardedDeviceOperation` (the eight in a `sliding` decode step span **0.48-2.63 us**; a `full` step has six, 0.48-2.58) out of 2710, and only for batches that have a `batch`-core rectangle. Recorded as a hard-check gap rather than a silent omission. |

## 5. Commands

```bash
D=$PWD/models/autoports/meta_models_muse_glimmer_30b/doc/fused_decoder
T=models/autoports/meta_models_muse_glimmer_30b/tests/test_fused_decoder.py

# candidate screening (wall-clock A/B against the functional decoder, min of 3
# rounds with the per-round spread printed)
python $D/bench/ab_latency.py --decode-iters 128 \
    --impl functional,fused,packed_gate_up,swiglu,packed_qkv_gate,fused_kv_update  # logs/variant_sweep.log

# probes kept as evidence
python $D/bench/norm_shard_probe.py            # logs/norm_shard_probe.log
python $D/bench/sdpa_chunk_sweep.py            # logs/sdpa_chunk_sweep.log
python $D/bench/chunked_sdpa_sweep.py         # logs/chunked_sdpa_sweep.log
python $D/bench/prefill_matmul_probe.py        # logs/prefill_matmul_probe.log

# MinimalMatmulConfig blocking: wall-clock sweep, its own noise control, then the
# device-kernel-time rounds that actually decide it (run under the profiler)
python $D/bench/prefill_matmul_kblock_probe.py    # logs/prefill_matmul_kblock_probe.log
python $D/bench/prefill_matmul_kblock_confirm.py  # logs/prefill_matmul_kblock_confirm.log
for r in "" 2 3 4 5; do        # 4 = attention gate, 5 = the 3072-row + K14/18 gaps
  python -m tracy -r -p -v $D/bench/prefill_matmul_kblock_device$r.py \
      > $D/logs/prefill_matmul_kblock_device$r.log 2>&1
done   # then slice each ops CSV back into named groups:
for r in "" 2 3 4 5; do
  cp "$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)" \
      $D/tracy/probes/kblock_device$r"_ops.csv"    # (done per run, newest first)
  python $D/bench/summarize_device_probe.py $D/logs/prefill_matmul_kblock_device$r.log \
      $D/tracy/probes/kblock_device$r"_ops.csv" > $D/logs/prefill_matmul_kblock_device$r"_summary.txt"
done
# rounds 1, 2 and 4 print enough profiler spam to pass the repo's 500 KB file
# hook, so those console logs (and two of the ops CSVs) are committed gzipped;
# summarize_device_probe.py reads .gz transparently
gzip -9 $D/logs/prefill_matmul_kblock_device{,2,4}.log
gzip -9 $D/tracy/probes/kblock_device{,4}_ops.csv

# derived from the suite run, never transcribed by hand (--check turns staleness
# into an exit code)
python $D/bench/summarize_pcc.py               # logs/pcc_summary.txt
python $D/bench/refresh_context_contract.py    # ../context_contract.json
python $D/bench/refresh_context_contract.py --check

# the one fidelity change: RMSNorm compute-kernel config, isolated against a
# float64 reference (run under the profiler; the log is gzipped afterwards)
python -m tracy -r -p -v $D/bench/norm_fidelity_probe.py \
    > $D/logs/norm_fidelity_probe.log 2>&1 && gzip -9 $D/logs/norm_fidelity_probe.log
# ...and its stage-level control: the same graph with both norm configs set to
# None, which is how logs/norm_fidelity_control.log was produced
#   (temporarily edit norm_compute_kernel_config to return None, then)
#   python -m pytest $T::test_fused_vs_functional_equivalence

# is ttnn.linear's auto-selected decode compute config the same as the explicit
# one _dense forwards?  (it is, to every digit -- the forwarding is a no-op)
python $D/bench/decode_dense_ck_probe.py     # logs/decode_dense_ck_probe.log

# the paged_fused_update_cache variant, tighter (5 rounds x 256 iters)
python $D/bench/ab_latency.py --impl fused,fused_kv_update --rounds 5 \
    --decode-iters 256 --tag kvupdate    # logs/kv_update_ab.log

# peer-merging the two decode RoPE gathers into one packed table (rejected)
python -m tracy -r -p -v $D/bench/decode_rope_gather_probe.py \
    > $D/logs/decode_rope_gather_probe.log 2>&1

# decode sharded-output matmul merge (rejected; run under the profiler)
python -m tracy -r -p -v $D/bench/decode_sharded_out_probe.py \
    > $D/logs/decode_sharded_out_probe.log 2>&1
python $D/bench/summarize_device_probe.py $D/logs/decode_sharded_out_probe.log \
    $D/tracy/probes/decode_sharded_out_ops.csv \
    --op-code MatmulDeviceOperation,InterleavedToSharded \
    > $D/logs/decode_sharded_out_probe_summary.txt

# the RoPE-gather probe emits several ops per announced call and two unannounced
# correctness calls, so it has its own slicer rather than the generic one
python $D/bench/summarize_rope_gather_probe.py \
    > $D/logs/decode_rope_gather_probe_summary.txt
python $D/bench/op_merge_probes.py             # logs/op_merge_probes.log
python $D/bench/dense_compute_kernel_probe.py  # logs/dense_compute_kernel_probe.log
# logs/minimal_matmul_sweep.log is the ttnn.linear-vs-minimal_matmul crossover
# sweep at 11 row counts (32..8192) x 5 projections, min of 3 rounds, both sides
# on the shipped compute-kernel config.  Written by bench/minimal_matmul_sweep.py.

# --- the behaviour-carrying chain, in one command ------------------------
# Runs everything below in the order the "Artifact freshness" section claims, so
# that ordering is a property of a committed script rather than of typing:
bash $D/bench/run_evidence_chain.sh     # ~15 min; progress in logs/chain.log
# (it runs black over the two sources first, so the artifacts below are never
#  older than the file the pre-commit hook would rewrite at commit time)

# full fused suite
python -m pytest $T -q --no-header --junitxml=$D/test_results.xml   # 94 passed

# watcher (18 node ids; a separate run from any profiling, and it moves/gzips
# the log itself because the repo ignores any "generated" path component and the
# check-large-files hook rejects anything over 500 KB)
bash $D/bench/run_watcher.sh          # logs/watcher_run.log, watcher/watcher.log.gz

# Tracy + tt-perf-report, 8 windows (2 kinds x {prefill 8192, prefill 16384,
# decode@2048, decode@131071}) plus the two 16384 functional baselines
bash $D/bench/run_tracy.sh
grep -c "markers were dropped" $D/logs/tracy_*.log   # must all be 0
```

## 6. Results

See `README.md` for the full before/after tables, the PCC surface and the
limitations. Headline, all measured with the Tracy device profiler and
`tt-perf-report --csv` `Device Time`:

| kind | window | ops/iter | device time / iter | speedup |
| --- | --- | --- | --- | --- |
| sliding | prefill 8192 (1 chunk) | 42 -> 24 | 101.23 -> **49.32 ms** | 2.05x |
| full | prefill 8192 (1 chunk) | 24 -> 22 | 99.38 -> **47.98 ms** | 2.07x |
| sliding | prefill 16384 (2 chunks) | 95 -> 61 | 214.37 -> **104.79 ms** | 2.05x |
| full | prefill 16384 (2 chunks) | 51 -> 47 | 221.58 -> **111.04 ms** | 2.00x |
| sliding | traced decode @ 2048 | 64 -> 44 | 3.163 -> **2.710 ms/token** | 1.17x |
| sliding | traced decode @ 131071 | 64 -> 44 | 3.160 -> **2.710 ms/token** | 1.17x |
| full | traced decode @ 2048 | 32 -> 34 | 3.080 -> **2.687 ms/token** | 1.15x |
| full | traced decode @ 131071 | 32 -> 34 | 3.575 -> **3.179 ms/token** | 1.13x |

The 16384 rows are the multi-chunk regime a long prompt actually runs, and the
only windows in which a `full` layer touches the paged
`chunked_scaled_dot_product_attention` at all; their baselines were captured the
same way and are committed as `tracy/<kind>/prefill_16384_baseline_*`.
Correctness: **94 passed**, 214 asserted PCC checks (202 HF-vs-TTNN, worst
**0.998152** against a 0.995 bar; 12 fused-vs-unfused, worst 0.996797) plus 12
accuracy controls: all six prefill comparisons improve (+3.3e-4 to +1.2e-3),
five of six decode comparisons improve and one drifts by -4.6e-5 (BF16
re-association; decode never changes matmul kernel).  The functional stage's worst check was
0.997422, so the accuracy floor moved up.

Watcher: 18 tests passed under `TT_METAL_WATCHER=10` in a run with no profiler
attached (`bash $D/bench/run_watcher.sh`, 187 s), covering both kinds'
multi-chunk prefill, decode, continuation prefill, traced replay, batch 13
(fallback head-concat) and batch 32, the non-zero cache slot, the awkward
page-count prefill that is the only case exercising the *halved* paged-SDPA
chunk, the graph audit, the norm-config shapes, the fused-vs-unfused comparison
and the 64-step stress soak.  `watcher/watcher.log.gz` contains **zero** occurrences of
`Watcher detected`, `tripped`, `sanitize`, `TT_ASSERT`, `DEBUG_ASSERT`,
`out of bounds`, `fault` or `Error` in 20490 lines with 38 periodic dumps.
Console log: `logs/watcher_run.log`.

Stress: `test_repeated_run_stress` replays a captured decode trace 64 times per
kind, advancing the position every step, after four re-prefills of the same
user; every step's output is checked finite and every 16th is PCC-checked
against HF (worst 0.998333).  It is the coverage the new L1-resident residual
stream needed and that no single-shot test provides.

Where the remaining time goes: decode is **93 % the BF16 weight-streaming
roofline** (968 MB of weights at the 383 GB/s the six matmuls achieve =
2.526 ms of a 2.710 ms step; everything else in the layer is 0.18 ms).  Prefill
is 65 % six `MinimalMatmul` ops at 228.5-255.5 TFLOPs (`full`: 228.6-255.5), none of which
`tt-perf-report` marks `SLOW` any more, and both the op's block-size config and
28 explicit 2D matmul grids were swept without finding anything better.  Both
remaining levers are precision/matmul-config, i.e. the optimized-decoder stage.

### Artifact freshness

Every measured number in this work log and in the README is re-derivable from a
committed artifact in this directory (the exceptions are arithmetic derived in
the text and labelled as such, e.g. per-chunk sums over the crossover sweep and
the per-chunk sums over the crossover sweep), and the
behaviour-carrying evidence chain was regenerated **in order** after the last
edit to `tt/fused_decoder.py` *or* `tests/test_fused_decoder.py`, by
`bash $D/bench/run_evidence_chain.sh` (§5), which runs, in this order: the
variant sweep, the multi-chunk A/B, the pytest suite with its junit XML, the PCC
summary and the context contract (both derived from that run, by
`bench/summarize_pcc.py` and `bench/refresh_context_contract.py`), all ten Tracy
captures (eight fused windows plus the two multi-chunk functional baselines),
and finally the watcher run.  Its own progress log is `logs/chain.log`.

Two checks keep that honest.  Since the artifacts are untracked until the
checkpoint commit, mtime *ordering* is the first one, and it has to include the
two source files, not just the artifacts:

```bash
ls -l --time-style=+%m-%d_%H:%M \
   models/autoports/meta_models_muse_glimmer_30b/tt/fused_decoder.py \
   models/autoports/meta_models_muse_glimmer_30b/tests/test_fused_decoder.py \
   doc/fused_decoder/logs/variant_sweep.log doc/fused_decoder/test_results.xml \
   doc/fused_decoder/tracy/sliding/prefill_ops.csv doc/fused_decoder/watcher/watcher.log.gz
```

Both sources must be **older** than every artifact.  A cheaper independent
check, which does not depend on mtimes at all: loguru stamps every log line
with the emitting source line, so the `test_perf_prefill:NNNN` /
`test_perf_decode_traced:NNNN` line numbers in `logs/tracy_*.log` must match the
current `tests/test_fused_decoder.py`, and the `assert_pcc` /
`test_fused_vs_functional_equivalence` line numbers in `logs/full_test_run.log`
must too.  In an earlier round they did not — the Tracy captures had been taken
before a docstring edit shifted those functions by four lines — which is exactly
the failure mode this paragraph exists to catch.

The *probe* logs are a separate class from the evidence chain: they are not
regenerated with it, because each one backs a shipped constant rather than a
reported result, and re-running them would only re-measure a decision already
made.  Eighteen of the twenty-one never construct a `FusedDecoder` at all — they measure
raw TTNN ops at this layer's shapes, so a change to the module cannot invalidate
them — and each backs a constant that has not changed since it was taken:

* `logs/norm_shard_probe.log` -> `DECODE_NORM_TARGET_CORES`
* `logs/sdpa_chunk_sweep.log` and `logs/chunked_sdpa_sweep.log` ->
  `PREFILL_SDPA_CHUNK`, at the in-memory and the paged SDPA call site
  respectively
* `logs/minimal_matmul_sweep.log` + `logs/prefill_matmul_probe.log` ->
  `MINIMAL_MATMUL_MIN_ROWS`
* `logs/prefill_matmul_kblock_*` and `tracy/probes/kblock_device*_ops.csv`
  (rounds 1, 2 and 4 gzipped for the 500 KB file hook) ->
  `MINIMAL_MATMUL_BLOCKS`, including round 4's attention-gate sweep and
  round 5's 3072-row and non-power-of-two `K_block` gaps
* `logs/norm_fidelity_probe.log.gz` -> `norm_compute_kernel_config()` (§3.5)
* `logs/decode_dense_ck_probe.log` -> `_dense` forwarding the compute-kernel
  config to `ttnn.linear` being a verified no-op
* `logs/decode_rope_gather_probe*` + `tracy/probes/decode_rope_gather_ops.csv`
  -> the cos/sin peer-merge rejection
* `logs/decode_sharded_out_probe*` + `tracy/probes/decode_sharded_out_ops.csv`
  -> the sharded-output matmul rejection
* `logs/op_merge_probes.log` and `logs/kv_coregrid_probe.log` -> the activation
  and KV-cache rejections

Three do build a `FusedDecoder`.  Two of them — `logs/norm_fidelity_control.log`
and `logs/kv_update_ab.log` — are the controls §3.5 and §4.1 quote.  Both were
taken *before* the round-12 fix that gave the prefill per-head QK norms the same
config as every other norm, which their mtimes show and which does not affect
what they measure: the norm control toggles **all** norms to the op default
either way (its header, written before that fix, enumerates only the four
hidden-size and the two decode QK norms), and the KV A/B is a traced-decode
latency comparison that no prefill norm config can move.  The third,
`logs/dense_compute_kernel_probe.log`, predates `MINIMAL_MATMUL_BLOCKS` — so its shipped side is
now about 0.57 ms/chunk faster than when it was taken.  It is quoted only for
the *sign and rough size* of the delta between two compute-kernel configs, and
`logs/minimal_matmul_sweep.log` confirms that sign independently at the op level
(`minimal` vs `minimal_opdefault` at M=8192, summing to ~1.0 ms per chunk over
the six dispatches against the probe's in-graph 2.3-2.5 ms).  The two
measurements disagree on magnitude — in-graph the slower kernel also shifts
dispatch overlap — so both are quoted rather than one.  The rejection does not
turn on which is right: the op default is slower under both.

Some scripts under `bench/` have mtimes newer than the logs they produced, from
docstring and comment corrections made during the review rounds.  Each script's
`print()` format strings still match its log line for line, and `variants.py`'s
candidate classes are unchanged — that is the check to run if it ever needs
re-verifying, and it is a stronger one than the mtime, because a comment edit
moves the mtime and changes nothing else.

Finally, `logs/rejected/prefill_perf_report_matmul_activation_{sliding,full}.txt`
is explicitly a capture of the *pre-`minimal_matmul`* graph, kept as the evidence
for that rejection and labelled as such in the README.

## 7. Stage review

`$stage-review` was run as an independent fresh subagent after every change to
the stage, sixteen rounds in total.  Round 16 returned **`clean-pass` with zero
required work**; rounds 1-15 returned `more-work-needed` and every finding was
fixed and re-reviewed rather than argued away.  The reviewer re-derived, from
the committed artifacts each round, the whole before/after table, both op-share
breakdowns, every config percentage, the crossover and chunk sweeps, the PCC
surface and the watcher result, and re-ran `bench/summarize_pcc.py`,
`bench/summarize_device_probe.py`, `bench/summarize_rope_gather_probe.py` and
`bench/refresh_context_contract.py --check` against the committed CSVs.

What the review actually changed, in order of consequence:

| round | finding | outcome |
| --- | --- | --- |
| 6 | the paged SDPA call site was still on the functional layer's chunk 128 | fixed; 1.59x on that op, and it is the dominant op of any multi-chunk prefill |
| 7 | the retune I had just made overran the page table on an awkward page count | real bug I introduced, fixed with a dual-constraint halving loop and pinned by `test_multi_chunk_prefill_page_table_bound` |
| 9 | the `MinimalMatmulConfig` sweep had never tried `K_block` above 4, against a default of 8 | reopened on device kernel time; two shapes now ship a config, worth 2.8-2.9 % on 65 % of prefill |
| 11 | every RMSNorm ran a higher-fidelity compute-kernel config than the op default, undisclosed, while three documents claimed "unchanged precision" | measured in isolation and at model level, kept, and documented as the stage's one fidelity change (§3.5) |
| 12 | the *prefill* per-head QK norms had been missed by that uplift, and the test pinning it read an attribute rather than the dispatch | fixed; worth 10x on the worst decode accuracy control, which let the decode tolerance tighten 5e-4 -> 2e-4 |
| 13 | the decode cos/sin gather's peer merge had never been attempted | built and measured: bit-identical, a wash, now an earned rejection |
| 10, 14, 15 | quoted numbers that no committed artifact produced | each re-derived; `summarize_pcc.py` and `refresh_context_contract.py` now generate what used to be transcribed, and `--check` turns staleness into an exit code |

## 8. Checkpoint commits

Local only, on `agentic-research/hous/muse-glimmer-30b`; nothing pushed.

| repo | commit | what |
| --- | --- | --- |
| tt-metal | `85daa112c57` | the stage: `tt/fused_decoder.py`, `tests/test_fused_decoder.py`, `doc/fused_decoder/`, `doc/context_contract.json` |
| tt-metal | `827a7ade4dc` | evidence regenerated after the pre-commit hooks reformatted the two sources |
| tt-metal | `190437a00ed` | the RMSNorm fidelity uplift disclosed and measured; attention-gate sweep; MLP 3072-row point |
| tt-metal | `70f8e685f10` | the prefill QK norms brought into that uplift; decode tolerance 5e-4 -> 2e-4 |
| tt-metal | `263ab2641b5` | the RoPE-gather peer merge measured and rejected; docstrings made capture-independent |
| tt-metal | `e45e9ce1068` | three quoted figures corrected; the last probe summary made regenerable |
| tt-metal | `3c0d549e7db` | three transcription corrections from round 15 |
| tt-metal | *(this commit)* | the round-16 `clean-pass` record and these SHAs |

No unrelated dirty state was included: every commit touches only
`models/autoports/meta_models_muse_glimmer_30b/{tt/fused_decoder.py,
tests/test_fused_decoder.py, doc/fused_decoder/**, doc/context_contract.json}`.
