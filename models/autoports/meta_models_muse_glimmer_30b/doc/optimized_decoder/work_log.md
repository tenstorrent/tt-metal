# Optimized decoder work log — meta-models/Muse-Glimmer-30B

Stage 02. Chronological: what was measured, what broke, what the evidence said. The
summarised result is in [`README.md`](README.md); the generated numbers are in
[`evidence_tables.md`](evidence_tables.md).

Hardware for every run: one Blackhole `p300c` chip of the 4-chip host, `(1, 1)` mesh,
compute grid 11x10 (110 cores), 8 DRAM banks, 1461504 B L1 per core, 34.18 GB DRAM.
`timeout 60 tt-smi -ls --local` at the start of the stage listed all four boards healthy;
no reset or recovery was needed at any point, and no ARC/ERISC/Ethernet signature appeared.

## 1. Operation-topology audit of the measured path

Read before touching anything: the stage-01 `tt-perf-report` tables
(`doc/functional_decoder/tracy/*/`) together with `tt/functional_decoder.py`.

Decode, batch 1, context 4096, sliding_rope — 3.21 ms of device time per token:

| op | count/token | µs each | µs/token | share | reading |
|---|---|---|---|---|---|
| `MatmulDeviceOperation 32 x 6656 x 19968` (SwiGLU gate, up) | 2 | 695, 693 | 1388 | 43% | DRAM-bound, 75% of peak BW, BF16 weights |
| `MatmulDeviceOperation 32 x 19968 x 6656` (SwiGLU down) | 1 | 692 | 692 | 22% | same |
| `LayerNormDeviceOperation` (hidden-size) | 4 | 138 | 552 | 17% | **1 core**, ~3 GB/s |
| `MatmulDeviceOperation 32 x 6656 x 4608` (QKV) | 1 | 160 | 160 | 5% | DRAM-bound |
| `MatmulDeviceOperation 32 x 4096 x 6656` (output proj) | 1 | 147 | 147 | 5% | DRAM-bound |
| `MatmulDeviceOperation 32 x 6656 x 4096` (attention gate) | 1 | 140 | 140 | 4% | DRAM-bound |
| `SdpaDecodeDeviceOperation` | 1 | 38 | 38 | 1% | BF16 cache |
| everything else | ~22 | 1-13 | ~90 | 3% | head creation, RoPE, cache update, elementwise |

Candidate replacements derived from the model's own dataflow, and what happened to each:

| observation | candidate | dtype/fidelity constraint | action | evidence |
|---|---|---|---|---|
| 2527 µs of the 3210 µs is matmul at 75% of DRAM BW with BF16 weights | reduce weight dtype: the traffic *is* the cost | attention and MLP projections tolerate BFP4 on real weights; norms and activations stay BF16 | **taken** | §3, candidate table |
| Four hidden-size RMSNorms on 1 core | L1 width-sharded residual + `LayerNormShardedMultiCoreProgramConfig` | needs a rectangular shard grid | **taken**: 552 → 37 µs | §2 |
| Matmuls use the default program config with DRAM-interleaved weights | DRAM-sharded matmul, weights width-sharded over the 8 DRAM banks | in0 must be L1 width-sharded on a core count dividing both tiled K and tiled N | **taken**: 1.402 → 1.065 ms at the same dtype | candidate table |
| QKV *and* the attention gate consume the same post-norm activation (2 matmuls, one input) | pack into one 8704-wide projection, split the output | split must land on a shard boundary; 4608 is not a multiple of 8704/16 | **tried, rejected**: 1.0651 vs 1.0636 ms, a tie; kept separate as the simpler contract | candidate table |
| SwiGLU gate and up consume the same activation | pack into one 39936-wide projection, split the output | splits at 624 tiles = 26 shards of 24 tiles, exactly on a boundary | **tried, rejected**: 1.0929 vs 1.0636 ms; the two slices cost more than the saved launch | candidate table |
| `ttnn.silu` then `ttnn.multiply` over the 19968-wide intermediate | fuse SiLU into the multiply (`input_tensor_a_activations`) | none | **taken** | 3.2% of functional prefill device time |
| `ttnn.sigmoid` then `ttnn.multiply` on the attention gate | fuse sigmoid into the multiply (`input_tensor_b_activations`) | none | **taken** | — |
| BF16 KV cache | BFP8 cache; fill tensors cast to the cache dtype, decode update tensors stay BF16 | `paged_fill_cache` wants the cache dtype, `paged_update_cache` takes BF16 in | **taken**: no latency change at 4096 but halves the cache footprint | candidate table |
| Prefill matmuls all `SLOW`, `in0_block_w=1`, DRAM-interleaved in0 | explicit 2D program configs, sequence folded into L1-sized row blocks | per-core output block must fit L1 | **taken**: 97.7 → 54.6 ms at 8192 | §4 |
| Prefill: `nlp_create_qkv_heads` / `nlp_concat_heads` / SDPA already composite ops | nothing to replace — SDPA, FlashDecode, paged fill/update and the head helpers are already the optimized composite ops for this contract | — | **already optimal** | — |
| No collectives in the measured path (single device) | — | — | not applicable at this stage; stage 03 owns CCL topology | — |
| No MoE / routed experts in this model (dense SwiGLU) | — | — | not applicable | `text_config` has no expert config |
| No LM head in this layer (softcap and `output_multiplier` live on the model, not the layer) | — | — | out of scope for stage 02, owned by the full-model stage | `README.md` scope note |

## 2. Sharded decode residual and norms

`6656 = 208 tiles = 16 x 13`. Core counts that both divide 208 and form a rectangle inside
an 11x10 grid: 1, 2, 4, 8, 16 — 26/52/104 would need a 13-wide row. 16 cores it is, at 416
elements (13 tiles) per shard.

First attempt failed immediately:

```
TT_FATAL: Sharded layernorm does not support non-rectangular core grids.
The shard spec grid has 16 cores but its bounding box spans 22 cores (11 x 2).
```

Not from `create_sharded_memory_config` — that produces a clean 8x2 rectangle — but from
the *matmul output*. `MatmulMultiCoreReuseMultiCastDRAMSharded` ignores the requested
output memory config and builds its own from `num_cores_to_corerangeset(num_cores, grid,
row_wise=True)`, which for 16 cores on an 11-wide grid is the 11+5 set. So the layer models
what the op actually produces (`_matmul_output_memcfg`), keeps the elementwise consumers on
exactly the producer's layout, and reshards only where the residual contract needs a
rectangle: after the output projection and after the down projection, before their norms.
Four `Reshard` ops per token, 7.1 µs, 0.7%.

Norms: 138 µs on 1 core → 9.2 µs on 16 cores, four per layer.

## 3. Precision / fidelity policy

Six named policies, one tensor group at a time
(`tt/optimized_decoder.py::POLICIES`). Measured PCC on synthetic weights at the real
per-tensor scales, then again on **real checkpoint weights** and real activations (the
checkpoint's own embedding rows run through the real preceding layers) loaded from the
local snapshot:

| policy | synthetic prefill | synthetic decode | real prefill | real decode | traced decode ms |
|---|---|---|---|---|---|
| `bf16_baseline` | 0.999178 / 0.999066 | 0.998918 / 0.999232 | 0.999899 / 0.999935 | 0.999672 / 0.999947 | 2.2020 |
| `bfp8_all_hifi2` | 0.998722 / 0.998476 | 0.998606 / 0.998617 | 0.999824 / 0.999898 | 0.999399 / 0.999958 | 1.8454 |
| `bfp8_all_lofi` | 0.998643 / 0.998391 | 0.998190 / 0.998693 | 0.999869 / 0.999921 | 0.999659 / 0.999948 | 1.2540 |
| `bfp8_attn_bfp4_gateup` | 0.995062 / 0.993910 | 0.994555 / 0.994158 | 0.999533 / 0.999644 | 0.999497 / 0.999480 | 1.1534 |
| `bfp8_attn_bfp4_mlp` | 0.993321 / 0.991733 | 0.992784 / 0.991905 | 0.999053 / 0.999183 | 0.999161 / 0.998850 | 1.0987 |
| **`bfp4_all`** | 0.968243 / 0.963489 | 0.969057 / 0.960099 | **0.998470 / 0.998871** | **0.996974 / 0.998443** | **1.0639** |

(sliding_rope / full_nope, sequence length 512, both weight sources measured in one
process on the final code by `scripts/sweep_optimized_precision.py`; raw artifact
[`pcc/policy_sweep.json`](pcc/policy_sweep.json). Latencies from the candidate table.)

The synthetic and real columns disagree sharply for the BFP4 policies and agree for
everything else. That is the expected signature of block-float quantisation: BFP4 shares
one exponent across a 16-element block, which is cheap on the checkpoint's strongly
correlated weight blocks and expensive on i.i.d. Gaussian ones. It is not an implementation
defect — the same code at BFP8 passes on synthetic weights at 0.9983 and above.

`$optimize` OPT-012 says a synthetic-only failure does not veto a real-weight win, and
prescribes adding same-contract real-weight coverage for the disputed conditions instead.
Done: `test_real_weights_non_aligned_and_traced` covers a non-aligned prompt length, the
paged prefill→decode transition and a 3-step traced replay on real weights at the 0.995
bar, and `test_stress_repeated_prefill_decode` repeats prefill + 8 decode steps three times
on real weights. `bfp4_all` is selected. The suite keeps the synthetic structural coverage
at full strength by running it at `bfp8_all_lofi` — same code path, same 0.995 bar — so a
page-table or chunking regression still fails loudly on cheap weights.

OPT-007 (attention-projection precision as its own search) is satisfied by the
`bfp8_attn_bfp4_*` rows: BFP4 attention weights were tried on real weights, pass, and are
kept; the BFP8-attention fallback is measured at +3.3% and left as a one-argument switch.

Compute-kernel flags were swept separately from dtype: `packer_l1_acc=False` costs 26%
(1.343 vs 1.064 ms) and `fp32_dest_acc_en=True` costs 2%.

## 4. DRAM-sharded decode matmul geometry

Reading `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp` changed the model
of this op materially, and the numbers follow from it:

* the op picks its **own worker set** — 12 cores on this p300c, from
  `get_optimal_dram_bank_to_reader_assignment` — and that is *not* the program config's core
  count, nor the DRAM bank count. The factory happens to name its local variable
  `num_dram_banks` (`num_dram_banks = all_worker_cores_ordered.size()`), which is where the
  two got conflated: the bank count is 8 (`mesh_device.dram_grid_size().x`, the grid the
  weights are width-sharded over and the value `_prefill_grid_x` must match), the reader
  worker count is 12, and the profiler's `Cores` column reports the latter;
* `per_core_N_compute = ceil(N_tiles / workers)` — the program config's `per_core_N` only
  sets the output *storage* shard. The layer's L1 model deliberately uses 8 rather than 12
  as the divisor, which over-estimates `per_core_N_compute` and therefore the weight
  circular buffer, so the `in0_block_w` it accepts is always legal and never optimistic;
* the weight circular buffer is **triple** buffered:
  `3 * per_core_N_compute * in0_block_w * dram_aligned_tile_bytes`.

That last line is why the K block is dtype-dependent, and why the first attempt died:

```
TT_THROW: Statically allocated circular buffers on core range [0-0 - 7-9]
grow to 2080640 B which is beyond max L1 size of 1572864 B
```

at `in0_block_w=13`, `per_core_N=39` for the SwiGLU gate/up projection on the 16-core
residual grid. The layer now derives `in0_block_w` from an L1 budget model built on the
factory's own arithmetic, sweeping downward through the legal divisors of the per-core K
tile count; at BF16 the SwiGLU projection affords 2, at BFP4 it affords 8.

The residual grid cannot host the SwiGLU projections at all —
`gcd(208, 624) = 208` allows many more cores than `gcd(208, 144) = 16` does — so the
SwiGLU block gets a **phase-specific working shard** (OPT-011), resharded in and out:

| SwiGLU working cores | in0_block_w | traced decode ms |
|---|---|---|
| 8 | 8 | 1.2199 |
| 13 | — | L1 overflow |
| 16 (the residual grid) | 1 | 1.5056 |
| 26 | 8 | 1.0798 |
| **52** | **4** | **1.0643** |
| 104 | 2 | 1.1160 |

Per-role `in0_block_w` was then swept against the budget-derived defaults; the defaults win
every role (full table in [`evidence_tables.md`](evidence_tables.md)). The costs of getting
it wrong are large: `in0_block_w=1` on the SwiGLU projections costs 37%, on QKV/gate 17%.

Every dominant decode matmul ends at `in0_block_w` 13 (QKV, gate), 8 (output projection),
4 (SwiGLU gate/up) and 12 (SwiGLU down) — none at the 1-2 the skill flags. The 4 on
gate/up is the largest legal divisor of `208 / 52 = 4`; buying a larger one means fewer
cores, which the table above shows losing.

## 5. Prefill: two silent-wrong-answer bugs

### 5.1 `per_core_N * grid_x > N_tiles` is accepted

A grid whose width does not divide the tiled N is accepted by
`MatmulMultiCoreReuseMultiCastProgramConfig` and produces wrong results. Caught by
`test_paged_prefill_decode_pcc[32-sliding_rope]`: prefill PCC 0.692 instead of 0.999.

### 5.2 DRAM width-sharded weights require `grid_x == num_dram_banks`

Fixing 5.1 by pinning `grid_x` to a divisor of the tiled N was **not enough**: the QKV
projection's 144 tiles divide by 9 exactly, and a 9-wide grid still produced NaN — layer
PCC 0.765, paged K cache entirely NaN, at every sequence length. A standalone matmul at the
same shape, dtype, program config and grid was *correct*, which narrowed it to the one
remaining difference: the layer's weights are DRAM width-sharded and the standalone repro's
were interleaved.

[`../../scripts/repro_prefill_matmul_grid_x9.py`](../../scripts/repro_prefill_matmul_grid_x9.py)
isolates it; the committed run is [`logs/repro_prefill_matmul_grid_x9.log`](logs/repro_prefill_matmul_grid_x9.log):

```
DRAM banks=8, in1 shard width=576 elements = 18 tiles
           in1 / input shape      grid  per_core_N  subblock_w   covers N         PCC
        interleaved[1,1,512]    8x8             18           6      exact    0.999826
        interleaved[1,1,512]    9x8             16           8      exact    0.999826
        interleaved[1,1,512]   10x8             15       OVER    0.999826
        dram_sharded[1,1,512]    8x8            18           6      exact    0.999826
        dram_sharded[1,1,512]    9x8            16           8      exact         nan
        dram_sharded[1,1,512]   10x8            15       OVER         nan
```

So: **a 2D multicast matmul with a DRAM width-sharded `in1` silently returns NaN unless
`per_core_N` equals the in1 DRAM shard width in tiles**, i.e. unless
`grid_x == num_dram_banks`. Both this and 5.1 want a validation check in the matmul
program-config validator. `_prefill_grid_x` pins `grid_x` to the DRAM bank count, which
makes both unreachable.

This one is worth dwelling on: the illegal 9-wide and 10-wide grids measured ~5% *faster*
than the legal one in a wall-clock sweep, and were nearly selected on that basis. Only the
PCC suite caught it. A wall-clock sweep without a correctness gate in the same loop is not
evidence.

### 5.3 The row fold

A single `M = 8192` matmul against the 19968-wide SwiGLU projection needs megabytes of
output circular buffer per core, so the sequence is folded into `block_rows`-row blocks
along a leading batch dimension. The first implementation required `rows % 512 == 0`, which
blew L1 for every non-aligned length:

```
TT_THROW: Statically allocated circular buffers on core range [0-0 - 7-4]
grow to 2411008 B which is beyond max L1 size of 1572864 B
```

(2080 rows = 65 tiles → `grid_y = 5`, `per_core_M = 13`, `per_core_N = 78`.) Eight tests
failed: 2080, 3000, 12345 and the ragged-slot test, on both layer kinds. `_prefill_fold`
now searches the legal folds, zero-pads the row count up to a multiple of the chosen block
when the length has no useful divisor (3008 rows is 2 x 47 tiles, whose only exact folds
are 1 and 2 tiles, i.e. 8 or 16 of 110 cores), and slices the padding off the output. The
padding is mathematically inert — zero activation rows produce zero output rows, removed
before anything else in the layer sees them — and it does not touch the page table or the
cache, unlike padding the prompt itself would.

Candidates are ranked by `grid_x * grid_y * rows / padded_rows`, filtered first to those
that still support the largest legal `in0_block_w`: a fold with a big `per_core_M` can only
afford `in0_block_w=1`, which costs far more than the extra cores gain. That filter matters
— an earlier scoring that maximised cores alone picked `per_core_M = 1` and measured
80.4 ms against 54.1 ms.

### 5.4 Prefill grid height

`grid_y = 8` is a sharp optimum on this Blackhole grid, not a Wormhole habit:

| grid_y cap | prefill 8192 ms |
|---|---|
| 2 | 134.03 |
| 4 | 78.91 |
| 6 | 111.82 |
| **8** | **54.23** |
| 10 | 80.36 |

Fold cutoff: 512 rows (54.12 ms) beats 1024 (57.77) and 2048 (57.79).

## 6. Stage-review follow-ups

An independent review of §1-§5 returned `more-work-needed` on six items. What changed:

### 6.1 The decode matmul family rejection was not earned

The review's headline finding: the "a wider worker set is not expressible" conclusion in
`perf/accounting.json` rested on a comparison against `ttnn.linear` with **no program
config at all**. `$optimize` OPT-004/OPT-014 require an explicitly configured
alternative-geometry candidate under the *same* dtype/fidelity before a dominant `SLOW` row
is accepted.

Built and measured (`decode_matmul="mcast1d"`): an explicit
`MatmulMultiCoreReuseMultiCast1DProgramConfig` per role — `mcast_in0=True`, in0 L1
width-sharded, DRAM-interleaved weights, per-role `in0_block_w` from the same L1 budget
model, BFP4/LoFi.

| decode matmul family | cores | traced decode ms |
|---|---|---|
| DRAM-sharded (selected) | 12, op-chosen | **1.0644** |
| 1D multicast, 8x2 = 16 cores | 16 | 1.2311 |
| 1D multicast, 4x4 = 16 cores | 16 | 1.2532 |
| 1D multicast, 8x1 = 8 cores | 8 | L1 overflow |
| no program config, DRAM-interleaved weights | ~104 | 1.4019 |

So more than 12 compute cores *is* expressible for these shapes, and it loses by 16%. 16
cores is the ceiling for that family: `num_cores` must divide both the tiled K and the
tiled N, `gcd(208, 144) = gcd(208, 128) = gcd(128, 208) = 16`, and 26/52/104 have no
rectangle inside an 11x10 grid. The accounting text now says that instead of
"not expressible".

### 6.2 `PREFILL_IN0_BLOCK_W_CAP = 8` was an unmeasured constant

The three attention prefill rows sat at an 8-tile K block while their tiled K (208 and 128)
has 13, 16 and 26 as legal divisors that the L1 model says fit. Swept:

| cap | prefill 8192 ms |
|---|---|
| 8 (original) | 53.94 / 53.86 |
| 13 | 52.91 |
| 16 | 52.84 |
| **26** | **52.55 / 53.10** |
| 52 | 62.21 |
| 104 | 62.29 |
| 208 | 62.29 |

26 is now the default. In the final committed report the per-role K blocks are 26 (QKV),
26 (attention gate), 16 (output projection), 8 (SwiGLU gate/up, where the L1 budget stops
it) and 26 (SwiGLU down); four of the five rows are `FLOP`-bound rather than `SLOW`, and the
down projection went from 71.4% to 77.6% of the LoFi peak. Above cap 26 the attention rows
regress sharply.

### 6.3 The tilize claim was false

`README.md` claimed no tilize appears in the committed reports;
`tracy/sliding_rope/decode_1_perf_report.csv` has 2 `TilizeWithValPaddingDeviceOperation`
per token, 11.2 µs, 1.07% of the batch-1 decode step. It is the
`ttnn.embedding` -> `rotary_embedding_hf` layout boundary in `_rope_decode_mats` (the decode
cos/sin tables must be ROW_MAJOR for the gather and TILE for the rotary kernel), inherited
unchanged from the functional stage at 135.0 µs; only its *share* changed, because
everything around it got 3x faster. It is absent from the batch-32 report, where 32 users
fill the tile row.

`decode_rope_pad_to_tile` (now on by default) gathers a whole tile row of positions instead,
and the op **disappears from the report**: the RoPE metadata path is now Embeddings 5.2 µs +
Transpose 8.3 µs + Slice 3.2 µs per token against the previous Embeddings + Tilize +
Transpose 17.7 µs — the same total, with no tilize. Whole-step latency is a tie (1.0630 vs
1.0640 ms). The audit paragraph now describes how it was removed rather than asserting it was
never there.

### 6.4 The SwiGLU prefill asymmetry is classified

See *Anomalies* in [`README.md`](README.md). Summary: in `sliding_rope/prefill_8192` the ten
`6656 x 19968` rows measure `8943/9577  8948/8945  8943/8937  8936/9773  8936/10034` µs -
seven at 8936-8948 and three at 9577-10034, the three carrying `SLOW`. It is sporadic, not
strictly positional: two of the five iterations are symmetric. Controls: all ten rows are
stable at 4096 on the same path (4468-4473) and all ten are stable at 8192 on the
full-attention path (8937-8953); the down projection is stable everywhere. The sliding path
at 8192 is the only one that allocates and frees the non-chunked SDPA's 8192-token Q/K/V
immediately before the SwiGLU block, whose two projections then allocate 327 MB outputs back
to back. The reading is DRAM allocator/refresh state, not compute - which is why it is
intermittent and why the down projection, running after both intermediates are freed, never
shows it. Both candidate mitigations lose (packed gate/up 56.6 vs 53.2 ms; sub-blocking adds
a concat per chunk), so it stays inside the headline number rather than being worked around.
The two documentation claims it contradicted are corrected.

### 6.5 The context contract was advertising stage-01 evidence

`doc/context_contract.json` claimed 131072 with a `test_functional_decoder.py` reference,
for a prefill path this stage rewrote. `test_full_context_prefill_and_decode` now exists in
the optimized suite and passes on both layer kinds: prefill at 131072 and at 131071, decode
at position 131071, prefill PCC 0.9982-0.9984, K-cache 0.9998, decode 0.9986-0.9990. The
reference is driven with a query-block filter (first, middle, last block of each prompt) so
one 131072-token host reference per kind stays affordable; every PCC record carries its
coverage string, and the shorter lengths keep full every-position coverage.

### 6.6 DRAM banks and DRAM readers were conflated

The docs used "DRAM bank count" for both 8 (`mesh_device.dram_grid_size().x`, the grid the
weights are width-sharded over and the value `_prefill_grid_x` must match) and 12 (the
worker set `get_optimal_dram_bank_to_reader_assignment` picks, and the `Cores` column of
every dominant decode row). They are now named separately. The roofline constants are also
cited rather than back-computed: 512 GB/s is
`tt_perf_report.ArchitectureSpec("blackhole").dram_bandwidth_gb_s` — chip-wide, the same
constant its `DRAM %` column divides by, not worker-relative — and 5.5296 TFLOP/s per core
is its LoFi `tflops_per_core`.

### 6.7 Math fidelity was only swept as a whole-policy switch

The review noted that `PrecisionPolicy` carries `attn_fidelity`, `mlp_fidelity` and
`mlp_down_fidelity` separately but §3 only compared LoFi against HiFi2 at BFP8 weights.
Fidelity is a knob independent of dtype, so it is now isolated per projection group at the
shipped BFP4 weights (`bfp4_all_hifi2_attn`, `bfp4_all_hifi2_mlp`):

| candidate | traced decode ms | warmed prefill 8192 ms |
|---|---|---|
| **LoFi everywhere (shipped)** | **1.0632** | **52.86** |
| HiFi2 on the attention projections | 1.1950 (+12%) | — |
| HiFi2 on the SwiGLU gate/up projections only | 1.4812 (+39%) | 53.02 |
| HiFi2 on the SwiGLU down projection only | 1.2755 (+20%) | 52.99 |
| HiFi2 on both SwiGLU projections | 1.6752 (+58%) | — |
| HiFi2 on the prefill attention projections only | — | 56.87 (+7.6%) |
| HiFi2 on the prefill SwiGLU projections only | — | 71.58 (+35%) |
| HiFi2 on all prefill matmuls | — | 74.86 (+42%) |

Every group is isolated, in both phases: gate/up separately from down (the second review
noted the first pass moved them together), and each prefill group separately from the
aggregate. LoFi is the right pairing for BFP4 everywhere by a wide margin, which is what the
expected pairing predicted but had not been measured for this model.

### 6.8 The re-stamped 131072 evidence, and a length-dependence finding

The second review found the re-stamped context evidence measured at `bfp8_all_lofi`, not the
shipped `bfp4_all`, with quoted PCC ranges that appear nowhere in the artifact. Both are
fixed: `test_full_context_prefill_and_decode` is now parameterised over both policies (four
cases, all passing), every quoted number is regenerated from `pcc/pcc_results.json`, and the
contract records which policy each row came from.

Running it at BFP4 surfaced something the first pass had not looked for. On **synthetic**
weights the `full_nope` prefill PCC drifts with length — 0.9576 / 0.9555 / 0.9498 at 1000 /
8192 / 32768 tokens for the last query block. That is the one failure mode a fixed-length PCC
bar cannot see: a `full_attention` layer attends over the whole paged prefix, so error in the
cached K/V compounds with the number of keys in the softmax, while a sliding layer never sees
past its 2048-token window.

`scripts/probe_length_dependence.py` isolates it
([`pcc/length_dependence.json`](pcc/length_dependence.json)). Two findings:

1. it is **BFP4 attention weights**, not the MLP: `bfp8_attn_bfp4_mlp` — which differs from
   `bfp4_all` only there — is flat at 0.99149 / 0.99131 / 0.99097 over the same lengths;
2. it **does not happen on real weights**: `bfp4_all` measures 0.99887 / 0.99886 / 0.99867 at
   1000 / 8192 / 12345, a drift of 0.0002 against 0.0079 on synthetic, and 25x inside the bar.

So the shipped policy stands, and the OPT-012 discipline that selected it now also covers the
length axis. `test_real_weights_length_independence` gates it for both the shipped and the
conservative policy: both lengths at 0.995 absolute, and the long length within 0.005 of the
short one. The synthetic long-context assertion is a length-*shape* check against its own
1000-token measurement rather than an absolute floor, because an absolute floor on synthetic
BFP4 would be arbitrary.

### 6.9 The declared noise floor was wrong, and two selections rested on it

The claimed +-0.17% spread was contradicted by rows in the same file. Six repeats of the
shipped configuration in one process measure 1.0632 / 1.0792 / 1.0776 / 1.0775 / 1.0806 /
1.0764 ms — **1.6% peak-to-peak**, with the first repeat consistently fastest, i.e. a
within-process warm-up drift larger than several reported differences. Sweep position
therefore biases a single measurement by more than a 1-2% effect.

The two selections inside that band were re-measured **paired** (A/B/A/B), which cancels the
drift:

| paired comparison | pairs | effect |
|---|---|---|
| SwiGLU working cores 52 vs 26 | 1.0638/1.0886, 1.0811/1.0953, 1.0780/1.0951 | 26 slower by 1.42%, 1.31%, 1.59% — consistent |
| prefill `in0_block_w` cap 26 vs 8 | 53.334/54.360, 53.385/54.294 | cap 8 slower by 1.92%, 1.70% — consistent |

Both selections hold. Everything else under ~1.7% is now reported as a tie rather than a win,
and the README lists which results those are.

### 6.10 `out_subblock_h` was hard-coded, and the stated blocker was wrong

`out_subblock_h = 1` for every prefill matmul, never swept, and the README explained the
output projection's `SLOW` marker with a claim about `out_subblock_w` divisors that only holds
at `h = 1`. `prefill_out_subblock_h` is now a knob and was swept at the shipped policy:
`h=1` 52.86 ms, `h=2` 53.14 ms, `h=4` 53.34 ms at 8192 tokens. `h=1` stays — the choice was
right, the recorded reason was not, and the difference is inside the noise floor either way.

### 6.11 The rejected 1D matmul candidate was handicapped

`_mcast1d_pc` capped its output subblock at 4 tiles even though decode runs with
`fp32_dest_acc_en=False`, where 8 is available. Raised to 8 and re-measured: 1.2282 ms
against 1.2311 ms before, still 15% behind the DRAM-sharded 1.0644. The candidate is also now
PCC-gated by the sweep harness (0.9665 prefill / 0.9596 decode on synthetic, matching every
other BFP4 candidate), so the rejection rests on a candidate that was computing the right
answer.

Also from the review, smaller: the shadowed duplicate `_prefill_in0_block_w` definition was
removed (the two copies differed in their no-fit return value, which `_prefill_fold`
depends on); a stale comment naming the wrong default policy was fixed; the decode SDPA
output memory config gained an L1 candidate (a tie, kept because it is the rule-consistent
choice); a real-weight batch-4 test was added
(`test_real_weights_batched`), since OPT-012 lists larger batch among the conditions a
synthetic-only precision failure must be re-checked under; the sweep harness gained a
``PCC_SEQ`` correctness gate so no future candidate table is wall-clock-only; the QK RMSNorms
are named as the one unsharded norm pair with their 8 µs cost; and the `Slice`/`Pad` ops the
new RoPE gather introduced are accounted for in the data-movement audit.

## 7. Evidence collected

* **Correctness**: 105 tests, all passing, both layer kinds
  ([`logs/full_suite.log`](logs/full_suite.log), PCC records in
  [`pcc/pcc_results.json`](pcc/pcc_results.json)). Every record carries a code fingerprint
  over the optimized layer, the functional layer, the host reference and the test files;
  `render_optimized_evidence.py` fails if any record is stale.
* **Performance**: `tt-perf-report` tables, filtered CSVs, summaries with advice, and the
  gzipped raw Tracy ops CSV for prefill 4096/8192 and decode batch 1/32 on both layer kinds
  ([`tracy/`](tracy)), plus wall clock in [`perf/perf_summary.json`](perf/perf_summary.json).
* **Candidate table**: [`perf/candidates.json`](perf/candidates.json), 11 sections, 66
  measured candidates.
* **Precision sweep**: [`pcc/policy_sweep.json`](pcc/policy_sweep.json) — every named policy
  on synthetic *and* real weights, both layer kinds, with the dtype/fidelity each one
  actually resolved to.
* **Length-dependence probe**: [`pcc/length_dependence.json`](pcc/length_dependence.json) —
  last-query-block prefill PCC per policy at 1000/8192/32768 synthetic and 1000/8192/12345
  real, which is what established that BFP4's synthetic length drift does not exist on real
  weights (see §6.8).
* **Accounting**: [`perf/accounting.json`](perf/accounting.json) — roofline, device time and
  end-to-end from the same runs, with named limitations.
* **Watcher**: 54 tests under `TT_METAL_WATCHER=10`, zero findings
  ([`watcher/WATCHER_SUMMARY.md`](watcher/WATCHER_SUMMARY.md)).
* **Stress**: `test_stress_repeated_prefill_decode` — three cycles of prefill + 8 decode
  steps on real weights, PCC asserted every cycle and outputs asserted bit-identical
  across cycles.

### `tt-perf-report` advice, and what was done with it

Advice is left enabled in the committed `*_perf_report.summary.txt`. The remaining
recommendations on the optimized reports and their disposition:

| advice | on | disposition |
|---|---|---|
| `in0_block_w=1 is small, try in0_block_w=2 or above` | gone from every row | **fixed** — the smallest surviving value on any dominant row is 4 (decode SwiGLU gate/up) and 8 (prefill SwiGLU gate/up); the rest are 8-26 |
| `If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)` | gone from every decode row | **fixed** — decode in0 is `L1_WIDTH_SHARDED` everywhere |
| `If possible place input 0 in L1` | still on the prefill matmul rows | **rejected with reason**: prefill activations at 8192 x 6656 are 104 MB in BF16 and cannot be held in the 157 MB of aggregate L1 alongside the weights and intermediates; the skill's own guidance is that prefill activations belong in DRAM interleaved |
| `No output subblock size found` | the DRAM-sharded decode rows | **not actionable**: the DRAM-sharded factory computes its own subblocks from `per_core_N_compute`; the field is not exposed on the program config |
| `Use HiFi2 or HiFi4 with BF16 activations for improved accuracy` | the LoFi rows | **rejected with evidence**: this is accuracy advice on a path measured at 0.9970-0.9990 real-weight PCC against a 0.995 bar, and HiFi2 costs 32% (§3) |

One more advice item appears in the prefill summaries and was omitted from an earlier
version of this table: `High Op-to-Op Gap`, self-reported at 0.0% of the window (the summed
gap is 22 µs per 51 ms prefill iteration and 50 µs per 1.06 ms decode window). Immaterial,
and the end-to-end/device reconciliation in §7 already accounts for the host term.

The `SLOW` marker remains on the three attention prefill rows (59-63% of the LoFi FLOP
peak), on the second SwiGLU projection of the sliding path at 8192 (see §6.4), and on all
five decode rows. On the decode rows it reflects the 12-core DRAM-reader worker set, which
§6.1 measured an explicit larger-grid alternative against.

## 8. Exact commands

```bash
# the full-context contract on the optimized path (§6.5, ~4 min)
python -m pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_optimized_decoder.py -q \
  -k long_context -s

# precision policy sweep, synthetic and real weights (the §3 table)
python models/autoports/meta_models_muse_glimmer_30b/scripts/sweep_optimized_precision.py

# correctness (97 tests, ~7 min)
python -m pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_optimized_decoder.py -q

# performance artifacts, one Tracy session per (mode, kind, size)
for kind in sliding_rope full_nope; do
  for spec in "decode 1" "decode 32" "prefill 4096" "prefill 8192"; do
    set -- $spec
    bash models/autoports/meta_models_muse_glimmer_30b/scripts/collect_optimized_perf.sh $1 $kind $2
  done
done

# unprofiled wall clock (the honest end-to-end numbers)
python -m pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_optimized_decoder_perf.py -q

# watcher (separate run, never with the profiler)
WD=models/autoports/meta_models_muse_glimmer_30b/doc/optimized_decoder/watcher
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
  TT_METAL_LOGS_PATH=$PWD/$WD \
  python -m pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_optimized_decoder.py -q \
    -k "paged_prefill_decode_pcc or page_block_sizes or batched or ragged or short_prefill or \
        multichunk or continued or sliding_window or traced or real_weights or stress or \
        determinism or no_runtime_host_fallback"

# the TTNN matmul repro
python models/autoports/meta_models_muse_glimmer_30b/scripts/repro_prefill_matmul_grid_x9.py

# regenerate the evidence tables (exits non-zero if the artifacts disagree with the code)
python models/autoports/meta_models_muse_glimmer_30b/scripts/render_optimized_evidence.py
```

The precision and geometry sweeps were driven by
[`../../scripts/sweep_optimized_decoder.py`](../../scripts/sweep_optimized_decoder.py),
which takes a JSON list of `from_state_dict` overrides and measures each candidate in one
process on one device:

```bash
CANDIDATES='[{"label":"bfp4_all","precision":"bfp4_all"},
             {"label":"bfp8_lofi","precision":"bfp8_all_lofi"}]' \
KINDS=sliding_rope DECODE_ITERS=32 CONTEXT=4096 PREFILL_SEQ=8192 \
  python models/autoports/meta_models_muse_glimmer_30b/scripts/sweep_optimized_decoder.py
```

## 9. Checkpoint

Stage-owned changes committed locally; nothing pushed. The commit SHA is recorded in §9
after the fact, because the artifacts this log describes necessarily precede the commit
that contains them.

## 10. Commit SHAs

* `48dd8e2caa7` — optimized decoder, tests, sweep/collect/repro scripts, docs and evidence.
* `df5c3988f12` — the seven stage-review items in §6, their measurements, and the
  regenerated artifacts.
