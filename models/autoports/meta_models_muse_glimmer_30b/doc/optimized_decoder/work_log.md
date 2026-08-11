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
| `bf16_baseline` | 0.999435 / 0.999369 | 0.999036 / 0.999431 | 0.999921 / 0.999946 | 0.999701 / 0.999956 | 2.2020 |
| `bfp8_all_hifi2` | 0.999339 / 0.999197 | 0.998824 / 0.999154 | 0.999918 / 0.999942 | 0.999378 / 0.999956 | 1.8454 |
| `bfp8_all_lofi` | 0.998985 / 0.998785 | 0.998325 / 0.998494 | 0.999906 / 0.999939 | 0.999690 / 0.999948 | 1.2540 |
| `bfp8_attn_bfp4_gateup` | 0.995402 / 0.994298 | 0.994665 / 0.993967 | 0.999591 / 0.999677 | 0.999562 / 0.999459 | 1.1534 |
| `bfp8_attn_bfp4_mlp` | 0.993672 / 0.992139 | 0.992878 / 0.991679 | 0.999166 / 0.999265 | 0.999130 / 0.998838 | 1.0987 |
| **`bfp4_all`** | 0.968571 / 0.963860 | 0.969769 / 0.960209 | **0.998612 / 0.998959** | **0.996970 / 0.998435** | **1.0639** |

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

* the **worker set is one core per DRAM bank** (12 on this p300c, per
  `get_optimal_dram_bank_to_reader_assignment`), *not* the program config's core count;
* `per_core_N_compute = ceil(N_tiles / num_dram_banks)` — the program config's
  `per_core_N` only sets the output *storage* shard;
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

## 6. Evidence collected

* **Correctness**: 97 tests, all passing, both layer kinds
  ([`logs/full_suite.log`](logs/full_suite.log), 241 PCC records in
  [`pcc/pcc_results.json`](pcc/pcc_results.json)). Every record carries a code fingerprint
  over the optimized layer, the functional layer, the host reference and the test files;
  `render_optimized_evidence.py` fails if any record is stale.
* **Performance**: `tt-perf-report` tables, filtered CSVs, summaries with advice, and the
  gzipped raw Tracy ops CSV for prefill 4096/8192 and decode batch 1/32 on both layer kinds
  ([`tracy/`](tracy)), plus wall clock in [`perf/perf_summary.json`](perf/perf_summary.json).
* **Candidate table**: [`perf/candidates.json`](perf/candidates.json), 8 sections, 45
  measured candidates.
* **Precision sweep**: [`pcc/policy_sweep.json`](pcc/policy_sweep.json) — every named policy
  on synthetic *and* real weights, both layer kinds, with the dtype/fidelity each one
  actually resolved to.
* **Accounting**: [`perf/accounting.json`](perf/accounting.json) — roofline, device time and
  end-to-end from the same runs, with named limitations.
* **Watcher**: 52 tests under `TT_METAL_WATCHER=10`, zero findings
  ([`watcher/WATCHER_SUMMARY.md`](watcher/WATCHER_SUMMARY.md)).
* **Stress**: `test_stress_repeated_prefill_decode` — three cycles of prefill + 8 decode
  steps on real weights, PCC asserted every cycle and outputs asserted bit-identical
  across cycles.

### `tt-perf-report` advice, and what was done with it

Advice is left enabled in the committed `*_perf_report.summary.txt`. The remaining
recommendations on the optimized reports and their disposition:

| advice | on | disposition |
|---|---|---|
| `in0_block_w=1 is small, try in0_block_w=2 or above` | gone from every row | **fixed** — the smallest surviving value on any dominant row is 4 |
| `If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)` | gone from every decode row | **fixed** — decode in0 is `L1_WIDTH_SHARDED` everywhere |
| `If possible place input 0 in L1` | still on the prefill matmul rows | **rejected with reason**: prefill activations at 8192 x 6656 are 104 MB in BF16 and cannot be held in the 157 MB of aggregate L1 alongside the weights and intermediates; the skill's own guidance is that prefill activations belong in DRAM interleaved |
| `No output subblock size found` | the DRAM-sharded decode rows | **not actionable**: the DRAM-sharded factory computes its own subblocks from `per_core_N_compute`; the field is not exposed on the program config |
| `Use HiFi2 or HiFi4 with BF16 activations for improved accuracy` | the LoFi rows | **rejected with evidence**: this is accuracy advice on a path measured at 0.9970-0.9990 real-weight PCC against a 0.995 bar, and HiFi2 costs 32% (§3) |

The `SLOW` marker remains on three of the five prefill matmul rows (the three attention
projections) and on all five decode rows. On the decode rows it reflects the 12-core DRAM
worker set, not a config choice — see the README's *Performance accounting*. On the prefill
attention rows it reflects 59-63% of the LoFi FLOP peak; the two dominant SwiGLU rows are
no longer `SLOW` at all.

## 7. Exact commands

```bash
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

## 8. Checkpoint

Stage-owned changes committed locally; nothing pushed. The commit SHA is recorded in §9
after the fact, because the artifacts this log describes necessarily precede the commit
that contains them.

## 9. Commit SHAs

* `TBD` — optimized decoder, tests, sweep/collect/repro scripts, docs and evidence.
