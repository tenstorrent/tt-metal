# AUTODEBUG: Gemma-4 optimized-decoder performance gate

Date: 2026-07-29
Scope: source-only review of the staged optimized single-device decoder, its tests, and
`doc/optimized_decoder` evidence. No device work was performed.

## Verdict

The optimized stage is not complete. The final decode profile still runs the inherited
functional residual/RMS chain in interleaved DRAM. Seven hidden-width DRAM-input norms
consume **300.815 us / 23.95%** of device time. Including Q/K/V and the post-expert
L1-input norms, norms consume **358.252 us / 28.52%**. This directly confirms the
OPT-003 failure and makes a coherent sharded residual chain the first required
intervention.

`OptimizedDecoder` does not override `decode_forward` or `_rms_norm`.
`functional_decoder.py::decode_forward` repeatedly invokes `_rms_norm`, performs
residual adds with DRAM output, and `_rms_norm` explicitly requests DRAM output with
the correctness compute config. This is not an inference from timing alone; it is the
executed staged source path.

The prior `GEMMA4_OPT_DENSE_DECODE_DRAM_SHARDED=1` candidate does not refute this
hypothesis. `_linear` independently converts each input to a sharded layout and each
output back to DRAM. It is an isolated-matmul experiment, from an earlier unpacked
source state, rather than a persistent activation chain.

The independently produced `stage_review.md` reaches the same `more-work-needed`
verdict and identifies this norm/residual topology as its leading P1 item.

## Direct profile observations

Source: `tracy_final_retry/decode_perf_report.csv`, sliding layer 0, batch 1,
position 1024, traced decode. Device-op total is 1255.961 us.

| Rows | Device time | Share | Relevant contract |
|---|---:|---:|---|
| 7 hidden-width DRAM-input RMS norms | 300.815 us | 23.95% | one core, default program config, DRAM output |
| Q/K/V plus post-expert L1-input RMS norms | 57.437 us | 4.57% | still one core/default program |
| All norms | 358.252 us | 28.52% | performance gate failure |
| Hidden residual adds | 6.653 us | 0.53% | DRAM output; more important as chain-breaking boundaries |
| Sparse expert gate/up pair | 176.241 us | 14.03% | both read the same DRAM-interleaved expert input |

The final candidate's important dense rows are:

| Role | Tiled `K x N` | Current geometry | Time |
|---|---:|---|---:|
| sliding QKV | `88 x 256` | 86 cores, `in0_block_w=2`, `per_core_N=3`, subblock `1x3` | 113.801 us |
| sliding output projection | `128 x 88` | 88 cores, block 2, `per_core_N=1`, subblock `1x1` | 87.156 us |
| packed shared gate/up | `88 x 132` | 66 cores, block 2, `per_core_N=2`, subblock `1x2` | 53.117 us |
| shared down | `66 x 88` | 88 cores, block 2, `per_core_N=1`, subblock `1x1` | 46.127 us |

The packed shared MLP rows are BFP8/HiFi2. No recorded candidate exercises the dense
MLP dtype/fidelity environment controls.

## Ranked hypotheses and smallest experiments

### 1. Keep the hidden residual stream sharded across the whole decode layer

**Confidence: definitive cause, highest expected impact.**

The physical decode activation is BF16 `[1, 1, 32, 2816]`: 88 width tiles and
180,224 bytes. Two exact width-sharded residual contracts on the P300 worker grid are:

| Candidate | Grid | Shard shape | Bytes/core | RMS program in tiles |
|---|---|---:|---:|---|
| R11 | `11x1` | `[32, 256]` | 16 KiB | `block_h=1`, `block_w=8`, `subblock_w=4` |
| R22 | `11x2` | `[32, 128]` | 8 KiB | `block_h=1`, `block_w=4`, `subblock_w=4` |

Both batch 1 and batch 32 have physical height 32, so this contract covers both decode
fixtures. Use width-sharded L1, row-major shard orientation, and a
`LayerNormShardedMultiCoreProgramConfig`; keep the norm output in the input memory
configuration. R11 is the lowest-core candidate. R22 offers more norm parallelism and
an exact shared-MLP split.

The code intervention should be an optimized decode path, not a blanket change to
functional `_rms_norm`:

1. Convert the decode activation to R11 or R22 once at layer entry.
2. Keep hidden RMS outputs, residual adds, scalar residual scaling, and branch merges
   in that same memory configuration.
3. For the shared MLP, first use **separate** gate/up projections. R11 has
   `K=8`, `N=6` tiles/core for gate/up and `K=6`, `N=8` for down. R22 has
   `K=4`, `N=3` for gate/up and `K=3`, `N=4` for down. GELU and multiply must retain
   the shared intermediate shard. The current packed output joins two 66-tile halves;
   an L1-only slice/paired-core exchange must be demonstrated before packed mode can
   be called a coherent chain.
4. At attention, narrowly reshard to an 8-core width grid: hidden input is 11
   tiles/core; sliding/full QKV output is 32/40 tiles/core; sliding/full output
   projection has `K=16/32` and returns 11 hidden tiles/core. Reshard the projection
   output back to R11/R22 without an interleaved-DRAM round trip. Head-layout
   conversions may remain local L1 boundaries.
5. Treat router and sparse experts as explicit branches. Convert the normalized
   expert input once at the sparse boundary if the sparse kernels cannot consume the
   residual shard, then return the expert output to R11/R22 once. Do not convert
   around each RMS norm.
6. Convert out only at the externally required layer/test boundary.

**Smallest experiment:** implement R11 and R22 as two otherwise identical decode
candidates and profile one complete traced layer. An isolated
`interleaved_to_sharded -> RMS -> sharded_to_interleaved` microprobe is useful only
to select the norm grid; it is not acceptance evidence.

Required evidence:

- no hidden-width one-core/default RMS rows;
- no hidden RMS or residual-add output in interleaved DRAM;
- an enumerated entry, attention, expert, and exit conversion count;
- real-weight PCC for sliding and full attention, traced batch 1 and batch 32, both
  natural and shared cache paths;
- whole-layer latency, not just the norm-kernel sum.

### 2. Retune dense geometry, then sweep dense precision under the same geometry

**Confidence: high that coverage is missing; performance winner requires hardware
measurement.**

The final dense matmuls use small default blocks and subblocks. Start with only two
regular-1D geometry alternatives per shared-MLP role:

| Role | Candidate A | Candidate B |
|---|---|---|
| packed gate/up `K88,N132` | 44 cores, block 11, `per_core_N=3`, subblock `1x3` | 22 cores, block 22, `per_core_N=6`, subblock `1x3` |
| down `K66,N88` | 44 cores, block 6, `per_core_N=2`, subblock `1x2` | 22 cores, block 11, `per_core_N=4`, subblock `1x4` |

Measure each role alone, then the full shared MLP. If the coherent residual experiment
is ready, prefer its exact R11/R22 split geometry over these interleaved candidates.
The existing single `GEMMA4_OPT_DRAM_BLOCK_W` is insufficient: QKV, projection,
gate/up, and down have different legal tiled K values, so retained DRAM-sharded
experiments need role-specific block/program controls.

With the winning geometry fixed, run this minimal existing-control matrix:

```text
# Current precision, lower fidelity
GEMMA4_OPT_MLP_WEIGHT_DTYPE=bfp8
GEMMA4_OPT_MLP_FIDELITY=lofi
GEMMA4_OPT_PACKED_DENSE_GATE_UP=1

# Lower precision and fidelity
GEMMA4_OPT_MLP_WEIGHT_DTYPE=bfp4
GEMMA4_OPT_MLP_FIDELITY=lofi
GEMMA4_OPT_PACKED_DENSE_GATE_UP=1
```

If all-BFP4 fails correctness, it does not reject BFP4 gate/up with BFP8 down. Add
separate gate/up and down dtype/fidelity controls and test:

```text
gate/up=BFP4+LoFi, down=BFP8+LoFi
gate/up=BFP8+LoFi, down=BFP4+LoFi   # guarded down-only candidate
```

Every result must show the runtime profiler row's actual weight dtype, compute
fidelity, grid, block width, per-core N, and subblock. Do not infer that an
environment variable took effect.

After the dense sweep, cover attention precision independently rather than treating
the combined BFP4/LoFi rejection as two rejections. The bounded matrix is BFP8 at the
current fidelity followed by BFP8/LoFi, using
`GEMMA4_OPT_ATTENTION_WEIGHT_DTYPE=bfp8`,
`GEMMA4_OPT_ATTENTION_FIDELITY={hifi4,lofi}` for sliding attention, and
`GEMMA4_OPT_FULL_ATTENTION_FIDELITY={hifi2,lofi}` for full attention.

### 3. Measure the existing expert-input-L1 candidate cumulatively

**Confidence: high, small bounded experiment.**

Set only:

```text
GEMMA4_OPT_EXPERT_DECODE_INPUT_L1=1
```

on top of the final packed/BFP8 expert policy. The source performs one L1 conversion
before the two sparse gate/up matmuls, which currently total 176.241 us and both read
DRAM. Compare the two individual rows and complete-layer time. Retain it only if the
saved read time exceeds the single conversion cost. In the residual-chain candidate,
replace this with one residual-shard-to-supported-L1 boundary and test direct sharded
sparse input if the kernel accepts it.

### 4. Test BFP8 KV cache with a complete fill/update contract

**Confidence: medium-high; entirely unmeasured.**

The current tests allocate BF16 cache, and the optimized decoder inherits the
functional fill path. The smallest valid experiment is:

1. Allocate otherwise identical K/V caches as `ttnn.bfloat8_b`.
2. Cast the aligned/full prefill K/V tensors to `cache.dtype` before
   `paged_fill_cache`.
3. Keep the non-aligned tail and decode `paged_update_cache` input tensors BF16, as
   in the common attention path.
4. Run real-weight prefill followed by decode for sliding and full attention, both
   cache-sharing modes and traced batch 1/32.
5. Compare the same-context SDPA row and full-layer time, and record cache bytes.

A decode-only timing against a pre-populated synthetic cache is not sufficient because
it does not validate the fill/update dtype boundary.

### 5. Repair candidate provenance before using rejected policies as conclusions

**Confidence: definitive documentation gap.**

The attention BFP4 trial changed dtype and fidelity together and records a
PCC 0.938611 summary, but lacks a candidate-tagged raw PCC artifact, first failing
tensor, and exact failing output. The expert BFP4 rejection records a minimum PCC
0.993908 summary, but the retained trace JSON now represents the final default
candidate, not immutable per-user failure evidence. The fresh DRAM-sharded timing is
also from the earlier unpacked source state and has no operator-row evidence.

For every rejected correctness candidate, preserve:

- exact command, environment, source hash, fixtures, seed, and weights;
- raw per-tensor/per-user PCC output and the first failing tensor;
- candidate-specific JSON/log filenames that later runs do not overwrite;
- profiler rows proving the requested dtype/fidelity/program actually executed.

Re-test attention BFP4 dtype and LoFi fidelity independently. Preserve all per-user
expert PCCs for the near-threshold BFP4 result rather than only its minimum.

## Recommended order

1. R11 versus R22 RMS microprobe, then immediately the coherent full residual-chain
   comparison.
2. Expert-input-L1 cumulative A/B while the larger chain work is evaluated.
3. Shared-MLP geometry A/B, followed by BFP8/LoFi and BFP4/LoFi on the winner.
4. BFP8 KV fill-plus-decode experiment.
5. Reproduce rejected precision candidates with immutable raw provenance.

Also collect a full-attention final profiler: the retained raw Tracy report covers
only the sliding layer, while full attention has different QKV/cache shapes and a
different selected fidelity. Reconcile theoretical roofline, device/signpost time,
and same-run host time only after the remediated winner is selected.

No implementation correctness bug beyond the performance-contract violation is
proven by this source-only review; hardware measurements are required to choose
between R11 and R22 and to keep any geometry or precision candidate.
