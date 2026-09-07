# Prefill: two general optimizations the bringup pipeline missed

Prefill was the weak axis of this port — `doc/qwen38_checkpoint_swap` records TTFT
missing its graded target by **59x**, at a flat ~28 ms per input token from ISL
128 to 65,536. Both fixes here are **model-agnostic**: neither uses a fused op
written for this architecture, and both are things a bringup pipeline could
check on any model.

Combined: **16.2x** on the linear-attention prefill layer, **3.94x** on
full-model TTFT at the shipped batch.

## Results

Single linear-attention layer, S=128 prefill, single chip, synthetic weights:

| stage | batch 32 | batch 1 | PCC (b32) |
|---|---:|---:|---:|
| baseline | 8543.8 ms | 272.8 ms | 0.99999576 |
| + matmul program configs | 2031.2 ms | 68.4 ms | 0.99999524 |
| **+ sequential recurrence** | **527.8 ms** | **44.2 ms** | 0.99999543 |
| | **16.2x** | **6.2x** | |

Full model, 64 layers, TP4, real Qwen3.8-27B weights, warm TTFT at ISL 128:

| stage | batch 32 | batch 1 |
|---|---:|---:|
| baseline | 106614 ms | 3612 ms |
| + matmul program configs | 37160 ms | 1443 ms |
| **+ sequential recurrence** | **27051 ms** | **1127 ms** |
| | **3.94x** | **3.20x** |

The model-level gain is smaller than the per-layer gain because TP4 splits
`value_heads` 48 -> 12, so the recurrence's batched matmuls run at `b=384`
rather than `b=1536` and are less efficient, and because the 16 full-attention
layers, projections and LM head do not change.

Reproduce (both knobs default off/`hillis`; set them to opt in):

```bash
QWEN36_SCAN_MATMUL_GRID=8x10 QWEN36_PREFILL_SCAN=sequential \
python models/autoports/qwen_qwen3_6_27b/tests/linear_attention_synthetic_pcc.py \
  --mode prefill --sequence 128 --optimized --candidate linear_final --batch 32 --iterations 2
```

## 1. Batched matmuls had no program config (4.21x)

`tt-perf-report` on the batch-32 prefill layer showed 77% of device time in two
batched matmuls running on **16 and 4 of 110 cores**, while every cheap
elementwise op used all 110. They were plain `ttnn.matmul` calls, so the program
was auto-selected.

Swept at the real per-device shape `[384, 32, 128, 128]`:

| config | us |
|---|---:|
| default (no `program_config`) | 35475 |
| `MatmulMultiCoreReuse`, grid 8x10 (80 cores) | **2694** |
| same at grid 4x4 (**16 cores**) | 3361 |

**13.17x.** The 16-core row is the point: an explicit batched-reuse config on the
*same* core count is still 10.5x faster, so this was never mainly about grid
size — the default picks the wrong program class for a batched matmul with small
M/N/K and a huge batch. Achieved rate 1.45 -> 19.1 TFLOP/s.

A finer sweep of `in0_block_w` and output subblocks then found nothing: best
2680.5 us against 2694, i.e. 0.5%. The op is done.

Applied via `_scan_matmul` in `functional_decoder.py`, used by both the
single-chip and multichip prefill paths.

### Why the pipeline missed it

`.agents/skills/optimize/SKILL.md` has detailed, correct core-grid guidance —
and every instance of it is scoped to **decode**:

> "For any matmul or repeated matmul group that is one of the largest
> **decode-time** consumers: swept legal program configs ... including core grid"

There is no equivalent gate for prefill, so the stage could pass with prefill's
dominant matmul never swept. Across every optimize-stage log, the word "cores"
appears twice, and zero times in a prefill context.

Compounding it, the only prefill tracy capture in the whole pipeline is from the
*functional* stage at **S=5**, where the scan matmul is 9.1% of the layer and the
projections (96-109 cores) dominate — the picture looks healthy. The op that is
70% of prefill at the shipped shape was 9% at the only shape ever profiled.

**Pipeline fix:** drop "decode-time" from the checklist — require the sweep for
the largest consumers *in each measured phase* — and profile prefill at the
shipped ISL and batch, not a smoke-test length.

## 2. A parallel scan bought parallelism the device already had (3.85x)

The prefill state recurrence was a Hillis-Steele affine scan: materialise a
`[K, K]` = 128x128 transition per token, then compose them pairwise in
`log2(chunk)` doubling steps.

The recurrence it encodes is a **rank-1 update**:

```
S_t = d_t * S_{t-1} + k_t^T ( beta_t * (v_t - k_t * (d_t * S_{t-1})) )
```

which is O(K*V) = 32 K FLOP per token. Composing transitions is O(K^3) = 4.2 M
FLOP — **128x more** — and the scan does it `log2(32) = 5` times over. It also
slice+concat shifts a **1.61 GB** `[1536, 32, 128, 128]` array twice per step,
which is why Concat/Slice/BinaryNg were 58% of the layer with nothing a config
could touch.

Parallel scans trade work for depth. That trade only pays when the device is
starved — and here `groups = batch * value_heads = 1536` against **110 cores**,
saturating it more than 14x over by the batch dimension alone. The depth was
free; the `log2(chunk)` work multiplier was not.

So `_sequential_recurrence` applies the rank-1 update one token at a time — the
same maths `_linear_attention_decode` already runs, so no new algebra. Measured
2031.2 -> 527.8 ms, and PCC is marginally *better* (0.99999543 vs 0.99999524):
fewer matrix compositions, less accumulated error.

Two things worth recording:

- **Dispatch is not the cost.** The sequential form runs 2078 ops per layer
  against 650, and op-to-op gap is **0.1%**. The fear that motivates parallel
  scans — dispatch overhead — did not materialise.
- The first sequential implementation was only 1.47x because its matmuls were
  plain `ttnn.matmul` again, at 4 and 16 cores. Routing them through
  `_scan_matmul` took it from 1384.1 to 527.8 ms. The two fixes compose.

### The chunk-size knob's cost model is inverted

`_linear_prefill_chunk_size`'s docstring argues for *larger* chunks:

> "(S/chunk) * log2(chunk) sequential scan **steps** — a *decreasing* function of
> chunk ... Larger chunks therefore reduce both sequential depth and host traffic"

That counts steps, not work. Work per step scales with the batch dimension
`groups * chunk`, so total work is `groups * S * log2(chunk)` — **increasing** in
chunk. Since the layer is 100% device-bound with ~0 gap, depth is irrelevant and
work is everything. The recommended direction is also unreachable: chunk 64 dies
with `Out of Memory: Not enough space to allocate 3221225472 B DRAM buffer`,
because the scan materialises `[groups, chunk, K, K]`.

The knob is not the lever either way — 32 is the tile-alignment floor. It is
recorded because the reasoning error is the same one that produced the parallel
scan: optimizing depth for a workload that was never depth-bound.

**Pipeline fix:** before choosing a parallel formulation, check occupancy. If
`batch * heads >> cores`, use the work-efficient sequential recurrence. State
cost models in work, not steps, and say which resource is the binding one.

## Correctness

`linear_attention_synthetic_pcc.py`, `linear_final`, both modes:

| shape | hillis | sequential |
|---|---:|---:|
| b32 S128 | 0.99999524 | 0.99999543 |
| b1 S128 | 0.99999528 | 0.99999548 |
| b1 S33 (ragged tail) | 0.99999543 | 0.99999562 |
| b1 S65 | — | 0.99999552 |
| b32 S96 | 0.99999523 | 0.99999543 |

Bar is 0.995. Note the synthetic conv weights are tap-degenerate — see
`doc/kda_conv_swap` — so `check_conv_taps.py` is the harness that actually
exercises conv history; it is unaffected by these changes but should be re-run
before landing.

One bug found and fixed while implementing: `ttnn.concat` of a single tensor
aliases its input, so deallocating the inputs freed the result. A ragged tail
chunk of one token hits this. Guarded.

## Status

Not landed. Both knobs default to the old behaviour
(`QWEN36_SCAN_MATMUL_GRID=8x10` is the default but `_scan_matmul` falls back with
`0`; `QWEN36_PREFILL_SCAN` defaults to `hillis`). Before landing:

1. multichip decode matrix and the full-model decode A/B, since `_scan_matmul`
   is shared with `multichip_decoder.py`
2. `check_conv_taps.py` across single-chip and TP4
3. long-ISL re-measure — the recorded sweep timed out at ISL 131072, and at
   3.94x that point should complete well inside a limit that already passes
   ISL 65536 at 1837 s
