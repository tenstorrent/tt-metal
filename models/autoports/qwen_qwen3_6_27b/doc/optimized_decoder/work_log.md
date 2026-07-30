# Optimized decoder work log

## Operation-topology audit

| current operation | candidate | action | evidence |
|---|---|---|---|
| same-input Q/K/V and linear input projections | packed projection | retained fused-stage packing | avoids repeated input reads |
| MLP gate/up projections | packed gate/up | rejected | fused-stage measurement was slower |
| decode interleaved BFP8/HiFi2 matmuls | BFP8 LoFi, BFP4 LoFi | selected BFP4 LoFi | full b1 2.440 -> 1.387 ms before sharding |
| decode interleaved weights | eight-bank DRAM width sharding | selected, `in0_block_w=8` | full b1 1.387 -> 1.006 ms |
| max block width 4/8/10/20 | sweep | max 8 selected | actual w8 divisors are 5 (K=5120) and 4 (K=17408); max10 failed a static-CB/L1 collision after adaptation; max20 requested 2,243,328 B vs 1,572,864 B L1 |
| decode residual/norm reshards | coherent width-sharded residual | selected | sharded RMSNorm and adds remain on eight workers |
| narrow packed QKV/beta slices | retain sharding | rejected | physical shard shapes are incompatible; one L1-interleaved boundary is required |
| prefill DRAM-sharded weights | large-M interleaved copy | selected | seq33 passed PCC 0.9997609 without a runtime reshard |
| default prefill matmul factory | explicit large-M multicast | selected for physical M >= 10 tiles | forced/selected matrix in `logs/prefill_config_*` and `logs/prefill_selected_*` |
| SDPA | dedicated composite op | retained | fused functional path already uses TTNN SDPA |

## Candidate matrix

Warmed decode medians (ms), ordered full-b1/full-b32/linear-b1/linear-b32:

- BFP8 LoFi interleaved: 1.633 / 1.831 / 2.324 / 20.629.
- BFP4 LoFi interleaved: 1.387 / 1.584 / 2.073 / 20.353.
- BFP4 LoFi DRAM-sharded w8: 1.006 / 1.208 / 1.670 / 19.125.

All listed candidates passed the functional PCC threshold; the selected path
was exactly deterministic across repetitions. Batch 1 and 32 were swept
separately. The selected decode factory uses `per_core_M=1`, which is legal
for batch 1 because the activation is tile padded.

The max-width-10 experiment is preserved in
`logs/candidate_w10_full_b1.log`: its K=5120 projections select literal
`in0_block_w=10`, but the generated program's static circular-buffer region
ends at 1,474,816 while an L1 buffer begins at 1,351,680. Max-width 20 failed
the stronger total-L1 capacity check. These are hard program-resource limits,
not an unsupported-family inference.

### Prefill program-config sweep

Sequence 33 warmed medians (ms):

| config | full b1 | full b32 | linear b1 | linear b32 |
|---|---:|---:|---:|---:|
| default factory | 2.244 | 39.425 | 81.782 | 2538.571 |
| forced explicit multicast | 2.955 | 12.858 | 81.903 | 2513.046 |
| selected by physical M | 2.234 | 12.853 | 81.361 | 2512.844 |

The selected explicit config uses an 8-column grid, up to 10 rows,
`in0_block_w=4`, `per_core_M=ceil(physical_M_tiles/grid_y)`, and projection
specific `per_core_N`: 56 packed-QKV, 65 packed-linear, 68 MLP gate/up, and 20
for 5120-wide outputs. Batch-1 sequence 33 has two physical M tiles and keeps
the default factory; batch 32 has 64 and uses a 8x10 config. Initial 8x8/10x8
attempts exceeded L1; their error logs are retained.

### Linear decode layout-conversion audit

The final linear-b32 report attributes 3.921 ms of 19.094 ms to 11
tilize/untilize operations. Equivalent fused and functional profiler controls
have the same topology and approximately 3.978/3.963 ms per replay, proving
this traffic is inherited rather than introduced by DRAM sharding. The source
boundaries are:

- the 10,240-wide mixed projection permuted into singleton-temporal-last
  orientation for the four-token depthwise convolution cache, then permuted
  back after concat/multiply/reduce;
- 48-wide beta/decay reshape-permute boundaries, which are not tile aligned;
- recurrent state shaped `[B, 48, 128, 128]`.

The optimized packed-output sharded-to-interleaved crossing is only 3.481 us
and is required before taking the 48-wide slices because their logical width
cannot retain a packed 2,080-wide physical shard. No gated-delta composite
exists in TTNN. The available `conv1d` path itself converts TILE to row-major
and back; existing Mamba code must split at 5,120 channels for L1 capacity
while this layer has 10,240 channels, and another repository caller disables
its conv1d fast paths for numerical divergence. Removing this movement
therefore requires a new dedicated gated-delta composite, not a safe local
layout substitution. This is necessary movement in the validated primitive
topology, not an unexamined fallback.

## Checklist

- [x] Read optimization technical guidance and inspected source/data flow.
- [x] Profiled and ranked operation topology.
- [x] Swept fidelity, weight precision, layout, DRAM geometry, and block width.
- [x] Swept default versus explicit large-M prefill program configs at both
  batches and selected them independently by physical M.
- [x] Measured batch 1 and serving batch 32 for both meaningful layer kinds.
- [x] Preserved packed projections, SDPA, cache/state, determinism, and non-aligned prefill.
- [x] Removed host transfers/fallbacks from measured runtime; tests enforce this.
- [x] Recorded selected/rejected configurations and hard resource failure.
- [x] Ran static optimized-path tests and hardware PCC runs.
- [x] Ran 20-replay stress/determinism checks under watcher interval 10 for
  full and linear attention at serving batch 32 (`logs/watcher_*.log`).
- [x] Generated final profiler and advice-enabled `tt-perf-report` artifacts
  for both layer kinds at both batches (`tracy/final_w8_*`).

`tt-perf-report` modeled DRAM-roofline utilization was 38.6%/35.2% for full
attention at batch 1/32 and 26.0%/5.3% for linear attention. The lower linear
batch-32 aggregate is expected: recurrent elementwise/state traffic dominates
after the projection matmuls. The reports classify the material matmuls as
DRAM-sharded BFP4/LoFi; they also flag create-QKV-heads, repeat, and SDPA decode
as currently unclassified report categories, not runtime fallbacks.

Limitations: decode contains one necessary sharded-to-interleaved crossing at
narrow packed-output slicing boundaries. Linear gated-delta tilize/untilize
traffic remains necessary until TTNN has a dedicated composite. Host timings
are not taken from watcher or profiler runs.

## Local checkpoints

- Fused-decoder starting point: `ea7b667c09d`
- Optimized decoder implementation and reviewed docs: `a864ec4e7c3`

The following bookkeeping commit adds the ignored evidence logs and records
these checkpoint identifiers. Nothing was pushed.
