# Phi-3.5 Mini advisor challenger

The batch-32 challenger ships a measured improvement. The frozen incumbent's
best real-weight repeat was `0.786969 ms`; the best measured set was
`0.746062 ms`, a reduction of `0.040907 ms` (`5.198%`). The real-weight
recorded-cache oracle passed at PCC `0.998920` against the incumbent's `0.995`
bar.

## Frozen incumbent

The incumbent was measured three times before advisor capture:
`0.788090`, `0.787479`, and `0.786969 ms`. The best repeat is the baseline and
the spread, `0.001121 ms`, is the noise floor. The executed policy came from
`doc/optimized_decoder/tracy_final/decode_b32_perf_report.csv` and the selected
cumulative path in `doc/optimized_decoder/README.md`: BFP4 projection weights,
LoFi kernels, BFP8 KV cache, eight decode cores, and four DRAM-sharded
projection weights.

## Capture and reconciliation

Phi-3.5 Mini has one dense decoder kind across all 32 layers. It was captured
once at batch 32 with the shipped BFP4 weight precision. All four eligible
matmuls were considered for DRAM sharding and all four were advised, but the
incumbent already used DRAM-sharded weights for all four.

The pinned advisor lacks a handler for the shipped fused paged-cache update.
The capture-only adapter exposed it as two supported paged updates so tracing
could reach the O projection and MLP; projection topology, shapes, precision,
program configs, and math fidelity were unchanged. The adapter also replaced a
dynamic Q/K layout query with its declared height-sharded decode layout.

The stock `reconcile.py` command was run against the incumbent CSV, but its
current parsers do not recognize this branch's `#ttnn_layout` MLIR form or the
`Device Time` CSV column. `reconciliation.json` corrects the result from those
same two inputs and groups by summed chain share.

## Measured chains and combinations

The advisor recalled a manual RoPE chain that the incumbent left in DRAM.
The first two attempted geometries were illegal and were not treated as
measurements. The legal chain keeps Q/K and elementwise math height-sharded in
L1, uses interleaved L1 only for the non-tile-aligned 48-wide half-slices and
concat, and then restores the 32-core shard. It measured `0.746191 ms` with PCC
`0.998993`, versus `0.786969 ms` incumbent.

The advisor IR's down projection used `in0_block_w=32`; screened alone it
measured `0.786990 ms`, just `0.000021 ms` slower than the incumbent and inside
the noise floor, so the incumbent won that single-chain tie.

The required pairwise measurement of RoPE L1 plus down-width 32 measured
`0.746062 ms` with PCC `0.998920`, the lowest measured set. The shipped decoder
therefore enables the L1 RoPE chain and uses down block width 32. A fresh
post-application profile is in `tracy/final_ops.csv`; no op shape changed, so
the workflow correctly did not re-capture.

There are no sparse-matmul, SSM, or other terminal layer kinds in this model;
uncapturable layer share is `0%`. Only `tt-perf-report` outputs are stored under
`tracy/`; no raw Tracy capture is included.
