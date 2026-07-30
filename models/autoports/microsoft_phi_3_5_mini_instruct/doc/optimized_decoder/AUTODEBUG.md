# AutoDebug: Phi-3.5 optimized decoder decode geometry

## Symptom and evidence

Source-only investigation; no device experiment was run.

- The optimized decoder changes weight storage dtype only. It inherits
  `FusedDecoder._mlp`, whose norm, gate/up, down, and residual add all use TTNN
  defaults without memory/program/compute configs.
- The bounded Blackhole profiler reports the optimized BFP8 down projection
  (`32 x 8192 x 3072`) at about 182-183 us for both B1 and B32, `SLOW`, 96
  cores, HiFi2. Its reported DRAM utilization is only about 27.6-27.7%, versus
  about 54.6% for the BF16 baseline row at the same latency.
- Both decode norms remain default `LayerNormDeviceOperation` calls at about
  44 us on one core.
- The common `models/common/modules/mlp/mlp_1d.py` decode path explicitly
  reshards the activated MLP tensor, uses
  `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` for W2, then restores
  the residual memory config. The Phi path does none of these.

## Ranked hypotheses and focused experiments

### H1 — Default `ttnn.linear` selected an interleaved, bandwidth-starved down path

Confidence: high.

The down weight is merely typecast in place and remains DRAM interleaved.
There is no `program_config`, weight shard spec, or activation reshard in the
Phi call. The unchanged 183 us despite halving weight storage, combined with
the profiler's halved effective DRAM percentage, predicts that launch/dataflow
geometry—not BFP8 arithmetic—is limiting the row.

Verify/refute:

1. Add a test-local down-only A/B using the exact post-SiLU tensor and real
   layer-0 down weight for B1 and B32.
2. Log logical/padded shape, dtype, layout, memory config, shard spec, program
   config, and compute config for input/weight/output.
3. Compare the current default against DRAM-sharded BFP8 weights plus
   `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`.
4. Start with clean tile geometry for K=8192 (256 tiles), N=3072 (96 tiles):
   16 cores (`K/core=16`, `per_core_N=6`) and 32 cores
   (`K/core=8`, `per_core_N=3`). Sweep every legal `in0_block_w` divisor
   (16/8/4/2 for 16 cores; 8/4/2 for 32) and legal output subblocks. Try a
   padded 64-core candidate only if padding/slicing makes its N distribution
   mathematically inert.
5. Cross each material geometry with BFP8 HiFi2 and LoFi.

Verified if a DRAM-sharded candidate materially reduces the isolated row and
whole traced layer latency with real PCC >= 0.995. Refuted only if all legal
clean candidates are slower or an exact validation/L1 contract blocker is
captured.

### H2 — The activated MLP tensor has the wrong layout for the best down geometry

Confidence: high.

The packed gate/up result is sliced and multiplied under default memory
placement, then consumed directly by down. A first DRAM-sharded program-config
failure or regression would likely reflect this inherited input contract, not
an invalid DRAM-sharded down family.

Verify/refute:

1. In the same down-only harness, convert `activated` once to candidate
   width/block-sharded L1 configs whose shard K widths admit the H1
   `in0_block_w` values.
2. Measure conversion + down + conversion/restoration together, not the
   matmul alone.
3. Compare at least a 16-core wide-shard and 32-core wide-shard family for B1
   and B32. Record shard shape in tiles and L1 use.
4. Extend the best candidate backward through slice/fused-SiLU-multiply to see
   whether those consumers/producers can keep the working shard and eliminate
   a conversion.

Verified if the adapted chain wins while the same matmul on interleaved input
does not, or if wider shards unlock a winning block size. Refuted if conversion
cost dominates every legal candidate and no upstream producer can emit the
layout.

### H3 — Default HiFi2 is unnecessary for BFP8/BFP4 decode matmuls

Confidence: medium-high.

All optimized profiler matmuls still report HiFi2. No LoFi candidate is
present. The down row is bandwidth-limited, so LoFi alone may not fix it, but
LoFi can change kernel/L1 constraints and must be crossed with H1/H2 geometry.

Verify/refute:

For the final H1/H2 candidate matrix, run BFP8 down HiFi2 versus LoFi and BFP4
down LoFi as a guarded candidate. Separately run packed gate/up BFP4 HiFi2
versus LoFi under its best legal geometry. Use real layer weights and record
down-only PCC, whole-layer PCC, traced B1/B32 latency, and profiler-emitted
dtypes/fidelity. Do not let a synthetic-only PCC reject a real-weight winner.

### H4 — Keeping the 3072 residual stream sharded removes both 44-us norms and layout traffic

Confidence: medium-high.

`FunctionalDecoder._norm` calls default `ttnn.rms_norm`, and the residual is
DRAM interleaved at layer boundaries. The profiler proves one-core execution.
TTNN provides `LayerNormShardedMultiCoreProgramConfig`, and the common RMSNorm
module demonstrates sharded decode RMSNorm.

Verify/refute:

1. First isolate RMSNorm on the exact `[1,1,32,3072]` physical decode shape
   used for both logical B1 and B32. Compare default against legal sharded L1
   configs and `LayerNormShardedMultiCoreProgramConfig`; include input/output
   conversions.
2. Then measure a composite residual family:
   residual add -> sharded RMSNorm -> packed gate/up -> fused activation ->
   DRAM-sharded down -> residual add, restoring the public boundary only at
   layer exit.
3. Try shard grids compatible with both 3072 (96 tiles) and the selected down
   candidate, prioritizing 16 and 32 cores. Record whether RMSNorm requires a
   different working shard; if so, measure that reshard once across the whole
   MLP.

Verified if the composite chain beats the sum of isolated defaults while
preserving PCC. A faster isolated norm that loses after conversions is not a
chain win.

### H5 — B1 and B32 currently profile the same padded M=32 work, masking useful phase-specific choices

Confidence: medium.

Both reports show `M=32` rows and nearly identical down/norm times, although
the logical batches differ. This is expected tile padding, but it means B1
latency cannot improve by reducing M unless a legal narrow-B1 contract exists.

Verify/refute:

Inspect the exact logical and padded tensor shapes at each MLP boundary for B1
and B32. Run the same H1-H4 families at both batches and rank by whole trace,
not isolated op. Keep phase/batch-specific configs only if trace capture and
the public B1/B32 shapes remain unchanged. Do not shrink or special-case the
model-visible batch contract.

## Recommended experiment order

1. H1 down-only precision-locked DRAM-sharded geometry matrix.
2. H2 adapted activation-shard matrix; keep only end-to-end winners.
3. H3 fidelity/dtype cross on the best geometries.
4. H4 sharded RMSNorm, then the whole residual/MLP chain.
5. Reprofile warmed traced B1 and B32 with real weights; confirm emitted
   dtype/fidelity/config rows and rerun real PCC plus watcher.

Each experiment should change one hypothesis group at a time. Preserve
candidate tables with row latency, conversion latency, whole-layer trace
latency, PCC, and exact rejection reason.

## Source and artifacts inspected

- `tt/optimized_decoder.py`
- `tt/fused_decoder.py`
- `tt/functional_decoder.py`
- `tests/optimized_decoder_perf.py`
- `doc/optimized_decoder/tracy/decode_b1.txt`
- `doc/optimized_decoder/tracy/decode_b32.txt`
- `models/common/modules/mlp/mlp_1d.py`
- `models/common/modules/rmsnorm/rmsnorm_1d.py`
- DRAM-sharded matmul and sharded LayerNorm examples under `models/`

## Current conclusion

The strongest diagnosis is missing explicit dataflow geometry: the stored BFP8
down weight reaches runtime, but the inherited default interleaved linear path
does not exploit it. The first repair attempt should be an isolated,
precision-locked DRAM-sharded down sweep with adapted activation sharding, not
another dtype-only change. The norm issue is real but should be evaluated as a
composite residual-chain optimization so conversion costs are not hidden.
