# Exact residual elementwise fusion plan

CPU-only independent AutoFix hypothesis audit, 2026-11-14. No hardware runs or
production changes. This plan excludes exact projection multiply/reduction
fusion, which is being investigated separately, and makes no speedup claim.

**There is useful policy-preserving work, but this hypothesis alone cannot
close the FUNCTIONAL gap.** The historical stage6 reduced profile places all
identified norm/mixing/write kernels at a weighted16.983622ms, or20.390689ms
including gathered-down compaction. Even deleting every such kernel at zero
replacement cost would not close the approximately49.736174ms gap between
selected113.881454ms and the current prepared-weight-plus-KV10% target64.145280ms.
These are historical eager kernel envelopes, not rigorous full48 wall-time
bounds; the final source-matched profile and clean traces must measure gains.

## Source contract

Current code uses `MeshResidual`, not a class named `MeshGatedResidual`:

- [MeshResidual.read/write](../../tt/multichip_decoder.py), lines195–231.
- [FinalMixer](../../tt/model.py), lines52–80.
- [rms](../../tt/decoder_components.py), lines28–31.
- [Collectives.sum](../../tt/multichip_decoder.py), lines117–121.

Each rank retains one complete BF162560-wide residual stream, including across
decoder layers. There are two GR read/write pairs per layer, plus one final
mixer read:96 writes and97 read/mixing paths. Local fusion must preserve that
layout, the existing BF16 all-gathers, selected exact projections, and all
intermediate BF16 rounding boundaries. No native all-reduce, hidden-sliced
residual, new weight/activation dtype, native RMSNorm or changed matmul fidelity
is proposed.

Notation below: `B(x)` is the existing BF16 round-to-nearest-even boundary,
`F(x)` is exact BF16-to-FP32 widening. FP32 operations retain their original
ordering and selected SFPU implementation, including nonapproximate nonlinear
functions. Do not infer an allowed reassociation from algebraic equivalence.

## Highest-priority local chains

### 1. Gathered GR-down compaction plus scale/SiLU

The exact down projection returns96 padded columns per rank for80 useful
columns. `GatheredProjection.__call__` performs the existing all-gather,
extracts `[r*96:r*96+80]` for ranks0..3, then concatenates320 useful values.
`MeshResidual.read` and `FinalMixer.__call__` immediately evaluate:

```text
gathered BF16[384] -> four80-column slices -> concat BF16[320]
  -> BF16 multiply by0.25 -> widen FP32 -> SFPU SiLU -> BF16
```

Propose one local kernel after the unchanged all-gather. Read useful source
column `96*(j//80) + j%80`, emit320 columns in the identical order, and compute
`B(silu(F(B(down*0.25))))`. Explicitly preserve the intermediate scale BF16
boundary, including subnormal/midpoint controls. This bypasses the four slice
allocations and the single-core concat without changing the projection or CCL.
Use a consumer-specific wrapper/entry point; do not silently change every
`GatheredProjection` consumer.

The first stage6 instance has slice/concat report rows22..26 and scale/SiLU
rows27..30 (zero-based CSV data rows). Concat alone is30.354us. The weighted
whole-chain envelope is **4.348552ms**:3.407067ms compaction plus0.941485ms
scale/SiLU. This is the largest clearly identified local chain that needs no
reduction-order change. The exact-up input remains BF16[1,1,Q,320].

### 2. Mesh residual writer

Current eight-op chain is:

```text
F(block), F(injection) -> FP32 multiply -> B(update)
F(residual), F(update) -> FP32 add -> B(next_residual)
```

One native local kernel can implement
`B(F(residual) + F(B(F(block)*F(injection))))`, retaining the two explicit
rounding boundaries and separate multiply/add instructions. Preserve column
broadcast of the per-row injection scalar; do not contract into FMA.

The repository already has an exact implementation in
[fused_decoder.py::fused_residual_write](../../tt/fused_decoder.py), lines262–357,
and [probe_fused_residual_write.py](../../tests/probe_fused_residual_write.py).
Its selected single-chip layout is residual10240 with four streams and an
injection tensor `[1,4,Q,1]`. It is **not directly callable** on the mesh layout.
A mesh adapter needs local residual/block2560, local injection `[1,1,Q,1]`,
80 tiles per token tile instead of320, and local injection tile addressing
without the single-chip `stream=column/80` offset. Retain standard tile checks,
FP32 unpack, SFPU column broadcast, explicit typecasts, and independent output
storage. No CCL belongs in this kernel.

[Stage3 work log](../fused_decoder/work_log.md), entries around lines458–471,
records bitwise random and midpoint/cancellation controls,11 lengths and12
changed-input trace replays for the original geometry. That is implementation
precedent, **not mesh validation or a transferable latency measurement**.
The six identified mesh writer sequences give a weighted **3.004752ms**
kernel envelope. This is the lowest-risk first experiment.

### 3. Norm epilogue first; full norm only with exact reduction reproduction

The complete `rms` chain is:

```text
F(residual) -> FP32 square -> accurate SFPU width SUM over2560
  -> FP32 postmultiply1/2560 -> FP32 add epsilon -> precise rsqrt
  -> FP32 multiply original widened residual -> FP32 multiply norm weight
  -> BF16 normalized stream
```

The stage6 raw reduction attributes explicitly show `ReduceOpDim::W`,
`use_sfpu_reduce=true`, `post_mul_scaler=0.000390625`, FP32 DEST and packer-off.
The generic width reduction folds80 width tiles in order then performs the
SFPU intra-tile reduction; see
[reduce_helpers_compute.inl](../../../../../ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl),
around lines486–565. Preserve that fold, its row reduction and reciprocal
placement. The physical clone/fillpad ensures padding is inert; a fused reader
must initialize equivalent padding rather than accidentally reducing stale rows.

The simple candidate leaves square/mean unchanged and fuses the five-op
epsilon/rsqrt/two-multiplies/BF16 epilogue: **2.507136ms** weighted envelope.
Preserve multiplication order `(value*scale)*weight` and epsilon FP32 precision.
The larger custom exact norm could also remove square, clone/fill and reduction
dispatches: **6.121788ms** total envelope, including the epilogue. These two
numbers overlap and must not be added.

Do not reuse native RMSNorm: the prior rejection documents TF32/FPU intake,
different reduction order and epsilon handling
([stage3 AUTODEBUG](../fused_decoder/AUTODEBUG.md)). The safe implementation
strategy is copying the selected SFPU arithmetic into a fused producer/consumer,
not replacing it with a mathematically equivalent norm.

## Additional chains and hard barriers

| Chain | Required boundaries | Weighted historical envelope |
|---|---|---:|
| Exact-up output to weighted normalized stream | `B(sigmoid(F(up_BF16)))`, then existing BF16 multiply with normalized stream; preserve the sigmoid BF16 intermediate before multiply |1.297660ms |
| Injection projection epilogue | `B(injection*0.25)` -> FP32 sigmoid -> BF16 -> BF16 multiply2; leave native projection and its layout conversions unchanged |0.938583ms |
| Weighted-stream gather epilogue | Existing BF16 AG -> FP32 widening -> exact H-reduction over rank values -> FP32 multiply0.25 -> BF16 |4.679354ms |
| Only post-rank-sum scale/cast | Keep widening/reduction entirely unchanged, fuse multiply0.25 plus BF16 |0.693393ms, included in preceding row |

The weighted-stream all-gather stays exactly where it is. Raw stage6 attributes
show that `Collectives.sum(dim=1)` lowers through clone/fill, transpose,
`ReduceOpDim::H` accurate SFPU reduction, and transpose back. Although the source
comment describes rank-ordered accumulation, **do not replace this with an
assumed sequential four-add loop**. A fused local reducer must reproduce the
actual current H-reduction lane/tree order and pad initialization. Prototype
against the existing tensor output bitwise before considering integration.
This is local reduction fusion after an unchanged collective, not native AR.

Norm cannot be fused through a projection or a collective as a single local
elementwise kernel. Writer plus the next norm is possible only if the writer's
BF16 output remains explicit and available to the residual owner; it introduces
multi-output/liveness changes. Start with separate writer and norm kernels so
serving/cache/trace boundaries remain easy to test. The final mixer has no
injection/write and must use its own read-only variant.

## Stage6 attribution and reproducibility

Input is the1253-row reduced model CSV:
[model_perf_report.csv](../full_model/terminal_final_perf_v1/model_perf_report.csv),
SHA256 `152f944b22d00b966eb45e39ad936871cab1266dc3bf7afbf23eac418f1508c8`.
It is one eager B1/C262144 invocation with real layers0,1,3 and terminal work.
[Coverage](../full_model/terminal_final_perf_v1/coverage.json) and
[analysis](../full_model/terminal_final_analysis.json) prove complete joins.
Per-op times use the report's merged ranks, not one rank's critical path.
Do not include eager host gaps or multiply the whole reduced model by16.
The old QSA scorer is outside every region used here.

Exact source-order53-op GR-read signatures match at starts5,169,502,666,795,
1075. The final mixer starts1204 and matches the44-op prefix. These are
zero-based CSV **data-row indices**, not report IDs. Current source confirms
the same residual expression; the final selected profile must revalidate timing.

| Instance | Read start | Read prefix first/last Global Call Count | Write data-row range | Write first/last Global Call Count | Weight |
|---|---:|---|---|---|---:|
| L0 attention |5|4082691 /4126721|161..168|4242433 /4249602|35|
| L0 MLP |169|4250626 /4294656|290..297|4374528 /4381699|35|
| L1 attention |502|4591618 /4635651|658..665|4751362 /4758530|1|
| L1 MLP |666|4759552 /4803585|787..794|4883459 /4890626|1|
| L3 attention |795|4891650 /4935683|1067..1074|5170177 /5177346|12|
| L3 MLP |1075|5178371 /5222402|1196..1203|5302274 /5309441|12|
| Final mixer |1204|5310465 /5354499|none|none|1|

For read start `s`, sum `float(row['Device Time [μs]'])` over these half-open
offset ranges, then multiply by the instance weight and divide1000:

| Region | Offsets | Weighted ms |
|---|---|---:|
| Full norm |[0,10)|6.121788|
| Norm epilogue subset |[5,10)|2.507136|
| Down compaction |[17,22)|3.407067|
| Down scale/SiLU |[22,26)|0.941485|
| Up sigmoid/product |[31,35)|1.297660|
| Post-weighted-stream-AG epilogue |[36,44)|4.679354|
| Post-sum scale/cast subset |[42,44)|0.693393|
| Injection epilogue, first six instances only |[48,53)|0.938583|
| Residual writer, eight rows at each write start |[0,8)|3.004752|

The nonoverlapping total uses full norm, compaction, down scale/SiLU,
up sigmoid/product, full post-AG epilogue, injection and writer:20.390689ms.
A more conservative first wave retaining both original reductions uses norm
epilogue, compaction+SiLU, up sigmoid/product, post-sum scale/cast, injection
and writer:12.790076ms. Every replacement still costs nonzero device time;
these totals are savings ceilings in this profile, never predictions.

## Experiment order and acceptance

1. Adapt the existing exact writer to local stream2560. Independently compare
   gathered-down compaction+SiLU against its current complete local chain.
2. Capture actual mesh operands for all four ranks, GDN/QSA/PLE layers and the
   final mixer. Include BF16 midpoint/cancellation and extreme finite controls;
   exact existing-output equality is the component gate, not PCC alone.
3. Exercise physical Q1/2/4/8/16/32 plus valid logical31/33 via the generator,
   mixed requests and inactive fixed slots. Compare inactive rows and all
   observable residual/mixed/injection outputs. Warm before capture, replay
   changed inputs, audit fallback, allocation tracking, and Watcher separately.
4. Measure complete producer/consumer traces and selected full48 resident token-out
   with unchanged sampling; rerun integrated accuracy and serving contracts.
   Avoid claiming large gains from isolated op dispatch or reducing B32 support.
5. Only then attempt norm or local rank-reduction fusion, reproducing exact SFPU
   order. Preserve the original wrappers as controls; do not replace rejected
   native policies or reopen a datatype frontier.

Existing rejection evidence remains binding:
[stage5 topology audit](../optimized_multichip_decoder/operation_topology_audit.md)
records slower async/persistent CCL, subgroup dtype gains below0.2%, and hidden
residual whole-B32 regressions. None is revisited by these local candidates.
This plan is a bounded independent contribution to the gap repair, not grounds
to mark the strict10% gate passed or the overall AutoFix complete.

## Test-local writer candidate prepared

[exact_residual_write.py](../../tt/exact_residual_write.py) now implements the
local2560 variant using three new file-backed kernels. Compute and writer source
are copied byte-for-byte from the inherited native writer; only the reader's
local tile geometry and Python shape validation change. Production
`MeshResidual.write` remains the eight-op control.

[probe_exact_residual_write.py](../../tests/probe_exact_residual_write.py) tests
Q1/3/17/32, two synthetic families and four distinct ranks. It requires bitwise
BF16 storage equality, including signed zeros, against the actual
`MeshResidual.write`; it retains12 alternating-input trace checks per variant.
Tracked mode includes a retained-allocation positive control. Clean timings use
three64-replay nonblocking windows with one final synchronization and no interior
input copy/readback/synchronization. Runtime audit wraps eager, warm and captured
bodies. Only trace-owned outputs are acknowledged corruptible; only owned traces
are released. Source SHA256s are saved in the result.

CPU checks pass: Python compile/format, all eight shape/family fixtures have
rounding-boundary witnesses on each rank (omitting the intermediate BF16 rounding
changes results), and compute/writer equality against inherited source. No device
correctness, C++ compilation, performance or full-model acceptance is claimed.
Inputs have actual runtime geometry but are synthetic, not captured real model
activations. Run serially after current device ownership clears:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/environment.sh
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
timeout 600 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_exact_residual_write \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/exact_writer_tracked_v1.json

TT_METAL_TRACE_ALLOC_TRACKING=0 \
timeout 600 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_exact_residual_write \
  --clean-timing --output models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/exact_writer_clean_v1.json
```

Watcher is a separate tracked correctness run, never combined with clean timing.
Full-model opt-in validation remains necessary before selecting production dispatch.

The additional merged-norm source review found that
[merged_norm_layer0_v1.json](../fused_decoder/merged_norm_layer0_v1.json) passes
exact output/state but is slower in its tracked single-chip regime
(47.007233ms versus46.902691ms). The `norm_rhs` row of
[norm_rhs_matmul_v1.json](../fused_decoder/norm_rhs_matmul_v1.json) passes exact
output/state and measures46.942287ms versus46.965122ms, a negligible0.049% gain.
That file's aggregate failure belongs to its separate `matmul_activation` row.
RHS norm is not a demonstrated accuracy-rejected policy; neither old result
establishes a useful local2560 or full-model speedup.

An independent optional harness is prepared in
[probe_exact_mesh_norm.py](../../tests/probe_exact_mesh_norm.py). It compares
the unchanged `rms` against existing `merged_rms(rhs=True)`, retaining original
square/mean arithmetic. It uses real FP32 layer0 attention/MLP and final-mixer
norm weights, four distinct rank streams, Q1/3/17/32, random/cancellation and
near-epsilon residuals. Diagnostic FP32 scales are compared bitwise outside
timing, including the RHS activation chain evaluated on FP32 ones. Actual final
BF16 norm outputs are compared bitwise eagerly and across12 changed-input trace
replays. Complete norm traces receive three64-replay nonblocking timing windows.
The historically slower add/rsqrt variant is an explicit optional comparison,
not the default. Both harnesses print phase markers before submissions and
capture and preserve the last phase on failure. Provenance includes the reused
`fused_decoder.py` arithmetic source.

```bash
# After environment.sh, serially; start with clean comparison as directed by owner.
TT_METAL_TRACE_ALLOC_TRACKING=0 \
timeout 600 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_exact_mesh_norm \
  --clean-timing --output models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/exact_norm_clean_v1.json
# Repeat without --clean-timing and with tracking=1 for guarded validation.
# --include-add-rsqrt adds the historical slower exact comparison explicitly.
```

The checkpoint's three named norm tensors and rms_norm_eps=1e-6 were verified
from local metadata. CPU descriptor mocks verify exact one-owner coverage of80
local tiles for all four writer row counts, correct input addresses, bounded
CBs and rejection of counts outside1..32. Python compile/format checks pass for
both probes. Hardware compilation, bitwise output and timings remain untested.
