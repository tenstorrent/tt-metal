# Llama 3.1 70B optimized multichip decoder

This directory is the evidence record for the topology-first optimization of
the real-weight `meta-llama/Llama-3.1-70B-Instruct` decoder on four Blackhole
p300c devices.  The measured path is the production `MultiChipDecoder` on a
logical `1x4` TP4 ring; no single-chip or replicated-compute fallback is used
for the multichip result.  Llama 3.1 70B is dense and has one decoder-layer
kind, represented by real layer-39 weights and a direct two-layer composition
check.

## Result

The selected default enables the explicit Blackhole paged-SDPA decode program
(`8x8` grid and HiFi2 compute kernel).  It leaves prefill on the operation's
implicit program because a direct nonaligned prefill A/B was slightly faster
there.  All inherited BFP4/LoFi projection, BF16 activation/cache/CCL,
DRAM-sharded decode-matmul, packed-projection, and persistent RS+AG policies
remain selected.

Repeated isolated A/B runs showed the decode-only SDPA choice improving traced
replay from `0.602448/0.602594 ms` to `0.598484/0.598459 ms`.  The required
like-for-like stage comparison constructs the inherited and selected policies
side by side with the production prefill chunk default of 4096:

| Warmed production-default path | Before this pass | After this pass | Change |
| --- | ---: | ---: | ---: |
| nonaligned prefill | 1.267830 ms | 1.266436 ms | -0.11% |
| eager decode | 1.572428 ms | 1.454311 ms | -7.51% |
| traced decode replay | 0.603287 ms | 0.598957 ms | -0.72% |

That corrected paired run preserves prefill/decode PCC `1.0/0.9999979552` and
K/V PCC `1.0/1.0`.  The independent canonical final-default run reproduced
TP4 prefill/eager/traced latency at `1.281216/1.507943/0.597632 ms`; its
same-run optimized single-chip references were
`3.714869/1.855658/1.847464 ms`, for 2.8995x, 1.2306x, and 3.0913x speedups.
The frozen completed-stage and initial reproduction prefill figures used an
old test-only chunk-32 policy, so they remain historical tail provenance and
are deliberately excluded from the production-default before/after table.

## Final correctness and capability

`final_default.xml` and `final_default.log` are the canonical final-path run: 5
passed and 6 intentional opt-in tests skipped.  The final PCC deltas relative
to the reproduction are below the accepted baseline's natural run variance.

| Final check | PCC versus optimized TTNN reference |
| --- | ---: |
| nonaligned prefill output | 0.9999978879 |
| decode output | 0.9996770059 |
| contiguous K/V cache | 0.9999931706 / 0.9999924131 |
| nonidentity paged decode | 0.9996770059 |
| two-layer direct composition | 0.9984903965 |
| advancing trace outputs, positions 39/40 | 0.9996835100 / 0.9996977323 |
| advancing trace K/V minimum | 0.9998138782 |
| dynamic page-table position 64 | 0.9999930220 |

The public nonaligned contract is preserved.  Production-default prefill
length 39 executes with chunk limit 4096.  A separate direct check temporarily
sets the internal chunk limit to 32 and proves the implementation owns the
seven-token tail, padding, cache fill, masking, and slicing instead of
requiring aligned public inputs.  Decode changes page tables and
device-resident position/RoPE tensors without recapture.  The capacity gate
allocates full advertised-context local K and V tensors
`[2048,2,64,128]` per device, preserving 131072 tokens at batch 1.  The final
cache, activation, residual, and CCL dtypes did not change, so
`doc/context_contract.json` remains valid without modification.

## Operation topology and inter-layer contract

The pre-tuning audit is in `work_log.md`.  It covers the four projection
groups, repeated same-input Q/K/V and gate/up projections, dedicated attention
ops, all material reshard/layout conversions, both row-parallel collective
boundaries, fused CCL+matmul candidates, and lower-movement residual families.

The final inter-layer contract is:

- Input and output residual: replicated BF16 `[1,B,S,8192]` on all four
  devices; decode uses the existing L1 width-sharded residual memory config
  inside each layer.
- QKV and gate/up: packed column-parallel BFP4 weights, local heads/features,
  no projection-result collective.
- Output and down: row-parallel BFP4 DRAM-sharded matmuls.  Each is reduced
  inside the layer using persistent BF16 `reduce_scatter_minimal_async` plus
  `all_gather_async`, then added to the replicated residual.
- Layer boundary: the returned replicated residual feeds the next layer
  directly.  There is no gather, reshard, reduce-scatter, or all-reduce between
  decoder layers.  Full-model bringup must preserve this boundary and must not
  add an inter-layer collective.

The complete two-layer sharded-residual family carried a quarter-hidden shard
through both layer invocations and gathered only outside the measured stack.
It improved eager decode 4.0% but regressed trace 32.1%.  The adapted fused
all-gather-matmul family regressed trace 240.8%.  Those results prove the
replicated boundary is faster overall on this mesh; it is not retained merely
because the test restored an old contract.

## Coherent family decisions

All values below are real-weight, whole-layer TP4 measurements.  A first API
failure was never used as a material rejection.

| Family | Control -> candidate | Correctness | Decision |
| --- | --- | --- | --- |
| shard-advisor full L1/1-D chain | trace 0.603105 -> 0.691715 ms | output 0.9999973; K/V >=0.999847 | reject |
| sharded residual + decomposed RS | trace 0.603087 -> 0.796816 ms | two layers >=0.9999943 | reject |
| sharded residual + fused AG-matmul | trace 0.603379 -> 2.056379 ms | two layers >=0.9999935 | reject after 7 adaptations |
| fused matmul-reduce-scatter | trace 0.603028 -> 0.888325 ms | PCC/cache unchanged | reject after weight-layout retry |
| packed -> split QKV | trace 0.602976 -> 0.637713 ms | PCC/cache unchanged | keep packed |
| packed -> split gate/up | trace 0.603330 -> 0.618173 ms | PCC unchanged | keep packed |
| persistent -> composite CCL | trace 0.602637 -> 0.620267 ms | PCC unchanged | keep persistent buffers |
| BF16 -> BF8 CCL on explicit SDPA | eager 1.452764 -> 1.583507 ms; trace 0.599246 -> 0.597527 ms | PCC unchanged | reject coherent policy; material eager loss |
| implicit -> explicit decode SDPA, repeat 1 | trace 0.602448 -> 0.598484 ms | PCC/cache unchanged | accept |
| implicit -> explicit decode SDPA, repeat 2 | trace 0.602594 -> 0.598459 ms | PCC/cache unchanged | accept |
| implicit -> explicit prefill SDPA | prefill 1.276612 -> 1.279915 ms | PCC unchanged | keep implicit prefill |
| DRAM -> L1 inputs for every prefill matmul | prefill 1.027875 -> 1.035686 ms | output/K/V PCC 1.0 | keep DRAM prefill inputs |

The fused AG-matmul path was retried with rank-4 weights, legal 1-D programs,
DRAM gate output, split gate/up, shorter tensor lifetimes, released gather
buffers, and a DRAM down-gather.  The fused matmul-RS path was retried after
fixing interleaved weight scope, rank, and layout.  Exact logs for every
adaptation are under `candidates/`.

## Precision, fidelity, sharding, and program evidence

The final `tt-perf-report` rows verify BF16 activation x BFP4 weight -> BF16
at LoFi for all four dominant decode matmuls.  The policy is measured, not
inferred from constructor defaults.  Targeted real-weight alternatives were:

| Tensor group / advice | Control -> candidate traced decode | PCC or blocker | Decision |
| --- | ---: | --- | --- |
| attention compute LoFi -> HiFi2 | 0.602439 -> 0.660596 ms | unchanged | LoFi |
| gate/up LoFi -> HiFi2 | 0.602294 -> 0.782758 ms | unchanged | LoFi |
| down LoFi -> HiFi2 | 0.602442 -> 0.692127 ms | unchanged | LoFi |
| attention weights BFP4 -> BFP8 | 0.602568 -> 0.618186 ms | K/V 0.992178 / 0.993648 | BFP4 |
| gate/up weights BFP4 -> BFP8 | 0.602483 -> 0.652143 ms | first 8-wide block overflowed L1; legal 4-wide retry passed | BFP4 |
| down weights BFP4 -> BFP8 | 0.602648 -> 0.624361 ms | output 0.9999907 | BFP4 |
| attention activations BF16 -> BFP8 | 0.603069 -> 0.615181 ms | output 0.9999979 | BF16 |
| MLP activations BF16 -> BFP8 | 0.603180 -> 0.604791 ms | output 0.9999983 | BF16 |
| cache BF16 -> BFP8, 8 recurrent positions | 0.603134 -> 0.601699 ms | output min 0.9999935 | reject: eager and nonaligned prefill regressed |

`$shard-advise` captured the exact rewritten dense block and produced 13 ops,
12 choices, zero spills, `report.json`, and `final_ir.mlir`.  Its complete
recommended L1/program chain was applied after adapting the report's partial
75-core row from the misleading compact `11x6` label to the IR's legal `11x7`
envelope.  The 14.7% trace regression rejects the seed as a coherent chain.

The completed multichip stage's ten isolated real-weight DRAM-sharded program
trials (`Q1/Q2/Q3`, `O1/O2/O4`, `G1/G3`, `D1/D3`) remain the starting geometry
provenance and selected O2/G1.  This pass additionally tested the advisor's
per-op core grids, inner blocks, output blocks, output subblocks, and L1
layouts.  It also adapted reduced-core MLP geometries until legal variants
executed: gate/up `G4B` and `G5C` regressed trace
`0.599128 -> 0.603500 ms` and `0.598904 -> 0.601097 ms`; down `D4B` and `D5C`
regressed trace `0.599190 -> 0.636235 ms` and
`0.599041 -> 0.683617 ms`.  Larger inner-block variants have exact retained L1
CB-overflow evidence.  The current DRAM-sharded TTNN program type does not
expose output subblock fields; applying the advisor's legal 1-D program family
is the direct measurement of that profiler advice, and it lost end to end.  No actionable
profiler advice remains deferred.

## Final profiler and performance accounting

The final reduced real-weight layer was profiled separately from watcher.
Advice-enabled tables and machine CSVs are under `final_tracy/`; the merged raw
operation CSV is retained losslessly as `.csv.gz`.

| Window | Ops | Device time | Op gap | Main rows |
| --- | ---: | ---: | ---: | --- |
| nonaligned prefill | 26 | 1018.519 us | 69871.552 us harness/capture gaps | QKV/output/gate-up/down about 63/45/301/134 us |
| five traced decode replays | 230 | 2919.591 us | 238.385 us steady; 17014.079 us range-entry artifacts | QKV/output/gate-up/down 41/35/204/108 us |
| traced decode per replay | 46 | 583.918 us | 47.677 us merged steady gaps | two RS 21--23 us; two AG 13--14 us |

The final traced wall latency is 597.632 us versus 583.918 us summed device-op
time per replay, leaving 13.714 us (2.3%) for dispatch/runtime and overlap
accounting.  Three 5.67-ms cross-device first-op/signpost gaps total
17014.079 us; they are Tracy range-entry artifacts, reported separately and
excluded from the 238.385-us five-replay steady gap.  Merged op gaps are not
additive to wall time because operations from four devices overlap.  The
overall report observes 35.8% DRAM roofline for decode and 21.2% for prefill.
Gate/up is about 35% and down
about 18.5% of device time; both were attacked with geometry, sharding,
packing, activation, weight, and fidelity trials.  The explicit SDPA reduces
the decode SDPA row from roughly 16--17 us to 12 us while preserving the op
topology.

The required three-number accounting is:

| Path | Warmed wall | Device ops | Theoretical DRAM lower bound |
| --- | ---: | ---: | ---: |
| traced decode replay | 597.632 us | 583.918 us | about 235.03 us |

Each device reads 120,324,096 bytes of stored BFP4 tile weights per layer
(`576 bytes / 1024 elements` across local QKV/output/gate-up/down shapes
`8192x2560`, `2048x8192`, `8192x14336`, and `7168x8192`).  The measured short
context adds about 12 KB of BF16 K/V reads per device, so
`(120324096 + about 12288) / 512000 = about 235.03 us`.  Equivalently, the
four-device total is divided by the aggregate 2048 GB/s bandwidth.

As an independent profiler-model check, `tt-perf-report` attributes
106,954,752 bytes per replay to modeled ops:
`106954752 / 512000 = 208.896 us`, also
`583.918 us * 35.775% = 208.896 us`.  This lower internal estimate is not used
as the theoretical headline because it omits unclassified SDPA K/V traffic
and does not equal the complete stored BFP4 tile footprint.  The reduced
prefill profile separately records 1018.519 us of device work against the
canonical 1281.216-us warmed wall result.

Final trace device time improves from the current-pass baseline profiler's
587.0 us/replay to 583.9 us/replay.  The final default wall value, rather than
the best isolated candidate value, is the reported optimized number.

## Runtime, stress, and health gates

- The source-backed runtime audit checks all prefill/decode methods for
  `from_torch`, `to_torch`, `torch.`, and single-chip `super()` fallbacks; it
  passes and proves the async CCL, paged SDPA, and device embedding path.
- `watcher_final.xml`/`watcher_final.log`: 1 passed, 10 deselected; watcher was
  attached to Tensix on devices 0--3 for 17 eager positions, position 64, and
  100 advancing trace replays.  No assertion, timeout, stall, or non-finite
  output occurred.  ETH watcher instrumentation remains disabled because its
  instrumented fabric-router binary exceeds the hardware kernel-config buffer.
- `device_health_final.log` retains the post-watcher `tt-smi -s` report: all
  four p300c devices healthy, all DRAM channels enabled, and zero
  corrected/uncorrected GDDR errors.  No
  reset was needed during this optimization pass.
- `python -m py_compile` and `git diff --check` pass.
- The final independent `$stage-review` verdict is `clean-pass`; both earlier
  `more-work-needed` finding sets were fixed and rereviewed.

Known non-gating environment notices are firmware 19.8.0 being newer than the
latest fully tested 19.5.0 bundle, the unknown B850M-C topology fallback, low
`/dev/shm` headroom, and nanobind shutdown leak diagnostics already present in
the completed stage.  None caused a watcher or model failure.

## Artifact index

- `baseline/reproduction.{xml,log}`: current-pass before correctness and wall
  latency.
- `baseline/tracy/`: before advice tables, CSVs, plots, capture log, and merged
  operation provenance.
- `candidates/`: every current-pass real-weight candidate and adapted retry.
- `shard_advise/`: capture script, `report.json`, `report.txt`, `final_ir.mlir`,
  compressed decision trace, and tool/ABI provenance.
- `final_default.{xml,log}`: canonical final default correctness and latency.
- `final_tracy/`: final advice tables, CSVs, renderer logs, and compressed raw
  operation provenance.
- `watcher_final.{xml,log}`: final watcher/stress evidence.
- `device_health_final.log`: retained post-watcher device-health evidence.
- `static_gates.log`: retained compile, diff, whitespace, and artifact-hygiene
  gate results.
- `STAGE_REVIEW_INITIAL.md`, `STAGE_REVIEW_WIRING.md`, and
  `STAGE_REVIEW_FINAL.md`: independent findings and their final disposition.
- `work_log.md`: commands, complete decisions, limitations, and commit/review
  record.

Full-model, LM-head, sampling, text generation, and vLLM work are intentionally
outside this decoder-only stage and were not started.  MoE active-expert work
is inapplicable because Llama 3.1 70B is dense.
