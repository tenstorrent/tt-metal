# Optimized multichip decoder work log

Date: 2026-07-19

## Scope and frozen baseline

- Model: `meta-llama/Llama-3.1-70B-Instruct`.
- Target: the complete local four-Blackhole p300c mesh as logical `1x4`, TP4,
  ring topology. Single-chip and replicated-compute paths are references only.
- Starting commit: `9280cf5c2b0`; completed multichip implementation commit:
  `1087041f65f`.
- Stage scope is `tt/multichip_decoder.py`, its model-local tests, context
  contract if affected, and this documentation tree. No full-model or vLLM
  work is in scope.
- The unrelated pre-existing modification to
  `.agents/skills/forge-functional-decoder-from-ir/SKILL.md` is preserved and
  excluded from the stage commit.
- Frozen before evidence:
  `doc/multichip_decoder/final.xml`, `final.log`, and `tracy_final/`.
  Committed headline values are prefill `1.653101 ms`, eager decode
  `1.556849 ms`, and traced warmed decode `0.598285 ms`; current-pass
  reproduction is recorded below before candidate work. The old canonical
  test forced `prefill_mlp_chunk_size=32`, so its prefill number and the first
  current-pass reproduction are historical 32+7-tail measurements, not a
  production-default before/after comparison.

## Hardware orientation

```text
flock /tmp/tt-device.lock timeout 60 tt-smi -ls --local
```

Result: devices 0--3 are visible as four Blackhole p300c ASICs.

```text
flock /tmp/tt-device.lock env TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
  python <open MeshShape(1,4), close, print MESH_SMOKE_OK>
```

Result: `MESH_SMOKE_OK`. Known non-gating environment notices match the prior
stage: firmware 19.8.0 newer than tested 19.5.0, B850M-C bus-ID topology
fallback, and low `/dev/shm` headroom.

## Operation-topology audit before tuning

The audit combines the current source path with the frozen trace report. Times
below are per replay from the five-replay signposted baseline unless stated.

| Boundary / sequence | Current contract and material cost | Rewrite or lower-movement candidate | Constraints | Initial action |
| --- | --- | --- | --- | --- |
| input residual -> RMSNorm -> packed QKV | replicated TP4 residual, 32-core width-sharded L1; norm ~8 us; one packed local QKV matmul ~41 us | distributed/fractured residual norm; shard-advisor L1 chain; BFP4/LoFi vs BFP4/HiFi2 and BFP8 fidelities | QKV output is local heads; decode public batch remains 1 with 32 padded rows | measure as a coherent residual family; run shard-advisor seed |
| packed QKV -> create heads -> RoPE -> cache -> SDPA -> concat heads | dedicated create-heads, rotary, paged update, paged SDPA, concat-head ops already used; SDPA ~16--17 us | explicit Blackhole SDPA config; BFP8 cache; lower-movement head layouts | preserve two local KV heads, page-table mapping, nonaligned logical lengths, trace-dynamic positions | retain dedicated ops; sweep cache/config and inspect total attention cost |
| local output matmul -> reduction -> attention residual | local `[32,2048] x [2048,8192]`, ~35 us; persistent RS ~21--23 us + AG ~12--14 us restores replicated hidden | matmul+RS; RS carried into distributed residual/norm; fused AG-matmul with output-sharded weights; collective placement/dtype/persistence | candidate must carry its residual contract through post-attention norm/MLP instead of immediately restoring replication | implement/adapt and measure family; first validator error is not rejection |
| post-attention RMSNorm -> packed gate/up | replicated residual, norm ~8 us; one packed local gate/up matmul ~204 us | sharded/fractured norm; split gate/up with fused SiLU family; advisor 1-D candidate; geometry/precision cross-product | compare packed whole-MLP including two slices and fused `mul(SiLU(gate), up)` | measure packed and split under identical dtype/layout; keep whole-MLP winner |
| packed output slices -> fused SiLU multiply -> local down matmul | slices ~2 us each, fused binary ~13 us; local down ~108 us | phase-specific activation shard; larger/non-power-of-two `in0_block_w`; output subblocks; BFP4/LoFi vs higher fidelities | preserve BFP4 real-weight correctness; account for reshard once across projection group | precision-locked geometry sweep |
| local down matmul -> reduction -> layer output | persistent RS ~22 us + AG ~13--14 us restores replicated boundary | matmul+RS; sharded residual add; carry sharded/fractured layer output into next layer and next distributed norm; delayed gather only outside measured stack | final inter-layer contract may not insert gather/reshard/all-reduce; test harness may compose only outside measured path | implement two-layer-compatible lower-movement family or prove exact op blocker |
| prefill norms/matmuls/collectives | DRAM-interleaved activations; packed QKV/gate-up; output/down RS+AG composite; committed warmed prefill 1.653101 ms | 2-D grid/config sweep, collective placement/dtype, split vs packed projection comparison, internal chunk tail | preserve arbitrary logical sequence lengths and 131072-token batch-1 context | current-pass warmed/profiled measurement then candidate A/B |

### Dedicated-op and structural rewrite findings

- Already dedicated and retained: `ttnn.rms_norm`, packed QKV projection plus
  `nlp_create_qkv_heads_decode`, `rotary_embedding`, paged cache update,
  paged SDPA decode, `nlp_concat_heads_decode`, and fused SiLU input activation
  on the gate/up multiply.
- Q/K/V already share one packed BFP4 weight/matmul; no split Q/K/V topology
  exists in the measured default.
- Gate/up already share one packed BFP4 weight/matmul. A tuned separate family
  remains mandatory evidence because packing adds slices and constrains N/L1.
- No host/PyTorch conversion belongs in the runtime path. Runtime fallback and
  inter-layer collective audits are final gates.
- Material rewrite work is concentrated at the output/down row-parallel
  matmul-to-collective boundaries and the residual layout they feed.

## Current-pass baseline reproduction

```text
flock /tmp/tt-device.lock env \
  LLAMA_31_70B_REAL_WEIGHT_FILE=/home/mvasiljevic/hf-cache/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b/model-00015-of-00030.safetensors \
  TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
  pytest -sv models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_multichip_decoder.py \
  --junitxml=models/autoports/meta_llama_llama_3_1_70b_instruct/doc/optimized_multichip_decoder/baseline/reproduction.xml
```

Result: 5 passed, 4 opt-in candidate/profiler/watcher tests skipped.

| Baseline check | Current-pass value |
| --- | ---: |
| nonaligned prefill PCC | 0.9999978879 |
| decode PCC | 0.9996807609 |
| contiguous K/V PCC | 0.9999931706 / 0.9999924131 |
| two-layer direct composition PCC | 0.9984936517 |
| paged decode PCC | 0.9996794316 |
| advancing trace outputs | 0.9996876997 / 0.9997024905 |
| position-64 dynamic page trace | 0.9999930420 |
| single / TP4 warmed prefill | 3.719677 / 1.693824 ms |
| single / TP4 eager decode | 1.852614 / 1.562353 ms |
| single / TP4 traced warmed decode | 1.846958 / 0.598971 ms |

Artifacts: `baseline/reproduction.xml` and `baseline/reproduction.log`.

The baseline source used the inherited test-only chunk size 32. Its decode
numbers remain valid historical context; its prefill number is retained only
as tail-policy provenance. The later paired stage A/B and final canonical run
use the production default chunk size 4096.

## Current-pass baseline profiler

Watcher was not enabled. The reduced real-weight multichip layer path was
captured with:

```text
flock /tmp/tt-device.lock env \
  LLAMA_31_70B_REAL_WEIGHT_FILE="$LLAMA_31_70B_REAL_WEIGHT_FILE" \
  RUN_MULTICHIP_DECODER_PROFILER=1 \
  TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
  python -m tracy -p -r \
  -o models/autoports/meta_llama_llama_3_1_70b_instruct/doc/optimized_multichip_decoder/baseline/tracy \
  -m pytest -sv models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_multichip_decoder.py \
  -k profiler_smoke
```

`tt-perf-report` was run with advice enabled for both signposted windows; the
machine-readable CSV runs used `--csv`, and the human tables used
`--no-summary` without `--no-advice`.

| Window | Device ops | Device time | Op gap | Observed aggregate DRAM |
| --- | ---: | ---: | ---: | ---: |
| prefill | 26 | 1017 us | 69089 us harness/capture gaps | 109 GB/s / 21.3% |
| five traced decode replays | 230 | 2935 us | 263 us | 182 GB/s / 35.6% |
| traced decode per replay | 46 | 587.0 us | 52.6 us | same capture |

Decode rows reproduce the frozen topology: QKV/output/gate-up/down are about
41/35/204/108 us; each persistent BF16 reduce-scatter is 21--23 us and each
all-gather 13--14 us. The selected projection rows prove the measured policy
is BF16 activation x BFP4 weight -> BF16 at LoFi. Advice marks every projection
`SLOW`, notes absent output-subblock metadata for DRAM-sharded decode rows, and
suggests higher fidelity for accuracy. That advice is actionable only as an
A/B: the current real-weight LoFi path already passes, so HiFi2/HiFi4 must win
or improve required correctness enough to justify any latency cost.

Artifacts:

- `baseline/tracy/{prefill,decode}_perf_report.txt`: advice-backed tables.
- `baseline/tracy/{prefill,decode}_perf_report.csv`: filtered machine tables.
- `baseline/tracy/*perf_report.console.log`: exact renderer provenance.
- `baseline/tracy/reports/2026_07_19_06_10_16/ops_perf_results_2026_07_19_06_10_16.csv.gz`:
  losslessly compressed raw merged op provenance.

## Shard-advisor seed (OPT-015)

The current-pass capture uses exact TP4-local projection shapes and current
BFP4 weights: QKV `[8192,2560]`, output `[2048,8192]`, packed gate/up
`[8192,14336]`, and down `[7168,8192]`. Because the advisor does not model mesh
CCLs, the two row-parallel reductions are documented identity edges in the
capture only; the hardware candidate retains the real persistent RS+AG path.

The first invocation failed before capture because the prebuilt
`libTTMLIRRuntime.so` expected the tt-mlir third-party tt-metal
`moe_compute` ABI while the active checkout exported a newer signature. The
tool was realigned without rebuilding: its matching third-party tt-metal
Python package/library path and `_ttnncpp.so` were used. Capture then completed
with 13 ops, 12 final choices, zero spills, and no unfixable ops.

Authoritative `final_ir.mlir` seed:

| Role | Advisor input/output layout | 1-D program seed |
| --- | --- | --- |
| input norm | block-sharded 11 cores | 11x1 norm chain |
| QKV | width-sharded 32 -> 80 cores | 11x8, `in0_block_w=8`, `per_core_N=1`, subblock 1 |
| output | L1 interleaved -> width-sharded 86 cores | 11x8, block 2, `per_core_N=3`, subblock 3 |
| post norm | block-sharded 11 cores | 11x1 norm chain |
| packed gate/up | width-sharded 64 -> 90 cores | 11x9, block 2, `per_core_N=5`, subblock 5 |
| fused SiLU multiply | width-sharded 75 cores | reshard once to down input |
| down | width-sharded 56 -> 86 cores | 11x8, block 2, `per_core_N=3`, subblock 3 |

The first hardware application used the compact report's misleading `11x6`
label for the 75-core multiply output and failed because 75 cores cannot fit 66.
The authoritative IR explicitly has a seventh partial row (`66 + 9`); adapting
the envelope to `11x7` made the exact seed execute and pass.

| Seed A/B | Control | Advisor seed | Decision |
| --- | ---: | ---: | --- |
| eager decode | 1.677256 ms | 1.853851 ms | reject, 10.5% slower |
| traced warmed decode | 0.603105 ms | 0.691715 ms | reject, 14.7% slower |
| output PCC | reference | 0.9999973162 | pass |
| K/V PCC | reference | 0.9998474113 / 0.9998772803 | pass |

The seed uses per-device DRAM-interleaved BFP4 weights plus the full advised
L1 chain. It therefore fairly tests the advisor's 1-D strategy against the
existing DRAM-sharded BFP4/LoFi path; it is not rejected from a lone matmul or
an immediate layout restore. The DRAM-sharded search remains authoritative.

Artifacts:

- `shard_advise/advise_llama70b_tp4_local.py`.
- `shard_advise/run/{report.json,report.txt,final_ir.mlir}` and decision trace.
- `shard_advise/pipeline.stderr.log` (ABI realignment warnings/provenance).
- `candidates/advisor_1d_seed_attempt1_grid_error.{xml,log}`.
- `candidates/advisor_1d_seed.{xml,log}`.

## Residual-layout and fused-CCL families

The row-parallel output and down-projection boundaries were evaluated as
complete, two-layer-compatible residual-layout families.  Candidate gathers
used by the test harness happen only after the measured layer stack; no
candidate was rejected after immediately restoring the old replicated input
contract.

| Decode family | Layer 0 / layer 1 PCC | K / V cache PCC | Eager control / candidate | Trace control / candidate | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| replicated residual, persistent RS+AG (default) | baseline | baseline | 1.689609 / 1.689609 ms | 0.603087 / 0.603087 ms | control |
| sharded residual, decomposed matmul + RS | 0.9999973653 / 0.9999942507 | 0.9999053090 / 0.9999040060 | 1.689609 / 1.622209 ms | 0.603087 / 0.796816 ms | reject: trace +32.1% |
| sharded residual, fused AG+matmul | 0.9999969930 / 0.9999934828 | 0.9998444126 / 0.9998579711 | 1.704780 / 2.069338 ms | 0.603379 / 2.056379 ms | reject: trace +240.8% |

The decomposed family leaves each layer with a quarter-width hidden shard,
computes RMSNorm from a global sum-of-squares reduction, and feeds the shard
directly to the next layer.  Although eager latency improves by 4.0%, its
extra traced collectives and fragmented normalization increase the required
traced decode metric by 32.1%; it is therefore not a viable inter-layer
contract for this model.

The fused all-gather-matmul family went through the following adaptations
before measurement, so the rejection is not based on a first TTNN/API error:

1. rank-2 fused weights were changed to the required rank-4 representation;
2. QKV/output/gate/down layouts and 1-D matmul programs were made legal;
3. the packed gate/up projection was split after its gathered result exceeded
   L1, providing a real split-projection trial rather than a source argument;
4. intermediate fused-gather and projection tensors were explicitly released;
5. the down-projection gathered intermediate was placed in DRAM after the
   legal tile-padded tensor still exceeded available L1.

All four fused projection sites then executed.  The successful family has
good two-layer and cache PCC but is substantially slower both eagerly and
under trace; no fused all-gather-matmul remains in the selected path.

Artifacts:

- `candidates/sharded_residual_decomposed.{xml,log}`.
- `candidates/sharded_residual_fused_agmm_attempt{1,2,3_dram_gate,4_split_gate,5_lifetimes,6_gather_lifetimes}.{xml,log}`.
- `candidates/sharded_residual_fused_agmm_attempt7_dram_down_gather.{xml,log}`.

## Fused matmul-reduce-scatter family

The second adjacent CCL rewrite used `ttnn.experimental.matmul_reduce_scatter`
for the attention output and down projections, then gathered only because this
candidate intentionally retained the replicated residual contract. This is a
distinct fused-primitive measurement from the complete sharded-residual family
above.

Attempt one incorrectly exposed an interleaved QKV weight to the candidate
program. The measured retry scoped only output/down weights, changed them to
the required rank/layout, and used legal fused program configurations. It then
completed the whole layer with unchanged output/cache PCC.

| Fused MMRS A/B | Control | Candidate | Result |
| --- | ---: | ---: | --- |
| eager decode | 1.533670 ms | 1.512141 ms | 1.4% faster |
| traced warmed decode | 0.603028 ms | 0.888325 ms | 47.3% slower |

The traced metric rejects the compatibility family. The complete
stack-compatible lower-movement family is not inferred from this result; it is
the independently measured two-layer sharded-residual result above.

Artifacts: `candidates/fused_mmrs_attempt1.{xml,log}` and
`candidates/fused_mmrs_attempt2_weights.{xml,log}`.

## Projection packing, collective placement, and persistence

Packed QKV and packed gate/up are repeated same-input projection rewrites in
the inherited default. Legal separate alternatives were built as whole-layer
policies and include all split, activation, binary, and layout costs.

| Whole-layer family | Correctness | Eager control / candidate | Trace control / candidate | Decision |
| --- | ---: | ---: | ---: | --- |
| packed -> split QKV | output/cache unchanged | 1.668866 / 1.777587 ms | 0.602976 / 0.637713 ms | keep packed |
| packed -> split gate/up | output unchanged | 1.686212 / 1.600729 ms | 0.603330 / 0.618173 ms | keep packed |
| persistent RS+AG -> composite all-reduce | output/cache unchanged | 1.676519 / 1.654858 ms | 0.602637 / 0.620267 ms | keep persistent |

The split-QKV constructor signature was corrected after attempt one. The
separate gate/up prefill family also passed at the nonaligned logical length
and measured `1.274407 / 1.303309 ms`, so packing wins both required phases.

The current pass remeasured composite CCL under the current projection and
residual policies. The completed stage's topology evidence additionally
retains the following exact real-weight pairs:

| Inherited topology A/B | Control trace | Candidate trace | Other evidence | Decision |
| --- | ---: | ---: | --- | --- |
| persistent BF16 | 0.620704 ms | 0.603138 ms | eager 1.471944 / 1.451630 | selected starting policy |
| BF8 composite | 0.620839 ms | 0.619271 ms | eager 1.471385 / 1.509472 | reject eager loss |
| one link | 0.620960 ms | 0.642511 ms | exact PCC | reject |
| linear topology | 0.620872 ms | 0.629654 ms | output PCC 0.9999993 | reject |
| persistent BF8 | 0.620942 ms | 0.601112 ms | eager 1.543350 / 1.614094 | reject 4.6% eager loss for 0.34% trace gain over BF16 persistent |

The final decode CCL path is model-applicable async communication with two
links, ring topology, ten chunks per sync, two workers per link, and two
buffers per channel. Persistent intermediate, scattered, and gathered L1
buffers are allocated once and reused by both intra-layer reductions.

Artifacts: `candidates/{split_qkv_attempt1,split_qkv_attempt2_ctor,
split_gate_up_attempt1,prefill_split_gate_up,composite_current}.{xml,log}` and
the inherited `doc/multichip_decoder/topology_*.{xml,log}`.

## Collective dtype and precision/fidelity families

Every trial used real layer-39 weights and the full TP4 layer. The profiler's
higher-fidelity recommendation was applied independently to the material
attention, gate/up, and down groups rather than as a global switch.

| Policy | Key PCC / adaptation | Control -> candidate | Decision |
| --- | --- | ---: | --- |
| attention LoFi -> HiFi2 | unchanged | trace 0.602439 -> 0.660596 ms | reject |
| gate/up LoFi -> HiFi2 | unchanged | trace 0.602294 -> 0.782758 ms | reject |
| down LoFi -> HiFi2 | unchanged | trace 0.602442 -> 0.692127 ms | reject |
| attention weight BFP4 -> BFP8 | K/V 0.992178 / 0.993648 | trace 0.602568 -> 0.618186 ms | reject |
| gate/up weight BFP4 -> BFP8 | first 8-wide block exceeded L1; 4-wide retry passed | trace 0.602483 -> 0.652143 ms | reject |
| down weight BFP4 -> BFP8 | output 0.9999907 | trace 0.602648 -> 0.624361 ms | reject |
| attention activation BF16 -> BFP8 | output 0.9999979; K/V >=0.999871 | trace 0.603069 -> 0.615181 ms | reject |
| MLP activation BF16 -> BFP8 | output 0.9999983 | trace 0.603180 -> 0.604791 ms | reject |

The final BFP4/LoFi policy is therefore the fastest passing real-weight policy,
not a synthetic precision veto. The final profiler rows explicitly show
`LoFi BF16 x BFP4 => BF16` for QKV, output, gate/up, and down.

### BF8 collective dtype

On the inherited implicit-SDPA control, BF8 CCL passed but regressed eager
decode `1.679409 -> 1.756905 ms`; traced decode moved
`0.602976 -> 0.601548 ms`. Nonaligned prefill moved
`1.275328 -> 1.272675 ms`. Because the accepted SDPA configuration changed
the local attention cost, BF8 CCL was retried on that exact topology:

| Explicit-SDPA CCL A/B | BF16 | BF8 |
| --- | ---: | ---: |
| eager decode | 1.452764 ms | 1.583507 ms |
| traced warmed decode | 0.599246 ms | 0.597527 ms |

The 0.29% trace gain does not compensate for a 9.0% eager regression in the
coherent final policy, and the completed persistent-BF8 evidence independently
shows the same tradeoff. Final CCL remains BF16.

### BFP8 cache

A one-position BFP8-cache trial passed output and cache PCC and slightly
improved trace. It was expanded rather than accepted from that first result:

- eight recurrent positions: output PCC minimum 0.999993521, K/V
  0.999967/0.999976;
- eager `1.547977 -> 1.589394 ms`, trace `0.603134 -> 0.601699 ms`;
- nonaligned prefill output PCC 1.0, K/V 0.999975/0.999977, latency
  `1.266910 -> 1.291243 ms`.

The trace-only gain is outweighed by eager and prefill regressions. BF16 cache
remains selected, and the 131072-token capacity contract therefore requires no
dtype/layout update.

Artifacts: the `attention_*`, `gate_up_*`, `down_*`, `mlp_activation_bfp8`,
`ccl_bf8_*`, `prefill_ccl_bf8`, `cache_bfp8_*`, and `prefill_cache_bfp8`
files under `candidates/`.

## Accepted SDPA program rewrite

The final winner explicitly configures the existing optimized composite paged
SDPA op rather than rebuilding attention from primitive operations. The decode
program uses the legal Blackhole 8x8 grid and HiFi2 compute kernel.

| Repeated A/B | Implicit control | Explicit decode SDPA | Output/cache |
| --- | ---: | ---: | --- |
| eager run 1 | 1.686267 ms | 1.514553 ms | unchanged |
| trace run 1 | 0.602448 ms | 0.598484 ms | unchanged |
| eager run 2 | 1.607507 ms | 1.481564 ms | unchanged |
| trace run 2 | 0.602594 ms | 0.598459 ms | unchanged |

Nonaligned prefill was measured separately: `1.276612 / 1.279915 ms`.
Accordingly, the default optimization flags remain enabled for decode and
`MultiChipDecoder.__init__` sets only `prefill_sdpa_program_config=None`. This
keeps the public nonaligned prefill path on its faster implicit program.

Artifacts: `candidates/{explicit_sdpa,explicit_sdpa_repeat,
prefill_explicit_sdpa}.{xml,log}`.

## Production-default stage before/after

After the initial review identified that the frozen test policy was unsuitable
for a prefill headline, a dedicated real-weight A/B constructed the starting
and selected policies side by side. Both used `MultiChipConfig()`'s production
prefill chunk of 4096; the only difference was the two accepted explicit
decode-SDPA flags.

The selected policy versus starting-policy PCC is `1.0` for prefill,
`0.9999979552` for decode, and `1.0/1.0` for K/V.

| Paired stage A/B | Before | After | Change |
| --- | ---: | ---: | ---: |
| warmed prefill | 1.267830 ms | 1.266436 ms | -0.11% |
| eager decode | 1.572428 ms | 1.454311 ms | -7.51% |
| traced warmed decode | 0.603287 ms | 0.598957 ms | -0.72% |

Artifact: `candidates/stage_before_after_defaults.{xml,log}`. This is the
required like-for-like before/after comparison; the independently rerun final
default below is the final-path headline.

## Program geometry and profiler-advice closure

The completed multichip stage supplied ten isolated real-weight DRAM-sharded
decode geometry trials for all four dominant projections:
`Q1/Q2/Q3`, `O1/O2/O4`, `G1/G3`, and `D1/D3`. They sweep core count/grid,
larger legal inner blocks, and per-core output blocks. O2 and G1 are the
retained starting winners; QKV remains 16/16/5 and down remains 32/7/8. This
pass then applied the shard-advisor's exact per-op L1 layout, core grid, inner
block, output block, and output-subblock chain as a first current-pass seed.

| Geometry | Candidate `(input cores, in0 block, per-core N)` | Output PCC | Eager control / candidate | Trace control / candidate |
| --- | --- | ---: | ---: | ---: |
| Q1 | QKV `(8,32,5)` | 0.9999977696 | 1.477042 / 1.509512 ms | 0.635825 / 0.634991 ms |
| Q2 | QKV `(32,8,5)` | 0.9999982559 | 1.467788 / 1.449707 ms | 0.636806 / 0.639075 ms |
| Q3 | QKV `(8,32,2)` | 0.9999977696 | 1.562921 / 1.517662 ms | 0.636593 / 0.635800 ms |
| O1 | output `(16,4,8)` | 0.9999987711 | 1.478814 / 1.478611 ms | 0.636933 / 0.632112 ms |
| O2 | output `(8,8,8)` | 0.9999987121 | 1.521315 / 1.451859 ms | 0.636551 / 0.630560 ms |
| O4 | output `(8,8,4)` | 0.9999987121 | 1.476444 / 1.504659 ms | 0.636681 / 0.629556 ms |
| G1 | gate/up `(32,8,14)` | 0.9999985719 | 1.537236 / 1.470207 ms | 0.636795 / 0.626549 ms |
| G3 | gate/up `(64,4,7)` | 1.0000000000 | 1.523128 / 1.578473 ms | 0.636650 / 0.639572 ms |
| D1 | down `(16,14,8)` | 0.9999992766 | 1.513883 / 1.515429 ms | 0.636543 / 0.646495 ms |
| D3 | down `(16,14,4)` | 0.9999992766 | 1.468577 / 1.475176 ms | 0.636748 / 0.645673 ms |

These precision-locked BFP4/LoFi XML properties are in the inherited
`doc/multichip_decoder/geometry_*.xml` files. O4's 1.0-us absolute trace
advantage over O2 in separate runs is within the observed run variance, while
O2 improves eager by 4.6% and also improves its paired trace; O2 is the robust
output choice. The same paired-trace plus eager criterion retains the baseline
QKV geometry over the marginal Q1/Q3 trace movements.

The initial review requested lower-core MLP coverage beyond G1/G3 and D1/D3.
Every material L1 failure was adapted by reducing the inner block and/or
per-core N until a legal lower-core candidate executed:

| Candidate | Geometry | Evidence | Decision |
| --- | --- | --- | --- |
| G4/G4A | 16 cores, inner 16, N 14/7 | CB needs 1,839,872 bytes > 1,572,864 | adapt |
| G4B | 16 cores, inner 8, N 14 | eager 1.533124 -> 1.545209; trace 0.599128 -> 0.603500 ms | reject |
| G5/G5A/G5B | 8 cores, inner 32/32/16, N 14/7/14 | CB needs 3,453,696 or 1,839,872 bytes | adapt |
| G5C | 8 cores, inner 8, N 14 | eager 1.478591 -> 1.471172; trace 0.598904 -> 0.601097 ms | reject |
| D4/D4A | 8 cores, inner 28, N 8/4 | CB needs 1,839,872 bytes | adapt |
| D4B | 8 cores, inner 14, N 8 | eager 1.554721 -> 1.550240; trace 0.599190 -> 0.636235 ms | reject |
| D5/D5A/D5B | 4 cores, inner 56/56/28, N 8/4/8 | CB needs 3,502,848 or 1,839,872 bytes | adapt |
| D5C | 4 cores, inner 14, N 8 | eager 1.516453 -> 1.573458; trace 0.599041 -> 0.683617 ms | reject |

All four executing reduced-core variants preserve output/cache PCC. Artifacts
are `candidates/geometry_{G4,G4A,G4B,G5,G5A,G5B,G5C,D4,D4A,D4B,D5,D5A,D5B,D5C}.{xml,log}`.

`tt-perf-report` gave two classes of actionable advice:

1. move prefill inputs to L1 and use the shown inner/output-subblock choices;
2. use HiFi2/HiFi4 with BF16 activations for accuracy.

The full advisor chain is the legal decode L1/program-family attempt and
regressed trace 14.7%. Item one was also applied directly to all four prefill
matmuls as one whole-prefill family. It produced exact output/K/V PCC 1.0 and
regressed warmed sequence-7 prefill `1.027875 -> 1.035686 ms`; therefore
prefill's DRAM-interleaved inputs remain selected. The
DRAM-sharded decode program API does not expose output-subblock fields, so the
advisor's legal 1-D program family is also the direct output-subblock trial.
The per-group HiFi2 results above close item two. No applicable advice is
deferred.

Artifact: `candidates/prefill_l1_inputs.{xml,log}`.

## Final canonical default run

```bash
flock /tmp/tt-device.lock env \
  LLAMA_31_70B_REAL_WEIGHT_FILE=/home/mvasiljevic/hf-cache/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b/model-00015-of-00030.safetensors \
  TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
  pytest -sv models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_multichip_decoder.py \
  --junitxml=models/autoports/meta_llama_llama_3_1_70b_instruct/doc/optimized_multichip_decoder/final_default.xml
```

Result: 5 passed, 6 intentional opt-in tests skipped.

| Final check | Value |
| --- | ---: |
| nonaligned prefill PCC | 0.9999978879 |
| production prefill chunk limit | 4096 |
| explicit decode / implicit prefill policy | asserted in runtime JUnit properties |
| separate 32+7 internal-tail PCC | 0.9999978879 |
| decode PCC | 0.9996770059 |
| contiguous K/V PCC | 0.9999931706 / 0.9999924131 |
| two-layer direct composition PCC | 0.9984903965 |
| paged decode PCC | 0.9996770059 |
| advancing trace outputs | 0.9996835100 / 0.9996977323 |
| dynamic position-64 page output | 0.9999930220 |
| optimized single / TP4 prefill | 3.714869 / 1.281216 ms |
| optimized single / TP4 eager decode | 1.855658 / 1.507943 ms |
| optimized single / TP4 traced decode | 1.847464 / 0.597632 ms |

The full local cache allocation remains K and V each `[2048,2,64,128]` per
device. The final default is the measured path; candidate-only timing is not
substituted for these headline values.

## Final profiler and performance accounting

Profiler and watcher were separate hardware runs. Final capture command:

```bash
flock /tmp/tt-device.lock env \
  LLAMA_31_70B_REAL_WEIGHT_FILE="$LLAMA_31_70B_REAL_WEIGHT_FILE" \
  RUN_MULTICHIP_DECODER_PROFILER=1 \
  TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
  python -m tracy -p -r \
  -o models/autoports/meta_llama_llama_3_1_70b_instruct/doc/optimized_multichip_decoder/final_tracy \
  -m pytest -sv models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_multichip_decoder.py \
  -k profiler_smoke
```

`tt-perf-report --arch blackhole --no-color` was rendered with advice enabled,
using `MULTICHIP_PREFILL`..`MULTICHIP_PREFILL_END` and
`MULTICHIP_TRACE_DECODE`..`MULTICHIP_TRACE_DECODE_END`. Machine tables used
`--csv`; human tables used `--no-summary` without suppressing advice.

| Window | Ops | Device time | Op-to-op gap |
| --- | ---: | ---: | ---: |
| prefill | 26 | 1018.519 us | 69871.552 us capture/harness gaps |
| five traced decode replays | 230 | 2919.591 us | 238.385 us steady plus 17014.079 us range-entry artifacts |
| traced decode per replay | 46 | 583.918 us | 47.677 us merged steady gaps |

The final wall trace is 597.632 us, 13.714 us (2.3%) above summed device-op
time. Three cross-device first-operation/signpost boundaries contribute
17014.079 us of Tracy range-entry artifacts; they are not part of the steady
five-replay gap. The merged
op-gap column cannot be added to wall time because it merges overlapping
operations from four devices. Final projection rows are approximately
41/35/204/108 us for QKV/output/gate-up/down. The overall report observes
35.8% DRAM roofline for decode and 21.2% for prefill. Gate/up and down are about
35% and 18.5% of device time and were each attacked with packing, geometry,
sharding, weight, activation, and fidelity candidates. The two persistent
RS+AG pairs remain about 12% and have coherent placement/fusion/residual/dtype
evidence.

The required wall/device/theoretical latency accounting is:

| Path | Warmed wall | Device ops | Theoretical DRAM lower bound |
| --- | ---: | ---: | ---: |
| traced decode replay | 597.632 us | 583.918 us | about 235.03 us |

Per device, the four local projection weights contain:

| Stored BFP4 tile weight | Elements | Bytes at 576 bytes/1024 elements |
| --- | ---: | ---: |
| QKV `8192x2560` | 20,971,520 | 11,796,480 |
| output `2048x8192` | 16,777,216 | 9,437,184 |
| gate/up `8192x14336` | 117,440,512 | 66,060,288 |
| down `7168x8192` | 58,720,256 | 33,030,144 |
| total | 213,909,504 | 120,324,096 |

The measured short-context replay adds about 12 KB of BF16 K/V reads per
device. Thus `(120324096 + about 12288) / 512000 = about 235.03 us` at each
device's 512 GB/s peak; multiplying numerator and denominator by four gives
the same four-device aggregate result.

For a separate model consistency check, `tt-perf-report` attributes
534,773,760 bytes to modeled ops across five replays, or 106,954,752 per
replay. `106954752 / 512000 = 208.896 us`, matching
`583.918 us * 0.35775 = 208.896 us`. This internal estimate omits unclassified
SDPA K/V traffic and is smaller than the complete stored BFP4 tile footprint,
so it is not substituted for the analytic theoretical headline. The reduced
prefill profile records 1018.519 us of device work against the canonical
1281.216-us warmed wall result.

Compared with this pass's baseline profile, trace device time improves from
587.0 to 583.9 us/replay. The canonical synchronized production-default wall
run is the required prefill headline; the paired A/B is
`1.267830 -> 1.266436 ms` and the independent final run is `1.281216 ms`.

Artifacts:

- `final_tracy/{prefill,decode}_perf_report.{txt,csv}` and renderer console
  provenance;
- `final_tracy/reports/2026_07_19_08_14_06/ops_perf_results_2026_07_19_08_14_06.csv.gz`;
- corresponding before artifacts under `baseline/tracy/`.

Duplicated raw Tracy host/device captures were removed after the merged CSVs
were losslessly compressed; the exact capture command reproduces them.

## Runtime fallback, layout boundary, context, and watcher gates

The host source test inspects every runtime prefill/decode method and rejects
`from_torch`, `to_torch`, `torch.`, `super().prefill`, and `super().decode`.
It also asserts the real async all-reduce path, paged SDPA, device embedding,
TP4 mesh axes, and selected explicit SDPA flags. Constructor-time Torch weight
packing is intentionally outside runtime and is not a fallback.

The final inter-layer residual contract is replicated BF16
`[1,B,S,8192]`. Within decode, the residual is L1 width-sharded; QKV and
gate/up are local packed column-parallel projections; output and down are
row-parallel DRAM-sharded matmuls followed by persistent BF16 RS+AG reductions
inside the layer. The final residual add returns the same replicated contract
to the next layer. The two-layer test feeds output directly to the next
invocation without host conversion and proves there is no inter-layer gather,
reshard, or all-reduce. Full-model bringup must not add one.

Final dtypes/layouts and persistent allocations do not change the existing
131072-token context proof in `doc/context_contract.json`. Production-default
logical length 39, a separately forced seven-token internal chunk tail,
paged/nonidentity mappings, positions
39/40/64, and full local cache allocation remain covered. No context reduction
was made.

Watcher command:

```bash
flock /tmp/tt-device.lock env \
  LLAMA_31_70B_REAL_WEIGHT_FILE="$LLAMA_31_70B_REAL_WEIGHT_FILE" \
  RUN_MULTICHIP_DECODER_WATCHER=1 TT_METAL_WATCHER=10 \
  TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=10 \
  TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
  pytest -sv models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_multichip_decoder.py \
  -k watcher_stress \
  --junitxml=models/autoports/meta_llama_llama_3_1_70b_instruct/doc/optimized_multichip_decoder/watcher_final.xml
```

Result: 1 passed, 10 deselected. Watcher attached to Tensix on devices 0--3;
17 eager positions, the page boundary, and 100 advancing trace replays
completed without assertion, timeout, stall, or non-finite result. ETH watcher
instrumentation remains disabled because the instrumented fabric-router binary
cannot fit its 25,600-byte kernel-config buffer, as proven in the completed
stage. `device_health_final.log` retains the post-run `tt-smi -s`: all four
devices and DRAM channels are healthy, and all corrected/uncorrected GDDR error
counters are zero. No reset was needed.

Static gates also pass:

```text
python -m py_compile <multichip_decoder.py> <test_multichip_decoder.py>
git diff --check
```

Known non-gating notices are the newer-than-tested firmware 19.8.0, unknown
B850M-C bus-ID fallback, low `/dev/shm` headroom, and existing nanobind shutdown
leak diagnostics.

## Independent-review remediation

The initial `$stage-review` verdict was `more-work-needed`; its complete record
is `STAGE_REVIEW_INITIAL.md`. The three required findings were closed as
follows:

1. Production-default accounting: the canonical test now times chunk 4096,
   records that default explicitly, and isolates the 32+7 tail as correctness
   only. `candidates/stage_before_after_defaults.{xml,log}` supplies the paired
   starting/final comparison, while `final_default.{xml,log}` is the final
   source-snapshot run.
2. Lower-core MLP geometries: G4/G5 and D4/D5 variants were adapted through
   smaller inner blocks/per-core N until G4B, G5C, D4B, and D5C executed. Every
   result or exact CB blocker is retained under `candidates/geometry_*`.
3. Prefill L1 advice: all four matmul inputs were moved as one whole-prefill
   family. Exact PCC passed and latency regressed, as retained in
   `candidates/prefill_l1_inputs.{xml,log}`.

The secondary health-provenance concern is closed by
`device_health_final.log`. Final wall and device-profiler numbers are clearly
identified as separate runs, and the three cross-device Tracy range-entry gaps
are not treated as steady decode overhead.

The next fresh rereview found a production wiring contradiction and missing
roofline accounting; its complete `more-work-needed` record is
`STAGE_REVIEW_WIRING.md`. Both findings were remediated:

1. The implicit-prefill override was moved from the provenance 2-D constructor
   to production `MultiChipDecoder.__init__`. The host audit checks that exact
   source assignment, and the canonical hardware test asserts/records explicit
   decode program+compute and implicit prefill program. Paired A/B, final
   default, whole-prefill L1 advice, Tracy, watcher, health, and static gates
   were regenerated. The refreshed raw prefill SDPA row records
   `program_config=std::nullopt`.
2. The profiler section now reports wall, device, and theoretical lower-bound
   latency for prefill and traced decode, with the stored-dtype modeled-byte
   formula and the utilization cross-check.

The final fresh rereview is retained in `STAGE_REVIEW_FINAL.md` and returned
`clean-pass` with no required work. It independently reconciled artifact
timestamps, production SDPA wiring, raw program attributes, PCC/performance,
roofline arithmetic, topology/precision family evidence, watcher/health, and
scope boundaries.

## Applicability and no-defer audit

- Dense decoder: one meaningful layer kind; real layer-39 and two direct
  invocations cover it. MoE active-expert requirements are inapplicable.
- Decoder-only stage: LM head, sampling, text generation, full-model, vLLM,
  and qualitative generation checks are inapplicable and were not started.
- Batch/context: batch 1 is the latency target; the inherited public batch and
  full batch-1 131072-token capacity contract remain unchanged. The emitted
  batch-32 provenance is not hard-coded into runtime.
- Async CCLs, fused AG-matmul, fused matmul-RS, placement, residual layout,
  activation sharding, persistent/preallocated buffers, DRAM-sharded decode
  matmuls, projection packing, activation/CCL/cache dtype, weight dtype, and
  per-group compute fidelity all have passing or adapted failure evidence.
- No applicable optimization or advice is deferred. Material candidates that
  initially failed were adapted through legal shape, layout, padding, weight,
  lifetime, or memory changes until they executed, or were superseded by a
  complete executing family.

## Artifact index and stage ownership

- `baseline/reproduction.{xml,log}` and `baseline/tracy/`: current-pass before.
- `candidates/`: all current-pass alternatives and retries.
- `shard_advise/`: exact capture, structured report, IR, compressed decision
  trace, and ABI provenance.
- `final_default.{xml,log}`: canonical final correctness and warmed wall timing.
- `final_tracy/`: advice-enabled final tables, CSVs, and operation provenance.
- `watcher_final.{xml,log}`: final stress/watcher run.
- `device_health_final.log`: post-watcher `tt-smi -s` health evidence.
- `static_gates.log`: compile, diff, whitespace, and artifact-hygiene gates.
- `STAGE_REVIEW_INITIAL.md`, `STAGE_REVIEW_WIRING.md`, and
  `STAGE_REVIEW_FINAL.md`: findings, remediation, and final clean-pass verdict.
- `README.md`: compact handoff and inter-layer contract.

Starting commit is `9280cf5c2b0`; completed multichip implementation commit is
`1087041f65f`. Stage-review verdict and local stage commit SHAs are appended
after the independent review gate. The pre-existing unrelated
`.agents/skills/forge-functional-decoder-from-ir/SKILL.md` modification is not
stage-owned and will not be committed.
