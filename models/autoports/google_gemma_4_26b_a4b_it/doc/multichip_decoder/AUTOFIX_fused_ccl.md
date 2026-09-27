# AutoFix: fused CCL producer/consumer probes

## Linear MMRS attempt

At repo HEAD `9a529836fc`, ran:

```bash
HF_HUB_OFFLINE=1 timeout 180 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_multichip_fused_ccl --family mm_rs --layer sliding_attention --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/fused_mmrs_sliding.json
```

First setup failed because this TTNN build does not export
`BlackholeComputeKernelConfig`. Changed the probe to
`init_device_compute_kernel_config(mesh.arch(), ...)`, matching the baseline.
This API mismatch does not reject any fused family. Initial log:
`fused_mmrs_sliding.log`. Reset/list/mesh-smoke logs `fused_reset1.log`,
`fused_list1.log`, `fused_smoke1.log` all completed exit0, four ASICs visible.

Retry log: `fused_mmrs_sliding_retry1.log`. Native separate matmul, Linear RS,
distributed norm, and residual completed. Fused operation then stalled.
`fused_mmrs_triage.txt` identifies op7
`MatmulReduceScatterAsyncDeviceOperation`, input [1,1,32,1024] FP32, weight
[1,1,1024,2816] BFP8, FP32 intermediate [1,1,32,2816] and output
[1,1,32,704]. The preceding op6 residual add completed. No completed fused
PCC, trace, or timing measurement exists.

Triage callstacks show ring RS readers on multiple chips waiting for
intermediate-ready semaphore at
`reduce_scatter_minimal_async/device/kernels/ring_reduce_scatter_minimal_async_reader.cpp:416`.
Matmul compute cores are no longer listed as running. Other CCL RISCs wait on
CBs. This localizes the stall inside the fused RS program rather than the
following distributed norm. Triage summary statuses describe diagnostic
execution, not all checks being healthy: the report also flags live fabric
NoC counter mismatches, which are not alone proof of a NoC hardware fault.

Collected triage before terminating only probe PID1483 (`SIGTERM`, exit143).
Then `timeout 180 tt-smi -r`, `timeout 60 tt-smi -ls --local`, and FABRIC_1D
1x4 open/close smoke all returned exit0 (`fused_reset2.log`, `fused_list2.log`,
`fused_smoke2.log`). Four ASICs visible; no locks cleared or second reset needed.
Hardware ownership was explicitly returned to root after recovery.

## Next hypothesis and adaptation

Source evidence:

- `matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_program_factory.cpp:84`
  unconditionally calls `build_ring_reduce_scatter_minimal_async_program_artifacts`.
- Standalone `reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.cpp:39`
  chooses `RingReduceScatterMeshWorkloadFactory` for Ring and
  `LineReduceScatterMeshWorkloadFactory` for Linear.
- The ring helper marks first/last-chip variables unused and generates ring
  schedules; passing Linear does not select the standalone Line implementation.

Hypothesis: the fused wrapper sends Linear peers to a ring schedule, leaving
endpoint intermediate-ready expectations unsatisfied. The callstacks support
but do not by themselves prove this. A legal Ring retry isolates topology from
shape, grid, precision, and next-consumer layout. Probe now accepts
`--topology ring`, selecting FABRIC_1D_RING and Topology.Ring together.
Auto-discovery logs report four participating chips with physical and logical
degree histograms `{2:4}`, consistent with a four-device cycle. Mesh open under
Ring remains a required runtime gate; the histogram alone is not performance
evidence.

Next command adds `--topology ring` and a distinct output/log filename. Keep
FP32, grid8x6, K-block2, N/core11 and the directly consuming norm/residual
unchanged. Independently test AGMM afterward; its runtime is not rejected by
this MMRS-specific result. No hardware retry started while root owns devices.

## Ring validation and geometry adaptation

Ring initialized and completed on all four devices. The same FP32 sliding
MMRS shape/config that stalled under Linear passes under Ring. This supports
the wrapper/topology diagnosis; the model can use Ring without a C++ change.
No additional failure or reset occurred in the eight Ring runs. Each process
closed all devices normally (exit0). Hardware returned to root afterward.

All results use synthetic BFP8 weights, FP32 activations/output, HiFi2 with
FP32 destination accumulation. Numbers are median of20 warmed trace host
replays of the entire producer/consumer component, not device time or whole
layer latency. PCC compares fused and separate native TT implementations with
identical inputs, weights, config and sharded consumer; repeated trace output
is bit-identical. Default shape32 represents physical decode tile rows.

| Family/type | Grid / K block / subblock W | Separate host us | Fused host us | PCC | JSON evidence |
| --- | --- | ---: | ---: | ---: | --- |
| mmrs/sliding | 8x6 / 2 / 1 | 93.6880 | 98.5520 | 1.000000000 | fused_mmrs_ring_sliding.json |
| mmrs/sliding | 11x6 / 16 / 4 | 87.8975 | 92.2605 | 1.000000000 | fused_mmrs_ring_sliding_tuned.json |
| mmrs/full | 8x6 / 2 / 1 | 111.1815 | 115.9300 | 0.999999940 | fused_mmrs_ring_full.json |
| mmrs/full | 11x6 / 16 / 4 | 97.2850 | 100.6070 | 0.999999940 | fused_mmrs_ring_full_tuned.json |
| agmm/sliding | 8x6 / 2 / 1 | 116.7015 | 111.0615 | 1.000000000 | fused_agmm_ring_sliding.json |
| agmm/sliding | 8x6 / 22 / 4 | 103.5070 | 97.7710 | 1.000000000 | fused_agmm_ring_sliding_tuned.json |
| agmm/full | 8x6 / 2 / 1 | 126.8165 | 122.9490 | 0.999999940 | fused_agmm_ring_full.json |
| agmm/full | 8x6 / 22 / 4 | 116.5415 | 104.7540 | 0.999999940 | fused_agmm_ring_full_tuned.json |

The adaptation materially improves both implementations: MMRS grid11 gives
N/core8 instead of11, and K16/subblock4 reduces loop and pack overhead; AGMM
K22 aligns with the22-tile H shard and subblock4 divides both local output
widths. Tested config changes preserve the physical tensor shapes and norm/
residual contract. Tuned fused MMRS remains slower than the equally tuned
separate matmul plus RS for both attention kinds. This rejects the measured
MMRS fusion configurations as a component latency win; it does not reject
sharded residuals themselves or claim no possible faster kernel exists.

Tuned fused AGMM saves5.736us sliding and11.7875us full in this component
comparison. It is a viable candidate for sharded normalized input to QKV,
subject to integration into real-weight layer tests and complete-layer
measurement. These savings cannot be added directly to a layer result or used
as telemetry headline performance. No BF16 fallback was needed.

Reproduce any row using the probe's family `mm_rs` or `ag_mm`, layer
`sliding_attention` or `full_attention`, `--topology ring`, `--grid-x` and
`--block-k` from the table, and `--subblock-w` for tuned rows. Each JSON stores
all arguments and the source SHA256. Logs have the same basename with `.log`.
The original four runs predate only the addition of configurable subblockW;
source SHA256 `24ce6b4d39feb6dedc848173eb91970df7bd1619d478349a91d0c0e183016676`.
Tuned runs use SHA256
`8212857de436d21498b18435fad1e92977ec92026363c4f5b8a7b6eff2c7e73d`.

## Integration candidate (patch only)

`agmm_candidate.patch` is an unapplied, AST-parsed patch against runtime SHA256
`3f3fc2ed3ad8ff21bcc19cb37519438187dbac5fd1e4cfdcfcf6aacd66504a20`.
`git apply --check` passed when generated. Resulting source SHA256 would be
`26b120d80ba9fb52245a1a679f55ed51a97d0155760f4437c22c03deb95cc1c9`.
No runtime or runner files were edited by this subtask.

Adds optional `fused_agmm` plus topology selection. Fusion requires sharded
residuals and Ring; caller must open FABRIC_1D_RING first. All decoder CCL
paths use the selected topology. The wrapper reuses existing BFP8 packed QKV
weights and baseline decode compute precision (HiFi2 sliding, LoFi full),
2D grid8x6/K22/subblock4, FP32 output, and setup-owned persistent gathered
buffer and semaphores. Prefill explicitly gathers normalized input and calls
the existing `_Projection`; decode passes normalized H/4 directly into fused
AGMM. Outputs preserve the layer's sharded H704 residual contract.

Source audit: `DecodeAttention.__call__` dispatches via `is_decode`;
`FusedAttention.decode` checks only logical row count1 before `heads`, which
calls the QKV projection before head splitting. There is no pre-projection
H2816 width check in this path. Functional prefill explicitly sets
`is_decode=False`, including short tails; decode loops batch slots to row1.
The patch's setup buffer uses logical row1 with padded row32 and defensively
slices a returned physical-row output to logical row1. The existing head
splitter handles DRAM-to-L1 migration.

Remaining integration risks requiring real tests:

- Component evidence used logical row32, while model decode uses logical row1;
  persistent buffer metadata and fused output logical shape need execution.
- Component compute used HiFi2/packer accumulation on both kinds; integration
  intentionally preserves baseline LoFi full and packer setting. Revalidate PCC.
- Per-layer persistent buffer/semaphore reuse is serialized on CQ0; batched
  slot-loop and repeated trace tests must verify fresh inputs and no stale data.
- Component savings include synthetic norms and weights; complete-layer timing
  determines whether sharded residual plus AGMM beats the replicated path.

The unapplied integration patch was rebased after the shared-decode precision
option was added. Current base SHA256 is
`5f75aa5f250b0424b93cfdd5a9fa036f9b322fca38bb7eab40b8c2f5c4764dff`;
candidate SHA256 is
`fe57ebea3a0ac6fe098b3c00b9b6b03100caeb4d6c02ffa77621a401bc8e5eb5`.
AST parse and `git apply --check` passed again. Shared-MLP changes are preserved.
`MeshConfig.allreduce` uses `ccl_manager.topology` for both constituent CCLs,
and `allgather` defaults to the manager topology; explicit decoder RS now uses
`self.topology`. No runtime/hardware action was taken during this rebase.

Latest unapplied patch refresh: runtime base
`d3dee9c50f549b6a48e657f0f8b2745a42c436a81ebecb8d8b099e1ae34f454b`,
candidate `8c73f97c110ed5159521e48eacb23502b2c9f9585be734fa2eefe609b4b62615`.
The separate `agmm_runner.patch` adds `--ring` and `--fused-agmm`; fusion
requires both Ring and sharded residual flags. TP1 keeps fabric disabled;
TP4 selects matching fabric/topology. Results record both flags. Runner base
SHA256 `6088e95b3cf4aece2659ef1c816bc7e2c91cee1eecce92fb3b41b5a73e4239f7`;
candidate `2669b63f2febba02a8f7adc05129608182fa7f5675f1d81c3e9b1b44a2c90979`.
Both AST checks and `git apply --check` passed; patches remain unapplied.

Suggested validation order after applying both patches:

1. Run layer0 and5, length65, steps8, `--trace --check-cache --ring
   --sharded-residual --fused-agmm` against TP1. This checks S1 projection
   metadata and real cache/head layout before a long experiment.
2. For each layer0/5, run length4096/steps128 with `--trace --ring
   --sharded-residual`, then repeat with `--fused-agmm`. Keep expert and
   shared-MLP policies identical across the A/B.
3. Compare the complete-layer Ring-sharded paths to the selected replicated
   path using the same workload. The component AGMM result alone does not
   establish a winning residual contract.

The attention dispatch audit still applies: model decode supplies logical
[1,1,1,704] to the wrapper, which gathers to [1,1,1,2816] before multiplying
packed local QKV weights. Only the produced QKV is passed to head splitting;
head ownership and cache layout do not see the fractured hidden width.

## Real-weight layer integration results

Applied the runtime and runner patches, then validated logical S1 on short
length65/8-step traced/cache smokes for both layer kinds. All passed. Extended
to exact4096-input/128-decode, batch1, hybrid experts and optimized shared
weights. These are paired real-weight TT1 versus TT4 layer checks. All runs
use Ring fabric/topology and hidden-sharded H704 residuals. The independent
TP1 baseline runs in the same process with fabric disabled.

| Layer | QKV path | TP4 prefill host us | TP4 decode host us | Minimum output PCC | Minimum cache PCC | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| sliding_attention | separate | 89719.092 | 936.422 | 0.999328527 | 0.999982003 | agmm_layer0_headline_separate.json |
| sliding_attention | fused2D DRAM | 88621.625 | 953.689 | 0.999328527 | 0.999982003 | agmm_layer0_headline_fused.json |
| sliding_attention | fused1D DRAM | 88730.113 | 949.466 | 0.999328527 | 0.999982003 | agmm1d_layer0_headline_fused.json |
| sliding_attention | fused1D L1 | 89654.340 | 943.435 | 0.999328527 | 0.999982003 | agmm1d_l1_layer0_headline_fused.json |
| full_attention | separate | 74611.518 | 937.344 | 0.999337831 | 0.999965247 | agmm_layer5_headline_separate.json |
| full_attention | fused2D DRAM | 74794.946 | 942.187 | 0.999337831 | 0.999965247 | agmm_layer5_headline_fused.json |
| full_attention | fused1D DRAM | 74517.922 | 938.635 | 0.999337831 | 0.999965247 | agmm1d_layer5_headline_fused.json |
| full_attention | fused1D L1 | 74536.096 | 933.927 | 0.999337831 | 0.999965247 | agmm1d_l1_layer5_headline_fused.json |

All results pass PCC>=.995, per-device local KV cache comparison, runtime
fallback guard, refreshed tensor positions/inputs, and identical repeated
trace output for every decode step. Timings are host-wall medians, not device
time; no profiler or Watcher ran during this experiment. Raw per-step timings,
TP1 timings, exact argv and runtime hashes are in each JSON. No measured run
failed after applying the integration patches; all closed normally, so no
recovery was needed. Hardware and edit ownership returned to root afterward.

Adaptation was necessary before assessing the family: the component test used
a 2D matmul for both sides, while the optimized whole layer's separate QKV
uses 1D64-core sliding/48-core full programs. The first integrated fused2D
variant lost. The fused API explicitly dispatches 1D at
`all_gather_matmul_async/device/all_gather_matmul_async_program_factory.cpp:96`,
so the second variant reused the exact `_Projection.program`: sliding grid8x8,
K22,N/core1,subblock1; full grid8x6,K22,N/core2,subblock2. CCL moved to offset
(0,8), outside both compute grids. Semaphores span the actual device grid.
This compiled and ran on target hardware. Finally, matching the existing
projection's L1 output avoids the DRAM-to-L1 copy in head splitting.

Final fused1D+L1 still loses about7.0us on sliding relative to the optimized
separate Ring-sharded path; full saves about3.4us. This is a narrow workload
comparison, not a general rejection of AGMM. The complete sharded layer is
around934–943us here, so the root stage must compare it to its selected
replicated-residual result. Component-only savings cannot justify replacing a
faster complete residual contract.

Current opt-in integration remains in runtime for reproducibility; default
behavior stays separate/Linear unless the caller selects flags. Snapshots:

- `runtime_agmm_candidate.py.txt`: fused2D DRAM, SHA256
  `8c73f97c110ed5159521e48eacb23502b2c9f9585be734fa2eefe609b4b62615`.
- `runtime_agmm1d_candidate.py.txt`: fused1D DRAM, SHA256
  `e1994ff8634fa6e7d3518726bb3b5d53bb55fc7c7d56f45545f712f89821a420`.
- `runtime_agmm1d_l1_candidate.py.txt`: fused1D L1, SHA256
  `d4c8f9d90de881c73899fe9aaf0e423b6ad4329915d3ee695ff165ce477a314c`.

The `.patch` files preserve the initial integration, not the subsequent 1D/L1
adaptations. Snapshots and measured JSON hashes identify those variants exactly.
