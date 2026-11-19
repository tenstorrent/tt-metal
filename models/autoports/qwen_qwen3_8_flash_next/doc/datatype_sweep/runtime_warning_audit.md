# Runtime warning audit

CPU-only source/log audit at **2026-11-16 05:06:43 UTC**. No device calls, source changes or warning suppression. Baseline [log](logs/inherited_e4_lofi_v3.log) completed with exit0; [combined log](logs/combined8_lofi_v1.log) was still running, ending at natural-text prefill output in the captured prefix. Counts are a snapshot, **not a claim that the combined run completed**. [Exact normalized signatures/counts](runtime_warning_counts.json) preserve bytes/lines read and the partial log boundary.

Normalization removes only timestamp/severity/logger columns (`line.split('|',3)[-1].strip()`); it retains full messages, numbers, layouts and source locations. Both snapshots have **2366 warnings, six identical distinct signatures, zero new or missing signatures**. The baseline includes its second serial repeat; combined has not yet finished its first repeat. Equal counts alone do not establish correctness; source semantics and remaining numerical/capability gates are evaluated below.

| Signature | Completed baseline | Combined prefix | Actual meaning / disposition |
| --- | ---: | ---: | --- |
| Unknown B850M-C motherboard, bus0x1→tray ID | 1 | 1 | Physical discovery metadata fallback; carried platform condition, not precision fallback. |
| Fabric4352-byte packet,2048-byte pages, recommended8192 | 1 | 1 | CCL transport packing opportunity; no data conversion or packet loss indicated. |
| Output memory: computed1 core vs requested8, shard32×32 | 1248 | 1248 | Actual output shard-grid override; corresponds by geometry to GR projections. |
| Output memory: computed7 cores vs requested8, shard32×224 | 468 | 468 | Actual output shard-grid override; corresponds by geometry to GDN packed Z/A/B. |
| Output memory: computed5 cores vs requested8, shard32×64 | 624 | 624 | Actual output shard-grid override; corresponds by geometry to packed shared gate/up. |
| Sparse multicast disabled: num_links0, has_fabric=false | 24 | 24 | Explicit local-only dispatch domain; expected implementation selection, not lost EP communication. |

## Matmul output memory warnings

[Matmul validation source](../../../../../ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp) `validate_matmul_optional_tensors`, lines213–300, first requires matching output layout and buffer type (or specifically supported1D layout equivalence). In the no-output-tensor case it warns when the requested shard specification differs from the computed one. It uses the **computed output MemoryConfig**. Thus one requested field really is overridden: output core-grid geometry. It is not accurate to say the whole requested MemoryConfig was honored.

For `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, the same file around3100–3141 computes `num_cores=ceil(N/per_core_N)*ceil(M/per_core_M)`, constructs that exact active grid and preserves the requested output dtype, memory-layout class and buffer type. [DRAM projection helper](../../tt/dram_projection.py) requests an eight-core L1 output grid using its activation grid, while output width may need fewer cores. For Q1..32:

- GR output width1 pads to32, so one32-column shard/core suffices.
- GDN packed output width1536+12+12 pads to1568; seven224-column shards suffice.
- Shared gate/up total320 columns needs five64-column shards.

These source-derived role mappings are an inference from geometry, not operation IDs in the warning text. The counts also fit the model inventory:1248/96 GR groups,468/36 GDN groups and624/48 shared groups each equal13 calls. The computed shard has sufficient padded output coverage; no output columns are discarded. `_PreparedGroup.__call__` consumes the returned tensor, converts its actual layout to DRAM and slices logical widths; it does not reinterpret it using the stale eight-core request. Therefore this override is compatible with the current consumer and explains the warning without assuming identical logs imply safety.

This warning does **not** replace weight dtype, output dtype or math fidelity. `_PreparedGroup` passes actual typed weight tensors and an explicit compute config. `create_matmul_attributes` around3458–3468 initializes from `parameters.compute_kernel_config`; its default-fidelity calculation is not this memory-config warning's behavior. The combined run's already-recorded `runtime_precision` has actual GR/GDN/shared BFP8 tensors and LoFi compute, plus BFP8/LoFi head; full-model gates remain responsible for numerical acceptance. No blanket claim about every TTNN kernel honoring every possible fidelity is made.

Disposition: **understood output-layout override; no datatype/fidelity blocker found**. Preserve it in the anomaly ledger, with actual returned layout as authoritative. A future consumer that assumes eight output cores would need correction; current interleaved-DRAM conversion avoids that dependency. Do not change runtime geometry midway through the sweep merely to silence logs.

## Physical-discovery fallback

[Physical-system discovery](../../../../../tt_metal/fabric/physical_system_discovery.cpp), unknown-motherboard branch around147–162, warns once per distinct motherboard name per process and returns the PCI bus ID as tray ID. It does not substitute devices, precision or compute kernels. Its practical risk is inaccurate physical placement metadata on unrecognized hosts; topology correctness cannot be inferred from the motherboard label alone.

The configured1×4 mesh, explicit mesh-graph path and completed baseline full-model collectives provide evidence for this tested lane. The combined run must still complete its own trace/accuracy gates. Retain the fallback warning as a platform metadata limitation; no detected dtype issue, and no hardware remapping/reset is justified by this message alone.

## Fabric packet recommendation

[CCL packet check](../../../../../ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp), lines45–83, calculates ideal whole pages per packet from architecture payload limit and a four-scatter-chunk cap. On Blackhole with2048-byte pages it recommends `min(floor(15232/2048),4)*2048=8192` bytes, versus configured4352. It logs each size combination once per process. The warning is a throughput recommendation, not a numerical fallback, rejected packet or unsupported fabric condition.

The2048-byte page is BF16 data transport; changing projection weights to BFP8 does not imply all collective payloads should become BFP8. Combined policy retains BF16 CCL and outputs, so this warning is consistent with its policy. Packet size remains the same across the compared policies; measured performance includes this fixed transport choice. Changing it could change router buffering, runtime memory and all baselines, so it is a separate optimization experiment requiring revalidation, not a datatype-compliance repair.

## Local dispatch and sparse multicast

[Dispatch API](../../../../../ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/dispatch.cpp), lines40–65, defines local-only dispatch when the selected domain is singleton and `dispatch_group_size==1`; it then explicitly selects `num_links=0` and Linear topology. [Model routed-expert call](../../tt/multichip_decoder.py), `MeshMoe.__call__`, uses `dispatch_group_size=1`, `cluster_axis=0` on a1×4 mesh, with a zero local dispatch table. Expert tensors/IDs are already EP-partitioned; each rank executes its own128 experts and cross-rank contributions are handled by the model reduction path.

[Dispatch factory](../../../../../ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/dispatch_program_factory.cpp), lines422–469, enables sparse multicast only for1D fabric, positive links and bounded hops. Here only the positive-link gate is false, intentionally. The “has_fabric=false” text describes this dispatch operation, **not the whole model fabric**. The same warning-level log reports ENABLED as well as DISABLED, so severity alone does not imply an error. The alternative per-expert/local path remains active; this is not dense all-expert debug evaluation and does not replace requested expert dtype/fidelity.

Disposition: **expected local-dispatch path**, justified by the call's singleton domain rather than merely its presence in baseline logs. If dispatch_group_size/axis/mesh changes, this justification must be rechecked; a newly disabled required cross-device route would be a capability defect.

## Anomaly ledger and completion boundary

- **A1, output shard-grid override:** real requested-versus-used layout discrepancy, understood and consumed safely by current DRAM conversion. No precision field discarded; retained in report.
- **A2, unknown motherboard metadata:** carried platform fallback; validated only for the explicit tested lane, not general host placement.
- **A3, packet packing:** carried performance opportunity under fixed transport settings; no correctness evidence of loss.
- **A4, sparse multicast disabled:** expected singleton local-dispatch selection, with model EP reduction still present.
- **A5, combined run incomplete at snapshot:** open execution/accuracy/capability work. No pass inferred from this warning audit. Final reviewer should compare the finished log against the captured six signatures and investigate any new message or changed parameters.

The parent retains all accuracy/capability checks. This audit neither suppresses warnings nor promotes a candidate based on common baseline behavior.
