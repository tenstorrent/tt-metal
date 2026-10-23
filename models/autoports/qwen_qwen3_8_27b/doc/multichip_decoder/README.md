# Qwen3.8-27B multichip decoder

Stage04 implements a four-ASIC tensor-parallel decoder in `tt/multichip_decoder.py`, using `OptimizedDecoder` as its single-chip mathematical baseline. The target is this machine's four Blackhole p300c ASICs in a 1×4 mesh, FABRIC_1D_RING, with two Ethernet links per edge. Smaller meshes are intentionally unsupported. All required validation and profiling are complete. Independent `stage_review.md` returns **clean-pass** with no required work.

## Tensor and phase contracts

`mesh_plan.md` records the strategy chosen before implementation and the measured selection. Full attention owns six query/gate heads and one KV head per device. Linear attention owns four key heads and twelve value heads, with local convolution and FP32 recurrence. The MLP owns 4,352 intermediate channels per device. Gate/up uses one packed weight representation for both phases; unused separate copies are not allocated. Projection weights are BFP4/LoFi, activations and collective arithmetic BF16, KV BFP8. This is a dense model, with no MoE experts.

The selected prefill interface is hidden-sharded `[B,S,1280]` per device, with the same output layout. Distributed RMSNorm exchanges sum-of-squares statistics; normalized activations gather at column projections, and row projections reduce-scatter into the residual. Decode consumes and returns replicated `[B,1,5120]`; row projections use reduce-scatter plus all-gather. Both contracts compose directly across decoder layers. State ownership does not change between phases. The caller supplies the phase layout; no host conversion occurs inside forward execution.

Prefill accepts valid non-aligned logical lengths. Internal tile/chunk padding is removed at output boundaries and does not advance recurrent state for padded tokens. Continuation starts retain the optimized baseline's 128-token alignment requirement; decode handles individual positions. Native context remains 262,144. Local paged caches are `[physical_pages,1,32,256]`, with replicated INT32 page-table IDs and per-user current positions. Page-table rows contain a multiple of eight IDs (32-byte sticks); allocation rounds physical capacity to 256 tokens while preserving the requested logical context. Linear state is BF16 `[B,3,2560]` history plus FP32 `[B,12,128,128]` recurrence.

Weights, metadata buffers, state snapshots and update-copy programs are prepared before capture. Trace owners retain every referenced state/input/output allocation until release. Correctness runs enable `TT_METAL_TRACE_ALLOC_TRACKING=1`, include program-cache allocations, use no exemptions, replay advancing inputs and positions, and compare three restored-state replays bitwise. The harness audits runtime torch use and TTNN host conversion. Profiling separately disables tracker and watcher.

## Evidence and performance

Real checkpoint weights and recorded hidden activations are compared directly with single-chip optimized TTNN. `batch_validation.json` covers batches 8, 16 and 32 at 4096+128; `watcher_l0.json` and `watcher_l3.json` cover batch 1. Minimum PCC is gated at 0.995 for prefill, every advancing decode, and reconstructed local state. All 43 final-source cases pass; watcher logs include 53–59 completed scans per case with all features enabled; `validation_summary.json` checks implementation hashes and every finite output/state PCC. Boundary, continuation, stack, native-context and watcher suite manifests record commands and per-case logs.

The headline workload is exactly 4,096 supplied hidden positions plus 128 advancing decode steps, batch 1 and one request. These are layer measurements, not end-to-end token generation. Host timing remains diagnostic and is never used as a device-time roofline denominator. Decode device timing is the final warmed step at context 4,224; prefill is a complete warmed trace replay.

`performance.json` records the measured final default. Efficiency is speedup divided by four participating ASICs.

| Layer kind | Single prefill / TP4 (µs) | Prefill speedup / efficiency | Single decode / TP4 (µs) | Decode speedup / efficiency |
|---|---:|---:|---:|---:|
| linear_attention | 16057.13 / 8326.15 | 1.929× / 48.21% | 815.30 / 508.85 | 1.602× / 40.06% |
| full_attention | 17831.05 / 10628.84 | 1.678× / 41.94% | 704.05 / 424.10 | 1.660× / 41.50% |

`tracy/<label>/whole_layer_timing.json` uses the maximum complete per-device firmware envelope, including attention, norms, CCL, data movement and gaps. It never subtracts unsynchronized clocks from different ASICs. Human-readable `*_perf_report.txt`, compact report CSVs and complete measured-window `*_ops.csv` preserve dtype, fidelity and collective evidence. Full-run raw captures remain local. Rooflines use all four participating ASICs' theoretical peaks; DRAM bytes are explicitly estimated compulsory traffic, not hardware bus counters.

Geometry sweeps preserve BFP4/LoFi across candidates. `prefill_geometry_results.json` and `prefill_selection.json` record larger legal K blocks, smaller grids and the repeated full-layer confirmation; they support retaining the selected prefill geometry. `geometry_results.json` and `extra_geometry_results.json` motivate 40-core/K4 mixed projections, 16-core/K3 output projections and 17-core/K8 down projections. The coherent fused CCL alternatives keep sharded residuals through downstream consumers. Their initial worker-order stall was diagnosed and repaired (`AUTODEBUG.md`, `AUTOFIX.md`). Final combined-fusion medians are about 682/599 µs versus the selected approximately 490/409 µs; coherent RS-only and combined AG+RS families are rejected on measured decode latency. `anomaly_ledger.md` classifies observed discrepancies, including the profiler's UINT8 staging-output label.

`memory_capacity_plan.json` accounts for all 64 layer setups, conservative BF16 embedding/head storage, local KV/recurrent state and estimated activation/trace reserves. The planned 32×32K working set uses about 18.47 GB per device, below the observed 34.18 GB allocator capacity. This is a full-stack feasibility ledger, not a measured full-model allocation or serving claim. Single-layer batch/context execution evidence is recorded separately. No full-model or vLLM implementation is part of this stage.

## Reproduction

Run inside the existing `qwen38-functional-build` container from `/workspace`:

```bash
bash models/autoports/qwen_qwen3_8_27b/tests/multichip_run.sh selected_l0 --layer 0 --seq 4096 --steps 128
bash models/autoports/qwen_qwen3_8_27b/tests/multichip_run.sh selected_l3 --layer 3 --seq 4096 --steps 128 --remap
python -m models.autoports.qwen_qwen3_8_27b.tests.validate_multichip --suite boundary
bash models/autoports/qwen_qwen3_8_27b/tests/profile_multichip.sh final_profile_l0 --layer 0 --multichip-only
bash models/autoports/qwen_qwen3_8_27b/tests/profile_multichip.sh final_profile_l3 --layer 3 --multichip-only
```

The shell wrappers set the writable cache, repository path and trace flags. For direct Python suite invocation, set `USER=mvasiljev`, `TT_METAL_HOME=/workspace`, `TT_METAL_CACHE=/workspace/.cache/tt-metal-cache`, and `PYTHONPATH=/workspace`. Hardware use is serialized; watcher and profiler runs are separate. `work_log.md` records recovery, review and checkpoint details.
