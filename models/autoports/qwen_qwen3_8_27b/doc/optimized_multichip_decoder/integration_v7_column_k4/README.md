# Optimized multichip decoder

Stage05 implements Qwen/Qwen3.8-27B decoder optimization on four Blackhole p300c ASICs (two cards), mesh1x4, TP4, two-link ring fabric. The stage04 checkpoint is `ad9ed5d3c46f649ae4a0fa0804e899e7999d8180`. This stage contains no full-model generation or vLLM implementation. Final acceptance and independent review are in progress.

## Selected implementation

Weights and attention/recurrent state remain partitioned across all four devices. B1–7 carry replicated L1 residuals with pooled BFP8 asynchronous all-reduce; B8–32 carry1280 hidden channels per device and use native fused matmul/reduce-scatter for both row projections. The next layer consumes the same residual directly. No inter-layer gather, reshard or all-reduce is inserted by the stack. The complete construction, workspace ownership, preparation and retirement contract is in [residual_contract.md](residual_contract.md).

Decode projection weights are BFP4 with LoFi math. Mix/out/down use8 input-storage cores with K blocks20/6/17; packed gate/up uses80 storage cores with K2. Reader counts are1/2/1/2 respectively. Input-storage cores are distinct from the eight DRAM banks and their reader workers. Linear mix has physical width4608 and full-attention mix3584; owned padding is removed before model computation. Larger legal K blocks and adapted working shards/readers were measured and rejected by complete-layer latency or documented capacity limits. See [geometry_contract.md](geometry_contract.md), `matmul_search.csv`, and reader microbenchmarks.

Prefill uses4096-token internal chunks. Column projections use K4 and11x10 grids except full-attention mix, which uses11x8 and a4x1 output subblock at4096 tokens. Full single-user4096-row out/down projections use native fused matmul/RS (outK48/downK17,H4,sub2x4,8x8 grid) with BFP8 output and preallocated DRAM buffers; other shapes retain separate TP4 matmul/RS. Standalone prefill collectives use8 workers/link, BFP8 hidden-AG/RS payload, and BFP8 mix/gateup inputs. Full-attention mix AG writes directly to L1 for at most4096 padded working rows; larger row counts use DRAM. Norm statistics/residuals stay BF16. Per-role subblock selection remains valid for shorter/nonaligned chunks. Public lengths need not be aligned. Linear B8–32 decode packs four temporal rows per user into one convolution launch, with internal padding and exact per-user history extraction. `packed_conv_exact.json` checks all four devices and advancing histories against the former per-user TTNN loop.

KV remains BFP8 and recurrence FP32. Full-attention decode SDPA uses LoFi/FP16 and chunk256; its chunk guard respects the physical page-table capacity. Norms/residuals remain BF16. This is a dense model; routed MoE execution does not apply.

## Evidence and reproduction

The headline workload is4096 input tokens,128 advancing decode steps, batch1, one request, using recorded real model activations and checkpoint weights. PCC compares the accepted single-chip TTNN reference. It is layer accuracy, not end-to-end generated text quality.

Run commands from the checkout; the container has the built TTNN runtime:

```bash
docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh python -m models.autoports.qwen_qwen3_8_27b.tests.run_multichip_decoder --layer 0 --seq 4096 --steps 128 --prefill-repeats 15 --output <fresh.json>
docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh python -m models.autoports.qwen_qwen3_8_27b.tests.validate_optimized_multichip --suite <batch|boundary|continuation|context|stack|watcher>
docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh bash models/autoports/qwen_qwen3_8_27b/tests/profile_optimized_multichip.sh <fresh-label> --layer 0 --multichip-only
```

Use layer3 for full attention. Profile B32 with `--batch 32`. Hardware commands must remain serialized. Watcher and profiler are separate runs. Timing profiles disable allocation tracking; acceptance enables it including program-cache allocations. Do not overwrite evidence files; preserve previous runs in an archive before rerunning suites.

`performance.json` and `tracy/final_profile_l*/whole_layer_timing.json` will hold final device timing. The denominator is the complete per-device layer firmware envelope, including attention, normalization, movement, CCL and gaps; the reported layer duration is the maximum participating-device envelope. Roofline work uses useful dense FLOPs and modeled compulsory DRAM traffic with physical BFP4 padding/BFP8 headers. It is not a bus-counter measurement. Physical peak includes all four participating ASICs,120 cores/ASIC at1.35GHz and512GB/s/ASIC. The [official p300 specifications](https://docs.tenstorrent.com/aibs/blackhole/p300.html) list two ASICs and1024GB/s per card; this run uses two cards. Do not substitute host timing into device fields.

`candidate_summary.csv`, `topology_audit.md` and `work_log.md` retain coherent family comparisons, failed/repaired attempts, command provenance and rejection reasons. `runtime_contract.json` and `batched_runtime_contract.json` will verify actual runtime dtypes, fidelity, projection geometry, fused CCLs, convolution count and device gaps. Advice-enabled human-readable tables and CSVs are under each profile directory.

## Final-default correctness

Same4096+128,B1 workload and accepted single-chip TTNN reference; gatePCC>=0.995 for outputs and states.

| Layer kind | Before prefill PCC | After prefill PCC | Before min decode PCC | After min decode PCC |
|---|---:|---:|---:|---:|
| linear_attention | 0.99996187 | 0.99994652 | 0.99996624 | 0.99997376 |
| full_attention | 0.99997559 | 0.99996260 | 0.99999184 | 0.99982168 |

Evidence: `before_l0.json`, `before_l3.json`, and `final_accepted_l0.json`, `final_accepted_l3.json`. Reduced communication/compute precision causes small measured PCC changes; all remain above the inherited gate. Broader validation is recorded separately in `validation_summary.json`.

## Capacity and limitations

Native262144 context remains supported; `context_contract.json` records the contract and probes. The shared owner preallocates native RS buffers for batches8–32 before capture:16,384,000 DRAM bytes/device and102,400 L1 bytes/bank, plus1,392,640 bytes/device of shared AR scratch. The shared prefillRS pool adds55,705,600 DRAM bytes/device once across all layers; its two roles and fixed4096-row signature are allocated before capture. Layers must explicitly share that owner. `memory_capacity_plan.json` accounts for complete-model weights/cache/reserves, but is an estimate, not a full-stack allocation claim.

Native DRAM-sharded fused all-gather/matmul has source-backed descriptor/K-order defects, preserved triage and adapted controls. Valid interleaved-weight fused AG was measured in the compatible B32 native-RS family and did not improve the whole layer. Native fused RS was repaired by matching output block width20 to its bank descriptor; the selected path uses that repair. See `AUTODEBUG_fused.md` and `AUTOFIX_fused.md`. Prefill fusion was separately repaired and compared with legal full4096 and internally chunked configurations, BF8 inputs, wider11x8 grids, native RS and coherent L1 residual families; adapted AG fusion was rejected, while matched-precision repeated comparisons selected BFP8-output native RS for4096-row single-user prefill. `prefill_ccl_findings.md`, `AUTOTRIAGE_prefill.md`, `AUTOFIX_prefill.md` and `prefill_ccl_candidates.csv` retain the evidence.

The mixed-stack trace runner needed warmup with a retained output before its second capture. All six stack cases pass that repair without relaxing allocation tracking; the exact missing C++ program hash was not observed. `AUTODEBUG_trace.md` distinguishes the source-supported L1-selector hypothesis from measured facts. Mixed-batch trace checks restore state on device and prove ownership/replay behavior; advancing-decode PCC and page-remap checks supply separate semantic coverage.

Final review, local commit SHAs and the compact telemetry receipt will be recorded here after all gates pass. Large raw captures remain local and compressed; compact tables/provenance are the review artifacts.

Runtime-clean means no host/single-chip model fallback and passing correctness/trace/watcher checks, not literally warning-free logs. Rank-normalizing `Tensor::view` drops stale ND provenance while preserving its existing equivalent2D shard distribution (`ttnn/core/tensor/tensor_ops.cpp:390–420`). The fused matmul helper derives `allowed_worker_cores` from the exact selected8x1 grid (`ccl_fusion_mcast_2d.cpp:3483–3497`); it neither expands the grid nor selects another kernel family. Both warnings are controlled by source inspection and passing measured paths. Unknown motherboard tray-ID and baseline subset-MMIO warnings concern discovery/single-chip reference setup, not the TP4 implementation.

Some accepted setup logs report AICLK settling at1343MHz versus requested1350MHz. UMD explicitly accepts values within5% after polling (blackhole_device_firmware.cpp:235–254); these runs complete with passing PCC. This is a firmware-accepted setup clock warning, not an exact-frequency measurement. Roofline peaks remain labeled nominal/theoretical1.35GHz; profile cycle conversion uses each capture's reported clock header.

Fused matmul/RS profiler rows contain two compute kernels. The scalar `MATH FIDELITY` column reports the last kernel (HiFi4 reduction), while captured matmul attributes and the exact compiled `chlkc_descriptors.h` identify LoFi matmuls. `kernel_fidelity_evidence/` and the runtime audits preserve both values and hash-matched descriptors. This is a profiler attribution limitation; the underlying matmul policy reaches the measured kernel.
