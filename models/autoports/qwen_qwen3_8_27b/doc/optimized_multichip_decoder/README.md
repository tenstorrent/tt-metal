# Optimized multichip decoder

Stage05 implements Qwen/Qwen3.8-27B decoder optimization on four Blackhole p300c ASICs (two cards), mesh1x4, TP4, two-link ring fabric. The stage04 checkpoint is `ad9ed5d3c46f649ae4a0fa0804e899e7999d8180`. This stage contains no full-model generation or vLLM implementation. Final v5 acceptance is complete and independent review returned `clean-pass`; complete v4 acceptance and four profiles are preserved in integration_v8_rs_rope.

## Selected implementation

Weights and attention/recurrent state remain partitioned across all four devices. B1–7 carry replicated L1 residuals with pooled BFP8 asynchronous all-reduce; B8–32 carry1280 hidden channels per device and use native fused matmul/reduce-scatter for both row projections. The next layer consumes the same residual directly. No inter-layer gather, reshard or all-reduce is inserted by the stack. The complete construction, workspace ownership, preparation and retirement contract is in [residual_contract.md](residual_contract.md).

Decode projection weights are BFP4 with LoFi math. The regular DRAM-sharded matmul path uses8 input-storage cores for mix/out/down with K blocks20/6/17; packed gate/up uses80 storage cores with K2. Reader counts are1/2/1/2 respectively. Input-storage cores are distinct from the eight DRAM banks and their reader workers. Linear mix has physical width4608 and full-attention mix3584; owned padding is removed before model computation. Larger legal K blocks and adapted working shards/readers were measured and rejected by complete-layer latency or documented capacity limits. See [geometry_contract.md](geometry_contract.md), `matmul_search.csv`, and reader microbenchmarks.

Prefill uses4096-token internal chunks. Linear-attention mix uses K16/11x10; full-attention mix retains K4/11x8 with a4x1 subblock at4096 tokens. Separate gate/up use11x8,K20,H8,sub8x1 with one shared BF8 DRAM gather and fused gate SiLU. Packed gate/up remains for decode and one-row prefill. Output-block height and subblock height adapt to divisors of the runtime row geometry for arbitrary lengths. Full single-user4096-row out/down projections use native fused matmul/RS (outK48/downK17,H4,sub2x3,11x8 grid,N15) with BFP8 output and preallocated DRAM buffers; other shapes retain separate TP4 matmul/RS. Standalone prefill collectives use8 workers/link, BFP8 hidden-AG/RS payload, and BFP8 mix/gate/up inputs. Full-attention mix AG writes directly to L1 for at most4096 padded working rows; larger row counts use DRAM. Norm statistics/residuals stay BF16. Per-role subblock selection remains valid for shorter/nonaligned chunks. Public lengths need not be aligned. Linear B8–32 decode packs four temporal rows per user into one convolution launch, with internal padding and exact per-user history extraction. `packed_conv_exact.json` checks all four devices and advancing histories against the former per-user TTNN loop.

Native batched decodeRS uses10x1/N16/sub1x4, outK6 for linear attention andK24 for full attention, downK17. Larger role-isolated K24/48/down34 comparisons and K68/136 capacity bounds are in [decode_rs_geometry.md](decode_rs_geometry.md). Full-attention RoPE shares transient packed cos/sin tables acrossQ/K: decodeB>=4 uses logical users as sequence rows, prefillB>=8 folds tile-aligned internal user/sequence rows. The original loop remains for other shapes. No persistent position cache is introduced; padding and final DRAM boundaries remain internal. [rope_optimization.md](rope_optimization.md) records exact proofs, source precision-assignment findings and measured comparisons.

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

`performance.json`, `performance.csv` and `tracy/final_profile_l*/whole_layer_timing.json` hold final B1 device timing. The denominator is the complete per-device layer firmware envelope, including attention, normalization, movement, CCL and gaps; the reported layer duration is the maximum participating-device envelope. Roofline work uses useful dense FLOPs and modeled compulsory DRAM traffic with physical BFP4 padding/BFP8 headers. It is not a bus-counter measurement. Physical peak includes all four participating ASICs,120 cores/ASIC at1.35GHz and512GB/s/ASIC. The [official p300 specifications](https://docs.tenstorrent.com/aibs/blackhole/p300.html) list two ASICs and1024GB/s per card; this run uses two cards. Do not substitute host timing into device fields.

`candidate_summary.csv`, `topology_audit.md` and `work_log.md` retain coherent family comparisons, failed/repaired attempts, command provenance and rejection reasons. `runtime_contract.json` and `batched_runtime_contract.json` verify actual runtime dtypes, fidelity, projection geometry, fused CCLs, convolution/RoPE counts and device gaps. Advice-enabled human-readable tables and CSVs are under each profile directory.

## Final-default device performance

Exact4096input/128decode steps, B1, concurrency1, TP4. Before and after use the same whole-layer device-envelope basis. These are device microseconds, not host-wall phase time.

|Layer kind|Before prefill us|After prefill us|Before decode us|After decode us|Prefill useful FLOPs/peak|Decode estimated DRAM/peak|
|---|---:|---:|---:|---:|---:|---:|
|linear_attention|8237.61|5667.22|509.79|460.83|21.03%|24.12%|
|full_attention|10626.33|5919.61|421.19|368.27|20.73%|28.99%|

Rooflines use the complete layer envelope, including gaps, and all four ASICs' theoretical peak. Work/traffic assumptions, physical padding, precision/fidelity, exact source hashes and evidence paths are in performance.json. Final default host prefill/decode medians are5870.45/445.78us (linear) and6127.69/359.46us (full); profiler and ordinary timing regimes are kept separate.

## Final-default correctness

Same4096+128,B1 workload and accepted single-chip TTNN reference; gatePCC>=0.995 for outputs and states.

| Layer kind | Before prefill PCC | After prefill PCC | Before min decode PCC | After min decode PCC |
|---|---:|---:|---:|---:|
| linear_attention | 0.99996187 | 0.99994851 | 0.99996624 | 0.99997386 |
| full_attention | 0.99997559 | 0.99996109 | 0.99999184 | 0.99982168 |

Evidence: `before_l0.json`, `before_l3.json`, and `final_accepted_l0.json`, `final_accepted_l3.json`. Reduced communication/compute precision causes small measured PCC changes; all remain above the inherited gate. Broader validation is recorded separately in `validation_summary.json`.

## Capacity and limitations

Native262144 context remains supported; `context_contract.json` records the contract and probes. The shared owner preallocates native RS buffers for batches8–32 before capture:16,384,000 DRAM bytes/device and102,400 L1 bytes/bank, plus1,392,640 bytes/device of shared AR scratch. The shared prefillRS pool adds55,705,600 DRAM bytes/device once across all layers; its two roles and fixed4096-row signature are allocated before capture. Layers must explicitly share that owner. Separate prefill gate/up adds25,067,520 layer-owned bytes/device/layer (1,604,321,280 across64 layers); this is not charged as a shared buffer. The updated B32×32768 planning footprint is30,879,801,344bytes/device versus34,178,731,008bytes physical capacity, including12GiB activation/CCL and1GiB trace reserves. `memory_capacity_plan.json` accounts for complete-model weights/cache/reserves, but is an estimate, not a full-stack allocation claim.

Native DRAM-sharded fused all-gather/matmul has source-backed descriptor/K-order defects, preserved triage and adapted controls. Valid interleaved-weight fused AG was measured in the compatible B32 native-RS family and did not improve the whole layer. Native fused RS was repaired by matching output block width to the native reader per-coreN; the initial20-wide repair generalizes to selected16-wide decode and15-wide prefill. See `AUTODEBUG_fused.md` and `AUTOFIX_fused.md`. Prefill fusion was separately repaired and compared with legal full4096 and internally chunked configurations, BF8 inputs, wider11x8 grids, native RS and coherent L1 residual families; adapted AG fusion was rejected, while matched-precision repeated comparisons selected BFP8-output native RS for4096-row single-user prefill. `prefill_ccl_findings.md`, `AUTOTRIAGE_prefill.md`, `AUTOFIX_prefill.md` and `prefill_ccl_candidates.csv` retain the evidence.

The mixed-stack trace runner needed warmup with a retained output before its second capture. All six stack cases pass that repair without relaxing allocation tracking; the exact missing C++ program hash was not observed. `AUTODEBUG_trace.md` distinguishes the source-supported L1-selector hypothesis from measured facts. Mixed-batch trace checks restore state on device and prove ownership/replay behavior; advancing-decode PCC and page-remap checks supply separate semantic coverage.

Independent [stage review](stage_review.md) returned `clean-pass` with no required work. Local checkpoint SHAs are recorded in work_log.md by the receipt commit; the compact telemetry packet is stored in the experiment telemetry directory. Large raw captures remain local and compressed; compact tables/provenance are the review artifacts.

Runtime-clean means no host/single-chip model fallback and passing correctness/trace/watcher checks, not literally warning-free logs. Rank-normalizing `Tensor::view` drops stale ND provenance while preserving its existing equivalent2D shard distribution (`ttnn/core/tensor/tensor_ops.cpp:390–420`). The earlier fused matmul helper derived `allowed_worker_cores` from its selected8x1 grid (`ccl_fusion_mcast_2d.cpp:3483–3497`); v5 supplies the selected10x1/11x8 core sets explicitly. Both warnings are controlled by source inspection and passing measured paths. Unknown motherboard tray-ID and baseline subset-MMIO warnings concern discovery/single-chip reference setup, not the TP4 implementation.

Some accepted setup logs report AICLK settling at1331/1337/1343MHz versus requested1350MHz. UMD explicitly accepts values within5% after polling (blackhole_device_firmware.cpp:235–254); these runs complete with passing PCC. This is a firmware-accepted setup clock warning, not an exact-frequency measurement. Roofline peaks remain labeled nominal/theoretical1.35GHz; profile cycle conversion uses each capture's reported clock header.

Fused matmul/RS profiler rows contain two compute kernels. The scalar `MATH FIDELITY` column reports the last kernel (HiFi4 reduction), while captured matmul attributes and the exact compiled `chlkc_descriptors.h` identify LoFi matmuls. `kernel_fidelity_evidence/` and the runtime audits preserve both values and hash-matched descriptors. This is a profiler attribution limitation; the underlying matmul policy reaches the measured kernel.

## Final-default larger-batch regression evidence

These are warmed host trace timings at4096+128, not device-time roofline values. All eight output/state PCC gates pass. Larger-batch prefill is validated, with no claim of globally optimal per-batch geometry.

|Batch|Layer kind|Prefill host us|Median decode host us|
|---:|---|---:|---:|
|4|linear_attention|24384.37|730.83|
|4|full_attention|25733.00|577.19|
|8|linear_attention|52671.11|777.06|
|8|full_attention|49333.65|631.97|
|16|linear_attention|109184.18|1251.27|
|16|full_attention|97604.46|751.57|
|32|linear_attention|218596.40|2000.69|
|32|full_attention|198021.67|989.93|

Evidence: batch{4,8,16,32}_l{0,3}.json. Final B32 profiles separately verify nativeRS, packed linear convolution and two full-attention RoPE operations per phase/ASIC.

Medians do not describe every host-wall step. Historical v4 batch32_l3 had an isolated3164.21us host tail versus1475.43us median; its origin was not measured. That run is preserved under integration_v8_rs_rope. Device profiles use separate whole-layer execution envelopes and are not substituted for host timing.

## Final B32 device profiles

|Layer kind|Prefill device us|Decode device us|
|---|---:|---:|
|linear_attention|211522.11|2035.43|
|full_attention|187301.46|1010.19|

These are whole-layer all-four-ASIC envelope measurements from tracy/final_b32_profile_l{0,3}/whole_layer_timing.json. Runtime audits confirm22-core nativeRS (10MM+12CCL), one linear decode convolution, and two full-attention RoPE operations per phase/ASIC. Compared with the archived v4 profiles, linear timings are effectively flat. Full decode drops1686.13→1010.19us, while full prefill186950.19→187301.46us is also effectively flat (+0.19%); no stable device-prefill gain is claimed for RoPE packing. Its modest selection gain comes from paired ordinary host measurements, and those regimes remain separate. Ordinary fullB32 decode1475.43→989.93us is a separate measurement from the profiler envelope.

Final acceptance:67 cases,4 all-device profiles,18convolution+54decodeRoPE+14prefillRoPE exact checks,2524nonblocking replays and4watcher-clean runs. The supplemental B8/S4097 case exercises packed prefill across the4096-token boundary and its one-token tail before8 cache-consuming decode steps. All output/state PCC values exceed.995; the minimum across67cases is.9993398186. Validation details/source hashes are in validation_summary.json. Raw captures/logs remain local with SHA256/size receipts in local_artifact_manifest.json; compact reports/tables/provenance are commit-owned.

Telemetry packet: `/home/mvasiljev/agentic-research-data/experiment-26/bringup-logs/qwen38-27b/20261023-astra-medium-exp26-clean/telemetry/packets/a07f750e-34af-4431-bb22-90aaafd34e60.json`, generated from the identically named template. It contains only compact target-workload device/roofline/PCC evidence, with dataset sample population left unknown.
