# PB.MM256: the 8192 chunk's matmuls re-swept on the 8x4 (MM_TUNED256 off | on)

Model deepseek_v41_flash_d_p, galaxy_8x4, layers 0-20 (DECODER_SKIP=decode), 8192 chunk (256 rows a chip). Attempt 4 (2026-11-23).

## 1. Inventory: every matmul of a real 8192 chunk (single, H = 139264), largest first

Device us a chunk: the op-mode real-time profile of the off side (generated/ds41_mm256/ops_single.json, slowest chip); config source: tt/mm_configs (`rule` = the generic rule, the GALAXY table has no 8x4 256-row entry). 4x2k (chip-aligned packed) makes the same calls with the same shapes (ops_4x2k.json: within 1 % except wq_b 5412 us).

| op / site | x · w | dtypes (x, w -> out), fidelity, fp32 acc | calls | device us / chunk | today's config (source) |
|---|---|---|---|---|---|
| bringup.moe_block.routed_experts_combine routed experts gate_up + down (bfp4, LoFi) + combine | 8192x5120 bf16 · 1x384 u32 | fork | 20 | 23464.0 | fork moe_block: blocking fixed at load (flat layout / planner); no call-time knob, not swept |
| bringup.moe_block.moe_head router + shared expert (fused head) | 1024x1280 bf16 · 2661120x32 bf16 | fork | 20 | 7130.5 | fork moe_block: blocking fixed at load (flat layout / planner); no call-time knob, not swept |
| linear attn.wq_b | 256x1280 bfp8 · 1280x32768 bfp8 | BFLOAT8_B, BFLOAT8_B -> BFLOAT16, HiFi2, 1 | 20 | 5046.2 | MatmulMultiCoreReuseMultiCast1D (rule) |
| linear attn.wo_b | 256x8192 bfp8 · 8192x5120 bfp8 | BFLOAT8_B, BFLOAT8_B -> BFLOAT16, HiFi2, 1 | 20 | 3629.6 | MatmulMultiCoreReuseMultiCast1D (rule) |
| bringup.grouped_matmul attn.wo_a | 8x256x4096 bfp8 · 8x4096x1024 bfp8 | BFLOAT8_B, BFLOAT8_B -> BFLOAT8_B, HiFi2, 1 | 20 | 3147.5 | grouped_matmul bw4 depth4 rt1 (WO_A_GMM_*) |
| linear q_a | 256x5120 bfp8 · 5120x1792 bfp8 | BFLOAT8_B, BFLOAT8_B -> BFLOAT16, HiFi2, 1 | 20 | 1609.0 | MatmulMultiCoreReuseMultiCast1D (rule) |
| experimental.minimal_matmul engram_kv | 1024x6144 bfp8 · 6144x6400 bfp8 | BFLOAT8_B, BFLOAT8_B -> BFLOAT16, HiFi2, 1 | 2 | 845.0 | minimal: MinimalMatmulConfig (rule) |
| linear compressor | 256x5120 bf16 · 5120x1024 bf16 | BFLOAT16, BFLOAT16 -> FLOAT32, HiFi2, 1 | 3 | 133.6 | MatmulMultiCoreReuseMultiCast (rule) |
| linear indexer.wq_b | 256x1280 bfp8 · 1280x4096 bfp8 | BFLOAT8_B, BFLOAT8_B -> BFLOAT16, HiFi2, 1 | 3 | 117.1 | MatmulMultiCoreReuseMultiCast1D (rule) |
| matmul hc.fn | 256x20480 bf16 · 20480x32 fp32 | BFLOAT16, FLOAT32 -> FLOAT32, HiFi2, 1 | 1 | 93.4 | MatmulMultiCoreReuseMultiCast (rule) |
| linear indexer.weights_proj | 256x5120 bf16 · 5120x32 bf16 | BFLOAT16, BFLOAT16 -> BFLOAT16, HiFi2, 1 | 3 | 51.1 | MatmulMultiCoreReuseMultiCast (rule) |
| linear compressor | 256x5120 bf16 · 5120x512 bf16 | BFLOAT16, BFLOAT16 -> FLOAT32, HiFi2, 1 | 1 | 30.4 | MatmulMultiCoreReuseMultiCast (rule) |
| linear compressor | 512x512 bf16 · 512x128 bf16 | BFLOAT16, BFLOAT16 -> BFLOAT16, HiFi2, 1 | 3 | 19.1 | MatmulMultiCoreReuseMultiCast (rule) |
| linear compressor | 1024x512 bf16 · 512x128 bf16 | BFLOAT16, BFLOAT16 -> BFLOAT16, HiFi2, 1 | 1 | 10.8 | MatmulMultiCoreReuseMultiCast (rule) |

The two moe_block fork rows (~30.6 ms a chunk) hold the routed / shared expert and router matmuls inside fused programs; their K / N blocking comes from the load-time flat routed-expert layout (MoEBlockConfig, FlatLayoutOptions), so a call-time switch cannot change it (it would need a reload and its own A/B). The small compressor call 512x512 · 512x128 (x3) had no faster config.

## 2. Tuned table (MM_TUNED256=on: tt/mm_configs.GALAXY256 / GMM256)

Sweep: tests/perf/test_matmul_sweep.py -k test_sweep_card, 12 single-card processes in parallel (P23_SHARD=i/12, P23_EXACT=1), over the capture of the real chunk; candidates on the 11 x 10 matmul grid (FIX.15) and the model's L1 (1399 KiB free for CBs); fidelity, dtypes, accumulation unchanged. Confirmed on the whole mesh (test_mm256_offon confirm: every chip, 2 repeats each side bit-identical, 20 calls timed). Kept: faster, deterministic, accurate (OWNER rule 3: not bit-identity).

| site | shape | calls / chunk | today config | today us | tuned config | tuned us (sweep / mesh) | gain us / chunk |
|---|---|---|---|---|---|---|---|
| q_a | 256x5120 bfp8 · 5120x1792 bfp8 | 20 | 1d bw8 | 79.4 | 2d 10x8 pm1 pn6 ob1x6 bw32 sb1x3 | 49.0 / 79.7 -> 50.1 | 608 |
| attn.wo_b | 256x8192 bfp8 · 8192x5120 bfp8 | 20 | 1d bw8 | 180.9 | 1d in0 pm8 pn2 bw16 sb4x1 | 156.0 / 181.1 -> 157.2 | 497 |
| attn.wq_b | 256x1280 bfp8 · 1280x32768 bfp8 | 20 | 1d bw8 | 250.1 | minimal M2 K8 N16 sb1x4 | 236.3 / 250.3 -> 238.2 | 275 |
| attn.wo_a.gmm | 8x256x4096 bfp8 · 8x4096x1024 bfp8 | 20 | gmm bw4 depth4 rt1 | 154.2 | gmm bw2 depth2 rt1 | 149.6 / 154.9 -> 151.0 | 93 |
| engram_kv | 1024x6144 bfp8 · 6144x6400 bfp8 | 2 | minimal M4 K4 N8 (rule) | 420.9 | minimal M4 K8 N4 sb4x1 | 393.7 / 421.1 -> 393.8 | 54 |
| indexer.wq_b | 256x1280 bfp8 · 1280x4096 bfp8 | 3 | 1d bw8 | 38.7 | 2d 11x8 pm1 pn12 ob1x12 bw8 sb1x4 | 33.7 / 38.8 -> 33.7 | 15 |
| hc.fn | 256x20480 bf16 · 20480x32 fp32 | 1 | 2d bw16 | 93.2 | 2d 1x4 pm2 pn1 ob2x1 bw32 sb2x1 | 81.9 / 92.8 -> 81.9 | 11 |
| indexer.weights_proj | 256x5120 bf16 · 5120x32 bf16 | 3 | 2d bw16 | 16.8 | 2d 1x8 pm1 pn1 ob1x1 bw32 sb1x1 | 15.6 / 16.8 -> 15.6 | 4 |
| compressor | 256x5120 bf16 · 5120x1024 bf16 | 3 | 2d bw16 | 42.9 | 2d 11x8 pm1 pn3 ob1x3 bw32 sb1x3 | 42.2 / 44.0 -> 43.2 | 2 |
| compressor | 1024x512 bf16 · 512x128 bf16 | 1 | 2d bw16 | 10.6 | 1d in1 pm1 pn4 bw16 sb1x1 | 8.9 / 10.7 -> 9.1 | 2 |
| compressor | 256x5120 bf16 · 5120x512 bf16 | 1 | 2d bw16 | 29.0 | 2d 8x8 pm1 pn2 ob1x2 bw32 sb1x2 | 27.7 / 30.0 -> 29.5 | 1 |

Sum: ~1.56 ms a chunk (sweep times x calls). wq_b: 'minimal auto' (230.8 us) takes the card's 12 x 10 grid (> 110 cores): refused; M2 K8 N16 kept. wq_b and wo_a change the fp32 summation order over K (DS41-390); every other winner keeps it.


## 3. Device ms off vs on, one load (test_mm256_offon, H = 139264, slowest chip, median of 2 profiled runs a side)

| section | single off | single on | 4x2k off | 4x2k on |
|---|---|---|---|---|
| attn_hc | 0.274 | 0.264 | 0.278 | 0.263 |
| attn_norm | 0.021 | 0.021 | 0.021 | 0.021 |
| q_a | 2.435 | 1.833 | 2.055 | 1.541 |
| kv | 0.370 | 0.370 | 0.315 | 0.315 |
| attn_region | - | - | 5.988 | 5.773 |
| attention | 27.108 | 26.378 | 26.060 | 25.601 |
| history_gather | - | - | 1.182 | 1.157 |
| attn_residual | 5.630 | 4.815 | 5.267 | 5.165 |
| experts | 40.474 | 39.724 | 40.974 | 40.840 |
| ffn_residual | 4.156 | 4.279 | 5.413 | 5.415 |
| engram_rows | 0.894 | 0.959 | 0.891 | 0.891 |
| engram_kv | 1.204 | 1.131 | 1.181 | 1.132 |
| engram_add | 0.780 | 0.781 | 0.780 | 0.783 |
| indexer_keys | - | - | 0.851 | 1.080 |
| compressor | 1.955 | 1.912 | 2.841 | 2.919 |
| indexer | 3.786 | 3.766 | 4.190 | 4.146 |
| indexer_topk | 1.385 | 1.385 | 1.474 | 1.474 |
| attn_index | - | - | 0.432 | 0.432 |
| **chunk (device)** | **87.288** | **85.185** | **95.975** | **95.079** |

Runs: single off [87.288, 86.606] / on [85.178, 85.185]; 4x2k off [95.687, 95.975] / on [94.837, 95.079]. Table: the orchestrator measure run (ab/PB.MM256/new); the gate unit's run before it: single 87.446 -> 85.390, 4x2k 95.891 -> 95.286 ms. Single: -2.1 ms (-2.4 %, q_a -0.57, attention (wq_b / wo_a / wo_b) -1.0, the rest in noise); 4x2k: -0.6 to -0.9 ms (q_a -0.51, attention -0.76, offset by +0.4 in history_gather / attn_residual run to run).

## 4. Accuracy and determinism (switch on)

- test_chunk8192_reference (40 layers, gate unit, MM_TUNED256=on): PASS. final hidden 0.1400 / limit 0.2586 (from0), 0.1345 / 0.2422 (last); top5 1.000 / 0.9943 both; every err_layer_L* within its limit (L39 0.868 / 1.492 from0, 0.662 / 1.533 last).
- mm256_repeat_max_abs_diff_single = 0, _4x2k = 0 (on vs on in two slot sets over the same history: every layer output and every served record bitwise), _confirm = 0 (every winner, every chip, 2 repeats).
- Off vs on (diagnostic, rel L2 over 64 sampled rows): single bit-identical on layers 0-16, 1.5e-3 at L19 (the wq_b / wo_a order change, DS41-390); 4x2k bit-identical on all 20 layers in this run.
- no-rebuild (cold, strict): no_rebuild_pass_* = 1 for single8192, one4k, one1600, lanes4x2k, lanes4x1600s, lanes2x3k, lanes6mix; prepares 0, cache entries 0.

## 5. Ladder, switch on (orchestrator measure, ab/PB.MM256/new; tok/s). Off side: the orchestrator's A/B after this step

| case | 51200 | 131072 | 139264 | 262144 | 524288 | 786432 | 1024000 |
|---|---|---|---|---|---|---|---|
| single 8192 steady | 82999 | 81920 | 82915 | 81512 | 76848 | 70988 | 68153 |
| single 8192 resumed | 79292 | 78064 | 79527 | 76147 | 73577 | 66207 | 65402 |
| 4x2k steady | 75018 | 74676 | 73208 | 71049 | 67148 | 64301 | 61272 |
| 4x2k resumed | 70914 | 69628 | 69252 | 67192 | 63366 | 60616 | 58050 |
| 4x1792s steady | 73143 | 71923 | 71923 | 68210 | 64913 | 62014 | 60280 |
| 4x1792s resumed | 66891 | 66773 | 66014 | 63784 | 61100 | 57473 | 55205 |

## 6. Not done / notes

- moe_block fork matmuls (~30.6 ms a chunk, the largest matmul work) not swept: no call-time knob (section 1).
- No re-sweep this attempt: attempt 2's pass-1 sweep (generated/ds41_mm256/pass1/sweep.json) already accepted every deterministic candidate that matched today's output on random inputs, which for bfp8 inputs rejected none (random bfp8 sums are exact, DS41-390); only the fp32-output compressor calls had rejects (0.16 ms a chunk in total there).
- The 5120 path (160 rows a chip): no GALAXY256 / GMM256 key matches; attempt 3's edit of the GALAXY engram entry reverted, so off and 5120 are today's bytes.
