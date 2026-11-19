# Final29 advice inventory

All12 measured phase CSVs across six final-default TP4 workloads on source `2d4ba9dd20c86f6ae9d710c5fe79c259bdd66fe6007b6af4be5437ca01554c0b` are included. All six profile result JSONs pass. CPU-only reconciliation; no CSV modification or new experiment.

**60 occurrences, five distinct exact advice texts, all prefill; zero decode advice.** Every nonempty Advice row has exactly one disposition in `final29_advice.json`. Remaining new experiments are explicitly `not_tried_stop_rule` under `user_wrapup28.json`; measured historical kept/rejected scope remains explained. No generic-text obsolete-operation exemption is used.

Final reports verify GDN attention HiFi4 (joint projection and attention-output MMRS), GDN MLP HiFi2 and full-attention LoFi. Old LoFi GDN accuracy text/N4608 exemption are absent. New56/64-core joint-projection and88-core GDN SwiGLU advice is carried explicitly.

| CSV | Rows | Advice | SHA256 |
|---|---:|---:|---|
| final29_profile_l0_b1_s512/prefill_perf_report.csv | 34 | 4 | `7a89f35890dd7246cd0b1e0a26a6fb93934f2498c4b336e2a114022c02f0e1c7` |
| final29_profile_l0_b1_s512/decode_perf_report.csv | 33 | 0 | `c22bd9b1134433455c4976a6322895958b6cef880941ebbc5c77173e73b1d2eb` |
| final29_profile_l0_b32_s33/prefill_perf_report.csv | 55 | 4 | `e1efbb58ac43f6c97ef0e7332f35b5541832c25cd2ee9b9598f62f7f6a57e74e` |
| final29_profile_l0_b32_s33/decode_perf_report.csv | 33 | 0 | `75f550b3b80459da843b4a16700008376a087bf43c022f8d3fb4217d3f2c3c86` |
| final29_profile_l0_b32_s513/prefill_perf_report.csv | 134 | 7 | `4c542c3a074bb3e7587bd40437ef8e64abbb7f3031450663f6e033095e813281` |
| final29_profile_l0_b32_s513/decode_perf_report.csv | 33 | 0 | `875c5cda591a12e81755dcd21d95edf5d5574cc48bc3311feec58dc76206fecc` |
| final29_profile_l3_b1_s512/prefill_perf_report.csv | 30 | 4 | `8e5a653c52d85941907f86202c978a4ff58af3b613befb7c000759a9d7508b67` |
| final29_profile_l3_b1_s512/decode_perf_report.csv | 33 | 0 | `c0d6b805947f305fd3f3b857c29d9a2ee9218458a9932033e9f6adfdf66725d5` |
| final29_profile_l3_b32_s33/prefill_perf_report.csv | 62 | 5 | `da5bee6258fc16bcff36d5aee238b6e6f955634bb3c0e51a0d2b654d93f2b965` |
| final29_profile_l3_b32_s33/decode_perf_report.csv | 32 | 0 | `087cd82ce8b10f047f2cf348d9bdd7b6adefbeb6965acfd329297f5816d7e469` |
| final29_profile_l3_b32_s513/prefill_perf_report.csv | 555 | 36 | `bce3f7f6fb6b837d2208ecd7c9234a6a6cf1d03635a39c414403ad85f3b24899` |
| final29_profile_l3_b32_s513/decode_perf_report.csv | 32 | 0 | `54c9eb4f8870dd37272838bd6602e36f801c53a95894bf14f6c41e853c16f2b4` |

## Exact occurrences

JSON retains workload,phase,op ID,shape,dtypes/fidelity,device/cores,CSV line/hash and result/hash. Report K/subblock blanks remain null; minimal-SwiGLU configs recovered from pinned source are separately labeled, not fabricated report fields. MMRS has no exposed program_config.

| Profile / phase | Op ID | Operation | Fidelity | Cores | K/H/W reported | Advice group |
|---|---|---|---|---:|---|---|
| final29_profile_l0_b1_s512 / prefill | 1482 | MatmulReduceScatterPrefillDeviceOperation 512 x 1536 x 5120 | HiFi4 BF16 x BFP8 => BF16 | 82 | ?/?/? | 1 |
| final29_profile_l0_b1_s512 / prefill | 1510 | MatmulReduceScatterPrefillDeviceOperation 512 x 4352 x 5120 | HiFi2 BF16 x BFP8 => BF16 | 82 | ?/?/? | 1 |
| final29_profile_l0_b32_s33 / prefill | 1814 | MatmulReduceScatterPrefillDeviceOperation 1056 x 1536 x 5120 | HiFi4 BF16 x BFP8 => BF16 | 90 | ?/?/? | 1 |
| final29_profile_l0_b32_s33 / prefill | 1837 | MinimalMatmulDeviceOperation 1056 x 5120 x 8704 | HiFi2 BF16 x BFP8 => BF16 | 110 | ?/?/? | 1 |
| final29_profile_l0_b32_s33 / prefill | 1842 | MatmulReduceScatterPrefillDeviceOperation 1056 x 4352 x 5120 | HiFi2 BF16 x BFP8 => BF16 | 90 | ?/?/? | 1 |
| final29_profile_l0_b32_s513 / prefill | 2786 | MatmulReduceScatterPrefillDeviceOperation 16384 x 1536 x 5120 | HiFi4 BF16 x BFP8 => BF16 | 92 | ?/?/? | 1 |
| final29_profile_l0_b32_s513 / prefill | 2814 | MatmulReduceScatterPrefillDeviceOperation 16384 x 4352 x 5120 | HiFi2 BF16 x BFP8 => BF16 | 90 | ?/?/? | 1 |
| final29_profile_l0_b32_s513 / prefill | 3058 | MinimalMatmulDeviceOperation 32 x 5120 x 8704 | HiFi2 BF16 x BFP8 => BF16 | 110 | ?/?/? | 1 |
| final29_profile_l3_b1_s512 / prefill | 1898 | MatmulReduceScatterPrefillDeviceOperation 512 x 1536 x 5120 | LoFi BF16 x BFP8 => BF16 | 82 | ?/?/? | 1 |
| final29_profile_l3_b1_s512 / prefill | 1913 | MinimalMatmulDeviceOperation 512 x 5120 x 8704 | LoFi BF16 x BFP8 => BF16 | 88 | ?/?/? | 1 |
| final29_profile_l3_b1_s512 / prefill | 1918 | MatmulReduceScatterPrefillDeviceOperation 512 x 4352 x 5120 | LoFi BF16 x BFP8 => BF16 | 82 | ?/?/? | 1 |
| final29_profile_l3_b32_s33 / prefill | 3370 | MatmulReduceScatterPrefillDeviceOperation 1056 x 1536 x 5120 | LoFi BF16 x BFP8 => BF16 | 90 | ?/?/? | 1 |
| final29_profile_l3_b32_s33 / prefill | 3392 | MinimalMatmulDeviceOperation 1056 x 5120 x 8704 | LoFi BF16 x BFP8 => BF16 | 110 | ?/?/? | 1 |
| final29_profile_l3_b32_s33 / prefill | 3398 | MatmulReduceScatterPrefillDeviceOperation 1056 x 4352 x 5120 | LoFi BF16 x BFP8 => BF16 | 90 | ?/?/? | 1 |
| final29_profile_l3_b32_s513 / prefill | 9462 | MatmulReduceScatterPrefillDeviceOperation 16384 x 1536 x 5120 | LoFi BF16 x BFP8 => BF16 | 92 | ?/?/? | 1 |
| final29_profile_l3_b32_s513 / prefill | 9483 | MinimalMatmulDeviceOperation 16384 x 5120 x 8704 | LoFi BF16 x BFP8 => BF16 | 110 | ?/?/? | 1 |
| final29_profile_l3_b32_s513 / prefill | 9490 | MatmulReduceScatterPrefillDeviceOperation 16384 x 4352 x 5120 | LoFi BF16 x BFP8 => BF16 | 90 | ?/?/? | 1 |
| final29_profile_l3_b32_s513 / prefill | 11238 | MinimalMatmulDeviceOperation 32 x 5120 x 8704 | LoFi BF16 x BFP8 => BF16 | 110 | ?/?/? | 1 |
| final29_profile_l3_b1_s512 / prefill | 1811 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s33 / prefill | 3183 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s33 / prefill | 3191 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9107 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9115 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9123 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9134 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9141 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9149 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9158 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9165 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9174 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9182 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9187 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9195 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9206 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9214 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9221 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9230 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9237 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9245 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9253 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9262 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9270 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9277 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9285 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9294 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9302 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9310 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9318 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9325 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9334 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9342 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9349 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l3_b32_s513 / prefill | 9357 | MatmulDeviceOperation 512 x 5120 x 3584 | LoFi BF16 x BFP8 => BF16 | 88 | 8/2/1 | 2 |
| final29_profile_l0_b32_s33 / prefill | 1662 | MatmulDeviceOperation 1056 x 5120 x 7168 | HiFi4 BF16 x BFP8 => FP32 | 56 | 8/1/4 | 3 |
| final29_profile_l0_b1_s512 / prefill | 1405 | MatmulDeviceOperation 512 x 5120 x 7168 | HiFi4 BF16 x BFP8 => FP32 | 64 | 8/1/4 | 4 |
| final29_profile_l0_b32_s513 / prefill | 2614 | MatmulDeviceOperation 4096 x 5120 x 7168 | HiFi4 BF16 x BFP8 => FP32 | 64 | 8/1/4 | 4 |
| final29_profile_l0_b32_s513 / prefill | 2623 | MatmulDeviceOperation 4096 x 5120 x 7168 | HiFi4 BF16 x BFP8 => FP32 | 64 | 8/1/4 | 4 |
| final29_profile_l0_b32_s513 / prefill | 2635 | MatmulDeviceOperation 4096 x 5120 x 7168 | HiFi4 BF16 x BFP8 => FP32 | 64 | 8/1/4 | 4 |
| final29_profile_l0_b32_s513 / prefill | 2647 | MatmulDeviceOperation 4096 x 5120 x 7168 | HiFi4 BF16 x BFP8 => FP32 | 64 | 8/1/4 | 4 |
| final29_profile_l0_b1_s512 / prefill | 1504 | MinimalMatmulDeviceOperation 512 x 5120 x 8704 | HiFi2 BF16 x BFP8 => BF16 | 88 | ?/?/? | 5 |

## Advice groups

1. **18 occurrences:** If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED) • No program_config specified, try using one to override in0_block_w and out_subblock_h/w
   `not_tried_stop_rule`: Remaining coherent placement/planner and phase-specific fused SwiGLU bulk/tail families; historical measured chunks/L1/minimal experiments remain recorded rather than relabeled untried.

2. **35 occurrences:** If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED) • in0_block_w=8 and output subblock 2x1 look good 🤷 • Use HiFi2 or HiFi4 with BF16 activations for improved accuracy
   `not_tried_stop_rule`: Matched full-attention BF8 higher-fidelity/input-L1/row-cap with consumer and concat cost; no S33-to-S513 extrapolation.

3. **1 occurrences:** Increase grid size (currently using 56 of 110)
   `not_tried_stop_rule`: Matched final-HiFi4 joint projection active-grid/block/row-cap adaptation including combine movement and live L1 capacity.

4. **5 occurrences:** Increase grid size (currently using 64 of 110)
   `not_tried_stop_rule`: Matched final-HiFi4 joint projection active-grid/block/row-cap adaptation including combine movement and live L1 capacity.

5. **1 occurrences:** Increase grid size (currently using 88 of 110)
   `not_tried_stop_rule`: B1/S512 fused minimal SwiGLU larger-grid/block experiment including input/output movement and L1 footprint.

## Limits

Historical candidate timings are not final-default numbers. All28 final performance jobs and both final-default HF contracts are separate evidence, not replaced by advice inventory. Both B1 paired95% nonregression bounds remain inconclusive (+0.037668ms GDN,+0.057086ms attention), despite improved point estimates. No changed gate,new tuning,70% roofline assertion,formal review or completion claim is made. Final profile accounting and aggregate checker own roofline/device/host reconciliation.

Inventory timestamp: 2026-11-13T22:20:42.878261+00:00.
