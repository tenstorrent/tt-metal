### Molmo2-8B Performance Summary

This document summarizes performance changes for Molmo2-8B between branches `ssinghal/Molmo2-8B` (initial) and `ign/fs/molmo2_8B`.

- **Initial**: results from `ssinghal/Molmo2-8B`
- **Final**: results from `ign/fs/molmo2_8B`
- **Optimization %**: computed as `(Initial - Final) / Initial * 100`


### Table of Results

| Test Module | Initial (us) | Final (us) | Optimization % |
| --- | --- | --- | --- |
| models/demos/molmo2/tests/test_vision_backbone.py::test_vision_backbone_encode_only | 44781 | 29326 | 34.5% |
| models/demos/molmo2/tests/test_vision_transformer.py::test_vision_transformer | 55,951 | 29,598 | 47.10014119 |
| models/demos/molmo2/tests/test_vision_transformer.py::test_vision_transformer_feature_layers | 45,078 | 29,628 | 34.2739252|
| models/demos/molmo2/tests/test_image_projector.py | 2964| 2367 | 20.1417004|
| models/demos/molmo2/tests/test_image_pooling.py | 465 | 430 | 7.52688172|
| models/demos/molmo2/tests/test_text_block.py | 2566 | 2508| 2.260327358 |
| models/demos/molmo2/tests/test_text_mlp.py | 1538 | 1511 | 1.755526658 |
| models/demos/molmo2/tests/test_vision_full_pcc.py | 91,521 | 83,123 | 9.176036101 |
| models/demos/molmo2/tests/test_molmo2_model.py | 6331 | 5848 | 7.62912652|

### Consolidated Changes

| Sl No | Module | Changes | Remarks |
| --- | --- | --- | --- |
| 1 | `text_attention.py` | Changed rotary embedding (Llama-style) to half-span `rotary_embedding` to match the reference. | Text block PCC for layer 17 and layer 35 was low; the change fixed it. |
| 2 | `test_attention.py` | Fused QKV in text prefill. | |
| 3 | `test_pcc_all_layers.py`, `test_attention`, `test_block`, `test_model` | Converted weights to bfloat8 while maintaining PCC. | |
| 4 | `functional.py`, `test_molmo2_model.py` | Aligned to cross-attention implementation of TTNN. | `test_molmo2_work` was failing. |
| 5 | `text_model.py`, `molmo2_model.py`, `test_e2e_pcc.py` | Overall matmul improvement with multi-device implementation of `MatmulDeviceOperation` 32 × 4096 × 151936. | Improved execution time from 15,325.80 μs to 12,198.93 μs. |
| 6 | `vision_transformer.py`, `vision_attention.py` | L1 width sharding for matmuls and use of `MatmulMultiCoreReuseMultiCast1DProgramConfig` to maximize worker cores. | Improved vision processing from 397 ms to 345 ms when vision trace enabled. The video demo with trace enabled is failing with L1 CB issue. |
| 7 | `vision_layernorm.py` | Width sharding to block sharding. | 2078 μs to 2052 μs. |
| 8 | Unified trace | Unified trace working for both video and image. | End-to-end TTFT (vision + fusion + prefill): 2190.24 ms to TTFT (vision + prefill): 155.72 ms. |
| 9 | Decode trace | Error due to `rotary_embedding` current position. | Fixed indexing. |
| 10 | `image_projector.py` | When tiling data is high, use explicit multicast config. | Optimized from 4,134.18 μs to 3,411.74 μs. |
