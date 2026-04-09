# GPT-OSS Prefill MoE DeepSeek Integration — FIXED

## Status: WORKING (2026-04-09)
36L batch=128 demo PASSED with coherent text for all 128 users.
TTFT: 73.78ms. No fallback errors.

## Root Causes Fixed

### 1. Wrong SwiGLU activation function
TtRoutedExpert used standard SwiGLU: silu(gate)*up
GPT-OSS requires: (up+1)*gate*sigmoid(gate*1.702) with clamping at [-7,7]
Fix: Import and use _apply_swiglu from decode.py

### 2. Missing expert biases
TtRoutedExpert had no bias support. GPT-OSS has gate_up_proj_bias and down_proj_bias.
Fix: Use ThroughputExpertWeights which include biases (w1_bias, w3_bias, w2_bias)

### 3. Expert-to-device mapping mismatch
ThroughputExpertWeights use ShardTensorToMesh (LINEAR ordering: device 0=experts 0-3, device 1=experts 4-7...)
DeepSeek dispatch uses GROUP-BASED ordering (column 0=experts 0-15 across 4 rows)
Fix: Permute state_dict expert dimension before loading, so ShardTensorToMesh gives GROUP-BASED order

### 4. NaN in dispatch buffer padding
Unfilled expert buffer slots contain stale DRAM data (NaN/Inf). Fused matmul produces garbage for padding rows.
Fix: ttnn.where(ttnn.isfinite(buf), buf, 0.0) before expert compute

## Key Files Modified
- prefill_deepseek.py: DeepSeek dispatch/combine + GPT-OSS expert compute via ThroughputExpertWeights
- mlp.py: Permutes state_dict for GROUP-BASED ordering, loads separate permuted weights
- __init__.py: Passes weights and program_config to forward_prefill_deepseek
- text_demo.py: enable_prefill_trace temporarily False (restore after trace fix)

## Known Issues
- Double weight loading (standard + permuted) doubles init time (~25min for 36L first run, cached after)
- Prefill trace disabled (enable_prefill_trace=False) — needs trace-compatible cleanup
- TTFT 73ms (vs fallback 66ms) — could be improved with trace
