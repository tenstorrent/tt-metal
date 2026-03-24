# Qwen3.5-9B Decode Throughput Optimization — Design Spec

**Date:** 2026-03-23
**Device:** Blackhole P150 (single device)
**Model:** Qwen3.5-9B hybrid (24 DeltaNet + 8 Gated Full Attention layers)
**Current Performance:** 8.3 tok/s (~120ms/token), batch=1, max_seq_len=2048
**Goal:** Maximize decode throughput (tok/s), no accuracy regression (PCC > 0.999)
**Constraints:** Experimental ops modifiable with caution (validated with PCC tests)

---

## Architecture Context

Qwen3.5-9B decode processes 32 layers sequentially per token:
- **24 Gated DeltaNet layers** — conv1d + recurrent state update + projections (O(1) memory per token)
- **8 Gated Full Attention layers** — SDPA + KV cache append + projections
- **32 SwiGLU MLPs** — 3 matmuls each (gate_proj + up_proj + down_proj)
- **64 RMSNorms** — 2 per layer (pre-attention/deltanet, pre-MLP)

At ~120ms for 32 layers, average is ~3.75ms/layer. The actual distribution across layer types and op categories is unknown — profiling is required.

---

## Phase 1: Profiling Infrastructure

### Goal
Measure per-layer and per-op timing to identify the actual bottleneck breakdown.

### Design
- Insert `ttnn.synchronize_device()` + `time.perf_counter()` around each major section of a decode step
- Measure per layer: norm time, attention/deltanet time, MLP time
- Also measure: embedding lookup, final norm, LM head
- Run 20 decode steps, discard first 2 (warmup), average the rest
- Output a table: layer index, layer type, attention/deltanet ms, MLP ms, norm ms, total ms

### Test
`test_decode_profiling` — loads model, runs decode loop, prints per-layer timing breakdown. Asserts total decode time is within expected range (80-200ms).

### Success Criteria
- Clear identification of which layer type (DeltaNet vs Attention) and which op category (projections, element-wise, SDPA, conv) dominates decode time
- Timing data guides Phase 2 optimization priorities

---

## Phase 2: Targeted Op Optimization

### Goal
Reduce time in the hottest ops identified by profiling, without changing correctness.

### Optimization Targets (ordered by expected impact)

#### 2a. DeltaNet Recurrent Path (24 layers)
- `recurrent_gated_delta_rule_decode_ttnn()` in `ttnn_delta_rule_ops.py` has many small element-wise ops — each incurs dispatch overhead
- Reduce intermediate tensor allocations via algebraic simplification (e.g., combine sequential multiply/add into fewer ops). Note: true kernel fusion would require custom TTNN kernels, which is out of scope — focus on reducing op count through restructuring.
- Ensure all T=1 tensors are L1-resident; add explicit `memory_config` to ops that currently pass `None`
- Ensure `try/except` norm paths resolve to the same code path during warmup and trace capture
- **Validation:** PCC > 0.999 per-layer against torch reference

#### 2b. MLP (32 layers × 3 matmuls)
- Ensure all linear ops use `memory_config=ttnn.L1_MEMORY_CONFIG`
- Verify fused SiLU activation is active (not a separate op)
- Optimal `WormholeComputeKernelConfig` on all projections
- **Validation:** PCC > 0.999 for MLP output

#### 2c. Gated Attention (8 layers)
- **KV cache pre-allocation (required, not optional):** The current wrapper uses legacy `ttnn.concat` path for KV cache, but the experimental op already supports pre-allocated cache with `kv_cache_key`/`kv_cache_value`/`cache_pos` parameters. Switch the wrapper to use pre-allocated cache — this eliminates per-step concat overhead and is a prerequisite for trace capture (Phase 3).
- **Fused Q+K+V projection:** Currently 3 separate `ttnn.linear` calls for Q (2x wide), K, V. Fuse into a single matmul + split, matching the DeltaNet pattern.
- **RoPE pre-computation:** Currently computed on host (CPU) every decode step via `torch.full()` + CPU computation + device transfer. Pre-compute full RoPE table on device and index into it — this is both a perf win and a trace capture prerequisite.
- SDPA program config tuning (chunk sizes, grid sizes for Blackhole)
- Add explicit `memory_config` to all `ttnn.linear` calls (currently passing `None`)
- **Validation:** PCC > 0.999 for attention output

#### 2d. RMSNorm (64 norms + 16 Q/K norms in attention)
- Ensure `ttnn.rms_norm` is used everywhere (no manual 6-op fallbacks)
- **Specific target:** `rms_norm_zero_centered_ttnn` in `ttnn_gated_attention.py` always uses the manual 6-op fallback, never attempts fused `ttnn.rms_norm`. Called twice per attention layer (Q norm + K norm) = 16 invocations with 6 ops each.
- L1 memory config for norm outputs
- Check for native `ttnn.rotary_embedding` op on Blackhole (could replace manual `rotate_half_ttnn` which creates intermediate tensors)
- **Validation:** PCC > 0.999 for norm output

### Test Approach
For each optimization:
1. Run `test_decode_profiling` before → record baseline
2. Apply optimization
3. Run `test_single_layer.py` + `test_model_e2e.py` → PCC still good?
4. Run `test_decode_profiling` after → measure improvement
5. Update throughput baseline

---

## Phase 3: Trace Capture for Decode Path

### Goal
Eliminate Python dispatch overhead by capturing the entire decode graph as a trace and replaying it. Expected 2-5x speedup.

### Prerequisites
- Profiling confirms significant dispatch overhead
- Phase 2 optimizations already applied

### Design

#### 3a. Trace-Compatible Decode Methods
Create `forward_decode()` in the model that takes fixed-shape input (batch=1, T=1) and returns fixed-shape output. No Python conditionals, no dynamic shapes, no torch↔ttnn conversions mid-graph.

#### 3b. Pre-Allocated Buffers
- Input embedding buffer: `[1, 1, 4096]` on device
- Output logits buffer: `[1, 1, vocab_size]` on device
- All 24 recurrent states, 24 conv states, 8 KV caches already on device

#### 3c. Capture Flow
```python
# After warmup decode steps (program cache populated):
trace_id = ttnn.begin_trace_capture(device)
output = model.forward_decode(input_buffer, position)
ttnn.end_trace_capture(device, trace_id)

# Subsequent steps:
ttnn.copy(new_embedding, input_buffer)
ttnn.execute_trace(device, trace_id)
result = ttnn.to_torch(output_buffer)
```

#### 3d. Position Tracking
The `cache_pos` changes each step. If trace doesn't support changing scalar inputs, pre-allocate a position tensor and write to it before `execute_trace`.

#### 3e. Known Trace Blockers (must resolve before capture)
1. **Host RoPE computation** — `qwen35_model.py` creates `torch.full()` and computes RoPE on CPU every decode step. Must pre-compute on device (addressed in Phase 2c).
2. **KV cache concat** — produces variable-length tensors, fundamentally incompatible with trace. Must switch to pre-allocated cache (addressed in Phase 2c).
3. **`try/except` in RMSNorm** — first call may take exception path; trace captures whichever path runs. Warmup must exercise the same path that trace will capture.
4. **`ttnn.to_torch()` in conv FIR fallback** — `ttnn_gated_deltanet.py` line 172 has a fallback that calls `ttnn.to_torch(weight)`. For decode with pre-computed weight_taps this should not be hit, but must be explicitly verified.
5. **Dynamic reshapes with `memory_config=None`** — many `ttnn.reshape` calls in delta rule ops pass no memory config. Trace may need deterministic memory placement.

#### 3f. Risk Mitigation
If experimental ops remain trace-incompatible after addressing the above:
- Refactor those specific paths for trace compatibility
- Or fall back to non-traced decode with Phase 2 gains

#### 3g. Memory Budget for Trace
All buffers must be pre-allocated for trace. Estimate:
- 24 recurrent states: ~24MB (fixed)
- 24 conv states: ~2MB (fixed)
- 8 KV caches at max_seq_len=2048: ~32MB
- Weights: ~9GB in bf8b (DRAM)
- Activations: ~few MB per layer (L1, reused)
- Must verify total fits within DRAM budget with pre-allocation

### Test
`test_traced_decode`:
- Run 10 decode steps without trace, capture outputs
- Capture trace, run 10 more steps with trace replay
- Assert PCC > 0.999 between traced and non-traced outputs
- Assert traced decode is at least 1.5x faster

---

## Phase 4: Iterative Measurement & Regression Prevention

### Test Suite

| Test | Scope | Validates |
|---|---|---|
| `test_decode_profiling` | New | Per-layer timing breakdown, bottleneck identification |
| `test_optimization_regression` | New | Decode throughput >= baseline (8.3 tok/s), updated as we improve |
| `test_traced_decode` | New | PCC between traced/non-traced, trace speedup |
| `test_single_layer.py` | Existing | Per-layer PCC > 0.999 (correctness guard) |
| `test_model_e2e.py` | Existing | End-to-end generation quality |

### Workflow Per Optimization
1. Run `test_decode_profiling` → identify target
2. Implement change
3. Run `test_single_layer` + `test_model_e2e` → PCC still good?
4. Run `test_decode_profiling` again → did the target improve?
5. Update throughput baseline in `test_optimization_regression`

### Timeout Policy
All device tests run with `--timeout=300` (5 min). If a test hangs:
1. Kill the process
2. Run `tt-smi -r` to reset the device
3. Debug the hang before retrying

---

## Files Modified

### New Files
- `tests/test_decode_profiling.py` — per-layer timing breakdown
- `tests/test_optimization_regression.py` — throughput baseline guard
- `tests/test_traced_decode.py` — trace capture validation

### Modified Files (Qwen wrapper code)
- `tt/qwen35_model.py` — trace capture/replay orchestration, `forward_decode()`
- `tt/qwen35_mlp.py` — L1 memory configs, compute kernel configs
- `tt/qwen35_decoder.py` — L1 memory configs, profiling hooks
- `tt/qwen35_gated_deltanet.py` — decode-specific forward, trace support
- `tt/qwen35_gated_attention.py` — decode-specific forward, pre-allocated KV cache

### Modified Files (Experimental ops, with caution)
- `ttnn_delta_rule_ops.py` — op count reduction, L1 memory configs for T=1 path, explicit memory_config on reshapes
- `ttnn_gated_deltanet.py` — memory config optimization, verify no ttnn.to_torch() in decode path
- `ttnn_gated_attention.py` — SDPA program config tuning, fused Q/K norms, fused QKV projection, memory configs on linear ops

---

## Success Criteria

- Per-layer profiling data available and actionable
- Each optimization measurably improves throughput (verified by profiling)
- PCC > 0.999 maintained at every step
- Throughput significantly improved from 8.3 tok/s baseline
