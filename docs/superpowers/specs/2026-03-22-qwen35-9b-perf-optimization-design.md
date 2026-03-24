# Qwen3.5-9B Performance Optimization & Pytest Conversion

## Goal

Improve decode throughput from 6 tok/s toward 15-30 tok/s on Blackhole P150, and convert the standalone demo into a proper pytest using standard fixtures.

## Current State

- Model: Qwen3.5-9B hybrid (24 DeltaNet + 8 Gated Full Attention layers)
- Hardware: Blackhole P150, batch=1, max_seq_len=2048
- Decode throughput: ~6.1 tok/s (165ms/token)
- Demo: standalone CLI script (`demo/demo.py`), not a pytest
- Existing optimizations: program cache, bf8b weights, fused QKV, device-cached RoPE, explicit deallocate

## Design

### Part 1: Pytest Conversion

Convert demo into `tests/test_qwen35_demo.py`:

- Use standard `device` fixture from root `conftest.py` with `device_params` parametrization
- Module-scoped fixtures for model loading (expensive) and tokenizer
- Single test `test_qwen35_text_generation`:
  - Prefills a chat-formatted prompt
  - Decodes up to 50 tokens
  - Asserts: valid output shape, no NaN, at least 1 token generated
  - Prints throughput metrics (tok/s, TTFT)
- Device params: `{"l1_small_size": 24576, "num_command_queues": 2}`

### Part 2: Phase 1 - Program Config & Memory Optimizations

Target: 10-25% improvement (6 -> 7-8 tok/s). Low risk, no experimental op changes.

1. **MLP fused activation** (`qwen35_mlp.py`): Fuse SiLU into gate_proj matmul via `activation="silu"` in `ttnn.linear`, eliminating separate `ttnn.silu` op and intermediate tensor.

2. **Explicit L1 memory configs** (`qwen35_mlp.py`, `qwen35_decoder.py`): Add `memory_config=ttnn.L1_MEMORY_CONFIG` to all `ttnn.linear` calls in decode path.

3. **Compute kernel configs**: Add `ttnn.WormholeComputeKernelConfig(math_fidelity=LoFi, fp32_dest_acc_en=True)` to MLP projections and QKV projections.

4. **Device params**: `num_command_queues=2` via pytest fixture for async dispatch.

5. **Incremental measurement**: Run pytest after each change to measure impact.

### Part 3: Phase 2 - Trace Capture for Decode Path

Target: 2-5x improvement. Medium risk, requires decode-specific wrappers.

1. **Separate decode path**: Create `forward_decode()` methods in `qwen35_gated_deltanet.py` and `qwen35_gated_attention.py` that hardcode T=1, mode="decode", is_causal=False - no Python conditionals.

2. **Pre-allocate state buffers**: Before trace capture, allocate all recurrent states, conv states, KV caches with known shapes on device.

3. **Trace capture in `qwen35_model.py`**:
   - After first real decode step, `ttnn.begin_trace_capture`
   - Run one decode step through trace-compatible path
   - `ttnn.end_trace_capture` -> trace_id
   - Subsequent steps: write token embedding into pre-allocated input, `ttnn.execute_trace`

4. **Validation**: PCC > 0.99 between traced and non-traced outputs.

5. **Fallback**: If experimental ops are incompatible with trace, keep non-traced decode with Phase 1 optimizations.

## Success Criteria

- Pytest runs and passes on BH P150
- Decode throughput measurably improved (target: as fast as possible, minimum 7+ tok/s from Phase 1)
- No correctness regression (PCC validation, generated text quality)

## Files Modified

- `tests/test_qwen35_demo.py` (new) - pytest
- `tt/qwen35_mlp.py` - fused activation, L1 memory configs, compute kernel configs
- `tt/qwen35_decoder.py` - L1 memory configs for norm outputs
- `tt/qwen35_gated_deltanet.py` - decode-specific forward, trace support
- `tt/qwen35_gated_attention.py` - decode-specific forward, trace support
- `tt/qwen35_model.py` - trace capture/replay orchestration
