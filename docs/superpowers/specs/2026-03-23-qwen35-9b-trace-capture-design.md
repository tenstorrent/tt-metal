# Qwen3.5-9B Trace Capture Design

## Problem

Decode throughput is ~9 tok/s (~110ms/token). Trace capture can yield 2-5x speedup by recording device ops once and replaying them, eliminating per-step host overhead. Current trace attempt fails because:

1. **DeltaNet recurrent state mutation**: `h = fused_decay_and_write(h, ...)` creates a new tensor each step. Trace replays read stale address A instead of updated address B.
2. **Conv state mutation**: `new_state = ttnn.concat([conv_state[:, 1:, :], x], dim=1)` creates new tensors.
3. **KV cache concat**: `ttnn.concat([past_key, key_states], dim=2)` changes tensor shape each step.
4. **Try/except in RMSNorm**: Control flow breaks trace.
5. **Dynamic slicing**: KV cache `[:, :, :valid_len, :]` varies per step.

## Approach: Pre-allocated Buffers + `ttnn.copy()`

### Core Pattern

For every mutable state tensor, pre-allocate a fixed-size buffer. After computing updated state, `ttnn.copy()` the result back into the pre-allocated buffer. Trace records the copy op — on replay, it writes to the same address, keeping state consistent.

```
# Before trace capture:
h_buffer = ttnn.zeros([B, H, K, V], device=device)

# During trace (captured once, replayed many times):
new_h = fused_decay_and_write(h_buffer, k, delta, decay, beta)
ttnn.copy(new_h, h_buffer)  # Write back to fixed address
ttnn.deallocate(new_h)
```

### Component Changes

#### 1. DeltaNet Recurrent State (24 layers)
- Pre-allocate `h_buffer` of shape `[1, 32, 128, 128]` per layer at init
- After `fused_decay_and_write_ttnn()` produces `new_h`, do `ttnn.copy(new_h, h_buffer)`
- Pass `h_buffer` as `initial_state` on every call
- On `reset_state()`: `ttnn.copy(zeros, h_buffer)` instead of reassigning

#### 2. DeltaNet Conv State (24 layers)
- Pre-allocate `fused_conv_buffer` of shape `[1, 3, 8192]` per layer
- In `_causal_conv1d_decode_t1`: after computing `new_state`, `ttnn.copy(new_state, conv_buffer)`
- Return `conv_buffer` instead of `new_state`

#### 3. KV Cache (8 attention layers)
- Pre-allocate `kv_cache_key` and `kv_cache_value` of shape `[1, 4, max_seq_len, 256]` per layer
- Use existing pre-allocated cache path in `gated_attention_forward_ttnn` (lines 185-201)
- Pass `cache_pos` as integer, write new K/V via `ttnn.copy()`
- For decode SDPA: attend to full pre-allocated cache; zeros in unfilled positions get near-zero softmax weight
- Pre-allocate `cur_pos` device tensor if SDPA supports it; otherwise use full-cache attention

#### 4. RMSNorm Fix
- Remove all `try/except` blocks
- Test which path works for T=1 decode shapes and hardcode it
- For `[1, 1, 4096]` shapes: `ttnn.rms_norm` should work (standard hidden dim)
- For `[1, 1, 32, 128]` shapes (per-head in DeltaNet): may need manual fallback — determine and hardcode

#### 5. Trace I/O Buffers
- Pre-allocate: `trace_input [1, 1, 4096]`, `trace_cos [1, 1, 64]`, `trace_sin [1, 1, 64]`, `trace_cache_pos [1]`
- Pre-allocate: `trace_output` (returned by forward during capture)
- Before each replay: `ttnn.copy()` new embedding, cos, sin, cache_pos into trace buffers

### Memory Budget

Per DeltaNet layer (24 layers):
- Recurrent state: 1 x 32 x 128 x 128 x 2B = 1MB
- Conv state: 1 x 3 x 8192 x 2B = 48KB
- Subtotal: ~1.05MB x 24 = ~25MB

Per Attention layer (8 layers):
- KV cache: 2 x 1 x 4 x 2048 x 256 x 2B = 8MB
- Subtotal: 8MB x 8 = 64MB

Total state: ~89MB (well within DRAM budget)

### Modified Call Flow

```
# One-time capture:
model.reset_state(batch_size=1)
model.prefill(token_ids)               # Populates state, untraced
model.warmup_decode(device)             # 1 untraced decode for program cache
model.capture_decode_trace(device)      # Record trace

# Per-token decode:
model.decode_traced(token_id, pos)      # Copy inputs, replay trace
```

## Implementation Order

1. Remove try/except from RMSNorm — determine correct path, hardcode
2. Pre-allocate DeltaNet recurrent state buffers + copy-back logic
3. Pre-allocate DeltaNet conv state buffers + copy-back logic
4. Pre-allocate KV cache for attention layers + wire up pre-allocated cache path
5. Update `capture_decode_trace()` and `decode_traced()` with all buffers
6. Test trace capture + replay PCC
7. Measure throughput improvement

## Success Criteria

- Trace capture completes without error
- Traced decode PCC > 0.99 vs non-traced decode
- Decode throughput >= 15 tok/s (2x improvement)
- Coherent text generation with traced decode
