# Qwen3.5-9B Trace Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable TTNN trace capture/replay for decode, achieving 2-5x decode throughput improvement (from ~9 to 18+ tok/s).

**Architecture:** Pre-allocate all mutable state tensors (DeltaNet recurrent state, conv states, KV caches) at fixed sizes. Replace all state mutations with `ttnn.copy()` into pre-allocated buffers so trace replay reads/writes consistent addresses. Remove control flow (try/except) from traced path.

**Tech Stack:** TTNN, Python, pytest, Blackhole P150

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `tt/qwen35_decoder.py` | Modify | Remove try/except from `rms_norm_ttnn`, add trace-safe RMSNorm |
| `experimental/.../ttnn_gated_deltanet.py` | Modify | Add `_causal_conv1d_decode_t1_inplace` with copy-back to pre-allocated conv buffer |
| `experimental/.../ttnn_delta_rule_ops.py` | Modify | Add `recurrent_gated_delta_rule_decode_inplace_ttnn` with copy-back to pre-allocated state buffer |
| `tt/qwen35_gated_deltanet.py` | Modify | Pre-allocate recurrent state + conv state buffers, use inplace variants |
| `experimental/.../ttnn_gated_attention.py` | Modify | Wire up pre-allocated KV cache path for decode (already partially implemented) |
| `tt/qwen35_gated_attention.py` | Modify | Pre-allocate KV caches, pass cache_pos, wire pre-allocated path |
| `tt/qwen35_model.py` | Modify | New `capture_decode_trace()` + `decode_traced()` with all pre-allocated buffers |
| `tests/test_traced_decode.py` | Modify | Update test to validate trace works, measure throughput |

---

### Task 1: Fix RMSNorm — Remove try/except

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_decoder.py:14-32`
- Modify: `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py:25-51`
- Modify: `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py:49-58,155-173`

The try/except in RMSNorm breaks trace capture. We need to determine which path works for each shape and hardcode it.

- [ ] **Step 1: Test which RMSNorm path works for decode shapes**

Run a quick test to determine if `ttnn.rms_norm` works for the shapes used during decode:
- `[1, 1, 4096]` — decoder block input/output norms
- `[1, 1, 16, 256]` — Q/K norms in gated attention
- `[1, 1, 32, 128]` — output norm in gated DeltaNet

```python
# Quick test script — run interactively
import torch, ttnn
device = ttnn.open_device(0)

# Test [1, 1, 4096] — most common
x = ttnn.from_torch(torch.randn(1, 1, 4096, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
w = ttnn.from_torch(torch.ones(4096, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
try:
    r = ttnn.rms_norm(x, weight=w, epsilon=1e-6)
    print("[1,1,4096]: fused works")
except Exception as e:
    print(f"[1,1,4096]: fused fails: {e}")

# Test [1, 1, 16, 256] — Q/K norms
x2 = ttnn.from_torch(torch.randn(1, 1, 16, 256, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
w2 = ttnn.from_torch(torch.ones(256, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
try:
    r2 = ttnn.rms_norm(x2, weight=w2, epsilon=1e-6)
    print("[1,1,16,256]: fused works")
except Exception as e:
    print(f"[1,1,16,256]: fused fails: {e}")

# Test [1, 1, 32, 128] — DeltaNet output norm
x3 = ttnn.from_torch(torch.randn(1, 1, 32, 128, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
w3 = ttnn.from_torch(torch.ones(128, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
try:
    r3 = ttnn.rms_norm(x3, weight=w3, epsilon=1e-6)
    print("[1,1,32,128]: fused works")
except Exception as e:
    print(f"[1,1,32,128]: fused fails: {e}")

ttnn.close_device(device)
```

- [ ] **Step 2: Replace try/except with hardcoded paths**

In `qwen35_decoder.py`, replace `rms_norm_ttnn`:

```python
def rms_norm_ttnn(x, weight, eps=1e-6, memory_config=None):
    """Zero-centered RMSNorm — fused path for standard shapes, manual for others."""
    ndim = len(x.shape)
    if ndim <= 3:
        # [B, T, D] — fused ttnn.rms_norm works
        return ttnn.rms_norm(x, weight=weight, epsilon=eps, memory_config=memory_config)
    else:
        # [B, T, H, D] — manual path
        mc = memory_config
        x_sq = ttnn.multiply(x, x, memory_config=mc)
        variance = ttnn.mean(x_sq, dim=-1, keepdim=True, memory_config=mc)
        inv_rms = ttnn.rsqrt(ttnn.add(variance, eps), memory_config=mc)
        x_normed = ttnn.multiply(x, inv_rms, memory_config=mc)
        return ttnn.multiply(x_normed, weight, memory_config=mc)
```

Apply the same pattern in `ttnn_gated_deltanet.py` for `rms_norm_gated_ttnn` and `rms_norm_ttnn`, and in `ttnn_gated_attention.py` for the Q/K norm blocks.

- [ ] **Step 3: Run existing test to verify no regression**

Run: `pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py::TestDecodeLoop::test_generate_tokens -v -s --timeout=600`
Expected: PASS, coherent output, no NaN

- [ ] **Step 4: Commit**

```
git add -p && git commit -m "fix: remove try/except from RMSNorm for trace compatibility"
```

---

### Task 2: Pre-allocate DeltaNet Recurrent State with Copy-back

**Files:**
- Modify: `models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py:472-579`
- Modify: `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py:298-613`
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py:244-344`

- [ ] **Step 1: Add inplace decode function to delta_rule_ops.py**

After `recurrent_gated_delta_rule_decode_ttnn` (line 579), add:

```python
def recurrent_gated_delta_rule_decode_inplace_ttnn(
    q, k, v, beta, g, state_buffer, scale=None, device=None,
):
    """Like recurrent_gated_delta_rule_decode_ttnn but writes state back to pre-allocated buffer.

    For trace capture: state_buffer is a pre-allocated [B, H, K, V] tensor.
    After computing the new state, we copy it back to state_buffer so the
    trace replay reads the correct address on the next iteration.
    """
    o, new_h = recurrent_gated_delta_rule_decode_ttnn(
        q=q, k=k, v=v, beta=beta, g=g,
        scale=scale, initial_state=state_buffer, device=device,
    )
    # Copy new state back to pre-allocated buffer for trace consistency
    ttnn.copy(new_h, state_buffer)
    ttnn.deallocate(new_h)
    return o, state_buffer
```

- [ ] **Step 2: Wire inplace decode in gated_deltanet_forward_ttnn**

In `ttnn_gated_deltanet.py`, add `use_inplace_state=False` parameter. When True and T==1, call `recurrent_gated_delta_rule_decode_inplace_ttnn` instead:

At the import section (line 14), add:
```python
from tt.ttnn_delta_rule_ops import recurrent_gated_delta_rule_decode_inplace_ttnn
```

In the T==1 branch (line 575-584), add:
```python
    elif T == 1:
        if use_inplace_state and recurrent_state is not None:
            o, new_state = recurrent_gated_delta_rule_decode_inplace_ttnn(
                q=q, k=k, v=v, beta=beta, g=g,
                state_buffer=recurrent_state, device=device,
            )
        else:
            o, new_state = recurrent_gated_delta_rule_decode_ttnn(
                q=q, k=k, v=v, beta=beta, g=g,
                initial_state=recurrent_state, device=device,
            )
```

- [ ] **Step 3: Pre-allocate state buffer in Qwen35GatedDeltaNet**

In `qwen35_gated_deltanet.py`, modify `_init_recurrent_state` to allocate a persistent buffer. Add a `use_inplace_state` flag:

```python
def _init_recurrent_state(self, batch_size):
    state = torch.zeros(batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim, dtype=torch.bfloat16)
    self.recurrent_state = ttnn.from_torch(state, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

def enable_inplace_state(self):
    """Enable inplace state updates for trace capture."""
    self.use_inplace_state = True

def reset_state_inplace(self):
    """Zero out pre-allocated state buffer without reallocating."""
    if self.recurrent_state is not None:
        zeros = ttnn.from_torch(
            torch.zeros_like(ttnn.to_torch(self.recurrent_state)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
        )
        ttnn.copy(zeros, self.recurrent_state)
        ttnn.deallocate(zeros)
```

Pass `use_inplace_state=getattr(self, 'use_inplace_state', False)` to `gated_deltanet_forward_ttnn`.

- [ ] **Step 4: Run test to verify non-traced path still works**

Run: `pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py::TestDecodeLoop::test_generate_tokens -v -s --timeout=600`
Expected: PASS

- [ ] **Step 5: Commit**

---

### Task 3: Pre-allocate Conv State with Copy-back

**Files:**
- Modify: `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py:82-117`
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py:257-333`

- [ ] **Step 1: Add inplace conv decode function**

In `ttnn_gated_deltanet.py`, add after `_causal_conv1d_decode_t1`:

```python
def _causal_conv1d_decode_t1_inplace(x, conv_buffer, kernel_size, device, memory_config=None,
                                      weight_taps=None, bias_dev=None):
    """Conv1d decode with copy-back to pre-allocated buffer for trace capture.

    conv_buffer: pre-allocated [B, kernel_size-1, D] tensor (fixed address).
    Computes output, then updates conv_buffer in-place via ttnn.copy().
    """
    mc = memory_config

    # Compute output: weighted sum of state positions and current input
    out = ttnn.multiply(x, weight_taps[kernel_size - 1], memory_config=mc)
    for k in range(kernel_size - 1):
        s_k = conv_buffer[:, k:k+1, :]
        s_k = ttnn.to_layout(s_k, ttnn.TILE_LAYOUT)
        term = ttnn.multiply(s_k, weight_taps[k], memory_config=mc)
        out = ttnn.add(out, term, memory_config=mc)

    if bias_dev is not None:
        out = ttnn.add(out, bias_dev, memory_config=mc)

    # Update conv buffer in-place: shift left and append x
    new_state = ttnn.concat([conv_buffer[:, 1:, :], x], dim=1, memory_config=mc)
    new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT)
    ttnn.copy(new_state, conv_buffer)
    ttnn.deallocate(new_state)

    return ttnn.silu(out, memory_config=mc), conv_buffer
```

- [ ] **Step 2: Wire inplace conv in fused conv decode path**

In `gated_deltanet_forward_ttnn`, when `use_inplace_state=True` and `use_fused_conv`, use `_causal_conv1d_decode_t1_inplace` instead of `_causal_conv1d_decode_t1`.

- [ ] **Step 3: Pre-allocate fused conv buffer in Qwen35GatedDeltaNet**

In `qwen35_gated_deltanet.py`, add method to allocate fused conv buffer:

```python
def _init_fused_conv_buffer(self, batch_size):
    """Pre-allocate fused conv state buffer for trace capture."""
    D_total = self.args.linear_q_dim + self.args.linear_k_dim + self.args.linear_v_dim
    state = torch.zeros(batch_size, self.conv_kernel_size - 1, D_total, dtype=torch.bfloat16)
    self.fused_conv_buffer = ttnn.from_torch(
        state, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
    )
```

After prefill, when fusing conv states, copy into the pre-allocated buffer instead of creating a new tensor.

- [ ] **Step 4: Run test**

Run: `pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py::TestDecodeLoop::test_generate_tokens -v -s --timeout=600`
Expected: PASS

- [ ] **Step 5: Commit**

---

### Task 4: Pre-allocate KV Cache for Attention Layers

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_attention.py:22-122`
- Modify: `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py:184-216`

- [ ] **Step 1: Pre-allocate KV cache in Qwen35GatedAttention**

```python
def __init__(self, args, state_dict, layer_num, device, weight_cache_path=None):
    # ... existing init ...

    # Pre-allocated KV cache for trace capture
    self.max_seq_len = args.max_seq_len
    self.kv_cache_key = None  # Allocated on first use or enable_trace
    self.kv_cache_value = None
    self.cache_pos = 0
    self.use_preallocated_cache = False

def enable_preallocated_cache(self, batch_size=1):
    """Allocate fixed-size KV cache for trace-compatible decode."""
    self.kv_cache_key = ttnn.from_torch(
        torch.zeros(batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
    )
    self.kv_cache_value = ttnn.from_torch(
        torch.zeros(batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
    )
    self.use_preallocated_cache = True
    self.cache_pos = 0
```

- [ ] **Step 2: Update forward to use pre-allocated cache**

```python
def forward(self, x, cos, sin):
    T = x.shape[1]
    mc = ttnn.L1_MEMORY_CONFIG if T == 1 else None

    if self.use_preallocated_cache:
        output, new_key, new_value = gated_attention_forward_ttnn(
            hidden_states=x,
            # ... all weight params ...
            cos=cos, sin=sin,
            kv_cache_key=self.kv_cache_key,
            kv_cache_value=self.kv_cache_value,
            cache_pos=self.cache_pos,
            cache_len=self.max_seq_len,
            memory_config=mc,
            # ... other params ...
        )
        self.cache_pos += T
        return output
    else:
        # Existing concat-based path (unchanged)
        ...
```

- [ ] **Step 3: Handle prefill → decode transition**

After prefill with T tokens, `cache_pos = T`. During decode, each step writes at `cache_pos` and increments.

For trace: `cache_pos` needs to be a device tensor that can be overwritten. But the slicing `kv_cache[:, :, cache_pos:cache_pos+T, :]` uses Python ints. For decode (T=1), the slice range is fixed-size (1 token), but the offset changes.

Alternative for trace: after each decode step, shift the cache write position. Since SDPA with `is_causal=False` attends to all positions, unfilled zero entries get near-zero softmax weight.

For the trace path, we can use a rotating write approach:
- Pre-allocate position tensor on device
- Use `ttnn.experimental.paged_update_cache` if available, or fall back to slice+copy

If slicing with dynamic position doesn't work in trace, we can attend to the full cache (zeros give ~0 attention weight after softmax).

- [ ] **Step 4: Run test**

Run: `pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py -v -s --timeout=600`
Expected: PASS

- [ ] **Step 5: Commit**

---

### Task 5: Wire Up Trace Capture in Model

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py:177-235`

- [ ] **Step 1: Add enable_trace method**

```python
def enable_trace(self):
    """Prepare all layers for trace-compatible decode."""
    for layer in self.layers:
        if layer.is_full_attention:
            layer.attention.enable_preallocated_cache(batch_size=1)
        else:
            layer.attention.enable_inplace_state()
            layer.attention._init_fused_conv_buffer(batch_size=1)
```

- [ ] **Step 2: Update capture_decode_trace**

```python
def capture_decode_trace(self, device):
    """Capture trace for decode path after prefill + warmup."""
    self.enable_trace()

    # Pre-allocate I/O buffers
    self._trace_input = ttnn.from_torch(
        torch.zeros(1, 1, self.args.dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )
    self._trace_cos = ttnn.from_torch(
        torch.zeros(1, 1, self.args.rope_head_dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )
    self._trace_sin = ttnn.from_torch(
        torch.zeros(1, 1, self.args.rope_head_dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )

    # Capture
    self._trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    self._trace_output = self.forward_decode(self._trace_input, self._trace_cos, self._trace_sin)
    ttnn.end_trace_capture(device, self._trace_id, cq_id=0)
```

- [ ] **Step 3: Update decode_traced**

```python
def decode_traced(self, token_ids, current_pos):
    """Traced decode: write inputs to pre-allocated buffers, replay trace."""
    token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
    x = ttnn.embedding(token_ids_ttnn, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
    ttnn.deallocate(token_ids_ttnn)

    ttnn.copy(x, self._trace_input)
    ttnn.deallocate(x)

    cos_pos = self.rope.cos_device[:, current_pos:current_pos+1, :]
    sin_pos = self.rope.sin_device[:, current_pos:current_pos+1, :]
    ttnn.copy(cos_pos, self._trace_cos)
    ttnn.copy(sin_pos, self._trace_sin)

    ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)
    return self._trace_output
```

- [ ] **Step 4: Add reset_state_for_trace**

```python
def reset_state_for_trace(self, batch_size=1):
    """Reset state using inplace copy (preserves pre-allocated buffer addresses)."""
    for layer in self.layers:
        if layer.is_full_attention:
            layer.attention.reset_cache_inplace()
        else:
            layer.attention.reset_state_inplace()
```

- [ ] **Step 5: Commit**

---

### Task 6: Update Traced Decode Test

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tests/test_traced_decode.py`

- [ ] **Step 1: Update test to use new trace flow**

The test should:
1. Load model, prefill
2. Run 2 untraced decode steps (warmup for program cache)
3. Reset state, prefill again
4. Enable trace, capture trace
5. Run traced decode, compare PCC with baseline
6. Measure throughput

Key change: remove the `pytest.skip` on PCC < 0.999 and replace with `assert min_pcc > 0.95`.

- [ ] **Step 2: Run trace test**

Run: `pytest models/demos/blackhole/qwen3_5_9b/tests/test_traced_decode.py -v -s --timeout=600`
Expected: PASS with PCC > 0.95 and improved throughput

- [ ] **Step 3: Commit**

---

### Task 7: Validate and Measure

**Files:**
- Test: `models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py`
- Test: `models/demos/blackhole/qwen3_5_9b/tests/test_traced_decode.py`

- [ ] **Step 1: Run full e2e test (non-traced path regression)**

Run: `pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py -v -s --timeout=600`
Expected: PASS, same throughput as before

- [ ] **Step 2: Run traced decode test (new path)**

Run: `pytest models/demos/blackhole/qwen3_5_9b/tests/test_traced_decode.py -v -s --timeout=600`
Expected: PASS, throughput >= 15 tok/s

- [ ] **Step 3: Run demo with traced decode**

Run: `python models/demos/blackhole/qwen3_5_9b/demo/demo.py --prompt "Explain quantum computing" --max-tokens 50`
Expected: Coherent output, improved throughput displayed

- [ ] **Step 4: Final commit with perf numbers in STATUS.md**
