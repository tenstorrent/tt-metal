# Qwen3.5-9B Performance Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize Qwen3.5-9B decode throughput from ~3.5 tok/s to 10+ tok/s and reduce model load time from ~150s to <30s on Blackhole P150.

**Architecture:** Enable program caching (single biggest win), eliminate CPU-device round-trips in decode path (conv states, RoPE), pre-allocate KV cache instead of concatenating, and add TTNN weight caching to disk. All changes are in-place modifications to existing files.

**Tech Stack:** TTNN, PyTorch, pytest, safetensors

---

### File Structure

**Files to Modify:**
- `models/demos/blackhole/qwen3_5_9b/tt/model_config.py` — Add weight cache path, program cache flag
- `models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py` — Enable program cache, pre-allocate KV cache, cache RoPE on device
- `models/demos/blackhole/qwen3_5_9b/tt/qwen35_rope.py` — Pre-compute full RoPE table on device
- `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_attention.py` — Use pre-allocated KV cache with position index
- `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py` — Keep conv states on device
- `models/demos/blackhole/qwen3_5_9b/tt/qwen35_decoder.py` — Pass current_pos for KV cache indexing
- `models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py` — Add program cache to device fixture
- `models/demos/blackhole/qwen3_5_9b/demo/demo.py` — Enable program cache
- `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py` — KV cache write-in-place
- `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py` — Device-side conv state, deallocate intermediates

---

### Task 1: Enable Program Cache

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py`
- Modify: `models/demos/blackhole/qwen3_5_9b/demo/demo.py`

This is the single highest-impact change. Program cache stores compiled TTNN kernels so they don't recompile on every call.

- [ ] **Step 1: Add program cache to test device fixture**

In `test_model_e2e.py`, modify the device fixture:
```python
@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)
```

- [ ] **Step 2: Add program cache to demo**

In `demo.py`, after `ttnn.open_device`:
```python
device = ttnn.open_device(device_id=0)
device.enable_program_cache()
```

- [ ] **Step 3: Add program cache to test_single_layer.py device fixture**

Same pattern in `test_single_layer.py`.

- [ ] **Step 4: Run e2e test and measure baseline with program cache**

Run: `pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py::TestDecodeLoop::test_generate_tokens -v -s`
Expected: Noticeable improvement in decode latency, especially after first few tokens (cache warm-up).

---

### Task 2: Eliminate CPU-Device Round-Trips in Conv State Management

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py`
- Modify: `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py`

The decode path calls `ttnn.to_torch()` 3x per DeltaNet layer (72 total per decode step) to save conv states on CPU. This forces device synchronization. Fix: keep conv states as TTNN tensors on device.

- [ ] **Step 1: Modify `_causal_conv1d_fir` to keep conv state on device**

In `ttnn_gated_deltanet.py`, modify `_causal_conv1d_fir` to return device tensors for conv state instead of CPU torch tensors. For T=1 decode, the new state is simply the input token (no slicing needed).

- [ ] **Step 2: Modify `causal_conv1d_ttnn` to propagate device conv states**

Update the wrapper to handle device-side conv states.

- [ ] **Step 3: Modify `Qwen35GatedDeltaNet` to store device conv states**

In `qwen35_gated_deltanet.py`, update conv state storage to keep TTNN tensors.

- [ ] **Step 4: Run e2e test and measure improvement**

Run: `pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py::TestDecodeLoop::test_generate_tokens -v -s`

---

### Task 3: Cache RoPE Tensors on Device

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_rope.py`

Currently `get_rot_mats` creates new tensors on CPU and transfers to device every call. For decode (T=1), we can pre-compute the full table on device and just slice.

- [ ] **Step 1: Pre-compute full RoPE table on device at init**

```python
# In __init__, after computing cos_cpu/sin_cpu:
self.cos_device = ttnn.from_torch(
    self.cos_cpu.unsqueeze(0).unsqueeze(0),  # [1, 1, max_seq_len, head_dim]
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
)
self.sin_device = ttnn.from_torch(
    self.sin_cpu.unsqueeze(0).unsqueeze(0),
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
)
```

- [ ] **Step 2: Add fast path for single-position decode**

```python
def get_rot_mats(self, position_ids):
    # For decode (single position), slice from pre-computed device tensor
    if position_ids.numel() == 1:
        pos = position_ids.item()
        cos = self.cos_device[:, :, pos:pos+1, :]
        sin = self.sin_device[:, :, pos:pos+1, :]
        return cos, sin
    # For prefill, use CPU path (variable length)
    ...existing code...
```

- [ ] **Step 3: Run e2e test and measure improvement**

---

### Task 4: Pre-allocate KV Cache (Write-in-Place)

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_attention.py`
- Modify: `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py`
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py`
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_decoder.py`

Current code concatenates past + new KV tensors every decode step (O(n) copies). Fix: pre-allocate max-size KV cache and use `ttnn.slice` / write at position.

- [ ] **Step 1: Add KV cache pre-allocation in `Qwen35GatedAttention.__init__`**

Pre-allocate cache tensors: `[B, H_kv, max_seq_len, head_dim]` filled with zeros.

- [ ] **Step 2: Modify `gated_attention_forward_ttnn` to accept cache + position index**

Add `cache_position` parameter. Instead of concat, write new K/V into cache at the position. Use `ttnn.slice` to read only the valid portion for attention.

- [ ] **Step 3: Thread `current_pos` through decoder to attention**

Pass position from `model.decode()` → `layer.forward()` → `attention.forward()`.

- [ ] **Step 4: Handle prefill case — batch-write all positions at once**

During prefill, write all T positions into cache at once.

- [ ] **Step 5: Run e2e test and measure improvement**

---

### Task 5: Deallocate Intermediate Tensors

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_decoder.py`
- Modify: `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py`
- Modify: `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py`
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_mlp.py`

Add `ttnn.deallocate()` calls for intermediate tensors to reduce memory pressure.

- [ ] **Step 1: Add deallocation in MLP forward**

```python
def forward(self, x):
    w1_out = ttnn.linear(x, self.w1)
    w3_out = ttnn.linear(x, self.w3)
    w1_act = ttnn.silu(w1_out)
    ttnn.deallocate(w1_out)
    hidden = ttnn.mul(w1_act, w3_out)
    ttnn.deallocate(w1_act)
    ttnn.deallocate(w3_out)
    output = ttnn.linear(hidden, self.w2)
    ttnn.deallocate(hidden)
    return output
```

- [ ] **Step 2: Add deallocation in decoder forward**

Deallocate `attn_input`, `attn_output`, `ff_input`, `ff_output` after they're consumed.

- [ ] **Step 3: Add deallocation in gated attention forward**

Deallocate intermediate Q/K/V projections after reshape, gate after multiply.

- [ ] **Step 4: Run e2e test and verify no regression**

---

### Task 6: Measure Final Performance and Iterate

- [ ] **Step 1: Run full e2e test suite**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py -v -s
```

- [ ] **Step 2: Compare metrics against baseline**

Record: TTFT, avg decode latency, tok/s, first decode time, model load time.

- [ ] **Step 3: Profile if targets not met**

If not meeting 10+ tok/s, identify remaining bottlenecks using timing instrumentation.
