# Qwen3.5-9B Performance Phase 2 + Pytest Conversion

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Qwen3.5-9B decode throughput from ~6 tok/s as fast as possible on BH P150, and convert the CLI demo into a proper pytest.

**Architecture:** Three phases: (1) pytest conversion using standard device fixture, (2) program config / memory config / fused activation optimizations, (3) trace capture for decode path. Each phase is independently valuable.

**Tech Stack:** TTNN, PyTorch, pytest, safetensors, transformers

---

### File Structure

**New Files:**
- `models/demos/blackhole/qwen3_5_9b/tests/test_qwen35_demo.py` — Pytest demo test

**Files to Modify:**
- `models/demos/blackhole/qwen3_5_9b/tt/qwen35_mlp.py` — Fused SiLU, L1 memory config, compute kernel config
- `models/demos/blackhole/qwen3_5_9b/tt/qwen35_decoder.py` — L1 memory config for norm outputs
- `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_attention.py` — L1 memory config, compute kernel config
- `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py` — L1 memory config, compute kernel config
- `models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py` — Trace capture/replay for decode
- `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py` — L1 memory configs, compute kernel config for linear projections
- `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py` — L1 memory configs, compute kernel config for linear projections

---

### Task 1: Create Pytest Demo Test

**Files:**
- Create: `models/demos/blackhole/qwen3_5_9b/tests/test_qwen35_demo.py`

- [ ] **Step 1: Write the test file**

```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pytest demo for Qwen3.5-9B text generation on Blackhole P150."""
import time
import torch
import pytest
import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_model import Qwen35Model

CHECKPOINT_DIR = "/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(CHECKPOINT_DIR)


@pytest.fixture(scope="module")
def model(device):
    t0 = time.time()
    model = Qwen35Model.from_pretrained(device, CHECKPOINT_DIR, max_batch_size=1, max_seq_len=2048)
    print(f"\n[PERF] Model load: {time.time() - t0:.1f}s")
    return model


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "num_command_queues": 2}], indirect=True
)
def test_qwen35_text_generation(device, model, tokenizer):
    """Test text generation: prefill + decode loop with performance metrics."""
    device.enable_program_cache()

    prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs["input_ids"]
    prompt_len = token_ids.shape[1]

    model.reset_state(batch_size=1)

    # Prefill
    t0 = time.time()
    logits = model.prefill(token_ids)
    ttnn.synchronize_device(device)
    ttft = time.time() - t0

    logits_torch = ttnn.to_torch(logits).squeeze()
    assert not torch.isnan(logits_torch).any(), "NaN in prefill logits"
    next_token = logits_torch.argmax().item()

    # Decode
    max_new_tokens = 50
    generated_tokens = [next_token]
    decode_times = []

    for i in range(max_new_tokens - 1):
        next_input = torch.tensor([[next_token]], dtype=torch.long)

        t_step = time.time()
        logits = model.decode(next_input, current_pos=prompt_len + i)
        ttnn.synchronize_device(device)
        decode_times.append(time.time() - t_step)

        logits_torch = ttnn.to_torch(logits).squeeze()
        assert not torch.isnan(logits_torch).any(), f"NaN in decode logits at step {i}"
        next_token = logits_torch.argmax().item()

        if next_token == tokenizer.eos_token_id:
            break
        generated_tokens.append(next_token)

    # Assertions
    assert len(generated_tokens) >= 1, "Should generate at least 1 token"

    # Performance metrics
    avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"\n[PERF] TTFT: {ttft:.3f}s")
    print(f"[PERF] Avg decode: {avg_decode*1000:.0f}ms/token ({1/avg_decode:.1f} tok/s)")
    print(f"[PERF] Generated {len(generated_tokens)} tokens")
    print(f"[TEXT] {text[:200]}")
```

- [ ] **Step 2: Run the test to verify it works**

Run: `pytest models/demos/blackhole/qwen3_5_9b/tests/test_qwen35_demo.py -v -s`
Expected: PASS, prints TTFT and tok/s metrics (~6 tok/s baseline)

- [ ] **Step 3: Commit**

```bash
git add models/demos/blackhole/qwen3_5_9b/tests/test_qwen35_demo.py
git commit -m "Add pytest demo for Qwen3.5-9B text generation"
```

---

### Task 2: Fused SiLU Activation in MLP

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_mlp.py:29-39`

Fuse the separate `ttnn.silu(w1_out)` into the `ttnn.linear` call for gate_proj, eliminating one op and one intermediate tensor.

- [ ] **Step 1: Modify MLP forward to use fused activation**

In `qwen35_mlp.py`, change `forward()`:

```python
def forward(self, x):
    w1_out = ttnn.linear(x, self.w1, activation="silu")
    w3_out = ttnn.linear(x, self.w3)
    hidden = ttnn.mul(w1_out, w3_out)
    ttnn.deallocate(w1_out)
    ttnn.deallocate(w3_out)
    output = ttnn.linear(hidden, self.w2)
    ttnn.deallocate(hidden)
    return output
```

This removes the separate `ttnn.silu` call and the `w1_act` intermediate tensor.

- [ ] **Step 2: Run pytest to verify correctness and measure impact**

Run: `pytest models/demos/blackhole/qwen3_5_9b/tests/test_qwen35_demo.py -v -s`
Expected: PASS, check tok/s in output

- [ ] **Step 3: Commit**

```bash
git add models/demos/blackhole/qwen3_5_9b/tt/qwen35_mlp.py
git commit -m "Fuse SiLU activation into MLP gate_proj matmul"
```

---

### Task 3: Add Compute Kernel Configs to Linear Projections

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_mlp.py:13-28`
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py:39-81`
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_attention.py:22-48`

Add `WormholeComputeKernelConfig(math_fidelity=LoFi, fp32_dest_acc_en=True)` to matmul-heavy operations for faster compute.

- [ ] **Step 1: Add compute kernel config to MLP**

In `qwen35_mlp.py`, add to `__init__`:

```python
self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)
```

Update `forward()` to pass it to all `ttnn.linear` calls:

```python
def forward(self, x):
    w1_out = ttnn.linear(x, self.w1, activation="silu", compute_kernel_config=self.compute_kernel_config)
    w3_out = ttnn.linear(x, self.w3, compute_kernel_config=self.compute_kernel_config)
    hidden = ttnn.mul(w1_out, w3_out)
    ttnn.deallocate(w1_out)
    ttnn.deallocate(w3_out)
    output = ttnn.linear(hidden, self.w2, compute_kernel_config=self.compute_kernel_config)
    ttnn.deallocate(hidden)
    return output
```

- [ ] **Step 2: Add compute kernel config to gated attention projections**

In `qwen35_gated_attention.py`, add to `__init__`:

```python
self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)
```

Pass it when calling `gated_attention_forward_ttnn()` — this requires adding a `compute_kernel_config` parameter to the experimental op's signature.

- [ ] **Step 3: Add compute kernel config to gated deltanet projections**

Same pattern for `qwen35_gated_deltanet.py` — pass compute_kernel_config through to `gated_deltanet_forward_ttnn()`.

- [ ] **Step 4: Update experimental ops to accept and use compute_kernel_config**

In `ttnn_gated_deltanet.py`, add `compute_kernel_config=None` parameter to `gated_deltanet_forward_ttnn()`. Pass it to all `ttnn.linear()` calls inside.

In `ttnn_gated_attention.py`, add `compute_kernel_config=None` parameter to `gated_attention_forward_ttnn()`. Pass it to all `ttnn.linear()` calls inside.

- [ ] **Step 5: Run pytest to verify and measure**

Run: `pytest models/demos/blackhole/qwen3_5_9b/tests/test_qwen35_demo.py -v -s`
Expected: PASS, check tok/s improvement

- [ ] **Step 6: Commit**

```bash
git add models/demos/blackhole/qwen3_5_9b/tt/qwen35_mlp.py \
      models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_attention.py \
      models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py \
      models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py \
      models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py
git commit -m "Add LoFi compute kernel configs to all linear projections"
```

---

### Task 4: Add L1 Memory Configs to Decode Path

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_mlp.py`
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py:110-127`

Explicitly route decode-path outputs through L1 instead of defaulting to DRAM.

- [ ] **Step 1: Add L1 memory config to MLP linear calls**

In `qwen35_mlp.py`, add `memory_config=ttnn.L1_MEMORY_CONFIG` to all `ttnn.linear` calls in `forward()`:

```python
def forward(self, x):
    w1_out = ttnn.linear(x, self.w1, activation="silu",
                         memory_config=ttnn.L1_MEMORY_CONFIG,
                         compute_kernel_config=self.compute_kernel_config)
    w3_out = ttnn.linear(x, self.w3,
                         memory_config=ttnn.L1_MEMORY_CONFIG,
                         compute_kernel_config=self.compute_kernel_config)
    hidden = ttnn.mul(w1_out, w3_out, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(w1_out)
    ttnn.deallocate(w3_out)
    output = ttnn.linear(hidden, self.w2,
                         memory_config=ttnn.L1_MEMORY_CONFIG,
                         compute_kernel_config=self.compute_kernel_config)
    ttnn.deallocate(hidden)
    return output
```

- [ ] **Step 2: Add L1 memory config to LM head in model decode**

In `qwen35_model.py`, modify `decode()` method:

```python
logits = ttnn.linear(x, self.lm_head_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
```

Note: The LM head is [4096, 248320] which is large. L1 may OOM for this one — if it does, revert to no memory_config for the LM head only.

- [ ] **Step 3: Run pytest to verify and measure**

Run: `pytest models/demos/blackhole/qwen3_5_9b/tests/test_qwen35_demo.py -v -s`
Expected: PASS (or if L1 OOM on LM head, revert that one change)

- [ ] **Step 4: Commit**

```bash
git add models/demos/blackhole/qwen3_5_9b/tt/qwen35_mlp.py \
      models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py
git commit -m "Route decode-path outputs through L1 memory for lower latency"
```

---

### Task 5: Trace Capture for Decode Path

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py`

This is the high-impact optimization. Trace capture records the decode step's device operations once, then replays them without Python overhead.

**Key insight:** For T=1 decode, all tensor shapes are fixed and the recurrent loop in `recurrent_gated_delta_rule_ttnn` runs exactly once (T=1). The `mode` and Python conditionals are resolved before trace capture since we always call with mode="decode".

- [ ] **Step 1: Add pre-allocated input/output buffers to model**

In `qwen35_model.py`, add to `__init__`:

```python
# Pre-allocate decode buffers for trace capture
self._decode_input_ids = ttnn.from_torch(
    torch.zeros((1, 1), dtype=torch.int32),
    dtype=ttnn.uint32, device=device
)
self._trace_id = None
```

- [ ] **Step 2: Add trace capture method**

```python
def capture_decode_trace(self, warmup_token_id, warmup_pos):
    """Capture a decode step trace after the program cache is warm."""
    # Write warmup token into pre-allocated buffer
    warmup_input = torch.tensor([[warmup_token_id]], dtype=torch.int32)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(warmup_input, dtype=ttnn.uint32),
        self._decode_input_ids,
    )

    self._trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
    self._trace_output = self._decode_step(self._decode_input_ids, warmup_pos)
    ttnn.end_trace_capture(self.device, self._trace_id, cq_id=0)
```

- [ ] **Step 3: Add _decode_step helper and decode_traced method**

```python
def _decode_step(self, token_ids_ttnn, current_pos):
    """Single decode step — must be trace-compatible (no Python conditionals on tensors)."""
    x = ttnn.embedding(token_ids_ttnn, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)

    position_ids = torch.full((1, 1), current_pos, dtype=torch.long)
    cos, sin = self.rope.get_rot_mats(position_ids)

    for layer in self.layers:
        x = layer.forward(x, cos=cos, sin=sin, mode="decode")

    x = rms_norm_ttnn(x, self.norm_weight, eps=self.norm_eps)
    logits = ttnn.linear(x, self.lm_head_weight)
    ttnn.deallocate(x)
    return logits

def decode_traced(self, token_id_int, current_pos):
    """Execute traced decode step."""
    if self._trace_id is None:
        # Fallback to non-traced path
        token_ids = torch.tensor([[token_id_int]], dtype=torch.long)
        return self.decode(token_ids, current_pos)

    # Update input tensor
    new_input = torch.tensor([[token_id_int]], dtype=torch.int32)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(new_input, dtype=ttnn.uint32),
        self._decode_input_ids,
    )

    # Update RoPE for new position
    position_ids = torch.full((1, 1), current_pos, dtype=torch.long)
    cos, sin = self.rope.get_rot_mats(position_ids)
    # Note: RoPE tensors need to be updated in the trace's input slots

    ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=True)
    return self._trace_output
```

**Important caveat:** Trace capture requires that ALL input tensors (including RoPE cos/sin) are written to the same device memory locations. The RoPE position changes each step, which means we need to either:
- Pre-allocate RoPE input buffers and copy into them before each trace execution
- Or accept that RoPE changes break trace and use trace only for the non-RoPE layers

If trace doesn't work due to RoPE or experimental op issues, fall back to the non-traced path (we still get Phase 1 gains).

- [ ] **Step 4: Update test to try trace capture**

Update `test_qwen35_demo.py` to attempt trace capture after a few warmup decode steps:

```python
# After first 2 decode steps (program cache warm), try trace
if i == 2:
    try:
        model.capture_decode_trace(next_token, prompt_len + i)
        use_trace = True
        print("[TRACE] Decode trace captured successfully")
    except Exception as e:
        print(f"[TRACE] Trace capture failed: {e}, continuing without trace")
        use_trace = False

if use_trace:
    logits = model.decode_traced(next_token, current_pos=prompt_len + i)
else:
    logits = model.decode(next_input, current_pos=prompt_len + i)
```

- [ ] **Step 5: Run pytest and measure trace impact**

Run: `pytest models/demos/blackhole/qwen3_5_9b/tests/test_qwen35_demo.py -v -s`
Expected: If trace works, significant speedup (2-5x). If trace fails, graceful fallback.

- [ ] **Step 6: Commit**

```bash
git add models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py \
      models/demos/blackhole/qwen3_5_9b/tests/test_qwen35_demo.py
git commit -m "Add trace capture for decode path"
```

---

### Task 6: Measure and Iterate

- [ ] **Step 1: Run full test with all optimizations**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_qwen35_demo.py -v -s
```

Record final metrics: TTFT, avg decode latency, tok/s.

- [ ] **Step 2: If trace failed, investigate and fix**

Common trace issues:
- Host-device transfers inside trace: find and eliminate
- Dynamic shapes: ensure all tensors have fixed shapes
- Python conditionals on tensor values: resolve before trace boundary

- [ ] **Step 3: If performance still needs improvement, profile individual ops**

Add per-layer timing to identify the slowest operations and target them specifically.
