# Qwen3.5-9B Decode Throughput Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Maximize decode throughput (tok/s) on Blackhole P150 from 8.3 tok/s baseline, with no accuracy regression (PCC > 0.999).

**Architecture:** Profile-first approach — build timing infrastructure, identify bottlenecks, optimize the hottest code paths, then attempt trace capture for the decode loop.

**Tech Stack:** TTNN, PyTorch (reference), pytest, Blackhole P150

**Spec:** `docs/superpowers/specs/2026-03-23-qwen35-9b-perf-optimization-design.md`

**IMPORTANT:** Do NOT commit anything. Keep all changes local. All device tests must use `--timeout=300`. If a test hangs, kill the process and run `tt-smi -r` before retrying.

---

## File Map

### New Files
| File | Responsibility |
|---|---|
| `models/demos/blackhole/qwen3_5_9b/tests/test_decode_profiling.py` | Per-layer timing breakdown for decode |
| `models/demos/blackhole/qwen3_5_9b/tests/test_optimization_regression.py` | Throughput baseline guard |
| `models/demos/blackhole/qwen3_5_9b/tests/test_traced_decode.py` | Trace capture PCC validation |

### Modified Files
| File | Changes |
|---|---|
| `models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py` | Enhanced profiling, RoPE optimization |
| `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_attention.py` | Pre-allocated KV cache, fused QKV |
| `models/demos/blackhole/qwen3_5_9b/tt/qwen35_decoder.py` | L1 memory configs for norms |
| `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py` | Fused Q/K norms, memory configs on linears |
| `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py` | Memory configs for decode path ops |
| `models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py` | L1 memory configs, reduce redundant to_layout calls |

---

## Task 1: Build Profiling Test

**Files:**
- Create: `models/demos/blackhole/qwen3_5_9b/tests/test_decode_profiling.py`

- [ ] **Step 1: Write the profiling test**

```python
# models/demos/blackhole/qwen3_5_9b/tests/test_decode_profiling.py
"""Per-layer decode profiling for Qwen3.5-9B.

Measures timing breakdown: embedding, each layer (norm + attn/deltanet + MLP), final norm + LM head.
Run: pytest models/demos/blackhole/qwen3_5_9b/tests/test_decode_profiling.py -v -s --timeout=300
"""
import time
import pytest
from loguru import logger
import torch
import ttnn

from models.demos.blackhole.qwen3_5_9b.tt.qwen35_model import Qwen35Model
from models.common.utility_functions import run_for_blackhole

CHECKPOINT_DIR = "/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"
pytestmark = run_for_blackhole()


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def model(device):
    m = Qwen35Model.from_pretrained(device, CHECKPOINT_DIR, max_batch_size=1, max_seq_len=2048)
    return m


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(CHECKPOINT_DIR)


class TestDecodeProfile:
    def test_per_layer_timing(self, model, tokenizer, device):
        """Profile decode: per-layer breakdown with sync barriers."""
        prompt = "Hello, my name is"
        inputs = tokenizer(prompt, return_tensors="pt")
        token_ids = inputs["input_ids"]
        T = token_ids.shape[1]

        # Prefill first
        model.reset_state(batch_size=1)
        logits = model.prefill(token_ids)
        ttnn.synchronize_device(device)
        logits_torch = ttnn.to_torch(logits).squeeze()
        next_token = logits_torch.argmax().item()

        # Warmup 2 decode steps (populate program cache)
        for i in range(2):
            next_input = torch.tensor([[next_token]], dtype=torch.long)
            logits = model.decode(next_input, current_pos=T + i)
            ttnn.synchronize_device(device)
            logits_torch = ttnn.to_torch(logits).squeeze()
            next_token = logits_torch.argmax().item()

        # Profile 10 decode steps
        num_steps = 10
        all_timings = []  # list of dicts per step

        for step in range(num_steps):
            timings = {}
            next_input = torch.tensor([[next_token]], dtype=torch.long)
            pos = T + 2 + step

            ttnn.synchronize_device(device)
            t_total_start = time.perf_counter()

            # Embedding
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            token_ids_ttnn = ttnn.from_torch(next_input, dtype=ttnn.uint32, device=device)
            x = ttnn.embedding(token_ids_ttnn, model.tok_embeddings, layout=ttnn.TILE_LAYOUT)
            ttnn.deallocate(token_ids_ttnn)
            ttnn.synchronize_device(device)
            timings["embedding"] = time.perf_counter() - t0

            # RoPE
            t0 = time.perf_counter()
            position_ids = torch.full((1, 1), pos, dtype=torch.long)
            cos, sin = model.rope.get_rot_mats(position_ids)
            ttnn.synchronize_device(device)
            timings["rope"] = time.perf_counter() - t0

            # Per-layer
            from models.demos.blackhole.qwen3_5_9b.tt.qwen35_decoder import rms_norm_ttnn
            layer_timings = []
            for i, layer in enumerate(model.layers):
                ttnn.synchronize_device(device)
                tl0 = time.perf_counter()

                # Full layer forward
                x = layer.forward(x, cos=cos, sin=sin, mode="decode")

                ttnn.synchronize_device(device)
                tl1 = time.perf_counter()
                layer_type = "attn" if layer.is_full_attention else "dnet"
                layer_timings.append({"idx": i, "type": layer_type, "ms": (tl1 - tl0) * 1000})

            # Final norm + LM head
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            x = rms_norm_ttnn(x, model.norm_weight, eps=model.norm_eps)
            logits = ttnn.linear(x, model.lm_head_weight)
            ttnn.deallocate(x)
            ttnn.synchronize_device(device)
            timings["norm_lmhead"] = time.perf_counter() - t0

            ttnn.synchronize_device(device)
            timings["total"] = time.perf_counter() - t_total_start
            timings["layers"] = layer_timings

            logits_torch = ttnn.to_torch(logits).squeeze()
            next_token = logits_torch.argmax().item()

            all_timings.append(timings)

        # Aggregate and print
        logger.info("=" * 80)
        logger.info("DECODE PROFILING RESULTS (averaged over {} steps)".format(num_steps))
        logger.info("=" * 80)

        avg_embed = sum(t["embedding"] for t in all_timings) / num_steps * 1000
        avg_rope = sum(t["rope"] for t in all_timings) / num_steps * 1000
        avg_norm_lmhead = sum(t["norm_lmhead"] for t in all_timings) / num_steps * 1000
        avg_total = sum(t["total"] for t in all_timings) / num_steps * 1000

        logger.info(f"  Embedding:        {avg_embed:.2f} ms")
        logger.info(f"  RoPE:             {avg_rope:.2f} ms")

        # Per-layer averages
        n_layers = len(all_timings[0]["layers"])
        dnet_times = []
        attn_times = []
        for li in range(n_layers):
            avg_ms = sum(t["layers"][li]["ms"] for t in all_timings) / num_steps
            ltype = all_timings[0]["layers"][li]["type"]
            if ltype == "dnet":
                dnet_times.append(avg_ms)
            else:
                attn_times.append(avg_ms)

        if dnet_times:
            logger.info(f"  DeltaNet (24):    {sum(dnet_times):.2f} ms total, {sum(dnet_times)/len(dnet_times):.2f} ms avg/layer")
        if attn_times:
            logger.info(f"  Attention (8):    {sum(attn_times):.2f} ms total, {sum(attn_times)/len(attn_times):.2f} ms avg/layer")
        logger.info(f"  Norm + LM Head:   {avg_norm_lmhead:.2f} ms")
        logger.info(f"  TOTAL:            {avg_total:.2f} ms ({1000/avg_total:.1f} tok/s)")
        logger.info("")

        # Print per-layer detail
        logger.info("  Per-layer breakdown:")
        for li in range(n_layers):
            avg_ms = sum(t["layers"][li]["ms"] for t in all_timings) / num_steps
            ltype = all_timings[0]["layers"][li]["type"]
            bar = "#" * int(avg_ms / 0.2)
            logger.info(f"    Layer {li:2d} ({ltype}): {avg_ms:6.2f} ms {bar}")

        logger.info("=" * 80)

        # Assertions
        assert avg_total < 200, f"Decode too slow: {avg_total:.1f}ms (expected < 200ms)"
        assert avg_total > 0, "Decode timing is zero — profiling broken"

        # Return data for other tests to use
        return {
            "embedding_ms": avg_embed,
            "rope_ms": avg_rope,
            "deltanet_total_ms": sum(dnet_times),
            "attention_total_ms": sum(attn_times),
            "norm_lmhead_ms": avg_norm_lmhead,
            "total_ms": avg_total,
            "tok_per_sec": 1000 / avg_total,
        }
```

- [ ] **Step 2: Run the profiling test**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_decode_profiling.py -v -s --timeout=300
```

Expected: PASS with detailed timing breakdown printed. This tells us where time is spent and guides all subsequent optimization work.

- [ ] **Step 3: Record baseline numbers**

Record the profiling results in a comment at the top of the test file for future comparison. Note which layer type dominates.

---

## Task 2: Build Regression Guard Test

**Files:**
- Create: `models/demos/blackhole/qwen3_5_9b/tests/test_optimization_regression.py`

- [ ] **Step 1: Write the regression test**

```python
# models/demos/blackhole/qwen3_5_9b/tests/test_optimization_regression.py
"""Throughput regression guard for Qwen3.5-9B decode.

Asserts that decode throughput stays above baseline. Update BASELINE_TOK_S as optimizations land.
Run: pytest models/demos/blackhole/qwen3_5_9b/tests/test_optimization_regression.py -v -s --timeout=300
"""
import time
import pytest
from loguru import logger
import torch
import ttnn

from models.demos.blackhole.qwen3_5_9b.tt.qwen35_model import Qwen35Model
from models.common.utility_functions import run_for_blackhole

CHECKPOINT_DIR = "/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"
pytestmark = run_for_blackhole()

# Update this as optimizations land
BASELINE_TOK_S = 7.0  # Conservative: below current 8.3 to avoid flaky failures


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def model(device):
    return Qwen35Model.from_pretrained(device, CHECKPOINT_DIR, max_batch_size=1, max_seq_len=2048)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(CHECKPOINT_DIR)


class TestThroughputRegression:
    def test_decode_throughput_above_baseline(self, model, tokenizer, device):
        """Decode 20 tokens, assert throughput >= baseline."""
        prompt = "Explain quantum computing"
        inputs = tokenizer(prompt, return_tensors="pt")
        token_ids = inputs["input_ids"]
        T = token_ids.shape[1]

        model.reset_state(batch_size=1)
        logits = model.prefill(token_ids)
        ttnn.synchronize_device(device)
        logits_torch = ttnn.to_torch(logits).squeeze()
        next_token = logits_torch.argmax().item()

        # Warmup 2 steps
        for i in range(2):
            next_input = torch.tensor([[next_token]], dtype=torch.long)
            logits = model.decode(next_input, current_pos=T + i)
            ttnn.synchronize_device(device)
            logits_torch = ttnn.to_torch(logits).squeeze()
            next_token = logits_torch.argmax().item()

        # Measure 20 steps
        decode_times = []
        for i in range(20):
            next_input = torch.tensor([[next_token]], dtype=torch.long)
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            logits = model.decode(next_input, current_pos=T + 2 + i)
            ttnn.synchronize_device(device)
            dt = time.perf_counter() - t0
            decode_times.append(dt)
            logits_torch = ttnn.to_torch(logits).squeeze()
            next_token = logits_torch.argmax().item()

        avg_ms = sum(decode_times) / len(decode_times) * 1000
        tok_s = 1000 / avg_ms
        logger.info(f"[PERF] Decode: {avg_ms:.1f} ms/token, {tok_s:.2f} tok/s (baseline: {BASELINE_TOK_S})")

        assert tok_s >= BASELINE_TOK_S, (
            f"Throughput regression: {tok_s:.2f} tok/s < {BASELINE_TOK_S} baseline"
        )
```

- [ ] **Step 2: Run the regression test**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_optimization_regression.py -v -s --timeout=300
```

Expected: PASS with throughput above 7.0 tok/s.

---

## Task 3: Optimize Gated Attention — Pre-Allocated KV Cache

Switch from concat-based KV cache to pre-allocated cache with index writes. The experimental op already supports this via `kv_cache_key`/`kv_cache_value`/`cache_pos` parameters — the wrapper just isn't using them.

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_attention.py`

- [ ] **Step 1: Run existing e2e test to confirm baseline**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py::TestDecodeLoop::test_generate_tokens -v -s --timeout=300
```

Expected: PASS with current throughput numbers.

- [ ] **Step 2: Modify wrapper to use pre-allocated KV cache**

Replace `qwen35_gated_attention.py` with pre-allocated cache mode. The experimental op at `ttnn_gated_attention.py:171-187` already has this path via `kv_cache_key`/`kv_cache_value`/`cache_pos` parameters.

**Important notes:**
- `ttnn.copy` into a sliced view may not be supported. If it fails, first try the approach and if `ttnn.copy` errors, fall back to keeping the concat-based approach but investigate `ttnn.tensor.update_cache` or similar scatter-write ops.
- The pre-allocated cache read at line 180 (`kv_cache_key[:, :, :valid_len, :]`) produces a dynamic-length tensor. This is acceptable for Phase 2 (eliminates concat overhead) but will need to be addressed for trace capture in Phase 3 (pass full cache + use max_seq_len attention mask instead).
- `reset_cache` should just reset `cache_pos = 0` without re-allocating — old values are overwritten naturally.

```python
# In __init__, after loading weights:
self.max_seq_len = args.max_seq_len
self.max_batch_size = args.max_batch_size
self.cache_pos = 0
# Pre-allocate KV cache on device (ROW_MAJOR for slice writes)
import torch
kv_shape = [args.max_batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim]
self.kv_cache_key = ttnn.from_torch(
    torch.zeros(kv_shape, dtype=torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
self.kv_cache_value = ttnn.from_torch(
    torch.zeros(kv_shape, dtype=torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

```python
# In forward:
def forward(self, x, cos, sin):
    T = x.shape[1]
    output, new_key, new_value = gated_attention_forward_ttnn(
        hidden_states=x,
        q_proj_weight=self.q_proj_weight,
        k_proj_weight=self.k_proj_weight,
        v_proj_weight=self.v_proj_weight,
        o_proj_weight=self.o_proj_weight,
        q_norm_weight=self.q_norm_weight,
        k_norm_weight=self.k_norm_weight,
        cos=cos,
        sin=sin,
        num_attention_heads=self.num_heads,
        num_key_value_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        device=self.device,
        norm_eps=self.norm_eps,
        compute_kernel_config=self.compute_kernel_config,
        use_optimized_concat=True,
        kv_cache_key=self.kv_cache_key,
        kv_cache_value=self.kv_cache_value,
        cache_pos=self.cache_pos,
    )
    self.cache_pos += T
    return output
```

```python
# In reset_cache — just reset position, don't re-allocate:
def reset_cache(self):
    self.cache_pos = 0
```

**Fallback:** If `ttnn.copy` into a sliced view fails at runtime, revert to the concat-based approach and skip this task. The concat approach still works correctly — it's just slower.

- [ ] **Step 3: Run e2e test to verify no regression**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py -v -s --timeout=300
```

Expected: PASS. Generated text should be coherent. Check that PCC/output quality is maintained.

- [ ] **Step 4: Run profiling test to measure improvement**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_decode_profiling.py -v -s --timeout=300
```

Expected: Attention layer times should decrease (no more concat overhead per step).

---

## Task 4: Optimize Gated Attention — Add Memory Configs to Linears

The `gated_attention_forward_ttnn` function doesn't pass `memory_config` to any `ttnn.linear` call. Adding L1 memory config should reduce DRAM round-trips.

**Files:**
- Modify: `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py`

- [ ] **Step 1: Run single-layer test as baseline**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_single_layer.py::TestGatedAttentionLayer -v -s --timeout=300
```

Expected: PASS.

- [ ] **Step 2: Add memory_config to linear calls in gated_attention_forward_ttnn**

In `ttnn_gated_attention.py`, add a `memory_config` parameter to `gated_attention_forward_ttnn` (default `None`) and thread it through. The Qwen wrapper will pass `ttnn.L1_MEMORY_CONFIG` for decode (T=1) and `None` for prefill (to avoid L1 OOM with larger T).

Add parameter to signature:
```python
def gated_attention_forward_ttnn(
    ...,
    memory_config=None,  # New: L1 for decode, None/DRAM for prefill
):
```

Then use it at lines 145, 157, 163, 229:
```python
mc = memory_config

# Line 145: Q projection
qg = ttnn.linear(hidden_states, q_proj_weight, compute_kernel_config=ckc, memory_config=mc)

# Line 157: K projection
key_states = ttnn.linear(hidden_states, k_proj_weight, compute_kernel_config=ckc, memory_config=mc)

# Line 163: V projection
value_states = ttnn.linear(hidden_states, v_proj_weight, compute_kernel_config=ckc, memory_config=mc)

# Line 229: Output projection
attn_output = ttnn.linear(attn_output, o_proj_weight, compute_kernel_config=ckc, memory_config=mc)
```

In `qwen35_gated_attention.py`, pass `memory_config=ttnn.L1_MEMORY_CONFIG` for decode:
```python
# Determine memory config based on sequence length
T = x.shape[1]
mc = ttnn.L1_MEMORY_CONFIG if T == 1 else None

output, new_key, new_value = gated_attention_forward_ttnn(
    ...,
    memory_config=mc,
)
```

- [ ] **Step 3: Run single-layer and e2e tests**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_single_layer.py::TestGatedAttentionLayer -v -s --timeout=300 && pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py::TestDecodeLoop -v -s --timeout=300
```

Expected: Both PASS. If L1 OOM occurs on any linear, fall back to `ttnn.DRAM_MEMORY_CONFIG` for that specific op.

- [ ] **Step 4: Run profiling to measure improvement**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_decode_profiling.py -v -s --timeout=300
```

Expected: Attention layer times decrease.

---

## Task 5: Optimize Gated Attention — Fuse Q/K RMSNorm

The `rms_norm_zero_centered_ttnn` in `ttnn_gated_attention.py` always uses the manual 6-op fallback. Replace it with the fused `ttnn.rms_norm` path (with pre-offset weights, same pattern as decoder norms).

**Files:**
- Modify: `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py`
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_attention.py`

- [ ] **Step 1: Pre-offset Q/K norm weights in wrapper**

In `qwen35_gated_attention.py`, change `load_1d` for norm weights to pre-add +1 (same as decoder norms):

```python
# Replace load_1d calls for norm weights (lines 61-62):
def load_norm_1d(name):
    """Load 1D norm weight with +1 offset for zero-centered RMSNorm."""
    t = state_dict[f"{prefix}.{name}"] + 1.0  # Pre-offset
    return ttnn.as_tensor(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=weight_cache_path / f"{prefix}.{name}" if weight_cache_path else None,
    )

self.q_norm_weight = load_norm_1d("q_norm.weight")
self.k_norm_weight = load_norm_1d("k_norm.weight")
```

- [ ] **Step 2: Do NOT modify the shared `rms_norm_zero_centered_ttnn` — it has other callers**

Instead, modify `gated_attention_forward_ttnn` to accept a `norm_weights_pre_offset` boolean parameter (default `False`). When `True`, use the fused `ttnn.rms_norm` directly instead of calling `rms_norm_zero_centered_ttnn`:

In `ttnn_gated_attention.py`, add the parameter to `gated_attention_forward_ttnn` signature and use it at lines 153 and 159:

```python
def gated_attention_forward_ttnn(
    ...,
    norm_weights_pre_offset=False,  # New parameter
):
    ...
    # Line 153: Q norm
    if norm_weights_pre_offset:
        try:
            query_states = ttnn.rms_norm(query_states, weight=q_norm_weight, epsilon=norm_eps)
        except Exception:
            query_states = rms_norm_zero_centered_ttnn(query_states, q_norm_weight, eps=norm_eps)
    else:
        query_states = rms_norm_zero_centered_ttnn(query_states, q_norm_weight, eps=norm_eps)

    # Line 159: K norm (same pattern)
    if norm_weights_pre_offset:
        try:
            key_states = ttnn.rms_norm(key_states, weight=k_norm_weight, epsilon=norm_eps)
        except Exception:
            key_states = rms_norm_zero_centered_ttnn(key_states, k_norm_weight, eps=norm_eps)
    else:
        key_states = rms_norm_zero_centered_ttnn(key_states, k_norm_weight, eps=norm_eps)
```

Then in the Qwen wrapper (`qwen35_gated_attention.py`), pass `norm_weights_pre_offset=True` in the `forward` call.

- [ ] **Step 3: Run single-layer test**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_single_layer.py::TestGatedAttentionLayer -v -s --timeout=300
```

Expected: PASS. If fused norm works, we save 16 × (6→1 op) = 80 fewer ops per decode step.

- [ ] **Step 4: Run e2e test**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py -v -s --timeout=300
```

Expected: PASS with coherent text.

- [ ] **Step 5: Run profiling to measure improvement**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_decode_profiling.py -v -s --timeout=300
```

---

## Task 6: Optimize DeltaNet — Memory Configs for Decode Path

Add explicit L1 memory configs to the DeltaNet recurrent decode path. Currently most ops pass `memory_config=None`.

**Files:**
- Modify: `models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py`

- [ ] **Step 1: Run single-layer DeltaNet test as baseline**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_single_layer.py::TestDeltaNetLayer -v -s --timeout=300
```

Expected: PASS.

- [ ] **Step 2: Add L1 memory config to recurrent_gated_delta_rule_decode_ttnn**

In `ttnn_delta_rule_ops.py`, function `recurrent_gated_delta_rule_decode_ttnn` (line 471), replace `memory_config=None` with `memory_config=ttnn.L1_MEMORY_CONFIG` for small tensor ops (reshapes, element-wise). Keep `ttnn.DRAM_MEMORY_CONFIG` for the large state tensor `h` (which is `[B, H, K, V]` = `[1, 32, 128, 128]`).

Key changes in `recurrent_gated_delta_rule_decode_ttnn`:
```python
# Line 515: q scale
q = ttnn.multiply(q, scale, memory_config=ttnn.L1_MEMORY_CONFIG)

# Lines 518-522: reshapes — use L1 for small vectors
q_t = ttnn.reshape(q, [B, H, K], memory_config=ttnn.L1_MEMORY_CONFIG)
k_t = ttnn.reshape(k, [B, H, K], memory_config=ttnn.L1_MEMORY_CONFIG)
v_t = ttnn.reshape(v, [B, H, V], memory_config=ttnn.L1_MEMORY_CONFIG)
beta_t = ttnn.reshape(beta, [B, H], memory_config=ttnn.L1_MEMORY_CONFIG)
g_t = ttnn.reshape(g, [B, H], memory_config=ttnn.L1_MEMORY_CONFIG)

# Line 525: decay
decay_t = ttnn.exp(g_t, memory_config=ttnn.L1_MEMORY_CONFIG)
```

Similarly in `l2_norm_ttnn` (line 313), add L1 memory config:
```python
def l2_norm_ttnn(x, dim=-1, eps=1e-6):
    mc = ttnn.L1_MEMORY_CONFIG
    x_sq = ttnn.multiply(x, x, memory_config=mc)
    norm_sq = ttnn.sum(x_sq, dim=dim, keepdim=True, memory_config=mc)
    inv_norm = ttnn.rsqrt(ttnn.add(norm_sq, eps, memory_config=mc), memory_config=mc)
    return ttnn.multiply(x, inv_norm, memory_config=mc)
```

And in `fused_decay_and_write_ttnn` (line 321), use L1 for broadcast scalars only. **Do NOT** put `k_col` and `d_row` in L1 — they are immediately moved to DRAM at lines 358-359 for the matmul. Adding L1 would create an unnecessary extra memory move.
```python
# Lines 344-348: broadcast scalars in L1 (tiny tensors)
decay = ttnn.typecast(decay_t, ttnn.bfloat16)
decay = ttnn.reshape(decay, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
beta_expanded = ttnn.reshape(beta_t, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
# k_col and d_row: leave as memory_config=None — they go to DRAM for matmul
```

- [ ] **Step 3: Run single-layer and e2e tests**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_single_layer.py::TestDeltaNetLayer -v -s --timeout=300 && pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py -v -s --timeout=300
```

Expected: PASS. If any L1 OOM, revert that specific op to `None` (DRAM).

- [ ] **Step 4: Run profiling to measure improvement**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_decode_profiling.py -v -s --timeout=300
```

Expected: DeltaNet layer times decrease.

---

## Task 7: Optimize DeltaNet — Reduce Redundant to_layout Calls

The decode path has several `ttnn.to_layout(x, ttnn.TILE_LAYOUT)` calls on tensors that are already in TILE_LAYOUT. Each is a no-op but still incurs dispatch overhead.

**Files:**
- Modify: `models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py`

- [ ] **Step 1: Remove redundant to_layout calls in recurrent decode path**

In `recurrent_gated_delta_rule_decode_ttnn` (starting line 471), identify and remove redundant `to_layout` calls. The key insight: during decode, `h` (recurrent state) is always returned as TILE_LAYOUT from the previous step's `fused_decay_and_write_ttnn`. So line 533 `h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)` is redundant after the first call.

Concrete changes:
```python
# Line 533: Remove — h is already TILE_LAYOUT from fused_decay_and_write_ttnn
# was: h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)
# Line 534: Keep — h still needs to be in DRAM for matmul
h = ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG)

# Lines 551-553: These ARE needed — reshape may change layout
# Keep as-is: q_t = ttnn.to_layout(q_t, ttnn.TILE_LAYOUT) etc.

# Lines 555-557: k_row reshape + to_layout + to_memory_config
# Combine into one reshape with explicit memory config:
k_row = ttnn.reshape(k_t, [B, H, 1, K], memory_config=ttnn.DRAM_MEMORY_CONFIG)
k_row = ttnn.to_layout(k_row, ttnn.TILE_LAYOUT)
# Remove the separate to_memory_config call (line 557)

# Lines 569-571: same pattern for q_row
q_row = ttnn.reshape(q_t, [B, H, 1, K], memory_config=ttnn.DRAM_MEMORY_CONFIG)
q_row = ttnn.to_layout(q_row, ttnn.TILE_LAYOUT)
# Remove the separate to_memory_config call (line 571)
```

**Note:** Only remove `to_layout` if you can verify the input is already in the right layout. When in doubt, keep it — a redundant `to_layout` is a no-op on device but the removal of a needed one causes incorrect results.

- [ ] **Step 2: Run tests**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_single_layer.py -v -s --timeout=300 && pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py -v -s --timeout=300
```

Expected: PASS.

- [ ] **Step 3: Run profiling**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_decode_profiling.py -v -s --timeout=300
```

---

## Task 8: Optimize Decoder — L1 Memory for Norm Outputs

The decoder's RMSNorm outputs currently use default memory config. Adding L1 should help since the result is immediately consumed by the next op.

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_decoder.py`

- [ ] **Step 1: Add memory_config to rms_norm_ttnn**

In `qwen35_decoder.py`, modify `rms_norm_ttnn` to accept and pass through `memory_config`:

```python
def rms_norm_ttnn(x, weight, eps=1e-6, memory_config=None):
    try:
        return ttnn.rms_norm(x, weight=weight, epsilon=eps, memory_config=memory_config)
    except Exception:
        mc = memory_config
        x_sq = ttnn.multiply(x, x, memory_config=mc)
        variance = ttnn.mean(x_sq, dim=-1, keepdim=True, memory_config=mc)
        inv_rms = ttnn.rsqrt(ttnn.add(variance, eps), memory_config=mc)
        x_normed = ttnn.multiply(x, inv_rms, memory_config=mc)
        return ttnn.multiply(x_normed, weight, memory_config=mc)
```

In `forward` method, pass L1 memory config to norms:

```python
def forward(self, x, cos=None, sin=None, mode="decode", chunk_size=64):
    mc = ttnn.L1_MEMORY_CONFIG if mode == "decode" else None
    attn_input = rms_norm_ttnn(x, self.attention_norm_weight, eps=self.norm_eps, memory_config=mc)
    # ... (rest unchanged)
    ff_input = rms_norm_ttnn(h, self.ff_norm_weight, eps=self.norm_eps, memory_config=mc)
```

- [ ] **Step 2: Run tests**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_single_layer.py -v -s --timeout=300 && pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py -v -s --timeout=300
```

Expected: PASS.

- [ ] **Step 3: Run profiling**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_decode_profiling.py -v -s --timeout=300
```

---

## Task 8b: Verify MLP Optimizations Already Applied

The spec (Phase 2b) calls for MLP optimization. The current `qwen35_mlp.py` already has:
- `activation="silu"` on gate_proj (fused SiLU)
- `memory_config=ttnn.L1_MEMORY_CONFIG` on all linears
- `WormholeComputeKernelConfig(LoFi, fp32_dest_acc_en=True)` on all matmuls

**Files:** None (verification only)

- [ ] **Step 1: Verify MLP code**

Read `models/demos/blackhole/qwen3_5_9b/tt/qwen35_mlp.py` and confirm all three optimizations are present. They are (lines 45-51). No changes needed.

- [ ] **Step 2: Verify via profiling that MLP is not the bottleneck**

Check the profiling output from Task 1. MLP time should be proportional (32 layers × ~1ms each or less). If MLP dominates, investigate further.

---

## Task 8c: Check for Native ttnn.rotary_embedding on Blackhole

The spec asks to check if Blackhole has a native RoPE op that could replace the manual `rotate_half_ttnn` (which creates intermediate tensors via slice+neg+concat).

**Files:** None (investigation only)

- [ ] **Step 1: Search for rotary_embedding in ttnn API**

```bash
python -c "import ttnn; print([x for x in dir(ttnn) if 'rotar' in x.lower()])" 2>/dev/null
python -c "import ttnn; print([x for x in dir(ttnn.experimental) if 'rotar' in x.lower()])" 2>/dev/null
python -c "import ttnn; print([x for x in dir(ttnn.transformer) if 'rotar' in x.lower()])" 2>/dev/null
```

- [ ] **Step 2: If found, evaluate whether to use it**

If a native `ttnn.rotary_embedding` exists, it could replace `apply_rotary_pos_emb_ttnn` (which does 8+ ops: 2× rotate_half + 4× multiply + 2× add + 2× concat). However, this is only called for 8 attention layers and only on Q/K, so the impact may be small. Add to backlog if impact is marginal.

---

## Task 9: Checkpoint — Measure Cumulative Improvement

After all Phase 2 optimizations, run the full test suite and profiling to measure total improvement.

**Files:** None (measurement only)

- [ ] **Step 1: Run full test suite**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_single_layer.py -v -s --timeout=300
pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py -v -s --timeout=300
```

Expected: All PASS.

- [ ] **Step 2: Run profiling**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_decode_profiling.py -v -s --timeout=300
```

- [ ] **Step 3: Run regression test**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_optimization_regression.py -v -s --timeout=300
```

- [ ] **Step 4: Update baseline**

If throughput has improved, update `BASELINE_TOK_S` in `test_optimization_regression.py` to the new floor (set it ~10% below measured to avoid flakiness).

- [ ] **Step 5: Decide next steps**

Based on profiling results:
- If dispatch overhead dominates → proceed to Task 10 (trace capture)
- If specific ops dominate → investigate further targeted optimization
- If throughput is already satisfactory → stop here

---

## Task 10: Trace Capture — Prepare Decode Path (Phase 3)

Create a trace-compatible decode path with fixed shapes and no host-side logic.

**Prerequisites:** Task 9 complete, profiling shows dispatch overhead is significant.

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py`

- [ ] **Step 1: Verify RoPE decode path is device-only**

Check that the decode RoPE path (T=1, B=1) in `qwen35_rope.py:80-84` only does device slicing (no `ttnn.from_torch`). Currently it does `self.cos_device[:, pos:pos+1, :]` which is a device-side slice — this is trace-compatible. Verify by reading the code.

- [ ] **Step 2: Create forward_decode method**

Add a `forward_decode` method to `Qwen35Model` that hardcodes T=1 behavior with no Python conditionals:

```python
def forward_decode(self, x, cos, sin):
    """Trace-compatible decode: fixed shape [B=1, T=1, D=4096], no Python conditionals."""
    for layer in self.layers:
        x = layer.forward(x, cos=cos, sin=sin, mode="decode")
    x = rms_norm_ttnn(x, self.norm_weight, eps=self.norm_eps)
    logits = ttnn.linear(x, self.lm_head_weight)
    ttnn.deallocate(x)
    return logits
```

- [ ] **Step 3: Add trace capture/replay to decode**

**Critical: RoPE position handling.** A trace captures a fixed computation graph — you cannot change which slice of the RoPE table is taken after capture. Solution: pre-allocate cos/sin buffers and `ttnn.copy` the position-specific values into them *before* executing the trace.

Add trace infrastructure to `Qwen35Model`:

```python
def capture_decode_trace(self, device):
    """Capture a trace for the decode path after warmup."""
    import torch

    # Pre-allocate fixed buffers for trace inputs
    self._trace_input = ttnn.from_torch(
        torch.zeros(1, 1, self.args.dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )
    # Pre-allocate cos/sin buffers (fixed shape [1, 1, rope_head_dim])
    self._trace_cos = ttnn.from_torch(
        torch.zeros(1, 1, self.args.rope_head_dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )
    self._trace_sin = ttnn.from_torch(
        torch.zeros(1, 1, self.args.rope_head_dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )

    # Capture: trace reads from these fixed buffers
    self._trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    self._trace_output = self.forward_decode(self._trace_input, self._trace_cos, self._trace_sin)
    ttnn.end_trace_capture(device, self._trace_id, cq_id=0)

def decode_traced(self, token_ids, current_pos):
    """Execute traced decode: write inputs into pre-allocated buffers, replay trace."""
    # Embed token (outside trace — this is a lookup, not a fixed compute)
    token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
    x = ttnn.embedding(token_ids_ttnn, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
    ttnn.deallocate(token_ids_ttnn)

    # Write embedding into trace input buffer
    ttnn.copy(x, self._trace_input)

    # Write position-specific RoPE values into trace cos/sin buffers
    cos_pos = self.rope.cos_device[:, current_pos:current_pos+1, :]
    sin_pos = self.rope.sin_device[:, current_pos:current_pos+1, :]
    ttnn.copy(cos_pos, self._trace_cos)
    ttnn.copy(sin_pos, self._trace_sin)

    # Replay the captured trace
    ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)
    return self._trace_output
```

**Note:** The embedding lookup and RoPE copy happen *outside* the trace (before `execute_trace`). The trace only captures the compute graph from input buffer → logits. This means embedding + RoPE copy overhead is not eliminated by trace, but they are small relative to the 32-layer forward pass.

- [ ] **Step 4: Create test_traced_decode.py**

```python
# models/demos/blackhole/qwen3_5_9b/tests/test_traced_decode.py
"""Trace capture validation: PCC between traced and non-traced decode.

Run: pytest models/demos/blackhole/qwen3_5_9b/tests/test_traced_decode.py -v -s --timeout=300
"""
import time
import pytest
from loguru import logger
import torch
import ttnn

from models.demos.blackhole.qwen3_5_9b.tt.qwen35_model import Qwen35Model
from models.common.utility_functions import run_for_blackhole

CHECKPOINT_DIR = "/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"
pytestmark = run_for_blackhole()
PCC_THRESHOLD = 0.999


def compute_pcc(a, b):
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-8)).item()


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def model(device):
    return Qwen35Model.from_pretrained(device, CHECKPOINT_DIR, max_batch_size=1, max_seq_len=2048)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(CHECKPOINT_DIR)


class TestTracedDecode:
    def test_traced_vs_non_traced_pcc(self, model, tokenizer, device):
        """Compare traced and non-traced decode outputs."""
        prompt = "Hello world"
        inputs = tokenizer(prompt, return_tensors="pt")
        token_ids = inputs["input_ids"]
        T = token_ids.shape[1]

        # Run 5 steps non-traced, collect logits
        model.reset_state(batch_size=1)
        logits = model.prefill(token_ids)
        ttnn.synchronize_device(device)
        next_token = ttnn.to_torch(logits).squeeze().argmax().item()

        non_traced_logits = []
        for i in range(5):
            inp = torch.tensor([[next_token]], dtype=torch.long)
            logits = model.decode(inp, current_pos=T + i)
            ttnn.synchronize_device(device)
            lt = ttnn.to_torch(logits).squeeze()
            non_traced_logits.append(lt)
            next_token = lt.argmax().item()

        # Capture trace (if supported)
        try:
            model.capture_decode_trace(device)
        except Exception as e:
            pytest.skip(f"Trace capture not supported: {e}")

        # Run 5 steps traced from same state
        model.reset_state(batch_size=1)
        logits = model.prefill(token_ids)
        ttnn.synchronize_device(device)
        next_token = ttnn.to_torch(logits).squeeze().argmax().item()

        traced_logits = []
        traced_times = []
        for i in range(5):
            inp = torch.tensor([[next_token]], dtype=torch.long)
            t0 = time.perf_counter()
            logits = model.decode_traced(inp, current_pos=T + i)
            ttnn.synchronize_device(device)
            traced_times.append(time.perf_counter() - t0)
            lt = ttnn.to_torch(logits).squeeze()
            traced_logits.append(lt)
            next_token = lt.argmax().item()

        # Compare PCC
        for i in range(5):
            pcc = compute_pcc(non_traced_logits[i], traced_logits[i])
            logger.info(f"  Step {i}: PCC = {pcc:.6f}")
            assert pcc > PCC_THRESHOLD, f"PCC regression at step {i}: {pcc:.6f}"

        avg_traced_ms = sum(traced_times) / len(traced_times) * 1000
        logger.info(f"  Traced decode: {avg_traced_ms:.1f} ms/token")
```

- [ ] **Step 5: Test trace capture**

Try capturing and replaying the trace. This step is exploratory — trace may fail if experimental ops use incompatible operations. If it fails:
1. Note which op caused the failure
2. Investigate if it can be made trace-compatible
3. If not, fall back to non-traced decode (Phase 2 gains are still valuable)

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_traced_decode.py -v -s --timeout=300
```

- [ ] **Step 5: If trace works, measure improvement**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_decode_profiling.py -v -s --timeout=300
pytest models/demos/blackhole/qwen3_5_9b/tests/test_optimization_regression.py -v -s --timeout=300
```

Expected: 2-5x improvement if trace capture eliminates dispatch overhead.

---

## Task 11: Final Measurement and Cleanup

- [ ] **Step 1: Run full test suite one final time**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/ -v -s --timeout=300
```

- [ ] **Step 2: Run profiling and record final numbers**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_decode_profiling.py -v -s --timeout=300
```

- [ ] **Step 3: Update STATUS.md with new performance numbers**

Update the performance table in `models/demos/blackhole/qwen3_5_9b/STATUS.md` with the new decode throughput, noting which optimizations contributed.
