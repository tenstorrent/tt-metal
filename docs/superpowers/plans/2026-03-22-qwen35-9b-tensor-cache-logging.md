# Qwen3.5-9B Tensor Cache & Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add disk-based tensor caching to eliminate repeated weight conversion on model load, and replace all `print()` with `loguru.logger` to match codebase conventions.

**Architecture:** Weight tensors loaded via `ttnn.from_torch()` are replaced with `ttnn.as_tensor()` + `cache_file_name`, which transparently caches converted tensors to `{checkpoint_dir}/tensor_cache_bfp8/`. Logging is migrated to loguru. Test fixtures get `run_for_blackhole()` markers.

**Tech Stack:** ttnn, loguru, pytest

**Spec:** `docs/superpowers/specs/2026-03-22-qwen35-9b-tensor-cache-logging-design.md`

**Important:** Do NOT commit anything. All changes stay local.

---

### Task 1: Add `weight_cache_path()` to `Qwen35ModelArgs`

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/model_config.py:79-84`

- [ ] **Step 1: Add the `weight_cache_path` method**

Add after line 84 (after `is_deltanet_layer`):

```python
def weight_cache_path(self, dtype=None):
    """Return cache directory path for converted weight tensors.

    Directory is created automatically by ttnn.as_tensor when first cache file is written.
    """
    if dtype is None:
        dtype = self.weight_dtype
    import ttnn
    if dtype == ttnn.bfloat8_b:
        suffix = "tensor_cache_bfp8"
    else:
        suffix = "tensor_cache_bf16"
    return Path(self.checkpoint_dir) / suffix
```

Note: `Path` is already imported at the top of the file (line 12).

- [ ] **Step 2: Verify import exists**

Confirm `from pathlib import Path` is already on line 12. No change needed.

---

### Task 2: Add tensor caching to `Qwen35MLP`

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_mlp.py:16-33`

- [ ] **Step 1: Add `weight_cache_path` parameter to `__init__`**

Change the signature on line 16 from:
```python
def __init__(self, args, state_dict, layer_num, device):
```
to:
```python
def __init__(self, args, state_dict, layer_num, device, weight_cache_path=None):
```

- [ ] **Step 2: Replace `ttnn.from_torch` with `ttnn.as_tensor` in `load_weight`**

Replace lines 26-29:
```python
def load_weight(name):
    """Load 2D weight, transpose to [in, out] for ttnn.linear."""
    t = state_dict[f"{prefix}.{name}"].T.contiguous()
    return ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
```

With:
```python
def load_weight(name):
    """Load 2D weight, transpose to [in, out] for ttnn.linear."""
    t = state_dict[f"{prefix}.{name}"].T.contiguous()
    return ttnn.as_tensor(
        t,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=weight_cache_path / f"{prefix}.{name}" if weight_cache_path else None,
    )
```

---

### Task 3: Add tensor caching to `Qwen35GatedAttention`

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_attention.py:22-48`

- [ ] **Step 1: Add `weight_cache_path` parameter to `__init__`**

Change line 22 from:
```python
def __init__(self, args, state_dict, layer_num, device):
```
to:
```python
def __init__(self, args, state_dict, layer_num, device, weight_cache_path=None):
```

- [ ] **Step 2: Replace `ttnn.from_torch` with `ttnn.as_tensor` in `load_weight`**

Replace lines 33-36:
```python
def load_weight(name):
    """Load 2D weight, transpose to [in, out] for ttnn.linear."""
    t = state_dict[f"{prefix}.{name}"].T.contiguous()
    return ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
```

With:
```python
def load_weight(name):
    """Load 2D weight, transpose to [in, out] for ttnn.linear."""
    t = state_dict[f"{prefix}.{name}"].T.contiguous()
    return ttnn.as_tensor(
        t,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=weight_cache_path / f"{prefix}.{name}" if weight_cache_path else None,
    )
```

- [ ] **Step 3: Replace `ttnn.from_torch` with `ttnn.as_tensor` in `load_1d`**

Replace lines 38-41:
```python
def load_1d(name):
    """Load 1D param (norm weight) — TILE_LAYOUT on device."""
    t = state_dict[f"{prefix}.{name}"]
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
```

With:
```python
def load_1d(name):
    """Load 1D param (norm weight) — TILE_LAYOUT on device."""
    t = state_dict[f"{prefix}.{name}"]
    return ttnn.as_tensor(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=weight_cache_path / f"{prefix}.{name}" if weight_cache_path else None,
    )
```

---

### Task 4: Add tensor caching to `Qwen35GatedDeltaNet`

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py:39-106`

- [ ] **Step 1: Add `weight_cache_path` parameter to `__init__`**

Change line 39 from:
```python
def __init__(self, args, state_dict, layer_num, device):
```
to:
```python
def __init__(self, args, state_dict, layer_num, device, weight_cache_path=None):
```

- [ ] **Step 2: Replace `ttnn.from_torch` with `ttnn.as_tensor` in `load_weight_2d`**

Replace lines 58-61:
```python
def load_weight_2d(name):
    """Load 2D weight, transpose to [in, out] for ttnn.linear."""
    t = state_dict[f"{prefix}.{name}"].T.contiguous()
    return ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
```

With:
```python
def load_weight_2d(name):
    """Load 2D weight, transpose to [in, out] for ttnn.linear."""
    t = state_dict[f"{prefix}.{name}"].T.contiguous()
    return ttnn.as_tensor(
        t,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=weight_cache_path / f"{prefix}.{name}" if weight_cache_path else None,
    )
```

- [ ] **Step 3: Replace `ttnn.from_torch` with `ttnn.as_tensor` in `load_1d`**

Replace lines 68-71:
```python
def load_1d(name):
    """Load 1D param — must use TILE_LAYOUT on device like all other tensors."""
    t = state_dict[f"{prefix}.{name}"]
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
```

With:
```python
def load_1d(name):
    """Load 1D param — must use TILE_LAYOUT on device like all other tensors."""
    t = state_dict[f"{prefix}.{name}"]
    return ttnn.as_tensor(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=weight_cache_path / f"{prefix}.{name}" if weight_cache_path else None,
    )
```

- [ ] **Step 4: Update QKV fused projection loading (line 76-78)**

Replace line 77:
```python
self.qkv_proj_weight = ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
```

With:
```python
self.qkv_proj_weight = ttnn.as_tensor(
    t,
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    cache_file_name=weight_cache_path / f"{prefix}.qkv_proj.weight" if weight_cache_path else None,
)
```

- [ ] **Step 5: Do NOT change `load_conv_weight` or `load_conv_bias_or_none`**

These functions (lines 63-67, 93-98) load with `memory_config=ttnn.L1_MEMORY_CONFIG` and no `device`. They stay on host. No caching benefit — leave them as-is.

---

### Task 5: Thread `weight_cache_path` through `Qwen35TransformerBlock`

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_decoder.py:41-67`

- [ ] **Step 1: Add `weight_cache_path` parameter to `__init__`**

Change line 41 from:
```python
def __init__(self, args, state_dict, layer_num, device):
```
to:
```python
def __init__(self, args, state_dict, layer_num, device, weight_cache_path=None):
```

- [ ] **Step 2: Replace `ttnn.from_torch` with `ttnn.as_tensor` in `load_norm`**

Replace lines 48-56:
```python
def load_norm(name):
    """Load norm weight with +1 offset for zero-centered RMSNorm.

    Qwen3.5 uses zero-centered RMSNorm: output = x_normed * (1 + weight).
    We pre-add 1 to the weight so the fused ttnn.rms_norm can be used directly.
    """
    t = state_dict[f"{prefix}.{name}"]
    t_offset = t + 1.0  # Pre-offset for zero-centered norm
    return ttnn.from_torch(t_offset, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
```

With:
```python
def load_norm(name):
    """Load norm weight with +1 offset for zero-centered RMSNorm.

    Qwen3.5 uses zero-centered RMSNorm: output = x_normed * (1 + weight).
    We pre-add 1 to the weight so the fused ttnn.rms_norm can be used directly.
    The +1 offset is baked into the cached tensor — safe on reload.
    """
    t = state_dict[f"{prefix}.{name}"]
    t_offset = t + 1.0  # Pre-offset for zero-centered norm
    return ttnn.as_tensor(
        t_offset,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=weight_cache_path / f"{prefix}.{name}" if weight_cache_path else None,
    )
```

- [ ] **Step 3: Pass `weight_cache_path` to child constructors**

Replace lines 62-67:
```python
if self.is_full_attention:
    self.attention = Qwen35GatedAttention(args, state_dict, layer_num, device)
else:
    self.attention = Qwen35GatedDeltaNet(args, state_dict, layer_num, device)

self.feed_forward = Qwen35MLP(args, state_dict, layer_num, device)
```

With:
```python
if self.is_full_attention:
    self.attention = Qwen35GatedAttention(args, state_dict, layer_num, device, weight_cache_path)
else:
    self.attention = Qwen35GatedDeltaNet(args, state_dict, layer_num, device, weight_cache_path)

self.feed_forward = Qwen35MLP(args, state_dict, layer_num, device, weight_cache_path)
```

---

### Task 6: Thread `weight_cache_path` through `Qwen35Model` and add caching to top-level weights

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py:26-86`

- [ ] **Step 1: Update `__init__` signature and add cache path**

Change line 26 from:
```python
def __init__(self, args, state_dict, device):
```
to:
```python
def __init__(self, args, state_dict, device, weight_cache_path=None):
```

- [ ] **Step 2: Replace embedding `ttnn.from_torch` with `ttnn.as_tensor`**

Replace lines 32-37:
```python
self.tok_embeddings = ttnn.from_torch(
    embed_weight.unsqueeze(0).unsqueeze(0),
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
)
```

With:
```python
self.tok_embeddings = ttnn.as_tensor(
    embed_weight.unsqueeze(0).unsqueeze(0),
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    cache_file_name=weight_cache_path / "tok_embeddings.weight" if weight_cache_path else None,
)
```

- [ ] **Step 3: Pass `weight_cache_path` to `Qwen35TransformerBlock`**

Change line 46 from:
```python
layer = Qwen35TransformerBlock(args, state_dict, i, device)
```
to:
```python
layer = Qwen35TransformerBlock(args, state_dict, i, device, weight_cache_path)
```

- [ ] **Step 4: Replace final norm `ttnn.from_torch` with `ttnn.as_tensor`**

Replace lines 51-53:
```python
self.norm_weight = ttnn.from_torch(
    norm_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
)
```

With:
```python
self.norm_weight = ttnn.as_tensor(
    norm_weight,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    cache_file_name=weight_cache_path / "norm.weight" if weight_cache_path else None,
)
```

- [ ] **Step 5: Replace LM head `ttnn.from_torch` with `ttnn.as_tensor`**

Replace lines 58-60:
```python
self.lm_head_weight = ttnn.from_torch(
    lm_head_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device,
)
```

With:
```python
self.lm_head_weight = ttnn.as_tensor(
    lm_head_weight,
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    cache_file_name=weight_cache_path / "output.weight" if weight_cache_path else None,
)
```

- [ ] **Step 6: Update `from_pretrained` to compute and pass `weight_cache_path`**

After line 71 (after `args = Qwen35ModelArgs(...)` block), add:
```python
cache_path = args.weight_cache_path()
```

And change line 86 from:
```python
return cls(args, state_dict, device)
```
to:
```python
return cls(args, state_dict, device, weight_cache_path=cache_path)
```

---

### Task 7: Replace `print()` with `loguru.logger` in model code

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py`

- [ ] **Step 1: Add loguru import**

After line 9 (`import ttnn`), add:
```python
from loguru import logger
```

- [ ] **Step 2: Replace print statements in `__init__`**

Replace line 43:
```python
print(f"Loading {args.n_layers} transformer layers...")
```
With:
```python
logger.info(f"Loading {args.n_layers} transformer layers...")
```

- [ ] **Step 3: Replace print statements in `from_pretrained`**

Replace line 73:
```python
print("Loading weights from safetensors...")
```
With:
```python
logger.info("Loading weights from safetensors...")
```

Replace line 82:
```python
print("Remapping weights...")
```
With:
```python
logger.info("Remapping weights...")
```

- [ ] **Step 4: Replace print in `decode` profiling block**

Replace lines 154-158:
```python
print(f"  [PROFILE] embed+rope: {(_t1-_t0)*1000:.1f}ms | "
      f"deltanet(24): {sum(_layer_times['deltanet'])*1000:.1f}ms (avg {dn_avg:.1f}ms) | "
      f"attention(8): {sum(_layer_times['attention'])*1000:.1f}ms (avg {att_avg:.1f}ms) | "
      f"norm+lmhead: {(_t3-_t2)*1000:.1f}ms | "
      f"total: {(_t3-_t0)*1000:.1f}ms")
```
With:
```python
logger.info(
    f"[PROFILE] embed+rope: {(_t1-_t0)*1000:.1f}ms | "
    f"deltanet(24): {sum(_layer_times['deltanet'])*1000:.1f}ms (avg {dn_avg:.1f}ms) | "
    f"attention(8): {sum(_layer_times['attention'])*1000:.1f}ms (avg {att_avg:.1f}ms) | "
    f"norm+lmhead: {(_t3-_t2)*1000:.1f}ms | "
    f"total: {(_t3-_t0)*1000:.1f}ms"
)
```

---

### Task 8: Replace `print()` with `loguru.logger` in test files

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py`
- Modify: `models/demos/blackhole/qwen3_5_9b/tests/test_qwen35_demo.py`
- Modify: `models/demos/blackhole/qwen3_5_9b/tests/test_component_pcc.py`

#### test_model_e2e.py

- [ ] **Step 1: Add loguru import and `run_for_blackhole` marker**

After line 8 (`import pytest`), add:
```python
from loguru import logger
```

After line 12 (after `from models...import Qwen35Model`), add:
```python
from models.common.utility_functions import run_for_blackhole
```

After line 15 (`CHECKPOINT_DIR = ...`), add:
```python

pytestmark = run_for_blackhole()
```

- [ ] **Step 2: Replace all `print()` with `logger.info()` in test_model_e2e.py**

Replace every `print(` with `logger.info(` throughout the file. Specifically:
- Line 33: `print(f"\n[PERF] Model load time: {load_time:.1f}s")` → `logger.info(f"[PERF] Model load time: {load_time:.1f}s")`
- Line 56: `print(...)` → `logger.info(...)`
- Line 72: `print(...)` → `logger.info(...)`
- Lines 73-75: `print(...)` → `logger.info(...)`
- Line 95: `print(...)` → `logger.info(...)`
- Lines 124-137: All `print(...)` → `logger.info(...)` (the performance summary block)

Remove leading `\n` from format strings — loguru adds its own line formatting.

#### test_qwen35_demo.py

- [ ] **Step 3: Add loguru import**

After line 7 (`import pytest`), add:
```python
from loguru import logger
```

- [ ] **Step 4: Replace all `print()` with `logger.info()` in test_qwen35_demo.py**

- Line 32: `print(f"\n[PERF] Model load: ...")` → `logger.info(f"[PERF] Model load: ...")`
- Line 78: `print(f"\n[PERF] TTFT: ...")` → `logger.info(f"[PERF] TTFT: ...")`
- Line 79: `print(f"[PERF] Avg decode: ...")` → `logger.info(f"[PERF] Avg decode: ...")`
- Line 80: `print(f"[PERF] Generated ...")` → `logger.info(f"[PERF] Generated ...")`
- Line 81: `print(f"[TEXT] ...")` → `logger.info(f"[TEXT] ...")`

#### test_component_pcc.py

- [ ] **Step 5: Add loguru import**

After line 7 (`import pytest`), add:
```python
from loguru import logger
```

- [ ] **Step 6: Replace all `print()` with `logger.info()` in test_component_pcc.py**

Replace every `print(` with `logger.info(` throughout. Key locations:
- Line 66-69: Embedding PCC results
- Line 91-92, 97-98, 106-107: LM Head results
- Line 129: RMSNorm PCC
- Line 150-152: MLP PCC
- Line 213-216: Gated Attention PCC
- Line 276-278: DeltaNet PCC
- Line 298-303: Decoder block output

---

### Task 9: Replace `print()` with `loguru.logger` in demo

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/demo/demo.py`

- [ ] **Step 1: Add loguru import**

After line 13 (`import time`), add:
```python
from loguru import logger
```

- [ ] **Step 2: Replace status `print()` with `logger.info()` — keep streaming prints**

Replace these status prints:
- Line 41: `print("Opening Blackhole P150 device...")` → `logger.info("Opening Blackhole P150 device...")`
- Line 49: `print("Loading Qwen3.5-9B model...")` → `logger.info("Loading Qwen3.5-9B model...")`
- Line 55: `print(f"Model loaded in {load_time:.1f}s")` → `logger.info(f"Model loaded in {load_time:.1f}s")`
- Line 66: `print(f"\nUser: {args.prompt}")` → `logger.info(f"User: {args.prompt}")`
- Line 67: `print(f"Formatted prompt ({prompt_len} tokens)")` → `logger.info(f"Formatted prompt ({prompt_len} tokens)")`
- Line 68: `print("-" * 60)` → `logger.info("-" * 60)`
- Line 77: `print(f"Prefill: ...")` → `logger.info(f"Prefill: ...")`
- Line 105: `print(f"\n\n{'-' * 60}")` → `logger.info("-" * 60)`
- Line 106: `print(f"Generated ...")` → `logger.info(f"Generated ...")`
- Line 107: `print(f"Avg decode: ...")` → `logger.info(f"Avg decode: ...")`

**Keep these streaming prints as-is** (they need `end=""` and `flush=True` for interactive token output):
- Line 80: `print(f"\nAssistant:", end=" ", flush=True)`
- Line 81: `print(tokenizer.decode([next_token]), end="", flush=True)`
- Line 99: `print(tokenizer.decode([next_token]), end="", flush=True)`

---

### Task 10: Add `run_for_blackhole` marker to `test_single_layer.py`

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tests/test_single_layer.py:1-17`

- [ ] **Step 1: Add import and pytestmark**

After line 7 (`import pytest`), add:
```python
from models.common.utility_functions import run_for_blackhole
```

After line 17 (`CHECKPOINT_DIR = ...`), before `PCC_THRESHOLD`, add:
```python

pytestmark = run_for_blackhole()
```

---

### Task 11: Verify changes work

- [ ] **Step 1: Run a quick syntax check**

Run: `python -c "from models.demos.blackhole.qwen3_5_9b.tt.qwen35_model import Qwen35Model; print('Import OK')"`

Expected: `Import OK` (no syntax errors)

- [ ] **Step 2: Verify tensor cache directory would be created**

Run: `python -c "from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs; a = Qwen35ModelArgs(); print(a.weight_cache_path())"`

Note: This will fail if `weight_dtype` is None (no device). That's expected — cache path is only meaningful when a device is present.

- [ ] **Step 3: Run test_weight_mapping (CPU-only, no device needed)**

Run: `pytest models/demos/blackhole/qwen3_5_9b/tests/test_weight_mapping.py -v --timeout=60`

Expected: All tests pass (these are CPU-only and should not be affected by changes)
