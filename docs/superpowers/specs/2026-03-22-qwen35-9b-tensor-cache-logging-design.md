# Qwen3.5-9B Tensor Cache & Logging Alignment

## Problem

1. **No tensor cache**: The Qwen3.5-9B model loads weights from safetensors and converts them via `ttnn.from_torch()` on every run. Other models (Llama, Falcon) use `ttnn.as_tensor()` with `cache_file_name` to cache converted tensors to disk, making subsequent runs significantly faster.

2. **Print-based logging**: All logging uses `print()` statements (~50+ locations across model and test files). The codebase standard is `loguru.logger`.

3. **Non-standard test fixtures**: `test_model_e2e.py`, `test_single_layer.py`, and `test_component_pcc.py` manually open/close devices instead of using the root conftest's `device` fixture.

## Design

### 1. Add `weight_cache_path()` to `Qwen35ModelArgs`

In `tt/model_config.py`, add:

```python
from pathlib import Path

def weight_cache_path(self, dtype=None):
    """Return cache directory for converted weight tensors."""
    if dtype is None:
        dtype = self.weight_dtype
    import ttnn
    if dtype == ttnn.bfloat8_b:
        suffix = "tensor_cache_bfp8"
    else:
        suffix = "tensor_cache_bf16"
    return Path(self.checkpoint_dir) / suffix
```

Note: Directory creation is not needed here — `ttnn.as_tensor` creates parent directories automatically when writing cache files.

### 2. Replace `ttnn.from_torch()` with `ttnn.as_tensor()` for all weights

**Important**: `ttnn.as_tensor()` requires `memory_config` when `device` is specified. All calls must include `memory_config=ttnn.DRAM_MEMORY_CONFIG` (or appropriate config).

**Files to change:**

#### `tt/qwen35_model.py`
- Thread `weight_cache_path` from `from_pretrained()` into `__init__()`
- Embedding: `ttnn.as_tensor(..., memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_file_name=...)`
- Final norm: same pattern
- LM head: same pattern

#### `tt/qwen35_decoder.py`
- Accept `weight_cache_path` in `__init__`
- Norm weights: `cache_file_name=cache_path / f"layers.{layer_num}.input_layernorm.weight"` etc.
- **Note**: Norm weights have `+1.0` pre-offset for zero-centered RMSNorm. This is correct — the cache stores the pre-offset value, so on reload the offset is already baked in.
- Pass `weight_cache_path` to MLP, GatedAttention, GatedDeltaNet constructors

#### `tt/qwen35_mlp.py`
- Accept `weight_cache_path` in `__init__`
- `load_weight(name)` helper: add `memory_config` and `cache_file_name`

#### `tt/qwen35_gated_attention.py`
- Accept `weight_cache_path` in `__init__`
- `load_weight(name)` and `load_1d(name)` helpers: add `memory_config` and `cache_file_name`

#### `tt/qwen35_gated_deltanet.py`
- Accept `weight_cache_path` in `__init__`
- `load_weight_2d(name)` and `load_1d(name)`: add `memory_config` and `cache_file_name` (these have `device=device`)
- `load_conv_weight(name)` and `load_conv_bias_or_none(name)`: these load with `memory_config=ttnn.L1_MEMORY_CONFIG` but **no device** — they stay on host. Skip caching for these (no `cache_file_name`) since there's no device transfer to skip. The conv weights are small and cheap to load.

**Key change pattern** (example from MLP):
```python
# Before
def load_weight(name):
    t = state_dict[f"{prefix}.{name}"].T.contiguous()
    return ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

# After
def load_weight(name):
    t = state_dict[f"{prefix}.{name}"].T.contiguous()
    return ttnn.as_tensor(
        t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=weight_cache_path / f"{prefix}.{name}" if weight_cache_path else None,
    )
```

**Fused/precomputed weights** (`ab_proj_weight`, `fused_conv_weight_taps`, `A_neg`, etc.) remain computed at init time from the (now-cached) base weights. They are derived tensors and don't need their own cache files. Note: these involve `ttnn.to_torch()` round-trips on the base weights, which is the same cost as before. A future optimization could cache the derived tensors too.

### 3. Replace `print()` with `loguru.logger`

**All files with print statements:**

#### `tt/qwen35_model.py`
- Add `from loguru import logger`
- `print(f"Loading {args.n_layers}...")` → `logger.info(...)` (line 43)
- `print("Loading weights from safetensors...")` → `logger.info(...)` (line 73)
- `print("Remapping weights...")` → `logger.info(...)` (line 82)
- Profile print block in `decode()` method (lines 154-158) → `logger.info(...)`

#### `tests/test_model_e2e.py`
- Add `from loguru import logger`
- All `print(f"[PERF]...")` → `logger.info(...)`
- Performance summary block → `logger.info(...)` lines

#### `tests/test_qwen35_demo.py`
- Add `from loguru import logger`
- All `print()` → `logger.info(...)`

#### `tests/test_component_pcc.py`
- Add `from loguru import logger`
- All `print()` → `logger.info(...)`

#### `demo/demo.py`
- Add `from loguru import logger`
- All `print()` → `logger.info()` for status messages
- **Keep** `print(..., end="", flush=True)` for streaming token output (interactive UX requires unbuffered character-by-character output)

### 4. Test Fixture Alignment

#### `tests/test_model_e2e.py` and `tests/test_single_layer.py`

These tests have module-scoped `device` fixtures with module-scoped `model`/`model_fixtures` that depend on them. The root conftest's `use_module_device` marker cannot be used here because pytest's scope rules prevent module-scoped fixtures from requesting the function-scoped `device` fixture (even though it delegates internally).

**Solution**: Keep the manual module-scoped `device` fixture (needed for the module-scoped `model` fixture dependency chain), but add the `@run_for_blackhole()` marker and replace `print()` with `logger.info()`:

```python
from models.common.utility_functions import run_for_blackhole

pytestmark = run_for_blackhole()

@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)
```

#### `tests/test_component_pcc.py`

This test is run with `--noconftest` (per its docstring). **Keep the manual device fixture** for this file — migrating to the conftest fixture would break that usage pattern. Only apply the logging changes.

## Files Changed

| File | Changes |
|------|---------|
| `tt/model_config.py` | Add `weight_cache_path()` method |
| `tt/qwen35_model.py` | Cache path threading, `ttnn.as_tensor` with `memory_config`, loguru |
| `tt/qwen35_decoder.py` | Accept + pass `weight_cache_path` |
| `tt/qwen35_mlp.py` | `ttnn.as_tensor` with cache + `memory_config` |
| `tt/qwen35_gated_attention.py` | `ttnn.as_tensor` with cache + `memory_config` |
| `tt/qwen35_gated_deltanet.py` | `ttnn.as_tensor` with cache + `memory_config` (skip conv weights) |
| `tests/test_model_e2e.py` | `run_for_blackhole` marker, loguru |
| `tests/test_single_layer.py` | `run_for_blackhole` marker |
| `tests/test_component_pcc.py` | loguru only (keep manual fixture due to `--noconftest`) |
| `tests/test_qwen35_demo.py` | loguru |
| `demo/demo.py` | loguru (keep streaming prints) |

## Testing

- First run: creates tensor cache directory, populates cache files. Should take roughly same time as before.
- Second run: loads from cache files. Should be significantly faster (skip dtype conversion + device transfer).
- Verify cache directory is created at `{checkpoint_dir}/tensor_cache_bfp8/`
- Verify logger output appears in pytest output with `-s` flag
- Verify all existing tests still pass
