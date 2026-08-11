# Weight Loading and TTNN Cache

The `__init__` of each module uses a 3-way branch:

```python
if torch_weights is not None:
    # Real weights provided — convert to TTNN, optionally write cache
    weights = self._convert_and_cache_weights(torch_weights, ..., cache_path, device=mesh_device)
elif weight_cache_path is not None:
    # Cache-only — pass None weights, load from .tensorbin files
    weights = self._convert_and_cache_weights(None, ..., cache_path, device=mesh_device)
else:
    # Testing without cache — random/dummy weights
    ...
```

With this, every TT module that holds weights follows the same 3-method pattern:

| Method | Type | Purpose |
|---|---|---|
| `check_cache_complete(cache_path, ...)` | `@staticmethod` | Returns `True` if all `.tensorbin` files exist for this component |
| `build_ttnn_cache(torch_weights, ..., cache_path)` | `@staticmethod` | Writes `.tensorbin` cache to disk without touching device memory. Delegates to `_convert_and_cache_weights(..., device=None)` |
| `_convert_and_cache_weights(torch_weights, ..., device)` | `@staticmethod` | Single workhorse that handles all weight paths. When `device=None` it builds cache; when `device=mesh_device` it loads to device. Accepts `None` for `torch_weights` — creates minimal `torch.empty()` placeholders with post-transform shapes, skipping expensive host operations (`torch.randn`, transpose, stacking, slicing). `ttnn.as_tensor` ignores the placeholder when a `.tensorbin` cache file exists |


Components implementing this pattern: `TtParallelEmbedding`, `TtDistributedRmsNorm`, `ttMLA`, `TtMoEGatePrefill`, `TtRoutedExpert`, `TtSharedExpert` (inherited by `TtFfn` as well). Note: the method signatures are not yet unified across modules — each component has different arguments reflecting its weight structure (single tensor, dict, list of dicts, state_dict, etc.).

---

## Cache Check Optimization

### Problem
The `check_cache_complete()` methods used `Path.glob()` to verify existence of `.tensorbin` files. Each glob scans the entire cache directory (~2303 files), taking ~24ms per call. With 2303 pattern checks: **2303 × 24ms = 56 seconds**.

### Solution
`utils/fast_cache_checker.py` replaces sequential glob calls with:
1. Single `iterdir()` to build file set (~40ms)
2. In-memory prefix/suffix matching (~0.03ms per pattern)

**Result: 56s → 240ms (230× speedup)**

---

## Performance: tt_transformer_creation

Benchmark: 61-layer transformer, 1024 seq len, 8×4 mesh (32 devices)

| Scenario | tt_transformer_creation | Notes |
|----------|------------------------|-------|
| TTNN cache creation | ~465s | Building `.tensorbin` files to disk |
| Cold OS page cache | ~435s | First run after reboot or `drop_caches` |
| Warm OS page cache | ~42s | Subsequent runs (tensorbin files in RAM) |

The 10× difference between cold and warm page cache is disk I/O: reading ~100GB of `.tensorbin` files from disk (~250MB/s) vs RAM.

**For benchmarking**: Always run twice and report the second run (warm cache) as the representative time.

To manually drop the OS page cache (requires root on host, not inside container):
```bash
echo 3 | sudo tee /proc/sys/vm/drop_caches
```
