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
