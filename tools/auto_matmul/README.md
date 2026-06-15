# Auto Matmul Cache Tuning

`tools/auto_matmul/tune_matmul_cache.py` pre-populates the measured auto-matmul cache from a JSON manifest.

## Manifest Shape

Each case may specify:

- `name`
- `lhs_shape`
- `rhs_shape`
- `bias_shape`
- `is_linear`
- `rhs_on_host`
- `lhs_shard_dim`
- `rhs_shard_dim`
- `bias_shard_dim`
- `lhs_mesh_dims`
- `rhs_mesh_dims`
- `bias_mesh_dims`
- `transpose_a`
- `transpose_b`
- `dtype`
- `layout`
- `memory_config`
- `activation`

Example:

```json
{
  "cases": [
    {
      "name": "single_device_linear",
      "lhs_shape": [1, 1, 32, 4096],
      "rhs_shape": [1, 1, 4096, 4096],
      "bias_shape": [1, 1, 1, 4096],
      "is_linear": true,
      "dtype": "bfloat16",
      "layout": "TILE_LAYOUT",
      "memory_config": "DRAM_MEMORY_CONFIG"
    }
  ]
}
```

For multi-device runs, use `*_shard_dim` for simple 1D sharding or `*_mesh_dims`
for explicit 2D mesh placement. If a mapper field is omitted on a mesh run,
that tensor is replicated.

## Usage

```bash
python3 tools/auto_matmul/tune_matmul_cache.py --manifest /path/to/cases.json --device-id 0
```

```bash
python3 tools/auto_matmul/tune_matmul_cache.py --manifest /path/to/cases.json --mesh-shape 1x8
```

Useful environment controls:

- `TTNN_AUTO_MATMUL_CACHE_DIR`
- `TTNN_AUTO_MATMUL_VERSION`
- `TTNN_AUTO_MATMUL_FORCE_RETUNE=1`
