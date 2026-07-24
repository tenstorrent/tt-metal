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

`--manifest` and `--save-report` are resolved relative to (and confined within)
the current working directory, so run this from the directory that holds your
manifest and where the report should be written.

```bash
python3 tools/auto_matmul/tune_matmul_cache.py --manifest cases.json --device-id 0
```

```bash
python3 tools/auto_matmul/tune_matmul_cache.py --manifest cases.json --mesh-shape 1x8
```

Useful environment controls:

- `TTNN_AUTO_MATMUL_CACHE_DIR`
- `TTNN_AUTO_MATMUL_VERSION`
- `TTNN_AUTO_MATMUL_FORCE_RETUNE=1`

## Full Timing Reports

Use `--save-report` to dump the full selector result for each case, including
the winning descriptor, every measured candidate timing, and selector
recommendations:

```bash
TTNN_AUTO_MATMUL_FORCE_RETUNE=1 \
python3 tools/auto_matmul/tune_matmul_cache.py \
  --manifest cases.json \
  --mesh-shape 1x8 \
  --save-report auto-matmul-report.json
```

This is the repo-native way to produce isolated op-level evidence that the
selector chose the fastest legal recipe for a representative shape set.

## Refresh / Retrain Workflow

When underlying matmul kernels or fused CCL implementations change:

1. Bump the cache version by setting `TTNN_AUTO_MATMUL_VERSION`, or let the
   default `ttnn.model_preprocessing.git_hash()` change invalidate the old
   cache automatically.
2. Re-run `tools/auto_matmul/tune_matmul_cache.py` on the supported single-device
   and multi-device shape manifests.
3. Save the full timing report with `--save-report` so winner changes and
   fallback behavior can be reviewed directly.
4. Re-run the correctness / model tests on the refreshed cache before updating
   model-side perf claims.
