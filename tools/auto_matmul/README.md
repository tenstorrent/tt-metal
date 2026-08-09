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

## Default runtime behavior (unseen shapes)

At runtime, `ttnn.experimental.auto_config.matmul/linear` resolves a shape in this order,
**offline-first** — it never sweeps candidates on device on first touch of a shape:

1. **Exact offline-tuned cache hit** — served directly (~ms lookup). Authoritative.
2. **Predictor (unseen shape)** — a regime-partitioned nearest-shape lookup over
   `index.json` reuses a nearby tuned winner's *scheme* and re-derives the config for the
   new dims. It is statically validated (in0_block_w | Kt, L1 budget, subblock rule) and
   run once guarded; any error falls back to the base op. Predictions live in the
   in-process runtime record only — **never written to the on-disk cache**.
3. **Base op** — the plain `ttnn.matmul` if there is no comparable shape to predict from.

The inline online candidate sweep is **opt-in**, not the default: set
`TTNN_AUTO_MATMUL_ONLINE_TUNE=1` (used by the tuner below and by developers). Production
ships a pre-tuned cache + index; it does not tune on the hot path.

## Regenerate → tune → index → validate → ship

```bash
# 1. Generate a bounded, reproducible corpus manifest.
python3 tools/auto_matmul/generate_corpus.py --out corpus.json \
    --dtypes bfloat16 --extra-shapes 2048x2880x5120 2048x4096x2880

# 2. Tune it on device (online sweep opt-in; exhaustive = ship the best config).
TTNN_AUTO_MATMUL_ONLINE_TUNE=1 TTNN_AUTO_MATMUL_EXHAUSTIVE=1 \
python3 tools/auto_matmul/tune_matmul_cache.py --manifest corpus.json --save-report report.json

# 3. Build the predictor index from the freshly tuned cache (offline, no device).
python3 -c "import ttnn; ttnn.experimental.auto_config.build_predictor_index()"

# 4. Validate optimality/correctness, then ship the version dir (cache entries + index.json).
```

Winners are stable across re-tunes because selection uses low-noise device-kernel timing
(median over repeats); `--save-report` records each winner's margin over the runner-up, so
near-ties are visible and a regeneration is auditable.

## Refresh when matmul internals change

The cache is keyed by a **version** that includes the inputs that can change a tuned
winner, so a change correctly marks old entries stale (a stale entry = a version mismatch,
served as a miss and re-tuned):

- `version = f(code, arch, schema)` by default — `code` = `git_hash()` / CI SHA / package
  version; `arch` = `ttnn.get_arch_name()` (a config tuned on one arch must not be served on
  another); `schema` = `_SELECTOR_SCHEMA_VERSION` in `_selector.py`.
- **Bump `_SELECTOR_SCHEMA_VERSION`** when tuning logic / config builders change, or when
  kernels are rebuilt in a way the git hash does not reflect (a local kernel rebuild) —
  this is the lever that closes the "internals changed but the hash didn't" case.
- `TTNN_AUTO_MATMUL_VERSION` pins the version verbatim (you own reproducibility under a pin).

Refresh procedure: bump the version (or let arch/git/schema change it), re-run the
regenerate→tune→index pipeline above, and re-run the correctness / model tests on the
refreshed cache before updating any perf claims.
