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

## Guarantees for cache misses and unseen shapes — exhaustive vs heuristic

Each resolution tier makes a **different** guarantee. Only the first is *exhaustive* (an
optimality guarantee); the rest are *correctness-guaranteed but best-effort* on performance.

| tier | how the config is chosen | search | correctness guarantee | performance guarantee |
|---|---|---|---|---|
| **1. exact cache hit** (known shape) | the config that won the **exhaustive** offline tune for this exact signature | **exhaustive** — full valid space `{scheme × grid × in0_block_w × out-subblock}` | rel-L2 ≤ 0.05 vs an fp32 golden (gated at tune time) | **optimal within the searched valid space** (the measured argmin) |
| **2. predictor** (unseen shape) | nearest **same-regime** tuned neighbor's *scheme*, params **re-derived** for the new dims | **heuristic** — no search, no device timing at predict time | statically validated (`in0_block_w \| Kt`, L1 budget, subblock rule) **+ run once guarded → base op on any error** | **none** — a fast best-guess; measured regret **mean 1.057, max 1.151** vs a full tune (scheme always correct) |
| **3. online tune** (opt-in, `TTNN_AUTO_MATMUL_ONLINE_TUNE=1`) | on-device sweep of an **even-sampled, ≤ `_MAX_PROGRAM_CONFIG_CANDIDATES`** subset | **bounded** (between exhaustive and heuristic) | same fp32-golden gate as the exhaustive tune | best of the sampled subset — **not** guaranteed ≤5%; measured regret up to **~1.22** vs the exhaustive best |
| **4. base op** | plain `ttnn.matmul` (no program config) | none | identical to stock ttnn | **never worse than not using auto-config** |

**The one-line contract:** *for a shape in the offline corpus you get the exhaustive optimum
(tier 1); for an unseen shape you get a correctness-guaranteed best-guess (tier 2) that is
never worse than the base op, not an optimality guarantee.* Optimality is a property of the
**offline corpus coverage**, not of any per-call online search — which is why the corpus +
predictor index are the shipped artifacts.

**Exhaustive vs heuristic, precisely**
- **Exhaustive** = the offline tune (`TTNN_AUTO_MATMUL_EXHAUSTIVE=1`): enumerate every valid
  config, time each on device-kernel duration, correctness-gate, take the argmin. Runs once,
  offline; its winner is cached and shipped. *This is the only tier with an optimality claim.*
- **Heuristic** = the predictor: reuse a nearby shape's winning scheme and re-derive the
  per-shape params. No enumeration, no timing — a scheme-correct guess, gated for correctness.

**How each guarantee is validated** (all on N300; commands under *Validate the guarantees* below)
- tier-1 exhaustiveness/optimality — the item-#1 optimality test: the selected config is within
  the device-kernel **noise floor** of the brute-forced oracle on **8/8** taxonomy shapes
  (pre-widening: 2/8, worst 1.476× off → after: worst 1.04×).
- tier-1/2/4 resolution + staleness — **host tests** (`test_auto_config_helpers.py`): cache
  **hit** served without tuning, **miss** → predict, **stale-by-version** and
  **stale-by-internals (arch/schema)** recognized, predictor **regime-isolation**, the cheap
  static gate (no fp32 golden on the predict hot path), and predictions kept **runtime-only**.
- tier-2 predictor accuracy — held-out shapes on device: **scheme reuse correct on every shape**
  (square → 2d, narrow → 1d), regret **mean 1.057 / max 1.151**.

### Validate the guarantees

```bash
# resolution order + staleness + predictor guarantees (device-free):
python3 -m pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_auto_config_helpers.py -q \
    -k "cache or stale or predict or regime or static or predicted or version or internals"

# tier-1 exhaustive optimality + tier-2 predictor accuracy reproduce from the committed tuner
# (see the pipeline below and the PR validation report for the on-device numbers).
```

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
