# BGE-M3 custom ops sandbox

Model-local `ttnn.generic_op` kernels and Python wrappers under perf-sweep
validation. **Not** imported by production model code (`attention.py`,
`mlp.py`, `model.py`, `encoder.py`) until a sweep demonstrates a real
device-time win **and** PCC stays ≥ 0.9999 vs. the stock reference op.

## Methodology

Each candidate op follows the same flow:

1. **Baseline lock-in.** A sweep `test_*_baseline_timing` re-measures the
   stock op(s) the new kernel would replace. We confirm the number matches
   the latest `test_bs1_optimized.log` (e.g. `nlp_create_qkv_heads` ≈
   29 µs/op × 24 = ~700 µs total).
2. **PCC harness.** A second sweep test runs the new op against the same
   inputs as the baseline and asserts PCC ≥ 0.9999 on every output tensor.
3. **Timing matrix.** A sweep enumerates config knobs (memory layout, dtype,
   per-core tiling, …). Each row writes a CSV entry: `(config, best_µs,
   avg_µs, pcc_min, status, notes)`. Failed configs (compile errors, OOM,
   PCC fail) are recorded as rows, never raised.
4. **Integration.** Only after we have a winning row do we touch
   `attention.py` etc. Behind a feature flag if it adds risk.

## Layout

    custom_ops/
        __init__.py              — package docstring
        README.md                — this file
        fused_qkv_heads/         — Sweep 1.1: scatter-writer QKV→Q/K/V
            __init__.py
            op.py
            kernels/
                writer_qkv_scatter.cpp
        (future)
        fused_sdpa_concat/       — Sweep 1.2: scatter SDPA out → concat layout
        ln_to_matmul_handoff/    — Sweep 1.3: sharded LN→matmul (memcfg-only)

## Sweep files (in `tests/perf/`)

| Sweep | Target bucket | Expected win |
|---|---|---|
| `sweep_fused_qkv_heads_writer.py` | `NlpCreateHeads` 702 µs total | −500 to −700 µs |
| `sweep_fused_sdpa_concat_writer.py` | `NLPConcatHeads` 254 µs total | −200 µs |
| `sweep_layernorm_to_matmul_layout.py` | reshard `S↔I` ~170 µs | −100 to −150 µs |

Sum target: 5.4 ms → ~4.0 ms (Phase 1). Phase 2 (4.0 → 3.0 ms) requires
deeper fusion (Matmul+LN, SDPA grid) and is not yet scaffolded.

## Running a sweep

From `/home/gtobar/new_test/`:

```bash
source local_env.sh

# Baseline-only run (fast, validates plumbing before kernel work)
TT_VISIBLE_DEVICES=0 pytest \
  models/demos/wormhole/bge_m3/tests/perf/sweep_fused_qkv_heads_writer.py::test_baseline_qkv_heads_timing \
  -sv

# Full sweep
TT_VISIBLE_DEVICES=0 pytest \
  models/demos/wormhole/bge_m3/tests/perf/sweep_fused_qkv_heads_writer.py \
  -sv

# Limit row count for quick iteration
BGE_M3_SWEEP_LIMIT=4 TT_VISIBLE_DEVICES=0 pytest \
  models/demos/wormhole/bge_m3/tests/perf/sweep_fused_qkv_heads_writer.py \
  -sv
```

CSV output lands in `tests/perf/sweep_results/<sweep_name>.csv`.

## Conventions

- Kernel paths in `op.py` are relative to `TT_METAL_HOME`, e.g.
  `"models/demos/wormhole/bge_m3/tt/custom_ops/fused_qkv_heads/kernels/writer_qkv_scatter.cpp"`.
- Each sub-package exposes `ttnn`-style entry points (callable, returns a
  ttnn tensor) so the sweep file can swap stock vs. custom with a flag.
- Sweep files never import from production `attention.py` / `mlp.py`. They
  build their own minimal harness so PCC failures here cannot crash the
  full-model perf test.
