# Gemma4 XY torus investigation

Measured September 7, 2026 on a 32-device Blackhole mesh: Gemma4 31B,
8×4 (CP=8, TP=4), 262,144 text tokens, 8,192-token chunks, final readback.
Baseline: `d8741bae6e0`. The experimental patch adapts the topology/CCL changes
from `20418a11b6fcc1d3100e8ddb1de9124a134007c1`; that reference used 4×8.

## Results

| Configuration | Device time | First chunk | Last chunk |
| --- | ---: | ---: | ---: |
| Baseline: FABRIC_2D | 13.564 s | 235.6 ms | 614.6 ms |
| XY torus + async gathers (this branch) | 13.705 s | 240.2 ms | 619.5 ms |
| XY torus + original gathers | 13.708 s | 240.2 ms | 620.0 ms |
| Y-only torus + original gathers | 13.556 s | 235.0 ms | 614.5 ms |

XY torus adds approximately 4.4 ms per chunk, or 1% overall. Restoring the
original gathers does not recover performance; removing X (TP-axis) wrapping
does. This isolates the regression to TP-axis wrapping in this configuration,
without identifying the specific kernel overhead. History-dependent growth is
nearly unchanged. Keep the baseline for performance; Y-only torus shows no
meaningful gain. This branch preserves the experimental XY patch for reference.

## Reproduction

```bash
HF_HUB_OFFLINE=1 \
HF_HOME=/localdev/$USER/huggingface \
HF_MODEL=google/gemma-4-31B-it \
TT_CACHE_PATH=/localdev/$USER/huggingface/tt_cache/google--gemma-4-31B-it \
PYTEST_TIMEOUT=1800 \
python_env/bin/python3 -m pytest \
  'models/demos/gemma4/demo/text_demo_prefill.py::test_prefill_long_context_traced[blackhole-readback_final-ctx_256k-chunk8192-text-8x4]' -sv
```

Run sequentially with cached weights. For the original-gather variants, restore
`tt/ccl.py` and `tt/model.py` under `models/demos/gemma4/` from the baseline,
keeping the fabric and ring-attention changes. For Y-only, also change
`FABRIC_2D_TORUS_XY` to `FABRIC_2D_TORUS_Y` in `tests/test_factory.py`.

One full run per configuration; all four passed. Device totals above sum the
32 logged chunk timings and exclude loading, compilation, staging and readback.
The test checks output finiteness, not numerical parity. No statistical
confidence or generalization to other meshes is claimed. The patch also passed
25 host tests in `test_ccl_topology.py` and applicable pre-commit checks.
Raw logs remain on the measurement host under `/tmp/gemma4-torus-ab/`
(`baseline.log`, `torus.log`, `torus_sync_gathers.log`,
`torus_y_sync_gathers.log`); they are not included in this branch.
