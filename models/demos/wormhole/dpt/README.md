## DPT-Large (Depth Estimation) — Wormhole Demo

This mirrors the ViT example layout:

- `tt/` – TTNN implementation modules (backbone, neck, head, traced/2CQ)
- `demo/` – runnable scripts
- `tests/` – unit/e2e/perf tests

### Quickstart

Run a single image on device, traced:

```
python -m models.demos.wormhole.dpt.demo.demo_dpt_inference --image path/to/img.jpg --tt-run --traced --device-id 0 --output out.png
```

Benchmark full pipeline:

```
python -m models.demos.wormhole.dpt.demo.demo_dpt_inference --benchmark --traced --warmup 5 --repeat 50
```

PCC vs reference (uses shared `comp_pcc`):

```
python -m models.demos.wormhole.dpt.demo.demo_dpt_inference --pcc-eval
```

### Tests

```
pytest -q models/demos/wormhole/dpt/tests/test_dpt_backbone.py::test_pcc_vs_reference
pytest -q models/demos/wormhole/dpt/tests/test_dpt_backbone.py::test_full_pipeline_smoke
```

