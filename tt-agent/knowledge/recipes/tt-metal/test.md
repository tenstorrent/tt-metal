# Run tt-metal model tests

## Unit test (single module)

```bash
pytest models/tt_transformers/tests/test_mlp.py --timeout 300
```

## E2E demo

```bash
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-token-matching" --timeout 420
```

## Multi-device test

```bash
MESH_DEVICE=T3K pytest models/demos/deepseek_v3/tests/test_mla.py -k "mode_decode" --timeout 900
```

## Architecture selection

Auto-detected from hardware. Override with:

```bash
pytest <test_path> --tt-arch wormhole_b0
```

## Key fixtures (from root conftest.py)

- `mesh_device` — multi-device mesh. Parametrize indirectly: `@pytest.mark.parametrize("mesh_device", [2], indirect=True)`
- `device_params` — dict with keys like `fabric_config`, `trace_region_size`. Parametrize indirectly.
- `reset_seeds` — seeds torch/numpy/random to 213919
- `ensure_gc` — autouse garbage collection

## Test structure pattern

1. `@torch.no_grad()` decorator
2. Parametrize `mesh_device` and `device_params`
3. Build reference PyTorch model, run forward
4. Build TT model, run forward
5. `comp_pcc(ref_output, tt_output, 0.99)` to validate

## Useful markers

- `@pytest.mark.timeout(1800)` — override default 300s timeout
- `@pytest.mark.models_performance_bare_metal` — perf test category
- `-k "performance"` — filter to performance tests
