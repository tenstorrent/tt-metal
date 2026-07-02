# Granite TTM-R1 on Wormhole

Wormhole bring-up for `ibm-granite/granite-timeseries-ttm-r1`.

## Tests

All tests live in `models/demos/granite_ttm_r1/tests/` following the Informer PR pattern:

| Directory | Description |
|---|---|
| `tests/pcc/` | Per-component PCC >= 0.99 validation |
| `tests/perf/` | Throughput, latency, model size |
| `tests/accuracy/` | Zero-shot ETTh1 benchmark (marked `slow`) |

## Quick Reference

```bash
# PCC tests (require attached Wormhole device)
pytest models/demos/granite_ttm_r1/tests/pcc/ -v

# Performance tests
pytest models/demos/granite_ttm_r1/tests/perf/ -v

# Model size (no device)
pytest models/demos/granite_ttm_r1/tests/perf/test_perf.py::test_model_size -v
```

## See Also

- Main model directory: `models/demos/granite_ttm_r1/`
- GitHub issue: https://github.com/tenstorrent/tt-metal/issues/32142
