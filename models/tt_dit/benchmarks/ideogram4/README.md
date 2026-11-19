# Ideogram 4 benchmarks

Manual, env-driven benchmarks for the Ideogram 4 pipeline on a BH Galaxy (4x8). They are not part of CI:
each test skips unless its input environment variables are set, and each module docstring lists them.

| File | Measures |
|---|---|
| `test_perf_ideogram4_gpu_bench.py` | Latency matched to the GPU benchmark's configuration (caption, sizes, presets, repetitions) |
| `test_perf_ideogram4_resolution_matrix.py` | Latency and images across the API's output sizes and presets |
| `test_ideogram4_prompt_batch.py` | Images for a batch of captions from one shared initial latent, so only the prompt varies |

Run one with pytest, for example:

```bash
IDEOGRAM4_WEIGHTS=ideogram-ai/ideogram-4-fp8 \
  pytest models/tt_dit/benchmarks/ideogram4/test_perf_ideogram4_gpu_bench.py
```

The CI tests (`test_perf_ideogram4.py`, `test_serving_ideogram4.py`) stay in `models/tt_dit/tests/models/ideogram4/`.
