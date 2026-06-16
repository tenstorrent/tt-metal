# Profiling recipe — HW perf counters

Authoritative doc lives in-repo: `tt_metal/tools/profiler/PERF_COUNTERS.md`.
This is the quick personal index.

## Trust the numbers (run first)

```bash
pytest tests/ttnn/tracy/test_counter_utilization_sanity.py -v
```

Grid-normalized FPU counter must match the achieved fraction of the analytical
peak. If it doesn't, the util math is wrong — fix it before believing any
downstream number. Validated bh 4x8: FPU 21.9% vs achieved 21.5%.

## One-command capture (pinned buffer + archive)

```bash
python -m tracy.capture_counters \
  --test "<pytest target>" \
  --groups fpu,instrn --programs-per-device <ops/device> [--compute-core-sample 3]
```

- Reuse the SAME `--programs-per-device` across runs → stable kernel hash →
  100% JIT cache. Changing it = ~14 min instrumented recompile.
- Compute-core sample shrinks post-process and is repeatable; preserves per-core
  util %, not grid-summed metrics. Never sample NoC counters.
- Artifacts archived to `~/traces/<ts>/` (reports dir wipes each run).

## Sizing helper

```python
from tracy.perf_counter_sizing import markers_per_zone, recommend_program_support_count
```

## Host-side post-process tests (no device)

```bash
PYTHONPATH=tools pytest tests/ttnn/tracy/test_counter_buffer_sizing.py \
  tests/ttnn/tracy/test_counter_scope.py tests/ttnn/tracy/test_counter_capture_plan.py -q
```
