# Local perf / accuracy target validation (opt-in)

In CI, model e2e tests are **warning-only** for perf/accuracy: `verify_perf` /
`verify_accuracy` log drift but never abort, so the run always writes complete
benchmark JSON. A separate CI step then enforces the targets in
[`models/model_targets.yaml`](../../model_targets.yaml) via
`.github/scripts/utils/validate_perf_targets.py`.

Known measurements are printed even when the matching entry is missing or
`status: TODO` (table status `no-target`). An entry may set `enforce_perf: false`
(the default is `true`) so perf is compared and printed as `report-only` without
failing the job. Tier 3 e2e uses that flag. Accuracy on an active entry still
fails the job.

You can run that **same** enforcement locally as an opt-in pytest gate.

## How to enable

Run your model demo/test with `VALIDATE_PERF_TARGETS=1`:

```bash
VALIDATE_PERF_TARGETS=1 pytest models/tt_transformers/demo/simple_text_demo.py -k "..."
```

A `pytest_sessionfinish` hook (in `models/conftest.py`) runs **after** the
session — once benchmark JSON is fully written — loads the CI validator, and
**fails the session** if any measured metric misses its target. It is off by
default, so a normal `pytest` run is unaffected.

## Requirements

- The test must emit benchmark data (it calls `create_benchmark_data(...).save(...)`,
  writing `generated/benchmark_data/complete_run_*.json`).
- An entry for your `model` + `sku` must exist in `models/model_targets.yaml`.

## Knobs

| Env var | Default | Effect |
| --- | --- | --- |
| `VALIDATE_PERF_TARGETS` | `0` | `1` enables the local gate. |
| `VALIDATE_PERF_TARGETS_STRICT_MISSING` | `0` | `1` also fails when no target entry exists (TODO/missing); otherwise missing targets only warn. |
| `VALIDATE_PERF_TARGETS_SKU` | _(auto)_ | Override the SKU. If unset, it is auto-detected from the device, falling back to the `card_type` in the benchmark JSON. |

## Notes

- Same comparison logic as CI (single source of truth) — a local pass means a CI pass for those metrics.
- Validation runs once at session end, so it never truncates a test mid-run.
