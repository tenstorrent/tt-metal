<!--
SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0
-->

# DPT-Large (MiDaS 3.0) — Experimental

This implementation follows the standard model layout:

- `tt/` – TTNN + reference pipeline modules (backbone/neck/head, configs, weights)
- `demo/` – runnable scripts
- `tests/` – unit/e2e tests

## Demo

CPU fallback run (downloads HF weights on first run):

```sh
python -m models.experimental.dpt_large.demo.runner --image path/to/image.jpg --device cpu
```

TT run (requires a TT device):

```sh
python -m models.experimental.dpt_large.demo.runner --image path/to/image.jpg --tt-run --device wormhole_n300
```

Dump depth outputs:

```sh
python -m models.experimental.dpt_large.demo.runner \
  --image path/to/image.jpg \
  --tt-run --device wormhole_n300 \
  --dump-depth generated/depth.npy \
  --dump-depth-color generated/depth.png
```

TT run with perf reporting (writes a perf JSON and an adjacent header JSON):

```sh
python -m models.experimental.dpt_large.demo.runner \
  --image path/to/image.jpg \
  --tt-run --device wormhole_n300 \
  --dump-perf generated/dpt_large_perf.json
```

N300 data-parallel throughput run (2 chips in a `1x2` mesh, B=1 per chip), using traced execution:

```sh
python -m models.experimental.dpt_large.demo.runner \
  --images-dir /tmp/dpt_eval_imgs \
  --tt-run --device wormhole_n300 \
  --image-size 384 \
  --dp 2 --batch-size 2 \
  --tt-execution-mode trace_2cq --warmup 8 --repeat 30 \
  --dump-perf /tmp/dpt_runner_perf_dp.json
```

## Accuracy + Throughput (PCC/FPS)

Compute PCC (TT vs CPU) and per-image FPS on a folder of images:

```sh
python -m models.experimental.dpt_large.demo.eval_pcc \
  --images-dir /tmp/dpt_eval_imgs \
  --tt-run --device wormhole_n300 \
  --image-size 384 \
  --dp 2 --batch-size 2 \
  --dump-json /tmp/dpt_eval.json
```

### Perf Metrics (Device vs End-to-End)

For `--tt-execution-mode trace` / `trace_2cq`, the perf JSON reports two FPS values:

- `fps`: Stage-1 throughput metric (device-only). This is `fps_device`.
- `fps_e2e`: End-to-end FPS including readback and CPU normalization.

The device-only timing is derived from `trace_wall_ms`, which is measured using a host-visible completion
barrier (`ttnn.synchronize_device`) so it is stable across runs and does not include host jitter from
`cpu().to_torch()` and normalization.

When `--dump-perf` is enabled, the runner enables strict mode and fails if any TTNN program-config fast path
throws and falls back silently. In perf runs, `program_config_fallback_total` must be `0`.

Run CPU-only (no TT device) to get a baseline:

```sh
python -m models.experimental.dpt_large.demo.eval_pcc \
  --images-dir path/to/images \
  --device cpu \
  --dump-json generated/dpt_large_cpu_eval.json
```

## Tests

CPU-only smoke test:

```sh
pytest -q models/experimental/dpt_large/tests/test_e2e_fallback.py
```

Hardware PCC test (TT vs CPU reference):

```sh
pytest -q models/experimental/dpt_large/tests/test_tt_pcc.py
```

Hardware perf smoke test (writes `generated/dpt_large_tt_perf_smoke.json`):

```sh
pytest -q models/experimental/dpt_large/tests/test_tt_perf.py -s
```
