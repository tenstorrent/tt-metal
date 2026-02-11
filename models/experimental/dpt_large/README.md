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

TT run with perf reporting (writes a perf JSON and an adjacent header JSON):

```sh
python -m models.experimental.dpt_large.demo.runner \
  --image path/to/image.jpg \
  --tt-run --device wormhole_n300 \
  --dump-perf generated/dpt_large_perf.json
```

## Accuracy + Throughput (PCC/FPS)

Compute PCC (TT vs CPU) and per-image FPS on a folder of images:

```sh
python -m models.experimental.dpt_large.demo.eval_pcc \
  --images-dir path/to/images \
  --tt-run --device wormhole_n300 \
  --dump-json generated/dpt_large_eval.json
```

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
