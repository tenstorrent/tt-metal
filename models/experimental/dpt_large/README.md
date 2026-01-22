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

## Tests

CPU-only smoke test:

```sh
pytest -q models/experimental/dpt_large/tests/test_e2e_fallback.py
```

TT parity tests (require hardware + opt-in env vars):

```sh
DPT_RUN_TT_TESTS=1 pytest -q models/experimental/dpt_large/tests/test_e2e_tt_vs_cpu.py
DPT_RUN_TT_BACKBONE_PARITY=1 pytest -q models/experimental/dpt_large/tests/test_backbone_tt_vs_cpu_small.py
```
