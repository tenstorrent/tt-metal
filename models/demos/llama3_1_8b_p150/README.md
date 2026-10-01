<!-- SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# llama3_1_8b_p150 — fixed-identity single-model wrapper

This directory is a **thin, fixed-identity wrapper** over the tt_transformers
Llama implementation, pinned to **`meta-llama/Llama-3.1-8B-Instruct`** on a
**P150** device. It exists so the perf_automation optimize tool has a
single-model target with a fixed identity — `models/tt_transformers` itself is
rejected because it is a multi-model framework whose model is chosen at runtime
via `HF_MODEL`.

Everything here pins the identity at invocation:

- `tt/pipeline.py` — re-exports (by import, never copied) the tt_transformers
  Llama model builder + decode-forward entry; `HF_MODEL` pinned at module scope.
- `demo/demo.py` — hard-codes `HF_MODEL=meta-llama/Llama-3.1-8B-Instruct` and
  delegates to the proven `simple_text_demo` performance run.
- `tests/e2e/conftest.py` — sets `HF_MODEL` at collection so the identity is
  fixed the instant the tool inspects this target.
- `tests/e2e/test_pcc.py` — the e2e correctness gate. Single hard-coded floor
  (`LLAMA31_8B_TOP1_MIN = 0.86`), no per-model lookup; reuses the upstream
  `TokenAccuracy` token-matching logic (which loads
  `Llama-3.1-8B-Instruct.refpt`).

## The OPTIMIZABLE source lives upstream

This wrapper contains **no model math to tune**. The files the optimize tool
should mutate all live in `models/tt_transformers/tt/`:

- `models/tt_transformers/tt/model_config.py`
- `models/tt_transformers/tt/attention.py`
- `models/tt_transformers/tt/mlp.py`
- `models/tt_transformers/tt/lm_head.py`

## Proven on-device entry (≈17.55 tok/s on P150)

```bash
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
MESH_DEVICE=P150 \
TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
pytest models/tt_transformers/demo/simple_text_demo.py::test_demo_text -k "performance-batch-1"
```

The demo (`demo/demo.py`) and the e2e gate (`tests/e2e/test_pcc.py`) both
delegate to this same upstream flow with the identity pre-pinned.
