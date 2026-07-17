<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# tt-emule end-to-end model tests

Emule-owned wrappers that drive tt-metal model demos end-to-end under software
emulation, without editing the upstream demo (we are codeowners of `tests/emule/`
only).

## What's here

- `test_tt_transformers_text_demo.py` — a **vendored copy** of
  `models/tt_transformers/demo/simple_text_demo.py`. The only deltas from
  upstream are two lines tagged `# emule:` (see the file header):
  1. `_supports_on_device_sampling = False` — force host sampling.
  2. `enable_trace = False` — emule is slow-dispatch / teacher-forcing only.
- `conftest.py` — vendored copy of the demo's options-only conftest, so the
  `request.config.getoption(...)` calls resolve. All fixtures come from the root
  tt-metal `conftest.py`.

These are exercised by the tt-emule runner scripts
(`tt-emule/scripts/run_e2e_models_<arch>.sh`) and the nightly CI job
(`e2e-models` in `tt-emule/.github/workflows/nightly-metal-upstream.yml`).

## Running locally (wormhole N150)

```bash
export TT_METAL_DIR=$(git rev-parse --show-toplevel)   # this tt-metal checkout
HF_MODEL=unsloth/Llama-3.2-1B-Instruct MESH_DEVICE=N150 \
TT_METAL_EMULE_MODE=1 TT_METAL_SLOW_DISPATCH_MODE=1 \
PYTHONPATH=$TT_METAL_DIR/ttnn:$TT_METAL_DIR \
  pytest -v $TT_METAL_DIR/tests/emule/models/test_tt_transformers_text_demo.py::test_demo_text \
  -k "performance-ci-token-matching"
```

Or via the tt-emule runner (recommended — sets all env + curated entries):
`TT_METAL_DIR=$TT_METAL_DIR bash tt-emule/scripts/run_e2e_models_wormhole.sh`.

## Adding a model

- **Same demo, new model**: no code — add a `run_model` entry in
  `run_e2e_models_<arch>.sh` with a different `HF_MODEL` and `-k` selector.
- **Different demo / model family**: vendor that demo the same way (copy + its
  `# emule:` edits + header), add a matching runner entry.

## Re-syncing on a tt-metal pin bump

The vendored copy tracks upstream only as of the pin SHA recorded in the file
header. On a pin bump, re-run the `git show <new-pin>:...` copy and re-apply the
two `# emule:` markers. This is a `/uplift` checklist item — the demo is not in
the C++ regression, so upstream drift is otherwise uncaught.
