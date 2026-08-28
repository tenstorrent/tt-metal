# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bridge the bring-up test fixtures into `demo/`.

`serve_mistral4_interactive.py` builds the served model through the same `run_model` the bring-up
tests use, so it needs their fixtures. Those live in `../tests/conftest.py`, which pytest only
applies to tests under `tests/` — a sibling `demo/` module gets none of them and fails at setup with
"fixture 'variant' not found". (Collection succeeds either way, so `--collect-only` does not catch
this; only an actual run does.)

Importing the fixture functions into this conftest's namespace is enough for pytest to find them.
Deliberately an explicit list rather than a star-import: `tests/conftest.py` also defines pytest
hooks and command-line options, and pulling those into a second conftest would register them twice.

The list is dependency-closed — `model_path` and `tokenizer` are here because `variant`'s dependents
need them (and `run_model` resolves `model_path` at runtime via `request.getfixturevalue`), not
because the demo names them directly. Everything else it uses (`mesh_device`, `device_params`,
`is_ci_env`, `is_ci_v2_env`) is provided by the global tt-metal conftest.
"""

from models.demos.deepseek_v3_d_p.tests.conftest import (  # noqa: F401
    config_only,
    model_path,
    tokenizer,
    variant,
    weight_cache_path,
)
