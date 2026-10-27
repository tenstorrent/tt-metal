# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Every model's settings and serving adapter still import after a framework change.

The DeepSeek-V3 test folder's conftest loads every registered adapter (models/demos/common/prefill/adapter.py), so one
model's stale settings file stops every test there from loading: F137 removed Settings(max_precision) and missed
xing40_a4b_d_p, which broke the forks' source tests that live in that folder (combine, dispatch, offset_cumsum) for
six days (F198)."""

import importlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
SETTINGS = sorted(p.relative_to(REPO) for p in REPO.glob("models/demos/*/tt/settings.py"))


@pytest.mark.parametrize("path", SETTINGS, ids=[p.parts[2] for p in SETTINGS])
def test_every_models_settings_file_imports(path):
    importlib.import_module(".".join(path.with_suffix("").parts))


def test_every_registered_serving_adapter_loads():
    from models.demos.common.prefill.adapter import ADAPTER_PATHS, get_adapter

    for name in ADAPTER_PATHS:
        get_adapter(name)
