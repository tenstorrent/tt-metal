# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Package-level pytest fixtures for `models/demos/llama31_8b_d_p`.

Shape copied from `models/demos/minimax_m3/conftest.py`: a session-scoped `state_dict` fixture and a
`--skip-model-load` option, so a whole run pays the checkpoint load at most once and can opt out of
it entirely.

`mesh_device` and `reset_seeds` are NOT redefined here — they come from the repo root
`conftest.py:554` and `conftest.py:34` respectively (recipe P1 step 3).
"""

import os

import pytest

from models.demos.llama31_8b_d_p.tests.test_factory import load_hf_state_dict


def pytest_addoption(parser):
    parser.addoption("--skip-model-load", action="store_true", default=False, help="Skip loading the model state dict")


@pytest.fixture(scope="session")
def state_dict(request):
    """The full checkpoint state dict, or `{}` when `HF_MODEL` is unset or `--skip-model-load`.

    `models/demos/minimax_m3/conftest.py:22` routes this through `ModelArgs.load_state_dict`; this
    package has no `ModelArgs` until P6.2, so the load goes straight to the safetensors shards
    (`DEC-012`). Tensors are returned at the checkpoint dtype (bf16) — casting is the caller's job,
    per the reference dtype policy (`bringup_log/01_REFERENCE.md` §3).
    """
    load_model = not request.config.getoption("--skip-model-load")
    model_path = os.getenv("HF_MODEL", None)
    if model_path is None or not load_model:
        return {}
    return load_hf_state_dict(model_path=model_path)
