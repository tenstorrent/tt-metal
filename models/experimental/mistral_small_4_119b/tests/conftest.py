# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pin HF checkpoint for all tests in this package (ModelArgs reads HF_MODEL)."""

import pytest

from models.experimental.mistral_small_4_119b.constants import HF_MODEL_ID


@pytest.fixture(autouse=True)
def _mistral_small_4_hf_model_env(monkeypatch):
    monkeypatch.setenv("HF_MODEL", HF_MODEL_ID)
