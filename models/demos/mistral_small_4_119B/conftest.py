# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pytest bridge for Mistral Small 4 demo tests.

Real fixture/hook implementations live in ``tt_utils/conftest.py``.
This file exists at the demo root so pytest auto-discovers it for all tests under ``tests/``.
"""

from models.demos.mistral_small_4_119B.tt_utils import conftest as _tt_utils_conftest

pytest_addoption = _tt_utils_conftest.pytest_addoption
pytest_configure = _tt_utils_conftest.pytest_configure
pytest_collection_modifyitems = _tt_utils_conftest.pytest_collection_modifyitems
mistral_snapshot_dir = _tt_utils_conftest.mistral_snapshot_dir
_mistral_text_config_store = _tt_utils_conftest._mistral_text_config_store
mistral_text_config = _tt_utils_conftest.mistral_text_config
mistral_sharded_checkpoint = _tt_utils_conftest.mistral_sharded_checkpoint
