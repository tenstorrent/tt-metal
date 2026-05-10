# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight checks for ModelArgs registration of Mistral Small 4 (no mesh / no weights)."""

from models.tt_transformers.tt.common import get_base_model_name
from models.tt_transformers.tt.model_config import ModelArgs


def test_hf_tail_maps_to_expected_base_model_name():
    assert get_base_model_name("Mistral-Small-4-119B-2603") == "Mistral-Small-4-119B"


def test_local_hf_params_includes_small_4_dummy_entry():
    assert "Mistral-Small-4-119B-2603" in ModelArgs.LOCAL_HF_PARAMS
    assert "mistralai/Mistral-Small-4-119B-2603" in ModelArgs.LOCAL_HF_PARAMS["Mistral-Small-4-119B-2603"]
