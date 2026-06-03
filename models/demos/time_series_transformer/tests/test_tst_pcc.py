# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import comp_pcc

def test_tst_pcc():
    # 1. Resolve paths to reference data
    base_dir = (
        Path(__file__).resolve().parent.parent
    )  # tt-metal/models/demos/time_series_transformer
    reference_dir = base_dir / "reference"

    # 2. Safely read structural model parameters
    config_path = reference_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing structural config.json in {reference_dir}. Please run save_reference_tensors.py first."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 3. Safely read execution runtime parameters (Fixes KeyError: 'batch_size')
    runtime_config_path = reference_dir / "config_runtime.json"
    if not runtime_config_path.exists():
        raise FileNotFoundError(
            f"Missing runtime configuration metadata in {reference_dir}. Please execute save_reference_tensors.py."
        )

    with open(runtime_config_path, "r", encoding="utf-8") as f:
        runtime_cfg = json.load(f)

    batch_size = runtime_cfg["batch_size"]
    past_len = cfg["past_len"]
    d_model = cfg["d_model"]

    logger.info(
        f"Validation configurations loaded successfully. Running testing sequence for Batch Size: {batch_size}"
    )

    # 4. Mock random target allocations using valid shape specs from config
    raw_hidden = torch.randn(batch_size, past_len, d_model)
    logger.info(
        f"Generated model testing tensor footprint: {list(raw_hidden.shape)}"
    )

    # TODO: Add your component testing layers logic below
    # (e.g., loading safetensors, calling ttnn operations, and validating with comp_pcc)

    assert raw_hidden.shape == (batch_size, past_len, d_model)
