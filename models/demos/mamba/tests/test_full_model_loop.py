# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from transformers import AutoTokenizer
from typing import Optional
import ttnn
from models.demos.mamba.reference.decode_model import MambaPretrainedModelName
from models.demos.mamba.tests.test_full_model import run_inference
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Not supported on Grayskull")
def test_inference_loop(
    device: ttnn.Device,
    use_program_cache,
    model_version="state-spaces/mamba-2.8b",
    batch=32,
    pcc=0.92,
    cache_dir=None,
    num_layers=64,
    iterations=10,
):
    run_inference(device, use_program_cache, model_version, batch, pcc, cache_dir, num_layers, iterations)
