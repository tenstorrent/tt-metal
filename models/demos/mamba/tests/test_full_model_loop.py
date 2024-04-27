# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.demos.mamba.tests.test_full_model import run_inference
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Not supported on Grayskull")
def test_inference_loop(
    device: ttnn.Device,
    use_program_cache,
    model_version="state-spaces/mamba-2.8b",
    batch=32,
    pcc=0.88,
    cache_dir=None,
    num_layers=64,
    iterations=10,
):
    run_inference(device, use_program_cache, model_version, batch, pcc, cache_dir, num_layers, iterations)
