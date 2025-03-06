# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.utility_functions import run_for_wormhole_b0

from models.demos.squeezebert.tests.squeezebert_performant import (
    run_squeezebert_inference,
    run_squeezebert_trace_inference,
)


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    ((8, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
def test_run_squeezebert_inference(
    device, use_program_cache, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator
):
    run_squeezebert_inference(device, batch_size, act_dtype, weight_dtype, math_fidelity, model_location_generator)


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 800768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    ((8, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
@pytest.mark.parametrize("enable_async_mode", (False,), indirect=True)
def test_run_squeezebert_trace_inference(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    enable_async_mode,
    model_location_generator,
):
    run_squeezebert_trace_inference(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        model_location_generator,
    )
