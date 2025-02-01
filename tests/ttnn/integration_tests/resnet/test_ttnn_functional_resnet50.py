# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from loguru import logger

from models.demos.ttnn_resnet.tests.resnet50_test_infra import create_test_infra
from models.utility_functions import (
    run_for_blackhole,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    (
        (16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
        (16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),
    ),
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [True, False],
    ids=[
        "pretrained_weight_true",
        "pretrained_weight_false",
    ],
)
def test_resnet_50(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    use_pretrained_weight,
    model_location_generator,
):
    test_infra = create_test_infra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        use_pretrained_weight,
        model_location_generator=model_location_generator,
    )
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    # First run configures convs JIT
    test_infra.run()
    # Optimized run
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    # More optimized run with caching
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    passed, message = test_infra.validate()
    assert passed, message
