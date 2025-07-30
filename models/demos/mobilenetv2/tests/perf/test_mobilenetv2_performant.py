# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.mobilenetv2.tests.perf.mobilenetv2_common import MOBILENETV2_L1_SMALL_SIZE
from models.demos.mobilenetv2.tests.perf.mobilenetv2_performant import (
    run_mobilenetv2_inference,
    run_mobilenetv2_trace_2cqs_inference,
    run_mobilenetv2_trace_inference,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": MOBILENETV2_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("device_batch_size", [(1)])
def test_run_mobilenetv2_inference(
    device,
    device_batch_size,
    model_location_generator,
):
    run_mobilenetv2_inference(
        device,
        device_batch_size,
        model_location_generator,
    )


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": MOBILENETV2_L1_SMALL_SIZE, "trace_region_size": 3686400}], indirect=True
)
@pytest.mark.parametrize("device_batch_size", [(1)])
def test_run_mobilenetv2_trace_inference(
    device,
    device_batch_size,
    model_location_generator,
):
    run_mobilenetv2_trace_inference(
        device,
        device_batch_size,
        model_location_generator,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": MOBILENETV2_L1_SMALL_SIZE, "trace_region_size": 3686400, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("device_batch_size", [(1)])
def test_run_mobilenetv2_trace_2cq_inference(device, device_batch_size, model_location_generator):
    run_mobilenetv2_trace_2cqs_inference(
        device=device,
        device_batch_size=device_batch_size,
        model_location_generator=model_location_generator,
    )
