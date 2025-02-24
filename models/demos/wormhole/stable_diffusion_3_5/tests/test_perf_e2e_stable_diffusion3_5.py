# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import time

from models.utility_functions import run_for_wormhole_b0
from models.demos.wormhole.stable_diffusion_3_5.tests.perf_e2e_stable_diffusion3_5 import SD35mTrace  # run_perf_sd35,


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 13631488, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size",
    ((1),),
)
# @pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_run_sd35m_trace_inference(
    device,
    use_program_cache,
    batch_size,
    enable_async_mode=True,
    model_location_generator=None,
):
    sd35m_trace = SD35mTrace()

    sd35m_trace.initialize_sd35m_trace_inference(
        device,
        batch_size,
        model_location_generator=None,
    )
    for iter in range(10):
        t0 = time.time()
        output = sd35m_trace.execute_sd35m_trace_inference()
        t1 = time.time()
        print("TIME", t1 - t0)

    sd35m_trace.release_sd35m_trace_inference()
