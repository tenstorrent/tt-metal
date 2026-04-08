# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

# Modified test_stability.py using TT-CNN pipeline
import time
import pytest
from loguru import logger
from tqdm import tqdm

from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.mobileNetV3.runner.performant_runner_infra import MobileNetV3PerformanceRunnerInfra
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config
import ttnn


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1702912, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("resolution", [(224, 224)])
@pytest.mark.parametrize("test_duration", [5])
@pytest.mark.parametrize("pcc_check_interval", [5])
def test_MobileNetV3_stability(
    device,
    batch_size,
    model_location_generator,
    resolution,
    test_duration,
    pcc_check_interval,
):
    test_infra = MobileNetV3PerformanceRunnerInfra(
        device,
        batch_size,
        resolution=resolution,
        model_location_generator=None,
        input_path=".models/experimental/mobileNetV3/resources/dog.jpeg",
    )

    tt_inputs_host, dram_input_mem_config, l1_input_mem_config, channels = test_infra.setup_sharded_input(device)

    original_height = resolution[0]
    original_width = resolution[1]
    batch_per_device = tt_inputs_host.shape[2] // (original_height * original_width)

    def model_wrapper(input_tensor):
        reshaped_input = ttnn.reshape(input_tensor, (batch_per_device, original_height, original_width, channels))
        test_infra.input_tensor = reshaped_input
        test_infra.run()
        return test_infra.tt_output

    pipeline = create_pipeline_from_config(
        device=device,
        model=model_wrapper,
        config=PipelineConfig(
            use_trace=True,
            num_command_queues=2,
            all_transfers_on_separate_command_queue=False,
        ),
        dram_input_memory_config=dram_input_mem_config,
        l1_input_memory_config=l1_input_mem_config,
    )

    pipeline.compile(tt_inputs_host)

    logger.info(f"Running stability test for MobileNetV3 with resolution: {resolution} and batch size: {batch_size}")

    pcc_iter = 0
    start_time = time.time()

    with tqdm(total=test_duration, desc="Executing on device", unit="sec", mininterval=1) as pbar:
        while True:
            elapsed_time = round(time.time() - start_time, 1)
            pbar.update(min(elapsed_time, test_duration) - pbar.n)

            if elapsed_time >= test_duration:
                break

            check_pcc = elapsed_time >= pcc_iter * pcc_check_interval
            if check_pcc:
                pcc_iter += 1

            # Run through pipeline
            outputs = list(pipeline.enqueue([tt_inputs_host]).pop_all())

            if check_pcc:
                for i, output in enumerate(outputs):
                    test_infra.validate(output)
                    logger.info(f"Output {i} validation passed")

    pipeline.cleanup()
