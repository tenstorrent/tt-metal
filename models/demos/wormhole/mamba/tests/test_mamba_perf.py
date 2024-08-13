# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import (
    profiler,
    disable_persistent_kernel_cache,
    skip_for_grayskull,
)
from tt_metal.tools.profiler.process_model_log import get_samples_per_s

from models.demos.wormhole.mamba.reference.prefill_decode_model import Mamba
from models.demos.wormhole.mamba.tt import model_config
from models.demos.wormhole.mamba.tt.model_config import ModelMode
from models.demos.wormhole.mamba.tt.mamba_model import MambaTT


def is_nearby(actual: float, expected: float, lower_margin: float = 0.03, upper_margin: float = 0.03):
    lower_threshold = (1 - lower_margin) * expected
    upper_threshold = (1 + upper_margin) * expected
    return lower_threshold <= actual <= upper_threshold


NUM_LAYERS = 64
MARGIN = 0.05


@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "model_version, mode, batch_size, sequence_length, iterations, expected_compile_time, expected_inference_time",
    (
        ("state-spaces/mamba-2.8b", ModelMode.DECODE, 32, 1, 8, 12.50, 0.110),
        ("state-spaces/mamba-2.8b", ModelMode.PREFILL, 1, 128, 8, 23.50, 0.520),
    ),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_mamba_perf_e2e(
    device,
    model_version,
    mode,
    batch_size,
    sequence_length,
    iterations,
    expected_compile_time,
    expected_inference_time,
    use_program_cache,
    reset_seeds,
    get_tt_cache_path,
    is_ci_env,
):
    profiler.clear()

    logger.info(
        f"Testing end-to-end performance in {'PREFILL' if mode == ModelMode.PREFILL else 'DECODE'} mode with sequence length {sequence_length}"
    )

    logger.warning(f"Disabling persistent kernel cache due to hang on CI (#8606)")
    disable_persistent_kernel_cache()

    profiler.start(f"initialize_ref_model")
    reference_model = Mamba.from_pretrained(model_version, batch_size=batch_size)
    reference_model.args.mode = mode
    reference_model.eval()
    profiler.end(f"initialize_ref_model")

    if mode == ModelMode.DECODE:
        assert sequence_length == 1, "Sequence length must be 1 in decode mode"
        assert batch_size == 32, "Batch size must be 1 in decode mode"
    else:
        assert batch_size == 1, "Batch size must be 1 in prefill mode"

    profiler.start(f"initialize_model")
    config = model_config.create_model_config(
        batch_size, reference_model.args.d_model, mode=mode, seq_len=sequence_length
    )
    model = MambaTT(
        reference_model, device, config, tt_cache_path=get_tt_cache_path(model_version), num_layers=NUM_LAYERS
    )
    profiler.end(f"initialize_model")
    logger.info(f"Done initializing model in {profiler.get('initialize_model'):.2f} s")

    input = torch.randn((batch_size, sequence_length))
    logger.info(f"Measuring performance with input of shape {list(input.shape)}")

    logger.info(f"Compiling model with warmup run")
    profiler.start(f"inference_and_compile_time")
    model(input)
    profiler.end(f"inference_and_compile_time")

    inference_and_compile_time = profiler.get("inference_and_compile_time")
    logger.info(f"Model compiled with warmup run in {(inference_and_compile_time):.2f} s")

    logger.info(f"Running inference for {iterations} iterations")
    for idx in range(iterations):
        profiler.start("inference_time")
        profiler.start(f"inference_time_{idx}")
        model(input)
        profiler.end(f"inference_time_{idx}")
        profiler.end("inference_time")

    mean_inference_time = profiler.get("inference_time")
    inference_time = profiler.get(f"inference_time_{iterations - 1}")
    compile_time = inference_and_compile_time - inference_time
    logger.info(f"Inference time on last iterations was completed in {(inference_time * 1000.0):.2f} ms")
    logger.info(f"Mean inference time was {(mean_inference_time * 1000.0):.2f} ms")
    logger.info(f"Model compilation took {compile_time:.2f} s")

    comment = (
        f"mode-{'prefill' if mode == ModelMode.PREFILL else 'decode'}_layers-{NUM_LAYERS}_seqlen-{sequence_length}"
    )
    prep_perf_report(
        model_name=f"{model_version}",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comment,
    )

    lower_margin = MARGIN if is_ci_env else 1.0  # CI machines are generally slower
    upper_margin = MARGIN
    if not is_nearby(inference_time, expected_inference_time, lower_margin=lower_margin, upper_margin=upper_margin):
        logger.warning(
            "Inference time does not match (within some margin) the expected value (was {inference_time:2f} but expected {expected_inference_time:2f})"
        )

    if not is_nearby(compile_time, expected_compile_time, lower_margin=lower_margin, upper_margin=upper_margin):
        logger.warning(
            f"Compile time does not match (within some margin) the expected value (was {compile_time:2f} but expected {expected_compile_time:2f})"
        )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(600)
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch, warmup, expected_device_fw_duration_ms",
    ((32, True, 1.66),),
)
def test_mamba_perf_device(batch, warmup, expected_device_fw_duration_ms, reset_seeds):
    subdir = "ttnn_mamba"
    margin = 0.03
    if warmup:
        inference_iterations = 2
    else:
        inference_iterations = 1
    command = f"pytest models/demos/wormhole/mamba/tests/test_mamba_model.py::test_device_perf[{inference_iterations}]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    # Convert expected perf (ms) to samples/s
    expected_device_fw_duration_ns = expected_device_fw_duration_ms * 1e6  # ms to ns
    expected_total_device_fw_samples = get_samples_per_s(expected_device_fw_duration_ns * inference_iterations, batch)

    inference_time_key = "AVG DEVICE FW SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_total_device_fw_samples}

    post_processed_results = run_device_perf(command, subdir, 1, cols, batch)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)
    comment = ""
    prep_device_perf_report(
        model_name=f"mamba-2.8b_batch_{batch}",
        batch_size=batch,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=comment,
    )
