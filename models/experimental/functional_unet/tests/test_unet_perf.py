# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_unet.tt.model_preprocessing import (
    create_unet_input_tensors,
    create_unet_model_parameters,
)
from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn
from models.experimental.functional_unet.tests.common import (
    check_pcc_conv,
    is_n300_with_eth_dispatch_cores,
    is_t3k_with_eth_dispatch_cores,
)

from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import (
    profiler,
    skip_for_grayskull,
)


def synchronize_devices(device):
    devices = device.get_devices()
    for device in devices:
        ttnn.synchronize_device(device)


@skip_for_grayskull("UNet not currently supported on GS")
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch, groups, expected_device_perf_fps",
    ((2, 1, 656.0),),
)
def test_unet_perf_device(batch: int, groups: int, expected_device_perf_fps: float, reset_seeds):
    command = f"pytest models/experimental/functional_unet/tests/test_unet_model.py::test_unet_model[device_params0-{groups}-{batch}]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE FW SAMPLES/S"
    post_processed_results = run_device_perf(
        command, subdir="unet_shallow", num_iterations=1, cols=cols, batch_size=batch
    )
    expected_perf_cols = {inference_time_key: expected_device_perf_fps}
    expected_results = check_device_perf(
        post_processed_results, margin=0.02, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )
    prep_device_perf_report(
        model_name=f"unet-shallow_batch-{batch}_groups-{groups}",
        batch_size=batch,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )


@skip_for_grayskull("UNet not currently supported on GS")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize(
    "batch, groups, iterations, expected_compile_time, expected_inference_time_ms",
    ((2, 1, 16, 25.0, 39.0),),
)
def test_unet_perf_e2e(
    batch: int,
    groups: int,
    iterations: int,
    expected_compile_time: float,
    expected_inference_time_ms: float,
    device,
    use_program_cache,
    reset_seeds,
):
    profiler.clear()

    torch_input, ttnn_input = create_unet_input_tensors(device, batch, groups, pad_input=True)

    profiler.start(f"initialize_ref_model")
    model = unet_shallow_torch.UNet.from_random_weights(groups=1)
    profiler.end(f"initialize_ref_model")

    profiler.start(f"initialize_model")
    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)
    profiler.end(f"initialize_model")

    torch_output_tensor = model(torch_input)

    logger.info(f"Compiling model with warmup run")
    profiler.start(f"inference_and_compile_time")
    output_tensor = ttnn_model(ttnn_input).cpu()
    profiler.end(f"inference_and_compile_time")

    inference_and_compile_time = profiler.get("inference_and_compile_time")
    logger.info(f"Model compiled with warmup run in {(inference_and_compile_time):.2f} s")

    logger.info(f"Running inference for {iterations} iterations")
    for idx in range(iterations):
        profiler.start("inference_time")
        profiler.start(f"inference_time_{idx}")
        output_tensor = ttnn_model(ttnn_input).cpu()
        profiler.end(f"inference_time_{idx}")
        profiler.end("inference_time")

    mean_inference_time = profiler.get("inference_time")
    inference_time = profiler.get(f"inference_time_{iterations - 1}")
    compile_time = inference_and_compile_time - inference_time
    logger.info(f"Model compilation took {compile_time:.1f} s")
    logger.info(f"Inference time on last iterations was completed in {(inference_time * 1000.0):.2f} ms")
    logger.info(
        f"Mean inference time for {batch} (batch) images was {(mean_inference_time * 1000.0):.2f} ms ({batch / mean_inference_time:.2f} fps)"
    )

    expected_inference_time = expected_inference_time_ms * 1e-3
    prep_perf_report(
        model_name=f"unet_shallow",
        batch_size=batch,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
    )

    logger.info(f"Running sanity check against reference model output")
    B, C, H, W = torch_output_tensor.shape
    ttnn_tensor = ttnn.to_torch(output_tensor).reshape(B, H, W, -1)[:, :, :, :C].permute(0, 3, 1, 2)
    assert_with_pcc(torch_output_tensor, ttnn_tensor, 0.97)


@pytest.mark.skip("Crashes on N300/T3K - see issue #12685")
@skip_for_grayskull("UNet not currently supported on GS")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("enable_async_mode", (True,), indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize(
    "batch, groups, iterations, expected_compile_time, expected_inference_time_ms",
    ((2, 1, 16, 25.0, 61.0),),
)
def test_unet_data_parallel_perf_e2e(
    batch: int,
    groups: int,
    iterations: int,
    expected_compile_time: float,
    expected_inference_time_ms: float,
    mesh_device,
    use_program_cache,
    reset_seeds,
    enable_async_mode,
):
    if not is_n300_with_eth_dispatch_cores(mesh_device) and not is_t3k_with_eth_dispatch_cores(mesh_device):
        pytest.skip("Test is only valid for N300 or T3000")

    profiler.clear()

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    torch_input, ttnn_input = create_unet_input_tensors(mesh_device, batch, groups, pad_input=True)

    profiler.start(f"initialize_ref_model")
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)
    profiler.end(f"initialize_ref_model")

    profiler.start(f"initialize_model")
    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=mesh_device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device=mesh_device, mesh_mapper=weights_mesh_mapper)
    profiler.end(f"initialize_model")

    num_devices = len(mesh_device.get_device_ids())
    total_batch = num_devices * batch
    torch_input, ttnn_input = create_unet_input_tensors(
        mesh_device, total_batch, groups, pad_input=True, mesh_mapper=inputs_mesh_mapper
    )
    logger.info(f"Created reference input tensors: {list(torch_input.shape)}")
    logger.info(
        f"Created multi-device input tensors: shape={list(ttnn_input.shape)} on devices={mesh_device.get_device_ids()}"
    )

    torch_output_tensor = model(torch_input)

    logger.info(f"Compiling model with warmup run")
    profiler.start(f"inference_and_compile_time")
    output_tensor = ttnn.from_device(ttnn_model(ttnn_input), blocking=True)
    profiler.end(f"inference_and_compile_time")

    inference_and_compile_time = profiler.get("inference_and_compile_time")
    logger.info(f"Model compiled with warmup run in {(inference_and_compile_time):.2f} s")

    logger.info(f"Running inference for {iterations} iterations")
    for idx in range(iterations):
        profiler.start("inference_time")
        profiler.start(f"inference_time_{idx}")
        logger.info(f"running iter {idx}")
        output_tensor = ttnn.from_device(ttnn_model(ttnn_input), blocking=True)
        logger.info(f"done running iter {idx}")
        profiler.end(f"inference_time_{idx}")
        profiler.end("inference_time")
    synchronize_devices(mesh_device)

    mean_inference_time = profiler.get("inference_time")
    inference_time = profiler.get(f"inference_time_{iterations - 1}")
    compile_time = inference_and_compile_time - inference_time
    logger.info(f"Model compilation took {compile_time:.1f} s")
    logger.info(f"Inference time on last iterations was completed in {(inference_time * 1000.0):.2f} ms")
    logger.info(
        f"Mean inference time for {total_batch} (batch) images was {(mean_inference_time * 1000.0):.2f} ms ({total_batch / mean_inference_time:.2f} fps)"
    )

    expected_inference_time = expected_inference_time_ms * 1e-3
    prep_perf_report(
        model_name=f"unet_shallow-data_parallel",
        batch_size=total_batch,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=f"batch_{total_batch}-num_devices_{num_devices}",
    )

    logger.info(f"Running sanity check against reference model output")
    check_pcc_conv(torch_output_tensor, output_tensor, mesh_composer=output_mesh_composer, pcc=0.97)
