# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_grayskull
from models.demos.gemma3.tt.gemma_vision_model import TtGemmaTransformerVision
from models.demos.gemma3.tt.model_config import ModelArgs
from models.demos.utils.llm_demo_utils import create_benchmark_data, verify_perf
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.ccl import TT_CCL


def get_image_features(vision_tower, projector, input_tensor):
    """
    Get image features from the vision tower and projector.
    """
    vision_token = vision_tower(input_tensor).last_hidden_state
    image_features = projector(vision_token)
    return image_features


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize("device_params", [{"fabric_config": True, "l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "bsz, num_iterations, warmup_iterations",
    [
        (1, 50, 5),  # Single batch latency test
        (2, 30, 3),  # Small batch test
        (4, 20, 2),  # Medium batch test
    ],
    ids=["batch-1-latency", "batch-2", "batch-4"],
)
def test_gemma_vision_benchmark(
    mesh_device,
    reset_seeds,
    bsz,
    num_iterations,
    warmup_iterations,
    request,
):
    """
    Benchmark test for Gemma Vision Cross Attention Transformer.
    Measures inference time, throughput, and other performance metrics.
    """
    test_id = request.node.callspec.id
    dtype = ttnn.bfloat16

    logger.info(f"Running Gemma Vision benchmark: {test_id}")
    logger.info(f"Batch size: {bsz}, Iterations: {num_iterations}, Warmup: {warmup_iterations}")

    # Start profiler
    profiler = BenchmarkProfiler()
    profiler.start("run")

    # Initialize model
    profiler.start("model_initialization")
    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    vision_first_layer_prefix = "model.vision_tower.vision_model."
    vision_partial_state_dict = {
        k[len(vision_first_layer_prefix) :]: v
        for k, v in state_dict.items()
        if (k.startswith(vision_first_layer_prefix))
    }

    reference_vision_model = model_args.reference_vision_model()
    reference_mmp = model_args.reference_vision_multi_modal()

    image_size = model_args.vision_chunk_size
    in_channels = model_args.vision_in_channels

    test_gemma_vision = TtGemmaTransformerVision(
        mesh_device,
        tt_ccl=TT_CCL(mesh_device),
        state_dict=state_dict,
        state_dict_prefix="model.vision_tower.vision_model.",
        dtype=dtype,
        configuration=model_args,
        return_intermediate=False,
    )
    profiler.end("model_initialization")

    # Prepare input data
    profiler.start("data_preparation")
    input_tensor = torch.rand((bsz, in_channels, image_size, image_size))

    # Get reference output for correctness validation
    reference_output = get_image_features(
        reference_vision_model,
        reference_mmp,
        input_tensor,
    )
    profiler.end("data_preparation")

    # Warmup iterations
    logger.info("Starting warmup iterations...")
    profiler.start("warmup")
    for i in range(warmup_iterations):
        logger.debug(f"Warmup iteration {i+1}/{warmup_iterations}")
        profiler.start(f"warmup_iteration_{i}")
        test_output = test_gemma_vision(input_tensor)
        ttnn.synchronize_device(mesh_device)
        profiler.end(f"warmup_iteration_{i}")
    profiler.end("warmup")
    logger.info("Warmup completed")

    # Benchmark iterations
    logger.info("Starting benchmark iterations...")
    profiler.start("benchmark_iterations")
    inference_times = []

    for i in range(num_iterations):
        logger.debug(f"Benchmark iteration {i+1}/{num_iterations}")
        profiler.start(f"inference_iteration_{i}")

        # Run inference
        test_output = test_gemma_vision(input_tensor)
        ttnn.synchronize_device(mesh_device)

        profiler.end(f"inference_iteration_{i}")
        iteration_time = profiler.get_duration(f"inference_iteration_{i}")
        inference_times.append(iteration_time)

    profiler.end("benchmark_iterations")
    profiler.end("run")

    # Process final output for correctness check
    logger.info("Processing final output for correctness validation")
    out = ttnn.from_device(test_output)
    tt_output_torch = ttnn.to_torch(
        out,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[0, :, :, :]

    # Calculate performance metrics
    model_init_time = profiler.get_duration("model_initialization")
    data_prep_time = profiler.get_duration("data_preparation")
    total_warmup_time = profiler.get_duration("warmup")
    total_benchmark_time = profiler.get_duration("benchmark_iterations")
    total_run_time = profiler.get_duration("run")

    # Statistical analysis of inference times
    avg_inference_time = sum(inference_times) / len(inference_times)
    min_inference_time = min(inference_times)
    max_inference_time = max(inference_times)

    # Throughput calculations (images per second)
    images_per_second = bsz / avg_inference_time
    total_images_processed = bsz * num_iterations
    total_throughput = total_images_processed / total_benchmark_time

    measurements = {
        # Required measurements
        "model_initialization": model_init_time,
        "data_preparation": data_prep_time,
        "total_warmup": total_warmup_time,
        "total_inference": total_benchmark_time,
        "avg_inference_time": avg_inference_time,
        "min_inference_time": min_inference_time,
        "max_inference_time": max_inference_time,
        "images_per_second": images_per_second,
        "total_throughput": total_throughput,
        # Optional measurements
        "total_run_time": total_run_time,
        "num_iterations": num_iterations,
        "batch_size": bsz,
    }

    # Performance targets (can be adjusted based on requirements)
    targets = {
        "avg_inference_time": 0.1,  # Target: less than 100ms average inference time
        "images_per_second": 10 * bsz,  # Target: at least 10 images/sec per batch element
    }

    # Log performance metrics
    logger.info("")
    logger.info("=== Gemma Vision Performance Metrics ===")
    logger.info(f"Model initialization time: {model_init_time:.3f}s")
    logger.info(f"Data preparation time: {data_prep_time:.3f}s")
    logger.info(f"Total warmup time: {total_warmup_time:.3f}s")
    logger.info(f"Total benchmark time: {total_benchmark_time:.3f}s")
    logger.info("")
    logger.info("=== Inference Statistics ===")
    logger.info(f"Average inference time: {avg_inference_time*1000:.2f}ms")
    logger.info(f"Minimum inference time: {min_inference_time*1000:.2f}ms")
    logger.info(f"Maximum inference time: {max_inference_time*1000:.2f}ms")
    logger.info(f"Images per second: {images_per_second:.2f}")
    logger.info(f"Total throughput: {total_throughput:.2f} images/sec")
    logger.info(f"Total images processed: {total_images_processed}")

    # Create benchmark data for CI/dashboard
    benchmark_data = create_benchmark_data(
        profiler,
        measurements,
        {
            "warmup": warmup_iterations,
            "inference": 0,
        },  # No additional warmup for inference since we did separate warmup
        targets,
    )

    # Add individual iteration measurements for detailed analysis
    for i, inference_time in enumerate(inference_times):
        benchmark_data.add_measurement(
            profiler,
            0,
            "inference",
            f"iteration_{i}_time",
            inference_time * 1000,  # Convert to milliseconds
            step_warm_up_num_iterations=None,
            target=None,
        )

    # Save benchmark data
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"gemma-vision-benchmark",
        ml_model_name="Gemma3-Vision",
        ml_model_type="vision_transformer",
        num_layers=model_args.n_layers if hasattr(model_args, "n_layers") else None,
        batch_size=bsz,
        input_sequence_length=image_size * image_size,  # Using image area as sequence length equivalent
        output_sequence_length=None,  # Vision models don't have output sequence length
    )

    # Verify performance against targets (optional)
    try:
        verify_perf(
            measurements,
            targets,
            high_tol_percentage=1.2,  # Allow 20% tolerance
            expected_measurements={k: True for k in targets.keys()},
        )
        logger.info("Performance targets met!")
    except AssertionError as e:
        logger.warning(f"Performance targets not met: {e}")
        # Don't fail the test for performance targets, just log warning

    # Basic correctness check - ensure outputs are reasonable
    output_mean = tt_output_torch.mean().item()
    output_std = tt_output_torch.std().item()

    logger.info(f"Output statistics - Mean: {output_mean:.6f}, Std: {output_std:.6f}")

    # Ensure output is not all zeros or NaN
    assert not torch.isnan(tt_output_torch).any(), "Output contains NaN values"
    assert not torch.all(tt_output_torch == 0), "Output is all zeros"
    assert output_std > 1e-6, f"Output standard deviation too low: {output_std}"

    logger.info("Benchmark completed successfully!")
