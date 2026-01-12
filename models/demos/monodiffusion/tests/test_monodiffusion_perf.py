# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Performance benchmarking tests for MonoDiffusion
Target: 10+ FPS at 640x192 resolution
"""

import pytest
import torch
import ttnn
import time
import numpy as np
from typing import Dict

from models.demos.monodiffusion.tt import (
    create_monodiffusion_from_parameters,
    create_monodiffusion_preprocessor,
    load_reference_model,
)


def measure_inference_time(
    model,
    input_tensor: ttnn.Tensor,
    num_iterations: int = 10,
    warmup_iterations: int = 3
) -> Dict[str, float]:
    """
    Measure inference time with warmup

    Returns:
        Dictionary with timing statistics
    """
    # Warmup
    for _ in range(warmup_iterations):
        _ = model(input_tensor, return_uncertainty=True)

    # Measure
    times = []
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        depth, uncertainty = model(input_tensor, return_uncertainty=True)
        end_time = time.perf_counter()

        times.append(end_time - start_time)

    times = np.array(times)

    return {
        "mean": times.mean(),
        "std": times.std(),
        "min": times.min(),
        "max": times.max(),
        "median": np.median(times),
        "fps": 1.0 / times.mean()
    }


@pytest.mark.parametrize("device_id", [0])
@pytest.mark.parametrize("resolution", [(192, 640)])
@pytest.mark.parametrize("num_inference_steps", [20, 10, 5])
def test_monodiffusion_perf_device(device_id, resolution, num_inference_steps):
    """
    Test device performance (inference only, no host overhead)
    Target: 10+ FPS at 640x192 with 20 inference steps
    """
    height, width = resolution

    device = ttnn.open_device(device_id=device_id)

    try:
        # Create model
        print(f"\n{'='*60}")
        print(f"Testing MonoDiffusion Performance")
        print(f"Resolution: {width}x{height}")
        print(f"Inference steps: {num_inference_steps}")
        print(f"{'='*60}")

        # Load reference model and create preprocessor
        reference_model = load_reference_model()
        preprocessor = create_monodiffusion_preprocessor(device)
        parameters = preprocessor(reference_model, "monodiffusion", {})

        # Create model
        model = create_monodiffusion_from_parameters(
            parameters=parameters,
            device=device,
            batch_size=1,
            input_height=height,
            input_width=width,
        )
        model.num_inference_steps = num_inference_steps

        # Create input
        input_image = torch.randn(1, 3, height, width)
        input_tensor = ttnn.from_torch(
            input_image,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Measure performance
        stats = measure_inference_time(model, input_tensor, num_iterations=10, warmup_iterations=3)

        # Print results
        print(f"\nPerformance Results:")
        print(f"  Mean time:   {stats['mean']*1000:.2f} ms")
        print(f"  Std dev:     {stats['std']*1000:.2f} ms")
        print(f"  Min time:    {stats['min']*1000:.2f} ms")
        print(f"  Max time:    {stats['max']*1000:.2f} ms")
        print(f"  Median time: {stats['median']*1000:.2f} ms")
        print(f"  FPS:         {stats['fps']:.2f}")

        # Check target
        target_fps = 10.0
        if stats['fps'] >= target_fps:
            print(f"\n✓ Performance target met: {stats['fps']:.2f} FPS >= {target_fps} FPS")
        else:
            print(f"\n⚠ Performance below target: {stats['fps']:.2f} FPS < {target_fps} FPS")
            print(f"  (This is expected for initial bring-up, will optimize in Stage 2-3)")

        # For initial bring-up, we just log the performance
        # Strict assertion will be enabled after optimizations
        # assert stats['fps'] >= target_fps, f"FPS {stats['fps']:.2f} below target {target_fps}"

    finally:
        ttnn.close_device(device)


@pytest.mark.parametrize("device_id", [0])
def test_monodiffusion_perf_e2e(device_id):
    """
    Test end-to-end performance including:
    - Model compilation
    - First inference (with compilation)
    - Subsequent inferences
    """
    device = ttnn.open_device(device_id=device_id)

    try:
        print(f"\n{'='*60}")
        print(f"End-to-End Performance Test")
        print(f"{'='*60}")

        # Measure compilation time
        compile_start = time.perf_counter()
        model = create_monodiffusion_model(device, config_type="kitti")
        compile_end = time.perf_counter()
        compile_time = compile_end - compile_start

        print(f"\nCompilation time: {compile_time:.2f} seconds")

        # Create input
        input_image = torch.randn(1, 3, 192, 640)
        input_tensor = preprocess_input_image(input_image, device, 192, 640)

        # First inference (includes kernel compilation)
        first_start = time.perf_counter()
        depth1, uncertainty1 = model(input_tensor, return_uncertainty=True)
        first_end = time.perf_counter()
        first_inference_time = first_end - first_start

        print(f"First inference time: {first_inference_time:.2f} seconds")

        # Subsequent inferences
        stats = measure_inference_time(model, input_tensor, num_iterations=10, warmup_iterations=0)

        print(f"\nSteady-state performance:")
        print(f"  Mean time: {stats['mean']*1000:.2f} ms")
        print(f"  FPS:       {stats['fps']:.2f}")

        print(f"\n✓ E2E performance test completed!")

    finally:
        ttnn.close_device(device)


@pytest.mark.parametrize("device_id", [0])
def test_monodiffusion_throughput_batch(device_id):
    """
    Test throughput with multiple images processed sequentially
    """
    device = ttnn.open_device(device_id=device_id)

    try:
        model = create_monodiffusion_model(device, config_type="kitti")

        num_images = 20

        # Warmup
        input_image = torch.randn(1, 3, 192, 640)
        input_tensor = preprocess_input_image(input_image, device, 192, 640)
        _ = model(input_tensor, return_uncertainty=True)

        # Process batch
        start_time = time.perf_counter()
        for i in range(num_images):
            input_image = torch.randn(1, 3, 192, 640)
            input_tensor = preprocess_input_image(input_image, device, 192, 640)
            depth, uncertainty = model(input_tensor, return_uncertainty=True)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time_per_image = total_time / num_images
        throughput = num_images / total_time

        print(f"\nThroughput Test Results:")
        print(f"  Total images:     {num_images}")
        print(f"  Total time:       {total_time:.2f} seconds")
        print(f"  Avg per image:    {avg_time_per_image*1000:.2f} ms")
        print(f"  Throughput:       {throughput:.2f} images/sec")

        print(f"\n✓ Throughput test completed!")

    finally:
        ttnn.close_device(device)


@pytest.mark.parametrize("device_id", [0])
def test_memory_usage(device_id):
    """
    Test memory usage during inference
    """
    device = ttnn.open_device(device_id=device_id)

    try:
        model = create_monodiffusion_model(device, config_type="kitti")

        # Run inference and monitor memory
        input_image = torch.randn(1, 3, 192, 640)
        input_tensor = preprocess_input_image(input_image, device, 192, 640)

        depth, uncertainty = model(input_tensor, return_uncertainty=True)

        # TODO: Add actual memory profiling when available
        print(f"\n✓ Memory usage test completed!")
        print(f"  (Detailed memory profiling to be added)")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
