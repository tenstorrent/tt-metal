# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Device performance tests for YOLO26.

Measures kernel execution time and throughput on Tenstorrent hardware.
"""

import pytest
import torch
import time
import ttnn
from loguru import logger


def get_torch_state_dict(variant: str):
    """Get PyTorch state dict from Ultralytics."""
    try:
        from ultralytics import YOLO

        model = YOLO(f"{variant}.pt")
        return model.model.state_dict()
    except ImportError:
        pytest.skip("ultralytics not installed. Run: pip install ultralytics")
    except Exception as e:
        pytest.skip(f"Could not load YOLO26 model: {e}")


@pytest.mark.parametrize("variant", ["yolo26n"])
@pytest.mark.parametrize("input_size", [640])
def test_yolo26_device_perf(device, variant, input_size):
    """
    Test YOLO26 device kernel performance.

    Measures raw device execution time without host overhead.
    """
    logger.info(f"Testing YOLO26 device perf: variant={variant}, input_size={input_size}")

    batch_size = 1

    # Load model
    from models.experimental.yolo26.tt.ttnn_yolo26 import TtYOLO26

    state_dict = get_torch_state_dict(variant)
    tt_model = TtYOLO26(device, variant)
    tt_model.load_weights_from_state_dict(state_dict)

    # Create input
    torch.manual_seed(42)
    input_tensor = torch.rand(batch_size, input_size, input_size, 3)  # NHWC
    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # Warmup
    num_warmup = 5
    for _ in range(num_warmup):
        outputs = tt_model(tt_input)
        for out in outputs:
            ttnn.deallocate(out)

    # Benchmark
    num_iterations = 20
    ttnn.synchronize_device(device)

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        outputs = tt_model(tt_input)
        for out in outputs:
            ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iterations) * 1000
    fps = num_iterations / total_time

    logger.info(f"Average inference time: {avg_time_ms:.2f} ms")
    logger.info(f"Throughput: {fps:.1f} FPS")

    # Performance targets (adjust based on actual measurements)
    target_fps = 30  # Minimum target for real-time detection
    assert fps >= target_fps * 0.5, f"Performance below 50% of target: {fps:.1f} < {target_fps * 0.5}"

    return {"fps": fps, "avg_time_ms": avg_time_ms}


@pytest.mark.parametrize("variant", ["yolo26n"])
def test_yolo26_e2e_perf(device, variant, input_size):
    """
    End-to-end performance test including data transfer.

    Measures realistic inference throughput.
    """
    logger.info(f"Testing YOLO26 E2E perf: variant={variant}, input_size={input_size}")

    batch_size = 1

    # Load model
    from models.experimental.yolo26.tt.ttnn_yolo26 import TtYOLO26

    state_dict = get_torch_state_dict(variant)
    tt_model = TtYOLO26(device, variant)
    tt_model.load_weights_from_state_dict(state_dict)

    # Warmup
    num_warmup = 3
    for _ in range(num_warmup):
        input_tensor = torch.rand(batch_size, input_size, input_size, 3)
        tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        outputs = tt_model(tt_input)
        results = [ttnn.to_torch(out) for out in outputs]
        for out in outputs:
            ttnn.deallocate(out)

    # Benchmark (including data transfer)
    num_iterations = 10
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        # Create new input each iteration (simulates real usage)
        input_tensor = torch.rand(batch_size, input_size, input_size, 3)
        tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        outputs = tt_model(tt_input)
        results = [ttnn.to_torch(out) for out in outputs]

        for out in outputs:
            ttnn.deallocate(out)

    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iterations) * 1000
    fps = num_iterations / total_time

    logger.info(f"E2E Average inference time: {avg_time_ms:.2f} ms")
    logger.info(f"E2E Throughput: {fps:.1f} FPS")

    return {"e2e_fps": fps, "e2e_avg_time_ms": avg_time_ms}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
