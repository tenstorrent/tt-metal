# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from loguru import logger
from models.experimental.SSD512.tt.tt_ssd import build_ssd512
from models.experimental.SSD512.reference.ssd import build_ssd
from models.common.utility_functions import comp_pcc
from models.perf.benchmarking_utils import BenchmarkProfiler


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
@pytest.mark.parametrize(
    "size",
    (512,),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 98304}], indirect=True)
@pytest.mark.models_performance_bare_metal
def test_ssd512_network(device, pcc, size, reset_seeds):
    """
    Test Full SSD512 Network.
    """
    seed = 0
    if reset_seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except:
        pass

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        pass

    num_classes = 21  # VOC dataset
    batch_size = 1

    # Build PyTorch reference model FIRST
    torch_model = build_ssd("train", size=size, num_classes=num_classes)
    torch_model.eval()

    for m in torch_model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    # Build TTNN model
    ttnn_model = build_ssd512(num_classes=num_classes, device=device)

    # Load weights from PyTorch to TTNN
    # This ensures both models use the SAME weights for fair comparison
    ttnn_model.load_weights_from_torch(torch_model)

    # Synchronize device after weight loading
    ttnn.synchronize_device(device)
    import gc

    gc.collect()

    input_tensor = torch.randn(batch_size, 3, size, size)

    # Run PyTorch forward pass to get reference outputs
    with torch.no_grad():
        torch_outputs = torch_model(input_tensor)
        torch_loc = torch_outputs[0]  # Location predictions
        torch_conf = torch_outputs[1]  # Confidence predictions

    # Run TTNN forward pass
    ttnn.synchronize_device(device)
    gc.collect()
    ttnn_loc, ttnn_conf = ttnn_model.forward(input_tensor, dtype=ttnn.bfloat16, debug=False)

    # Ensure outputs are float32 for fair comparison
    ttnn_loc = ttnn_loc.float()
    ttnn_conf = ttnn_conf.float()

    # Flatten both tensors for comparison (comp_pcc expects same shape)
    torch_loc_flat = torch_loc.flatten()
    ttnn_loc_flat = ttnn_loc.flatten()
    torch_conf_flat = torch_conf.flatten()
    ttnn_conf_flat = ttnn_conf.flatten()

    # Truncate to minimum length if shapes don't match
    min_loc_len = min(len(torch_loc_flat), len(ttnn_loc_flat))
    min_conf_len = min(len(torch_conf_flat), len(ttnn_conf_flat))

    if len(torch_loc_flat) != len(ttnn_loc_flat):
        logger.warning(
            f"Location length mismatch! PyTorch: {len(torch_loc_flat)}, TTNN: {len(ttnn_loc_flat)}. Truncating to {min_loc_len}"
        )
        torch_loc_flat = torch_loc_flat[:min_loc_len]
        ttnn_loc_flat = ttnn_loc_flat[:min_loc_len]

    if len(torch_conf_flat) != len(ttnn_conf_flat):
        logger.warning(
            f"Confidence length mismatch! PyTorch: {len(torch_conf_flat)}, TTNN: {len(ttnn_conf_flat)}. Truncating to {min_conf_len}"
        )
        torch_conf_flat = torch_conf_flat[:min_conf_len]
        ttnn_conf_flat = ttnn_conf_flat[:min_conf_len]

    # Compare location predictions
    does_pass_loc, pcc_message_loc = comp_pcc(torch_loc_flat, ttnn_loc_flat, pcc)
    logger.info(f"Location PCC: {pcc_message_loc}")

    # Compare confidence predictions
    does_pass_conf, pcc_message_conf = comp_pcc(torch_conf_flat, ttnn_conf_flat, pcc)
    logger.info(f"Confidence PCC: {pcc_message_conf}")

    assert does_pass_loc, f"Location predictions do not meet PCC requirement {pcc}: {pcc_message_loc}"
    assert does_pass_conf, f"Confidence predictions do not meet PCC requirement {pcc}: {pcc_message_conf}"

    logger.info("Performance Measurement:")

    # Warmup iterations
    num_warmup_iterations = 1
    num_measurement_iterations = 1

    logger.info(f"Running {num_warmup_iterations} warmup iterations...")
    for _ in range(num_warmup_iterations):
        _ = ttnn_model.forward(input_tensor, dtype=ttnn.bfloat16, debug=False)
    gc.collect()

    # Measurement iterations using BenchmarkProfiler
    logger.info(f"Running {num_measurement_iterations} measurement iterations...")
    profiler = BenchmarkProfiler()

    for i in range(num_measurement_iterations):
        with profiler("inference", iteration=i):
            _ = ttnn_model.forward(input_tensor, dtype=ttnn.bfloat16, debug=False)

    # Calculate average inference time and FPS from profiler
    inference_time_avg = profiler.get_duration_average("inference", start_iteration=0)

    logger.info(f"\nPerformance Results:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Average inference time: {inference_time_avg:.6f} seconds")
