# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.bevformer.config.encoder_config import get_preset_config
from models.experimental.bevformer.tests.layer_common import build_layer_fixture
from models.experimental.bevformer.tests.test_utils import check_with_pcc, check_with_tolerances


@pytest.mark.parametrize(
    "config_name, bev_size, batch_size, expected_pcc, expected_abs_error, expected_rel_error, expected_high_error_ratio",
    [
        ("nuscenes_base", (100, 100), 1, 0.997, 0.05, 0.8, 0.5),
        ("nuscenes_tiny", (100, 100), 1, 0.996, 0.05, 0.8, 0.5),
        ("carla_base", (100, 100), 1, 0.997, 0.05, 0.8, 0.5),
        ("carla_tiny", (100, 100), 1, 0.995, 0.05, 0.8, 0.5),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [0])
def test_bevformer_layer_forward(
    device,
    config_name,
    bev_size,
    batch_size,
    expected_pcc,
    expected_abs_error,
    expected_rel_error,
    expected_high_error_ratio,
    seed,
):
    """Test a single TTBEVFormerLayer against the PyTorch reference layer."""
    torch.manual_seed(seed)

    config = get_preset_config(config_name)
    if config is None:
        pytest.fail(f"Configuration '{config_name}' not found")

    fixture = build_layer_fixture(device, config, bev_size, batch_size)

    with torch.no_grad():
        ref_output = fixture.ref_model(**fixture.ref_inputs)

    tt_output = fixture.tt_model(**fixture.tt_inputs)
    tt_output_torch = ttnn.to_torch(tt_output, dtype=torch.float32)

    logger.info(f"Reference layer output shape: {ref_output.shape}")
    logger.info(f"TT layer output shape: {tt_output_torch.shape}")

    check_with_tolerances(
        ref_output,
        tt_output_torch,
        pcc_threshold=expected_pcc,
        abs_error_threshold=expected_abs_error,
        rel_error_threshold=expected_rel_error,
        max_error_ratio=expected_high_error_ratio,
        tensor_name="bevformer_layer_output",
    )

    passed, message = check_with_pcc(ref_output, tt_output_torch, expected_pcc)
    assert passed, f"PCC check failed: {message}"

    logger.info("✅ All BEVFormer layer tolerance checks passed successfully!")
