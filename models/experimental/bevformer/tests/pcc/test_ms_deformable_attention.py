# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from models.experimental.bevformer.tt.tt_ms_deformable_attention import TTMSDeformableAttention
from models.experimental.bevformer.reference.ms_deformable_attention import MSDeformableAttention

from models.experimental.bevformer.config import DeformableAttentionConfig

from models.experimental.bevformer.config.encoder_config import (
    get_preset_config,
)

from models.experimental.bevformer.tests.test_utils import (
    check_with_tolerances,
    check_with_pcc,
)

from models.experimental.bevformer.tt.model_preprocessing import (
    create_ms_deformable_attention_parameters,
)

from loguru import logger

# Enable/disable logging output
ENABLE_LOGGING = True

# Default Test Configuration                                                  #
PRINT_DETAILED_COMPARISON_FLAG = False


@pytest.mark.parametrize(
    "config_name, batch_size, num_queries, expected_pcc, expected_abs_error, expected_rel_error, expected_high_error_ratio",
    [
        ("nuscenes_tiny", 1, 900, 0.999, 0.02, 0.38, 0.36),  # NuScenes tiny model
        ("nuscenes_base", 1, 10000, 0.999, 0.02, 0.21, 0.23),  # NuScenes base model with larger queries
        ("carla_base", 1, 12000, 0.999, 0.02, 0.15, 0.18),  # CARLA base model
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [0])
def test_ms_deformable_attention_forward(
    device,
    config_name,
    batch_size,
    num_queries,
    expected_pcc,
    expected_abs_error,
    expected_rel_error,
    expected_high_error_ratio,
    seed,
):
    """Test TTMSDeformableAttention against PyTorch reference implementation using configurations."""
    torch.manual_seed(seed)

    # Get configuration from preset
    preset_config = get_preset_config(config_name)
    if preset_config is None:
        pytest.fail(f"Configuration '{config_name}' not found")

    dataset_config = preset_config.dataset_config
    model_config = preset_config.model_config

    # Extract parameters from configs
    embed_dims = model_config.embed_dims
    num_heads = model_config.num_heads
    num_levels = model_config.num_levels
    num_points = model_config.num_points

    # Use spatial shapes from dataset config (limited to num_levels)
    spatial_shapes_list = dataset_config.spatial_shapes[:num_levels]
    spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.long)
    total_key_length = [h * w for h, w in spatial_shapes.tolist()]

    # --------------------------------------------------------------------------- #
    # Generate Inputs                                                             #
    # --------------------------------------------------------------------------- #

    # Create input tensors
    query = torch.randn(batch_size, num_queries, embed_dims, dtype=torch.float32)
    value = torch.randn(batch_size, sum(total_key_length), embed_dims, dtype=torch.float32)

    reference_points = torch.rand(batch_size, num_queries, num_levels, 2, dtype=torch.float32)

    # Convert tensors to ttnn format for ttnn model
    tt_query = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_value = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_reference_points = ttnn.from_torch(reference_points, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Create DeformableAttentionConfig from extracted parameters
    config = DeformableAttentionConfig(
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
    )

    # --------------------------------------------------------------------------- #
    # Models Init                                                                 #
    # --------------------------------------------------------------------------- #

    # Create PyTorch reference model
    ref_model = MSDeformableAttention(config)
    ref_model.eval()

    # Create preprocessed parameters from PyTorch model
    tt_parameters = create_ms_deformable_attention_parameters(
        torch_model=ref_model,
        device=device,
        config=config,
        dtype=ttnn.float32,
    )

    # Create ttnn model with preprocessed parameters
    tt_model = TTMSDeformableAttention(
        config=config,
        device=device,
        params=tt_parameters,
    )

    # --------------------------------------------------------------------------- #
    # Models Forward                                                              #
    # --------------------------------------------------------------------------- #

    ref_model_output = ref_model(
        query,
        value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
    )

    tt_model_output = tt_model(
        query=tt_query,
        value=tt_value,
        reference_points=tt_reference_points,
        spatial_shapes=spatial_shapes,
    )
    tt_model_output = ttnn.to_torch(tt_model_output)

    # --------------------------------------------------------------------------- #
    # Output Comparison                                                           #
    # --------------------------------------------------------------------------- #

    # Comprehensive comparison using enhanced test utilities
    if ENABLE_LOGGING:
        logger.info(f"Reference model output type: {type(ref_model_output)}, shape: {ref_model_output.shape}")
    if ENABLE_LOGGING:
        logger.info(f"TT model output type: {type(tt_model_output)}")

    if PRINT_DETAILED_COMPARISON_FLAG:
        # Print detailed statistical comparison
        print_detailed_comparison(
            ref_model_output,
            tt_model_output,
            tensor_name="ms_deformable_attention_output",
            show_histograms=False,  # Set to True for even more detailed analysis
        )

    # Comprehensive tolerance checking with multiple criteria
    check_with_tolerances(
        ref_model_output,
        tt_model_output,
        pcc_threshold=expected_pcc,
        abs_error_threshold=expected_abs_error,
        rel_error_threshold=expected_rel_error,
        max_error_ratio=expected_high_error_ratio,
        tensor_name="ms_deformable_attention_output",
    )

    passed, results = check_with_pcc(
        ref_model_output,
        tt_model_output,
        pcc=expected_pcc,
    )

    assert passed, f"PCC check failed: {results}"

    if ENABLE_LOGGING:
        logger.info("✅ All tolerance checks passed successfully!")
