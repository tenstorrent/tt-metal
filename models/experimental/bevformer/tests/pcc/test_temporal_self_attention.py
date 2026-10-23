# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from models.experimental.bevformer.tt.tt_temporal_self_attention import TTTemporalSelfAttention
from models.experimental.bevformer.reference.temporal_self_attention import TemporalSelfAttention


from models.experimental.bevformer.tests.common import PRESETS

from models.experimental.bevformer.tests.test_utils import (
    print_detailed_comparison,
    check_with_tolerances,
    check_with_pcc,
)

from models.experimental.bevformer.tt.model_preprocessing import (
    create_temporal_self_attention_parameters,
)

from loguru import logger

# Enable/disable logging output
ENABLE_LOGGING = True

# Default Test Configuration
PRINT_DETAILED_COMPARISON_FLAG = False

# Module-scoped device: opens once per file instead of once per test case.
pytestmark = pytest.mark.use_module_device({"l1_small_size": 10 * 1024})


@pytest.mark.parametrize(
    "config_name, batch_size, bev_h, bev_w, num_bev_queue, expected_pcc, expected_abs_error, expected_rel_error, expected_high_error_ratio",
    [
        ("nuscenes_tiny", 1, 30, 30, 2, 0.999, 0.02, 0.15, 0.2),  # NuScenes tiny model - 30x30 BEV grid
        ("nuscenes_base", 1, 50, 50, 2, 0.999, 0.02, 0.11, 0.3),  # NuScenes base model - 50x50 BEV grid
        ("nuscenes_base", 1, 100, 100, 2, 0.999, 0.03, 0.17, 0.4),  # NuScenes base model - 100x100 BEV grid
        ("nuscenes_base", 2, 30, 30, 2, 0.999, 0.02, 0.06, 0.2),  # Batch size 2
        ("nuscenes_base", 1, 200, 200, 2, 0.999, 0.06, 0.58, 0.4),  # Large BEV grid
        # Off: TSA reads only embed_dims/num_heads/num_points, identical here to
        # nuscenes_base, and takes its grid from bev_h/bev_w -- same workload.
        # ("carla_base", 1, 100, 100, 2, 0.999, 0.03, 0.17, 0.4),  # CARLA base model
    ],
)
@pytest.mark.parametrize("seed", [0])
# The TT module also takes sequence-first features; the reference always runs batch-first.
@pytest.mark.parametrize("batch_first", [True, False], ids=["batch_first", "seq_first"])
def test_temporal_self_attention_forward(
    device,
    config_name,
    batch_size,
    bev_h,
    bev_w,
    num_bev_queue,
    expected_pcc,
    expected_abs_error,
    expected_rel_error,
    expected_high_error_ratio,
    seed,
    batch_first,
):
    """Test TTTemporalSelfAttention against PyTorch reference implementation using configurations."""
    torch.manual_seed(seed)

    # Get configuration from preset
    config = PRESETS[config_name]

    # Extract parameters from configs
    embed_dims = config.embed_dims
    num_heads = config.num_heads
    num_points = config.tsa_num_points
    num_queries = bev_h * bev_w

    # BEV spatial shapes - single level for temporal self attention
    bev_spatial_shapes = torch.tensor([[bev_h, bev_w]], dtype=torch.long)
    num_levels = len(bev_spatial_shapes)

    # --------------------------------------------------------------------------- #
    # Generate Inputs                                                             #
    # --------------------------------------------------------------------------- #

    # Create input tensors for temporal self attention
    current_bev = torch.randn(batch_size, num_queries, embed_dims, dtype=torch.float32)

    # BEV reference points in 2D space [batch_size, num_queries, num_levels, 2]
    reference_points_2d = torch.rand(batch_size, num_queries, num_levels, 2, dtype=torch.float32)

    # Level start index
    indices = bev_spatial_shapes.prod(1).cumsum(0)
    level_start_index = torch.cat([torch.tensor([0], dtype=torch.long), indices[:-1]], 0)

    # Convert tensors to ttnn format for ttnn model
    tt_current_bev = ttnn.from_torch(
        current_bev if batch_first else current_bev.permute(1, 0, 2),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_reference_points_2d = ttnn.from_torch(
        reference_points_2d, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
    )

    # --------------------------------------------------------------------------- #
    # Models Init                                                                 #
    # --------------------------------------------------------------------------- #

    # Create PyTorch reference model using extracted parameters
    ref_model = TemporalSelfAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        num_bev_queue=num_bev_queue,
        batch_first=True,
    )
    ref_model.eval()

    # Create preprocessed parameters from PyTorch model
    tt_parameters = create_temporal_self_attention_parameters(ref_model, device, dtype=ttnn.bfloat16)

    # Create ttnn model with preprocessed parameters using extracted parameters
    tt_model = TTTemporalSelfAttention(
        device=device,
        params=tt_parameters,
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        num_bev_queue=num_bev_queue,
        batch_first=batch_first,
        spatial_shapes=bev_spatial_shapes,
    )

    # --------------------------------------------------------------------------- #
    # Models Forward                                                              #
    # --------------------------------------------------------------------------- #

    # Forward pass with PyTorch reference model
    # The upstream reference takes one set of points per stacked BEV map; without a previous BEV
    # both maps are the query, sampled at the same points.
    ref_model_output = ref_model(
        query=current_bev,
        reference_points=torch.stack([reference_points_2d] * num_bev_queue, 1).flatten(0, 1),
        spatial_shapes=bev_spatial_shapes,
    )

    # Forward pass with ttnn model
    tt_model_output = tt_model(
        query=tt_current_bev,
        reference_points=tt_reference_points_2d,
    )
    if not batch_first:
        tt_model_output = ttnn.permute(tt_model_output, (1, 0, 2))

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
            tensor_name="temporal_self_attention_output",
            show_histograms=False,  # Set to True for even more detailed analysis
        )

    # Comprehensive tolerance checking with expected metrics from test parameters
    check_with_tolerances(
        ref_model_output,
        tt_model_output,
        pcc_threshold=expected_pcc,
        abs_error_threshold=expected_abs_error,
        rel_error_threshold=expected_rel_error,
        max_error_ratio=expected_high_error_ratio,
        tensor_name="temporal_self_attention_output",
    )

    passed, message = check_with_pcc(
        ref_model_output,
        tt_model_output,
        pcc=expected_pcc,
    )
    assert passed, f"PCC check failed: {message}"

    if ENABLE_LOGGING:
        logger.info("✅ All TSA tolerance checks passed successfully!")
