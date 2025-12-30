# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import math

from models.experimental.bevformer.tt.tt_temporal_self_attention import TTTemporalSelfAttention
from models.experimental.bevformer.reference.temporal_self_attention import TemporalSelfAttention

from models.experimental.bevformer.config import DeformableAttentionConfig

from models.experimental.bevformer.tests.test_utils import (
    print_detailed_comparison,
    check_with_tolerances,
    check_with_pcc,  # Legacy compatibility
    save_comparison_report,
)

from models.experimental.bevformer.tt.model_preprocessing import (
    create_temporal_self_attention_parameters,
)

from loguru import logger


@pytest.mark.parametrize(
    "batch_size, num_query, embed_dims, num_heads, num_bev_queue",
    [
        (1, 900, 256, 8, 2),  # nuScenes 30x30 BEV grid
        (1, 2500, 256, 8, 2),  # 50x50 BEV grid
        (1, 10000, 256, 8, 2),  # 100x100 BEV grid
        (2, 900, 256, 8, 2),  # Batch size 2
        (1, 900, 256, 4, 2),  # Different number of heads
        (1, 40000, 256, 4, 2),  # 200x200 BEV grid
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [0])
def test_temporal_self_attention_forward(
    device,
    batch_size,
    num_query,
    embed_dims,
    num_heads,
    num_bev_queue,
    seed,
):
    torch.manual_seed(seed)
    print_detailed_comparison_flag = False

    bev_spatial_shapes = torch.tensor([[int(math.sqrt(num_query))] * 2], dtype=torch.long)
    bev_h, bev_w = bev_spatial_shapes[0]
    num_levels = len(bev_spatial_shapes)

    # Create input tensors for temporal self attention
    current_bev = torch.randn(batch_size, num_query, embed_dims, dtype=torch.float32)
    prev_bev = torch.randn(batch_size, num_query, embed_dims, dtype=torch.float32)

    # BEV reference points in 2D space [batch_size, num_query, num_levels, 2]
    reference_points_2d = torch.rand(batch_size, num_query, num_levels, 2, dtype=torch.float32)

    # Level start index
    indices = bev_spatial_shapes.prod(1).cumsum(0)
    level_start_index = torch.cat([torch.tensor([0], dtype=torch.long), indices[:-1]], 0)

    # Create configuration
    config = DeformableAttentionConfig(
        embed_dims=embed_dims, num_heads=num_heads, num_levels=num_levels, num_points=4, batch_first=True
    )

    # Create PyTorch reference model
    ref_model = TemporalSelfAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=4,
        num_bev_queue=num_bev_queue,
        batch_first=True,
    )
    ref_model.eval()

    # Create preprocessed parameters from PyTorch model
    tt_parameters = create_temporal_self_attention_parameters(
        torch_model=ref_model,
        device=device,
        dtype=ttnn.float32,
    )

    # Create ttnn model with preprocessed parameters
    tt_model = TTTemporalSelfAttention(
        device=device,
        params=tt_parameters,
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=4,
        num_bev_queue=num_bev_queue,
        batch_first=True,
    )

    # Forward pass with PyTorch reference model
    ref_model_output = ref_model(
        query=current_bev,
        reference_points=reference_points_2d,
        spatial_shapes=bev_spatial_shapes,
        level_start_index=level_start_index,
        bev_h=bev_h,
        bev_w=bev_w,
    )

    # Forward pass with ttnn model
    tt_model_output = tt_model(
        query=current_bev,
        reference_points=reference_points_2d,
        spatial_shapes=bev_spatial_shapes,
        level_start_index=level_start_index,
        bev_h=bev_h,
        bev_w=bev_w,
    )

    # Comprehensive comparison using enhanced test utilities
    logger.info(f"Reference model output type: {type(ref_model_output)}, shape: {ref_model_output.shape}")
    logger.info(f"TT model output type: {type(tt_model_output)}")

    if print_detailed_comparison_flag:
        # Print detailed statistical comparison
        print_detailed_comparison(
            ref_model_output,
            tt_model_output,
            tensor_name="temporal_self_attention_output",
            show_histograms=False,  # Set to True for even more detailed analysis
        )

    # Comprehensive tolerance checking with multiple criteria
    passed, results = check_with_tolerances(
        ref_model_output,
        tt_model_output,
        pcc_threshold=0.999,
        abs_error_threshold=6e-2,
        rel_error_threshold=6e-1,
        max_error_ratio=4e-1,
        tensor_name="temporal_self_attention_output",
    )

    # Assert that the comprehensive check passes
    assert passed, f"Comprehensive tolerance check failed. Results: {results['individual_checks']}"

    logger.info("✅ All TSA tolerance checks passed successfully!")


@pytest.mark.parametrize(
    "batch_size, num_query, embed_dims, num_heads",
    [
        (1, 100, 256, 4),  # Small test case
        (2, 256, 256, 4),  # Different batch size and num_query
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
def test_temporal_self_attention_without_prev_bev(
    device,
    batch_size,
    num_query,
    embed_dims,
    num_heads,
):
    """Test TSA when no previous BEV is provided (should duplicate current BEV)."""
    torch.manual_seed(42)

    # Simple BEV spatial configuration - match the number of queries
    # For 100 queries: 10x10, for 256 queries: 16x16
    if num_query == 100:
        bev_spatial_shapes = torch.tensor([[10, 10]], dtype=torch.long)
        bev_h, bev_w = 10, 10
    elif num_query == 256:
        bev_spatial_shapes = torch.tensor([[16, 16]], dtype=torch.long)
        bev_h, bev_w = 16, 16
    else:
        # Default: try to find square root for other cases
        import math

        side_length = int(math.sqrt(num_query))
        if side_length * side_length == num_query:
            bev_spatial_shapes = torch.tensor([[side_length, side_length]], dtype=torch.long)
            bev_h, bev_w = side_length, side_length
        else:
            raise ValueError(f"num_query {num_query} is not a perfect square, cannot create square BEV grid")

    # Create test inputs
    current_bev = torch.randn(batch_size, num_query, embed_dims, dtype=torch.float32)
    reference_points_2d = torch.rand(batch_size, num_query, 1, 2, dtype=torch.float32)
    level_start_index = torch.tensor([0], dtype=torch.long)

    # Create reference model
    ref_model = TemporalSelfAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=1,
        num_points=4,
        num_bev_queue=2,
        batch_first=True,
    )
    ref_model.eval()

    # Forward pass without previous BEV (using default value=None)
    output = ref_model(
        query=current_bev,
        reference_points=reference_points_2d,
        spatial_shapes=bev_spatial_shapes,
        level_start_index=level_start_index,
        # No value parameter - will use default behavior (stack query num_bev_queue times)
        bev_h=bev_h,
        bev_w=bev_w,
    )

    # Verify output shape matches input query shape
    assert output.shape == current_bev.shape, f"Output shape {output.shape} != input shape {current_bev.shape}"

    # Verify output is not NaN or Inf
    assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

    logger.info("✅ TSA without previous BEV test passed!")


@pytest.mark.parametrize(
    "batch_size, num_query, embed_dims, num_heads",
    [
        (1, 64, 128, 4),  # Smaller dimensions for validation
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
def test_temporal_self_attention_shape_consistency(
    device,
    batch_size,
    num_query,
    embed_dims,
    num_heads,
):
    """Test that TSA maintains shape consistency and residual connections work."""
    torch.manual_seed(123)

    # Simple configuration
    bev_spatial_shapes = torch.tensor([[8, 8]], dtype=torch.long)

    # Create test inputs
    current_bev = torch.randn(batch_size, num_query, embed_dims, dtype=torch.float32)
    prev_bev = torch.randn(batch_size, num_query, embed_dims, dtype=torch.float32)
    reference_points_2d = torch.rand(batch_size, num_query, 1, 2, dtype=torch.float32)
    level_start_index = torch.tensor([0], dtype=torch.long)

    # Create reference model
    ref_model = TemporalSelfAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=1,
        num_points=2,  # Small number of points
        num_bev_queue=2,
        batch_first=True,
    )
    ref_model.eval()

    # Test with explicit identity
    identity = torch.randn_like(current_bev)

    output = ref_model(
        query=current_bev,
        identity=identity,
        reference_points=reference_points_2d,
        spatial_shapes=bev_spatial_shapes,
        level_start_index=level_start_index,
        bev_h=8,
        bev_w=8,
    )

    # Verify output shape matches input query shape
    assert output.shape == current_bev.shape, f"Output shape {output.shape} != input shape {current_bev.shape}"

    # Verify output is not NaN or Inf
    assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

    # Check that residual connection is working (output should be different from identity alone)
    residual_diff = torch.abs(output - identity).mean()
    assert residual_diff > 1e-6, f"Residual difference too small: {residual_diff}, attention may not be working"

    logger.info("✅ TSA shape consistency and residual connection test passed!")
