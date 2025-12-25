# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from models.experimental.bevformer.tt.tt_spatial_cross_attention import TTSpatialCrossAttention
from models.experimental.bevformer.reference.spatial_cross_attention import SpatialCrossAttention

from models.experimental.bevformer.config import DeformableAttentionConfig

from models.experimental.bevformer.tests.test_utils import (
    print_detailed_comparison,
    check_with_tolerances,
    check_with_pcc,  # Legacy compatibility
    save_comparison_report,
    print_sparsity_analysis,
)

from models.experimental.bevformer.tt.model_preprocessing import (
    create_spatial_cross_attention_parameters,
)

from loguru import logger


@pytest.mark.parametrize(
    "batch_size, num_query, embed_dims, num_cams",
    [
        (1, 900, 256, 6),  # nuScenes 30x30 BEV grid
        (1, 2500, 256, 6),  # 50x50 BEV grid
        (1, 10000, 256, 6),  # 100x100 BEV grid
        (1, 40000, 256, 6),  # 200x200 BEV grid
        (2, 900, 256, 6),  # Batch size 2
    ],
)
@pytest.mark.parametrize(
    "spatial_shapes",
    [
        [[200, 113], [100, 57], [50, 29], [25, 15]],  # nuScenes, input size 1600x900
        [[160, 90], [80, 45], [40, 23], [20, 12]],  # nuScenes, input size 1280x720
        [[120, 80], [60, 40], [30, 20], [15, 10]],  # Waymo, input size 960x640
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [0])
def test_spatial_cross_attention_forward(
    device,
    batch_size,
    num_query,
    embed_dims,
    num_cams,
    spatial_shapes,
    seed,
):
    torch.manual_seed(seed)
    print_detailed_comparison_flag = False

    spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long)

    sca_total_key_length = [h * w for h, w in spatial_shapes.tolist()]

    # Create input tensors
    bev_queries = torch.randn(batch_size, num_query, embed_dims, dtype=torch.float32)

    # Camera features: [num_cams, H*W, batch_size, embed_dims] - use only first level
    camera_features = torch.randn(num_cams, sum(sca_total_key_length), batch_size, embed_dims, dtype=torch.float32)

    # Pre-projected reference points: [num_cams, batch_size, num_query, D, 2]
    D = 4  # Number of points per pillar
    reference_points_cam = torch.rand(num_cams, batch_size, num_query, D, 2, dtype=torch.float32)

    # Validity mask: [num_cams, batch_size, num_query, D]
    bev_mask = torch.ones(num_cams, batch_size, num_query, D, dtype=torch.bool)
    # Randomly mask out some invalid points for realism
    # Randomly mask out 95% of points as invalid for realism
    total_points = bev_mask.numel()
    num_invalid = int(0.95 * total_points)
    invalid_indices = torch.randperm(total_points)[:num_invalid]
    bev_mask.view(-1)[invalid_indices] = False

    # Level start index for single level
    indices = spatial_shapes.prod(1).cumsum(0)
    level_start_index = torch.cat([torch.tensor([0], dtype=torch.long), indices[:-1]], 0)

    # Create configuration
    config = DeformableAttentionConfig(embed_dims=embed_dims, num_heads=4, num_levels=4, num_points=8, batch_first=True)

    # Create PyTorch reference model
    ref_model = SpatialCrossAttention(
        embed_dims=embed_dims,
        num_cams=num_cams,
        dropout=0.0,  # No dropout for testing
        batch_first=True,
        deformable_attention={"embed_dims": embed_dims, "num_levels": 4, "num_points": 8, "num_heads": 4},
    )
    ref_model.eval()

    # Create preprocessed parameters from PyTorch model
    tt_parameters = create_spatial_cross_attention_parameters(
        torch_model=ref_model,
        device=device,
        dtype=ttnn.float32,
    )

    # Create ttnn model with preprocessed parameters
    tt_model = TTSpatialCrossAttention(
        device=device,
        params=tt_parameters,
        embed_dims=embed_dims,
        num_cams=num_cams,
        dropout=0.0,
        batch_first=True,
        deformable_attention={"embed_dims": embed_dims, "num_levels": 4, "num_points": 8, "num_heads": 4},
    )

    # Forward pass with PyTorch reference model
    ref_model_output = ref_model(
        query=bev_queries,
        reference_points_cam=reference_points_cam,
        bev_mask=bev_mask,
        key=camera_features,
        value=camera_features,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )

    tt_bev_queries = ttnn.from_torch(bev_queries, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_camera_features = ttnn.from_torch(camera_features, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_reference_points_cam = ttnn.from_torch(
        reference_points_cam, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    tt_bev_mask = ttnn.from_torch(bev_mask, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_level_start_index = ttnn.from_torch(
        level_start_index, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    # Forward pass with ttnn model
    tt_model_output = tt_model(
        query=tt_bev_queries,
        reference_points_cam=tt_reference_points_cam,
        bev_mask=tt_bev_mask,
        key=tt_camera_features,
        value=tt_camera_features,
        spatial_shapes=spatial_shapes,
        level_start_index=tt_level_start_index,
    )

    tt_model_output = ttnn.to_torch(tt_model_output, dtype=torch.float32)

    # Comprehensive comparison using enhanced test utilities
    logger.info(f"Reference model output type: {type(ref_model_output)}, shape: {ref_model_output.shape}")
    logger.info(f"TT model output type: {type(tt_model_output)}")

    if print_detailed_comparison_flag:
        # Print detailed statistical comparison
        print_detailed_comparison(
            ref_model_output,
            tt_model_output,
            tensor_name="spatial_cross_attention_output",
            show_histograms=False,  # Set to True for even more detailed analysis
        )

        # Individual sparsity analysis for each tensor
        print_sparsity_analysis(
            ref_model_output,
            tensor_name="spatial_cross_attention_output_torch",
            # block_size=16,  # Analyze sparsity in 16x16 blocks
        )

        print_sparsity_analysis(
            tt_model_output,
            tensor_name="spatial_cross_attention_output_ttnn",
            # block_size=16,  # Analyze sparsity in 16x16 blocks
        )

    # Comprehensive tolerance checking with multiple criteria
    passed, results = check_with_tolerances(
        ref_model_output,
        tt_model_output,
        pcc_threshold=0.998,  # 0.998 only in case of 200x200 BEV grid
        abs_error_threshold=4e-2,
        rel_error_threshold=1.3,
        max_error_ratio=0.5,
        tensor_name="spatial_cross_attention_output",
    )

    # Assert that the comprehensive check passes
    assert passed, f"Comprehensive tolerance check failed. Results: {results['individual_checks']}"

    logger.info("✅ All SCA tolerance checks passed successfully!")
