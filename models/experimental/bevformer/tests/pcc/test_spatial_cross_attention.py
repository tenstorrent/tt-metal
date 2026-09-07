# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from models.experimental.bevformer.tt.tt_spatial_cross_attention import TTSpatialCrossAttention
from models.experimental.bevformer.reference.spatial_cross_attention import SpatialCrossAttention


from models.experimental.bevformer.config.encoder_config import (
    get_preset_config,
)

from models.experimental.bevformer.tests.test_utils import (
    print_detailed_comparison,
    check_with_tolerances,
    check_with_pcc,
    print_sparsity_analysis,
)

from models.experimental.bevformer.tt.model_preprocessing import (
    create_spatial_cross_attention_parameters,
)

from loguru import logger

# Enable/disable logging output
ENABLE_LOGGING = True

# Default Test Configuration
PRINT_DETAILED_COMPARISON_FLAG = False

# Used for single query row testing
MAX_QUERY_ROW_REL_ERROR = 0.3


@pytest.mark.parametrize(
    "config_name, batch_size, bev_h, bev_w, expected_pcc, expected_abs_error, expected_rel_error, expected_high_error_ratio, blank_cams, valid_per_pair",
    [
        ("nuscenes_tiny", 1, 30, 30, 0.997, 0.06, 0.5, 0.5, (), None),  # NuScenes tiny model - 30x30 BEV grid
        ("nuscenes_base", 1, 50, 50, 0.999, 0.04, 1.3, 0.5, (), None),  # NuScenes base model - 50x50 BEV grid
        ("nuscenes_base", 1, 100, 100, 0.999, 0.04, 1.3, 0.5, (), None),  # NuScenes base model - 100x100 BEV grid
        ("nuscenes_base", 1, 200, 200, 0.998, 0.04, 1.3, 0.5, (), None),  # NuScenes base model - 200x200 BEV grid
        ("carla_base", 1, 100, 100, 0.998, 0.04, 1.3, 0.5, (), None),  # CARLA base model
        ("nuscenes_base", 2, 30, 30, 0.998, 0.04, 1.3, 0.5, (), None),  # Batch size 2
        ("nuscenes_base", 1, 50, 50, 0.999, 0.04, 1.3, 0.5, ((0, 0),), None),  # Blind camera: all-padding row
        ("nuscenes_base", 2, 30, 30, 0.998, 0.04, 1.3, 0.5, ((2, 1),), None),  # Same, in one batch item only
        ("nuscenes_base", 2, 30, 30, 0.998, 0.04, 1.3, 0.5, "all", None),  # max_len == 0: residual-only return
        ("nuscenes_base", 2, 30, 30, 0.998, 0.04, 1.3, 0.5, ((2, 1),), 32),  # max_len == rebatch_len: no padding
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [0])
def test_spatial_cross_attention_forward(
    device,
    config_name,
    batch_size,
    bev_h,
    bev_w,
    expected_pcc,
    expected_abs_error,
    expected_rel_error,
    expected_high_error_ratio,
    blank_cams,
    valid_per_pair,
    seed,
):
    """Test TTSpatialCrossAttention against PyTorch reference implementation using configurations."""
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
    num_cams = dataset_config.num_cams
    num_queries = bev_h * bev_w

    # Use spatial shapes from dataset config (limited to num_levels)
    spatial_shapes_list = dataset_config.spatial_shapes[:num_levels]
    spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.long)

    sca_total_key_length = [h * w for h, w in spatial_shapes.tolist()]

    # --------------------------------------------------------------------------- #
    # Generate Inputs                                                             #
    # --------------------------------------------------------------------------- #

    # Create input tensors
    bev_queries = torch.randn(batch_size, num_queries, embed_dims, dtype=torch.float32)

    # Camera features: [num_cams, H*W, batch_size, embed_dims] - use only first level
    camera_features = torch.randn(num_cams, sum(sca_total_key_length), batch_size, embed_dims, dtype=torch.float32)

    # Pre-projected reference points: [num_cams, batch_size, num_queries, D, 2]
    D = 4  # Number of points per pillar
    reference_points_cam = torch.rand(num_cams, batch_size, num_queries, D, 2, dtype=torch.float32)

    # Validity mask: [num_cams, batch_size, num_queries, D]
    if valid_per_pair is None:
        bev_mask = torch.ones(num_cams, batch_size, num_queries, D, dtype=torch.bool)
        # Randomly mask out some invalid points for realism
        # Randomly mask out 95% of points as invalid for realism
        total_points = bev_mask.numel()
        num_invalid = int(0.95 * total_points)
        invalid_indices = torch.randperm(total_points)[:num_invalid]
        bev_mask.view(-1)[invalid_indices] = False
    else:
        # Strided visible queries, alternating between two offsets so cameras share queries (which
        # exercises accumulation) and the parity flips per batch item so the items differ.
        assert valid_per_pair * 2 <= num_queries, f"valid_per_pair ({valid_per_pair}) too large for {num_queries}"
        bev_mask = torch.zeros(num_cams, batch_size, num_queries, D, dtype=torch.bool)
        stride = num_queries // valid_per_pair
        for cam_idx in range(num_cams):
            for batch_idx in range(batch_size):
                offset = ((cam_idx + batch_idx) % 2) * (stride // 2)
                visible = torch.arange(valid_per_pair) * stride + offset
                bev_mask[cam_idx, batch_idx, visible] = True

    if blank_cams == "all":
        bev_mask[:] = False
    else:
        for cam_idx, batch_idx in blank_cams:
            bev_mask[cam_idx, batch_idx] = False

    # Level start index
    indices = spatial_shapes.prod(1).cumsum(0)
    level_start_index = torch.cat([torch.tensor([0], dtype=torch.long), indices[:-1]], 0)

    # Convert tensors to ttnn format for ttnn model
    tt_bev_queries = ttnn.from_torch(bev_queries, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_camera_features = ttnn.from_torch(camera_features, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_reference_points_cam = ttnn.from_torch(
        reference_points_cam, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    tt_bev_mask = ttnn.from_torch(bev_mask, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_level_start_index = ttnn.from_torch(
        level_start_index, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    # --------------------------------------------------------------------------- #
    # Models Init                                                                 #
    # --------------------------------------------------------------------------- #

    # Create PyTorch reference model using extracted parameters
    ref_model = SpatialCrossAttention(
        embed_dims=embed_dims,
        num_cams=num_cams,
        batch_first=True,
        deformable_attention={
            "embed_dims": embed_dims,
            "num_levels": num_levels,
            "num_points": num_points,
            "num_heads": num_heads,
        },
    )
    ref_model.eval()

    # Create preprocessed parameters from PyTorch model
    tt_parameters = create_spatial_cross_attention_parameters(
        torch_model=ref_model,
        device=device,
        dtype=ttnn.float32,
    )

    # Create ttnn model with preprocessed parameters using extracted parameters
    tt_model = TTSpatialCrossAttention(
        device=device,
        params=tt_parameters,
        embed_dims=embed_dims,
        num_cams=num_cams,
        batch_first=True,
        deformable_attention={
            "embed_dims": embed_dims,
            "num_levels": num_levels,
            "num_points": num_points,
            "num_heads": num_heads,
        },
    )

    # --------------------------------------------------------------------------- #
    # Models Forward                                                              #
    # --------------------------------------------------------------------------- #

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

    # --------------------------------------------------------------------------- #
    # Output Comparison                                                           #
    # --------------------------------------------------------------------------- #

    if not bev_mask.any():
        # max_len == 0: both models return the residual, which on TT is the bfloat16 query.
        expected_residual = bev_queries.to(torch.bfloat16).to(torch.float32)
        assert torch.equal(tt_model_output, expected_residual), (
            "Residual-only branch did not return the query unchanged; max abs diff "
            f"{(tt_model_output - expected_residual).abs().max():.6f}."
        )

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

    # Comprehensive tolerance checking with expected metrics from test parameters
    check_with_tolerances(
        ref_model_output,
        tt_model_output,
        pcc_threshold=expected_pcc,
        abs_error_threshold=expected_abs_error,
        rel_error_threshold=expected_rel_error,
        max_error_ratio=expected_high_error_ratio,
        tensor_name="spatial_cross_attention_output",
    )

    passed, message = check_with_pcc(
        tt_model_output,
        ref_model_output,
        pcc=expected_pcc,
    )

    assert passed, f"PCC check failed: {message}"

    # Per-row error relative to the row's own magnitude, so one bound fits every config.
    row_error = (tt_model_output - ref_model_output).flatten(0, -2).norm(dim=-1)
    row_norm = ref_model_output.flatten(0, -2).norm(dim=-1)
    row_rel_error = row_error / row_norm.clamp(min=1e-6)
    worst_row = int(row_rel_error.argmax())
    assert row_rel_error[worst_row] <= MAX_QUERY_ROW_REL_ERROR, (
        f"Query row {worst_row % num_queries} of batch item {worst_row // num_queries} has relative "
        f"error {row_rel_error[worst_row]:.4f} > {MAX_QUERY_ROW_REL_ERROR} "
        f"(row norm {row_norm[worst_row]:.4f}); mean over rows is {row_rel_error.mean():.4f}."
    )

    if ENABLE_LOGGING:
        logger.info("✅ All SCA tolerance checks passed successfully!")
