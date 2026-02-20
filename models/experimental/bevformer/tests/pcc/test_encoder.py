# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from typing import Dict, Any, List

from models.experimental.bevformer.tt.tt_encoder import TTBEVFormerEncoder
from models.experimental.bevformer.reference.encoder import BEVFormerEncoder
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
    create_bevformer_encoder_parameters,
)

from loguru import logger

# Enable/disable logging output
ENABLE_LOGGING = True

# --------------------------------------------------------------------------- #
# Default Test Configuration                                                  #
# --------------------------------------------------------------------------- #
torch.manual_seed(0)

# Get default configuration for testing (can be overridden in individual tests)
DEFAULT_TEST_CONFIG = get_preset_config("nuscenes_base")  # NuScenes + Base model
DEFAULT_DATASET_CONFIG = DEFAULT_TEST_CONFIG.dataset_config
DEFAULT_MODEL_CONFIG = DEFAULT_TEST_CONFIG.model_config

# Flag to control detailed comparison output
PRINT_DETAILED_COMPARISON_FLAG = False


def create_sample_img_metas(
    batch_size: int, num_cams: int = DEFAULT_DATASET_CONFIG.num_cams, image_shape: tuple = (900, 1600)
) -> List[Dict[str, Any]]:
    """Create img_metas with random lidar2img matrices (matching reference implementation).

    Args:
        batch_size: Number of batches
        num_cams: Number of cameras
        image_shape: Tuple of (height, width) for camera images
    """
    img_metas = []

    for batch_idx in range(batch_size):
        # Generate random lidar2img matrices for each camera
        lidar2img_matrices = [torch.randn(4, 4).tolist() for _ in range(num_cams)]

        height, width = image_shape
        meta = {
            "img_shape": [(height, width, 3)] * num_cams,
            "lidar2img": lidar2img_matrices,
        }
        img_metas.append(meta)

    return img_metas


@pytest.mark.parametrize(
    "config_name, bev_size, num_layers, batch_size, expected_pcc, expected_abs_error, expected_rel_error, expected_high_error_ratio",
    [
        ("nuscenes_base", (100, 100), 6, 1, 0.997, 0.05, 0.8, 0.5),  # NuScenes base model
        ("nuscenes_tiny", (100, 100), 3, 1, 0.996, 0.05, 0.8, 0.5),  # NuScenes tiny model
        ("carla_base", (100, 100), 6, 1, 0.997, 0.05, 0.8, 0.5),  # CARLA base model
        ("carla_tiny", (100, 100), 3, 1, 0.995, 0.05, 0.8, 0.5),  # CARLA tiny model
        ("nuscenes_base_fast", (100, 100), 6, 1, 0.996, 0.05, 0.8, 0.5),  # CARLA base fast model
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [0])
def test_bevformer_encoder_forward(
    device,
    config_name,
    bev_size,
    num_layers,
    batch_size,
    expected_pcc,
    expected_abs_error,
    expected_rel_error,
    expected_high_error_ratio,
    seed,
):
    """Test TTBEVFormerEncoder against PyTorch reference implementation using configurations."""
    torch.manual_seed(seed)

    # Get configuration
    config = get_preset_config(config_name)
    if config is None:
        pytest.fail(f"Configuration '{config_name}' not found")

    dataset_config = config.dataset_config
    model_config = config.model_config

    # Extract parameters from configs
    bev_h, bev_w = bev_size
    num_queries = bev_h * bev_w
    embed_dims = model_config.embed_dims
    num_cams = dataset_config.num_cams
    num_levels = model_config.num_levels

    # Use spatial shapes from dataset config (limited to num_levels)
    image_shape = dataset_config.input_size  # Use actual input size from config
    spatial_shapes_list = dataset_config.spatial_shapes[:num_levels]  # Take required number of levels
    spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.long)

    # --------------------------------------------------------------------------- #
    # Generate Inputs                                                             #
    # --------------------------------------------------------------------------- #

    # Create input tensors
    bev_query = torch.randn(batch_size, num_queries, embed_dims, dtype=torch.float32)
    bev_pos = torch.randn(batch_size, num_queries, embed_dims, dtype=torch.float32)

    # Camera features: [num_cams, H*W, batch_size, embed_dims]
    # Calculate total key length from spatial shapes
    spatial_total_length = sum(h * w for h, w in spatial_shapes.tolist())
    camera_features = torch.randn(num_cams, spatial_total_length, batch_size, embed_dims, dtype=torch.float32)

    # Level start index using centralized utility function
    level_start_index = config.get_level_start_index()
    # Limit to actual num_levels being used in this test
    if len(level_start_index) > num_levels:
        level_start_index = level_start_index[:num_levels]

    # Camera metadata for point sampling (convert width, height to height, width for img_metas)
    img_shape = (image_shape[1], image_shape[0])  # (height, width) for img_metas
    img_metas = create_sample_img_metas(batch_size, num_cams, img_shape)

    # Convert tensors to ttnn format for ttnn model
    tt_bev_query = ttnn.from_torch(bev_query, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_bev_pos = ttnn.from_torch(bev_pos, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_camera_features = ttnn.from_torch(camera_features, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_level_start_index = ttnn.from_torch(
        level_start_index, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    # --------------------------------------------------------------------------- #
    # Models Init                                                                 #
    # --------------------------------------------------------------------------- #

    # Create PyTorch reference model using config
    encoder_kwargs = config.get_encoder_kwargs()

    encoder_kwargs.update(
        {
            "num_layers": num_layers,  # Test-specific layer count
            "batch_first": True,
            "return_intermediate": False,
        }
    )

    ref_model = BEVFormerEncoder(**encoder_kwargs)
    ref_model.eval()

    # Create preprocessed parameters from PyTorch model
    tt_parameters = create_bevformer_encoder_parameters(
        torch_model=ref_model,
        device=device,
        dtype=ttnn.bfloat16,
    )

    # Create ttnn model with preprocessed parameters using config
    tt_model = TTBEVFormerEncoder(
        device=device,
        params=tt_parameters,
        **encoder_kwargs,
    )

    # --------------------------------------------------------------------------- #
    # Models Forward                                                              #
    # --------------------------------------------------------------------------- #

    # Forward pass with PyTorch reference model
    with torch.no_grad():
        ref_output = ref_model(
            bev_query=bev_query,
            key=camera_features,
            value=camera_features,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=None,  # No temporal attention
            img_metas=img_metas,
        )

    # Forward pass with ttnn model
    tt_output = tt_model(
        bev_query=tt_bev_query,
        key=tt_camera_features,
        value=tt_camera_features,
        bev_pos=tt_bev_pos,
        bev_h=bev_h,
        bev_w=bev_w,
        spatial_shapes=spatial_shapes,
        level_start_index=tt_level_start_index,
        prev_bev=None,  # No temporal attention
        img_metas=img_metas,
    )

    # Convert output back to torch
    tt_output_torch = ttnn.to_torch(tt_output, dtype=torch.float32)

    # --------------------------------------------------------------------------- #
    # Output Comparison                                                           #
    # --------------------------------------------------------------------------- #

    # Comprehensive comparison using enhanced test utilities
    if ENABLE_LOGGING:
        logger.info(f"Reference model output shape: {ref_output.shape}")
    if ENABLE_LOGGING:
        logger.info(f"TT model output shape: {tt_output_torch.shape}")

    if PRINT_DETAILED_COMPARISON_FLAG:
        # Print detailed statistical comparison
        print_detailed_comparison(
            ref_output,
            tt_output_torch,
            tensor_name="bevformer_encoder_output",
            show_histograms=False,
        )

        # Individual sparsity analysis for each tensor
        print_sparsity_analysis(
            ref_output,
            tensor_name="bevformer_encoder_output_torch",
        )

        print_sparsity_analysis(
            tt_output_torch,
            tensor_name="bevformer_encoder_output_ttnn",
        )

    # Comprehensive tolerance checking with multiple criteria
    check_with_tolerances(
        ref_output,
        tt_output_torch,
        pcc_threshold=expected_pcc,  # Lower threshold for complex encoder
        abs_error_threshold=expected_abs_error,
        rel_error_threshold=expected_rel_error,
        max_error_ratio=expected_high_error_ratio,
        tensor_name="bevformer_encoder_output",
    )

    passed, message = check_with_pcc(
        ref_output,
        tt_output_torch,
        expected_pcc,
    )

    assert passed, f"PCC check failed: {message}"

    if ENABLE_LOGGING:
        logger.info("✅ All BEVFormer encoder tolerance checks passed successfully!")
