# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import ttnn
import pytest
from typing import Dict, Any, List

from models.experimental.bevformer.tt.tt_encoder import TTBEVFormerEncoder
from models.experimental.bevformer.tt.tt_spatial_cross_attention import REBATCH_HEADROOM
from models.experimental.bevformer.reference.encoder import BEVFormerEncoder
from models.experimental.bevformer.tests.common import PRESETS, img_metas_for_dataset, rig_sequence_metas


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
DEFAULT_TEST_CONFIG = PRESETS["nuscenes_base"]  # NuScenes + Base model

# Flag to control detailed comparison output
PRINT_DETAILED_COMPARISON_FLAG = False

# Module-scoped device: opens once per file instead of once per test case.
pytestmark = pytest.mark.use_module_device({"l1_small_size": 32 * 1024})


def create_sample_img_metas(batch_size: int, dataset_config=DEFAULT_TEST_CONFIG) -> List[Dict[str, Any]]:
    """Create img_metas from the dataset's deterministic surround-view rig.

    ``lidar2img`` decides ``bev_mask`` and therefore the spatial-cross-attention
    rebatch length, so random matrices would make every spatial-path tensor shape
    depend on the RNG draw order.

    Args:
        batch_size: Number of batches
        dataset_config: Dataset config supplying camera count, resolution, and rig
    """
    return img_metas_for_dataset(dataset_config, batch_size)


def random_inputs(device, config, bev_size, batch_size):
    """Random BEV queries, positions and camera features: as the reference takes them, and on device."""
    num_queries = bev_size[0] * bev_size[1]
    spatial_total_length = sum(h * w for h, w in config.spatial_shapes[: config.num_levels])
    bev_query = torch.randn(batch_size, num_queries, config.embed_dims, dtype=torch.float32)
    bev_pos = torch.randn(batch_size, num_queries, config.embed_dims, dtype=torch.float32)
    camera_features = torch.randn(
        config.num_cams, spatial_total_length, batch_size, config.embed_dims, dtype=torch.float32
    )

    def upload(tensor):
        return ttnn.from_torch(tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # The encoder takes the camera features batch-first, [batch_size, num_cams, H*W, embed_dims].
    tt_inputs = (upload(bev_query), upload(bev_pos), upload(camera_features.permute(2, 0, 1, 3)))
    return (bev_query, bev_pos, camera_features), tt_inputs


def build_encoders(device, config, bev_size, num_layers):
    """The reference encoder with random weights, and the TT encoder built from them."""
    spatial_shapes = torch.tensor(config.spatial_shapes[: config.num_levels], dtype=torch.long)
    encoder_kwargs = dict(
        num_layers=num_layers,
        embed_dims=config.embed_dims,
        num_heads=config.num_heads,
        num_levels=config.num_levels,
        num_points=config.num_points,
        num_cams=config.num_cams,
        pc_range=list(config.pc_range),
    )
    ref_model = BEVFormerEncoder(**encoder_kwargs, tsa_num_points=config.tsa_num_points)
    ref_model.eval()
    tt_parameters = create_bevformer_encoder_parameters(ref_model, device, dtype=ttnn.bfloat16)
    tt_model = TTBEVFormerEncoder(
        device=device,
        params=tt_parameters,
        bev_h=bev_size[0],
        bev_w=bev_size[1],
        spatial_shapes=spatial_shapes,
        batch_first=True,
        return_intermediate=False,
        feedforward_channels=tt_parameters.config.feedforward_channels,
        temporal_num_points=config.tsa_num_points,
        **encoder_kwargs,
    )
    return ref_model, tt_model, spatial_shapes


@torch.no_grad()
def reference_forward(ref_model, inputs, bev_size, spatial_shapes, img_metas):
    """The reference encoder on a first frame, without a previous BEV."""
    bev_query, bev_pos, camera_features = inputs
    # The upstream reference takes the queries sequence-first and returns them batch-first.
    return ref_model(
        bev_query=bev_query.permute(1, 0, 2),
        value=camera_features,
        bev_h=bev_size[0],
        bev_w=bev_size[1],
        bev_pos=bev_pos.permute(1, 0, 2),
        spatial_shapes=spatial_shapes,
        prev_bev=None,
        img_metas=img_metas,
    )


@pytest.mark.parametrize(
    "config_name, bev_size, num_layers, batch_size, expected_pcc, expected_abs_error, expected_rel_error, expected_high_error_ratio",
    [
        ("nuscenes_base", (100, 100), 6, 1, 0.997, 0.05, 0.8, 0.5),  # NuScenes base model
        ("nuscenes_base", (100, 100), 1, 1, 0.997, 0.05, 0.8, 0.5),  # NuScenes base, single layer
        ("nuscenes_tiny", (100, 100), 3, 2, 0.996, 0.05, 0.8, 0.5),  # NuScenes tiny model, bs>1 grid widening
        # bs>1 near the base grid: the layer's intermediates outgrow L1 and must go to DRAM. Below
        # torch.quantile's 16M-element limit, which the PCC check hits at 200x200.
        ("nuscenes_base", (180, 180), 1, 2, 0.997, 0.05, 0.8, 0.5),
        ("carla_tiny", (100, 100), 3, 1, 0.995, 0.05, 0.8, 0.5),  # CARLA tiny model
        # Off: (arch, camera rig) corners the three above already cover.
        # ("carla_base", (100, 100), 6, 1, 0.997, 0.05, 0.8, 0.5),  # CARLA base model
        # ("nuscenes_base_fast", (100, 100), 6, 1, 0.996, 0.05, 0.8, 0.5),  # NuScenes base fast model
    ],
)
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

    config = PRESETS[config_name]
    inputs, (tt_bev_query, tt_bev_pos, tt_camera_features) = random_inputs(device, config, bev_size, batch_size)
    # Camera metadata for point sampling; the rig derives resolution and camera
    # count from the dataset config itself
    img_metas = create_sample_img_metas(batch_size, config)
    ref_model, tt_model, spatial_shapes = build_encoders(device, config, bev_size, num_layers)

    ref_output = reference_forward(ref_model, inputs, bev_size, spatial_shapes, img_metas)
    tt_output = tt_model(
        bev_query=tt_bev_query,
        key=tt_camera_features,
        value=tt_camera_features,
        bev_pos=tt_bev_pos,
        prev_bev=None,  # No temporal attention
        img_metas=img_metas,
    )
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


# Per frame, the rig's (yaw in degrees, image zoom), see ``rig_sequence_metas``. The first frame sizes
# the rebatch plan and the second stays within it; then one frame falls below it, one outgrows it,
# one falls back within it and the last outgrows it again.
RIG_SEQUENCE = ((0, 1.0), (5, 1.0), (10, 1.1), (15, 0.9), (20, 0.95), (25, 0.8))


@pytest.mark.parametrize(
    "config_name, expected_max_len, expected_pcc",
    [
        # The busiest camera's visible BEV cells per frame of RIG_SEQUENCE, from the reference's
        # projection at 100x100.
        ("nuscenes_base", (2472, 2474, 2295, 2662, 2559, 2797), 0.997),  # nuScenes rig
        ("carla_tiny", (2146, 2161, 2053, 2300, 2233, 2448), 0.995),  # ring rig
    ],
)
def test_bevformer_encoder_over_a_rig_sequence(device, config_name, expected_max_len, expected_pcc):
    """One encoder over frames whose camera geometry changes. Per frame: the cameras' visible counts
    against the reference projection's, the rebatch plan's capacity and reallocations against the
    counts so far, and the output against the reference's. One layer suffices, as every layer
    shares the frame's plan."""
    torch.manual_seed(0)
    config = PRESETS[config_name]
    bev_size = (100, 100)
    inputs, (tt_bev_query, tt_bev_pos, tt_camera_features) = random_inputs(device, config, bev_size, 1)
    ref_model, tt_model, spatial_shapes = build_encoders(device, config, bev_size, num_layers=1)
    planner = tt_model.rebatch_planner

    frames = rig_sequence_metas(config, RIG_SEQUENCE)
    assert len(frames) == len(expected_max_len)
    capacity, reallocations, rebatch_len = 0, 0, None
    for i, img_metas in enumerate(frames):
        _, ref_mask = ref_model.point_sampling(*bev_size, 1, img_metas)
        ref_counts = ref_mask.any(-1)[:, 0].sum(-1)
        assert int(ref_counts.max()) == expected_max_len[i], f"frame {i}: the sequence no longer has its geometry"

        cameras = tt_model.project_cameras(img_metas)
        plan = planner(cameras.reference_points_cam, cameras.visibility)
        tt_counts = ttnn.to_torch(cameras.visibility.visible_count).flatten().round().to(torch.int64)
        logger.info(f"frame {i}: visible {tt_counts.tolist()}, reference {ref_counts.tolist()}")
        assert torch.equal(tt_counts, ref_counts), f"frame {i}: visible counts differ from the reference's"

        max_len = int(tt_counts.max())
        grew = max_len > capacity
        if grew:
            reallocations += capacity > 0
            capacity = math.ceil(max_len * REBATCH_HEADROOM)
        else:
            assert plan.rebatch_len == rebatch_len, f"frame {i} changed the plan's shapes within its capacity"
        rebatch_len = plan.rebatch_len
        logger.info(f"frame {i}: max_len {max_len}, capacity {planner.capacity}, rebatch_len {rebatch_len}")
        assert planner.capacity == capacity >= max_len, f"frame {i}"
        assert planner.reallocations == reallocations, f"frame {i}"

        ref_output = reference_forward(ref_model, inputs, bev_size, spatial_shapes, img_metas)
        tt_output = tt_model(
            bev_query=tt_bev_query,
            key=tt_camera_features,
            value=tt_camera_features,
            bev_pos=tt_bev_pos,
            cameras=cameras._replace(rebatch_plan=plan),
        )
        passed, message = check_with_pcc(ref_output, ttnn.to_torch(tt_output, dtype=torch.float32), expected_pcc)
        logger.info(f"frame {i}: {message}")
        assert passed, f"frame {i}: {message}"

    assert reallocations == 2, "the sequence should outgrow the plan twice"
