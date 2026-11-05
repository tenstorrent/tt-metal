# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from models.experimental.bevformer.tt import tt_common
from models.experimental.bevformer.tt.tt_ms_deformable_attention import grouped_softmax
from models.experimental.bevformer.tt.tt_temporal_self_attention import TTTemporalSelfAttention
from models.experimental.bevformer.reference.temporal_self_attention import TemporalSelfAttention


from models.experimental.bevformer.reference.point_sampling_3d_2d import bev_reference_points
from models.experimental.bevformer.tests.common import (
    PRESETS,
    TSA_LOGIT_STD,
    TSA_OFFSET_STD_PX,
    _spread,
    sample_ego_shifts,
)

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
    query_pos = torch.randn(batch_size, num_queries, embed_dims, dtype=torch.float32)

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
    tt_query_pos = ttnn.from_torch(
        query_pos if batch_first else query_pos.permute(1, 0, 2),
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
    # Upstream's init zeroes the offset and attention Linears' weights, so neither would read the
    # query; spread them so the head-major (head, map, level, point) channels the op reads as its
    # (head, level, point) and the per-map softmax are exercised.
    generator = torch.Generator().manual_seed(seed)
    _spread(ref_model.sampling_offsets, TSA_OFFSET_STD_PX, generator)
    _spread(ref_model.attention_weights, TSA_LOGIT_STD, generator)

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
        query_pos=query_pos,
        reference_points=torch.stack([reference_points_2d] * num_bev_queue, 1).flatten(0, 1),
        spatial_shapes=bev_spatial_shapes,
    )

    # Forward pass with ttnn model
    tt_model_output = tt_model(
        query=tt_current_bev,
        query_pos=tt_query_pos,
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


# shift_scale 1.0 is a few cells of ego motion, 8.0 highway speed (~34 px at 200x200): the
# previous map's float32 reference points must carry the shift without the rounding bfloat16
# offsets would add.
# "dram" forces every group of intermediates around the fused op out of L1, as when they do not
# fit, which also gives the offsets a page per head where DRAM's alignment exceeds a head's. The
# fused op's own value placement is decided apart, by ``value_memory``.
@pytest.mark.parametrize(
    "batch_size, bev_h, bev_w, shift_scale, glue_intermediates",
    [
        (1, 50, 50, 1.0, "fit"),
        (2, 30, 30, 1.0, "fit"),
        (1, 200, 200, 1.0, "fit"),
        (1, 200, 200, 8.0, "fit"),
        (2, 30, 30, 1.0, "dram"),
    ],
    ids=["bs1_50x50", "bs2_30x30", "bs1_200x200", "bs1_200x200_highway", "bs2_30x30_dram"],
)
def test_temporal_self_attention_previous_bev(
    device, batch_size, bev_h, bev_w, shift_scale, glue_intermediates, monkeypatch
):
    """A previous BEV distinct from the queries, each sample shifted by its own ego motion."""
    if glue_intermediates == "dram":
        monkeypatch.setattr(tt_common, "L1_INTERMEDIATE_SHARE", 0.0)
    torch.manual_seed(0)
    config = PRESETS["nuscenes_base"]
    embed_dims, num_heads, num_points = config.embed_dims, config.num_heads, config.tsa_num_points
    num_queries = bev_h * bev_w
    spatial_shapes = torch.tensor([[bev_h, bev_w]], dtype=torch.long)

    ref_model = TemporalSelfAttention(
        embed_dims=embed_dims, num_heads=num_heads, num_levels=1, num_points=num_points, batch_first=True
    ).eval()
    generator = torch.Generator().manual_seed(0)
    _spread(ref_model.sampling_offsets, TSA_OFFSET_STD_PX, generator)
    _spread(ref_model.attention_weights, TSA_LOGIT_STD, generator)

    query = torch.randn(batch_size, num_queries, embed_dims)
    query_pos = torch.randn(batch_size, num_queries, embed_dims)
    prev_bev = torch.randn(batch_size, num_queries, embed_dims)
    # As upstream's encoder stacks them: per sample, the previous BEV, then the queries.
    value = torch.stack([prev_bev, query], 1).flatten(0, 1)
    ref_2d = bev_reference_points(bev_h, bev_w, batch_size)
    shift = sample_ego_shifts(batch_size) * shift_scale
    hybrid_ref_2d = torch.stack([ref_2d + shift[:, None, None, :], ref_2d], 1).flatten(0, 1)

    with torch.no_grad():
        expected = ref_model(
            query,
            value=value,
            query_pos=query_pos,
            reference_points=hybrid_ref_2d,
            spatial_shapes=spatial_shapes,
        )

    def to_device(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(tensor, device=device, dtype=dtype, layout=layout)

    tt_model = TTTemporalSelfAttention(
        device=device,
        params=create_temporal_self_attention_parameters(ref_model, device, dtype=ttnn.bfloat16),
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=1,
        num_points=num_points,
        spatial_shapes=spatial_shapes,
    )
    tt_output = tt_model(
        query=to_device(query),
        value=to_device(value),
        query_pos=to_device(query_pos),
        reference_points=to_device(ref_2d, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
        shift=to_device(shift.view(batch_size, 1, 1, 2), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    )

    passed, message = check_with_pcc(expected, ttnn.to_torch(tt_output).float(), pcc=0.999)
    logger.info(f"TSA previous BEV: {message}")
    assert passed, f"PCC check failed: {message}"


# TSA's groups of 4, the decoder's (one tile wide), SCA's of 8 at one camera level, and groups a
# tile wide.
@pytest.mark.parametrize(
    "num_groups, group_size",
    [(16, 4), (8, 4), (8, 8), (8, 32)],
    ids=["groups_of_4", "decoder_groups_of_4", "groups_of_8", "groups_of_32"],
)
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG], ids=["l1", "dram"])
def test_grouped_softmax_matches_per_group_softmax(device, num_groups, group_size, memory_config):
    """The (Q, groups*size) softmax must match softmax over each group of columns.

    Rows 0 and 1 put one group 40 and 200 below the others. Row 2 reproduces the
    [-100, -100, -100, -90] case beside zero logits. Every group must retain its distribution.
    """
    torch.manual_seed(0)
    num_queries = 96
    logits = torch.randn(1, num_queries, num_groups * group_size) * 3.0
    logits[0, 0, group_size : 2 * group_size] -= 40.0
    logits[0, 1, group_size : 2 * group_size] -= 200.0
    logits[0, 2] = 0.0
    logits[0, 2, group_size : 2 * group_size] = -100.0
    logits[0, 2, 2 * group_size - 1] = -90.0
    # Compare against the logits the device receives, including bf16 rounding at large offsets.
    logits = logits.bfloat16().float()
    expected = torch.softmax(logits.reshape(1, num_queries, num_groups, group_size), dim=-1)

    got = grouped_softmax(
        ttnn.from_torch(
            logits, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        ),
        group_size,
    )
    assert got.memory_config() == memory_config
    got = ttnn.to_torch(got).float().reshape(1, num_queries, num_groups, group_size)

    assert torch.isfinite(got).all()
    torch.testing.assert_close(got.sum(-1), torch.ones(1, num_queries, num_groups), atol=0.02, rtol=0)
    torch.testing.assert_close(got, expected, atol=0.01, rtol=0.02)
    passed, message = check_with_pcc(expected, got, pcc=0.9999)
    assert passed, message
