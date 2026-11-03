# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import torch
import ttnn
import pytest

from models.experimental.bevformer.tt import tt_common
from models.experimental.bevformer.tt.tt_ms_deformable_attention import TTMSDeformableAttention
from models.experimental.bevformer.tt.tt_spatial_cross_attention import (
    SCARebatchPlanner,
    TTSpatialCrossAttention,
    _rebatch_len,
)
from models.experimental.bevformer.reference.spatial_cross_attention import SpatialCrossAttention


from models.experimental.bevformer.tests.common import PRESETS

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

# Empirical threshold for gather-scatter row-routing regressions that global PCC can dilute; not an accuracy tolerance.
MAX_REBATCH_ROW_REL_ERROR = 0.3

# Module-scoped device: opens once per file instead of once per test case.
pytestmark = pytest.mark.use_module_device({"l1_small_size": 10 * 1024})


@pytest.mark.parametrize(
    "config_name, batch_size, bev_h, bev_w, expected_pcc, expected_abs_error, expected_rel_error, expected_high_error_ratio, blank_cams, valid_per_pair",
    [
        ("nuscenes_tiny", 1, 30, 30, 0.997, 0.06, 0.5, 0.5, (), None),  # NuScenes tiny model - 30x30 BEV grid
        ("nuscenes_base", 1, 100, 100, 0.999, 0.04, 1.3, 0.5, (), None),  # NuScenes base model - 100x100 BEV grid
        ("nuscenes_base", 1, 200, 200, 0.998, 0.04, 1.3, 0.5, (), None),  # NuScenes base model - 200x200 BEV grid
        # CARLA's rig projects symmetrically, NuScenes' does not, so max_len spreads
        # across cameras differently and pads the rebatch differently.
        ("carla_base", 1, 100, 100, 0.998, 0.04, 1.3, 0.5, (), None),  # CARLA base model
        ("nuscenes_base", 2, 30, 30, 0.998, 0.04, 1.3, 0.5, ((2, 1),), None),  # bs=2, blind cam in one item
        ("nuscenes_base", 1, 30, 30, 0.998, 0.04, 1.3, 0.5, "all", None),  # max_len == 0: residual-only return
        # max_len at the tile edge: the extra row still leaves the padding row unfilled slots read.
        ("nuscenes_base", 1, 30, 30, 0.998, 0.04, 1.3, 0.5, ((2, 0),), 32),
        # Off: a grid size between the 30x30 and 100x100 cases, no distinct path.
        # ("nuscenes_base", 1, 50, 50, 0.999, 0.04, 1.3, 0.5, (), None),  # NuScenes base model - 50x50 BEV grid
    ],
)
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
    config = PRESETS[config_name]

    # Extract parameters from configs
    embed_dims = config.embed_dims
    num_heads = config.num_heads
    num_levels = config.num_levels
    num_points = config.num_points
    num_cams = config.num_cams
    num_queries = bev_h * bev_w

    # Use spatial shapes from dataset config (limited to num_levels)
    spatial_shapes_list = config.spatial_shapes[:num_levels]
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
    # SCA takes camera features batch-first; the encoder owns the permute.
    tt_camera_features = ttnn.from_torch(
        camera_features.permute(2, 0, 1, 3), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    tt_reference_points_cam = ttnn.from_torch(
        reference_points_cam, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    tt_bev_mask = ttnn.from_torch(bev_mask, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # --------------------------------------------------------------------------- #
    # Models Init                                                                 #
    # --------------------------------------------------------------------------- #

    # Create PyTorch reference model using extracted parameters
    ref_model = SpatialCrossAttention(
        embed_dims=embed_dims,
        num_cams=num_cams,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
    )
    ref_model.eval()

    # Create preprocessed parameters from PyTorch model
    tt_parameters = create_spatial_cross_attention_parameters(ref_model, device, dtype=ttnn.float32)

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
        spatial_shapes=spatial_shapes,
    )

    # --------------------------------------------------------------------------- #
    # Models Forward                                                              #
    # --------------------------------------------------------------------------- #

    # Forward pass with PyTorch reference model
    ref_model_output = ref_model(
        query=bev_queries,
        value=camera_features,
        reference_points_cam=reference_points_cam,
        bev_mask=bev_mask,
        spatial_shapes=spatial_shapes,
    )

    # Forward pass with ttnn model
    tt_model_output = tt_model(
        query=tt_bev_queries,
        reference_points_cam=tt_reference_points_cam,
        bev_mask=tt_bev_mask,
        value=tt_camera_features,
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
        return

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
    assert row_rel_error[worst_row] <= MAX_REBATCH_ROW_REL_ERROR, (
        f"Query row {worst_row % num_queries} of batch item {worst_row // num_queries} has relative "
        f"error {row_rel_error[worst_row]:.4f} > {MAX_REBATCH_ROW_REL_ERROR} "
        f"(row norm {row_norm[worst_row]:.4f}); mean over rows is {row_rel_error.mean():.4f}."
    )

    if ENABLE_LOGGING:
        logger.info("✅ All SCA tolerance checks passed successfully!")


# The rebatch plan on its own: its routing must be exact, which a global PCC would dilute.

NUM_DEPTH_LEVELS = 4


def random_mask(num_cams, bs, num_queries, visible, generator, later_samples_extra=0):
    """``[num_cams, bs, num_queries, D]`` with ``visible[i]`` queries of camera ``i`` seen at one or
    more depth levels in the first sample, ``later_samples_extra`` more in the others."""
    mask = torch.zeros(num_cams, bs, num_queries, NUM_DEPTH_LEVELS, dtype=torch.bool)
    for i in range(num_cams):
        for j in range(bs):
            n = max(0, visible[i] + (later_samples_extra if j else 0))
            queries = torch.randperm(num_queries, generator=generator)[:n]
            levels = torch.randint(0, NUM_DEPTH_LEVELS, (queries.numel(),), generator=generator)
            mask[i, j, queries, levels] = True
            # A second level on some queries: the plan counts a query once however many levels hit.
            mask[i, j, queries[::3], (levels[::3] + 1) % NUM_DEPTH_LEVELS] = True
    return mask


def host_plan(mask, rebatch_len):
    """The plan's indices built on the host with ``nonzero``: every sample gathers the queries the
    first sample's cameras see, a query's slots fill in camera order."""
    num_cams, bs, num_queries, _ = mask.shape
    seen_by = mask.any(-1)
    visible = seen_by[:, :1].expand_as(seen_by)
    compact = torch.full((bs, num_cams, rebatch_len), num_queries - 1, dtype=torch.int64)
    max_cameras = int(visible.sum(0).max())
    inverse = torch.full((max_cameras, bs, num_queries), -1, dtype=torch.int64)
    next_slot = torch.zeros(bs, num_queries, dtype=torch.int64)
    for j in range(bs):
        for i in range(num_cams):
            ids = torch.nonzero(visible[i, j]).squeeze(-1)
            compact[j, i, : ids.numel()] = ids
            inverse[next_slot[j, ids], j, ids] = (j * num_cams + i) * rebatch_len + torch.arange(ids.numel())
            next_slot[j, ids] += 1
    return dict(
        compact=compact,
        inverse=inverse,
        max_cameras=max_cameras,
        num_valid=visible.sum(-1).t().flatten(),
        count=seen_by.sum(0).clamp(min=1),
    )


def upload_frame(mask, device, generator):
    num_cams, bs, num_queries, _ = mask.shape
    points = torch.rand(num_cams, bs, num_queries, NUM_DEPTH_LEVELS, 2, generator=generator)
    return (
        points,
        ttnn.from_torch(points, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
        ttnn.from_torch(mask.to(torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
    )


def check_plan(plan, mask, points):
    """Assert ``plan`` routes every row as the host build at the plan's own ``rebatch_len``; padded
    rows may hold anything, except the last of each camera, which the attention zeroes."""
    num_cams, bs, num_queries, _ = mask.shape
    rebatch_len = plan.rebatch_len
    expected = host_plan(mask, rebatch_len)
    assert int(expected["num_valid"].max()) < rebatch_len, "the plan holds fewer rows than a camera sees"
    assert plan.num_valid_queries.dtype == ttnn.uint32
    assert ttnn.to_torch(plan.num_valid_queries).tolist() == expected["num_valid"].flatten().tolist()

    expected_rows = torch.arange(rebatch_len) < expected["num_valid"].view(-1, 1)

    # Padded rows gather some query of their own sample; which one is not part of the plan.
    query_index = ttnn.to_torch(plan.query_index).flatten().to(torch.int64).view(bs * num_cams, rebatch_len)
    offsets = (torch.arange(bs) * num_queries).view(bs, 1, 1)
    want = (expected["compact"] + offsets).view(bs * num_cams, rebatch_len)
    assert torch.equal(query_index[expected_rows], want[expected_rows])
    sample_start = offsets.expand(bs, num_cams, rebatch_len).reshape(bs * num_cams, rebatch_len)
    padded_ids = (query_index - sample_start)[~expected_rows]
    assert torch.all((padded_ids >= 0) & (padded_ids < num_queries))

    assert torch.equal(ttnn.to_torch(plan.count).flatten().to(torch.int64), expected["count"].flatten())

    # Filled slots point at exactly the host's row; unfilled ones, including slots past this frame's
    # most cameras per query, at the last row of camera q % num_cams of their own sample.
    assert len(plan.inverse_index) >= expected["max_cameras"]
    unfilled_row = (
        (torch.arange(bs).view(bs, 1) * num_cams + torch.arange(num_queries).view(1, num_queries) % num_cams)
        * rebatch_len
        + rebatch_len
        - 1
    )
    for slot, index in enumerate(plan.inverse_index):
        rows = ttnn.to_torch(index).flatten().to(torch.int64).view(bs, num_queries)
        want = expected["inverse"][slot] if slot < expected["max_cameras"] else torch.full_like(rows, -1)
        filled = want >= 0
        assert torch.equal(rows[filled], want[filled]), f"slot {slot} routes a contribution to the wrong row"
        assert torch.equal(
            rows[~filled], unfilled_row[~filled]
        ), f"slot {slot} reads the wrong row for an unfilled slot"

    # The gathered reference points are the compacted queries' own, clamped as upstream.
    gathered = ttnn.to_torch(plan.reference_points_batched).float()
    host_points = points.clamp(-10.0, 10.0).to(torch.bfloat16).float().permute(1, 0, 2, 3, 4)
    for j in range(bs):
        for i in range(num_cams):
            n = int(expected["num_valid"][j * num_cams + i])
            assert torch.equal(gathered[j * num_cams + i, :n], host_points[j, i, expected["compact"][j, i, :n]])
    return expected


@pytest.mark.parametrize(
    "num_cams, bs, num_queries, visible, later_samples_extra",
    [
        (6, 1, 2500, (473, 400, 380, 300, 440, 470), 0),
        (6, 2, 900, (200, 0, 150, 120, 180, 199), 0),  # a blind camera
        (6, 2, 900, (200, 100, 150, 120, 180, 199), 60),  # the second sample sees more than the first
        (6, 2, 900, (200, 100, 150, 120, 180, 199), -60),  # and fewer
        (6, 1, 40000, (9885, 9500, 9400, 9000, 9700, 9800), 0),  # base grid
        (2, 1, 64, (31, 32), 0),  # camera counts one below and at the tile edge
    ],
)
def test_rebatch_plan_matches_host(device, num_cams, bs, num_queries, visible, later_samples_extra):
    generator = torch.Generator().manual_seed(0)
    mask = random_mask(num_cams, bs, num_queries, visible, generator, later_samples_extra)
    points, tt_points, tt_mask = upload_frame(mask, device, generator)

    planner = SCARebatchPlanner(device, headroom=1.0)
    plan = planner(tt_points, planner.visibility(tt_mask))
    # Sized by the first sample's cameras, whatever the others see.
    assert planner.capacity == max(visible), "device and host max_len differ"
    assert plan.rebatch_len == _rebatch_len(max(visible))
    check_plan(plan, mask, points)


def test_rebatch_plan_empty_frame(device):
    generator = torch.Generator().manual_seed(0)
    mask = torch.zeros(6, 1, 900, NUM_DEPTH_LEVELS, dtype=torch.bool)
    _, tt_points, tt_mask = upload_frame(mask, device, generator)
    planner = SCARebatchPlanner(device)
    plan = planner(tt_points, planner.visibility(tt_mask))
    assert plan.is_empty and plan.rebatch_len == 0
    assert planner.capacity == 0


@pytest.mark.parametrize("bs", [1, 2])
def test_rebatch_plan_over_a_sequence(device, bs):
    """One planner over a frame sequence: capacity and slots grow only when a frame outgrows them and
    never shrink, frames within them keep the plan's shapes, and every frame's plan matches the host
    build at that capacity, however far below it the frame is."""
    generator = torch.Generator().manual_seed(0)
    num_cams, num_queries = 6, 2500
    planner = SCARebatchPlanner(device, headroom=1.01)
    # Per-frame visible queries of the busiest camera; 0 is a frame no camera sees anything in.
    sequence = [600, 603, 597, 606, 607, 650, 0, 640, 655, 300, 657]
    capacity, num_slots, reallocations, previous_len = 0, 0, 0, None
    for frame, busiest in enumerate(sequence):
        visible = tuple(max(0, busiest - drop) if busiest else 0 for drop in (0, 50, 100, 20, 80, 10))
        mask = random_mask(num_cams, bs, num_queries, visible, generator)
        points, tt_points, tt_mask = upload_frame(mask, device, generator)
        plan = planner(tt_points, planner.visibility(tt_mask))

        if busiest == 0:
            assert plan.is_empty, f"frame {frame}"
            assert (planner.capacity, planner.num_slots) == (capacity, num_slots), f"frame {frame}"
            continue
        expected = check_plan(plan, mask, points)
        grew = busiest > capacity or expected["max_cameras"] > num_slots
        reallocations += grew and capacity > 0
        if busiest > capacity:
            capacity = math.ceil(busiest * 1.01)
        num_slots = max(num_slots, expected["max_cameras"])
        assert (planner.capacity, planner.num_slots) == (capacity, num_slots), f"frame {frame}"
        assert planner.reallocations == reallocations, f"frame {frame}"
        assert len(plan.inverse_index) == num_slots, f"frame {frame}"
        if not grew:
            assert plan.rebatch_len == previous_len, f"frame {frame} changed shape within capacity"
        previous_len = plan.rebatch_len

    # 607 outgrows ceil(600 * 1.01) = 606, and 650 outgrows ceil(607 * 1.01) = 614.
    assert capacity == math.ceil(650 * 1.01)
    assert reallocations >= 2


def test_spatial_cross_attention_glue_intermediates_in_dram(device, monkeypatch):
    """The 100x100 base case with every group of intermediates around the fused op in DRAM, as
    ``tt_common.l1_if_fits`` places them when they do not fit L1. Where the fused op stages its
    own value is decided apart, by ``value_memory`` (``test_fused_msda.py``)."""
    monkeypatch.setattr(tt_common, "L1_INTERMEDIATE_SHARE", 0.0)
    test_spatial_cross_attention_forward(
        device, "nuscenes_base", 1, 100, 100, 0.999, 0.04, 1.3, 0.5, blank_cams=(), valid_per_pair=None, seed=0
    )


@pytest.mark.parametrize(
    "config_name, batch_size, bev_h, bev_w, blank_cams",
    [("nuscenes_base", 1, 100, 100, ()), ("nuscenes_base", 2, 30, 30, ((2, 1),))],
)
def test_spatial_cross_attention_ignores_unwritten_padding(
    device, monkeypatch, config_name, batch_size, bev_h, bev_w, blank_cams
):
    """MSDA leaves each camera's rows past its valid count unwritten. Fill them with NaN: the
    aggregation zeroes every camera's last row and routes every unfilled slot there, so no NaN may
    reach the output, for each sample of the batch."""
    original = TTMSDeformableAttention.forward

    def poisoned(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        counts = ttnn.to_torch(kwargs["num_valid_queries_tensor"]).flatten().to(torch.int64)
        rows = ttnn.to_torch(out)
        assert rows.shape[0] == counts.numel(), "one valid count per camera row block"
        for block, count in enumerate(counts.tolist()):
            rows[block, count:] = float("nan")
        return ttnn.from_torch(
            rows, dtype=out.dtype, layout=out.layout, device=self.device, memory_config=out.memory_config()
        )

    monkeypatch.setattr(TTMSDeformableAttention, "forward", poisoned)
    test_spatial_cross_attention_forward(
        device,
        config_name,
        batch_size,
        bev_h,
        bev_w,
        0.998,
        0.04,
        1.3,
        0.5,
        blank_cams=blank_cams,
        valid_per_pair=None,
        seed=0,
    )
