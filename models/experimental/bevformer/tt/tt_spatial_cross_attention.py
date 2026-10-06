# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of BEVFormer's spatial cross-attention (``reference/spatial_cross_attention.py``).

Which BEV queries a camera sees depends on the camera geometry only, not on the activations.
:class:`SCARebatchPlan` holds, per frame, the gather and scatter indices, the cameras' float32
reference points and each query's camera count, built on the host from the frame's projected
pillar points and visibility mask (``TTBEVFormerEncoder.prepare_frame``). Its device buffers are
allocated once, with a fixed capacity per camera, and refilled in place for every later frame,
so a captured trace stays valid from frame to frame. The forward then gathers each camera's
queries, runs ``MSDeformableAttention3D`` (a ``TTMSDeformableAttention`` with neither output
projection nor shortcut) over the camera features, scatters the results back, averages them and
projects, all on device.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import ttnn

from ..config import DeformableAttentionConfig
from .tt_common import GRID_DTYPE
from .tt_ms_deformable_attention import TTMSDeformableAttention


def _index_dtype(num_rows):
    """The narrowest scatter index that covers ``num_rows``."""
    return ttnn.uint16 if num_rows <= 0xFFFF else ttnn.uint32


@dataclass
class SCARebatchPlan:
    """One frame's spatial cross-attention rebatch plan, shared by every encoder layer.

    Each camera's visible queries are compacted into ``capacity`` rows (tile-aligned, at least one
    tile), so the deformable attention runs on those rows instead of all queries; unused rows
    gather the sample's last query (any valid row; the result is discarded) and scatter into a
    sink row past the last query, which the forward drops.

    Attributes:
        capacity: Rows per camera; fixed when the plan is built, as the buffers' shapes depend on it.
        batch_size, num_cams, num_queries, num_points_in_pillar: The frame shape the plan is built
            for; a refill must match it.
        query_index: ``[1, 1, 1, bs * num_cams * capacity]`` uint32, the stacked query row each
            rebatched row gathers.
        reference_points: ``[bs * num_cams, capacity, num_points_in_pillar, 2]`` ROW_MAJOR, in the
            ``grid_dtype`` the plan was built with (float32 in the encoder),
            the rebatched rows' pillar points in their camera's normalized image coordinates.
        scatter_ids: ``[bs, num_cams * capacity, 1]`` ROW_MAJOR, the query each row adds into, or the
            sink row ``num_queries``.
        inverse_count: ``[bs, num_queries, 1]``, one over the number of cameras that see each query
            (at least one).
        scatter_base: Zeros ``[bs, num_queries + 1, embed_dims]`` ROW_MAJOR that the camera outputs
            are scatter-added into, out of place; created here, as creating it in the forward is a
            host write.
    """

    capacity: int
    batch_size: int
    num_cams: int
    num_queries: int
    num_points_in_pillar: int
    query_index: ttnn.Tensor
    reference_points: ttnn.Tensor
    scatter_ids: ttnn.Tensor
    inverse_count: ttnn.Tensor
    scatter_base: ttnn.Tensor


def _host_plan(reference_points_cam, bev_mask, capacity):
    """The plan's host tensors. As upstream, every sample gathers the queries the first sample's
    cameras see; the camera count and the reference points are each sample's own."""
    num_cams, bs, num_queries, depth = bev_mask.shape
    visible = bev_mask[:, 0].sum(-1) > 0  # [num_cams, num_queries]
    scatter_ids = torch.full((bs, num_cams, capacity), num_queries, dtype=torch.int32)
    reference_points = torch.zeros(bs, num_cams, capacity, depth, 2)
    for i in range(num_cams):
        index = torch.nonzero(visible[i], as_tuple=False).squeeze(-1)
        assert (
            index.numel() <= capacity
        ), f"camera {i} sees {index.numel()} queries, over the plan's capacity {capacity}"
        scatter_ids[:, i, : index.numel()] = index.to(torch.int32)
        reference_points[:, i, : index.numel()] = reference_points_cam[i][:, index]
    gather_ids = scatter_ids.clamp(max=num_queries - 1) + (torch.arange(bs, dtype=torch.int32) * num_queries).view(
        bs, 1, 1
    )
    count = torch.clamp((bev_mask.sum(-1) > 0).permute(1, 2, 0).sum(-1), min=1.0)
    return SimpleNamespace(
        query_index=gather_ids.reshape(1, 1, 1, -1),
        reference_points=reference_points.reshape(bs * num_cams, capacity, depth, 2),
        scatter_ids=scatter_ids.reshape(bs, num_cams * capacity, 1),
        inverse_count=(1.0 / count).unsqueeze(-1),
    )


def _plan_dtypes(num_queries, grid_dtype):
    """Each host-built plan buffer's device dtype and layout, shared by the build and the refill so
    ``copy_host_to_device_tensor`` sees matching tensors."""
    return dict(
        query_index=(ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        reference_points=(grid_dtype, ttnn.ROW_MAJOR_LAYOUT),
        scatter_ids=(_index_dtype(num_queries + 1), ttnn.ROW_MAJOR_LAYOUT),
        inverse_count=(ttnn.bfloat16, ttnn.TILE_LAYOUT),
    )


def full_capacity(num_queries):
    """Every query, tile-aligned: no rig can see more. The deformable attention then runs on every
    query for every camera, which fits the tiny grid but not the base one: at 200x200 the encoder
    fails to allocate its sampling buffers, one of them 3.9 GB. On the base grid use a bound for the
    rig; with nuScenes' rig the busiest camera sees about a quarter of the grid."""
    return -(-num_queries // ttnn.TILE_SIZE) * ttnn.TILE_SIZE


def build_rebatch_plan(
    reference_points_cam, bev_mask, embed_dims, device, grid_dtype=GRID_DTYPE, capacity=None
) -> SCARebatchPlan:
    """A plan for one frame, on device.

    Args:
        reference_points_cam: float32 torch tensor ``[num_cams, bs, num_queries, num_points_in_pillar, 2]``
            (``point_sampling_3d_2d.camera_geometry``).
        bev_mask: Bool torch tensor ``[num_cams, bs, num_queries, num_points_in_pillar]``.
        embed_dims: Query width, for the scatter base.
        device: Device the plan's buffers live on.
        grid_dtype: Dtype of the rebatched reference points. In bfloat16 a point near the image's
            right edge moves in steps of 2^-8, 0.8 px on the 200-wide first FPN level.
        capacity: Rows per camera. By default the most queries any camera sees in this frame,
            tile-aligned, which fits this frame only: a plan refilled by :func:`update_rebatch_plan`
            needs a capacity that covers every later frame: a bound for the rig, or
            :func:`full_capacity` on grids smaller than the base one.
    """
    num_cams, bs, num_queries, depth = bev_mask.shape
    if capacity is None:
        most_visible = int((bev_mask[:, 0].sum(-1) > 0).sum(-1).max())
        capacity = max(ttnn.TILE_SIZE, -(-most_visible // ttnn.TILE_SIZE) * ttnn.TILE_SIZE)
    assert capacity % ttnn.TILE_SIZE == 0, f"capacity {capacity} must be tile-aligned"
    host = _host_plan(reference_points_cam, bev_mask, capacity)
    tensors = {
        name: ttnn.from_torch(getattr(host, name), dtype=dtype, layout=layout, device=device)
        for name, (dtype, layout) in _plan_dtypes(num_queries, grid_dtype).items()
    }
    return SCARebatchPlan(
        capacity=capacity,
        batch_size=bs,
        num_cams=num_cams,
        num_queries=num_queries,
        num_points_in_pillar=depth,
        scatter_base=ttnn.zeros(
            (bs, num_queries + 1, embed_dims), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        ),
        **tensors,
    )


def update_rebatch_plan(plan: SCARebatchPlan, reference_points_cam, bev_mask):
    """Refill ``plan`` in place for a new frame: the buffers keep their shapes and addresses, so a
    trace captured with the plan replays on the new frame. The frame must have the plan's batch
    size, cameras, BEV size and pillar depth, and no camera may see more queries than the plan's
    capacity; the cameras' poses may differ."""
    shape = (plan.num_cams, plan.batch_size, plan.num_queries, plan.num_points_in_pillar)
    assert tuple(bev_mask.shape) == shape, (
        f"the plan is for (num_cams, bs, num_queries, num_points_in_pillar) {shape}, the frame is "
        f"{tuple(bev_mask.shape)}; build a new plan"
    )
    host = _host_plan(reference_points_cam, bev_mask, plan.capacity)
    for name, (dtype, layout) in _plan_dtypes(plan.num_queries, plan.reference_points.dtype).items():
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(getattr(host, name), dtype=dtype, layout=layout), getattr(plan, name)
        )


class TTSpatialCrossAttention:
    """Spatial cross-attention over a frame's :class:`SCARebatchPlan`.

    ``MSDeformableAttention3D`` folds ``spatial_shapes`` into its sampling-offset Linear, which
    consumes ``params.deformable_attention.sampling_offsets``: each instance needs its own params.
    """

    def __init__(
        self,
        params,
        device,
        *,
        spatial_shapes,
        embed_dims=256,
        num_cams=6,
        num_heads=8,
        num_levels=4,
        num_points=8,
        grid_dtype=GRID_DTYPE,
        grid_sample_compute_config=None,
    ):
        """``params`` from ``model_preprocessing.create_spatial_cross_attention_parameters``;
        ``spatial_shapes`` the camera feature levels as (h, w); ``num_points`` sampling points per
        head and level, split evenly over the pillar's points."""
        self.params = params
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.deformable_attention = TTMSDeformableAttention(
            DeformableAttentionConfig(
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                batch_first=True,
            ),
            device,
            params.deformable_attention,
            spatial_shapes=spatial_shapes,
            grid_dtype=grid_dtype,
            grid_sample_compute_config=grid_sample_compute_config,
            residual=False,
            output_proj=False,
        )

    def frame_inputs(self, plan):
        """The device tensors every layer of a frame shares, derived from ``plan`` on device: the
        full-width scatter index and the deformable attention's grid bias. They are derived in the
        forward, not stored in the plan, so a refilled plan reaches them under trace replay."""
        assert (
            plan.reference_points.dtype == self.deformable_attention.grid_dtype
        ), f"plan points {plan.reference_points.dtype}, attention grid {self.deformable_attention.grid_dtype}"
        depth = plan.num_points_in_pillar
        assert (
            self.deformable_attention.num_points % depth == 0
        ), f"num_points ({self.deformable_attention.num_points}) must split evenly over the {depth} pillar points"
        return SimpleNamespace(
            plan=plan,
            scatter_index=ttnn.repeat(plan.scatter_ids, ttnn.Shape((1, 1, self.embed_dims))),
            grid_bias=self.deformable_attention.grid_bias_for(plan.reference_points, depth),
        )

    def __call__(self, query, value, frame):
        """``query`` bfloat16 ``(bs, num_queries, C)``, ``value`` ``(bs * num_cams, num_keys, C)``
        (each sample's cameras in turn), ``frame`` from :meth:`frame_inputs`. Returns
        ``(bs, num_queries, C)``. Deformable attention scores no key: two Linears on the query predict
        where to sample and with what weight. No positional encoding enters: upstream's encoder adds
        it in the self-attention only."""
        plan = frame.plan
        bs, num_queries, embed_dims = query.shape
        assert (bs, num_queries) == (
            plan.batch_size,
            plan.num_queries,
        ), f"query {(bs, num_queries)}, plan built for {(plan.batch_size, plan.num_queries)}"
        assert (
            query.dtype == ttnn.bfloat16
        ), f"the rebatch gathers query rows, so it must be bfloat16, got {query.dtype}"
        rows = bs * self.num_cams

        query_rows = ttnn.reshape(ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT), (1, 1, bs * num_queries, embed_dims))
        queries = ttnn.reshape(
            ttnn.embedding(plan.query_index, query_rows, layout=ttnn.TILE_LAYOUT), (rows, plan.capacity, embed_dims)
        )
        assert value.shape[0] == rows, f"value has {value.shape[0]} camera rows, expected bs * num_cams = {rows}"
        attended = self.deformable_attention(
            query=queries, value=value, reference_points=plan.reference_points, grid_bias=frame.grid_bias
        )

        # Unused rows add into the sink row past the last query, which the slice drops.
        attended = ttnn.to_layout(
            ttnn.reshape(attended, (bs, self.num_cams * plan.capacity, embed_dims)), ttnn.ROW_MAJOR_LAYOUT
        )
        slots = ttnn.scatter_add(plan.scatter_base, dim=1, index=frame.scatter_index, src=attended)
        slots = ttnn.to_layout(ttnn.slice(slots, (0, 0, 0), (bs, num_queries, embed_dims)), ttnn.TILE_LAYOUT)
        slots = ttnn.mul(slots, plan.inverse_count)
        slots = ttnn.linear(slots, self.params.output_proj.weight, bias=self.params.output_proj.bias)
        return ttnn.add(slots, query)
