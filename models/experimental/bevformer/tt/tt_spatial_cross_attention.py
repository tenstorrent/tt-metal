# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of BEVFormer's spatial cross-attention (``reference/spatial_cross_attention.py``).

Which BEV queries a camera sees depends on the camera geometry only. :func:`build_rebatch_plan`
turns a frame's projected pillar points and visibility mask into gather and scatter indices,
the cameras' float32 reference points and the per-query camera count, on the host, once per
frame (``TTBEVFormerEncoder.prepare_frame``). The forward then gathers each camera's queries,
runs ``MSDeformableAttention3D`` (a ``TTMSDeformableAttention`` with neither output projection nor
shortcut) over the camera features, scatters the results back, averages them and projects, all
on device. Parameters come from ``model_preprocessing.create_spatial_cross_attention_parameters``.
"""

import ttnn
import torch
from dataclasses import dataclass
from typing import Optional


from .tt_ms_deformable_attention import TTMSDeformableAttention
from ..config import DeformableAttentionConfig

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

from loguru import logger

# Enable/disable logging output
ENABLE_LOGGING = False


def _batch_offsets(bs: int, num_queries: int) -> torch.Tensor:
    """Convert batch-local query IDs to stacked row IDs."""
    return (torch.arange(bs, dtype=torch.int32) * num_queries).reshape(bs, 1, 1)


def _flat_row_index(row_ids: torch.Tensor, device) -> ttnn.Tensor:
    """Format row IDs for ``ttnn.embedding``."""
    return ttnn.from_torch(
        row_ids.reshape(1, 1, 1, row_ids.numel()),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _index_dtype(num_rows: int):
    """Use the narrowest scatter index that covers all rows."""
    return ttnn.uint16 if num_rows <= 0xFFFF else ttnn.uint32


@dataclass(frozen=True)
class SCARebatchPlan:
    """Per-frame SCA rebatch plan shared by every encoder layer.

    The plan is valid only for the reference points, visibility mask, tensor shapes, and device used to build it.
    Tensor fields are ``None`` when ``is_empty`` is true.

    Attributes:
        rebatch_len: Tile-aligned number of query rows processed per camera.
        query_index: Gather indices for query rows.
        reference_points_batched: Rebatched camera reference points, ROW_MAJOR, with shape
            ``[batch_size * num_cams, rebatch_len, num_depth_levels, 2]``.
        scatter_index: Expanded query indices used to accumulate camera outputs.
        scatter_base: Zeros ``[batch_size, num_queries + 1, embed_dims]``, ROW_MAJOR, that the
            camera outputs are scatter-added into (out of place, so every layer reuses it); the
            last row is the padding sink. Created here, as creating it in the forward is a host
            write.
        count: Camera-contributor count with shape ``[batch_size, num_queries, 1]``.
        is_empty: Whether the visibility mask contains no valid query-camera pairs.
    """

    rebatch_len: int
    query_index: Optional[ttnn.Tensor]
    reference_points_batched: Optional[ttnn.Tensor]
    scatter_index: Optional[ttnn.Tensor]
    scatter_base: Optional[ttnn.Tensor]
    count: Optional[ttnn.Tensor]
    is_empty: bool = False


def build_rebatch_plan(
    reference_points_cam, bev_mask, embed_dims: int, device, grid_dtype=ttnn.float32
) -> SCARebatchPlan:
    """Build a rebatch plan for one frame's camera projections, on the host.

    Args:
        reference_points_cam: float32 torch tensor ``[num_cams, batch_size, num_queries, num_depth_levels, 2]``,
            the pillar points in each camera's normalized image coordinates.
        bev_mask: Bool torch tensor ``[num_cams, batch_size, num_queries, num_depth_levels]``.
        embed_dims: Query embedding width used to expand scatter indices.
        device: Device on which plan tensors are allocated.
        grid_dtype: Dtype of the rebatched reference points. In bfloat16 a point near the image's
            right edge moves in steps of 2^-8, 0.8 px on the 200-wide first FPN level.

    The geometry depends on the cameras only, so it is built once per frame from the camera
    parameters, before the forward; the forward only gathers, attends and scatters on device.
    A plan with no visible query-camera pair has ``is_empty`` set and no tensors.
    """
    num_cams, bs, num_queries, num_depth_levels = bev_mask.shape
    valid_per_cam = bev_mask.sum(-1) > 0  # [num_cams, B, num_queries]
    max_len = int(valid_per_cam.sum(-1).max().item())
    if max_len == 0:
        return SCARebatchPlan(
            rebatch_len=0,
            query_index=None,
            reference_points_batched=None,
            scatter_index=None,
            scatter_base=None,
            count=None,
            is_empty=True,
        )

    # Tile-align so folding num_cams into the row dim stays a view; unaligned would move data.
    # Costs up to TILE_SIZE - 1 padded rows per camera through MSDA, discarded afterwards.
    rebatch_len = ((max_len + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE

    # Compact each camera to the queries it sees, so MSDA runs on rebatch_len rows, not all of
    # them. IDs are batch-local; num_queries is the padding sentinel: scatter_ids keeps it to
    # dump padding in a sink row sliced off later, gather_ids clamps it since embedding rejects
    # out-of-range indices.
    scatter_ids = torch.full((bs, num_cams, rebatch_len), num_queries, dtype=torch.int32)
    reference_points_batched = torch.zeros(bs, num_cams, rebatch_len, num_depth_levels, 2)
    for j in range(bs):
        for i in range(num_cams):
            valid_indices = torch.nonzero(valid_per_cam[i, j], as_tuple=False).squeeze(-1)
            scatter_ids[j, i, : valid_indices.numel()] = valid_indices.to(torch.int32)
            reference_points_batched[j, i, : valid_indices.numel()] = reference_points_cam[i, j, valid_indices]
    gather_ids = scatter_ids.clamp(max=num_queries - 1)

    # Expand row IDs on device to avoid transferring a full-width index.
    scatter_index = ttnn.repeat(
        ttnn.from_torch(
            scatter_ids.reshape(bs, num_cams * rebatch_len, 1),
            device=device,
            dtype=_index_dtype(num_queries + 1),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        ttnn.Shape((1, 1, embed_dims)),
    )
    count = torch.clamp(valid_per_cam.permute(1, 2, 0).sum(-1), min=1.0)
    count = ttnn.unsqueeze(ttnn.from_torch(count, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), -1)
    return SCARebatchPlan(
        rebatch_len=rebatch_len,
        query_index=_flat_row_index(gather_ids + _batch_offsets(bs, num_queries), device),
        reference_points_batched=ttnn.from_torch(
            reference_points_batched.reshape(bs * num_cams, rebatch_len, num_depth_levels, 2),
            device=device,
            dtype=grid_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        scatter_index=scatter_index,
        scatter_base=ttnn.zeros(
            (bs, num_queries + 1, embed_dims), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        ),
        count=count,
    )


class TTSpatialCrossAttention:
    """
    TTNN Spatial Cross Attention module for BEVFormer.

    This attention mechanism allows BEV queries to extract spatial features
    from regions of interest across camera views using deformable attention.
    Each BEV query can attend to multiple camera features at different scales
    and locations.

    Note: This module expects pre-projected reference points and validity masks.
    Point sampling/projection from 3D to camera coordinates should be handled
    by the encoder before calling this attention module.

    Args:
        device: TTNN device for computation
        params: Parameter dict containing weights and biases
        embed_dims (int): The embedding dimension.
        num_cams (int): Number of cameras.
        batch_first (bool): Whether the first dimension of input is batch_size.
        deformable_attention (dict): Config for MSDeformableAttention.
        spatial_shapes: Multi-scale feature shapes [num_levels, 2]
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        device,
        params,
        embed_dims: int = 256,
        num_cams: int = 6,
        batch_first: bool = True,
        deformable_attention: Optional[dict] = None,
        *,
        spatial_shapes,
        grid_dtype=None,
        grid_sample_compute_config=None,
        **kwargs,
    ):
        self.device = device
        self.params = params
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.batch_first = batch_first

        if deformable_attention is None:
            deformable_attention = dict(embed_dims=embed_dims, num_levels=4, num_points=8, num_heads=8)

        deform_config = DeformableAttentionConfig(
            embed_dims=deformable_attention.get("embed_dims", embed_dims),
            num_heads=deformable_attention.get("num_heads", 8),
            num_levels=deformable_attention.get("num_levels", 4),
            num_points=deformable_attention.get("num_points", 8),
            batch_first=batch_first,
        )

        # MSDeformableAttention3D: no output projection and no shortcut; the SCA projects the
        # merged slots and adds its own residual.
        self.deformable_attention = TTMSDeformableAttention(
            deform_config,
            device,
            params.deformable_attention,
            spatial_shapes=spatial_shapes,
            grid_dtype=grid_dtype,
            grid_sample_compute_config=grid_sample_compute_config,
            residual=False,
        )

    def forward(
        self,
        *,
        query,
        value,
        rebatch_plan,
        residual=None,
        query_pos=None,
        **kwargs,
    ):
        """
        Forward pass of TTNN Spatial Cross Attention.

        Args:
            query: Bfloat16 BEV queries [B, num_queries, embed_dims].
            value: Multi-camera features [num_cams, H*W, B, embed_dims]. Attention here takes
                no key: deformable attention never scores query against one. Two Linears on
                query predict where to sample and with what weight, replacing Q*K^T, so
                ``value`` is the only feature tensor the device path needs.
            residual: Residual connection input.
            query_pos: Query positional encoding.
            rebatch_plan: :class:`SCARebatchPlan` from :func:`build_rebatch_plan`, shared by every encoder layer of a frame.
            **kwargs: Additional arguments.

        Returns:
            Output features [B, num_queries, embed_dims].
        """
        if use_signpost:
            signpost(header="TTNN SCA Forward Start")

        assert value is not None, "value is required"

        # Handle input defaults
        if residual is None:
            inp_residual = query
        else:
            inp_residual = residual

        # Add query positional encoding
        if query_pos is not None:
            query = ttnn.add(query, query_pos)

        if ENABLE_LOGGING:
            logger.info("SCA Tensor Conversion Complete")

        bs, num_queries, _ = query.shape
        if rebatch_plan.is_empty:
            if ENABLE_LOGGING:
                logger.warning("No valid points found in SCA, returning residual")
            return inp_residual
        num_depth_levels = rebatch_plan.reference_points_batched.shape[2]
        assert self.deformable_attention.num_points % num_depth_levels == 0, (
            f"num_points ({self.deformable_attention.num_points}) must split evenly over the "
            f"{num_depth_levels} pillar points"
        )

        rebatch_len = rebatch_plan.rebatch_len

        if ENABLE_LOGGING:
            logger.info("SCA Rebatching Start")

        assert query.dtype == ttnn.bfloat16, f"SCA rebatch gathers query, so it must be bfloat16, got {query.dtype}."

        # Fold batch and camera into row IDs to gather all valid queries in one embedding call.
        # Reference points come from the plan — they do not depend on query.
        query_rows = ttnn.reshape(
            ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT), (1, 1, bs * num_queries, self.embed_dims)
        )
        queries_batched = ttnn.reshape(
            ttnn.embedding(rebatch_plan.query_index, query_rows, layout=ttnn.TILE_LAYOUT),
            (bs * self.num_cams, rebatch_len, self.embed_dims),
        )
        reference_points_batched = rebatch_plan.reference_points_batched

        if ENABLE_LOGGING:
            logger.info("SCA Rebatching Complete")

        _, L, _, _ = value.shape

        # [num_cams, L, bs, embed_dims] -> [bs * num_cams, L, embed_dims]
        value_reshaped = ttnn.permute(value, (2, 0, 1, 3))  # [bs, num_cams, L, embed_dims]
        value_reshaped = ttnn.reshape(value_reshaped, (bs * self.num_cams, L, self.embed_dims))

        if ENABLE_LOGGING:
            logger.info("SCA Calling Deformable Attention")
        queries_output = self.deformable_attention(
            query=queries_batched,
            value=value_reshaped,
            reference_points=reference_points_batched,
            **kwargs,
        )

        if isinstance(queries_output, torch.Tensor):
            queries_output = ttnn.from_torch(
                queries_output, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )

        if ENABLE_LOGGING:
            logger.info("SCA Deformable Attention Complete")

        if ENABLE_LOGGING:
            logger.info("SCA Feature Aggregation Start")

        # Accumulate camera features by query ID. Unclamped IDs, so padding lands in the sink row
        # the slice below drops.
        scatter_src = ttnn.to_layout(
            ttnn.reshape(queries_output, (bs, self.num_cams * rebatch_len, self.embed_dims)),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        slots = ttnn.scatter_add(
            rebatch_plan.scatter_base,
            dim=1,
            index=rebatch_plan.scatter_index,
            src=scatter_src,
        )
        slots = ttnn.to_layout(ttnn.slice(slots, (0, 0, 0), (bs, num_queries, self.embed_dims)), ttnn.TILE_LAYOUT)

        if ENABLE_LOGGING:
            logger.info("SCA Feature Aggregation Complete")

        slots = ttnn.div(slots, rebatch_plan.count)

        # Output projection
        if hasattr(self.params, "output_proj") and self.params.output_proj is not None:
            slots = ttnn.to_layout(slots, ttnn.TILE_LAYOUT)
            slots = ttnn.linear(slots, self.params.output_proj.weight, bias=self.params.output_proj.bias)

        if ENABLE_LOGGING:
            logger.info("SCA Adding Residual")

        # Residual connection
        output = ttnn.add(slots, inp_residual)

        if use_signpost:
            signpost(header="TTNN SCA Forward End")

        return output

    def __call__(self, *args, **kwargs):
        """Make the class callable"""
        return self.forward(*args, **kwargs)

    def extra_repr(self) -> str:
        """String representation for debugging"""
        return f"embed_dims={self.embed_dims}, num_cams={self.num_cams}, " f"batch_first={self.batch_first}"
