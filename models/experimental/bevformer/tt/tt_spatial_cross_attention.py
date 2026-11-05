# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Spatial Cross Attention (SCA) module for BEVFormer.

This module implements the spatial cross-attention mechanism using TTNN operations
to enable BEV queries to extract spatial features from regions of interest across
multiple camera views using deformable attention.
"""

import math

import ttnn
import torch
from dataclasses import dataclass
from typing import NamedTuple, Optional


from .tt_common import signpost
from .tt_ms_deformable_attention import TTMSDeformableAttention
from ..model_config import DeformableAttentionConfig

from loguru import logger

# Enable/disable logging output
ENABLE_LOGGING = False


# Room for the visible counts to drift up between frames without changing the plan's shapes.
REBATCH_HEADROOM = 1.01


def _rebatch_len(capacity: int) -> int:
    """Tile-aligned rows per camera for up to ``capacity`` visible queries.

    Tile alignment keeps folding num_cams into the row dim a view. Aligning capacity + 1 keeps at
    least one padded row per camera, which aggregation needs as a zero source for queries that
    fewer than the plan's slots of cameras see.
    """
    return ((capacity // ttnn.TILE_SIZE) + 1) * ttnn.TILE_SIZE


def _row_index(rows: ttnn.Tensor) -> ttnn.Tensor:
    """Float32 row IDs, exact below 2**24, formatted for ``ttnn.embedding``."""
    rows = ttnn.to_layout(ttnn.typecast(rows, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.reshape(rows, (1, 1, 1, rows.logical_volume()))


@dataclass(frozen=True)
class SCARebatchPlan:
    """Per-frame SCA rebatch plan shared by every encoder layer.

    The plan is valid only for the reference points, visibility mask, tensor shapes, and device used to build it.
    Tensor fields are ``None`` and ``inverse_index`` is empty when ``is_empty`` is true.

    Attributes:
        rebatch_len: Tile-aligned number of query rows processed per camera.
        query_index: Gather indices for query rows.
        reference_points_batched: Rebatched camera reference points with shape
            ``[batch_size * num_cams, rebatch_len, num_depth_levels, 2]``.
        row_mask: Per-camera-row validity with shape ``[batch_size * num_cams, rebatch_len, 1]``.
        inverse_index: One gather index per contribution slot, each selecting the camera row that
            feeds a query. Slots a query does not fill point at a padded row.
        count: Camera-contributor count with shape ``[batch_size, num_queries, 1]``.
        num_valid_queries: ``(batch_size * num_cams,)`` uint32 device tensor, the valid row count of
            each camera row block in ``row_mask`` order. MSDA computes only these rows, reading the
            counts on device; the padding past them is masked by ``row_mask``.
        is_empty: Whether the visibility mask contains no valid query-camera pairs.
    """

    rebatch_len: int
    query_index: Optional[ttnn.Tensor]
    reference_points_batched: Optional[ttnn.Tensor]
    row_mask: Optional[ttnn.Tensor]
    inverse_index: tuple[ttnn.Tensor, ...]
    count: Optional[ttnn.Tensor]
    num_valid_queries: Optional[ttnn.Tensor] = None
    is_empty: bool = False


class SCAVisibility(NamedTuple):
    """One frame's camera visibility on device, and once read, its counts on their way to the host.

    Attributes:
        seen_by: ``[num_cams, batch_size, num_queries, 1]``, 1 where the camera sees the query.
        visible: Float32 ``[1, 1, num_cams, num_queries]``, the first sample's ``seen_by``.
        visible_count: Float32 ``[1, 1, num_cams, 1]``, the queries each camera sees.
        max_cameras: Float32 scalar, the most cameras seeing one query.
        host_counts: ``visible_count`` and ``max_cameras`` read back without blocking, valid once
            ``counts_ready`` completes; None until :meth:`SCARebatchPlanner.read_counts`.
        counts_ready: Event recorded after the readback.
    """

    seen_by: ttnn.Tensor
    visible: ttnn.Tensor
    visible_count: ttnn.Tensor
    max_cameras: ttnn.Tensor
    host_counts: Optional[tuple] = None
    counts_ready: object = None


class SCARebatchPlanner:
    """Builds each frame's :class:`SCARebatchPlan` on device, at a capacity that only grows.

    A frame takes four steps, so a traced caller can keep the host's readback and sizing out of its
    traces:
    :meth:`count_visible` and :meth:`build` only queue device work; :meth:`read_counts` queues the
    counts' readback and :meth:`size` waits for it. :meth:`visibility` and :meth:`__call__` chain
    them for an eager caller. While :attr:`generation` holds, :meth:`build` queues the same ops on
    the same constant tensors and on tensors of the same shapes.

    The visibility mask stays on device: a cumulative sum places each camera's visible queries in
    its row block and a scatter compacts their IDs there. The host reads back only the per-camera
    visible counts and the most cameras any query is seen by, and only to size the plan. MSDA takes its
    exact row counts from a device tensor, so a captured trace follows them from frame to frame.
    Capacity is the largest count seen so far times ``headroom``, rounded up, so a frame that sees a
    few more queries than the last keeps the plan's shapes.

    Attributes:
        capacity: Visible queries per camera the plan holds; each camera's block has
            ``_rebatch_len(capacity)`` rows.
        num_slots: Contribution slots, the most cameras seen to share one query so far. It only grows,
            like capacity, so the plan's number of tensors and ops stays fixed as well.
        reallocations: How many times a frame outgrew capacity or slots after the first frame; each
            changes the plan's shapes or its number of ops.
        generation: Changes whenever :meth:`build` would record different ops, shapes or constant
            tensors: on every growth and every rebuild of the constants. A trace of :meth:`build` is
            valid while it holds.
    """

    def __init__(self, device, headroom: float = REBATCH_HEADROOM):
        assert headroom >= 1.0, f"headroom {headroom} would allocate less than the observed count"
        self.device = device
        self.headroom = headroom
        self.capacity = 0
        self.num_slots = 0
        self.reallocations = 0
        self.generation = 0
        self._constants_key = None
        self._constants = None

    def reserve(self, capacity: int, num_slots: int):
        """Grow, as a frame needing them would, to at least ``capacity`` visible queries per camera and
        ``num_slots`` slots; a frame of the plan's shapes must have been sized. A caller that knows
        the counts it will see sizes the plan once instead of growing it frame by frame."""
        assert self._constants_key is not None, "size() a frame before reserve()"
        self._resize(max(self.capacity, capacity), max(self.num_slots, num_slots))
        num_cams, bs, num_queries, _ = self._constants_key
        self._constants_for(num_cams, bs, num_queries, _rebatch_len(self.capacity))

    def _grow(self, max_len: int, max_cameras: int):
        if self.capacity and (max_len > self.capacity or max_cameras > self.num_slots):
            logger.info(
                f"SCA rebatch plan grows: {max_len} visible queries for capacity {self.capacity}, "
                f"{max_cameras} cameras sharing a query for {self.num_slots} slots"
            )
        capacity = math.ceil(max_len * self.headroom) if max_len > self.capacity else self.capacity
        self._resize(capacity, max(self.num_slots, max_cameras))

    def _resize(self, capacity: int, num_slots: int):
        if (capacity, num_slots) == (self.capacity, self.num_slots):
            return
        if self.capacity:
            self.reallocations += 1
        self.generation += 1
        self.capacity, self.num_slots = capacity, num_slots

    def _constants_for(self, num_cams: int, bs: int, num_queries: int, rebatch_len: int):
        """Device constants of one plan shape, rebuilt only when the shape changes."""
        key = (num_cams, bs, num_queries, rebatch_len)
        if key != self._constants_key:

            def upload(tensor, dtype, layout):
                return ttnn.from_torch(tensor.contiguous(), device=self.device, dtype=dtype, layout=layout)

            cams = torch.arange(num_cams, dtype=torch.float32).view(1, 1, num_cams, 1)
            samples = torch.arange(bs, dtype=torch.float32).view(bs, 1, 1, 1)
            self._constants = dict(
                # Scatter source and destination: query IDs, and padding that gathers the last query.
                query_ids=upload(
                    torch.arange(num_queries, dtype=torch.int32).expand(1, 1, num_cams, num_queries),
                    ttnn.int32,
                    ttnn.ROW_MAJOR_LAYOUT,
                ),
                padding_ids=upload(
                    torch.full((1, 1, num_cams, rebatch_len), num_queries - 1, dtype=torch.int32),
                    ttnn.int32,
                    ttnn.ROW_MAJOR_LAYOUT,
                ),
                # Row offsets: of a sample's queries, of a (camera, sample)'s reference points, of a
                # camera's block of contributions and of a sample's blocks.
                query_offset=upload(samples * num_queries, ttnn.float32, ttnn.TILE_LAYOUT),
                point_offset=upload((cams * bs + samples) * num_queries, ttnn.float32, ttnn.TILE_LAYOUT),
                camera_block=upload(cams * rebatch_len, ttnn.float32, ttnn.TILE_LAYOUT),
                sample_block=upload(samples * num_cams * rebatch_len, ttnn.float32, ttnn.TILE_LAYOUT),
                rows=upload(
                    torch.arange(rebatch_len, dtype=torch.float32).view(1, rebatch_len, 1),
                    ttnn.float32,
                    ttnn.TILE_LAYOUT,
                ),
            )
            self._constants_key = key
            self.generation += 1
        return self._constants

    def visibility(self, bev_mask) -> "SCAVisibility":
        """:meth:`count_visible` and :meth:`read_counts`.

        Nothing here waits for the device, so the work queued after it, such as the backbone, runs
        while the counts come back.
        """
        return self.read_counts(self.count_visible(bev_mask))

    def read_counts(self, visibility: "SCAVisibility") -> "SCAVisibility":
        """``visibility`` with its counts' readback queued, without waiting for it."""
        return visibility._replace(
            host_counts=(visibility.visible_count.cpu(blocking=False), visibility.max_cameras.cpu(blocking=False)),
            counts_ready=ttnn.record_event(self.device, 0),
        )

    def count_visible(self, bev_mask) -> "SCAVisibility":
        """Count the cameras' visible queries, on device only.

        Args:
            bev_mask: Bfloat16 device tensor with shape
                ``[num_cams, batch_size, num_queries, num_depth_levels]``.
        """
        num_cams, _, num_queries, _ = bev_mask.shape
        # [num_cams, bs, num_queries, 1]: whether any depth level of the query lands in the camera.
        mask = ttnn.to_layout(bev_mask, ttnn.TILE_LAYOUT)
        seen_by = ttnn.gt(ttnn.sum(mask, dim=-1, keepdim=True), 0.0)
        # As upstream, every sample gathers the queries the first sample's cameras see, and divides by
        # the cameras its own mask counts. Float32 so positions are exact, and cameras on rows so a
        # tile holds 32 of them rather than one.
        visible = ttnn.transpose(ttnn.typecast(seen_by[:, :1], ttnn.float32), -2, -1)
        visible = ttnn.reshape(visible, (1, 1, num_cams, num_queries))
        visible_count = ttnn.sum(visible, dim=-1, keepdim=True)
        max_cameras = ttnn.max(ttnn.sum(visible, dim=-2, keepdim=True))
        return SCAVisibility(seen_by=seen_by, visible=visible, visible_count=visible_count, max_cameras=max_cameras)

    def __call__(self, reference_points_cam, visibility: "SCAVisibility") -> SCARebatchPlan:
        """Build a rebatch plan for one frame's camera projections: :meth:`size`, then :meth:`build`.

        Args:
            reference_points_cam: Bfloat16 device tensor with shape
                ``[num_cams, batch_size, num_queries, num_depth_levels, 2]``.
            visibility: The frame's :meth:`visibility`; waits only for its counts.

        Returns:
            A plan owned by these frame inputs. If no query-camera pair is valid, ``is_empty`` is true,
            ``rebatch_len`` is zero, and the plan carries no tensors.
        """
        if not self.size(visibility):
            return SCARebatchPlan(
                rebatch_len=0,
                query_index=None,
                reference_points_batched=None,
                row_mask=None,
                inverse_index=(),
                count=None,
                is_empty=True,
            )
        return self.build(reference_points_cam, visibility)

    def size(self, visibility: "SCAVisibility") -> bool:
        """Wait for :meth:`read_counts`' counts and grow the plan's capacity and slots to fit them.

        This is the plan's only readback. Growing uploads the constants of the new shapes here, so
        :meth:`build` queues device work only. Returns whether any camera sees a query.
        """
        # The only readback: these size the plan's tensors. MSDA reads its row counts on device.
        assert visibility.counts_ready is not None, "read_counts() before size()"
        ttnn.event_synchronize(visibility.counts_ready)
        host_count, host_max_cameras = visibility.host_counts
        num_visible = ttnn.to_torch(host_count).flatten().round().to(torch.int64)
        max_len = int(num_visible.max())
        if ENABLE_LOGGING:
            logger.info(f"SCA Valid Queries: {num_visible.tolist()}")
        if max_len == 0:
            return False
        self._grow(max_len, int(ttnn.to_torch(host_max_cameras).round().item()))
        num_cams, bs, num_queries, _ = tuple(visibility.seen_by.shape)
        rebatch_len = _rebatch_len(self.capacity)
        if max(num_cams * bs * num_queries, bs * num_cams * rebatch_len) > 2**24:
            raise ValueError("SCA rebatch row IDs are computed in float32, exact only up to 2**24")
        self._constants_for(num_cams, bs, num_queries, rebatch_len)
        return True

    def build(self, reference_points_cam, visibility: "SCAVisibility") -> SCARebatchPlan:
        """The plan at the current capacity and slots, as device work only.

        :meth:`size` must have seen this frame's counts: a camera seeing more queries than the
        capacity would scatter its compacted IDs past its row block.

        Args:
            reference_points_cam: Bfloat16 device tensor with shape
                ``[num_cams, batch_size, num_queries, num_depth_levels, 2]``.
            visibility: The frame's :meth:`count_visible`.
        """
        num_cams, bs, num_queries, num_depth_levels = tuple(reference_points_cam.shape)[:4]
        seen_by, visible, visible_count = visibility.seen_by, visibility.visible, visibility.visible_count
        rebatch_len = _rebatch_len(self.capacity)
        padding_row = rebatch_len - 1
        assert self.capacity > 0 and self._constants_key == (
            num_cams,
            bs,
            num_queries,
            rebatch_len,
        ), "size() a frame of these shapes before build()"
        constants = self._constants

        # Compact each camera to the queries it sees, so MSDA runs on its valid rows, not all of
        # them. A visible query's row in its camera's block is the number of visible queries before
        # it; the rest all land on the block's last row, which is always padding.
        position = ttnn.sub(ttnn.cumsum(visible, dim=-1), 1.0)
        destination = ttnn.where(visible, position, float(padding_row))
        destination = ttnn.to_layout(ttnn.typecast(destination, ttnn.int32), ttnn.ROW_MAJOR_LAYOUT)
        compact_ids = ttnn.scatter(constants["padding_ids"], -1, destination, constants["query_ids"])
        compact_ids = ttnn.typecast(ttnn.to_layout(compact_ids, ttnn.TILE_LAYOUT), ttnn.float32)

        # Adding a [bs, 1, 1, 1] or [bs, 1, num_cams, 1] offset lays the rows out in (sample, camera,
        # row) order, the order of the rebatched blocks.
        query_index = _row_index(ttnn.add(compact_ids, constants["query_offset"]))

        reference_points_cam = ttnn.clamp(reference_points_cam, -10.0, 10.0)
        assert (
            reference_points_cam.dtype == ttnn.bfloat16
        ), f"SCA rebatch gathers reference_points_cam, so it must be bfloat16, got {reference_points_cam.dtype}."
        # Reference points are camera-major, so their row IDs include camera and batch offsets.
        ref_rows = ttnn.reshape(
            ttnn.to_layout(reference_points_cam, ttnn.ROW_MAJOR_LAYOUT),
            (1, 1, num_cams * bs * num_queries, num_depth_levels * 2),
        )
        ref_index = _row_index(ttnn.add(compact_ids, constants["point_offset"]))
        # Split the gathered row back into MSDA's [.., rebatch_len, num_depth_levels, 2], the last
        # dim being each depth level's (x, y) camera coordinate. Changing the row width from
        # num_depth_levels * 2 to 2 requires a data-moving reshape before conversion to TILE layout.
        # TODO: profile whether tilizing first is cheaper.
        reference_points_batched = ttnn.to_layout(
            ttnn.reshape(
                ttnn.embedding(ref_index, ref_rows),
                (bs * num_cams, rebatch_len, num_depth_levels, 2),
            ),
            ttnn.TILE_LAYOUT,
        )

        # Invert the compaction: for every query, the rows of queries_output that contribute to it.
        # Row (j, i, p) of the compacted output is flat row (j * num_cams + i) * rebatch_len + p. A
        # query's slot in camera i is the number of cameras before i that see it, so each slot takes
        # at most one camera's row; unfilled slots read the last row of the sample's first camera,
        # which is always padding. Rows are offset by -padding_row so that the sum over cameras plus
        # padding_row is the filling camera's row, or padding_row where no camera fills the slot.
        slot = ttnn.sub(ttnn.cumsum(visible, dim=-2), visible)
        contribution = ttnn.sub(ttnn.add(position, constants["camera_block"]), float(padding_row))
        inverse_index = []
        for s in range(self.num_slots):
            fills = ttnn.logical_and(visible, ttnn.eq(slot, float(s)))
            rows = ttnn.sum(ttnn.where(fills, contribution, 0.0), dim=-2, keepdim=True)
            rows = ttnn.add(ttnn.add(rows, float(padding_row)), constants["sample_block"])
            inverse_index.append(_row_index(rows))

        num_valid = ttnn.reshape(visible_count, (1, num_cams, 1, 1))
        if bs > 1:
            num_valid = ttnn.repeat(num_valid, ttnn.Shape((bs, 1, 1, 1)))
        row_mask = ttnn.lt(constants["rows"], ttnn.reshape(num_valid, (bs * num_cams, 1, 1)))
        row_mask = ttnn.typecast(row_mask, ttnn.bfloat16)
        # Sums of 0/1 in float32, so exact; MSDA reads them as one uint32 row-major page.
        num_valid_queries = ttnn.typecast(ttnn.reshape(num_valid, (1, bs * num_cams)), ttnn.uint32)
        num_valid_queries = ttnn.reshape(ttnn.to_layout(num_valid_queries, ttnn.ROW_MAJOR_LAYOUT), (bs * num_cams,))

        count = ttnn.clamp(ttnn.sum(seen_by, dim=0, keepdim=True), min=1.0)
        count = ttnn.reshape(count, (bs, num_queries, 1))

        return SCARebatchPlan(
            rebatch_len=rebatch_len,
            query_index=query_index,
            reference_points_batched=reference_points_batched,
            row_mask=row_mask,
            inverse_index=tuple(inverse_index),
            count=count,
            num_valid_queries=num_valid_queries,
        )


def build_rebatch_plan(reference_points_cam, bev_mask, device) -> SCARebatchPlan:
    """One frame's plan at exactly its own capacity; see :class:`SCARebatchPlanner`."""
    planner = SCARebatchPlanner(device, headroom=1.0)
    return planner(reference_points_cam, planner.visibility(bev_mask))


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
        **kwargs,
    ):
        self.device = device
        self.params = params
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.batch_first = batch_first

        if deformable_attention is None:
            deformable_attention = dict(embed_dims=embed_dims, num_levels=4, num_points=4, num_heads=8)

        deform_config = DeformableAttentionConfig(
            embed_dims=deformable_attention.get("embed_dims", embed_dims),
            num_heads=deformable_attention.get("num_heads", 8),
            num_levels=deformable_attention.get("num_levels", 4),
            num_points=deformable_attention.get("num_points", 4),
            batch_first=batch_first,
        )

        # ``params.deformable_attention``, not ``params``: MSDA applies its own
        # ``output_proj``, while the one in the SCA namespace belongs to the slots
        # projection in forward. Sharing a namespace applies the SCA's matrix twice.
        self.deformable_attention = TTMSDeformableAttention(
            deform_config, device, params.deformable_attention, spatial_shapes=spatial_shapes, residual=False
        )

    def forward(
        self,
        *,
        query,
        reference_points_cam,
        bev_mask,
        value,
        residual=None,
        query_pos=None,
        key_padding_mask=None,
        rebatch_plan=None,
        **kwargs,
    ):
        """
        Forward pass of TTNN Spatial Cross Attention.

        Args:
            query: Bfloat16 BEV queries [B, num_queries, embed_dims].
            reference_points_cam: Bfloat16 camera projected reference points [num_cams, B, num_queries, D, 2].
            bev_mask: Valid mask for camera projections [num_cams, B, num_queries, D].
            value: Multi-camera features [B, num_cams, H*W, embed_dims], batch-first as the
                perception transformer lays them out. Attention here takes no key: deformable
                attention never scores query against one. Two Linears on query predict where to
                sample and with what weight, replacing Q*K^T, so ``value`` is the only feature
                tensor the device path needs.
            residual: Residual connection input.
            query_pos: Query positional encoding.
            key_padding_mask: Key padding mask.
            rebatch_plan: Prebuilt :class:`SCARebatchPlan`. Shared by every encoder layer; built here if absent.
            **kwargs: Additional arguments.

        Returns:
            Output features [B, num_queries, embed_dims].
        """
        signpost(header="TTNN SCA Forward Start")

        assert value is not None, "value is required"

        # Handle input defaults
        if residual is None:
            inp_residual = ttnn.clone(query)
        else:
            inp_residual = residual

        # Add query positional encoding
        if query_pos is not None:
            query = ttnn.add(query, query_pos)

        if ENABLE_LOGGING:
            logger.info("SCA Tensor Conversion Complete")

        bs, num_queries, _ = query.shape
        # Extract number of depth levels for 3D point sampling
        # Each BEV query samples points at multiple Z-coordinates (depth levels) in 3D space
        num_depth_levels = reference_points_cam.shape[3]

        # Validate sampling points divisibility to prevent runtime errors in deformable attention
        assert self.deformable_attention.num_points % num_depth_levels == 0, (
            f"num_points ({self.deformable_attention.num_points}) must be divisible by depth levels ({num_depth_levels}). "
            f"This is required for proper reshaping in deformable attention. Consider adjusting num_points in config."
        )

        # Every encoder layer in a forward shares one plan; building it here is the standalone path.
        if rebatch_plan is None:
            rebatch_plan = build_rebatch_plan(reference_points_cam, bev_mask, self.device)
        else:
            assert bev_mask.shape[0] == self.num_cams, (
                f"rebatch_plan built for num_cams={bev_mask.shape[0]} but this SCA is configured with "
                f"num_cams={self.num_cams}"
            )

        if rebatch_plan.is_empty:
            if ENABLE_LOGGING:
                logger.warning("No valid points found in SCA, returning residual")
            # As the reference: the output projection of all-zero slots, which is its bias. A traced
            # frame runs the full plan, all padding, and gets the same.
            output_proj = getattr(self.params, "output_proj", None)
            if output_proj is None or output_proj.bias is None:
                return inp_residual
            return ttnn.add(inp_residual, output_proj.bias)

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

        # A camera-first tensor has the same volume, so the reshape folds it into the wrong
        # rows instead of failing. We need to prevent this by asserting the shape is correct.
        assert value.shape[0] == bs and value.shape[1] == self.num_cams, (
            f"value {list(value.shape)} is not batch-first " f"[{bs}, {self.num_cams}, L, {self.embed_dims}]"
        )
        L = value.shape[2]
        value_reshaped = ttnn.reshape(value, (bs * self.num_cams, L, self.embed_dims))

        if ENABLE_LOGGING:
            logger.info("SCA Calling Deformable Attention")
        queries_output = self.deformable_attention(
            query=queries_batched,
            value=value_reshaped,
            reference_points=reference_points_batched,
            num_valid_queries_tensor=rebatch_plan.num_valid_queries,
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

        # Accumulate camera features by query ID. The plan inverts the compaction, so this is a
        # gather per contribution slot rather than a scatter: ttnn.scatter only writes along the
        # last dimension and transposes all three operands to get there.
        #
        # MSDA leaves the padded rows unwritten, so they can hold any bit pattern, NaN included.
        # Selecting zero makes them neutral; multiplying by row_mask would not.
        # Each contribution is already BF16. Adding those tensors rounds after every camera, so a
        # 1 sitting next to 256 disappears before the cancelling term arrives (256 + 1 + -256
        # comes back as 2). The running sum stays in FP32 and is rounded once, after the last camera.
        contribution_rows = ttnn.reshape(
            ttnn.to_layout(ttnn.where(rebatch_plan.row_mask, queries_output, 0.0), ttnn.ROW_MAJOR_LAYOUT),
            (1, 1, bs * self.num_cams * rebatch_len, self.embed_dims),
        )
        slots = None
        for index in rebatch_plan.inverse_index:
            gathered = ttnn.typecast(
                ttnn.to_layout(
                    ttnn.reshape(ttnn.embedding(index, contribution_rows), (bs, num_queries, self.embed_dims)),
                    ttnn.TILE_LAYOUT,
                ),
                ttnn.float32,
            )
            slots = gathered if slots is None else ttnn.add(slots, gathered)
        slots = ttnn.typecast(slots, ttnn.bfloat16)

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

        signpost(header="TTNN SCA Forward End")

        return output

    def __call__(self, *args, **kwargs):
        """Make the class callable"""
        return self.forward(*args, **kwargs)

    def extra_repr(self) -> str:
        """String representation for debugging"""
        return f"embed_dims={self.embed_dims}, num_cams={self.num_cams}, " f"batch_first={self.batch_first}"
