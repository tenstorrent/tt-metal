# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The spatial cross-attention's per-frame rebatch plan, built on device at a capacity that only
grows, as BEVFormer's ``SCARebatchPlanner`` builds it."""

import math
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
from loguru import logger

import ttnn

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
        reference_points_batched: Row-major rebatched camera reference points with shape
            ``[batch_size * num_cams, rebatch_len, num_depth_levels, 2]``.
        inverse_index: One gather index per contribution slot, each selecting the camera row that
            feeds a query. Slots a query does not fill point at a camera's last row, which is always
            padding.
        count: Camera-contributor count with shape ``[batch_size, num_queries, 1]``.
        num_valid_queries: ``(batch_size * num_cams,)`` uint32 device tensor, the valid row count of
            each camera row block, batch-major. MSDA computes only these rows, reading the counts on
            device, and leaves the padding past them unwritten.
        is_empty: Whether the visibility mask contains no valid query-camera pairs.
    """

    rebatch_len: int
    query_index: Optional[ttnn.Tensor]
    reference_points_batched: Optional[ttnn.Tensor]
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
            queries = torch.arange(num_queries, dtype=torch.float32).view(1, 1, 1, num_queries)
            padding_of_query = (queries % num_cams) * rebatch_len + rebatch_len - 1
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
                # The padding row an unfilled slot of query q reads, camera q % num_cams's last, so
                # those gathers spread over num_cams pages instead of all landing on one: within the
                # sample's blocks, and as the sample's row.
                padding_of_query=upload(padding_of_query, ttnn.float32, ttnn.TILE_LAYOUT),
                unfilled_row=upload(
                    samples * num_cams * rebatch_len + padding_of_query, ttnn.float32, ttnn.TILE_LAYOUT
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
        # dim being each depth level's (x, y) camera coordinate. It stays row-major, the layout fused
        # MSDA reads: in TILE the [num_depth_levels, 2] tail pads to a full tile per query, and every
        # layer sharing this plan would untilize it again.
        reference_points_batched = ttnn.reshape(
            ttnn.embedding(ref_index, ref_rows),
            (bs * num_cams, rebatch_len, num_depth_levels, 2),
        )

        # Invert the compaction: for every query, the rows of queries_output that contribute to it.
        # Row (j, i, p) of the compacted output is flat row (j * num_cams + i) * rebatch_len + p. A
        # query's slot in camera i is the number of cameras before i that see it, so each slot takes
        # at most one camera's row; an unfilled slot reads a camera's last row, which is always
        # padding and which the attention zeroes (constants["unfilled_row"]). Rows are offset by
        # the query's padding row so that the sum over cameras plus unfilled_row is the filling
        # camera's row, or the padding row where no camera fills the slot.
        slot = ttnn.sub(ttnn.cumsum(visible, dim=-2), visible)
        contribution = ttnn.sub(ttnn.add(position, constants["camera_block"]), constants["padding_of_query"])
        inverse_index = []
        for s in range(self.num_slots):
            fills = ttnn.logical_and(visible, ttnn.eq(slot, float(s)))
            rows = ttnn.sum(ttnn.where(fills, contribution, 0.0), dim=-2, keepdim=True)
            inverse_index.append(_row_index(ttnn.add(rows, constants["unfilled_row"])))

        num_valid = ttnn.reshape(visible_count, (1, num_cams, 1, 1))
        if bs > 1:
            num_valid = ttnn.repeat(num_valid, ttnn.Shape((bs, 1, 1, 1)))
        # Sums of 0/1 in float32, so exact; MSDA reads them as one uint32 row-major page.
        num_valid_queries = ttnn.typecast(ttnn.reshape(num_valid, (1, bs * num_cams)), ttnn.uint32)
        num_valid_queries = ttnn.reshape(ttnn.to_layout(num_valid_queries, ttnn.ROW_MAJOR_LAYOUT), (bs * num_cams,))

        count = ttnn.clamp(ttnn.sum(seen_by, dim=0, keepdim=True), min=1.0)
        count = ttnn.reshape(count, (bs, num_queries, 1))

        return SCARebatchPlan(
            rebatch_len=rebatch_len,
            query_index=query_index,
            reference_points_batched=reference_points_batched,
            inverse_index=tuple(inverse_index),
            count=count,
            num_valid_queries=num_valid_queries,
        )
