# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the BEV half of BEVFormer's ``PerceptionTransformer``
(``reference/perception_transformer.py``): FPN levels and BEV queries in, the frame's BEV out.

What changes from frame to frame and comes from the host, the CAN bus, the ego shift, the
previous BEV's rotation and the cameras' projections, is prepared by
:meth:`TtPerceptionTransformer.prepare_frame` into a :class:`PerceptionFrame` whose device
buffers are allocated once and refilled in place, so the forward runs on device only and a trace
captured with a frame replays on the next one.

The previous BEV's rotation is nearest-neighbour, so it moves whole cells: the host rotates an
image of cell indices with the reference's own ``rotate``, and the device gathers the previous
BEV's rows by them, which picks exactly the cells the reference picks. Cells rotated in from
outside the grid are zeroed by a mask.
"""

from dataclasses import dataclass

import torch
import ttnn
from torchvision.transforms.functional import rotate

from models.experimental.bevformer.model_config import CAN_BUS_DIMS, GRID_DTYPE
from models.experimental.bevformer.reference.perception_transformer import bev_grid_length, ego_shift
from models.experimental.bevformer.tt.tt_common import layer_norm
from models.experimental.bevformer.tt.tt_encoder import TTBEVFormerEncoder
from models.experimental.bevformer.tt.tt_spatial_cross_attention import SCARebatchPlan


@dataclass
class PerceptionFrame:
    """One frame's host-derived inputs, on device. Refilling it keeps every buffer's shape and
    address.

    Attributes:
        plan: The encoder's spatial cross-attention rebatch plan.
        can_bus: ``(bs, 1, CAN_BUS_DIMS)`` bfloat16, the relative CAN bus.
        shift: ``(bs, 1, 1, 2)`` float32 ROW_MAJOR, the ego translation in BEV-grid fractions.
        rotation_index: ``(1, 1, 1, bs * num_query)`` uint32 ROW_MAJOR, the stacked previous-BEV row each
            cell takes.
        rotation_mask: ``(bs, num_query, 1)`` bfloat16, zero where the cell rotated in from
            outside the grid.
    """

    plan: SCARebatchPlan
    can_bus: ttnn.Tensor
    shift: ttnn.Tensor
    rotation_index: ttnn.Tensor
    rotation_mask: ttnn.Tensor


def rotation_gather(img_metas, bev_h, bev_w, rotate_center):
    """Host ``(index, mask)`` for :class:`PerceptionFrame`: sample ``b``'s cell ``q`` takes stacked
    row ``index[..., b * num_query + q]`` of the previous BEV, or zero where ``mask`` is zero."""
    num_query = bev_h * bev_w
    # One-based so the zero ``rotate`` fills in marks outside; float32 holds these indices exactly.
    cells = torch.arange(1, num_query + 1, dtype=torch.float32).view(1, bev_h, bev_w)
    indices, masks = [], []
    for b, meta in enumerate(img_metas):
        source = rotate(cells, float(meta["can_bus"][-1]), center=list(rotate_center)).flatten().long()
        indices.append((source - 1).clamp(min=0) + b * num_query)
        masks.append((source > 0).float())
    return torch.cat(indices).view(1, 1, 1, -1), torch.stack(masks).unsqueeze(-1)


class TtPerceptionTransformer:
    """The encoder and its glue over a ``(bev_h, bev_w)`` grid and FPN levels of ``spatial_shapes``.

    Owns a :class:`TTBEVFormerEncoder`, so ``params`` are single-use like its own.
    """

    def __init__(self, params, device, *, bev_h, bev_w, spatial_shapes):
        """``params`` from ``model_preprocessing.create_perception_transformer_parameters``;
        ``spatial_shapes`` the FPN levels as (h, w)."""
        self.params = params
        self.device = device
        self.bev_h, self.bev_w = bev_h, bev_w
        self.embed_dims = params.config.embed_dims
        self.num_cams = params.config.num_cams
        self.rotate_center = params.config.rotate_center
        self.spatial_shapes = [tuple(int(v) for v in shape) for shape in spatial_shapes]
        self.encoder = TTBEVFormerEncoder(
            params.encoder, device, bev_h=bev_h, bev_w=bev_w, spatial_shapes=self.spatial_shapes
        )

    def _host_frame(self, img_metas):
        can_bus = torch.tensor([[list(map(float, meta["can_bus"]))] for meta in img_metas])
        assert can_bus.shape[-1] == CAN_BUS_DIMS, f"can_bus has {can_bus.shape[-1]} values, expected {CAN_BUS_DIMS}"
        grid_length = bev_grid_length(self.encoder.pc_range, self.bev_h, self.bev_w)
        shift = ego_shift(img_metas, self.bev_h, self.bev_w, grid_length).view(len(img_metas), 1, 1, 2)
        rotation_index, rotation_mask = rotation_gather(img_metas, self.bev_h, self.bev_w, self.rotate_center)
        return dict(
            can_bus=(can_bus, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            shift=(shift, GRID_DTYPE, ttnn.ROW_MAJOR_LAYOUT),
            rotation_index=(rotation_index, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            rotation_mask=(rotation_mask, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        )

    def prepare_frame(self, img_metas, frame=None, capacity=None):
        """This frame's :class:`PerceptionFrame` from ``img_metas`` (per sample ``lidar2img``,
        ``img_shape`` and the relative ``can_bus``). Without ``frame``, a new one, whose rebatch
        plan holds ``capacity`` rows per camera (see ``TTBEVFormerEncoder.prepare_frame``); with
        ``frame``, refilled in place for a frame of the same batch size."""
        host = self._host_frame(img_metas)
        if frame is None:
            return PerceptionFrame(
                plan=self.encoder.prepare_frame(img_metas, capacity=capacity),
                **{
                    name: ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=self.device)
                    for name, (tensor, dtype, layout) in host.items()
                },
            )
        self.encoder.prepare_frame(img_metas, plan=frame.plan)
        for name, (tensor, dtype, layout) in host.items():
            buffer = getattr(frame, name)
            assert tuple(buffer.shape) == tuple(
                tensor.shape
            ), f"{name}: the frame holds {tuple(buffer.shape)}, not {tuple(tensor.shape)}; prepare a new frame"
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(tensor, dtype=dtype, layout=layout), buffer)
        return frame

    def camera_features(self, mlvl_feats, bs):
        """``mlvl_feats`` per level ``(1, 1, bs * num_cams * h * w, C)``, as the FPN emits them ->
        ``(bs * num_cams, num_keys, C)`` with the camera and level embeddings added."""
        levels = []
        for (h, w), feat, embeds in zip(self.spatial_shapes, mlvl_feats, self.params.level_cams_embeds):
            # The FPN's levels may be L1-sharded, and the first keeps the backbone's dtype; the
            # per-camera split is a reshape of an interleaved bfloat16 tensor, free in ROW_MAJOR.
            feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)
            if feat.dtype != ttnn.bfloat16:
                feat = ttnn.typecast(feat, ttnn.bfloat16)
            feat = ttnn.reshape(
                ttnn.to_layout(feat, ttnn.ROW_MAJOR_LAYOUT), (bs, self.num_cams, h * w, self.embed_dims)
            )
            feat = ttnn.add(ttnn.to_layout(feat, ttnn.TILE_LAYOUT), embeds)
            levels.append(ttnn.reshape(feat, (bs * self.num_cams, h * w, self.embed_dims)))
        return ttnn.concat(levels, dim=1)

    def rotate_prev_bev(self, prev_bev, frame):
        """``prev_bev`` ``(bs, num_query, C)`` rotated per sample by the frame's heading change."""
        bs, num_query, embed_dims = prev_bev.shape
        rows = ttnn.reshape(ttnn.to_layout(prev_bev, ttnn.ROW_MAJOR_LAYOUT), (1, 1, bs * num_query, embed_dims))
        rotated = ttnn.embedding(frame.rotation_index, rows, layout=ttnn.TILE_LAYOUT)
        return ttnn.mul(ttnn.reshape(rotated, (bs, num_query, embed_dims)), frame.rotation_mask)

    def __call__(self, mlvl_feats, bev_queries, bev_pos, frame, prev_bev=None):
        """``mlvl_feats`` the FPN levels (see :meth:`camera_features`); ``bev_queries``
        ``(1, num_query, C)``, shared by the batch; ``bev_pos`` ``(bs, num_query, C)``; ``frame`` from
        :meth:`prepare_frame`; ``prev_bev`` the previous frame's ``(bs, num_query, C)`` BEV or None.
        Returns this frame's BEV, ``(bs, num_query, C)``."""
        bs = frame.plan.batch_size
        mlp = self.params.can_bus_mlp
        can_bus = ttnn.linear(frame.can_bus, mlp.linear1.weight, bias=mlp.linear1.bias, activation="relu")
        can_bus = ttnn.linear(can_bus, mlp.linear2.weight, bias=mlp.linear2.bias, activation="relu")
        can_bus = layer_norm(can_bus, mlp.norm)
        bev_queries = ttnn.add(bev_queries, can_bus)
        value = self.camera_features(mlvl_feats, bs)
        if prev_bev is not None:
            prev_bev = self.rotate_prev_bev(prev_bev, frame)
        return self.encoder(bev_queries, value, bev_pos, frame.plan, prev_bev=prev_bev, shift=frame.shift)
