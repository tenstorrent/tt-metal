# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of BEVFormer's encoder (``reference/encoder.py``).

The layers run batch-first, ``(bs, num_query, embed_dims)``; the camera features stay as the
reference takes them, ``(num_cams, num_keys, bs, embed_dims)``. Parameters come from
``model_preprocessing.create_bevformer_encoder_parameters``.

Camera geometry depends on the cameras only, not on the activations: once per frame,
:meth:`TTBEVFormerEncoder.prepare_frame` projects the pillar points into the cameras in float32 on
the host, as upstream's ``point_sampling`` does, and builds the spatial cross-attention's rebatch
plan, whose size the visibility mask decides. In bfloat16 the projected points are off by up to
a quarter of the image. The forward then runs on device only, the cells' reference points and
the ego shift included.
"""

from types import SimpleNamespace

import torch
import ttnn

from ..reference.encoder import PC_RANGE, BEVFormerEncoder
from ..reference.point_sampling_3d_2d import generate_reference_points, point_sampling_3d_to_2d
from .tt_common import layer_norm
from .tt_ms_deformable_attention import fp32_grid_sample_config
from .tt_spatial_cross_attention import TTSpatialCrossAttention, build_rebatch_plan
from .tt_temporal_self_attention import TTTemporalSelfAttention, tsa_grid_bias

# Sampling grids in float32: in bfloat16 a point in (0.5, 1) moves in steps of 2^-8, 0.8 px on a
# 200-wide BEV map.
GRID_DTYPE = ttnn.float32


class TTBEVFormerLayer:
    def __init__(
        self,
        params,
        device,
        *,
        bev_shape,
        spatial_shapes,
        embed_dims,
        num_heads,
        num_levels,
        num_points,
        tsa_num_points,
        num_cams,
        grid_sample_compute_config,
    ):
        self.params = params
        self.temporal_self_attention = TTTemporalSelfAttention(
            params.tsa,
            device,
            bev_shape=bev_shape,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_points=tsa_num_points,
            grid_dtype=GRID_DTYPE,
            grid_sample_compute_config=grid_sample_compute_config,
        )
        self.spatial_cross_attention = TTSpatialCrossAttention(
            device,
            params.sca,
            embed_dims=embed_dims,
            num_cams=num_cams,
            deformable_attention=dict(
                embed_dims=embed_dims, num_heads=num_heads, num_levels=num_levels, num_points=num_points
            ),
            spatial_shapes=spatial_shapes,
            grid_dtype=GRID_DTYPE,
            grid_sample_compute_config=grid_sample_compute_config,
        )

    def __call__(self, query, value, bev_pos, prev_bev, tsa_grid_bias, frame):
        """The positional encoding enters the self-attention only, as upstream."""
        query = self.temporal_self_attention(query, prev_bev, bev_pos, tsa_grid_bias)
        query = layer_norm(query, self.params.norms[0])
        query = self.spatial_cross_attention(query=query, value=value, rebatch_plan=frame.rebatch_plan)
        query = layer_norm(query, self.params.norms[1])
        ffn = self.params.ffn
        hidden = ttnn.linear(query, ffn.linear1.weight, bias=ffn.linear1.bias, activation="relu")
        hidden = ttnn.linear(hidden, ffn.linear2.weight, bias=ffn.linear2.bias)
        return layer_norm(hidden, self.params.norms[2], residual=query)


class TTBEVFormerEncoder:
    def __init__(
        self,
        params,
        device,
        *,
        bev_h,
        bev_w,
        spatial_shapes,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=8,
        tsa_num_points=4,
        num_cams=6,
        num_points_in_pillar=4,
        pc_range=PC_RANGE,
    ):
        """``params`` from ``create_bevformer_encoder_parameters``, one entry per layer;
        ``spatial_shapes`` the camera feature levels as (h, w)."""
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long)
        assert spatial_shapes.shape == (num_levels, 2), f"spatial_shapes {tuple(spatial_shapes.shape)}"
        self.device = device
        self.bev_h, self.bev_w = bev_h, bev_w
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.tsa_num_points = tsa_num_points
        self.pc_range = list(pc_range)
        self.spatial_shapes = spatial_shapes
        self.num_keys = int(spatial_shapes.prod(dim=1).sum())
        z_cfg = dict(num_points=num_points_in_pillar, start=self.pc_range[2], end=self.pc_range[5])
        self._z_cfg = z_cfg
        self._ref_2d = ttnn.from_torch(
            BEVFormerEncoder.reference_points_2d(bev_h, bev_w, 1),
            device=device,
            dtype=GRID_DTYPE,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        grid_sample_compute_config = fp32_grid_sample_config(device)
        self.layers = [
            TTBEVFormerLayer(
                layer_params,
                device,
                bev_shape=(bev_h, bev_w),
                spatial_shapes=spatial_shapes,
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                tsa_num_points=tsa_num_points,
                num_cams=num_cams,
                grid_sample_compute_config=grid_sample_compute_config,
            )
            for layer_params in params.layers
        ]

    def prepare_frame(self, img_metas):
        """Per-frame camera geometry: each pillar's points in every camera, which of them land
        in the image, and the rebatch plan the spatial cross-attention gathers with."""
        bs = len(img_metas)
        lidar2img = torch.stack([torch.as_tensor(meta["lidar2img"], dtype=torch.float32) for meta in img_metas])
        reference_points_3d = generate_reference_points(self.bev_h, self.bev_w, self._z_cfg, batch_size=bs)
        reference_points_cam, bev_mask = point_sampling_3d_to_2d(
            reference_points_3d, self.pc_range, lidar2img, img_metas=img_metas
        )
        rebatch_plan = build_rebatch_plan(reference_points_cam, bev_mask, self.embed_dims, self.device, GRID_DTYPE)
        return SimpleNamespace(rebatch_plan=rebatch_plan)

    def __call__(self, bev_query, value, bev_pos, frame, prev_bev=None, shift=None):
        """``bev_query``, ``bev_pos`` and ``prev_bev`` (already rotated to the current frame, or
        None) ``(bs, num_query, C)``; ``value`` ``(num_cams, num_keys, bs, C)``; ``frame`` from
        :meth:`prepare_frame`; ``shift`` ``(bs, 1, 1, 2)`` float32, the ego translation in BEV
        fractions. Returns ``(bs, num_query, C)``."""
        bs, num_query, embed_dims = bev_query.shape
        assert num_query == self.bev_h * self.bev_w, f"{num_query} queries for a {self.bev_h}x{self.bev_w} BEV"
        assert (
            value.shape[1] == self.num_keys
        ), f"{value.shape[1]} keys for spatial_shapes {self.spatial_shapes.tolist()}"

        ref_2d = self._ref_2d if bs == 1 else ttnn.repeat(self._ref_2d, ttnn.Shape((bs, 1, 1, 1)))
        if prev_bev is not None:
            previous_ref = ref_2d if shift is None else ttnn.add(ref_2d, shift)
            hybrid_ref = ttnn.concat([ttnn.unsqueeze(previous_ref, 1), ttnn.unsqueeze(ref_2d, 1)], dim=1)
            # The previous BEV is paired with the encoder's input query, not each layer's input.
            prev_bev = ttnn.concat([ttnn.unsqueeze(prev_bev, 1), ttnn.unsqueeze(bev_query, 1)], dim=1)
            prev_bev = ttnn.reshape(prev_bev, (bs * 2, num_query, embed_dims))
        else:
            hybrid_ref = ttnn.concat([ttnn.unsqueeze(ref_2d, 1), ttnn.unsqueeze(ref_2d, 1)], dim=1)
        hybrid_ref = ttnn.reshape(hybrid_ref, (bs * 2, num_query, 1, 2))
        grid_bias = tsa_grid_bias(hybrid_ref, self.num_heads, self.tsa_num_points, GRID_DTYPE)

        output = bev_query
        for layer in self.layers:
            output = layer(output, value, bev_pos, prev_bev, grid_bias, frame)
        return output
