# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of BEVFormer's encoder (``reference/encoder.py``).

The layers run batch-first, ``(bs, num_query, embed_dims)``; the camera features stay as the
reference takes them, ``(num_cams, num_keys, bs, embed_dims)``. Parameters come from
``model_preprocessing.create_bevformer_encoder_parameters``.

Camera geometry depends on the cameras only, not on the activations: once per frame,
:meth:`TTBEVFormerEncoder.prepare_frame` projects the pillar points into the cameras in float32 on
the host, as upstream's ``point_sampling`` does (``point_sampling_3d_2d.camera_geometry``), and
fills the spatial cross-attention's rebatch plan, whose size the visibility mask decides. In
bfloat16 the projection's homogeneous divide loses the points' precision. The forward then runs
on device only, the cells' reference points and the ego shift included.
"""


import ttnn
from models.experimental.bevformer.config.head_config import PC_RANGE

from ..reference.point_sampling_3d_2d import bev_reference_points, camera_geometry
from .tt_common import GRID_DTYPE, layer_norm
from .tt_ms_deformable_attention import fp32_grid_sample_config
from .tt_spatial_cross_attention import TTSpatialCrossAttention, build_rebatch_plan, update_rebatch_plan
from .tt_temporal_self_attention import TTTemporalSelfAttention, tsa_grid_bias


class TTBEVFormerLayer:
    """One encoder layer, ``self_attn -> norm -> cross_attn -> norm -> ffn -> norm``, as upstream's
    ``operation_order``. The positional encoding enters the self-attention only."""

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
        """``params`` is one entry of ``create_bevformer_encoder_parameters(...).layers``."""
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
            params.sca,
            device,
            spatial_shapes=spatial_shapes,
            embed_dims=embed_dims,
            num_cams=num_cams,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            grid_dtype=GRID_DTYPE,
            grid_sample_compute_config=grid_sample_compute_config,
        )

    def __call__(self, query, value, bev_pos, prev_bev, grid_bias, sca_frame):
        """``grid_bias`` from :func:`tsa_grid_bias`, ``sca_frame`` from
        ``TTSpatialCrossAttention.frame_inputs``; both shared by every layer of a frame."""
        query = self.temporal_self_attention(query, prev_bev, bev_pos, grid_bias)
        query = layer_norm(query, self.params.norms[0])
        query = self.spatial_cross_attention(query, value, sca_frame)
        query = layer_norm(query, self.params.norms[1])
        ffn = self.params.ffn
        hidden = ttnn.linear(query, ffn.linear1.weight, bias=ffn.linear1.bias, activation="relu")
        hidden = ttnn.linear(hidden, ffn.linear2.weight, bias=ffn.linear2.bias)
        return layer_norm(hidden, self.params.norms[2], residual=query)


class TTBEVFormerEncoder:
    """Encoder over a ``(bev_h, bev_w)`` BEV grid and camera features of ``spatial_shapes``.

    The deformable attentions fold the BEV and feature sizes into their sampling-offset Linears;
    the spatial cross-attention's fold consumes ``params.layers[*].sca.deformable_attention.sampling_offsets``,
    so each instance needs its own ``create_bevformer_encoder_parameters``, and reusing them raises.
    """

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
        ``spatial_shapes`` the camera feature levels as (h, w); ``num_points`` the spatial
        cross-attention's sampling points per head and level, ``tsa_num_points`` the self-attention's."""
        spatial_shapes = [tuple(int(v) for v in shape) for shape in spatial_shapes]
        assert len(spatial_shapes) == num_levels, f"{len(spatial_shapes)} spatial_shapes for {num_levels} levels"
        self.device = device
        self.bev_h, self.bev_w = bev_h, bev_w
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.tsa_num_points = tsa_num_points
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = list(pc_range)
        self.num_keys = sum(h * w for h, w in spatial_shapes)
        self._ref_2d = ttnn.from_torch(
            bev_reference_points(bev_h, bev_w, 1), device=device, dtype=GRID_DTYPE, layout=ttnn.ROW_MAJOR_LAYOUT
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

    def prepare_frame(self, img_metas, plan=None, capacity=None):
        """The spatial cross-attention's rebatch plan for this frame's cameras (``lidar2img`` and
        ``img_shape`` per sample). Without ``plan``, a new one, sized for this frame unless
        ``capacity`` (rows per camera) is given. With ``plan``, refilled in place, which keeps a
        trace captured with it valid; its capacity must cover this frame."""
        reference_points_cam, bev_mask = camera_geometry(
            img_metas, self.bev_h, self.bev_w, self.num_points_in_pillar, self.pc_range
        )
        if plan is None:
            return build_rebatch_plan(
                reference_points_cam, bev_mask, self.embed_dims, self.device, GRID_DTYPE, capacity
            )
        update_rebatch_plan(plan, reference_points_cam, bev_mask)
        return plan

    def __call__(self, bev_query, value, bev_pos, plan, prev_bev=None, shift=None):
        """``bev_query``, ``bev_pos`` and ``prev_bev`` (already rotated to the current frame, or
        None) ``(bs, num_query, C)``; ``value`` ``(num_cams, num_keys, bs, C)``; ``plan`` from
        :meth:`prepare_frame`; ``shift`` the ego translation in BEV fractions, a float32 ROW_MAJOR
        ``(bs, 1, 1, 2)`` device tensor. Returns ``(bs, num_query, C)``."""
        bs, num_query, embed_dims = bev_query.shape
        assert num_query == self.bev_h * self.bev_w, f"{num_query} queries for a {self.bev_h}x{self.bev_w} BEV"
        assert value.shape[1] == self.num_keys, f"{value.shape[1]} keys, expected {self.num_keys}"

        ref_2d = self._ref_2d if bs == 1 else ttnn.repeat(self._ref_2d, ttnn.Shape((bs, 1, 1, 1)))
        if prev_bev is not None:
            if shift is not None:
                assert (
                    tuple(shift.shape) == (bs, 1, 1, 2) and shift.dtype == GRID_DTYPE
                ), f"shift must be {GRID_DTYPE} (bs, 1, 1, 2), got {shift.dtype} {tuple(shift.shape)}"
            previous_ref = ref_2d if shift is None else ttnn.add(ref_2d, shift)
            # Per sample, the previous BEV's points (ego-shifted) then the current queries' points.
            hybrid_ref = ttnn.concat([ttnn.unsqueeze(previous_ref, 1), ttnn.unsqueeze(ref_2d, 1)], dim=1)
            # The previous BEV is paired with the encoder's input query, not each layer's input.
            prev_bev = ttnn.concat([ttnn.unsqueeze(prev_bev, 1), ttnn.unsqueeze(bev_query, 1)], dim=1)
            prev_bev = ttnn.reshape(prev_bev, (bs * 2, num_query, embed_dims))
        else:
            hybrid_ref = ttnn.concat([ttnn.unsqueeze(ref_2d, 1), ttnn.unsqueeze(ref_2d, 1)], dim=1)
        hybrid_ref = ttnn.reshape(hybrid_ref, (bs * 2, num_query, 1, 2))
        grid_bias = tsa_grid_bias(hybrid_ref, self.num_heads, self.tsa_num_points, GRID_DTYPE)
        sca_frame = self.layers[0].spatial_cross_attention.frame_inputs(plan)

        output = bev_query
        for layer in self.layers:
            output = layer(output, value, bev_pos, prev_bev, grid_bias, sca_frame)
        return output
