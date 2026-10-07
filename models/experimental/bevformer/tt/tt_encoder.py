# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of BEVFormer's encoder (``reference/encoder.py``).

The layers run batch-first, ``(bs, num_query, embed_dims)``; the camera features are
``(bs * num_cams, num_keys, embed_dims)``: each sample's cameras in turn, the FPN's order, with the
levels concatenated along ``num_keys``, as ``TtPerceptionTransformer.camera_features`` builds them.
Parameters come from ``model_preprocessing.create_bevformer_encoder_parameters``.

Camera geometry depends on the cameras only, not on the activations: once per frame,
:meth:`TTBEVFormerEncoder.prepare_frame` projects the pillar points into the cameras in float32 on
the host, as upstream's ``point_sampling`` does (``point_sampling_3d_2d.camera_geometry``), and
fills the spatial cross-attention's rebatch plan, whose per-camera capacity is fixed when it is
built, so later frames refill it in place. In bfloat16 the projection's homogeneous divide loses
the points' precision. The forward then runs on device only, the cells' reference points and the
ego shift included.
"""


import ttnn
from models.experimental.bevformer.model_config import GRID_DTYPE
from models.experimental.bevformer.reference.point_sampling_3d_2d import bev_reference_points, camera_geometry
from models.experimental.bevformer.tt.tt_common import TtFFN, layer_norm
from models.experimental.bevformer.tt.tt_ms_deformable_attention import fp32_grid_sample_config
from models.experimental.bevformer.tt.tt_spatial_cross_attention import (
    TTSpatialCrossAttention,
    build_rebatch_plan,
    update_rebatch_plan,
)
from models.experimental.bevformer.tt.tt_temporal_self_attention import TTTemporalSelfAttention


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
        self.ffn = TtFFN(params.ffn)
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
        """``grid_bias`` as ``tt_temporal_self_attention.tsa_grid_bias`` lays it out, ``sca_frame`` from
        ``TTSpatialCrossAttention.frame_inputs``; both shared by every layer of a frame."""
        query = self.temporal_self_attention(query, prev_bev, bev_pos, grid_bias)
        query = layer_norm(query, self.params.norms[0])
        query = self.spatial_cross_attention(query, value, sca_frame)
        query = layer_norm(query, self.params.norms[1])
        return layer_norm(self.ffn(query), self.params.norms[2], residual=query)


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
    ):
        """``params`` from ``create_bevformer_encoder_parameters``: one entry per layer and the
        reference encoder's configuration (heads, points, cameras, pillar depth, ``pc_range``);
        ``spatial_shapes`` the camera feature levels as (h, w)."""
        config = params.config
        spatial_shapes = [tuple(int(v) for v in shape) for shape in spatial_shapes]
        assert (
            len(spatial_shapes) == config.num_levels
        ), f"{len(spatial_shapes)} spatial_shapes for {config.num_levels} levels"
        self.device = device
        self.bev_h, self.bev_w = bev_h, bev_w
        self.embed_dims = config.embed_dims
        self.num_heads = config.num_heads
        self.tsa_num_points = config.tsa_num_points
        self.num_points_in_pillar = config.num_points_in_pillar
        self.pc_range = list(config.pc_range)
        self.num_keys = sum(h * w for h, w in spatial_shapes)
        # The self-attention's grid bias, ``2 * ref - 1`` in its offset Linear's queue-major channels
        # (tsa_grid_bias), depends on the BEV grid only; a frame adds its ego shift to the previous
        # BEV's half.
        channels_per_map = self.num_heads * self.tsa_num_points
        base = (2 * bev_reference_points(bev_h, bev_w, 1) - 1).expand(-1, -1, 2 * channels_per_map, -1)
        self._tsa_grid_base = ttnn.from_torch(
            base.reshape(1, bev_h * bev_w, 2 * channels_per_map * 2),
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
                embed_dims=config.embed_dims,
                num_heads=config.num_heads,
                num_levels=config.num_levels,
                num_points=config.num_points,
                tsa_num_points=config.tsa_num_points,
                num_cams=config.num_cams,
                grid_sample_compute_config=grid_sample_compute_config,
            )
            for layer_params in params.layers
        ]

    def prepare_frame(self, img_metas, plan=None, capacity=None):
        """The spatial cross-attention's rebatch plan for this frame's cameras (``lidar2img`` and
        ``img_shape`` per sample). Without ``plan``, a new one: by default sized for this frame
        only; a plan that later frames refill needs a ``capacity`` (rows per camera) covering
        them: a bound for the rig, or ``tt_spatial_cross_attention.full_capacity`` on grids smaller
        than the base one, where it does not fit in DRAM. With ``plan``,
        refilled in place, which keeps a trace captured with it valid."""
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
        None) ``(bs, num_query, C)``; ``value`` ``(bs * num_cams, num_keys, C)``; ``plan`` from
        :meth:`prepare_frame`; ``shift`` the ego translation in BEV fractions, a float32 ROW_MAJOR
        ``(bs, 1, 1, 2)`` device tensor. Returns ``(bs, num_query, C)``."""
        bs, num_query, embed_dims = bev_query.shape
        assert num_query == self.bev_h * self.bev_w, f"{num_query} queries for a {self.bev_h}x{self.bev_w} BEV"
        assert value.shape[1] == self.num_keys, f"{value.shape[1]} keys, expected {self.num_keys}"

        grid_bias = self._tsa_grid_base if bs == 1 else ttnn.repeat(self._tsa_grid_base, ttnn.Shape((bs, 1, 1)))
        if prev_bev is not None:
            if shift is not None:
                assert (
                    tuple(shift.shape) == (bs, 1, 1, 2) and shift.dtype == GRID_DTYPE
                ), f"shift must be {GRID_DTYPE} (bs, 1, 1, 2), got {shift.dtype} {tuple(shift.shape)}"
                # The previous BEV's points are ego-shifted: 2 * shift on its (head, point) channels,
                # the first half of each row; the current queries' half stays.
                channels_per_map = self.num_heads * self.tsa_num_points
                shift_bias = ttnn.repeat(ttnn.mul(shift, 2.0), ttnn.Shape((1, 1, channels_per_map, 1)))
                shift_bias = ttnn.reshape(shift_bias, (bs, 1, channels_per_map * 2))
                shift_bias = ttnn.pad(shift_bias, [(0, 0), (0, 0), (0, channels_per_map * 2)], 0.0)
                grid_bias = ttnn.add(grid_bias, shift_bias)
            # The previous BEV is paired with the encoder's input query, not each layer's input.
            prev_bev = ttnn.concat([ttnn.unsqueeze(prev_bev, 1), ttnn.unsqueeze(bev_query, 1)], dim=1)
            prev_bev = ttnn.reshape(prev_bev, (bs * 2, num_query, embed_dims))
        # Every layer's cross-attention has the same heads, levels and points, so the first layer's
        # frame inputs serve them all. Built here, inside the (traced) forward, from the plan.
        sca_frame = self.layers[0].spatial_cross_attention.frame_inputs(plan)

        output = bev_query
        for layer in self.layers:
            output = layer(output, value, bev_pos, prev_bev, grid_bias, sca_frame)
        return output
