# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import warnings
import torch
import ttnn
from models.experimental.vadv2.tt.tt_utils import fused_msda, msda_operands
from models.experimental.vadv2.tt.matmul_helpers import linear_flatten_batch


class TtSpatialCrossAttention:
    def __init__(
        self,
        device,
        params,
        spatial_shapes_hw,
        embed_dims=256,
        num_cams=6,
        pc_range=None,
        dropout=0.1,
        init_cfg=None,
        batch_first=False,
        deformable_attention=dict(type="MSDeformableAttention3D", embed_dims=256, num_levels=4),
        **kwargs,
    ):
        super(TtSpatialCrossAttention, self).__init__()

        self.device = device
        self.params = params
        self.init_cfg = init_cfg
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = TtMSDeformableAttention3D(
            device=self.device, params=params, spatial_shapes_hw=spatial_shapes_hw, num_levels=1
        )
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.batch_first = batch_first
        self._plan = None

    def prepare(self, reference_points_cam, bev_mask):
        """Build the per-camera query gather/scatter plan for a fixed camera rig.

        reference_points_cam: torch float32 (num_cams, 1, num_query, D, 2), bev_mask: torch bool
        (num_cams, 1, num_query, D), both from the encoder's host projection. The plan depends
        only on the calibration, so it is built here once and the forward stays on device.
        """
        num_cams, _, num_query, D, _ = reference_points_cam.shape
        assert num_cams == self.num_cams
        indexes = [mask[0].sum(-1).nonzero().squeeze(-1) for mask in bev_mask]
        max_len = max(len(index) for index in indexes)

        # Cameras are padded to max_len. Padded rows scatter into a sink row at index num_query,
        # sliced off after the scatter, and gather from row 0, which is harmless as their
        # results land in the sink.
        scatter_index = torch.full((num_cams, max_len), num_query, dtype=torch.int32)
        gather_index = torch.zeros((num_cams, max_len), dtype=torch.int32)
        reference_points_rebatch = torch.zeros((num_cams, max_len, D, 2), dtype=torch.float32)
        for cam, index in enumerate(indexes):
            scatter_index[cam, : len(index)] = index
            gather_index[cam, : len(index)] = index
            reference_points_rebatch[cam, : len(index)] = reference_points_cam[cam, 0, index]

        count = (bev_mask.sum(-1) > 0).permute(1, 2, 0).sum(-1).clamp(min=1.0).unsqueeze(-1).float()

        def upload(tensor, dtype, layout):
            return ttnn.from_torch(
                tensor, dtype=dtype, layout=layout, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        all_idx_expanded = upload(scatter_index.reshape(-1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        all_idx_expanded = ttnn.expand(ttnn.unsqueeze(all_idx_expanded, 1), [-1, self.embed_dims])
        self._plan = dict(
            num_query=num_query,
            max_len=max_len,
            all_idx_expanded=ttnn.to_memory_config(all_idx_expanded, ttnn.DRAM_MEMORY_CONFIG),
            all_idx_for_gather_q=upload(gather_index.reshape(-1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            # float32: the fused op forms the sampling locations from them in float32.
            reference_points_rebatch=upload(reference_points_rebatch, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
            count=upload(count, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            # Cloned per call: zero fills are host writes, and the forward must not issue them.
            slots_template=ttnn.zeros(
                (1, num_query, self.embed_dims), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            ),
            sentinel_row_template=ttnn.zeros(
                (1, self.embed_dims), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            ),
        )

    def __call__(
        self,
        query,
        key,
        value,
        residual=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        flag="encoder",
        **kwargs,
    ):
        assert self._plan is not None, "call prepare() with the rig's reference_points_cam / bev_mask first"
        plan = self._plan
        max_len = plan["max_len"]

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = ttnn.clone(plan["slots_template"])
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape
        assert bs == 1, "the device-side gather/scatter assumes bs=1"
        assert num_query == plan["num_query"]

        query_2d = ttnn.squeeze(query, 0)
        queries_gathered = ttnn.embedding(plan["all_idx_for_gather_q"], query_2d, layout=ttnn.TILE_LAYOUT)
        queries_rebatch = ttnn.reshape(queries_gathered, (bs * self.num_cams, max_len, self.embed_dims))
        num_cams, l, bs, embed_dims = key.shape

        key = ttnn.permute(key, (2, 0, 1, 3))
        key = ttnn.reshape(key, (bs * self.num_cams, l, self.embed_dims))

        value = ttnn.permute(value, (2, 0, 1, 3))
        value = ttnn.reshape(value, (bs * self.num_cams, l, self.embed_dims))
        queries = self.deformable_attention(
            query=queries_rebatch,
            key=key,
            value=value,
            reference_points=plan["reference_points_rebatch"],
        )
        ttnn.deallocate(queries_rebatch)

        # Padded rows go to the sink row at index num_query through the sentinel
        # in all_idx_expanded and are sliced off, so no mask multiply is needed.
        queries_flat = ttnn.reshape(queries, (self.num_cams * max_len, self.embed_dims))
        slots_2d = ttnn.squeeze(slots, 0)
        slots_2d = ttnn.to_layout(slots_2d, ttnn.TILE_LAYOUT)
        sentinel_row = ttnn.clone(plan["sentinel_row_template"])
        slots_extended = ttnn.concat([slots_2d, sentinel_row], dim=0)
        slots_extended = ttnn.scatter_add(slots_extended, 0, plan["all_idx_expanded"], queries_flat)
        slots = ttnn.slice(slots_extended, [0, 0], [num_query, self.embed_dims])
        slots = ttnn.unsqueeze(slots, 0)

        slots = ttnn.to_layout(slots, layout=ttnn.TILE_LAYOUT)
        slots = ttnn.div(slots, plan["count"])
        slots = linear_flatten_batch(slots, self.params.output_proj.weight, bias=self.params.output_proj.bias)
        ttnn.deallocate(key)
        ttnn.deallocate(value)

        output = slots + inp_residual
        ttnn.deallocate(slots)
        ttnn.deallocate(inp_residual)

        return output


class TtMSDeformableAttention3D:
    def __init__(
        self,
        device,
        params,
        spatial_shapes_hw,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=8,
        im2col_step=64,
        dropout=0.1,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False
        self.device = device
        self.params = params

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        assert len(spatial_shapes_hw) == num_levels, f"need {num_levels} (H, W) pairs, got {spatial_shapes_hw}"
        self.spatial_shapes_hw = [(int(h), int(w)) for h, w in spatial_shapes_hw]

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        **kwargs,
    ):
        params = self.params
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert sum(h * w for h, w in self.spatial_shapes_hw) == num_value
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

        value = linear_flatten_batch(value, params.value_proj.weight, bias=params.value_proj.bias)
        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            value = ttnn.where(mask, ttnn.zeros_like(value), value)
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        sampling_offsets, attention_weights = msda_operands(
            query, params, self.num_heads, self.num_levels * self.num_points
        )

        if reference_points.shape[-1] != 2:
            raise NotImplementedError(
                f"only 2-D reference points are supported, got {reference_points.shape[-1]} coordinates"
            )
        # "pillar": sampling point p reads the pillar's z-anchor p % D, as the reference's
        # (num_points // D, D) split of the points.
        assert self.num_points % reference_points.shape[2] == 0
        output = fused_msda(
            value, reference_points, sampling_offsets, attention_weights, self.spatial_shapes_hw, "pillar"
        )
        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        return output
