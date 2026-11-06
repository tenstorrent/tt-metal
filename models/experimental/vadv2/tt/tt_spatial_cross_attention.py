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
        # Each camera's last plan row is padding, the zero source for unfilled contribution slots.
        self.zero_rows = ttnn.from_torch(
            torch.zeros(1, num_cams, 1, embed_dims), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
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
        rebatch_plan=None,
        flag="encoder",
        **kwargs,
    ):
        """rebatch_plan: the frame's SCARebatchPlan, the cameras' visible queries compacted into
        row blocks; every encoder layer shares it."""
        assert rebatch_plan is not None, "the encoder builds the frame's rebatch plan"
        if value is None:
            value = key if key is not None else query
        inp_residual = query if residual is None else residual
        if query_pos is not None:
            query = query + query_pos
        bs, num_query, _ = query.shape
        assert bs == 1, "the rebatch plan is built for bs=1"

        if rebatch_plan.is_empty:
            # No camera sees any query: the attention contributes nothing but the projection's bias.
            return ttnn.add(inp_residual, self.params.output_proj.bias)

        rebatch_len = rebatch_plan.rebatch_len
        query_rows = ttnn.reshape(
            ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            (1, 1, bs * num_query, self.embed_dims),
        )
        queries_rebatch = ttnn.reshape(
            ttnn.embedding(rebatch_plan.query_index, query_rows, layout=ttnn.TILE_LAYOUT),
            (bs * self.num_cams, rebatch_len, self.embed_dims),
        )

        num_cams, l, bs, embed_dims = value.shape
        value = ttnn.permute(value, (2, 0, 1, 3))
        value = ttnn.reshape(value, (bs * self.num_cams, l, self.embed_dims))
        queries = self.deformable_attention(
            query=queries_rebatch,
            value=value,
            reference_points=rebatch_plan.reference_points_batched,
            num_valid_queries_tensor=rebatch_plan.num_valid_queries,
        )
        ttnn.deallocate(queries_rebatch)
        ttnn.deallocate(value)

        # The rows past each camera's valid count are left unwritten; its last one, which every
        # unfilled contribution slot reads, is zeroed.
        camera_rows = ttnn.reshape(queries, (1, bs * self.num_cams, rebatch_len, self.embed_dims))
        ttnn.experimental.slice_write(
            self.zero_rows,
            camera_rows,
            [0, 0, rebatch_len - 1, 0],
            [1, self.num_cams, rebatch_len, self.embed_dims],
            [1, 1, 1, 1],
        )
        contribution_rows = ttnn.reshape(camera_rows, (1, 1, bs * self.num_cams * rebatch_len, self.embed_dims))
        gather_layout = ttnn.TILE_LAYOUT if (bs * num_query) % ttnn.TILE_SIZE == 0 else ttnn.ROW_MAJOR_LAYOUT
        slots = None
        for index in rebatch_plan.inverse_index:
            gathered = ttnn.to_layout(
                ttnn.reshape(
                    ttnn.embedding(index, contribution_rows, layout=gather_layout), (bs, num_query, self.embed_dims)
                ),
                ttnn.TILE_LAYOUT,
            )
            slots = gathered if slots is None else ttnn.add(slots, gathered, dtype=ttnn.float32)
        ttnn.deallocate(queries)
        slots = ttnn.divide(slots, rebatch_plan.count, dtype=ttnn.bfloat16)
        slots = linear_flatten_batch(slots, self.params.output_proj.weight, bias=self.params.output_proj.bias)
        return ttnn.add(slots, inp_residual)


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
        num_valid_queries_tensor=None,
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
            value,
            reference_points,
            sampling_offsets,
            attention_weights,
            self.spatial_shapes_hw,
            "pillar",
            num_valid_queries_tensor=num_valid_queries_tensor,
        )
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        return output
