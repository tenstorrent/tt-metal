# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import warnings
from models.experimental.MapTR.tt.utils import multi_scale_deformable_attn


class TtCustomMSDeformableAttention:
    """TTNN implementation of CustomMSDeformableAttention for MapTR.

    This is the TTNN version of the CustomMSDeformableAttention module used in
    MapTR decoder layers for cross-attention with deformable sampling.

    Args:
        params: Preprocessed parameters containing:
            - sampling_offsets: weight and bias
            - attention_weights: weight and bias
            - value_proj: weight and bias
            - output_proj: weight and bias
        device: TTNN device
        embed_dims (int): The embedding dimension. Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map levels. Default: 1.
        num_points (int): The number of sampling points per head. Default: 4.
        im2col_step (int): The step used in image_to_column. Default: 64.
        dropout (float): Dropout rate. Default: 0.1.
        batch_first (bool): If True, inputs are (batch, n, embed_dim). Default: False.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        init_cfg: The Config for initialization. Default: None.
    """

    def __init__(
        self,
        params,
        device,
        embed_dims=256,
        num_heads=8,
        num_levels=1,
        num_points=4,
        im2col_step=64,
        dropout=0.1,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
    ):
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")
        dim_per_head = embed_dims // num_heads
        self.params = params
        self.device = device
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.fp16_enabled = False

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

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        flag="decoder",
        **kwargs,
    ):
        """Forward function of TtCustomMSDeformableAttention.

        Args:
            query: Query tensor with shape (num_query, bs, embed_dims) or (bs, num_query, embed_dims).
            key: Key tensor. Default: None.
            value: Value tensor. Default: None.
            identity: Identity tensor for residual connection. Default: None.
            query_pos: Positional encoding for query. Default: None.
            key_padding_mask: ByteTensor mask for key. Default: None.
            reference_points: Reference points with shape (bs, num_query, num_levels, 2) or (bs, num_query, num_levels, 4).
            spatial_shapes: Spatial shapes of features, shape (num_levels, 2).
            level_start_index: Start index of each level. Default: None.
            flag: Flag for decoder/encoder. Default: "decoder".

        Returns:
            Output tensor with same shape as query.
        """
        params = self.params

        if value is None:
            value = query

        if identity is None:
            identity = query

        if query_pos is not None:
            query = ttnn.add(query, query_pos)

        if not self.batch_first:
            # change to (bs, num_query, embed_dims)
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        # Verify spatial shapes (convert to torch since ttnn.sum doesn't support INT32)
        # Use float32 for calculation to avoid bfloat16 precision issues with large integers
        spatial_shapes_torch = ttnn.to_torch(spatial_shapes).float()
        expected_num_value = int((spatial_shapes_torch[:, 0] * spatial_shapes_torch[:, 1]).sum().item())
        assert expected_num_value == num_value, f"spatial_shapes product {expected_num_value} != num_value {num_value}"
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

        value = ttnn.linear(value, params.value_proj.weight, bias=params.value_proj.bias)
        # Deallocate value_proj parameters after use
        ttnn.deallocate(params.value_proj.weight)
        ttnn.deallocate(params.value_proj.bias)
        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            value = ttnn.where(mask, ttnn.zeros_like(value), value)
        value = ttnn.reshape(value, (bs, num_value, self.num_heads, -1))

        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        # Compute both linear operations using query before deallocating
        sampling_offsets = ttnn.linear(query, params.sampling_offsets.weight, bias=params.sampling_offsets.bias)
        attention_weights = ttnn.linear(query, params.attention_weights.weight, bias=params.attention_weights.bias)
        # Now safe to deallocate query and weight parameters
        ttnn.deallocate(params.sampling_offsets.weight)
        ttnn.deallocate(params.sampling_offsets.bias)
        ttnn.deallocate(params.attention_weights.weight)
        ttnn.deallocate(params.attention_weights.bias)
        ttnn.deallocate(query)
        # Reshape after deallocation
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        )
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels * self.num_points)
        )
        attention_weights = ttnn.softmax(attention_weights, -1)

        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels, self.num_points)
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = ttnn.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)
            bs_r, num_query, num_levels, _ = reference_points.shape
            reference_xy = ttnn.reshape(reference_points, (bs_r, num_query, 1, num_levels, 1, 2))
            offset_normalizer_xy = ttnn.reshape(
                offset_normalizer, (1, 1, 1, offset_normalizer.shape[0], 1, offset_normalizer.shape[1])
            )
            sampling_offsets = ttnn.to_layout(sampling_offsets, ttnn.TILE_LAYOUT)
            offset_normalizer_xy = ttnn.to_layout(offset_normalizer_xy, ttnn.TILE_LAYOUT)
            sampling_offsets = ttnn.squeeze(sampling_offsets, 0)
            sampling_offsets = ttnn.squeeze(sampling_offsets, 2)
            offset_normalizer_xy = ttnn.squeeze(offset_normalizer_xy, 0)
            offset_normalizer_xy = ttnn.squeeze(offset_normalizer_xy, 0)

            sampling_locations = ttnn.divide(sampling_offsets, offset_normalizer_xy, use_legacy=False)

            sampling_locations = ttnn.unsqueeze(sampling_locations, 2)
            sampling_locations = ttnn.unsqueeze(sampling_locations, 0)
            sampling_locations = ttnn.add(reference_xy, sampling_locations, use_legacy=False)

        elif reference_points.shape[-1] == 4:
            reference_points_reshape = ttnn.reshape(
                reference_points,
                [reference_points.shape[0], reference_points.shape[1], 1, reference_points.shape[2], 1, 2],
            )
            sampling_locations = (
                reference_points_reshape + sampling_offsets / self.num_points * reference_points_reshape * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be" f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        # Multi-scale deformable attention
        output = multi_scale_deformable_attn(value, spatial_shapes, sampling_locations, attention_weights, self.device)

        output = ttnn.linear(output, params.output_proj.weight, bias=params.output_proj.bias)
        ttnn.deallocate(params.output_proj.weight)
        ttnn.deallocate(params.output_proj.bias)
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        output = ttnn.add(output, identity)

        return output
