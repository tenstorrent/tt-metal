import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn.modules import ModuleList
import warnings
import torch.nn.functional as F
import copy


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """CPU version of multi-scale deformable attention.

    Args:
        value (Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class MultiheadAttention(nn.Module):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        dropout_layer=dict(type="Dropout", drop_prob=0.0),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        super(MultiheadAttention, self).__init__()
        if "dropout" in kwargs:
            warnings.warn(
                "The arguments `dropout` in MultiheadAttention "
                "has been deprecated, now you can separately "
                "set `attn_drop`(float), proj_drop(float), "
                "and `dropout_layer`(dict) "
            )
            attn_drop = kwargs["dropout"]
            dropout_layer["drop_prob"] = kwargs.pop("dropout")

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = nn.Dropout(p=0.1)

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape
                [num_queries, bs, embed_dims]
                if self.batch_first is False, else
                [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f"position encoding of key is" f"missing in {self.__class__.__name__}.")
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(query=query, key=key, value=value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)
        return identity + self.dropout_layer(self.proj_drop(out))


class CustomMSDeformableAttention(nn.Module):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=1,
        num_points=4,
        im2col_step=192,
        dropout=0.1,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

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
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(
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
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels, self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be" f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


class FFN(nn.Module):
    def __init__(self, embed_dim):
        super(FFN, self).__init__()
        self.activate = nn.ReLU(inplace=True)
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(embed_dim, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.1)),
            nn.Linear(512, embed_dim),
            nn.Dropout(p=0.1),
        )

    def forward(self, x, identity=None):
        if identity is None:
            identity = x
        return identity + self.layers(x)


class DetrTransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(DetrTransformerDecoderLayer, self).__init__()
        self.attentions = ModuleList([MultiheadAttention(embed_dim, num_heads), CustomMSDeformableAttention()])
        self.ffns = ModuleList([FFN(embed_dim)])
        self.norms = ModuleList([LayerNorm(embed_dim) for _ in range(3)])

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        reference_points=None,
        spatial_shapes=None,
        reg_branches=None,
        key_padding_mask=None,
    ):
        identity = query

        query = self.attentions[0](
            query, key=query, value=query, query_pos=query_pos, key_pos=query_pos, key_padding_mask=key_padding_mask
        )
        query = self.norms[0](query)

        query = self.attentions[1](
            query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
        )
        query = self.norms[1](query)

        query = self.ffns[0](query)
        query = self.norms[2](query)

        return query  # residual connection added back


# class BaseTransformerLayer(nn.Module):
# def __init__(self, embed_dim, num_heads=8):
#     super(BaseTransformerLayer, self).__init__()
#     self.attentions = ModuleList([
#         MultiheadAttention(embed_dim, num_heads)
#     ])
#     self.ffns = ModuleList([FFN(embed_dim)])
#     self.norms = ModuleList([LayerNorm(embed_dim) for _ in range(2)])

# def forward(self, query, key=None, value=None, query_pos=None,key_pos=None,  key_padding_mask = None, attn_masks = None):
#     identity = query

#     query = self.attentions[0](query, key=query, value=value, query_pos=query_pos, key_pos=key_pos,attn_mask = attn_masks , key_padding_mask=key_padding_mask)
#     query = self.norms[0](query)

#     query = self.ffns[0](query)
#     query = self.norms[1](query)

#     return query


class MapDetectionTransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads):
        super(MapDetectionTransformerDecoder, self).__init__()
        self.return_intermediate = True
        self.layers = ModuleList(
            [
                BaseTransformerLayer(
                    attn_cfgs=[
                        {"type": "MultiheadAttention", "embed_dims": embed_dim, "num_heads": num_heads, "dropout": 0.1},
                        {"type": "CustomMSDeformableAttention", "embed_dims": embed_dim, "num_levels": 1},
                    ],
                    ffn_cfgs=dict(
                        type="FFN",
                        embed_dims=embed_dim,
                        feedforward_channels=512,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                    operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                    norm_cfg=dict(type="LN"),
                    init_cfg=None,
                    batch_first=False,
                    kwargs={
                        "feedforward_channels": 512,
                        "ffn_dropout": 0.1,
                        "act_cfg": {"type": "ReLU", "inplace": True},
                        "ffn_num_fcs": 2,
                    },
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        reference_points=None,
        spatial_shapes=None,
        reg_branches=None,
        cls_branches=None,
        **kwargs,
    ):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                key=key,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
            )
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                # new_reference_points[..., 2:3] = tmp[
                #     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            a = torch.stack(intermediate)
            b = torch.stack(intermediate_reference_points)
            return a, b
        return output, reference_points


class DetectionTransformerDecoder(nn.Module):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, num_layers, embed_dim, num_heads):
        super(DetectionTransformerDecoder, self).__init__()
        self.return_intermediate = True
        self.layers = ModuleList(
            [
                BaseTransformerLayer(
                    attn_cfgs=[
                        {"type": "MultiheadAttention", "embed_dims": embed_dim, "num_heads": num_heads, "dropout": 0.1},
                        {"type": "CustomMSDeformableAttention", "embed_dims": embed_dim, "num_levels": 1},
                    ],
                    ffn_cfgs=dict(
                        type="FFN",
                        embed_dims=embed_dim,
                        feedforward_channels=512,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                    operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                    norm_cfg=dict(type="LN"),
                    init_cfg=None,
                    batch_first=False,
                    kwargs={
                        "feedforward_channels": 512,
                        "ffn_dropout": 0.1,
                        "act_cfg": {"type": "ReLU", "inplace": True},
                        "ffn_num_fcs": 2,
                    },
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        reference_points=None,
        spatial_shapes=None,
        reg_branches=None,
        cls_branches=None,
        **kwargs,
    ):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        print("-------reg_branches-----------", reg_branches)
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)  # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                key=key,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
            )
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class CustomTransformerDecoder(nn.Module):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default: `LN`.
    """

    def __init__(self, num_layers, return_intermediate=False, embed_dim=256, num_heads=8):
        super(CustomTransformerDecoder, self).__init__()
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False
        self.layers = self.layers = ModuleList(
            [
                BaseTransformerLayer(
                    attn_cfgs=[{"type": "MultiheadAttention", "embed_dims": 256, "num_heads": 8, "dropout": 0.1}],
                    ffn_cfgs={
                        "type": "FFN",
                        "embed_dims": 256,
                        "feedforward_channels": 512,
                        "num_fcs": 2,
                        "ffn_drop": 0.1,
                        "act_cfg": {"type": "ReLU", "inplace": True},
                    },
                    operation_order=("cross_attn", "norm", "ffn", "norm"),
                    norm_cfg={"type": "LN"},
                    init_cfg=None,
                    batch_first=False,
                    kwargs={"feedforward_channels": 512, "ffn_dropout": 0.1},
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        key_padding_mask=None,
        *args,
        **kwargs,
    ):
        intermediate = []
        for lid, layer in enumerate(self.layers):
            query = layer(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                key_padding_mask=key_padding_mask,
                *args,
                **kwargs,
            )

            if self.return_intermediate:
                intermediate.append(query)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return query


class BaseTransformerLayer(nn.Module):
    def __init__(
        self,
        attn_cfgs=None,
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.0,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        operation_order=None,
        norm_cfg=dict(type="LN"),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        deprecated_args = dict(
            feedforward_channels="feedforward_channels", ffn_dropout="ffn_drop", ffn_num_fcs="num_fcs"
        )
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f"The arguments `{ori_name}` in BaseTransformerLayer "
                    f"has been deprecated, now you should set `{new_name}` "
                    f"and other FFN related arguments "
                    f"to a dict named `ffn_cfgs`. "
                )
                ffn_cfgs[new_name] = kwargs[ori_name]

        super(BaseTransformerLayer, self).__init__()

        self.batch_first = batch_first

        assert set(operation_order) & set(["self_attn", "norm", "ffn", "cross_attn"]) == set(operation_order), (
            f"The operation_order of"
            f" {self.__class__.__name__} should "
            f"contains all four operation type "
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"
        )

        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), (
                f"The length "
                f"of attn_cfg {num_attn} is "
                f"not consistent with the number of attention"
                f"in operation_order {operation_order}."
            )

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                if "batch_first" in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]["batch_first"]
                else:
                    attn_cfgs[index]["batch_first"] = self.batch_first
                if attn_cfgs[index]["type"] == "MultiheadAttention":
                    type = attn_cfgs[index].pop("type")
                    attention = MultiheadAttention(**attn_cfgs[index])
                    attn_cfgs[index]["type"] = "MultiheadAttention"
                elif attn_cfgs[index]["type"] == "CustomMSDeformableAttention":
                    type = attn_cfgs[index].pop("type")
                    attention = CustomMSDeformableAttention(**attn_cfgs[index])
                    attn_cfgs[index]["type"] = "CustomMSDeformableAttention"

                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.pre_norm = operation_order[0] == "norm"

        self.embed_dims = self.attentions[0].embed_dims

        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")
        print("------------", num_attn)
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
            print(attn_cfgs)
        else:
            assert num_attn == len(attn_cfgs), (
                f"The length "
                f"of attn_cfg {num_attn} is "
                f"not consistent with the number of attention"
                f"in operation_order {operation_order}."
            )

        self.num_attn = num_attn
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count("ffn")

        for ffn_index in range(num_ffns):
            self.ffns.append(FFN(self.embed_dims))

        self.norms = nn.ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(nn.LayerNorm(self.embed_dims))

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                print("self_attn")
                temp_key = temp_value = query
                # print(kwargs)
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                print("norm")
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                print("cross_attn")
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                print("ffn")
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
