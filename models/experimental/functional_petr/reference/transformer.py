# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
##### should delete this file, only used in the commit, The replace file is petr_transformer


# import torch
# import torch.nn as nn
# import torch.nn.functional as f
# import torch

# # from torchview import draw_graph


# class PETRMultiheadAttention(nn.Module):
#     """ "
#     This module implements MultiheadAttention with identity connection,
#     and positional encoding  is also passed as input.
#     Args:
#         embed_dims (int): The embedding dimension.
#         num_heads (int): Parallel attention heads.
#         attn_drop (float): A Dropout layer on attn_output_weights.
#             Default: 0.0.
#         proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
#             Default: 0.0.
#         dropout_layer (obj:`ConfigDict`): The dropout_layer used
#             when adding the shortcut.
#         init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
#             Default: None.
#         batch_first (bool): When it is True,  Key, Query and Value are shape of
#             (batch, n, embed_dim), otherwise (n, batch, embed_dim).
#              Default to False.
#     """

#     def __init__(
#         self,
#         embed_dims,
#         num_heads,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         batch_first=False,
#         **kwargs,
#     ):
#         super(PETRMultiheadAttention, self).__init__()

#         self.embed_dims = embed_dims
#         self.num_heads = num_heads
#         self.batch_first = batch_first

#         self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)

#     def forward(
#         self,
#         query,
#         key=None,
#         value=None,
#         identity=None,
#         query_pos=None,
#         key_pos=None,
#         attn_mask=None,
#         key_padding_mask=None,
#         **kwargs,
#     ):
#         """Forward function for `MultiheadAttention`.

#         **kwargs allow passing a more general data flow when combining
#         with other operations in `transformerlayer`.
#         Args:
#             query (Tensor): The input query with shape [num_queries, bs,
#                 embed_dims] if self.batch_first is False, else
#                 [bs, num_queries embed_dims].
#             key (Tensor): The key tensor with shape [num_keys, bs,
#                 embed_dims] if self.batch_first is False, else
#                 [bs, num_keys, embed_dims] .
#                 If None, the ``query`` will be used. Defaults to None.
#             value (Tensor): The value tensor with same shape as `key`.
#                 Same in `nn.MultiheadAttention.forward`. Defaults to None.
#                 If None, the `key` will be used.
#             identity (Tensor): This tensor, with the same shape as x,
#                 will be used for the identity link.
#                 If None, `x` will be used. Defaults to None.
#             query_pos (Tensor): The positional encoding for query, with
#                 the same shape as `x`. If not None, it will
#                 be added to `x` before forward function. Defaults to None.
#             key_pos (Tensor): The positional encoding for `key`, with the
#                 same shape as `key`. Defaults to None. If not None, it will
#                 be added to `key` before forward function. If None, and
#                 `query_pos` has the same shape as `key`, then `query_pos`
#                 will be used for `key_pos`. Defaults to None.
#             attn_mask (Tensor): ByteTensor mask with shape [num_queries,
#                 num_keys]. Same in `nn.MultiheadAttention.forward`.
#                 Defaults to None.
#             key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
#                 Defaults to None.
#         Returns:
#             Tensor: forwarded results with shape
#             [num_queries, bs, embed_dims]
#             if self.batch_first is False, else
#             [bs, num_queries embed_dims].
#         """

#         if key is None:
#             key = query
#         if value is None:
#             value = key
#         if identity is None:
#             identity = query
#         if key_pos is None:
#             if query_pos is not None:
#                 # use query_pos if key_pos is not available
#                 if query_pos.shape == key.shape:
#                     key_pos = query_pos

#         if query_pos is not None:
#             query = query + query_pos
#         if key_pos is not None:
#             key = key + key_pos

#         if self.batch_first:
#             query = query.transpose(0, 1)
#             key = key.transpose(0, 1)
#             value = value.transpose(0, 1)
#         print()
#         out = self.attn(query=query, key=key, value=value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]

#         if self.batch_first:
#             out = out.transpose(0, 1)

#         return identity + out


# class FFN(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(FFN, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Sequential(
#                 nn.Linear(in_features=in_features[0], out_features=out_features[0], bias=True),
#                 nn.ReLU(inplace=True),
#             ),
#             nn.Linear(in_features=in_features[1], out_features=out_features[1], bias=True),
#         )

#     def forward(self, x):
#         input = x
#         x = self.layers(x)
#         return x + input


# class PETRTransformerDecoderLayer(nn.Module):
#     def __init__(
#         self,
#         embed_dims,
#         num_heads,
#         in_features,
#         out_features,
#         normalized_shape,
#         attn_drop=[0.1, 0.1],
#         proj_drop=[0.0, 0.0],
#         batch_first=[False, False],
#     ):
#         super(PETRTransformerDecoderLayer, self).__init__()
#         self.attentions = nn.ModuleList(
#             [
#                 PETRMultiheadAttention(
#                     embed_dims[0],
#                     num_heads[0],
#                     attn_drop[0],
#                     proj_drop[0],
#                     batch_first[0],
#                 ),
#                 PETRMultiheadAttention(
#                     embed_dims[1],
#                     num_heads[1],
#                     attn_drop[1],
#                     proj_drop[1],
#                     batch_first[1],
#                 ),
#             ]
#         )
#         self.ffns = nn.ModuleList([FFN(in_features, out_features)])
#         self.norms = nn.ModuleList(
#             [nn.LayerNorm(normalized_shape[0]), nn.LayerNorm(normalized_shape[1]), nn.LayerNorm(normalized_shape[2])]
#         )

#     def forward(self, query, key, value, key_pos, query_pos, key_padding_mask):
#         x = self.attentions[0](query, query, query, query_pos=query_pos, key_pos=query_pos)
#         x = self.norms[0](x)

#         x = self.attentions[1](x, key, value, query_pos=query_pos, key_pos=key_pos, key_padding_mask=key_padding_mask)
#         x = self.norms[1](x)

#         x = self.ffns[0](x)
#         x = self.norms[2](x)

#         return x


# class PETRTransformerDecoder(nn.Module):
#     def __init__(
#         self,
#     ):
#         super(PETRTransformerDecoder, self).__init__()
#         self.layers = nn.ModuleList(
#             [
#                 PETRTransformerDecoderLayer(
#                     [256, 256],
#                     [8, 8],
#                     [256, 2048],
#                     [2048, 256],
#                     [256, 256, 256],
#                 ),
#                 PETRTransformerDecoderLayer(
#                     [256, 256],
#                     [8, 8],
#                     [256, 2048],
#                     [2048, 256],
#                     [256, 256, 256],
#                 ),
#                 PETRTransformerDecoderLayer(
#                     [256, 256],
#                     [8, 8],
#                     [256, 2048],
#                     [2048, 256],
#                     [256, 256, 256],
#                 ),
#                 PETRTransformerDecoderLayer(
#                     [256, 256],
#                     [8, 8],
#                     [256, 2048],
#                     [2048, 256],
#                     [256, 256, 256],
#                 ),
#                 PETRTransformerDecoderLayer(
#                     [256, 256],
#                     [8, 8],
#                     [256, 2048],
#                     [2048, 256],
#                     [256, 256, 256],
#                 ),
#                 PETRTransformerDecoderLayer(
#                     [256, 256],
#                     [8, 8],
#                     [256, 2048],
#                     [2048, 256],
#                     [256, 256, 256],
#                 ),
#             ]
#         )
#         self.post_norm = nn.LayerNorm((256))

#     def forward(self, query, key, value, key_pos, query_pos, key_padding_mask):
#         x = self.layers[0](query, key, value, key_pos, query_pos, key_padding_mask)
#         x1 = self.post_norm(x)
#         x = self.layers[1](x, key, value, key_pos, query_pos, key_padding_mask)
#         x2 = self.post_norm(x)
#         x = self.layers[2](x, key, value, key_pos, query_pos, key_padding_mask)
#         x3 = self.post_norm(x)
#         x = self.layers[3](x, key, value, key_pos, query_pos, key_padding_mask)
#         x4 = self.post_norm(x)
#         x = self.layers[4](x, key, value, key_pos, query_pos, key_padding_mask)
#         x5 = self.post_norm(x)
#         x = self.layers[5](x, key, value, key_pos, query_pos, key_padding_mask)
#         x6 = self.post_norm(x)
#         x = torch.stack((x1, x2, x3, x4, x5, x6))
#         return x


# class PETRTransformer(nn.Module):
#     def __init__(self):
#         super(PETRTransformer, self).__init__()
#         self.decoder = PETRTransformerDecoder()

#     def forward(
#         self,
#         x,
#         mask,
#         query_embed,
#         pos_embed,
#     ):
#         bs, n, c, h, w = x.shape
#         memory = x.permute(1, 3, 4, 0, 2).reshape(-1, bs, c)  # [bs, n, c, h, w] -> [n*h*w, bs, c]
#         pos_embed = pos_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, c)  # [bs, n, c, h, w] -> [n*h*w, bs, c]
#         query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
#         mask = mask.view(bs, -1)  # [bs, n, h, w] -> [bs, n*h*w]
#         target = torch.zeros_like(query_embed)

#         out_dec = self.decoder(
#             query=target,
#             key=memory,
#             value=memory,
#             key_pos=pos_embed,
#             query_pos=query_embed,
#             key_padding_mask=mask,
#         )
#         out_dec = out_dec.transpose(1, 2)
#         memory = memory.reshape(n, h, w, bs, c).permute(3, 0, 4, 1, 2)
#         return out_dec, memory


# # model = PETRTransformer()
# # print("model")
# # model_graph = draw_graph(
# # model,
# # input_size=[(1, 6, 256, 20, 50),(1, 6, 20, 50),(900, 256),(1, 6, 256, 20, 50)],
# # expand_nested=True,
# # graph_name="model_petr_transformer_ref",
# # )
# # model_graph.visual_graph.render(format="pdf")
# # print("graph saved")

# # x = torch.rand(1, 6, 256, 20, 50)
# # mask = torch.zeros((1, 6, 20, 50), dtype=torch.bool)
# # query_embed = torch.rand(900, 256)
# # pos_embed = torch.rand(1, 6, 256, 20, 50)
# # out, mem = model(x, mask, query_embed, pos_embed)
