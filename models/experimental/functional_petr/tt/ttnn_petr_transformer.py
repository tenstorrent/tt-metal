# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch
from torch import nn
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias
from models.experimental.functional_petr.tt.ttnn_mha import TTPETRMultiheadAttention


def input_preprocessing(x, N, C, H, W):
    x = ttnn.to_torch(x)
    x = torch.permute(x, (0, 3, 1, 2))
    x = x.reshape(N, C, H, W)
    return x


class TTFFN:
    def __init__(self, device, parameter):
        super().__init__()
        self.device = device
        # Weights and biases for the first linear layer
        self.l1_weight = parameter[0][0].weight
        self.l1_bias = parameter[0][0].bias
        # Weights and biases for the second linear layer
        self.l2_weight = parameter[1].weight
        self.l2_bias = parameter[1].bias

    def __call__(
        self,
        x,
    ):
        input = x

        # Preprocess and move first layer weights to device
        self.l1_weight = preprocess_linear_weight(self.l1_weight, dtype=ttnn.bfloat16)
        self.l1_bias = preprocess_linear_bias(self.l1_bias, dtype=ttnn.bfloat16)
        self.l1_weight = ttnn.to_device(self.l1_weight, self.device)
        self.l1_bias = ttnn.to_device(self.l1_bias, self.device)

        # First linear transformation + ReLU
        x = ttnn.linear(x, self.l1_weight, bias=self.l1_bias)
        x = ttnn.relu(x)

        # Preprocess and move second layer weights to device
        self.l2_weight = preprocess_linear_weight(self.l2_weight, dtype=ttnn.bfloat16)
        self.l2_bias = preprocess_linear_bias(self.l2_bias, dtype=ttnn.bfloat16)
        self.l2_weight = ttnn.to_device(self.l2_weight, self.device)
        self.l2_bias = ttnn.to_device(self.l2_bias, self.device)

        # Second linear transformation
        x = ttnn.linear(x, self.l2_weight, bias=self.l2_bias)

        # Residual connection
        return input + x


class TTPETRTransformerDecoderLayer(nn.Module):
    def __init__(self, device, parameter):
        super().__init__()
        self.mha = TTPETRMultiheadAttention(device, parameter.attentions[0])
        self.petr_mha = TTPETRMultiheadAttention(device, parameter.attentions[1])
        self.ffns = TTFFN(device, parameter.ffns[0].layers)

        self.norm1_weight = ttnn.from_torch(
            parameter.norms[0].weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.norm1_bias = ttnn.from_torch(
            parameter.norms[0].bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.norm2_weight = ttnn.from_torch(
            parameter.norms[1].weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.norm2_bias = ttnn.from_torch(
            parameter.norms[1].bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.norm3_weight = ttnn.from_torch(
            parameter.norms[2].weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.norm3_bias = ttnn.from_torch(
            parameter.norms[2].bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def __call__(self, device, query, key, value, key_pos, query_pos, key_padding_mask):
        x, weight = self.mha(query, query, query, query_pos=query_pos, key_pos=query_pos)
        x = ttnn.layer_norm(x, weight=self.norm1_weight, bias=self.norm1_bias)

        x, weight = self.petr_mha(
            x, key, value, query_pos=query_pos, key_pos=key_pos, key_padding_mask=key_padding_mask
        )
        x = ttnn.layer_norm(x, weight=self.norm2_weight, bias=self.norm2_bias)

        x = self.ffns(x)
        x = ttnn.layer_norm(x, weight=self.norm3_weight, bias=self.norm3_bias)

        return x


class TTPETRTransformerDecoder(nn.Module):
    def __init__(self, device, parameter):
        super().__init__()
        self.decoder0 = TTPETRTransformerDecoderLayer(device, parameter.decoder.module.layers[0])
        self.decoder1 = TTPETRTransformerDecoderLayer(device, parameter.decoder.module.layers[1])
        self.decoder2 = TTPETRTransformerDecoderLayer(device, parameter.decoder.module.layers[2])
        self.decoder3 = TTPETRTransformerDecoderLayer(device, parameter.decoder.module.layers[3])
        self.decoder4 = TTPETRTransformerDecoderLayer(device, parameter.decoder.module.layers[4])
        self.decoder5 = TTPETRTransformerDecoderLayer(device, parameter.decoder.module.layers[5])

        self.post_norm_weight = ttnn.from_torch(
            parameter.decoder.module.post_norm.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.post_norm_bias = ttnn.from_torch(
            parameter.decoder.module.post_norm.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def __call__(self, device, query, key, value, key_pos, query_pos, key_padding_mask):
        x = self.decoder0(device, query, key, value, key_pos, query_pos, key_padding_mask)
        x1 = ttnn.layer_norm(x, weight=self.post_norm_weight, bias=self.post_norm_bias)

        x = self.decoder1(device, x, key, value, key_pos, query_pos, key_padding_mask)
        x2 = ttnn.layer_norm(x, weight=self.post_norm_weight, bias=self.post_norm_bias)

        x = self.decoder2(device, x, key, value, key_pos, query_pos, key_padding_mask)
        x3 = ttnn.layer_norm(x, weight=self.post_norm_weight, bias=self.post_norm_bias)

        x = self.decoder3(device, x, key, value, key_pos, query_pos, key_padding_mask)
        x4 = ttnn.layer_norm(x, weight=self.post_norm_weight, bias=self.post_norm_bias)

        x = self.decoder4(device, x, key, value, key_pos, query_pos, key_padding_mask)
        x5 = ttnn.layer_norm(x, weight=self.post_norm_weight, bias=self.post_norm_bias)

        x = self.decoder5(device, x, key, value, key_pos, query_pos, key_padding_mask)
        x6 = ttnn.layer_norm(x, weight=self.post_norm_weight, bias=self.post_norm_bias)

        x1 = ttnn.reshape(x1, (1, x1.shape[0], x1.shape[1], x1.shape[2]))
        x2 = ttnn.reshape(x2, (1, x2.shape[0], x2.shape[1], x2.shape[2]))
        x3 = ttnn.reshape(x3, (1, x3.shape[0], x3.shape[1], x3.shape[2]))
        x4 = ttnn.reshape(x4, (1, x4.shape[0], x4.shape[1], x4.shape[2]))
        x5 = ttnn.reshape(x5, (1, x5.shape[0], x5.shape[1], x5.shape[2]))
        x6 = ttnn.reshape(x6, (1, x6.shape[0], x6.shape[1], x6.shape[2]))

        x = ttnn.concat([x1, x2, x3, x4, x5, x6], dim=0)
        return x


class TTPETRTransformer:
    def __init__(self, device, parameter):
        super().__init__()
        self.decoder = TTPETRTransformerDecoder(device, parameter)

    def __call__(
        self,
        device,
        x,
        mask,
        query_embed,
        pos_embed,
    ):
        bs, n, c, h, w = x.shape
        print(x.shape)
        # memory = ttnn.to_torch(x)
        memory = x
        memory = ttnn.permute(memory, (1, 3, 4, 0, 2))
        # memory = ttnn.from_torch(memory, dtype=ttnn.bfloat16, device=device)
        memory = ttnn.reshape(memory, (-1, bs, c))
        pos_embed = ttnn.permute(pos_embed, (1, 3, 4, 0, 2))
        pos_embed = ttnn.reshape(pos_embed, (-1, bs, c))

        query_embed = ttnn.reshape(query_embed, (query_embed.shape[0], 1, query_embed.shape[1]))
        query_embed = ttnn.repeat(query_embed, ttnn.Shape([1, bs, 1]))

        mask = ttnn.reshape(mask, (bs, -1))

        target = torch.zeros((query_embed.shape[0], query_embed.shape[1], query_embed.shape[2]))

        target = ttnn.from_torch(target, dtype=ttnn.bfloat16, device=device)

        out_dec = self.decoder(
            device,
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
        )

        out_dec = ttnn.permute(out_dec, (0, 2, 1, 3))
        memory = ttnn.reshape(memory, (n, h, w, bs, c))
        memory = ttnn.permute(memory, (3, 0, 4, 1, 2))

        return out_dec, memory
