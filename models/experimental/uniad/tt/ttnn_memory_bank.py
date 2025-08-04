# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.uniad.tt.ttnn_utils import Instances
from models.experimental.uniad.tt.ttnn_multi_head_attention import TtMultiheadAttention


class TtMemoryBank:
    def __init__(
        self,
        dim_in=256,
        hidden_dim=256,
        dim_out=256,
        fp_ratio=0.3,
        memory_bank_score_thresh=0,
        memory_bank_len=4,
        params=None,
        device=None,
        eps=1e-05,
        model_args=None,
    ):
        self.memory_bank_len = memory_bank_len
        self.device = device
        self.params = params
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.dim_out = dim_out
        self.eps = eps

        self.temporal_attn = TtMultiheadAttention(device, params.temporal_attn, embed_dim=dim_in, num_heads=8)

        self.memory_bank_score_thresh = memory_bank_score_thresh
        self.save_period = 3

    def update(self, track_instances, memory_bank_score_thresh):
        embed = track_instances.output_embedding[:, None]

        scores = track_instances.scores
        mem_padding_mask = track_instances.mem_padding_mask

        save_period = track_instances.save_period

        cond1 = ttnn.eq(save_period, 0)
        cond2 = ttnn.gt(scores, memory_bank_score_thresh)
        saved_idxes = ttnn.logical_and(cond1, cond2)

        mask = ttnn.gt(save_period, 0)

        decrement = ttnn.where(mask, save_period - 1, save_period)
        save_period = decrement

        save_period = ttnn.where(saved_idxes, self.save_period, save_period)

    def _forward_temporal_attn(self, track_instances):
        key_padding_mask = track_instances.mem_padding_mask

        valid_idxes = key_padding_mask[:, -1] == 0

        mask_tensor = ttnn.from_torch(valid_idxes, dtype=ttnn.uint32, device=self.device)
        mask_tensor = ttnn.unsqueeze(mask_tensor, 0)
        mask_tensor = ttnn.unsqueeze(mask_tensor, 0)
        mask_tensor = ttnn.unsqueeze(mask_tensor, 0)

        output_tensor = ttnn.nonzero(mask_tensor)
        output_tensor1 = ttnn.to_torch(output_tensor[0])
        output_tensor2 = ttnn.to_torch(output_tensor[1])

        no_of_non_zero_indices = output_tensor1[..., 0].item()
        indices_tensor = output_tensor2[:, :, :, :no_of_non_zero_indices]
        indices_tensor = ttnn.from_torch(indices_tensor, device=self.device)

        indices_tensor = ttnn.reshape(indices_tensor, (-1,))
        indices_tensor = ttnn.to_layout(indices_tensor, ttnn.TILE_LAYOUT)
        indices_tensor = ttnn.typecast(indices_tensor, dtype=ttnn.uint32)
        ttnn_output = ttnn.embedding(indices_tensor, track_instances.output_embedding)
        ttnn_output = ttnn.to_layout(ttnn_output, ttnn.TILE_LAYOUT)
        ttnn_output = ttnn.typecast(ttnn_output, dtype=ttnn.float32)
        ttnn_output = ttnn.to_torch(ttnn_output)
        embed = ttnn_output

        embed = ttnn.to_torch(track_instances.output_embedding)[valid_idxes]

        if len(embed) > 0:
            A, B, C = track_instances.mem_bank.shape
            track_instances.mem_bank = ttnn.reshape(track_instances.mem_bank, (A, B * C))
            ttnn_output = ttnn.embedding(indices_tensor, track_instances.mem_bank)
            track_instances.mem_bank = ttnn.reshape(track_instances.mem_bank, (A, B, C))
            ttnn_output = ttnn.to_layout(ttnn_output, ttnn.TILE_LAYOUT)
            ttnn_output = ttnn.typecast(ttnn_output, dtype=ttnn.bfloat16)
            ttnn_output = ttnn.reshape(ttnn_output, (ttnn_output.shape[0], B, C))
            # ttnn_output = ttnn.to_torch(ttnn_output)
            prev_embed = ttnn_output

            x_tensor = ttnn.from_torch(key_padding_mask, dtype=ttnn.bfloat16, device=self.device)

            ttnn_output = ttnn.embedding(indices_tensor, x_tensor)
            ttnn_output = ttnn.to_layout(ttnn_output, ttnn.TILE_LAYOUT)
            ttnn_output = ttnn.typecast(ttnn_output, dtype=ttnn.float32)
            ttnn_output = ttnn.to_torch(ttnn_output)
            key_padding_mask = ttnn_output
            embed = embed.unsqueeze(0)
            embed = ttnn.from_torch(embed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            prev_embed = ttnn.permute(prev_embed, (1, 0, 2))
            embed2 = self.temporal_attn(
                embed,
                prev_embed,
                prev_embed,
                key_padding_mask=key_padding_mask,
            )[
                0
            ][0]

            out = embed + embed2

            embed = ttnn.layer_norm(
                out,
                weight=self.params.temporal_norm1.weight,
                bias=self.params.temporal_norm1.bias,
                epsilon=self.eps,
            )

            x = ttnn.linear(embed, self.params.temporal_fc1.weight, bias=self.params.temporal_fc1.bias)
            x = ttnn.relu(x)
            embed2 = ttnn.linear(x, self.params.temporal_fc2.weight, bias=self.params.temporal_fc2.bias)

            embed = ttnn.layer_norm(
                embed + embed2,
                weight=self.params.temporal_norm2.weight,
                bias=self.params.temporal_norm2.bias,
                epsilon=self.eps,
            )
            track_instances.output_embedding = ttnn.to_torch(track_instances.output_embedding, dtype=torch.float32)
            embed = ttnn.to_torch(embed, dtype=torch.float32)
            track_instances.output_embedding = track_instances.output_embedding.clone()

            track_instances.output_embedding[valid_idxes] = embed

        return track_instances

    def forward(self, track_instances: Instances, update_bank=True) -> Instances:
        track_instances = self._forward_temporal_attn(track_instances)
        if update_bank:
            self.update(track_instances, self.memory_bank_score_thresh)
        return track_instances
