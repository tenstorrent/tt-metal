# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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
        self.max_his_length = memory_bank_len
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
        track_instances.output_embedding = ttnn.to_torch(track_instances.output_embedding)
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
        saved_idxes = ttnn.to_torch((saved_idxes)).to(dtype=torch.bool)
        save_period = ttnn.to_torch(save_period)
        save_period[
            saved_idxes
        ] = (
            self.save_period
        )  # TODO Raised issue for this operation - <https://github.com/tenstorrent/tt-metal/issues/15553>
        track_instances.save_period = save_period
        track_instances.save_period = ttnn.from_torch(
            track_instances.save_period, dtype=ttnn.bfloat16, device=self.device
        )
        saved_embed = embed[saved_idxes]
        if len(saved_embed) > 0:
            track_instances.mem_bank = ttnn.to_torch(track_instances.mem_bank)
            prev_embed = track_instances.mem_bank[saved_idxes]
            saved_embed = ttnn.from_torch(saved_embed, dtype=ttnn.bfloat16, device=self.device)
            saved_embed = ttnn.to_layout(saved_embed, layout=ttnn.TILE_LAYOUT)
            save_embed = ttnn.linear(
                saved_embed,
                self.params.save_proj.weight,
                bias=self.params.save_proj.bias,
            )
            saved_embed = ttnn.to_torch(saved_embed)
            tensor1 = mem_padding_mask[
                saved_idxes, 1:
            ]  # TODO Raised issue for this operation - <https://github.com/tenstorrent/tt-metal/issues/15553>
            tensor1 = ttnn.from_torch(tensor1, dtype=ttnn.bfloat16, device=self.device)
            tensor2 = torch.zeros((len(saved_embed), 1), dtype=torch.bool)
            tensor2 = ttnn.from_torch(tensor2, dtype=ttnn.bfloat16, device=self.device)
            out = ttnn.concat(
                [
                    tensor1,
                    tensor2,
                ],
                dim=1,
            )
            mem_padding_mask[saved_idxes] = ttnn.to_torch(out).to(dtype=torch.bool)
            track_instances.mem_bank = track_instances.mem_bank.clone()
            prev_embed = ttnn.from_torch(prev_embed, dtype=ttnn.bfloat16, device=self.device)
            prev_embed = ttnn.to_layout(prev_embed, layout=ttnn.TILE_LAYOUT)
            out = ttnn.concat([prev_embed[:, 1:], save_embed], dim=1)
            track_instances.mem_bank[saved_idxes] = ttnn.to_torch(out)
            track_instances.mem_bank = ttnn.from_torch(
                track_instances.mem_bank, dtype=ttnn.bfloat16, device=self.device
            )
        track_instances.output_embedding = ttnn.from_torch(
            track_instances.output_embedding, dtype=ttnn.bfloat16, device=self.device
        )

    def _forward_temporal_attn(self, track_instances):
        key_padding_mask = track_instances.mem_padding_mask

        valid_idxes = key_padding_mask[:, -1] == 0

        return track_instances

    def __call__(self, track_instances: Instances, update_bank=True) -> Instances:
        track_instances = self._forward_temporal_attn(track_instances)
        if update_bank:
            self.update(track_instances, self.memory_bank_score_thresh)
        return track_instances
