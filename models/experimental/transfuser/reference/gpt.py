# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from models.experimental.transfuser.reference.gpt_block import Block


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(
        self,
        n_embd,
        n_head,
        block_exp,
        n_layer,
        img_vert_anchors,
        img_horz_anchors,
        lidar_vert_anchors,
        lidar_horz_anchors,
        seq_len,
        embd_pdrop,
        attn_pdrop,
        resid_pdrop,
        config,
        use_velocity=True,
    ):
        super().__init__()
        self.n_embd = n_embd
        # We currently only support seq len 1
        self.seq_len = 1

        self.img_vert_anchors = img_vert_anchors
        self.img_horz_anchors = img_horz_anchors
        self.lidar_vert_anchors = lidar_vert_anchors
        self.lidar_horz_anchors = lidar_horz_anchors
        self.config = config

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                self.seq_len * img_vert_anchors * img_horz_anchors
                + self.seq_len * lidar_vert_anchors * lidar_horz_anchors,
                n_embd,
            )
        )

        # velocity embedding
        self.use_velocity = use_velocity
        if use_velocity == True:
            self.vel_emb = nn.Linear(self.seq_len, n_embd)

        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)]
        )

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = self.seq_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.config.gpt_linear_layer_init_mean, std=self.config.gpt_linear_layer_init_std
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_tensor, lidar_tensor, velocity):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            velocity (tensor): ego-velocity
        """

        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]

        assert self.seq_len == 1
        image_tensor = (
            image_tensor.view(bz, self.seq_len, -1, img_h, img_w)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(bz, -1, self.n_embd)
        )
        lidar_tensor = (
            lidar_tensor.view(bz, self.seq_len, -1, lidar_h, lidar_w)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(bz, -1, self.n_embd)
        )

        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

        # project velocity to n_embed
        if self.use_velocity == True:
            velocity_embeddings = self.vel_emb(velocity)  # (B, C)
            # add (learnable) positional embedding and velocity embedding for all tokens
            x = self.drop(self.pos_emb + token_embeddings + velocity_embeddings.unsqueeze(1))  # (B, an * T, C)
        else:
            x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)

        x = x.view(
            bz,
            self.seq_len * self.img_vert_anchors * self.img_horz_anchors
            + self.seq_len * self.lidar_vert_anchors * self.lidar_horz_anchors,
            self.n_embd,
        )

        image_tensor_out = (
            x[:, : self.seq_len * self.img_vert_anchors * self.img_horz_anchors, :]
            .contiguous()
            .view(bz * self.seq_len, -1, img_h, img_w)
        )
        lidar_tensor_out = (
            x[:, self.seq_len * self.img_vert_anchors * self.img_horz_anchors :, :]
            .contiguous()
            .view(bz * self.seq_len, -1, lidar_h, lidar_w)
        )

        return image_tensor_out, lidar_tensor_out
