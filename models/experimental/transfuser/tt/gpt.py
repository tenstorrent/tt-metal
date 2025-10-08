# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.transfuser.tt.gpt_block import TTGptBlock


def generate_token_embeddings_tt(image_tensor, lidar_tensor, seq_len, n_embd):
    """
    Generate token embeddings from pre-flattened token format tensors.

    Args:
        image_tensor: (batch, 1, num_image_tokens, channels) - e.g., (1, 1, 110, 72)
        lidar_tensor: (batch, 1, num_lidar_tokens, channels) - e.g., (1, 1, 64, 72)
        seq_len: sequence length (should be 1)
        n_embd: embedding dimension (should be 72)

    Returns:
        token_embeddings: (batch, total_tokens, n_embd)
        Additional metadata for post-processing
    """
    print(f"{image_tensor.shape,lidar_tensor.shape=}")
    bz = image_tensor.shape[0]
    img_num_tokens = image_tensor.shape[2]  # 110
    img_c = image_tensor.shape[3]  # 72

    lidar_num_tokens = lidar_tensor.shape[2]  # 64
    lidar_c = lidar_tensor.shape[3]  # 72

    # Calculate original spatial dimensions
    # Assuming img_num_tokens = img_h * img_w = 5 * 22 = 110
    # Assuming lidar_num_tokens = lidar_h * lidar_w = 8 * 8 = 64
    img_h, img_w = 5, 22  # These should match img_vert_anchors, img_horz_anchors
    lidar_h, lidar_w = 8, 8  # These should match lidar_vert_anchors, lidar_horz_anchors

    # Reshape to remove the middle dimension
    # (1, 1, 110, 72) -> (1, 110, 72)
    image_tokens = ttnn.reshape(image_tensor, (bz, img_num_tokens, n_embd))

    # (1, 1, 64, 72) -> (1, 64, 72)
    lidar_tokens = ttnn.reshape(lidar_tensor, (bz, lidar_num_tokens, n_embd))

    # Concatenate image and lidar tokens along sequence dimension
    token_embeddings = ttnn.concat([image_tokens, lidar_tokens], dim=1)  # (1, 174, 72)

    return token_embeddings, bz, seq_len, img_h, img_w, lidar_h, lidar_w


def post_process_output_tt(
    x,
    bz,
    seq_len,
    img_vert_anchors,
    img_horz_anchors,
    lidar_vert_anchors,
    lidar_horz_anchors,
    n_embed,
    img_h,
    img_w,
    lidar_h,
    lidar_w,
):
    # Reshape to [bz, total_seq, n_embed]
    total_seq = seq_len * img_vert_anchors * img_horz_anchors + seq_len * lidar_vert_anchors * lidar_horz_anchors
    x = ttnn.reshape(x, (bz, total_seq, n_embed))

    # Split image and lidar tensors
    img_seq_len = seq_len * img_vert_anchors * img_horz_anchors

    # Slice image tensor
    image_tensor = x[:, :img_seq_len, :]
    image_tensor_out = ttnn.reshape(image_tensor, (bz * seq_len, -1, img_h, img_w))

    # Slice lidar tensor
    lidar_tensor = x[:, img_seq_len:, :]
    lidar_tensor_out = ttnn.reshape(lidar_tensor, (bz * seq_len, -1, lidar_h, lidar_w))

    return image_tensor_out, lidar_tensor_out


class TTGpt(LightweightModule):
    def __init__(
        self,
        device,
        parameters,
        n_head,
        n_layer,
        use_velocity,
        img_vert_anchors,
        img_horz_anchors,
        lidar_vert_anchors,
        lidar_horz_anchors,
        seq_len,
        n_embd,
        dropout_prob=0.0,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    ):
        self.dropout_prob = dropout_prob
        self.dropout_seed = 42
        self.device = device
        self.parameters = parameters
        self.n_head = n_head
        self.n_layer = n_layer
        self.use_velocity = use_velocity
        self.pos_emb = parameters["pos_emb"]

        if self.use_velocity:
            # Store velocity embedding weights and bias as TTNN tensors
            self.vel_emb_weight = parameters["vel_emb_weight"]
            self.vel_emb_bias = parameters["vel_emb_bias"]

        self.tt_blocks = []
        for i in range(n_layer):
            self.tt_blocks.append(
                TTGptBlock(device, parameters[f"blocks_{i}"], n_head, dtype=dtype, memory_config=memory_config)
            )
        self.dtype = dtype
        self.memory_config = memory_config
        self.img_vert_anchors = img_vert_anchors
        self.img_horz_anchors = img_horz_anchors
        self.lidar_vert_anchors = lidar_vert_anchors
        self.lidar_horz_anchors = lidar_horz_anchors
        self.seq_len = seq_len

    def __call__(self, tt_image_input, tt_lidar_input, velocity, n_embed):
        token_embeddings, bz, seq_len, img_h, img_w, lidar_h, lidar_w = generate_token_embeddings_tt(
            tt_image_input, tt_lidar_input, self.seq_len, n_embed
        )

        if self.use_velocity:
            # Convert velocity to TTNN if needed
            if isinstance(velocity, torch.Tensor):
                velocity = ttnn.from_torch(
                    velocity,
                    device=self.device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=self.dtype,
                    memory_config=self.memory_config,
                )

            # Apply linear transformation using ttnn.linear
            velocity_embeddings = ttnn.linear(
                velocity,
                self.vel_emb_weight,
                bias=self.vel_emb_bias,
                memory_config=self.memory_config,
                dtype=self.dtype,
            )

            # Add embeddings
            x = ttnn.add(self.pos_emb, token_embeddings)
            x = ttnn.add(x, velocity_embeddings)
        else:
            x = ttnn.add(self.pos_emb, token_embeddings)

        if self.dropout_prob > 0:
            x = ttnn.experimental.dropout(
                x, probability=self.dropout_prob, scale=1.0 / (1.0 - self.dropout_prob), seed=self.dropout_seed
            )

        # Continue with transformer blocks
        for i in range(self.n_layer):
            x = self.tt_blocks[i](x)

        x = ttnn.layer_norm(x, weight=self.parameters["ln_f_weight"], bias=self.parameters["ln_f_bias"])
        tt_image_output, tt_lidar_output = post_process_output_tt(
            x,
            bz,
            seq_len,
            self.img_vert_anchors,
            self.img_horz_anchors,
            self.lidar_vert_anchors,
            self.lidar_horz_anchors,
            n_embed,
            img_h,
            img_w,
            lidar_h,
            lidar_w,
        )
        return tt_image_output, tt_lidar_output
