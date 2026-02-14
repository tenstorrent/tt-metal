# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def ttnn_conv2d(
    x,
    weight,
    bias,
    device,
    in_channels,
    out_channels,
    batch_size,
    input_height,
    input_width,
    kernel_size=(3, 3),
    stride=(1, 1),
    padding=(1, 1),
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    # Functional wrapper for ttnn.conv2d with memory_config support
    return ttnn.conv2d(
        input_tensor=x,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=(1, 1),
        groups=1,
        device=device,
        return_output_dim=True,
        memory_config=memory_config,
    )


def ttnn_upsample(x, scale_factor):
    # Bilinear upsample is common in DPT
    # x is (B, C, H, W)
    return ttnn.upsample(x, scale_factor=scale_factor)


# ----------------------------------------------------------------------------
# DPT Components (Neck & Head)
# ----------------------------------------------------------------------------


class TtDPTReassembleLayer:
    def __init__(self, parameters, read_idx, config, device):
        self.parameters = parameters
        self.read_idx = read_idx
        self.config = config
        self.device = device
        self.sharded_config = config["l1_sharded_config"]

    def __call__(self, x):
        # x is (B, Seq, Hidden) - Padded to 2048 (64 tiles)
        batch_size, seq_len, hidden_size = x.shape
        grid_h, grid_w = 37, 37  # 518/14 = 37
        patch_count_all = grid_h * grid_w  # 1369

        # 1. Remove CLS tokens and Padding: (B, 2048, Hidden) -> (B, 1369, Hidden)
        # CLS is 32 tokens (1 tile). Patches start at index 32.
        # This slice is core-aligned in our 64-core grid!
        cls_size = 32
        x = ttnn.slice(x, (0, cls_size, 0), (batch_size, cls_size + patch_count_all, hidden_size))

        # 2. Reshape to Grid: (B, 1369, Hidden) -> (B, Hidden, 37, 37)
        # Move to RM for reshape/permute
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (batch_size, grid_h, grid_w, hidden_size))
        x = ttnn.permute(x, (0, 3, 1, 2))  # (B, Hidden, H, W)

        # 3. Projection: (B, Hidden, 37, 37) -> (B, 256, 37, 37)
        x = ttnn.permute(x, (0, 2, 3, 1))  # (B, H, W, Hidden)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.linear(
            x,
            self.parameters.projection.weight,
            bias=self.parameters.projection.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        x = ttnn.permute(x, (0, 3, 1, 2))  # (B, 256, H, W)

        # 4. Resample (Resize)
        if hasattr(self.parameters, "resize"):
            if self.read_idx == 0:
                x = ttnn_upsample(x, scale_factor=4)
            elif self.read_idx == 1:
                x = ttnn_upsample(x, scale_factor=2)
            elif self.read_idx == 3:
                # Downsample
                x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
                x, _ = ttnn_conv2d(
                    x_rm,
                    self.parameters.resize.weight,
                    self.parameters.resize.bias,
                    self.device,
                    256,
                    256,
                    batch_size,
                    grid_h,
                    grid_w,
                    stride=(2, 2),
                )
                ttnn.deallocate(x_rm)

        return x


class TtDPTFusionStage:
    def __init__(self, parameters, device):
        self.parameters = parameters
        self.device = device

    def _residual_block(self, x, params):
        residual = x
        # Force L1 Sharded for convolutions if possible
        batch_size, channels, h, w = x.shape
        # Move to RM for conv
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x, [h, w] = ttnn_conv2d(
            x,
            params.convolution1.weight,
            params.convolution1.bias,
            self.device,
            channels,
            channels,
            batch_size,
            h,
            w,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        x = ttnn.relu(x)
        x, [h, w] = ttnn_conv2d(
            x,
            params.convolution2.weight,
            params.convolution2.bias,
            self.device,
            channels,
            channels,
            batch_size,
            h,
            w,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Move back to tile for addition
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        res = ttnn.add_inplace(residual, x)
        ttnn.deallocate(x)
        return res

    def __call__(self, features):
        x = None
        for i in range(3, -1, -1):
            params = self.parameters.layers[i]
            feat_i = features[i]

            # Project current scale
            # feat_i is already in TILE_LAYOUT from encoder out
            feat_i = ttnn.permute(feat_i, (0, 2, 3, 1))
            feat_i = ttnn.linear(
                feat_i, params.projection.weight, bias=params.projection.bias, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            feat_i = ttnn.permute(feat_i, (0, 3, 1, 2))

            if i < 3:
                x_upsampled = ttnn_upsample(x, scale_factor=2)
                ttnn.deallocate(x)
                x = ttnn.add_inplace(x_upsampled, feat_i)
                ttnn.deallocate(feat_i)
            else:
                x = feat_i

            x = self._residual_block(x, params.residual_layer1)
            x = self._residual_block(x, params.residual_layer2)

        return x


class TtDPTHead:
    def __init__(self, parameters, device):
        self.parameters = parameters
        self.device = device

    def __call__(self, x):
        batch_size, channels, h, w = x.shape
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        # Conv 1
        x, [h, w] = ttnn_conv2d(
            x,
            self.parameters.conv1.weight,
            self.parameters.conv1.bias,
            self.device,
            channels,
            256,
            batch_size,
            h,
            w,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        x = ttnn.relu(x)

        # Upsample
        x = ttnn_upsample(x, scale_factor=2)
        h, w = h * 2, w * 2

        # Conv 2
        x, [h, w] = ttnn_conv2d(
            x,
            self.parameters.conv2.weight,
            self.parameters.conv2.bias,
            self.device,
            256,
            128,
            batch_size,
            h,
            w,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        x = ttnn.relu(x)

        # Conv 3 (Final Depth)
        x, [h, w] = ttnn_conv2d(
            x,
            self.parameters.conv3.weight,
            self.parameters.conv3.bias,
            self.device,
            128,
            1,
            batch_size,
            h,
            w,
            kernel_size=(1, 1),
            padding=(0, 0),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return x


# ----------------------------------------------------------------------------
# ViT Backbone (ttnn Implementation)
# ----------------------------------------------------------------------------


def vit_patch_embeddings(config, pixel_values, parameters, device):
    # pixel_values: (B, C, H, W)
    batch_size, img_c, img_h, img_w = pixel_values.shape
    patch_size = 14
    patch_count = img_h // patch_size  # 37
    patch_count_all = patch_count * patch_count  # 1369

    # 1. Patchify
    x = ttnn.to_layout(pixel_values, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (batch_size, img_c, patch_count, patch_size, patch_count, patch_size))
    x = ttnn.permute(x, (0, 2, 4, 3, 5, 1))
    x = ttnn.reshape(x, (batch_size, patch_count_all, patch_size * patch_size * img_c))

    # 2. Project
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    # Pad to match seqL_padded and projection width
    # In Stage 3, we target 2048 tokens total (including CLS).
    # Let's pad patches to 2016 (63 tiles) and then concat 32 tokens of CLS = 2048.
    patch_seq_padded = 2016
    x = ttnn.pad(x, (batch_size, patch_seq_padded, 608), (0, 0, 0), 0)

    x = ttnn.linear(
        x,
        parameters["projection"]["weight"],
        bias=parameters["projection"]["bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    return x  # (B, 2016, 1024)


def vit_embeddings(config, pixel_values, parameters, device):
    # 1. Patch Embeddings
    patch_embeddings = vit_patch_embeddings(config, pixel_values, parameters["patch_embeddings"], device)

    # 2. Concatenate CLS Token
    # CLS token is padded to 32 tokens to reach 2048 total (64 tiles) for 8x8 grid alignment
    cls_token = ttnn.pad(parameters["cls_token"], (1, 32, 1024), (0, 0, 0), 0)
    cls_token = ttnn.to_layout(cls_token, ttnn.TILE_LAYOUT)

    embedding_output = ttnn.concat([cls_token, patch_embeddings], dim=1)
    ttnn.deallocate(patch_embeddings)
    ttnn.deallocate(cls_token)

    # 3. Add Position Embeddings
    # pos_embeds is already padded to 2048 in preprocessor
    pos_embeds = ttnn.to_layout(parameters["position_embeddings"], ttnn.TILE_LAYOUT)

    embedding_output = ttnn.add_inplace(embedding_output, pos_embeds)
    ttnn.deallocate(pos_embeds)

    return embedding_output


def vit_layer(hidden_states, parameters, config):
    num_heads = config["num_attention_heads"]
    hidden_size = config["hidden_size"]
    head_size = hidden_size // num_heads
    sharded_config = config["l1_sharded_config"]

    # 1. Layernorm 1 (Sharded)
    ln1 = ttnn.layer_norm(
        hidden_states,
        weight=parameters["layernorm_before"]["weight"],
        bias=parameters["layernorm_before"]["bias"],
        memory_config=sharded_config,
        program_config=config["program_configs"]["layernorm_before_program_config"],
    )

    # 2. Fused QKV Matmul
    qkv = ttnn.linear(
        ln1,
        parameters["attention"]["qkv"]["weight"],
        bias=parameters["attention"]["qkv"]["bias"],
        memory_config=sharded_config,
        program_config=config["program_configs"]["qkv_matmul_program_config"],
    )

    # 3. Split Heads
    (query, key, value) = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    ttnn.deallocate(qkv)

    # 4. Attention Scores: Q * K^T
    attn_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    attn_scores = ttnn.mul(attn_scores, 1 / (head_size**0.5))

    # Use attention_softmax_ for in-place optimization
    attn_probs = ttnn.transformer.attention_softmax_(
        attn_scores,
        head_size=head_size,
    )

    # 5. Context Layer: Attn * V
    context_layer = ttnn.matmul(
        attn_probs,
        value,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )
    ttnn.deallocate(attn_probs)
    ttnn.deallocate(value)

    # 6. Merge Heads
    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=sharded_config,
    )

    # 7. Output Projection
    attn_out = ttnn.linear(
        context_layer,
        parameters["attention"]["output"]["dense"]["weight"],
        bias=parameters["attention"]["output"]["dense"]["bias"],
        memory_config=sharded_config,
    )
    ttnn.deallocate(context_layer)
    ttnn.deallocate(ln1)

    # 8. Residual 1
    hidden_states = ttnn.add_inplace(hidden_states, attn_out)
    ttnn.deallocate(attn_out)

    # 9. Layernorm 2 (Sharded)
    ln2 = ttnn.layer_norm(
        hidden_states,
        weight=parameters["layernorm_after"]["weight"],
        bias=parameters["layernorm_after"]["bias"],
        memory_config=sharded_config,
    )

    # 10. MLP: FC1 -> GELU -> FC2
    # Fusing FC1 and GELU
    mlp_out = ttnn.linear(
        ln2,
        parameters["intermediate"]["dense"]["weight"],
        bias=parameters["intermediate"]["dense"]["bias"],
        memory_config=sharded_config,
    )
    mlp_out = ttnn.gelu(mlp_out)

    mlp_out = ttnn.linear(
        mlp_out,
        parameters["output"]["dense"]["weight"],
        bias=parameters["output"]["dense"]["bias"],
        memory_config=sharded_config,
    )
    ttnn.deallocate(ln2)

    # 11. Residual 2
    hidden_states = ttnn.add_inplace(hidden_states, mlp_out)
    ttnn.deallocate(mlp_out)

    return hidden_states


# ----------------------------------------------------------------------------
# Main Model: Depth Anything V2
# ----------------------------------------------------------------------------


class TtDepthAnythingV2:
    def __init__(self, config, parameters, device):
        # Initialize optimized config
        self.config = get_model_config(1, device)
        self.device = device
        self.parameters = self._move_to_device(parameters, device)
        self.reassemble = [
            TtDPTReassembleLayer(self.parameters["neck"]["reassemble"][i], i, self.config, device) for i in range(4)
        ]
        self.fusion = TtDPTFusionStage(self.parameters["neck"]["fusion"], device)
        self.head = TtDPTHead(self.parameters["head"], device)

    def _move_to_device(self, params, device):
        if isinstance(params, ttnn.Tensor):
            return ttnn.to_device(params, device)
        elif isinstance(params, dict):
            return {k: self._move_to_device(v, device) for k, v in params.items()}
        elif isinstance(params, list):
            return [self._move_to_device(v, device) for v in params]
        else:
            return params

    def __call__(self, pixel_values):
        # 1. Embeddings
        hidden_states = vit_embeddings(
            self.config, pixel_values, self.parameters["backbone"]["embeddings"], self.device
        )

        # 2. Shard transition for encoder
        hidden_states = ttnn.to_memory_config(hidden_states, memory_config=self.config["l1_sharded_config"])

        # 3. Encoder (ViT-Large: 24 layers)
        features = []
        out_indices = [5, 11, 17, 23]

        for i in range(24):
            layer_params = self.parameters["backbone"]["encoder"]["layer"][i]
            hidden_states = vit_layer(hidden_states, layer_params, self.config)

            if i in out_indices:
                # Store features in DRAM for the neck
                features.append(ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG))

        # Final backbone norm
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.parameters["backbone"]["layernorm"]["weight"],
            bias=self.parameters["backbone"]["layernorm"]["bias"],
            memory_config=self.config["l1_sharded_config"],
            program_config=self.config["program_configs"]["layernorm_before_program_config"],
        )

        # We don't actually need hidden_states anymore, features[3] is the last layer output (post-norm usually)
        # But DPT uses features[3] which WE already stored.
        # Wait, usually features[3] is the output of the 24th layer *after* the final norm.
        # Let's replace the last feature with the post-norm version.
        ttnn.deallocate(features[-1])
        features[-1] = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(hidden_states)

        # 4. Neck (DPT Reassemble & Fusion)
        reassembled_features = [self.reassemble[i](features[i]) for i in range(4)]
        # Deallocate raw features
        for f in features:
            ttnn.deallocate(f)

        fused_feature = self.fusion(reassembled_features)
        # Deallocate reassembled features
        for f in reassembled_features:
            ttnn.deallocate(f)

        # 5. Head
        output = self.head(fused_feature)
        ttnn.deallocate(fused_feature)

        # Interleaved for output
        return ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)


def get_model_config(batch_size, device):
    if device is not None:
        core_grid = device.compute_with_storage_grid_size()
    else:
        core_grid = ttnn.CoreGrid(y=8, x=8)

    # Stage 3: Full 8x8 grid (64 cores)
    core_grid_8x8 = ttnn.CoreGrid(y=8, x=8)

    # Hidden Size: 1024, Heads: 16
    # 2048 padded tokens (64 tiles) to match 8x8 grid perfectly (1 tile per core)
    seqL_padded = 2048
    seqL_t = seqL_padded // 32  # 64
    dim_t = 1024 // 32  # 32

    # Blocks for 8x8 grid
    block_h = seqL_t // 8  # 8 tiles per row/column
    block_w = dim_t // 8  # 4 tiles per column

    program_configs = {
        "layernorm_before_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            subblock_w=block_w,
            block_h=block_h,
            block_w=block_w,
            inplace=False,
        ),
        "qkv_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=block_w,
            out_subblock_h=1,
            out_subblock_w=block_w,
            per_core_M=block_h,
            per_core_N=3 * block_w,
            transpose_mcast=False,
            fused_activation=None,
        ),
    }

    return {
        "num_attention_heads": 16,
        "hidden_size": 1024,
        "core_grid": core_grid,
        "core_grid_8x8": core_grid_8x8,
        "program_configs": program_configs,
        "l1_sharded_config": ttnn.create_sharded_memory_config(
            (1, seqL_padded, 1024),
            core_grid=core_grid_8x8,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        "l1_height_sharded_config": ttnn.create_sharded_memory_config(
            (1, seqL_padded, 1024),
            core_grid=core_grid_8x8,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    }


def custom_preprocessor(torch_model, name):
    parameters = {}

    # Helper to convert a single weight
    def convert(tensor, dtype=ttnn.bfloat16):
        return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    # helper for ROW_MAJOR tensors (bias, tokens, etc)
    def convert_rm(tensor, dtype=ttnn.bfloat16):
        return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    # 1. Backbone
    # Reshape patch projection weight: (1024, 3, 14, 14) -> (3*14*14, 1024) = (588, 1024)
    # 588 is not a multiple of 32, so we pad it to 608 (19 tiles)
    patch_weight = torch_model.backbone.embeddings.patch_embeddings.projection.weight
    patch_weight = patch_weight.permute(1, 2, 3, 0).reshape(-1, 1024)
    patch_weight_padded = torch.nn.functional.pad(patch_weight, (0, 0, 0, 608 - patch_weight.shape[0]))

    parameters["backbone"] = {
        "embeddings": {
            "patch_embeddings": {
                "projection": {
                    "weight": convert(patch_weight_padded, dtype=ttnn.bfloat8_b),
                    "bias": convert_rm(torch_model.backbone.embeddings.patch_embeddings.projection.bias),
                }
            },
            # Pre-pad CLS and Position Embeddings to match 32-multiple CLS padding and final 2048 seq length
            "cls_token": convert_rm(torch.nn.functional.pad(torch_model.backbone.embeddings.cls_token, (0, 0, 0, 31))),
            "position_embeddings": convert_rm(
                torch.nn.functional.pad(
                    torch_model.backbone.embeddings.position_embeddings,
                    (0, 0, 0, 2048 - torch_model.backbone.embeddings.position_embeddings.shape[1]),
                )
            ),
        },
        "encoder": {"layer": []},
        "layernorm": {
            "weight": convert_rm(torch_model.backbone.layernorm.weight),
            "bias": convert_rm(torch_model.backbone.layernorm.bias),
        },
    }

    # Encoder Layers
    for i, layer in enumerate(torch_model.backbone.encoder.layer):
        # QKV Fusion
        q_w = layer.attention.attention.query.weight
        k_w = layer.attention.attention.key.weight
        v_w = layer.attention.attention.value.weight
        qkv_weight = torch.cat([q_w, k_w, v_w], dim=0).transpose(0, 1)  # (1024, 3072) -> (1024, 3072)

        q_b = layer.attention.attention.query.bias
        k_b = layer.attention.attention.key.bias
        v_b = layer.attention.attention.value.bias
        qkv_bias = torch.cat([q_b, k_b, v_b], dim=0).unsqueeze(0)  # (1, 3072)

        layer_params = {
            "layernorm_before": {"weight": convert_rm(layer.norm1.weight), "bias": convert_rm(layer.norm1.bias)},
            "attention": {
                "qkv": {"weight": convert(qkv_weight, dtype=ttnn.bfloat8_b), "bias": convert(qkv_bias)},
                "output": {
                    "dense": {
                        "weight": convert(layer.attention.output.dense.weight.transpose(0, 1), dtype=ttnn.bfloat8_b),
                        "bias": convert_rm(layer.attention.output.dense.bias),
                    }
                },
            },
            "layernorm_after": {"weight": convert_rm(layer.norm2.weight), "bias": convert_rm(layer.norm2.bias)},
            "intermediate": {
                "dense": {
                    "weight": convert(layer.mlp.fc1.weight.transpose(0, 1), dtype=ttnn.bfloat8_b),
                    "bias": convert_rm(layer.mlp.fc1.bias),
                }
            },
            "output": {
                "dense": {
                    "weight": convert(layer.mlp.fc2.weight.transpose(0, 1), dtype=ttnn.bfloat8_b),
                    "bias": convert_rm(layer.mlp.fc2.bias),
                }
            },
        }
        parameters["backbone"]["encoder"]["layer"].append(layer_params)

    # 2. Neck (DPT Reassemble & Fusion)
    parameters["neck"] = {"reassemble": [], "fusion": {"layers": []}}

    for i, layer in enumerate(torch_model.neck.reassemble_stage.layers):
        p = {
            "projection": {
                "weight": convert(layer.projection.weight.permute(1, 2, 3, 0).reshape(-1, 256), dtype=ttnn.bfloat8_b),
                "bias": convert_rm(layer.projection.bias),
            }
        }

        if hasattr(layer, "resize") and hasattr(layer.resize, "weight"):
            # Handle Conv2d or ConvTranspose2d weights for resize
            p["resize"] = {
                "weight": convert_rm(layer.resize.weight, dtype=ttnn.bfloat8_b),
                "bias": convert_rm(layer.resize.bias),
            }
        parameters["neck"]["reassemble"].append(p)

    for i, layer in enumerate(torch_model.neck.fusion_stage.layers):
        layer_params = {
            "projection": {
                "weight": convert(layer.projection.weight.permute(1, 2, 3, 0).reshape(-1, 256), dtype=ttnn.bfloat8_b),
                "bias": convert_rm(layer.projection.bias),
            },
            "residual_layer1": {
                "convolution1": {
                    "weight": convert_rm(layer.residual_layer1.convolution1.weight, dtype=ttnn.bfloat8_b),
                    "bias": convert_rm(layer.residual_layer1.convolution1.bias),
                },
                "convolution2": {
                    "weight": convert_rm(layer.residual_layer1.convolution2.weight, dtype=ttnn.bfloat8_b),
                    "bias": convert_rm(layer.residual_layer1.convolution2.bias),
                },
            },
            "residual_layer2": {
                "convolution1": {
                    "weight": convert_rm(layer.residual_layer2.convolution1.weight, dtype=ttnn.bfloat8_b),
                    "bias": convert_rm(layer.residual_layer2.convolution1.bias),
                },
                "convolution2": {
                    "weight": convert_rm(layer.residual_layer2.convolution2.weight, dtype=ttnn.bfloat8_b),
                    "bias": convert_rm(layer.residual_layer2.convolution2.bias),
                },
            },
        }
        parameters["neck"]["fusion"]["layers"].append(layer_params)

    # 3. Head
    parameters["head"] = {
        "conv1": {
            "weight": convert_rm(torch_model.head.conv1.weight, dtype=ttnn.bfloat8_b),
            "bias": convert_rm(torch_model.head.conv1.bias),
        },
        "conv2": {
            "weight": convert_rm(torch_model.head.conv2.weight, dtype=ttnn.bfloat8_b),
            "bias": convert_rm(torch_model.head.conv2.bias),
        },
        "conv3": {
            "weight": convert_rm(torch_model.head.conv3.weight, dtype=ttnn.bfloat8_b),
            "bias": convert_rm(torch_model.head.conv3.bias),
        },
    }

    return parameters
