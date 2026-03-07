# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


# ============================================================================
# Helpers
# ============================================================================


def _dram_tile(t):
    """Normalize a tensor to TILE_LAYOUT + DRAM_MEMORY_CONFIG.
    Must be called on both operands before ttnn.add() to avoid
    broadcast / shard mismatches."""
    if t.layout != ttnn.TILE_LAYOUT:
        t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
    if t.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        t = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
    return t


class DictNamespace:
    """Wrap a plain dict so that keys are accessible via dot-notation.

    custom_preprocessor returns nested dicts; the model classes access
    weights/biases via attribute access (e.g. params.projection.weight),
    so every dict returned from the preprocessor must be wrapped here.
    """

    def __init__(self, d):
        self._d = d

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(f"DictNamespace has no attribute '{name}'")

    # Allow dict-style access as well so that both styles work.
    def __getitem__(self, key):
        return self._d[key]

    def __contains__(self, key):
        return key in self._d

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)


# ============================================================================
# Conv2d wrapper
# ============================================================================


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
    """Call ttnn.conv2d with the standard (B, C, H, W) -> (B, C, H, W) convention.

    ttnn.conv2d expects (B, H, W, C) internally; this function handles all
    required transposes and reshapes so the caller can stay in PyTorch layout.

    Weight must already be bfloat16 ROW_MAJOR -- never bfloat8_b for conv weights
    because bfloat8_b requires TILE_LAYOUT which is invalid for conv kernels.
    """

    # ---- weight: ensure 4-D [out_ch, in_ch, kH, kW] ----------------------
    if len(weight.shape) == 2:
        kH, kW = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
        weight = ttnn.reshape(weight, (out_channels, in_channels, kH, kW))

    # ---- bias: ensure [1, 1, 1, out_ch] ------------------------------------
    if bias is not None:
        if len(bias.shape) == 1:
            bias = ttnn.reshape(bias, (1, 1, 1, bias.shape[0]))
        elif len(bias.shape) == 2:
            bias = ttnn.reshape(bias, (1, 1, bias.shape[0], bias.shape[1]))

    # ---- input: (B, C, H, W) -> (B, H, W, C) --------------------------------
    x = ttnn.transpose(x, -3, -1)   # (B, C, W, H)
    x = ttnn.transpose(x, -2, -1)   # (B, H, W, C)

    out_tensor, [out_h, out_w] = ttnn.conv2d(
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

    # ttnn returns (1, 1, B*out_h*out_w, out_c); reshape then transpose back.
    out_tensor = ttnn.reshape(out_tensor, (batch_size, out_h, out_w, out_channels))
    out_tensor = ttnn.transpose(out_tensor, -3, -1)   # (B, out_c, out_w, out_h)
    out_tensor = ttnn.transpose(out_tensor, -2, -1)   # (B, out_c, out_h, out_w)

    return out_tensor, [out_h, out_w]


def ttnn_upsample(x, scale_factor):
    """Nearest-neighbour (or bilinear) upsample.  x is (B, C, H, W)."""
    return ttnn.upsample(x, scale_factor=scale_factor)


# ============================================================================
# DPT Neck -- Reassemble stage
# ============================================================================


class TtDPTReassembleLayer:
    """Project ViT features back to spatial tensors and optionally resample.

    neck_hidden_sizes = [256, 512, 1024, 1024]  (per HuggingFace config)
    reassemble_factors = [4, 2, 1, 0.5]
    read_idx 0 -> upsample x4
    read_idx 1 -> upsample x2
    read_idx 2 -> identity (no resize)
    read_idx 3 -> stride-2 conv (downsample)
    """

    def __init__(self, parameters, read_idx, config, device, feature_size=256):
        self.parameters = parameters
        self.read_idx = read_idx
        self.config = config
        self.device = device
        self.feature_size = feature_size  # per-layer output channels

    def __call__(self, x):
        # x: (B, seqL_padded, 1024) in TILE layout
        batch_size = x.shape[0]
        grid_h = grid_w = 37          # 518 / 14 = 37
        patch_count_all = grid_h * grid_w  # 1369
        feat = self.feature_size

        # 1. Linear projection: 1024 -> feat
        x = ttnn.linear(
            x,
            self.parameters.projection.weight,
            bias=self.parameters.projection.bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # 2. Drop the CLS region (padded to 32 tokens) and keep patch tokens.
        #    Result: (B, 1369, feat)
        cls_size = 32
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.slice(x, (0, cls_size, 0), (batch_size, cls_size + patch_count_all, feat))

        # 3. Reshape to spatial tensor (B, H, W, C) then -> (B, C, H, W)
        x = ttnn.reshape(x, (batch_size, grid_h, grid_w, feat))
        x = ttnn.transpose(x, -3, -1)   # (B, feat, grid_w, grid_h)
        x = ttnn.transpose(x, -2, -1)   # (B, feat, grid_h, grid_w)

        # 4. Resample based on reassemble_factors = [4, 2, 1, 0.5]
        if self.read_idx == 0:
            x = ttnn_upsample(x, scale_factor=4)
            grid_h *= 4
            grid_w *= 4
        elif self.read_idx == 1:
            x = ttnn_upsample(x, scale_factor=2)
            grid_h *= 2
            grid_w *= 2
        elif self.read_idx == 2:
            pass  # identity -- no resize
        elif self.read_idx == 3:
            # Stride-2 conv to halve spatial resolution
            x, [grid_h, grid_w] = ttnn_conv2d(
                x,
                self.parameters.resize.weight,
                self.parameters.resize.bias,
                self.device,
                feat, feat,
                batch_size, grid_h, grid_w,
                stride=(2, 2),
            )

        return x   # (B, feat, H', W')


# ============================================================================
# DPT Neck -- Fusion stage
# ============================================================================


class TtDPTFusionStage:
    """Top-down fusion of the four reassembled feature maps.

    Iterates from the deepest (index 3) to the shallowest (index 0):
    upsample the running feature, add the projected scale feature, then
    apply two residual conv blocks.

    Each scale goes through two linear ops:
      1. neck_conv:   C_i -> 256  (the normalization from DPTNeck.convs)
      2. projection:  256 -> 256  (the fusion layer's own 1x1 conv)
    Without step 1, the matmul crashes because C_i=1024 != weight_rows=256.
    """

    def __init__(self, parameters, device):
        self.parameters = parameters
        self.device = device

    # ------------------------------------------------------------------
    def _residual_block(self, x, params):
        """Two 3x3 convolutions with a skip connection."""
        residual = x
        batch_size, channels, h, w = x.shape

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        x, [h, w] = ttnn_conv2d(
            x,
            params.convolution1.weight,
            params.convolution1.bias,
            self.device,
            channels, channels,
            batch_size, h, w,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.relu(x)

        x, [h, w] = ttnn_conv2d(
            x,
            params.convolution2.weight,
            params.convolution2.bias,
            self.device,
            channels, channels,
            batch_size, h, w,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Residual add -- both must be TILE + DRAM to avoid layout mismatches.
        residual = _dram_tile(residual)
        x = _dram_tile(x)
        return ttnn.add(residual, x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # ------------------------------------------------------------------
    def __call__(self, features):
        """features: list of 4 tensors (B, C_i, H_i, W_i), index 0=shallow."""
        x = None
        for i in range(3, -1, -1):
            params = self.parameters.layers[i]
            feat_i = features[i]   # (B, C_i, H_i, W_i)

            # Transpose to (B, H, W, C_i) for linear ops
            feat_i = ttnn.transpose(feat_i, -2, -1)   # (B, C_i, W_i, H_i)
            feat_i = ttnn.transpose(feat_i, -3, -1)   # (B, H_i, W_i, C_i)

            # Step 1: neck_conv -- channel normalizer  C_i -> 256
            feat_i = ttnn.linear(
                feat_i,
                params.neck_conv.weight,
                bias=params.neck_conv.bias,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Step 2: fusion projection  256 -> 256
            feat_i = ttnn.linear(
                feat_i,
                params.projection.weight,
                bias=params.projection.bias,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Back to (B, 256, H_i, W_i)
            feat_i = ttnn.transpose(feat_i, -3, -1)   # (B, 256, W_i, H_i)
            feat_i = ttnn.transpose(feat_i, -2, -1)   # (B, 256, H_i, W_i)

            if x is None:
                # Deepest level -- nothing to add yet.
                x = feat_i
            else:
                x_up = ttnn_upsample(x, scale_factor=2)
                x_up = _dram_tile(x_up)
                feat_i = _dram_tile(feat_i)
                x = ttnn.add(x_up, feat_i, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            x = self._residual_block(x, params.residual_layer1)
            x = self._residual_block(x, params.residual_layer2)

        return x   # (B, 256, H_out, W_out)


# ============================================================================
# DPT Head
# ============================================================================


class TtDPTHead:
    """Final prediction head: two 3x3 convs with upsample, then 1x1 depth conv."""

    def __init__(self, parameters, device):
        self.parameters = parameters
        self.device = device

    def __call__(self, x):
        batch_size, channels, h, w = x.shape
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        # Conv 1: channels -> 256
        x, [h, w] = ttnn_conv2d(
            x,
            self.parameters.conv1.weight,
            self.parameters.conv1.bias,
            self.device,
            channels, 256,
            batch_size, h, w,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.relu(x)

        # Upsample x2
        x = ttnn_upsample(x, scale_factor=2)
        h, w = h * 2, w * 2

        # Conv 2: 256 -> 128
        x, [h, w] = ttnn_conv2d(
            x,
            self.parameters.conv2.weight,
            self.parameters.conv2.bias,
            self.device,
            256, 128,
            batch_size, h, w,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.relu(x)

        # Conv 3: 128 -> 1  (1x1 depth prediction)
        x, [h, w] = ttnn_conv2d(
            x,
            self.parameters.conv3.weight,
            self.parameters.conv3.bias,
            self.device,
            128, 1,
            batch_size, h, w,
            kernel_size=(1, 1),
            padding=(0, 0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return x   # (B, 1, H_out, W_out)


# ============================================================================
# ViT Backbone helpers
# ============================================================================


def vit_patch_embeddings(config, pixel_values, parameters, device):
    """Patchify an image and project patches to the embedding dimension.

    Input:  pixel_values  (B, 3, 518, 518)
    Output: patch_embeddings  (B, seqL_padded-32, 1024)  TILE layout
    """
    batch_size, img_c, img_h, img_w = pixel_values.shape
    patch_size = 14
    patch_count = img_h // patch_size          # 37
    patch_count_all = patch_count * patch_count  # 1369

    # ---- 1. Patchify --------------------------------------------------
    x = ttnn.to_layout(pixel_values, ttnn.ROW_MAJOR_LAYOUT)
    # (B, C, pH, pW, qH, qW) -- split spatial dims into patches
    x = ttnn.reshape(x, (batch_size, img_c, patch_count, patch_size, patch_count, patch_size))
    # (B, pH, qH, pW, qW, C)
    x = ttnn.permute(x, (0, 2, 4, 3, 5, 1))
    # (B, pH*qH, pW*qW*C)
    x = ttnn.reshape(x, (batch_size, patch_count_all, patch_size * patch_size * img_c))

    # ---- 2. Pad to tile boundaries and project -------------------------
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    # seq dimension: seqL_padded - 32 CLS tokens = 1376 patch slots
    patch_seq_padded = config.get("seqL_padded", 1408) - 32   # 1376
    pad_seq  = patch_seq_padded - x.shape[1]   # 1376 - 1369 = 7
    # feature dimension: patch_size*patch_size*C = 588; padded to 608 (19 tiles)
    pad_feat = 608 - x.shape[2]               # 608 - 588 = 20

    x = ttnn.pad(x, padding=((0, 0), (0, pad_seq), (0, pad_feat)), value=0)

    x = ttnn.linear(
        x,
        parameters["projection"]["weight"],
        bias=parameters["projection"]["bias"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    return x   # (B, 1376, 1024)


def vit_embeddings(config, pixel_values, parameters, device):
    """Combine patch embeddings, CLS token, and position embeddings."""

    # 1. Patch embeddings: (B, 1376, 1024)
    patch_embeddings = vit_patch_embeddings(config, pixel_values, parameters["patch_embeddings"], device)

    # 2. CLS token -- stored pre-padded to shape (1, 32, 1024) in ROW_MAJOR.
    #    _cls_pad = 32 - original_cls_len ensures tile alignment.
    _cls = parameters["cls_token"]
    _cls_pad = 32 - _cls.shape[1]
    if _cls_pad > 0:
        cls_token = ttnn.pad(_cls, padding=((0, 0), (0, _cls_pad), (0, 0)), value=0)
    else:
        cls_token = _cls
    cls_token = ttnn.to_layout(cls_token, ttnn.TILE_LAYOUT)

    # Concat along seq dim: (B, 32 + 1376, 1024) = (B, 1408, 1024)
    embedding_output = ttnn.concat([cls_token, patch_embeddings], dim=1)

    # 3. Position embeddings -- pre-padded to (1, 1408, 1024).
    pos_raw = parameters["position_embeddings"]
    seqL = embedding_output.shape[1]
    if pos_raw.shape[1] != seqL:
        pos_raw = ttnn.to_layout(pos_raw, ttnn.ROW_MAJOR_LAYOUT)
        pos_raw = ttnn.slice(pos_raw, (0, 0, 0), (1, seqL, 1024))
    pos_embeds = ttnn.to_layout(pos_raw, ttnn.TILE_LAYOUT)

    # Add: both operands must be TILE + DRAM.
    embedding_output = _dram_tile(embedding_output)
    pos_embeds = _dram_tile(pos_embeds)
    embedding_output = ttnn.add(embedding_output, pos_embeds, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return embedding_output   # (B, 1408, 1024)


def vit_layer(hidden_states, parameters, config):
    """Single ViT-Large transformer block.

    Uses fused QKV matmul + ttnn's split/concatenate_heads helpers.
    All ops stay on DRAM (no sharding) for N300 8x7 grid compatibility.
    """
    num_heads  = config["num_attention_heads"]   # 16
    hidden_size = config["hidden_size"]           # 1024
    head_size  = hidden_size // num_heads         # 64

    # ---- LayerNorm 1 ---------------------------------------------------
    # Do NOT pass memory_config or program_config: causes TT_FATAL when
    # the Gamma tensor is 1D ROW_MAJOR.
    ln1 = ttnn.layer_norm(
        hidden_states,
        weight=parameters["layernorm_before"]["weight"],
        bias=parameters["layernorm_before"]["bias"],
    )

    # ---- Fused QKV projection ------------------------------------------
    qkv = ttnn.linear(
        ln1,
        parameters["attention"]["qkv"]["weight"],
        bias=parameters["attention"]["qkv"]["bias"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(ln1)

    # ---- Split into Q, K, V heads --------------------------------------
    (query, key, value) = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    ttnn.deallocate(qkv)

    # ---- Scaled dot-product attention ----------------------------------
    attn_scores = ttnn.matmul(query, key, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    # Must move to DRAM before scalar multiply -- fails on L1-sharded tensors.
    attn_scores = ttnn.to_memory_config(attn_scores, ttnn.DRAM_MEMORY_CONFIG)
    attn_scores = ttnn.mul(
        attn_scores,
        1.0 / (head_size ** 0.5),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # ViT has NO attention mask -- use plain softmax (not attention_softmax_).
    attn_probs = ttnn.softmax(attn_scores, dim=-1)
    ttnn.deallocate(attn_scores)

    context_layer = ttnn.matmul(attn_probs, value, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_probs)
    ttnn.deallocate(value)

    # ---- Merge heads & output projection --------------------------------
    context_layer = ttnn.transformer.concatenate_heads(
        context_layer, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    attn_out = ttnn.linear(
        context_layer,
        parameters["attention"]["output"]["dense"]["weight"],
        bias=parameters["attention"]["output"]["dense"]["bias"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(context_layer)

    # ---- Residual 1 ----------------------------------------------------
    hidden_states = _dram_tile(hidden_states)
    attn_out      = _dram_tile(attn_out)
    hidden_states = ttnn.add(hidden_states, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_out)

    # ---- LayerNorm 2 ---------------------------------------------------
    ln2 = ttnn.layer_norm(
        hidden_states,
        weight=parameters["layernorm_after"]["weight"],
        bias=parameters["layernorm_after"]["bias"],
    )

    # ---- MLP: FC1 (GELU) -> FC2 ----------------------------------------
    mlp_out = ttnn.linear(
        ln2,
        parameters["intermediate"]["dense"]["weight"],
        bias=parameters["intermediate"]["dense"]["bias"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        activation="gelu",
    )
    ttnn.deallocate(ln2)

    mlp_out = ttnn.linear(
        mlp_out,
        parameters["output"]["dense"]["weight"],
        bias=parameters["output"]["dense"]["bias"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # ---- Residual 2 ----------------------------------------------------
    hidden_states = _dram_tile(hidden_states)
    mlp_out       = _dram_tile(mlp_out)
    hidden_states = ttnn.add(hidden_states, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(mlp_out)

    return hidden_states


# ============================================================================
# Main model
# ============================================================================


class TtDepthAnythingV2:
    """Depth Anything V2 Large ported to TTNN for Tenstorrent Wormhole N300.

    Architecture:
        pixel_values (B, 3, 518, 518)
            |  vit_embeddings
        ViT-Large encoder -- 24 layers
            |  features extracted at layers [5, 11, 17, 23]
        DPT Reassemble x4
            |
        DPT Fusion (top-down)
            |
        DPT Head
            |
        depth map (B, 1, H_out, W_out)
    """

    def __init__(self, config, parameters, device):
        self.config = get_model_config(1, device)
        self.device = device
        # Recursively move all weights to device and wrap dicts with DictNamespace.
        self.parameters = self._move_to_device(parameters, device)

        # neck_hidden_sizes per HuggingFace config: [256, 512, 1024, 1024]
        neck_sizes = [256, 512, 1024, 1024]
        self.reassemble = [
            TtDPTReassembleLayer(
                self.parameters["neck"]["reassemble"][i],
                read_idx=i,
                config=self.config,
                device=device,
                feature_size=neck_sizes[i],
            )
            for i in range(4)
        ]
        self.fusion = TtDPTFusionStage(self.parameters["neck"]["fusion"], device)
        self.head   = TtDPTHead(self.parameters["head"], device)

    # ------------------------------------------------------------------
    def _move_to_device(self, params, device):
        """Recursively send tensors to device and wrap dicts with DictNamespace."""
        if isinstance(params, ttnn.Tensor):
            return ttnn.to_device(params, device)
        elif isinstance(params, dict):
            return DictNamespace({k: self._move_to_device(v, device) for k, v in params.items()})
        elif isinstance(params, list):
            return [self._move_to_device(v, device) for v in params]
        return params

    # ------------------------------------------------------------------
    def __call__(self, pixel_values):
        # ---- 1. Embeddings ---------------------------------------------
        hidden_states = vit_embeddings(
            self.config,
            pixel_values,
            self.parameters["backbone"]["embeddings"],
            self.device,
        )

        # Ensure TILE + DRAM before encoder (no sharding -- N300 8x7 grid).
        hidden_states = _dram_tile(hidden_states)

        # ---- 2. ViT-Large encoder (24 layers) --------------------------
        features = []
        out_indices = {5, 11, 17, 23}

        for i in range(24):
            hidden_states = vit_layer(
                hidden_states,
                self.parameters["backbone"]["encoder"]["layer"][i],
                self.config,
            )
            if i in out_indices:
                features.append(
                    ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
                )

        # ---- 3. Final backbone LayerNorm --------------------------------
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.parameters["backbone"]["layernorm"]["weight"],
            bias=self.parameters["backbone"]["layernorm"]["bias"],
        )
        # Replace the last stored feature with the post-norm version.
        features[-1] = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)

        # ---- 4. DPT Neck -----------------------------------------------
        reassembled = [self.reassemble[i](features[i]) for i in range(4)]
        fused = self.fusion(reassembled)

        # ---- 5. Head ---------------------------------------------------
        output = self.head(fused)

        return ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)


# ============================================================================
# Model configuration
# ============================================================================


def get_model_config(batch_size, device):
    """Return DRAM-only model config for N300 (8x7 grid) compatibility.

    All ops use DRAM_MEMORY_CONFIG to avoid grid / shard mismatches.
    Sharding optimisations belong in a later stage.
    """
    if device is not None:
        raw  = device.compute_with_storage_grid_size()
        grid_x, grid_y = raw.x, raw.y
    else:
        grid_x, grid_y = 8, 7

    core_grid = ttnn.CoreGrid(y=grid_y, x=grid_x)

    # Padded sequence length aligned to tile boundary (32).
    # 1369 patches + 1 CLS = 1370 -> must reach a multiple of 32.
    # CLS is padded to 32 tokens (one full tile), so total = 1369 + 32 = 1401
    # -> rounded up to 1408 (= 44 x 32).
    min_tok    = 1369 + 32
    seqL_padded = ((min_tok + 31) // 32) * 32  # 1408

    return {
        "num_attention_heads": 16,
        "hidden_size": 1024,
        "core_grid": core_grid,
        "core_grid_8x8": core_grid,
        "seqL_padded": seqL_padded,
        "program_configs": {},
        # Stage 1: DRAM everywhere.  Stage 2+ can replace with L1 shard configs.
        "l1_sharded_config": ttnn.DRAM_MEMORY_CONFIG,
        "l1_height_sharded_config": ttnn.DRAM_MEMORY_CONFIG,
    }


# ============================================================================
# Weight preprocessor
# ============================================================================


def custom_preprocessor(torch_model, name):
    """Convert a HuggingFace Depth-Anything-V2-Large model to TTNN tensors.

    Key decisions
    -------------
    * Attention weights (QKV, output dense, MLP fc1/fc2):
        bfloat8_b + TILE_LAYOUT  -- best throughput on Wormhole matrix engines.
    * Conv weights (patch projection, reassemble resize, fusion residuals, head):
        bfloat16 + ROW_MAJOR_LAYOUT  -- bfloat8_b requires TILE_LAYOUT which is
        INVALID for conv kernels; use bfloat16 to avoid TT_FATAL.
    * LayerNorm weights / biases:
        bfloat16 + TILE_LAYOUT, shape (1, hidden_size)  -- 1-D ROW_MAJOR causes
        a Gamma assertion crash inside the LayerNorm kernel.
    * Bias tensors for linear layers:
        bfloat16 + ROW_MAJOR_LAYOUT.
    * CLS token / position embeddings:
        bfloat16 + ROW_MAJOR_LAYOUT (converted to TILE at runtime after padding).
    * Patch projection weight:
        (1024, 3, 14, 14) -> permute -> reshape -> (588, 1024) -> pad to (608, 1024)
        so the feature dimension is a multiple of 32.
    """

    parameters = {}

    def _tile(tensor, dtype=ttnn.bfloat16):
        """Convert to TILE_LAYOUT (for matmul weights, layernorm params)."""
        return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    def _rm(tensor, dtype=ttnn.bfloat16):
        """Convert to ROW_MAJOR_LAYOUT (for conv weights, biases, tokens)."""
        return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    # =========================================================
    # 1. Backbone
    # =========================================================

    # Patch projection weight: (out=1024, in=3, kH=14, kW=14)
    #   -> permute to (in, kH, kW, out) = (3, 14, 14, 1024)
    #   -> reshape  to (3*14*14, 1024)  = (588, 1024)
    #   -> pad rows to 608 (nearest 32-multiple >= 588)
    pw = torch_model.backbone.embeddings.patch_embeddings.projection.weight
    pw = pw.permute(1, 2, 3, 0).reshape(-1, 1024)
    pw = torch.nn.functional.pad(pw, (0, 0, 0, 608 - pw.shape[0]))

    parameters["backbone"] = {
        "embeddings": {
            "patch_embeddings": {
                "projection": {
                    "weight": _tile(pw, dtype=ttnn.bfloat8_b),
                    "bias":   _rm(torch_model.backbone.embeddings.patch_embeddings.projection.bias),
                }
            },
            # CLS token: original shape (1, 1, 1024); pad seq-dim to 32 so that
            # after ttnn.pad at runtime only one step is needed.
            "cls_token": _rm(
                torch.nn.functional.pad(
                    torch_model.backbone.embeddings.cls_token,
                    (0, 0, 0, 31),   # pad seq 1 -> 32
                )
            ),
            # Position embeddings: original (1, 1370, 1024) [1 CLS + 1369 patches].
            # Pre-pad to (1, 1408, 1024) to match seqL_padded.
            "position_embeddings": _rm(
                torch.nn.functional.pad(
                    torch_model.backbone.embeddings.position_embeddings,
                    (0, 0, 0, 1408 - torch_model.backbone.embeddings.position_embeddings.shape[1]),
                )
            ),
        },
        "encoder": {"layer": []},
        "layernorm": {
            # unsqueeze(0) -> (1, 1024) so LayerNorm kernel finds a 2-D TILE tensor.
            "weight": _tile(torch_model.backbone.layernorm.weight.unsqueeze(0)),
            "bias":   _tile(torch_model.backbone.layernorm.bias.unsqueeze(0)),
        },
    }

    # ---- Encoder layers (24) ------------------------------------------
    for layer in torch_model.backbone.encoder.layer:
        # Fuse Q, K, V weight matrices: cat along head dim then transpose.
        # Each is (1024, 1024); cat -> (3072, 1024); .T -> (1024, 3072)
        qkv_w = torch.cat(
            [layer.attention.attention.query.weight,
             layer.attention.attention.key.weight,
             layer.attention.attention.value.weight],
            dim=0,
        ).transpose(0, 1)

        qkv_b = torch.cat(
            [layer.attention.attention.query.bias,
             layer.attention.attention.key.bias,
             layer.attention.attention.value.bias],
            dim=0,
        ).unsqueeze(0)   # (1, 3072)

        lp = {
            "layernorm_before": {
                "weight": _tile(layer.norm1.weight.unsqueeze(0)),
                "bias":   _tile(layer.norm1.bias.unsqueeze(0)),
            },
            "attention": {
                "qkv": {
                    "weight": _tile(qkv_w, dtype=ttnn.bfloat8_b),
                    "bias":   _tile(qkv_b),
                },
                "output": {
                    "dense": {
                        "weight": _tile(
                            layer.attention.output.dense.weight.transpose(0, 1),
                            dtype=ttnn.bfloat8_b,
                        ),
                        "bias": _rm(layer.attention.output.dense.bias),
                    }
                },
            },
            "layernorm_after": {
                "weight": _tile(layer.norm2.weight.unsqueeze(0)),
                "bias":   _tile(layer.norm2.bias.unsqueeze(0)),
            },
            "intermediate": {
                "dense": {
                    "weight": _tile(layer.mlp.fc1.weight.transpose(0, 1), dtype=ttnn.bfloat8_b),
                    "bias":   _rm(layer.mlp.fc1.bias),
                }
            },
            "output": {
                "dense": {
                    "weight": _tile(layer.mlp.fc2.weight.transpose(0, 1), dtype=ttnn.bfloat8_b),
                    "bias":   _rm(layer.mlp.fc2.bias),
                }
            },
        }
        parameters["backbone"]["encoder"]["layer"].append(lp)

    # =========================================================
    # 2. Neck -- Reassemble stage
    # =========================================================

    # neck_hidden_sizes from HuggingFace config: [256, 512, 1024, 1024]
    neck_hidden_sizes = [256, 512, 1024, 1024]

    parameters["neck"] = {"reassemble": [], "fusion": {"layers": []}}

    for i, layer in enumerate(torch_model.neck.reassemble_stage.layers):
        out_ch = neck_hidden_sizes[i]

        # Projection weight: (out_ch, 1024, 1, 1) conv1x1.
        # permute -> (1024, 1, 1, out_ch); reshape -> (1024, out_ch).
        proj_w = layer.projection.weight.permute(1, 2, 3, 0).reshape(-1, out_ch)

        rp = {
            "projection": {
                "weight": _tile(proj_w, dtype=ttnn.bfloat8_b),
                "bias":   _rm(layer.projection.bias),
            }
        }

        if hasattr(layer, "resize") and hasattr(layer.resize, "weight"):
            # Stride-2 conv for layer 3 (downsample).
            # Conv weight must be bfloat16 ROW_MAJOR -- NOT bfloat8_b.
            rp["resize"] = {
                "weight": _rm(layer.resize.weight, dtype=ttnn.bfloat16),
                "bias":   _rm(layer.resize.bias),
            }

        parameters["neck"]["reassemble"].append(rp)

    # =========================================================
    # 3. Neck -- Fusion stage
    # =========================================================
    # HuggingFace DPTNeck has intermediate nn.ModuleList `neck.convs` between
    # reassemble and fusion:
    #   neck.convs[i]:               Conv2d(neck_hidden_sizes[i], 256, 1)
    #   fusion.layers[i].projection: Conv2d(256, 256, 1)
    # The missing neck.convs caused: TT_FATAL width=1024 height=256

    for i, layer in enumerate(torch_model.neck.fusion_stage.layers):
        # neck.convs[i]: (256, neck_hidden_sizes[i], 1, 1) -> permute -> (in_ch, 256)
        nc_w = torch_model.neck.convs[i].weight.permute(1, 2, 3, 0).reshape(-1, 256)

        # fusion projection: (256, 256, 1, 1) -> permute -> (256, 256)
        fproj_w = layer.projection.weight.permute(1, 2, 3, 0).reshape(-1, 256)

        fp = {
            # Channel normalizer: neck_hidden_sizes[i] -> 256 (applied first)
            "neck_conv": {
                "weight": _tile(nc_w, dtype=ttnn.bfloat8_b),
                "bias":   _rm(torch_model.neck.convs[i].bias),
            },
            # Fusion projection: 256 -> 256
            "projection": {
                "weight": _tile(fproj_w, dtype=ttnn.bfloat8_b),
                "bias":   _rm(layer.projection.bias),
            },
            "residual_layer1": {
                "convolution1": {
                    "weight": _rm(layer.residual_layer1.convolution1.weight, dtype=ttnn.bfloat16),
                    "bias":   _rm(layer.residual_layer1.convolution1.bias),
                },
                "convolution2": {
                    "weight": _rm(layer.residual_layer1.convolution2.weight, dtype=ttnn.bfloat16),
                    "bias":   _rm(layer.residual_layer1.convolution2.bias),
                },
            },
            "residual_layer2": {
                "convolution1": {
                    "weight": _rm(layer.residual_layer2.convolution1.weight, dtype=ttnn.bfloat16),
                    "bias":   _rm(layer.residual_layer2.convolution1.bias),
                },
                "convolution2": {
                    "weight": _rm(layer.residual_layer2.convolution2.weight, dtype=ttnn.bfloat16),
                    "bias":   _rm(layer.residual_layer2.convolution2.bias),
                },
            },
        }
        parameters["neck"]["fusion"]["layers"].append(fp)

    # =========================================================
    # 4. Head
    # =========================================================

    # Conv weights: bfloat16 ROW_MAJOR (never bfloat8_b for conv kernels).
    parameters["head"] = {
        "conv1": {
            "weight": _rm(torch_model.head.conv1.weight, dtype=ttnn.bfloat16),
            "bias":   _rm(torch_model.head.conv1.bias),
        },
        "conv2": {
            "weight": _rm(torch_model.head.conv2.weight, dtype=ttnn.bfloat16),
            "bias":   _rm(torch_model.head.conv2.bias),
        },
        "conv3": {
            "weight": _rm(torch_model.head.conv3.weight, dtype=ttnn.bfloat16),
            "bias":   _rm(torch_model.head.conv3.bias),
        },
    }

    return parameters