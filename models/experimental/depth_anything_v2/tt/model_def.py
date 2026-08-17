# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math

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
        if name == "_d" or name.startswith("__"):
            raise AttributeError(name)
        try:
            d = object.__getattribute__(self, "_d")
            return d[name]
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
    x = ttnn.permute(x, (0, 2, 3, 1))  # NCHW -> NHWC

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
    ttnn.deallocate(x)  # Free input after conv consumes it

    # ttnn returns (1, 1, B*out_h*out_w, out_c); reshape then permute back.
    out_tensor = ttnn.reshape(out_tensor, (batch_size, out_h, out_w, out_channels))
    out_tensor = ttnn.permute(out_tensor, (0, 3, 1, 2))  # NHWC -> NCHW

    return out_tensor, [out_h, out_w]


def ttnn_upsample(x, scale_factor):
    """Nearest-neighbour upsample.  x is (B, C, H, W).

    ttnn.upsample expects NHWC format, so we permute before and after.
    """
    x = ttnn.permute(x, (0, 2, 3, 1))  # NCHW -> NHWC
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.upsample(x, scale_factor=scale_factor)
    x = ttnn.permute(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


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
        grid_h = grid_w = 37  # 518 / 14 = 37
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
        x = ttnn.transpose(x, -3, -1)  # (B, feat, grid_w, grid_h)
        x = ttnn.transpose(x, -2, -1)  # (B, feat, grid_h, grid_w)

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
                feat,
                feat,
                batch_size,
                grid_h,
                grid_w,
                stride=(2, 2),
            )

        return x  # (B, feat, H', W')


# ============================================================================
# DPT Neck -- Fusion stage
# ============================================================================


class TtDPTFusionStage:
    """Top-down fusion matching HF DepthAnythingFeatureFusionStage exactly.

    HF fusion_stage.forward(features):
        hidden_states = features[::-1]  # reverse: deepest first
        for idx, (hidden_state, layer) in enumerate(...):
            size = next_level_shape if not last else None
            if first:
                fused = layer(hidden_state, residual=None, size=size)
            else:
                fused = layer(fused, hidden_state, size=size)

    HF DepthAnythingFeatureFusionLayer.forward(hidden_state, residual=None, size=None):
        if residual is not None:
            if shapes differ: residual = interpolate(residual, hidden_state.shape[2:])
            hidden_state = hidden_state + residual_layer1(residual)
        hidden_state = residual_layer2(hidden_state)
        hidden_state = interpolate(hidden_state, scale_factor=2 or size=size)
        hidden_state = projection(hidden_state)  # 1x1 conv
        return hidden_state

    HF DepthAnythingPreActResidualLayer.forward(x):
        residual = x
        x = activation1(x)  # ReLU BEFORE conv (pre-activation!)
        x = convolution1(x)
        x = activation2(x)  # ReLU
        x = convolution2(x)
        return x + residual
    """

    # Input channel counts for each reassembly output level
    NECK_IN_CHANNELS = [256, 512, 1024, 1024]

    def __init__(self, parameters, device):
        self.parameters = parameters
        self.device = device

    # ------------------------------------------------------------------
    def _pre_act_residual_block(self, x, params):
        """PRE-activation residual: ReLU → conv1 → ReLU → conv2 → add skip."""
        residual = x
        batch_size, channels, h, w = x.shape

        # activation1 (ReLU) BEFORE convolution1
        x = ttnn.relu(x)
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # activation2 (ReLU)
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Skip connection
        residual = _dram_tile(residual)
        x = _dram_tile(x)
        return ttnn.add(residual, x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # ------------------------------------------------------------------
    def _upsample_to(self, x, target_h, target_w):
        """Upsample x to (target_h, target_w), fully on-device.

        All upsampling uses on-device nearest-neighbor to avoid CPU
        round-trips and enable trace capture.  For non-exact-2x scales
        (e.g. 19->37), we upsample 2x then slice to the target size.
        """
        if target_h is None and target_w is None:
            return self._device_upsample_2x(x)
        cur_h, cur_w = x.shape[2], x.shape[3]
        if target_h == cur_h * 2 and target_w == cur_w * 2:
            return self._device_upsample_2x(x)
        return self._device_upsample_and_slice(x, target_h, target_w)

    def _device_upsample_2x(self, x):
        """On-device nearest-neighbor 2x upsample (no CPU round-trip)."""
        _, _, h, w = x.shape
        x = ttnn_upsample(x, scale_factor=2)
        return x, h * 2, w * 2

    def _device_upsample_and_slice(self, x, target_h, target_w):
        """On-device upsample to at least (target_h, target_w), then slice.

        For non-integer scale factors (e.g. 19->37 ≈ 1.95x), we:
          1. Compute the smallest integer scale_factor >= target/current
          2. Upsample by that factor using nearest-neighbor on-device
          3. Slice the result to the exact target dimensions

        This eliminates the CPU round-trip that was previously needed for
        bilinear interpolation, enabling trace capture for the full model.
        The PCC impact is negligible for near-2x scales (19->38->37).
        """
        _, channels, cur_h, cur_w = x.shape
        scale = int(math.ceil(target_h / cur_h))
        x = ttnn_upsample(x, scale_factor=scale)
        upsampled_h = cur_h * scale
        upsampled_w = cur_w * scale

        # Slice to exact target if overshot
        if upsampled_h != target_h or upsampled_w != target_w:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.slice(
                x,
                (0, 0, 0, 0),
                (1, channels, target_h, target_w),
            )

        return x, target_h, target_w

    # ------------------------------------------------------------------
    def _projection_1x1(self, x, params):
        """1x1 convolution implemented as ttnn.linear after BCHW→BHWC reshape."""
        batch_size, channels, h, w = x.shape
        out_channels = params.weight.shape[-1]  # Derive from weight, not hardcoded
        x = ttnn.permute(x, (0, 2, 3, 1))  # (B, H, W, C)
        x = _dram_tile(x)
        x = ttnn.reshape(x, (batch_size, h * w, channels))
        x = ttnn.linear(
            x,
            params.weight,
            bias=params.bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.reshape(x, (batch_size, h, w, out_channels))
        x = ttnn.permute(x, (0, 3, 1, 2))  # (B, out_channels, H, W)
        return x

    # ------------------------------------------------------------------
    def __call__(self, features):
        """features: list of 4 tensors (B, C_i, H_i, W_i), index 0=shallow.

        HF processes in REVERSE (deepest first).

        HF neck.forward:
            reassembled = reassemble_stage(hidden_states)   # [0..3]
            convolved = [convs[i](reassembled[i]) for i]    # convs[i] with features[i]
            output = fusion_stage(convolved)                # reverses internally

        HF fusion_stage.forward(hidden_states):
            hidden_states = hidden_states[::-1]  # [3,2,1,0]
            for idx, (hs, layer) in enumerate(zip(hidden_states, self.layers)):
                ...
            So: layers[0] processes convolved[3] (deepest)
                layers[1] processes convolved[2]
                layers[2] processes convolved[1]
                layers[3] processes convolved[0] (shallowest)

        Our parameters.layers[i] bundles:
            - neck_conv from neck.convs[i]  (expects features[i])
            - projection/residuals from fusion_stage.layers[i]
        So neck_conv[i] must be applied to features[i],
        but fusion layers[i] must process convolved[3-i].
        """
        # Step 1: Apply neck_convs — parameters.layers[i].neck_conv for features[i]
        convolved = []
        for i in range(4):
            params = self.parameters.layers[i]  # neck_conv[i] expects features[i]
            feat_i = features[i]
            batch_size, in_ch, h_i, w_i = feat_i.shape

            feat_i = ttnn.to_layout(feat_i, ttnn.ROW_MAJOR_LAYOUT)
            feat_i, [h_i, w_i] = ttnn_conv2d(
                feat_i,
                params.neck_conv.weight,
                params.neck_conv.bias,
                self.device,
                self.NECK_IN_CHANNELS[i],
                256,
                batch_size,
                h_i,
                w_i,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            convolved.append(feat_i)

        # Step 2: Fusion — process in reverse (deepest first)
        # reversed = [convolved[3], convolved[2], convolved[1], convolved[0]]
        # layers[0] → convolved[3], layers[1] → convolved[2], etc.
        reversed_features = convolved[::-1]
        fused = None

        for idx in range(4):
            # HF: fusion_stage.layers[idx] processes reversed_features[idx]
            fusion_params = self.parameters.layers[idx]  # fusion layers[idx]
            hidden_state = reversed_features[idx]

            # Target upsample size = next level's spatial dims
            if idx < 3:
                next_feat = reversed_features[idx + 1]
                target_h, target_w = next_feat.shape[2], next_feat.shape[3]
            else:
                target_h, target_w = None, None  # scale_factor=2

            if fused is None:
                # First layer (deepest) — no residual
                hidden_state = self._pre_act_residual_block(hidden_state, fusion_params.residual_layer2)
                hidden_state, _, _ = self._upsample_to(hidden_state, target_h, target_w)
                hidden_state = self._projection_1x1(hidden_state, fusion_params.projection)
                fused = hidden_state
            else:
                # Subsequent layers:
                # fused = running result, hidden_state = new feature from this level
                new_feature = hidden_state

                # Match spatial dims if needed
                fh, fw = fused.shape[2], fused.shape[3]
                nh, nw = new_feature.shape[2], new_feature.shape[3]
                if fh != nh or fw != nw:
                    new_feature, _, _ = self._upsample_to(new_feature, fh, fw)

                # residual_layer1(new_feature) + fused
                res1_out = self._pre_act_residual_block(new_feature, fusion_params.residual_layer1)
                fused = _dram_tile(fused)
                res1_out = _dram_tile(res1_out)
                fused = ttnn.add(fused, res1_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

                # residual_layer2
                fused = self._pre_act_residual_block(fused, fusion_params.residual_layer2)

                # Upsample to next level
                fused, _, _ = self._upsample_to(fused, target_h, target_w)

                # 1x1 projection
                fused = self._projection_1x1(fused, fusion_params.projection)

        return fused  # (B, 256, H_out, W_out)


# ============================================================================
# DPT Head
# ============================================================================


class TtDPTHead:
    """Final prediction head matching HF DepthAnythingDepthEstimationHead.

    HF forward order:
        conv1(256→128, 3x3)
        nearest-neighbor 2x upsample + slice (to patch_h*14 × patch_w*14)
        conv2(128→32, 3x3)
        activation1 (ReLU)
        conv3(32→1, 1x1)
        activation2 (ReLU) * max_depth
    """

    def __init__(self, parameters, device, patch_size=14, max_depth=80.0):
        self.parameters = parameters
        self.device = device
        self.patch_size = patch_size
        self.max_depth = max_depth

    def __call__(self, x, patch_height=37, patch_width=37):
        batch_size, channels, h, w = x.shape
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        # Conv 1: 256 -> 128 (NO ReLU after this in HF!)
        x, [h, w] = ttnn_conv2d(
            x,
            self.parameters.conv1.weight,
            self.parameters.conv1.bias,
            self.device,
            channels,
            128,
            batch_size,
            h,
            w,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Upsample to full input resolution (patch_h * 14, patch_w * 14)
        # On-device nearest-neighbor 2x upsample + slice to exact target.
        # 296 -> 592 (2x nearest) -> slice to 518x518.
        # This replaces the CPU bilinear round-trip to enable trace capture.
        target_h = patch_height * self.patch_size  # 518
        target_w = patch_width * self.patch_size  # 518
        scale = int(math.ceil(target_h / h))  # ceil(518/296) = 2
        x = ttnn_upsample(x, scale_factor=scale)
        upsampled_h, upsampled_w = h * scale, w * scale

        # Slice to exact target size (592 -> 518)
        if upsampled_h != target_h or upsampled_w != target_w:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.slice(
                x,
                (0, 0, 0, 0),
                (batch_size, 128, target_h, target_w),
            )
        h, w = target_h, target_w

        # Conv 2: 128 -> 32
        x, [h, w] = ttnn_conv2d(
            x,
            self.parameters.conv2.weight,
            self.parameters.conv2.bias,
            self.device,
            128,
            32,
            batch_size,
            h,
            w,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.relu(x)  # activation1

        # Conv 3: 32 -> 1  (1x1 depth prediction)
        x, [h, w] = ttnn_conv2d(
            x,
            self.parameters.conv3.weight,
            self.parameters.conv3.bias,
            self.device,
            32,
            1,
            batch_size,
            h,
            w,
            kernel_size=(1, 1),
            padding=(0, 0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.relu(x)  # activation2

        # Scale by max_depth (HF: predicted_depth * self.max_depth)
        x = ttnn.multiply(x, self.max_depth)

        return x  # (B, 1, H_out, W_out)


# ============================================================================
# ViT Backbone helpers
# ============================================================================


def vit_patch_embeddings(config, pixel_values, parameters, device):
    """Patchify an image and project patches to the embedding dimension.

    Uses ttnn.conv2d directly (kernel=14, stride=14) to match PyTorch Conv2d.
    ttnn.conv2d internally works in NHWC and returns (1,1,B*H*W,C) which maps
    directly to our sequence format (B, H*W, C) = (B, 1369, 1024).

    Input:  pixel_values  (B, 3, 518, 518) in NCHW
    Output: patch_embeddings  (B, patch_seq_padded, 1024) in TILE layout
    """
    batch_size, img_c, img_h, img_w = pixel_values.shape
    patch_size = 14

    # ---- 1. Convert input to NHWC for ttnn.conv2d ----------------------
    x = ttnn.transpose(pixel_values, -2, -1)  # (B, C, H, W) -> (B, C, W, H)
    x = ttnn.transpose(x, -3, -1)  # (B, C, W, H) -> (B, H, W, C)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    # Weight: (out_ch, in_ch, kH, kW) -- native Conv2d format
    weight = parameters["projection"]["conv_weight"]
    if len(weight.shape) == 2:
        weight = ttnn.reshape(weight, (config["hidden_size"], img_c, patch_size, patch_size))

    # Bias: needs (1, 1, 1, out_ch)
    bias = parameters["projection"]["bias"]
    if len(bias.shape) == 1:
        bias = ttnn.reshape(bias, (1, 1, 1, bias.shape[0]))
    elif len(bias.shape) == 2:
        bias = ttnn.reshape(bias, (1, 1, bias.shape[0], bias.shape[1]))

    # ---- 2. Conv2d: native NHWC path ----------------------------------
    out_tensor, [out_h, out_w] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=img_c,
        out_channels=config["hidden_size"],  # 1024
        batch_size=batch_size,
        input_height=img_h,
        input_width=img_w,
        kernel_size=(patch_size, patch_size),
        stride=(patch_size, patch_size),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        return_output_dim=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # out_tensor: (1, 1, B*out_h*out_w, out_ch) = (1, 1, 1369, 1024) -- NHWC flat

    # ---- 3. Reshape to (B, H*W, C) = (B, 1369, 1024) -----------------
    # The conv output is already in (N, H*W, C) order -- no transpose needed!
    x = ttnn.reshape(out_tensor, (batch_size, out_h * out_w, config["hidden_size"]))

    # ---- 4. Pad sequence to tile boundary ------------------------------
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    patch_seq_padded = config.get("seqL_padded", 1568) - 32  # 1536
    pad_seq = patch_seq_padded - x.shape[1]  # 1536 - 1369 = 167
    if pad_seq > 0:
        x = ttnn.pad(x, padding=((0, 0), (0, pad_seq), (0, 0)), value=0)

    return x  # (B, 1536, 1024)


def vit_embeddings(config, pixel_values, parameters, device):
    """Combine patch embeddings, CLS token, and position embeddings.

    Output shape: (B, seqL_padded, 1024) where seqL_padded comes from config.
    """
    assert pixel_values.shape[0] == 1, (
        f"Only batch_size=1 is supported (CLS token and position embeddings are not broadcast). "
        f"Got batch_size={pixel_values.shape[0]}."
    )

    seqL_padded = config.get("seqL_padded", 1568)

    # 1. Patch embeddings: (B, patch_seq_padded, 1024)
    patch_embeddings = vit_patch_embeddings(config, pixel_values, parameters["patch_embeddings"], device)

    # 2. CLS token -- stored pre-padded to shape (1, 32, 1024) in ROW_MAJOR.
    _cls = parameters["cls_token"]
    _cls_pad = 32 - _cls.shape[1]
    if _cls_pad > 0:
        cls_token = ttnn.pad(_cls, padding=((0, 0), (0, _cls_pad), (0, 0)), value=0)
    else:
        cls_token = _cls
    cls_token = ttnn.to_layout(cls_token, ttnn.TILE_LAYOUT)

    # Concat along seq dim: (B, 32 + patch_seq_padded, 1024)
    embedding_output = ttnn.concat([cls_token, patch_embeddings], dim=1)

    # Pad to seqL_padded if needed
    current_seq = embedding_output.shape[1]
    if current_seq < seqL_padded:
        pad_seq = seqL_padded - current_seq
        embedding_output = ttnn.pad(embedding_output, padding=((0, 0), (0, pad_seq), (0, 0)), value=0)

    embedding_output = ttnn.to_layout(embedding_output, ttnn.TILE_LAYOUT)

    # 3. Position embeddings -- pre-padded to (1, seqL_padded, 1024).
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

    return embedding_output  # (B, seqL_padded, 1024)


def vit_layer(hidden_states, parameters, config, attention_mask=None):
    """Single ViT-Large transformer block — L1 block-sharded + SDPA.

    Uses SDPA (scaled_dot_product_attention) instead of naive Q×K→softmax→V
    to avoid materializing the full 49×49 attention map in L1 (which would
    require 686 tiles/core = 686KB, exceeding the Wormhole L1 budget).

    SDPA processes attention in chunks (q_chunk_size=32) and keeps all
    intermediates within L1 per chunk.  Attention mask is passed directly
    to SDPA (no manual add needed).
    """
    num_heads = config["num_attention_heads"]  # 16
    hidden_size = config["hidden_size"]  # 1024
    head_size = hidden_size // num_heads  # 64
    pconfigs = config["program_configs"]

    # ---- LayerNorm 1 ---------------------------------------------------
    ln1 = ttnn.layer_norm(
        hidden_states,
        weight=parameters["layernorm_before"]["weight"],
        bias=parameters["layernorm_before"]["bias"],
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=pconfigs["layernorm_program_config"],
        compute_kernel_config=pconfigs["compute_kernel_config"],
    )

    # ---- Fused QKV projection ------------------------------------------
    qkv = ttnn.linear(
        ln1,
        parameters["attention"]["qkv"]["weight"],
        bias=parameters["attention"]["qkv"]["bias"],
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        program_config=pconfigs["qkv_matmul_program_config"],
    )
    ttnn.deallocate(ln1)

    # Move QKV to DRAM for the split (split_query_key_value is not perf-critical)
    qkv = ttnn.to_memory_config(qkv, ttnn.DRAM_MEMORY_CONFIG)

    # ---- Split into Q, K, V heads --------------------------------------
    # transpose_key=False: SDPA expects K in [B,H,S,D] not [B,H,D,S]
    (query, key, value) = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        num_heads=num_heads,
        transpose_key=False,
    )
    ttnn.deallocate(qkv)

    # ---- SDPA (FlashAttention) -----------------------------------------
    # Uses chunked attention to stay within L1 budget.
    # Never materializes the full (seqL × seqL) attention map.
    # ViT uses full attention (no mask) -- all tokens attend to all tokens.
    context_layer = ttnn.transformer.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=False,
        scale=1.0 / (head_size**0.5),
        attn_mask=attention_mask,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=pconfigs["compute_kernel_config"],
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)
    ttnn.deallocate(value)

    # ---- Merge heads & output projection --------------------------------
    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Reshard back to full grid for output dense matmul
    block_sharded_config_full = ttnn.create_sharded_memory_config(
        context_layer.padded_shape,
        core_grid=config["core_grid"],
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    context_layer = ttnn.to_memory_config(context_layer, block_sharded_config_full)

    attn_out = ttnn.linear(
        context_layer,
        parameters["attention"]["output"]["dense"]["weight"],
        bias=parameters["attention"]["output"]["dense"]["bias"],
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        program_config=pconfigs["output_matmul_program_config"],
    )
    ttnn.deallocate(context_layer)

    # ---- Residual 1 ----------------------------------------------------
    # DINOv2 LayerScale is fused into output_dense weight/bias at preprocessing.
    hidden_states = ttnn.add(
        attn_out,
        hidden_states,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(attn_out)

    # ---- LayerNorm 2 ---------------------------------------------------
    ln2 = ttnn.layer_norm(
        hidden_states,
        weight=parameters["layernorm_after"]["weight"],
        bias=parameters["layernorm_after"]["bias"],
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=pconfigs["layernorm_program_config"],
        compute_kernel_config=pconfigs["compute_kernel_config"],
    )

    # ---- MLP: FC1 -> GELU -> FC2 ----------------------------------------
    # DINOv2 MLP applies GELUActivation() after FC1.
    # NOTE: fused_activation=(GELU, False) in the program config was found
    # to produce numerically incorrect GELU (PCC≈0.40 vs PyTorch).  We use
    # an explicit ttnn.gelu() instead, which matches PyTorch exactly.
    mlp_out = ttnn.linear(
        ln2,
        parameters["intermediate"]["dense"]["weight"],
        bias=parameters["intermediate"]["dense"]["bias"],
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        program_config=pconfigs["ff1_matmul_program_config"],
    )
    ttnn.deallocate(ln2)

    # Explicit GELU activation (move to DRAM, apply, reshard)
    mlp_out = ttnn.to_memory_config(mlp_out, ttnn.DRAM_MEMORY_CONFIG)
    mlp_out = ttnn.gelu(mlp_out)
    mlp_gelu_shard = ttnn.create_sharded_memory_config(
        mlp_out.padded_shape,
        core_grid=config["core_grid"],
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    mlp_out = ttnn.to_memory_config(mlp_out, mlp_gelu_shard)

    mlp_out = ttnn.linear(
        mlp_out,
        parameters["output"]["dense"]["weight"],
        bias=parameters["output"]["dense"]["bias"],
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        program_config=pconfigs["ff2_matmul_program_config"],
    )

    # ---- Residual 2 ----------------------------------------------------
    # DINOv2 LayerScale is fused into FC2 weight/bias at preprocessing.
    hidden_states = ttnn.add(
        mlp_out,
        hidden_states,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
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
            |  features extracted at layers {4, 11, 17, 23} (0-indexed)
        DPT Reassemble x4
            |
        DPT Fusion (top-down)
            |
        DPT Head
            |
        depth map (B, 1, H_out, W_out)
    """

    BATCH_SIZE = 1  # Only batch_size=1 is supported (CLS/pos embeddings are not broadcast)

    def __init__(self, config, parameters, device):
        """Initialize the TT Depth Anything V2 model.

        Args:
            config: HuggingFace model config (used for architecture hyperparameters).
            parameters: Model parameters produced by custom_preprocessor.
            device: Target TT device.

        Note:
            Only batch_size=1 is supported.  CLS token and position embeddings
            are stored as (1, seqL, 1024) and are not broadcast across batches.
            TTNN execution config is derived internally from the target device and
            batch size, and is stored on ``self.config`` for internal use.
            The passed-in ``config`` is retained as ``self.hf_config``.
        """
        self.hf_config = config
        self.config = get_model_config(batch_size=self.BATCH_SIZE, device=device)
        self.device = device
        # Recursively move all weights to device and wrap dicts with DictNamespace.
        self.parameters = self._move_to_device(parameters, device)

        # neck_hidden_sizes per HuggingFace config; preserve defaults if absent.
        neck_sizes = getattr(self.hf_config, "neck_hidden_sizes", [256, 512, 1024, 1024])
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
        max_depth = getattr(self.hf_config, "max_depth", 80.0)
        self.head = TtDPTHead(self.parameters["head"], device, max_depth=max_depth)

        # Pre-build attention mask for padding tokens (fixed for all forward passes).
        # Sequence layout: [CLS(1 real + 31 pad), patches(1369 real + 167 pad)] = 1568
        # Real tokens: position 0 (CLS), positions 32-1400 (patches)
        # Padding tokens: positions 1-31, 1401-1567  -> set to -inf
        seqL = self.config["seqL_padded"]  # 1568
        mask = torch.zeros(1, 1, seqL, seqL)
        mask[:, :, :, 1:32] = float("-inf")  # CLS padding keys
        mask[:, :, :, 1401:seqL] = float("-inf")  # patch padding keys
        self.attention_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

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

        # ---- Shard into L1 before encoder --------------------------------
        # Move from interleaved DRAM to block-sharded L1 across all cores.
        seqL = self.config["seqL_padded"]  # 1568
        hidden_size = self.config["hidden_size"]  # 1024
        encoder_sharded_config = ttnn.create_sharded_memory_config(
            [hidden_states.shape[0], seqL, hidden_size],
            core_grid=self.config["core_grid"],
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        hidden_states = ttnn.to_memory_config(
            hidden_states,
            memory_config=encoder_sharded_config,
            dtype=ttnn.bfloat8_b,
        )

        # ---- Attention mask (pre-built in __init__) ------------------------
        attention_mask = self.attention_mask

        # ---- 2. ViT-Large encoder (24 layers) --------------------------
        features = []
        out_indices = {4, 11, 17, 23}  # HF out_indices=[5,12,18,24] are 1-indexed (include embedding)

        for i in range(24):
            hidden_states = vit_layer(
                hidden_states,
                self.parameters["backbone"]["encoder"]["layer"][i],
                self.config,
                attention_mask=attention_mask,
            )
            if i in out_indices:
                features.append(ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG))

        # ---- 3. Final backbone LayerNorm --------------------------------
        # Move to DRAM for the final layernorm (neck/head use DRAM anyway)
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.parameters["backbone"]["layernorm"]["weight"],
            bias=self.parameters["backbone"]["layernorm"]["bias"],
        )
        # Replace the last stored feature with the post-norm version.
        old_feature = features[-1]
        features[-1] = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(old_feature)

        # ---- 4. DPT Neck -----------------------------------------------
        reassembled = []
        for i in range(4):
            reassembled.append(self.reassemble[i](features[i]))
            ttnn.deallocate(features[i])  # Free encoder feature after reassembly
        del features

        fused = self.fusion(reassembled)
        for r in reassembled:
            ttnn.deallocate(r)  # Free reassembled features after fusion
        del reassembled

        # ---- 5. Head ---------------------------------------------------
        patch_height = 518 // 14  # 37
        patch_width = 518 // 14  # 37
        output = self.head(fused, patch_height=patch_height, patch_width=patch_width)
        ttnn.deallocate(fused)  # Free fusion output after head consumes it

        return ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)


# ============================================================================
# Model configuration
# ============================================================================


def get_model_config(batch_size, device):
    """Return L1 block-sharded model config for N300 (8x7 grid).

    All encoder ops use L1_BLOCK_SHARDED_MEMORY_CONFIG with explicit
    program configs for matmul, layernorm, and softmax.  This eliminates
    DRAM round-trips inside the ViT encoder, which is the primary
    bottleneck at Stage 1.

    Sequence is padded to 1568 (= 49 tiles) so that each of the 7 rows
    gets exactly 7 tiles, giving perfectly balanced compute.
    """
    assert device is not None, "get_model_config requires a TT device (device=None is not supported)"
    raw = device.compute_with_storage_grid_size()
    grid_x = raw.x  # 8 on N300
    # Force grid_y=7: 1568 tokens / 7 rows = 224 = 7×32 (tile-aligned).
    # With grid_y=8: 1568/8 = 196 which is NOT a multiple of 32 → TT_FATAL.
    grid_y = min(raw.y, 7)

    core_grid = ttnn.CoreGrid(y=grid_y, x=grid_x)

    TILE_HEIGHT = 32

    # Padded sequence length: 1369 patches + 32 CLS = 1401 real tokens.
    # Pad to 1568 (= 49 * 32) so that 49 / 7 = 7 tiles per core row.
    seqL_padded = 1568

    # Tile counts
    seqL_t = seqL_padded // TILE_HEIGHT  # 49 tiles
    dim_t = 1024 // TILE_HEIGHT  # 32 tiles
    dim_t__x = dim_t // grid_x  # 4 tiles per core (width)
    seqL_t__y = seqL_t // grid_y  # 7 tiles per core (height)
    # head_num = 16, head_size = dim_t // head_num = 2 tiles per head (used by SDPA internally)

    # MLP intermediate dimension: 4096
    mlp_dim_t__x = (4096 // TILE_HEIGHT) // grid_x  # 16 tiles per core

    # Sharded program configs (mirrors the reference optimized ViT WH)
    program_configs = {
        # LayerNorm (used for both LN1 and LN2 in each encoder layer)
        "layernorm_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            subblock_w=dim_t__x,  # 4
            block_h=seqL_t__y,  # 7
            block_w=dim_t__x,  # 4
            inplace=False,
        ),
        # Fused QKV matmul: (seqL, 1024) × (1024, 3072) → (seqL, 3072)
        "qkv_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=dim_t__x,  # 4
            out_subblock_h=1,
            out_subblock_w=dim_t__x,  # 4
            per_core_M=seqL_t__y,  # 7
            per_core_N=3 * dim_t__x,  # 12
            transpose_mcast=False,
            fused_activation=None,
        ),
        # SDPA (FlashAttention): replaces naive Q×K→softmax→V which would
        # overflow L1 (14×49 = 686 tiles/core = 686KB, too large for 1024KB L1).
        # SDPA processes attention in chunks, never materializing the full map.
        "sdpa_program_config": ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            q_chunk_size=32,  # process 32 seq tiles at a time (1024 tokens)
            k_chunk_size=32,  # same for keys
            exp_approx_mode=False,  # exact exp for accuracy
        ),
        # Output dense: (seqL, 1024) × (1024, 1024) → (seqL, 1024)
        "output_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=dim_t__x,  # 4
            out_subblock_h=1,
            out_subblock_w=dim_t__x,  # 4
            per_core_M=seqL_t__y,  # 7
            per_core_N=dim_t__x,  # 4
            transpose_mcast=False,
            fused_activation=None,
        ),
        # MLP FC1 with fused GELU: (seqL, 1024) × (1024, 4096)
        "ff1_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=dim_t__x,  # 4
            out_subblock_h=1,
            out_subblock_w=mlp_dim_t__x // 2,  # 8
            per_core_M=seqL_t__y,  # 7
            per_core_N=mlp_dim_t__x,  # 16
            transpose_mcast=False,
            fused_activation=None,  # GELU applied explicitly after FC1 for numerical accuracy
        ),
        # MLP FC2: (seqL, 4096) × (4096, 1024)
        # in0_block_w=4 (not 16) to fit in L1 with bfloat16 weights
        "ff2_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=dim_t__x,  # 4 (reduced from 16 to fit L1)
            out_subblock_h=1,
            out_subblock_w=dim_t__x,  # 4
            per_core_M=seqL_t__y,  # 7
            per_core_N=dim_t__x,  # 4
            transpose_mcast=False,
            fused_activation=None,
        ),
        # Compute kernel config for all matmuls and layernorms
        # HiFi4 + fp32 accumulation: required for ≥0.995 PCC across 24 layers.
        # HiFi2/bf16-acc introduced ~0.2% error per layer that compounded
        # through residual connections, causing L17-L19 PCC dip to 0.89-0.95.
        "compute_kernel_config": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    }

    return {
        "num_attention_heads": 16,
        "hidden_size": 1024,
        "core_grid": core_grid,
        "core_grid_8x8": core_grid,
        "seqL_padded": seqL_padded,
        "program_configs": program_configs,
        "l1_sharded_config": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        "l1_height_sharded_config": ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
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
        most use bfloat16 + ROW_MAJOR_LAYOUT; the fused QKV bias is stored as
        bfloat16 + TILE_LAYOUT to match the fused attention path.
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

    # Patch projection weight: keep original Conv2d format (out=1024, in=3, kH=14, kW=14)
    # for ttnn.conv2d.  Conv weights must be ROW_MAJOR, not TILE.
    pw = torch_model.backbone.embeddings.patch_embeddings.projection.weight.float()
    # pw shape: (1024, 3, 14, 14) — native Conv2d format

    parameters["backbone"] = {
        "embeddings": {
            "patch_embeddings": {
                "projection": {
                    "conv_weight": _rm(pw),
                    "bias": _rm(torch_model.backbone.embeddings.patch_embeddings.projection.bias),
                }
            },
            # CLS token: original shape (1, 1, 1024); pad seq-dim to 32 so that
            # after ttnn.pad at runtime only one step is needed.
            "cls_token": _rm(
                torch.nn.functional.pad(
                    torch_model.backbone.embeddings.cls_token,
                    (0, 0, 0, 31),  # pad seq 1 -> 32
                )
            ),
            # Position embeddings: original (1, 1370, 1024) [1 CLS + 1369 patches].
            # Our sequence layout: [CLS(32 tokens), patches(1369), pad(167)] = 1568.
            # HF layout: [CLS(1), patches(1369)] = 1370.
            # Must rearrange: CLS pos → slot 0 (pad 31 zeros to fill 32),
            # then patch positions → slots 32–1400, then pad to 1568.
            "position_embeddings": _rm(
                torch.cat(
                    [
                        torch_model.backbone.embeddings.position_embeddings[:, :1, :],  # CLS pos (1, 1, 1024)
                        torch.zeros(1, 31, 1024),  # pad CLS region to 32 tokens
                        torch_model.backbone.embeddings.position_embeddings[
                            :, 1:, :
                        ],  # patch positions (1, 1369, 1024)
                        torch.zeros(1, 1568 - 32 - 1369, 1024),  # pad to seqL_padded=1568
                    ],
                    dim=1,
                )
            ),
        },
        "encoder": {"layer": []},
        "layernorm": {
            # unsqueeze(0) -> (1, 1024) so LayerNorm kernel finds a 2-D TILE tensor.
            "weight": _tile(torch_model.backbone.layernorm.weight.unsqueeze(0)),
            "bias": _tile(torch_model.backbone.layernorm.bias.unsqueeze(0)),
        },
    }

    # ---- Encoder layers (24) ------------------------------------------
    for layer in torch_model.backbone.encoder.layer:
        # Fuse Q, K, V weight matrices: cat along head dim then transpose.
        # Each is (1024, 1024); cat -> (3072, 1024); .T -> (1024, 3072)
        qkv_w = torch.cat(
            [
                layer.attention.attention.query.weight,
                layer.attention.attention.key.weight,
                layer.attention.attention.value.weight,
            ],
            dim=0,
        ).transpose(0, 1)

        qkv_b = torch.cat(
            [
                layer.attention.attention.query.bias,
                layer.attention.attention.key.bias,
                layer.attention.attention.value.bias,
            ],
            dim=0,
        ).unsqueeze(
            0
        )  # (1, 3072)

        # ---- DINOv2 LayerScale: fuse into weights at preprocessing time ----
        # layer_scale1 (1024,) scales attention output before residual 1.
        # layer_scale2 (1024,) scales MLP output before residual 2.
        # Math: x + λ*(H@W+b) = x + H@(W*λ) + b*λ
        # So we pre-multiply output_dense weight/bias by λ1, and FC2 by λ2.
        # This avoids any runtime mul (zero L1 overhead, zero perf cost).
        ls1 = layer.layer_scale1.lambda1.detach()  # (1024,)
        ls2 = layer.layer_scale2.lambda1.detach()  # (1024,)

        # Fuse layer_scale1 into output_dense: W_out is (in=1024, out=1024)
        # Multiply each output column j by ls1[j]
        out_w = layer.attention.output.dense.weight.transpose(0, 1)  # (in, out)
        out_w_fused = out_w * ls1.unsqueeze(0)  # broadcast ls1 across rows
        out_b_fused = layer.attention.output.dense.bias * ls1

        lp = {
            "layernorm_before": {
                "weight": _tile(layer.norm1.weight.unsqueeze(0)),
                "bias": _tile(layer.norm1.bias.unsqueeze(0)),
            },
            "attention": {
                "qkv": {
                    "weight": _tile(qkv_w, dtype=ttnn.bfloat16),
                    "bias": _tile(qkv_b),
                },
                "output": {
                    "dense": {
                        "weight": _tile(out_w_fused, dtype=ttnn.bfloat16),
                        "bias": _rm(out_b_fused),
                    }
                },
            },
            "layernorm_after": {
                "weight": _tile(layer.norm2.weight.unsqueeze(0)),
                "bias": _tile(layer.norm2.bias.unsqueeze(0)),
            },
            "intermediate": {
                "dense": {
                    "weight": _tile(layer.mlp.fc1.weight.transpose(0, 1), dtype=ttnn.bfloat16),
                    "bias": _rm(layer.mlp.fc1.bias),
                }
            },
            "output": {
                "dense": {
                    # Fuse layer_scale2 into FC2: W_fc2 is (in=4096, out=1024)
                    "weight": _tile(
                        layer.mlp.fc2.weight.transpose(0, 1) * ls2.unsqueeze(0),
                        dtype=ttnn.bfloat16,
                    ),
                    "bias": _rm(layer.mlp.fc2.bias * ls2),
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
                "weight": _tile(proj_w, dtype=ttnn.bfloat16),
                "bias": _rm(layer.projection.bias),
            }
        }

        if hasattr(layer, "resize") and hasattr(layer.resize, "weight"):
            # Stride-2 conv for layer 3 (downsample).
            # Conv weight must be bfloat16 ROW_MAJOR -- NOT bfloat8_b.
            rp["resize"] = {
                "weight": _rm(layer.resize.weight, dtype=ttnn.bfloat16),
                "bias": _rm(layer.resize.bias),
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
        # neck.convs[i]: Conv2d with shape (256, in_ch, 3, 3)
        # Keep as conv weight: bfloat16 ROW_MAJOR (NOT bfloat8_b which is invalid for conv)
        nc_w = torch_model.neck.convs[i].weight

        # fusion projection: (256, 256, 1, 1) -> permute -> (256, 256)
        fproj_w = layer.projection.weight.permute(1, 2, 3, 0).reshape(-1, 256)

        nc_bias = torch_model.neck.convs[i].bias
        fp = {
            # Channel normalizer: neck_hidden_sizes[i] -> 256 (3x3 conv)
            "neck_conv": {
                "weight": _rm(nc_w, dtype=ttnn.bfloat16),
                "bias": _rm(nc_bias) if nc_bias is not None else None,
            },
            # Fusion projection: 256 -> 256
            "projection": {
                "weight": _tile(fproj_w, dtype=ttnn.bfloat16),
                "bias": _rm(layer.projection.bias),
            },
            "residual_layer1": {
                "convolution1": {
                    "weight": _rm(layer.residual_layer1.convolution1.weight, dtype=ttnn.bfloat16),
                    "bias": _rm(layer.residual_layer1.convolution1.bias),
                },
                "convolution2": {
                    "weight": _rm(layer.residual_layer1.convolution2.weight, dtype=ttnn.bfloat16),
                    "bias": _rm(layer.residual_layer1.convolution2.bias),
                },
            },
            "residual_layer2": {
                "convolution1": {
                    "weight": _rm(layer.residual_layer2.convolution1.weight, dtype=ttnn.bfloat16),
                    "bias": _rm(layer.residual_layer2.convolution1.bias),
                },
                "convolution2": {
                    "weight": _rm(layer.residual_layer2.convolution2.weight, dtype=ttnn.bfloat16),
                    "bias": _rm(layer.residual_layer2.convolution2.bias),
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
            "bias": _rm(torch_model.head.conv1.bias),
        },
        "conv2": {
            "weight": _rm(torch_model.head.conv2.weight, dtype=ttnn.bfloat16),
            "bias": _rm(torch_model.head.conv2.bias),
        },
        "conv3": {
            "weight": _rm(torch_model.head.conv3.weight, dtype=ttnn.bfloat16),
            "bias": _rm(torch_model.head.conv3.bias),
        },
    }

    return parameters
