"""
SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0
"""

import math
from typing import Optional, Tuple

import torch

import ttnn

from .common import (
    OptimizationConfig,
    convert_to_ttnn_tensor,
    create_l1_memory_config,
    create_sharded_memory_config,
    prepare_linear_weight_bias,
)


def yolos_patch_embeddings(
    pixel_values: ttnn.Tensor,
    parameters,
    config,
    device: ttnn.Device,
    opt_config: OptimizationConfig,
) -> ttnn.Tensor:
    """
    Convert image to patch embeddings using Conv2d projection.

    Args:
        pixel_values: Input image tensor [batch, channels, height, width]
        parameters: Patch embedding parameters (conv weight and bias)
        config: YOLOS configuration
        device: TTNN device
        opt_config: Optimization configuration

    Returns:
        Patch embeddings [batch, num_patches, hidden_size]
    """
    # Stage 2/3: Use optimized linear with optional sharding/L1 when available.
    # PyTorch patch embeddings use NCHW; we reshape into a sequence of
    # flattened patches and apply a linear projection.
    batch_size, _, height, width = pixel_values.shape
    in_channels = config.num_channels
    patch = config.patch_size

    num_patches_h = height // patch
    num_patches_w = width // patch

    # Reshape to [B, C, Hp, patch, Wp, patch]
    embeddings = ttnn.reshape(
        pixel_values,
        (batch_size, in_channels, num_patches_h, patch, num_patches_w, patch),
    )

    # Reorder to [B, Hp, Wp, C, patch, patch]
    embeddings = ttnn.permute(embeddings, (0, 2, 4, 1, 3, 5))

    # Flatten each patch: [B, Hp, Wp, C * patch * patch]
    embeddings = ttnn.reshape(
        embeddings,
        (batch_size, num_patches_h, num_patches_w, in_channels * patch * patch),
    )

    # Select memory configuration for the output activations.
    memory_config = None
    if opt_config.use_sharding:
        memory_config = create_sharded_memory_config(device, strategy="HEIGHT_SHARDED")
    elif opt_config.use_l1_for_intermediates:
        memory_config = create_l1_memory_config()

    # Linear projection per patch: last dim is in_features. We avoid passing
    # a bias directly, since some backends do not support bias with batched
    # inputs, and instead add it manually afterwards.
    embeddings = ttnn.linear(
        embeddings,
        parameters.projection.weight,
        bias=None,
        memory_config=memory_config,
    )

    # Manually add bias with broadcasting over patches.
    if parameters.projection.bias is not None:
        embeddings = ttnn.add(embeddings, parameters.projection.bias)

    # Flatten spatial dimensions: [B, Hp, Wp, hidden_size] -> [B, num_patches, hidden_size]
    batch_size, out_height, out_width, hidden_size = embeddings.shape
    num_patches = out_height * out_width
    embeddings = ttnn.reshape(embeddings, (batch_size, num_patches, hidden_size))

    return embeddings


def yolos_embeddings(
    pixel_values: ttnn.Tensor,
    parameters,
    config,
    device: ttnn.Device,
    opt_config: OptimizationConfig,
) -> ttnn.Tensor:
    """
    Construct full embeddings with CLS token, patches, and detection tokens.

    Args:
        pixel_values: Input image tensor
        parameters: Embedding layer parameters
        config: YOLOS configuration
        device: TTNN device
        opt_config: Optimization configuration

    Returns:
        Full embeddings [batch, seq_len, hidden_size]
        where seq_len = 1 (CLS) + num_patches + num_detection_tokens
    """
    batch_size = pixel_values.shape[0]

    # Get patch embeddings
    patch_embeds = yolos_patch_embeddings(pixel_values, parameters.patch_embeddings, config, device, opt_config)

    # Expand CLS token for batch
    cls_tokens = ttnn.repeat(parameters.cls_token, (batch_size, 1, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Expand detection tokens for batch
    detection_tokens = ttnn.repeat(
        parameters.detection_tokens, (batch_size, 1, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Concatenate: [CLS] + patches + detection_tokens
    embeddings = ttnn.concat([cls_tokens, patch_embeds, detection_tokens], dim=1)

    # Add position embeddings
    embeddings = ttnn.add(embeddings, parameters.position_embeddings)

    # Dropout is typically disabled during inference, so we skip it
    # If needed during training: embeddings = ttnn.dropout(embeddings, p=config.hidden_dropout_prob)

    return embeddings


def yolos_attention(
    hidden_states: ttnn.Tensor,
    parameters,
    config,
    device: ttnn.Device,
    opt_config: OptimizationConfig,
    attention_mask: Optional[ttnn.Tensor] = None,
) -> ttnn.Tensor:
    """
    Multi-head self-attention mechanism.

    Args:
        hidden_states: Input hidden states [batch, seq_len, hidden_size]
        parameters: Attention layer parameters
        config: YOLOS configuration
        device: TTNN device
        opt_config: Optimization configuration
        attention_mask: Optional attention mask

    Returns:
        Attention output [batch, seq_len, hidden_size]
    """
    batch_size, seq_length, hidden_size = hidden_states.shape
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads

    # Stage 2/3: Fused QKV projection
    if opt_config.fuse_qkv and hasattr(parameters, "qkv_weight"):
        # Use pre-fused QKV weight
        qkv = ttnn.linear(
            hidden_states,
            parameters.qkv_weight,
            bias=parameters.qkv_bias if hasattr(parameters, "qkv_bias") else None,
            memory_config=create_l1_memory_config() if opt_config.use_l1_for_intermediates else None,
        )

        # Split into Q, K, V
        qkv = ttnn.reshape(qkv, (batch_size, seq_length, 3, num_heads, head_dim))
        qkv = ttnn.permute(qkv, (2, 0, 3, 1, 4))  # [3, batch, num_heads, seq_len, head_dim]

        query_layer = qkv[0]
        key_layer = qkv[1]
        value_layer = qkv[2]
    else:
        # Stage 1: Separate Q, K, V projections
        query_layer = ttnn.linear(
            hidden_states,
            parameters.query.weight,
            bias=parameters.query.bias,
            memory_config=create_l1_memory_config() if opt_config.use_l1_for_intermediates else None,
        )

        key_layer = ttnn.linear(
            hidden_states,
            parameters.key.weight,
            bias=parameters.key.bias,
            memory_config=create_l1_memory_config() if opt_config.use_l1_for_intermediates else None,
        )

        value_layer = ttnn.linear(
            hidden_states,
            parameters.value.weight,
            bias=parameters.value.bias,
            memory_config=create_l1_memory_config() if opt_config.use_l1_for_intermediates else None,
        )

        # Reshape for multi-head attention: [batch, seq_len, hidden_size] -> [batch, num_heads, seq_len, head_dim]
        query_layer = ttnn.reshape(query_layer, (batch_size, seq_length, num_heads, head_dim))
        query_layer = ttnn.permute(query_layer, (0, 2, 1, 3))

        key_layer = ttnn.reshape(key_layer, (batch_size, seq_length, num_heads, head_dim))
        key_layer = ttnn.permute(key_layer, (0, 2, 1, 3))

        value_layer = ttnn.reshape(value_layer, (batch_size, seq_length, num_heads, head_dim))
        value_layer = ttnn.permute(value_layer, (0, 2, 1, 3))

    # Stage 3: Use fused scaled dot-product attention when available; otherwise
    # fall back to the manual attention path used in Stage 1/2.
    use_fused_sdpa = (
        opt_config.use_fused_sdpa
        and hasattr(ttnn, "transformer")
        and hasattr(ttnn.transformer, "scaled_dot_product_attention")
    )

    context_layer = None
    if use_fused_sdpa:
        try:
            # TTNN's fused SDPA operation uses `attn_mask` rather than the
            # PyTorch-style `attention_mask` keyword.
            context_layer = ttnn.transformer.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                is_causal=False,
                attn_mask=attention_mask,
                memory_config=None,  # SDPA expects DRAM-resident tensors
            )
        except RuntimeError as e:
            # If the backend requires DRAM tensors or SDPA is otherwise
            # unsupported, fall back to the manual path.
            if "Operands to SDPA need to be in DRAM" not in str(e):
                raise

    if context_layer is None:
        # Stage 1/2 (and Stage 3 fallback): Manual attention computation.
        # Compute raw attention scores: Q @ K^T.
        key_layer_transposed = ttnn.permute(key_layer, (0, 1, 3, 2))
        attention_scores = ttnn.matmul(query_layer, key_layer_transposed)

        # HF YOLOS uses an `eager_attention_forward` helper which applies
        # scaling and softmax in float32. When available, we mirror that
        # behavior via TTNN's transformer attention_softmax. Otherwise we
        # fall back to an explicit scale + softmax.
        if hasattr(ttnn, "transformer") and hasattr(ttnn.transformer, "attention_softmax"):
            attention_probs = ttnn.transformer.attention_softmax(
                attention_scores,
                head_size=head_dim,
                attention_mask=attention_mask,
            )
        else:
            # Scale
            scale = 1.0 / math.sqrt(head_dim)
            attention_scores = ttnn.multiply(attention_scores, scale)

            # Apply attention mask if provided
            if attention_mask is not None:
                attention_scores = ttnn.add(attention_scores, attention_mask)

            # Softmax
            attention_probs = ttnn.softmax(attention_scores, dim=-1)

        # Context: attention_probs @ V
        context_layer = ttnn.matmul(attention_probs, value_layer)

        # Clean up intermediate tensors in stage 3
        if opt_config.optimize_tensor_movement:
            ttnn.deallocate(attention_scores)
            ttnn.deallocate(attention_probs)

    # Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
    context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
    context_layer = ttnn.reshape(context_layer, (batch_size, seq_length, hidden_size))

    # Output projection
    attention_output = ttnn.linear(
        context_layer,
        parameters.output.dense.weight,
        bias=parameters.output.dense.bias,
        memory_config=create_l1_memory_config() if opt_config.use_l1_for_intermediates else None,
    )

    # Clean up context layer in stage 3
    if opt_config.optimize_tensor_movement:
        ttnn.deallocate(context_layer)

    return attention_output


def yolos_mlp(
    hidden_states: ttnn.Tensor,
    parameters,
    config,
    device: ttnn.Device,
    opt_config: OptimizationConfig,
) -> ttnn.Tensor:
    """
    Feed-forward MLP layer.

    Args:
        hidden_states: Input hidden states [batch, seq_len, hidden_size]
        parameters: MLP parameters (intermediate and output)
        config: YOLOS configuration
        device: TTNN device
        opt_config: Optimization configuration

    Returns:
        MLP output [batch, seq_len, hidden_size]
    """
    memory_config = create_l1_memory_config() if opt_config.use_l1_for_intermediates else None

    # Stage 2/3: Fused Linear + GELU
    if opt_config.use_operation_fusion:
        # First linear with fused GELU activation
        intermediate = ttnn.linear(
            hidden_states,
            parameters.intermediate.dense.weight,
            bias=parameters.intermediate.dense.bias,
            activation="gelu",
            memory_config=memory_config,
        )
    else:
        # Stage 1: Separate operations
        intermediate = ttnn.linear(
            hidden_states,
            parameters.intermediate.dense.weight,
            bias=parameters.intermediate.dense.bias,
            memory_config=memory_config,
        )
        intermediate = ttnn.gelu(intermediate)

    # Output projection
    output = ttnn.linear(
        intermediate,
        parameters.output.dense.weight,
        bias=parameters.output.dense.bias,
        memory_config=memory_config,
    )

    # Clean up intermediate in stage 3
    if opt_config.optimize_tensor_movement:
        ttnn.deallocate(intermediate)

    return output


def yolos_layer(
    hidden_states: ttnn.Tensor,
    parameters,
    config,
    device: ttnn.Device,
    opt_config: OptimizationConfig,
    attention_mask: Optional[ttnn.Tensor] = None,
) -> ttnn.Tensor:
    """
    Single transformer encoder layer with pre-norm architecture.

    Args:
        hidden_states: Input hidden states [batch, seq_len, hidden_size]
        parameters: Layer parameters
        config: YOLOS configuration
        device: TTNN device
        opt_config: Optimization configuration
        attention_mask: Optional attention mask

    Returns:
        Layer output [batch, seq_len, hidden_size]
    """
    # Pre-norm for attention
    attention_input = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_before.weight,
        bias=parameters.layernorm_before.bias,
        epsilon=config.layer_norm_eps,
    )

    # Self-attention
    attention_output = yolos_attention(
        attention_input,
        parameters.attention,
        config,
        device,
        opt_config,
        attention_mask=attention_mask,
    )

    # First residual connection
    hidden_states = ttnn.add(attention_output, hidden_states)

    # Clean up attention input if optimizing tensor movement
    if opt_config.optimize_tensor_movement:
        ttnn.deallocate(attention_input)
        ttnn.deallocate(attention_output)

    # Pre-norm for MLP
    mlp_input = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_after.weight,
        bias=parameters.layernorm_after.bias,
        epsilon=config.layer_norm_eps,
    )

    # MLP
    mlp_output = yolos_mlp(mlp_input, parameters, config, device, opt_config)

    # Second residual connection
    layer_output = ttnn.add(mlp_output, hidden_states)

    # Clean up intermediates
    if opt_config.optimize_tensor_movement:
        ttnn.deallocate(mlp_input)
        ttnn.deallocate(mlp_output)

    return layer_output


def yolos_encoder(
    hidden_states: ttnn.Tensor,
    parameters,
    config,
    device: ttnn.Device,
    opt_config: OptimizationConfig,
    attention_mask: Optional[ttnn.Tensor] = None,
) -> ttnn.Tensor:
    """
    Stack of transformer encoder layers.

    Args:
        hidden_states: Input embeddings [batch, seq_len, hidden_size]
        parameters: Encoder parameters (layer parameters and optional
            mid-position embeddings)
        config: YOLOS configuration
        device: TTNN device
        opt_config: Optimization configuration
        attention_mask: Optional attention mask

    Returns:
        Encoder output [batch, seq_len, hidden_size]
    """
    mid_pos = getattr(parameters, "mid_position_embeddings", None)

    # Process through all transformer layers
    for layer_idx in range(config.num_hidden_layers):
        layer_params = parameters.layer[layer_idx]

        hidden_states = yolos_layer(
            hidden_states,
            layer_params,
            config,
            device,
            opt_config,
            attention_mask=attention_mask,
        )

        # HF YOLOS adds interpolated mid-position embeddings after each
        # layer except the last when enabled.
        if mid_pos is not None and layer_idx < len(mid_pos):
            hidden_states = ttnn.add(hidden_states, mid_pos[layer_idx])

    return hidden_states


class YolosForObjectDetection:
    """
    TTNN YOLOS model for object detection.

    This class wraps the functional TTNN operations and manages model weights.
    Supports three optimization stages controlled by OptimizationConfig.
    """

    def __init__(
        self,
        config,
        device: ttnn.Device,
        reference_model: Optional[torch.nn.Module] = None,
        opt_config: Optional[OptimizationConfig] = None,
    ):
        """
        Initialize YOLOS model.

        Args:
            config: YOLOS configuration
            device: TTNN device
            reference_model: Optional PyTorch reference model to load weights from
            opt_config: Optimization configuration (defaults to Stage 1)
        """
        self.config = config
        self.device = device
        self.opt_config = opt_config or OptimizationConfig.stage1()

        # Initialize parameters
        self.parameters = self._initialize_parameters(reference_model)

    def _initialize_parameters(self, reference_model):
        """
        Initialize and convert model parameters from PyTorch to TTNN.

        Args:
            reference_model: PyTorch YOLOS model

        Returns:
            Namespace containing all TTNN parameters
        """
        if reference_model is None:
            raise ValueError("Reference model must be provided to load weights")

        from types import SimpleNamespace

        params = SimpleNamespace()

        # Weight dtype: use float32 for high-precision Stage 1 bring-up
        # (opt_config.use_fp32), and bfloat16 for optimized stages.
        # Activation dtype (including optional bfloat8 in Stage 3) is
        # controlled separately when creating input tensors.
        if getattr(self.opt_config, "use_fp32", False):
            dtype = ttnn.float32
        else:
            dtype = ttnn.bfloat16
        memory_config = ttnn.DRAM_MEMORY_CONFIG

        # === Embeddings ===
        params.embeddings = SimpleNamespace()

        # CLS token
        params.embeddings.cls_token = convert_to_ttnn_tensor(
            reference_model.yolos.embeddings.cls_token.data,
            self.device,
            dtype=dtype,
            memory_config=memory_config,
        )

        # Detection tokens
        params.embeddings.detection_tokens = convert_to_ttnn_tensor(
            reference_model.yolos.embeddings.detection_tokens.data,
            self.device,
            dtype=dtype,
            memory_config=memory_config,
        )

        # Position embeddings
        # HF YOLOS interpolates initial position embeddings to match the
        # current image resolution at runtime. Since this demo targets a
        # fixed resolution (config.image_size), we precompute the
        # interpolated position embeddings once using the HF logic and
        # convert that result to TTNN.
        height, width = self.config.image_size
        hf_pos = reference_model.yolos.embeddings.position_embeddings
        if hasattr(reference_model.yolos.embeddings, "interpolation"):
            with torch.no_grad():
                interpolated_pos = reference_model.yolos.embeddings.interpolation(hf_pos, (height, width))
            pos_tensor = interpolated_pos.data
        else:
            pos_tensor = hf_pos.data

        params.embeddings.position_embeddings = convert_to_ttnn_tensor(
            pos_tensor,
            self.device,
            dtype=dtype,
            memory_config=memory_config,
        )

        # Patch embeddings (implemented as linear over flattened patches)
        params.embeddings.patch_embeddings = SimpleNamespace()
        params.embeddings.patch_embeddings.projection = SimpleNamespace()

        conv_weight = reference_model.yolos.embeddings.patch_embeddings.projection.weight.data
        conv_bias = reference_model.yolos.embeddings.patch_embeddings.projection.bias.data

        # PyTorch conv weight is [out_channels, in_channels, kh, kw].
        # We flatten it to a linear weight over patches:
        # [out_channels, in_channels * kh * kw] -> transpose to
        # [in_features, out_features] for ttnn.linear.
        out_channels, in_channels, kh, kw = conv_weight.shape
        conv_weight_flat = conv_weight.reshape(out_channels, in_channels * kh * kw)
        conv_weight_flat = conv_weight_flat.transpose(0, 1).contiguous()

        params.embeddings.patch_embeddings.projection.weight = convert_to_ttnn_tensor(
            conv_weight_flat, self.device, dtype=dtype, memory_config=memory_config
        )
        params.embeddings.patch_embeddings.projection.bias = convert_to_ttnn_tensor(
            conv_bias, self.device, dtype=dtype, memory_config=memory_config
        )

        # === Encoder Layers ===
        params.encoder = SimpleNamespace()
        params.encoder.layer = []

        encoder = reference_model.yolos.encoder

        for layer_idx in range(self.config.num_hidden_layers):
            ref_layer = encoder.layer[layer_idx]
            layer_params = SimpleNamespace()

            # LayerNorm before attention
            layer_params.layernorm_before = SimpleNamespace()
            layer_params.layernorm_before.weight = convert_to_ttnn_tensor(
                ref_layer.layernorm_before.weight.data, self.device, dtype=dtype
            )
            layer_params.layernorm_before.bias = convert_to_ttnn_tensor(
                ref_layer.layernorm_before.bias.data, self.device, dtype=dtype
            )

            # Attention
            layer_params.attention = SimpleNamespace()

            # Stage 2/3: Fuse QKV weights if enabled
            if self.opt_config.fuse_qkv:
                # Concatenate Q, K, V weights
                q_weight, q_bias = prepare_linear_weight_bias(
                    ref_layer.attention.attention.query.weight.data, ref_layer.attention.attention.query.bias.data
                )
                k_weight, k_bias = prepare_linear_weight_bias(
                    ref_layer.attention.attention.key.weight.data, ref_layer.attention.attention.key.bias.data
                )
                v_weight, v_bias = prepare_linear_weight_bias(
                    ref_layer.attention.attention.value.weight.data, ref_layer.attention.attention.value.bias.data
                )

                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=1)
                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

                layer_params.attention.qkv_weight = convert_to_ttnn_tensor(
                    qkv_weight, self.device, dtype=dtype, memory_config=memory_config
                )
                layer_params.attention.qkv_bias = convert_to_ttnn_tensor(
                    qkv_bias, self.device, dtype=dtype, memory_config=memory_config
                )
            else:
                # Stage 1: Separate Q, K, V
                layer_params.attention.query = SimpleNamespace()
                layer_params.attention.key = SimpleNamespace()
                layer_params.attention.value = SimpleNamespace()

                q_weight, q_bias = prepare_linear_weight_bias(
                    ref_layer.attention.attention.query.weight.data, ref_layer.attention.attention.query.bias.data
                )
                layer_params.attention.query.weight = convert_to_ttnn_tensor(
                    q_weight, self.device, dtype=dtype, memory_config=memory_config
                )
                layer_params.attention.query.bias = convert_to_ttnn_tensor(
                    q_bias, self.device, dtype=dtype, memory_config=memory_config
                )

                k_weight, k_bias = prepare_linear_weight_bias(
                    ref_layer.attention.attention.key.weight.data, ref_layer.attention.attention.key.bias.data
                )
                layer_params.attention.key.weight = convert_to_ttnn_tensor(
                    k_weight, self.device, dtype=dtype, memory_config=memory_config
                )
                layer_params.attention.key.bias = convert_to_ttnn_tensor(
                    k_bias, self.device, dtype=dtype, memory_config=memory_config
                )

                v_weight, v_bias = prepare_linear_weight_bias(
                    ref_layer.attention.attention.value.weight.data, ref_layer.attention.attention.value.bias.data
                )
                layer_params.attention.value.weight = convert_to_ttnn_tensor(
                    v_weight, self.device, dtype=dtype, memory_config=memory_config
                )
                layer_params.attention.value.bias = convert_to_ttnn_tensor(
                    v_bias, self.device, dtype=dtype, memory_config=memory_config
                )

            # Attention output projection
            layer_params.attention.output = SimpleNamespace()
            layer_params.attention.output.dense = SimpleNamespace()

            out_weight, out_bias = prepare_linear_weight_bias(
                ref_layer.attention.output.dense.weight.data, ref_layer.attention.output.dense.bias.data
            )
            layer_params.attention.output.dense.weight = convert_to_ttnn_tensor(
                out_weight, self.device, dtype=dtype, memory_config=memory_config
            )
            layer_params.attention.output.dense.bias = convert_to_ttnn_tensor(
                out_bias, self.device, dtype=dtype, memory_config=memory_config
            )

            # LayerNorm after attention
            layer_params.layernorm_after = SimpleNamespace()
            layer_params.layernorm_after.weight = convert_to_ttnn_tensor(
                ref_layer.layernorm_after.weight.data, self.device, dtype=dtype
            )
            layer_params.layernorm_after.bias = convert_to_ttnn_tensor(
                ref_layer.layernorm_after.bias.data, self.device, dtype=dtype
            )

            # MLP
            layer_params.intermediate = SimpleNamespace()
            layer_params.intermediate.dense = SimpleNamespace()

            inter_weight, inter_bias = prepare_linear_weight_bias(
                ref_layer.intermediate.dense.weight.data, ref_layer.intermediate.dense.bias.data
            )
            layer_params.intermediate.dense.weight = convert_to_ttnn_tensor(
                inter_weight, self.device, dtype=dtype, memory_config=memory_config
            )
            layer_params.intermediate.dense.bias = convert_to_ttnn_tensor(
                inter_bias, self.device, dtype=dtype, memory_config=memory_config
            )

            layer_params.output = SimpleNamespace()
            layer_params.output.dense = SimpleNamespace()

            output_weight, output_bias = prepare_linear_weight_bias(
                ref_layer.output.dense.weight.data, ref_layer.output.dense.bias.data
            )
            layer_params.output.dense.weight = convert_to_ttnn_tensor(
                output_weight, self.device, dtype=dtype, memory_config=memory_config
            )
            layer_params.output.dense.bias = convert_to_ttnn_tensor(
                output_bias, self.device, dtype=dtype, memory_config=memory_config
            )

            params.encoder.layer.append(layer_params)

        # Mid-position embeddings (optional YOLOS-specific feature).
        # HF YOLOS can add learned mid-position embeddings after each
        # encoder layer except the last. We precompute and convert them
        # for the fixed image resolution if present.
        if getattr(encoder, "mid_position_embeddings", None) is not None and hasattr(encoder, "interpolation"):
            with torch.no_grad():
                interpolated_mid = encoder.interpolation(encoder.mid_position_embeddings, (height, width))
            # interpolated_mid: [num_layers-1, 1, seq_len, hidden]
            params.encoder.mid_position_embeddings = []
            for i in range(interpolated_mid.shape[0]):
                mid = interpolated_mid[i]  # [1, seq_len, hidden]
                params.encoder.mid_position_embeddings.append(
                    convert_to_ttnn_tensor(
                        mid,
                        self.device,
                        dtype=dtype,
                        memory_config=memory_config,
                    )
                )

        # === Final LayerNorm ===
        params.layernorm = SimpleNamespace()
        params.layernorm.weight = convert_to_ttnn_tensor(
            reference_model.yolos.layernorm.weight.data, self.device, dtype=dtype
        )
        params.layernorm.bias = convert_to_ttnn_tensor(
            reference_model.yolos.layernorm.bias.data, self.device, dtype=dtype
        )

        # === Detection Heads ===
        # Classification head (HF uses YolosMLPPredictionHead with `layers`)
        params.class_labels_classifier = SimpleNamespace()
        params.class_labels_classifier.layers = []

        for layer in reference_model.class_labels_classifier.layers:
            if isinstance(layer, torch.nn.Linear):
                layer_param = SimpleNamespace()
                weight, bias = prepare_linear_weight_bias(layer.weight.data, layer.bias.data)
                layer_param.weight = convert_to_ttnn_tensor(
                    weight, self.device, dtype=dtype, memory_config=memory_config
                )
                layer_param.bias = convert_to_ttnn_tensor(bias, self.device, dtype=dtype, memory_config=memory_config)
                params.class_labels_classifier.layers.append(layer_param)

        # Bounding box head (HF also uses YolosMLPPredictionHead with `layers`)
        params.bbox_predictor = SimpleNamespace()
        params.bbox_predictor.layers = []

        for i, layer in enumerate(reference_model.bbox_predictor.layers):
            if isinstance(layer, torch.nn.Linear):
                layer_param = SimpleNamespace()
                weight, bias = prepare_linear_weight_bias(layer.weight.data, layer.bias.data)
                layer_param.weight = convert_to_ttnn_tensor(
                    weight, self.device, dtype=dtype, memory_config=memory_config
                )
                layer_param.bias = convert_to_ttnn_tensor(bias, self.device, dtype=dtype, memory_config=memory_config)
                params.bbox_predictor.layers.append(layer_param)

        return params

    def __call__(
        self,
        pixel_values: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Forward pass of YOLOS model.

        Args:
            pixel_values: Input images [batch, channels, height, width]
            attention_mask: Optional attention mask

        Returns:
            Tuple of (logits, pred_boxes)
            - logits: Class predictions [batch, num_detection_tokens, num_classes+1]
            - pred_boxes: Bounding box predictions [batch, num_detection_tokens, 4]
        """
        # Embeddings
        hidden_states = yolos_embeddings(
            pixel_values,
            self.parameters.embeddings,
            self.config,
            self.device,
            self.opt_config,
        )

        # Encoder
        hidden_states = yolos_encoder(
            hidden_states,
            self.parameters.encoder,
            self.config,
            self.device,
            self.opt_config,
            attention_mask=attention_mask,
        )

        # Final LayerNorm
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.parameters.layernorm.weight,
            bias=self.parameters.layernorm.bias,
            epsilon=self.config.layer_norm_eps,
        )

        # Extract detection tokens (last num_detection_tokens in sequence)
        # Sequence is: [CLS] + patches + detection_tokens
        detection_tokens = hidden_states[:, -self.config.num_detection_tokens :, :]

        # Classification head (MLP)
        logits = detection_tokens
        for i, layer_params in enumerate(self.parameters.class_labels_classifier.layers):
            logits = ttnn.linear(
                logits,
                layer_params.weight,
                bias=layer_params.bias,
            )
            # Apply ReLU for all but the last layer
            if i < len(self.parameters.class_labels_classifier.layers) - 1:
                logits = ttnn.relu(logits)

        # Bounding box head (MLP)
        bbox_pred = detection_tokens
        for i, layer_params in enumerate(self.parameters.bbox_predictor.layers):
            bbox_pred = ttnn.linear(
                bbox_pred,
                layer_params.weight,
                bias=layer_params.bias,
            )
            # Apply ReLU for first two layers
            if i < len(self.parameters.bbox_predictor.layers) - 1:
                bbox_pred = ttnn.relu(bbox_pred)
            else:
                # Sigmoid for final layer to normalize bbox coordinates
                bbox_pred = ttnn.sigmoid(bbox_pred)

        return logits, bbox_pred

    def predict(
        self,
        pixel_values: ttnn.Tensor,
        threshold: float = 0.7,
    ):
        """
        Run inference and filter predictions by confidence threshold.

        Args:
            pixel_values: Input images [batch, channels, height, width]
            threshold: Confidence threshold for filtering detections

        Returns:
            Dictionary with scores, labels, boxes, and keep mask
        """
        logits, pred_boxes = self(pixel_values)

        # Convert to torch for post-processing
        logits_torch = ttnn.to_torch(logits)
        boxes_torch = ttnn.to_torch(pred_boxes)

        # Get probabilities
        probs = torch.softmax(logits_torch, dim=-1)

        # Get max probability and corresponding class (excluding background class)
        scores, labels = probs[..., :-1].max(-1)

        # Filter by threshold
        keep = scores > threshold

        return {
            "scores": scores,
            "labels": labels,
            "boxes": boxes_torch,
            "keep": keep,
        }
