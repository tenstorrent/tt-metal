# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Linear Attention (GatedDeltaNet) implementation for Qwen3.5-9B.

Based on HuggingFace transformers reference implementation.

Architecture (from reference):
    1. Project input to FUSED QKV (Query, Key, Value): in_proj_qkv → [key_dim*2 + value_dim]
    2. Apply Conv1d on FUSED QKV
    3. Split into Q, K, V
    4. Compute beta (from in_proj_b) and g (from in_proj_a)
    5. Repeat Q,K heads to match V heads if needed (Grouped Query Attention - GQA)
    6. Apply delta rule (chunk or recurrent)
    7. Apply RMSNormGated with z gate (from in_proj_z)
    8. Project output with out_proj

Key dimensions (Qwen3.5-9B):
    hidden_size: 4096
    key_dim: 2048 = num_k_heads * head_k_dim = 16 * 128
    value_dim: 4096 = num_v_heads * head_v_dim = 32 * 128
    conv_dim: 8192 = key_dim*2 + value_dim (Conv operates on FUSED QKV)

Tensor Layouts:
    - TILE_LAYOUT: Optimized for 2D tiled matrix operations (matmul, attention).
      Data is arranged in tiles for efficient parallel computation on TT hardware.
    - ROW_MAJOR_LAYOUT: Sequential memory layout for operations like conv1d that
      process data in a linear fashion. Layout conversions happen at operation boundaries.

Usage:
    model = TtLinearAttention(mesh_device, state_dict, args, layer_idx)
    output = model(hidden_states)
"""

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule

# Constants from HuggingFace config
EPSILON = 1e-6  # Numerical stability epsilon for RMSNorm (config.rms_norm_eps)

# Chunk size for parallel processing in prefill mode. Value of 64 matches HuggingFace
# reference implementation and balances parallelism with memory usage on TT hardware.
CHUNK_SIZE = 64

# Exponent for computing attention scale factor: scale = 1 / head_dim^0.5 = 1/sqrt(head_dim)
# This is the standard scaled dot-product attention normalization from "Attention Is All You Need"
ATTENTION_SCALE_EXPONENT = 0.5


class TtLinearAttention(LightweightModule):
    """
    GatedDeltaNet Linear Attention for Qwen3.5.

    Matches HuggingFace Qwen3_5GatedDeltaNet implementation exactly.
    """

    def __init__(self, mesh_device, state_dict, args, layer_idx, dtype=ttnn.bfloat16):
        """
        Initialize GatedDeltaNet Linear Attention with weights.

        Args:
            mesh_device: TT device or mesh of devices
            state_dict: HuggingFace model state dictionary with weights
            args: Model arguments (Qwen35ModelArgs)
            layer_idx: Index of this layer in the model
            dtype: Data type for weights (default: bfloat16)

        Expected state_dict keys (prefix = "model.layers.{layer_idx}.linear_attn"):
            Note: Keys from AutoModelForCausalLM are already in the correct format.
            Use standardize_hf_keys_multimodal() from load_checkpoints.py if loading
            from other sources.

            - in_proj_qkv.weight: [conv_dim, hidden_size]
            - in_proj_z.weight: [value_dim, hidden_size]
            - in_proj_b.weight: [num_v_heads, hidden_size]
            - in_proj_a.weight: [num_v_heads, hidden_size]
            - conv1d.weight: [conv_dim, 1, conv_kernel_size]
            - out_proj.weight: [hidden_size, value_dim]
            - A_log: [num_v_heads]
            - dt_bias: [num_v_heads]
            - norm.weight: [head_v_dim]
        """
        super().__init__()

        self.args = args
        self.layer_idx = layer_idx
        self.device = mesh_device
        self.dtype = dtype

        # Verify this is a linear attention layer
        if not args.is_linear_attention_layer(layer_idx):
            raise ValueError(f"Layer {layer_idx} is not a linear attention layer")

        # Reference dimensions (from HF implementation)
        self.hidden_size = args.dim  # 4096
        self.num_k_heads = args.linear_num_key_heads  # 16
        self.num_v_heads = args.linear_num_value_heads  # 32
        self.head_k_dim = args.linear_key_head_dim  # 128
        self.head_v_dim = args.linear_value_head_dim  # 128
        self.key_dim = args.linear_key_dim  # 2048
        self.value_dim = args.linear_value_dim  # 4096
        self.conv_kernel_size = args.linear_conv_kernel_dim  # 4
        self.conv_dim = self.key_dim * 2 + self.value_dim  # 8192

        # State dict prefix for loading weights
        prefix = args.get_state_dict_prefix("LinearAttention", layer_idx)

        logger.info(f"Initializing GatedDeltaNet (Layer {layer_idx})")
        logger.info(f"  hidden_size: {self.hidden_size}")
        logger.info(f"  num_k_heads: {self.num_k_heads}, num_v_heads: {self.num_v_heads}")
        logger.info(f"  key_dim: {self.key_dim}, value_dim: {self.value_dim}")
        logger.info(f"  conv_dim: {self.conv_dim} (for fused QKV)")

        # Precompute causal mask for chunk attention (constant, can be computed once)
        # Upper triangular mask: True above diagonal (positions to mask out for causality)
        self.causal_mask = torch.triu(torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool), diagonal=1)

        # Conv1d configuration for on-device execution
        # Use default config to avoid L1 memory issues with large conv_dim
        self.conv1d_config = None
        self.conv1d_compute_config = None

        # === Load weights from state dict ===
        # Projection weights (transpose for ttnn matmul: [in, out] -> [out, in])
        self.in_proj_qkv = ttnn.from_torch(
            state_dict[f"{prefix}.in_proj_qkv.weight"].transpose(0, 1),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )
        self.in_proj_z = ttnn.from_torch(
            state_dict[f"{prefix}.in_proj_z.weight"].transpose(0, 1),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )
        self.in_proj_b = ttnn.from_torch(
            state_dict[f"{prefix}.in_proj_b.weight"].transpose(0, 1),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )
        self.in_proj_a = ttnn.from_torch(
            state_dict[f"{prefix}.in_proj_a.weight"].transpose(0, 1),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )

        # Conv1d weight - keep as torch tensor, will be converted to ttnn when used
        # Shape: [conv_dim, 1, conv_kernel_size] = [8192, 1, 4]
        self.conv1d_weight_torch = state_dict[f"{prefix}.conv1d.weight"]
        # Initialize as None, will be created on first use
        self.conv1d_weight = None

        # Output projection
        self.out_proj = ttnn.from_torch(
            state_dict[f"{prefix}.out_proj.weight"].transpose(0, 1),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )

        # SSM parameters
        # A_log: State-Space Model state matrix parameter - controls decay dynamics.
        # exp(A_log) determines how quickly past information decays in the recurrent state.
        self.A_log = ttnn.from_torch(state_dict[f"{prefix}.A_log"], dtype=self.dtype, device=mesh_device)
        # dt_bias: Delta-time bias that adjusts decay rate. Larger values = slower decay,
        # allowing information to persist longer in the recurrent state.
        self.dt_bias = ttnn.from_torch(state_dict[f"{prefix}.dt_bias"], dtype=self.dtype, device=mesh_device)

        # RMSNorm weight
        self.norm_weight = ttnn.from_torch(state_dict[f"{prefix}.norm.weight"], dtype=self.dtype, device=mesh_device)

        logger.info(f"Loaded GatedDeltaNet weights for layer {layer_idx}")

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cache_params=None,
        attention_mask=None,
    ):
        """
        Forward pass matching HF reference.

        Args:
            hidden_states: [batch, seq_len, hidden_size] - Input tensor
            cache_params: Optional dict with keys:
                - 'conv_state': [batch, conv_dim, kernel_size-1] or None
                - 'recurrent_state': [batch, num_v_heads, head_k_dim, head_v_dim] or None
                - 'has_previous_state': bool (set to True by forward pass after first call)
            attention_mask: Not used in linear attention (causal masking is implicit)

        Returns:
            output: [batch, seq_len, hidden_size] - Attention output
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Determine if we're using cached states (decode mode)
        use_precomputed_states = (
            cache_params is not None and cache_params.get("has_previous_state", False) and seq_len == 1
        )

        # Get cached states if available
        conv_state = cache_params.get("conv_state") if cache_params else None
        recurrent_state = cache_params.get("recurrent_state") if cache_params else None

        # 1. Project input to FUSED QKV
        mixed_qkv = ttnn.linear(hidden_states, self.in_proj_qkv)  # [batch, seq_len, conv_dim]

        # 2. Project z, b, a
        z = ttnn.linear(hidden_states, self.in_proj_z)  # [batch, seq_len, value_dim]
        b = ttnn.linear(hidden_states, self.in_proj_b)  # [batch, seq_len, num_v_heads]
        a = ttnn.linear(hidden_states, self.in_proj_a)  # [batch, seq_len, num_v_heads]

        # 3. Apply causal conv1d on FUSED QKV
        mixed_qkv, new_conv_state = self._causal_conv1d(mixed_qkv, conv_state, use_precomputed_states)

        # 4-5. Split QKV, reshape to heads, and repeat Q/K (OPTIMIZED: single combined operation)
        # Output: [batch, num_v_heads, seq_len, head_dim] - Q and K already repeated to 32 heads
        query, key, value = self._split_qkv_to_heads(mixed_qkv)

        # 6. Compute beta and g
        beta = ttnn.sigmoid(b)  # [batch, seq_len, num_v_heads]
        g = self._compute_g(a)  # [batch, seq_len, num_v_heads]

        # 8. Apply delta rule
        if not use_precomputed_states:
            # Prefill mode: chunk-based processing
            core_attn_out, new_recurrent_state = self._chunk_gated_delta_rule(
                query, key, value, g, beta, initial_state=None, output_final_state=(cache_params is not None)
            )
        else:
            # Decode mode: recurrent processing
            core_attn_out, new_recurrent_state = self._recurrent_gated_delta_rule(
                query, key, value, g, beta, initial_state=recurrent_state, output_final_state=(cache_params is not None)
            )

        # 9. Apply RMSNormGated
        # Reshape for norm: [batch*seq_len, head_v_dim]
        core_attn_out = ttnn.reshape(core_attn_out, (batch_size * seq_len * self.num_v_heads, self.head_v_dim))
        z = ttnn.reshape(z, (batch_size * seq_len, self.value_dim))
        z = ttnn.reshape(z, (batch_size * seq_len * self.num_v_heads, self.head_v_dim))

        core_attn_out = self._rms_norm_gated(core_attn_out, z)

        # Reshape back: [batch, seq_len, value_dim]
        core_attn_out = ttnn.reshape(core_attn_out, (batch_size, seq_len, self.value_dim))

        # 10. Output projection
        output = ttnn.linear(core_attn_out, self.out_proj)

        # Update cache if needed
        if cache_params is not None:
            cache_params["conv_state"] = new_conv_state
            cache_params["recurrent_state"] = new_recurrent_state
            cache_params["has_previous_state"] = True

        return output

    def _causal_conv1d(self, x, conv_state, use_cached):
        """
        Apply causal conv1d on fused QKV tensor using on-device ttnn.conv1d.

        OPTIMIZED: Zero permutes! Works directly with [batch, seq_len, conv_dim] shape.

        Reference (from HF):
        - Prefill: F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        - Decode: torch_causal_conv1d_update with state management

        Args:
            x: [batch, seq_len, conv_dim] - Input tensor in TILE_LAYOUT
            conv_state: [batch, kernel_size-1, conv_dim] or None - Previous conv state for decode mode
            use_cached: bool - True for decode mode (seq_len=1), False for prefill mode

        Returns:
            tuple:
                - x_conv: [batch, seq_len, conv_dim] - Convolved and activated output in TILE_LAYOUT
                - new_conv_state: [batch, kernel_size-1, conv_dim] - Updated state for next iteration
        """
        batch_size, seq_len, _ = x.shape

        # Prepare conv1d weight on first use
        if self.conv1d_weight is None:
            # Weight shape: [conv_dim, 1, kernel_size] for depthwise conv
            self.conv1d_weight = ttnn.from_torch(
                self.conv1d_weight_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        # Prepare input with causal padding - work directly in [B, S, C] format!
        padding = self.conv_kernel_size - 1
        if use_cached and conv_state is not None:
            # Decode mode: concatenate with conv_state on sequence dimension
            x_with_state = ttnn.concat([conv_state, x], dim=1)  # [B, state_len + seq_len, C]
        else:
            # Prefill mode: left-pad with zeros for causality
            zeros = ttnn.zeros(
                (batch_size, padding, self.conv_dim),
                dtype=self.dtype,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
            )
            x_with_state = ttnn.concat([zeros, x], dim=1)  # [B, padding + seq_len, C]

        input_length = x_with_state.shape[1]

        # Convert to ROW_MAJOR layout for conv1d
        x_for_conv = ttnn.to_layout(x_with_state, ttnn.ROW_MAJOR_LAYOUT)

        # Reshape to [B, L, 1, C] - just adding a dimension, no permute needed!
        x_for_conv = ttnn.reshape(x_for_conv, [batch_size, input_length, 1, self.conv_dim])

        # Apply depthwise conv1d: groups=conv_dim means each channel has its own kernel,
        # so channels are convolved independently (no cross-channel mixing).
        [x_conv_out, out_length, [weights_device, _]] = ttnn.conv1d(
            input_tensor=x_for_conv,
            weight_tensor=self.conv1d_weight,
            device=self.device,
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            batch_size=batch_size,
            input_length=input_length,
            kernel_size=self.conv_kernel_size,
            stride=1,
            padding=0,
            groups=self.conv_dim,  # Depthwise: each of 8192 channels convolved separately
            bias_tensor=None,
            conv_config=self.conv1d_config,
            compute_config=self.conv1d_compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        # Update weight reference for next call
        self.conv1d_weight = weights_device

        # Convert back from sharded output
        x_conv_out = ttnn.sharded_to_interleaved(x_conv_out)

        # Reshape to [batch, out_length, conv_dim] - just removing the dimension
        x_conv = ttnn.reshape(x_conv_out, [batch_size, out_length, self.conv_dim])

        # Slice to get the correct output length (already in [B, S, C] format!)
        if use_cached:
            # Take last seq_len positions
            start_idx = out_length - seq_len
            x_conv = ttnn.slice(x_conv, (0, start_idx, 0), (batch_size, out_length, self.conv_dim))
        else:
            # Take first seq_len positions (causal) - output should already be seq_len
            if out_length > seq_len:
                x_conv = ttnn.slice(x_conv, (0, 0, 0), (batch_size, seq_len, self.conv_dim))

        # Convert to TILE_LAYOUT and apply SiLU - no permute needed!
        x_conv = ttnn.to_layout(x_conv, ttnn.TILE_LAYOUT)
        x_conv = ttnn.silu(x_conv)

        # Create new conv_state: last (kernel_size-1) positions of input
        # Store in [B, state_len, C] format to match our optimized flow
        state_len = self.conv_kernel_size - 1
        if seq_len >= state_len:
            new_conv_state = ttnn.slice(x, (0, seq_len - state_len, 0), (batch_size, seq_len, self.conv_dim))
        else:
            # Pad if sequence is shorter than state length
            pad_len = state_len - seq_len
            zeros_state = ttnn.zeros(
                (batch_size, pad_len, self.conv_dim),
                dtype=self.dtype,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
            )
            new_conv_state = ttnn.concat([zeros_state, x], dim=1)

        return x_conv, new_conv_state

    def _l2norm(self, x, dim=-1, eps=EPSILON):
        """
        L2 normalization: x / sqrt(sum(x^2) + eps)

        Used to normalize Q and K before applying delta rule for numerical stability.

        Reference (from HF):
            inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
            return x * inv_norm

        Args:
            x: Input tensor (typically Q or K)
            dim: Dimension to normalize over (default: -1, the feature dimension)
            eps: Small constant for numerical stability (default: EPSILON=1e-6)

        Returns:
            Normalized tensor with same shape as input
        """
        # Compute sum of squares along the specified dimension
        x_squared = ttnn.mul(x, x)
        sum_squared = ttnn.sum(x_squared, dim=dim, keepdim=True)

        # Add eps for numerical stability and compute inverse square root
        sum_squared_eps = ttnn.add(sum_squared, eps)
        inv_norm = ttnn.rsqrt(sum_squared_eps)

        # Normalize: x / ||x||_2
        return ttnn.mul(x, inv_norm)

    def _split_qkv_to_heads(self, mixed_qkv):
        """
        Split fused QKV tensor and reshape directly to head format.

        OPTIMIZED vs separate operations: Instead of calling _split_qkv() followed by
        separate reshape and _repeat_interleave operations in forward(), this function
        performs all transformations in one place, outputting tensors already in the
        [B, H, S, D] format needed by delta rule functions (no additional permutes needed).

        Q and K have 16 heads, V has 32 heads. This is Grouped Query Attention (GQA) where
        multiple V heads share the same Q/K heads. We repeat Q/K to match V's head count
        for parallel computation.

        Args:
            mixed_qkv: [batch, seq_len, conv_dim=8192]

        Returns:
            query: [batch, num_v_heads, seq_len, head_k_dim] = [B, 32, S, 128]
            key:   [batch, num_v_heads, seq_len, head_k_dim] = [B, 32, S, 128]
            value: [batch, num_v_heads, seq_len, head_v_dim] = [B, 32, S, 128]
        """
        batch_size, seq_len, _ = mixed_qkv.shape

        # Split along last dimension: conv_dim = key_dim + key_dim + value_dim
        # = 2048 + 2048 + 4096 = 8192
        q_flat = ttnn.slice(mixed_qkv, (0, 0, 0), (batch_size, seq_len, self.key_dim))
        k_flat = ttnn.slice(mixed_qkv, (0, 0, self.key_dim), (batch_size, seq_len, self.key_dim * 2))
        v_flat = ttnn.slice(mixed_qkv, (0, 0, self.key_dim * 2), (batch_size, seq_len, self.conv_dim))

        # Q/K have 16 heads, V has 32 heads (Grouped Query Attention).
        # Repeat Q/K heads to match V for parallel attention computation.
        # Q: [B, S, 2048] -> [B, S, 16, 128] -> [B, S, 32, 128] via repeat
        repeat_factor = self.num_v_heads // self.num_k_heads  # 32 // 16 = 2

        # Reshape to heads format with extra dim for repeat
        q_heads = ttnn.reshape(q_flat, (batch_size, seq_len, self.num_k_heads, 1, self.head_k_dim))
        k_heads = ttnn.reshape(k_flat, (batch_size, seq_len, self.num_k_heads, 1, self.head_k_dim))

        # Repeat along the new dimension
        q_heads = ttnn.repeat(q_heads, (1, 1, 1, repeat_factor, 1))
        k_heads = ttnn.repeat(k_heads, (1, 1, 1, repeat_factor, 1))

        # Reshape to final head format [B, S, num_v_heads, head_dim]
        q_heads = ttnn.reshape(q_heads, (batch_size, seq_len, self.num_v_heads, self.head_k_dim))
        k_heads = ttnn.reshape(k_heads, (batch_size, seq_len, self.num_v_heads, self.head_k_dim))
        v_heads = ttnn.reshape(v_flat, (batch_size, seq_len, self.num_v_heads, self.head_v_dim))

        # Permute to [B, num_heads, S, head_dim] for efficient batched attention
        query = ttnn.permute(q_heads, (0, 2, 1, 3))
        key = ttnn.permute(k_heads, (0, 2, 1, 3))
        value = ttnn.permute(v_heads, (0, 2, 1, 3))

        return query, key, value

    def _compute_g(self, a):
        """
        Compute decay rate (g) for temporal gating in delta rule.

        The decay rate controls how quickly information from past tokens decays in the
        recurrent state. Negative values ensure exponential decay over time.

        Reference (from HF):
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        Args:
            a: [batch, seq_len, num_v_heads] - Alpha gating values from in_proj_a

        Returns:
            g: [batch, seq_len, num_v_heads] - Decay rate (negative values)

        Shape notes:
            - dt_bias [num_v_heads] is broadcast to [1, 1, num_v_heads]
            - A_log [num_v_heads] is broadcast to [1, 1, num_v_heads]
            - All operations preserve [batch, seq_len, num_v_heads] shape
        """
        # Broadcast dt_bias (delta-time bias) to match input shape
        # dt_bias: [num_v_heads] -> [1, 1, num_v_heads]
        dt_bias_expanded = ttnn.reshape(self.dt_bias, (1, 1, self.num_v_heads))

        # Add dt_bias to alpha values
        a_biased = ttnn.add(a, dt_bias_expanded)

        # Apply softplus for smooth, positive output: softplus(x) = ln(1 + exp(x))
        a_softplus = ttnn.softplus(a_biased)

        # Compute SSM state matrix parameter: exp(A_log)
        A_exp = ttnn.exp(self.A_log)
        A_exp_expanded = ttnn.reshape(A_exp, (1, 1, self.num_v_heads))

        # Compute decay rate: g = -A_exp * softplus(a + dt_bias)
        # Negative sign ensures exponential decay (exp(g * dt) < 1 for dt > 0)
        g = ttnn.mul(A_exp_expanded, a_softplus)
        g = ttnn.neg(g)

        return g

    def _repeat_interleave(self, x, repeat_factor, dim):
        """
        Repeat tensor along dimension.

        Reference: query.repeat_interleave(repeat_factor, dim=2)
        Example: [batch, seq, 16, 128] -> [batch, seq, 32, 128] with repeat=2, dim=2

        Args:
            x: Input tensor [batch, seq_len, num_heads, head_dim]
            repeat_factor: Number of times to repeat (2 for Qwen3.5)
            dim: Dimension to repeat along (2 for head dimension)

        Returns:
            Repeated tensor [batch, seq_len, num_heads*repeat_factor, head_dim]
        """
        if repeat_factor == 1:
            return x

        # Get shape
        shape = list(x.shape)
        batch, seq_len, num_heads, head_dim = shape

        # Reshape to [batch, seq_len, num_heads, 1, head_dim]
        x_expanded = ttnn.reshape(x, (batch, seq_len, num_heads, 1, head_dim))

        # Repeat along new dimension
        x_repeated = ttnn.repeat(x_expanded, (1, 1, 1, repeat_factor, 1))

        # Reshape to [batch, seq_len, num_heads*repeat_factor, head_dim]
        x_output = ttnn.reshape(x_repeated, (batch, seq_len, num_heads * repeat_factor, head_dim))

        return x_output

    def _chunk_gated_delta_rule(self, q, k, v, g, beta, initial_state, output_final_state):
        """
        Chunk-based delta rule for prefill mode (efficient parallel processing).

        Processes sequence in chunks of CHUNK_SIZE tokens, applying delta rule with
        exponential decay within each chunk and maintaining recurrent state across chunks.

        Reference (from HF torch_chunk_gated_delta_rule):
        - Process sequence in chunks of 64 tokens
        - Within each chunk: apply delta rule with decay masks
        - Across chunks: maintain recurrent state

        Args:
            q, k: [batch, num_v_heads, seq_len, head_k_dim] - Query and key tensors (already transposed)
            v: [batch, num_v_heads, seq_len, head_v_dim] - Value tensor (already transposed)
            g, beta: [batch, seq_len, num_v_heads] - Decay rate and beta gating
            initial_state: [batch, num_v_heads, head_k_dim, head_v_dim] or None - Previous recurrent state
            output_final_state: bool - Whether to return final state

        Returns:
            tuple:
                - output: [batch, seq_len, num_v_heads, head_v_dim] - Attention output
                - final_state: [batch, num_v_heads, head_k_dim, head_v_dim] or None - Updated state
        """
        # QKV already in [batch, num_heads, seq_len, head_dim] format from _split_qkv_to_heads
        batch_size, num_heads, seq_len, k_head_dim = q.shape
        v_head_dim = v.shape[-1]

        # Apply L2 normalization to Q and K for numerical stability
        q = self._l2norm(q, dim=-1, eps=EPSILON)
        k = self._l2norm(k, dim=-1, eps=EPSILON)

        # Only permute beta and g (QKV already in correct format)
        beta = ttnn.permute(beta, (0, 2, 1))  # [batch, num_heads, seq_len]
        g = ttnn.permute(g, (0, 2, 1))  # [batch, num_heads, seq_len]

        # Pad sequence to multiple of CHUNK_SIZE for efficient chunk processing
        pad_size = (CHUNK_SIZE - seq_len % CHUNK_SIZE) % CHUNK_SIZE
        if pad_size > 0:
            q = ttnn.pad(q, [(0, 0), (0, 0), (0, pad_size), (0, 0)], 0.0)
            k = ttnn.pad(k, [(0, 0), (0, 0), (0, pad_size), (0, 0)], 0.0)
            v = ttnn.pad(v, [(0, 0), (0, 0), (0, pad_size), (0, 0)], 0.0)
            beta = ttnn.pad(beta, [(0, 0), (0, 0), (0, pad_size)], 0.0)
            g = ttnn.pad(g, [(0, 0), (0, 0), (0, pad_size)], 0.0)

        total_seq_len = seq_len + pad_size
        num_chunks = total_seq_len // CHUNK_SIZE

        # Apply scaled dot-product attention normalization: scale = 1/sqrt(head_dim)
        scale = 1.0 / (k_head_dim**ATTENTION_SCALE_EXPONENT)
        q = ttnn.mul(q, scale)

        # Compute v_beta and k_beta
        beta_expanded = ttnn.reshape(beta, (batch_size, num_heads, total_seq_len, 1))
        v_beta = ttnn.mul(v, beta_expanded)
        k_beta = ttnn.mul(k, beta_expanded)

        # Reshape to chunks: [batch, num_heads, num_chunks, CHUNK_SIZE, head_dim]
        q = ttnn.reshape(q, (batch_size, num_heads, num_chunks, CHUNK_SIZE, k_head_dim))
        k = ttnn.reshape(k, (batch_size, num_heads, num_chunks, CHUNK_SIZE, k_head_dim))
        v = ttnn.reshape(v, (batch_size, num_heads, num_chunks, CHUNK_SIZE, v_head_dim))
        k_beta = ttnn.reshape(k_beta, (batch_size, num_heads, num_chunks, CHUNK_SIZE, k_head_dim))
        v_beta = ttnn.reshape(v_beta, (batch_size, num_heads, num_chunks, CHUNK_SIZE, v_head_dim))
        g = ttnn.reshape(g, (batch_size, num_heads, num_chunks, CHUNK_SIZE))

        # Compute cumulative g for decay
        g = ttnn.cumsum(g, dim=-1)

        # Initialize state
        if initial_state is None:
            last_recurrent_state = ttnn.zeros(
                (batch_size, num_heads, k_head_dim, v_head_dim),
                dtype=self.dtype,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            last_recurrent_state = initial_state

        # Initialize output
        core_attn_out = ttnn.zeros(
            (batch_size, num_heads, num_chunks, CHUNK_SIZE, v_head_dim),
            dtype=self.dtype,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
        )

        # Use precomputed causal mask for intra-chunk attention
        # Convert boolean mask to float (0.0 for True/masked, 1.0 for False/unmasked)
        # Upper triangular mask (diagonal=1 means strictly above diagonal)
        mask_float = self.causal_mask.float()
        mask_tt = ttnn.from_torch(mask_float, device=self.device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)

        # Process each chunk
        for chunk_idx in range(num_chunks):
            # Extract chunk: [batch, num_heads, CHUNK_SIZE, head_dim]
            q_chunk = ttnn.slice(
                q, (0, 0, chunk_idx, 0, 0), (batch_size, num_heads, chunk_idx + 1, CHUNK_SIZE, k_head_dim)
            )
            k_chunk = ttnn.slice(
                k, (0, 0, chunk_idx, 0, 0), (batch_size, num_heads, chunk_idx + 1, CHUNK_SIZE, k_head_dim)
            )
            v_chunk = ttnn.slice(
                v, (0, 0, chunk_idx, 0, 0), (batch_size, num_heads, chunk_idx + 1, CHUNK_SIZE, v_head_dim)
            )
            k_beta_chunk = ttnn.slice(
                k_beta, (0, 0, chunk_idx, 0, 0), (batch_size, num_heads, chunk_idx + 1, CHUNK_SIZE, k_head_dim)
            )
            v_beta_chunk = ttnn.slice(
                v_beta, (0, 0, chunk_idx, 0, 0), (batch_size, num_heads, chunk_idx + 1, CHUNK_SIZE, v_head_dim)
            )
            g_chunk = ttnn.slice(g, (0, 0, chunk_idx, 0), (batch_size, num_heads, chunk_idx + 1, CHUNK_SIZE))

            # Squeeze chunk dimension
            q_chunk = ttnn.reshape(q_chunk, (batch_size, num_heads, CHUNK_SIZE, k_head_dim))
            k_chunk = ttnn.reshape(k_chunk, (batch_size, num_heads, CHUNK_SIZE, k_head_dim))
            v_chunk = ttnn.reshape(v_chunk, (batch_size, num_heads, CHUNK_SIZE, v_head_dim))
            k_beta_chunk = ttnn.reshape(k_beta_chunk, (batch_size, num_heads, CHUNK_SIZE, k_head_dim))
            v_beta_chunk = ttnn.reshape(v_beta_chunk, (batch_size, num_heads, CHUNK_SIZE, v_head_dim))
            g_chunk = ttnn.reshape(g_chunk, (batch_size, num_heads, CHUNK_SIZE))

            # Compute intra-chunk attention with decay
            # attn = (q @ k^T) * decay_mask, masked with causal mask
            attn = ttnn.matmul(
                q_chunk, ttnn.permute(k_chunk, (0, 1, 3, 2))
            )  # [batch, num_heads, CHUNK_SIZE, CHUNK_SIZE]

            # Apply decay mask: exp(g[i] - g[j]) for i >= j
            # This is a simplified version - full implementation would compute proper decay
            g_expanded_i = ttnn.reshape(g_chunk, (batch_size, num_heads, CHUNK_SIZE, 1))
            g_expanded_j = ttnn.reshape(g_chunk, (batch_size, num_heads, 1, CHUNK_SIZE))
            decay_diff = ttnn.sub(g_expanded_i, g_expanded_j)
            decay_mask = ttnn.exp(decay_diff)

            # Convert to TILE_LAYOUT for ttnn.where operation
            decay_mask = ttnn.to_layout(decay_mask, ttnn.TILE_LAYOUT)
            attn = ttnn.to_layout(attn, ttnn.TILE_LAYOUT)

            # Apply lower triangular mask (zero out upper triangle)
            decay_mask = ttnn.where(mask_tt, 0.0, decay_mask)
            attn = ttnn.mul(attn, decay_mask)

            # Apply causal mask
            attn = ttnn.where(mask_tt, 0.0, attn)

            # Convert v_beta_chunk to TILE_LAYOUT for matmul
            v_beta_chunk_tiled = ttnn.to_layout(v_beta_chunk, ttnn.TILE_LAYOUT)

            # Compute intra-chunk output: attn @ v_beta
            v_intra = ttnn.matmul(attn, v_beta_chunk_tiled)

            # Compute inter-chunk contribution from recurrent state
            # v_prime = (q_chunk * exp(g_chunk)) @ last_recurrent_state
            g_exp = ttnn.exp(g_chunk)
            g_exp_expanded = ttnn.reshape(g_exp, (batch_size, num_heads, CHUNK_SIZE, 1))
            q_decayed = ttnn.mul(q_chunk, g_exp_expanded)

            # Reshape for matmul: [batch, num_heads, CHUNK_SIZE, k_head_dim] @ [batch, num_heads, k_head_dim, v_head_dim]
            v_inter = ttnn.matmul(q_decayed, last_recurrent_state)

            # Combine intra and inter chunk contributions
            v_total = ttnn.add(v_intra, v_inter)

            # Store output using in-place slice write
            # Note: Using experimental API for efficient in-place tensor updates.
            # Monitor ttnn releases for stable alternative if API changes.
            v_total_expanded = ttnn.reshape(v_total, (batch_size, num_heads, 1, CHUNK_SIZE, v_head_dim))
            core_attn_out = ttnn.experimental.slice_write(
                v_total_expanded,
                core_attn_out,
                (0, 0, chunk_idx, 0, 0),
                (batch_size, num_heads, chunk_idx + 1, CHUNK_SIZE, v_head_dim),
                (1, 1, 1, 1, 1),
            )

            # Update recurrent state for next chunk
            # state = state * exp(g[-1]) + sum_t(outer(k_t, v_beta_t) * exp(g[-1] - g[t]))
            # This is equivalent to: state * exp(g[-1]) + k_decayed^T @ v_beta
            # where k_decayed = k * exp(g[-1] - g)
            g_last = ttnn.slice(g_chunk, (0, 0, CHUNK_SIZE - 1), (batch_size, num_heads, CHUNK_SIZE))
            g_last_expanded = ttnn.reshape(ttnn.exp(g_last), (batch_size, num_heads, 1, 1))
            last_recurrent_state = ttnn.mul(last_recurrent_state, g_last_expanded)

            # Compute decay-weighted k: k_decayed = k * exp(g[-1] - g[t]) for each position t
            g_last_for_decay = ttnn.reshape(g_last, (batch_size, num_heads, 1))  # [B, H, 1]
            g_decay = ttnn.sub(g_last_for_decay, g_chunk)  # g[-1] - g[t] for each t
            g_decay_exp = ttnn.exp(g_decay)  # [B, H, CHUNK_SIZE]
            g_decay_expanded = ttnn.reshape(g_decay_exp, (batch_size, num_heads, CHUNK_SIZE, 1))
            k_decayed = ttnn.mul(k_chunk, g_decay_expanded)  # [B, H, CHUNK_SIZE, k_dim]

            # Compute k_decayed^T @ v_beta
            k_decayed_t = ttnn.permute(k_decayed, (0, 1, 3, 2))  # [batch, num_heads, k_head_dim, CHUNK_SIZE]
            kv_update = ttnn.matmul(k_decayed_t, v_beta_chunk)  # [batch, num_heads, k_head_dim, v_head_dim]
            last_recurrent_state = ttnn.add(last_recurrent_state, kv_update)

        # Reshape back: [batch, num_heads, total_seq_len, v_head_dim]
        core_attn_out = ttnn.reshape(core_attn_out, (batch_size, num_heads, total_seq_len, v_head_dim))

        # Remove padding
        if pad_size > 0:
            core_attn_out = ttnn.slice(core_attn_out, (0, 0, 0, 0), (batch_size, num_heads, seq_len, v_head_dim))

        # Transpose back to [batch, seq_len, num_heads, v_head_dim]
        core_attn_out = ttnn.permute(core_attn_out, (0, 2, 1, 3))

        if not output_final_state:
            last_recurrent_state = None

        return core_attn_out, last_recurrent_state

    def _recurrent_gated_delta_rule(self, q, k, v, g, beta, initial_state, output_final_state):
        """
        Recurrent delta rule for decode mode (single token processing).

        OPTIMIZED for seq_len=1: No loops, no slices, uses matmul instead of elementwise + sum.

        Reference (from HF torch_recurrent_gated_delta_rule):
            last_recurrent_state = last_recurrent_state * exp(g)
            kv_mem = (last_recurrent_state * k).sum(dim=-2)  # = state @ k (matmul)
            delta = (v - kv_mem) * beta
            last_recurrent_state = last_recurrent_state + outer(k, delta)
            output = (last_recurrent_state * q).sum(dim=-2)  # = state @ q (matmul)

        Args:
            q, k: [batch, num_v_heads, seq_len, head_k_dim] - Query and key tensors (already transposed)
            v: [batch, num_v_heads, seq_len, head_v_dim] - Value tensor (already transposed)
            g, beta: [batch, seq_len, num_v_heads] - Decay rate and beta gating
            initial_state: [batch, num_v_heads, head_k_dim, head_v_dim] or None - Previous recurrent state
            output_final_state: bool - Whether to return final state (False for memory efficiency when not caching)

        Returns:
            tuple:
                - output: [batch, seq_len, num_v_heads, head_v_dim] - Attention output
                - final_state: [batch, num_v_heads, head_k_dim, head_v_dim] or None - Updated state
        """
        # QKV already in [batch, num_heads, seq_len, head_dim] format from _split_qkv_to_heads
        batch_size, num_heads, seq_len, k_head_dim = q.shape
        v_head_dim = v.shape[-1]

        # Decode mode should always be seq_len=1. For seq_len>1, use chunk mode instead.
        if seq_len != 1:
            logger.warning(
                f"Recurrent delta rule called with seq_len={seq_len}. "
                "This function is optimized for seq_len=1. Use chunk mode for prefill."
            )
            # Fall back to loop-based processing for seq_len > 1 (rare edge case)
            return self._recurrent_gated_delta_rule_loop(q, k, v, g, beta, initial_state, output_final_state)

        # === OPTIMIZED PATH FOR seq_len=1 (typical decode) ===
        # No loops, no slices needed - work directly with the tensors

        # Apply L2 normalization to Q and K for numerical stability
        q = self._l2norm(q, dim=-1, eps=EPSILON)
        k = self._l2norm(k, dim=-1, eps=EPSILON)

        # Apply attention scaling: scale = 1/sqrt(head_dim)
        scale = 1.0 / (k_head_dim**ATTENTION_SCALE_EXPONENT)
        q = ttnn.mul(q, scale)

        # Permute g and beta: [batch, 1, num_heads] -> [batch, num_heads, 1]
        g = ttnn.permute(g, (0, 2, 1))
        beta = ttnn.permute(beta, (0, 2, 1))

        # Initialize state if needed
        if initial_state is None:
            last_recurrent_state = ttnn.zeros(
                (batch_size, num_heads, k_head_dim, v_head_dim),
                dtype=self.dtype,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            last_recurrent_state = initial_state

        # Squeeze seq_len dimension: [B, H, 1, D] -> [B, H, D]
        # Use reshape instead of squeeze for explicit shape control
        q = ttnn.reshape(q, (batch_size, num_heads, k_head_dim))
        k = ttnn.reshape(k, (batch_size, num_heads, k_head_dim))
        v = ttnn.reshape(v, (batch_size, num_heads, v_head_dim))
        g = ttnn.reshape(g, (batch_size, num_heads))
        beta = ttnn.reshape(beta, (batch_size, num_heads))

        # 1. Decay state: state = state * exp(g)
        g_exp = ttnn.exp(g)
        g_exp = ttnn.reshape(g_exp, (batch_size, num_heads, 1, 1))
        last_recurrent_state = ttnn.mul(last_recurrent_state, g_exp)

        # 2. Memory retrieval: kv_mem = state @ k (matrix-vector multiply)
        # state: [B, H, k_dim, v_dim], k: [B, H, k_dim] -> k: [B, H, k_dim, 1]
        # Result: [B, H, v_dim, 1] -> squeeze to [B, H, v_dim]
        k_col = ttnn.reshape(k, (batch_size, num_heads, k_head_dim, 1))
        kv_mem = ttnn.matmul(
            ttnn.permute(last_recurrent_state, (0, 1, 3, 2)),  # [B, H, v_dim, k_dim]
            k_col,  # [B, H, k_dim, 1]
        )  # [B, H, v_dim, 1]
        kv_mem = ttnn.reshape(kv_mem, (batch_size, num_heads, v_head_dim))

        # 3. Compute delta: delta = (v - kv_mem) * beta
        beta = ttnn.reshape(beta, (batch_size, num_heads, 1))
        v_diff = ttnn.sub(v, kv_mem)
        delta = ttnn.mul(v_diff, beta)  # [B, H, v_dim]

        # 4. Update state: state = state + outer(k, delta)
        # outer(k, delta) = k[:, :, :, None] * delta[:, :, None, :]
        k_col = ttnn.reshape(k, (batch_size, num_heads, k_head_dim, 1))
        delta_row = ttnn.reshape(delta, (batch_size, num_heads, 1, v_head_dim))
        k_delta = ttnn.mul(k_col, delta_row)  # [B, H, k_dim, v_dim]
        last_recurrent_state = ttnn.add(last_recurrent_state, k_delta)

        # 5. Compute output: output = state @ q (matrix-vector multiply)
        q_col = ttnn.reshape(q, (batch_size, num_heads, k_head_dim, 1))
        output = ttnn.matmul(
            ttnn.permute(last_recurrent_state, (0, 1, 3, 2)),  # [B, H, v_dim, k_dim]
            q_col,  # [B, H, k_dim, 1]
        )  # [B, H, v_dim, 1]
        output = ttnn.reshape(output, (batch_size, num_heads, v_head_dim))

        # Reshape to [batch, seq_len=1, num_heads, v_head_dim]
        core_attn_out = ttnn.reshape(output, (batch_size, 1, num_heads, v_head_dim))

        # Set state to None for memory efficiency when caching is disabled
        if not output_final_state:
            last_recurrent_state = None

        return core_attn_out, last_recurrent_state

    def _recurrent_gated_delta_rule_loop(self, q, k, v, g, beta, initial_state, output_final_state):
        """
        Loop-based recurrent delta rule for seq_len > 1 (fallback, rarely used).

        This is the unoptimized version kept for edge cases where decode is called
        with seq_len > 1. For normal decode (seq_len=1), use _recurrent_gated_delta_rule.
        """
        batch_size, num_heads, seq_len, k_head_dim = q.shape
        v_head_dim = v.shape[-1]

        # Apply L2 normalization to Q and K
        q = self._l2norm(q, dim=-1, eps=EPSILON)
        k = self._l2norm(k, dim=-1, eps=EPSILON)

        # Permute g and beta
        g = ttnn.permute(g, (0, 2, 1))
        beta = ttnn.permute(beta, (0, 2, 1))

        # Apply attention scaling
        scale = 1.0 / (k_head_dim**ATTENTION_SCALE_EXPONENT)
        q = ttnn.mul(q, scale)

        # Initialize state
        if initial_state is None:
            last_recurrent_state = ttnn.zeros(
                (batch_size, num_heads, k_head_dim, v_head_dim),
                dtype=self.dtype,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            last_recurrent_state = initial_state

        # Initialize output
        core_attn_out = ttnn.zeros(
            (batch_size, num_heads, seq_len, v_head_dim),
            dtype=self.dtype,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
        )

        # Process each timestep (fallback loop for seq_len > 1)
        for i in range(seq_len):
            q_t = ttnn.slice(q, (0, 0, i, 0), (batch_size, num_heads, i + 1, k_head_dim))
            k_t = ttnn.slice(k, (0, 0, i, 0), (batch_size, num_heads, i + 1, k_head_dim))
            v_t = ttnn.slice(v, (0, 0, i, 0), (batch_size, num_heads, i + 1, v_head_dim))
            g_t = ttnn.slice(g, (0, 0, i), (batch_size, num_heads, i + 1))
            beta_t = ttnn.slice(beta, (0, 0, i), (batch_size, num_heads, i + 1))

            q_t = ttnn.reshape(q_t, (batch_size, num_heads, k_head_dim))
            k_t = ttnn.reshape(k_t, (batch_size, num_heads, k_head_dim))
            v_t = ttnn.reshape(v_t, (batch_size, num_heads, v_head_dim))
            g_t = ttnn.reshape(g_t, (batch_size, num_heads))
            beta_t = ttnn.reshape(beta_t, (batch_size, num_heads))

            # Decay state
            g_t_exp = ttnn.exp(g_t)
            g_t_expanded = ttnn.reshape(g_t_exp, (batch_size, num_heads, 1, 1))
            last_recurrent_state = ttnn.mul(last_recurrent_state, g_t_expanded)

            # Memory retrieval
            k_t_expanded = ttnn.reshape(k_t, (batch_size, num_heads, k_head_dim, 1))
            state_k = ttnn.mul(last_recurrent_state, k_t_expanded)
            kv_mem = ttnn.sum(state_k, dim=-2)

            # Compute delta
            beta_t_expanded = ttnn.reshape(beta_t, (batch_size, num_heads, 1))
            v_diff = ttnn.sub(v_t, kv_mem)
            delta = ttnn.mul(v_diff, beta_t_expanded)

            # Update state
            delta_expanded = ttnn.reshape(delta, (batch_size, num_heads, 1, v_head_dim))
            k_delta = ttnn.mul(k_t_expanded, delta_expanded)
            last_recurrent_state = ttnn.add(last_recurrent_state, k_delta)

            # Compute output
            q_t_expanded = ttnn.reshape(q_t, (batch_size, num_heads, k_head_dim, 1))
            state_q = ttnn.mul(last_recurrent_state, q_t_expanded)
            output_t = ttnn.sum(state_q, dim=-2)

            # Store output
            output_t_expanded = ttnn.reshape(output_t, (batch_size, num_heads, 1, v_head_dim))
            core_attn_out = ttnn.experimental.slice_write(
                output_t_expanded,
                core_attn_out,
                (0, 0, i, 0),
                (batch_size, num_heads, i + 1, v_head_dim),
                (1, 1, 1, 1),
            )

        # Transpose back
        core_attn_out = ttnn.permute(core_attn_out, (0, 2, 1, 3))

        if not output_final_state:
            last_recurrent_state = None

        return core_attn_out, last_recurrent_state

    def _rms_norm_gated(self, x, gate):
        """
        Apply RMSNorm with SiLU gating.

        Reference (from HF Qwen3_5RMSNormGated):
            variance = x.pow(2).mean(-1, keepdim=True)
            x_norm = x * rsqrt(variance + eps)
            x_norm = weight * x_norm
            x_norm = x_norm * silu(gate)

        Args:
            x: [N, head_v_dim] - input to normalize
            gate: [N, head_v_dim] - gating values

        Returns:
            normalized: [N, head_v_dim]
        """
        eps = EPSILON  # Numerical stability epsilon (same as HF config)

        # 1. Compute variance: mean of x^2 along last dimension
        x_squared = ttnn.pow(x, 2.0)
        variance = ttnn.mean(x_squared, dim=-1, keepdim=True)

        # 2. Normalize: x * rsqrt(variance + eps)
        variance_eps = ttnn.add(variance, eps)
        inv_std = ttnn.rsqrt(variance_eps)
        x_norm = ttnn.mul(x, inv_std)

        # 3. Apply weight (broadcast norm_weight: [head_v_dim] to [N, head_v_dim])
        # Weight is already on device from load_weights
        x_norm = ttnn.mul(x_norm, self.norm_weight)

        # 4. Apply SiLU gating: x_norm * silu(gate)
        gate_activated = ttnn.silu(gate)
        x_gated = ttnn.mul(x_norm, gate_activated)

        return x_gated
