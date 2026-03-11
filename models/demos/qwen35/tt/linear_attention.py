# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Linear Attention (GatedDeltaNet) implementation for Qwen3.5-9B.

Based on HuggingFace transformers reference implementation.

Architecture (from reference):
    1. Project input to FUSED QKV: in_proj_qkv → [key_dim*2 + value_dim]
    2. Apply Conv1d on FUSED QKV
    3. Split into Q, K, V
    4. Compute beta (from in_proj_b) and g (from in_proj_a)
    5. Repeat Q,K heads to match V heads if needed
    6. Apply delta rule (chunk or recurrent)
    7. Apply RMSNormGated with z gate (from in_proj_z)
    8. Project output with out_proj

Key dimensions (Qwen3.5-9B):
    hidden_size: 4096
    key_dim: 2048 = num_k_heads * head_k_dim = 16 * 128
    value_dim: 4096 = num_v_heads * head_v_dim = 32 * 128
    conv_dim: 8192 = key_dim*2 + value_dim (Conv operates on FUSED QKV)
"""

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule

# Constants from HuggingFace config
EPSILON = 1e-6  # Numerical stability epsilon for RMSNorm (config.rms_norm_eps)
CHUNK_SIZE = 64  # Token chunk size for efficient parallel processing in prefill mode
ATTENTION_SCALE_DIVISOR = 0.5  # Exponent for attention scaling: 1/sqrt(head_dim) = 1/(head_dim**0.5)


class TtLinearAttention(LightweightModule):
    """
    GatedDeltaNet Linear Attention for Qwen3.5.

    Matches HuggingFace Qwen3_5GatedDeltaNet implementation exactly.
    """

    def __init__(self, args, layer_idx, device, dtype=ttnn.bfloat16):
        super().__init__()

        self.args = args
        self.layer_idx = layer_idx
        self.device = device
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
        self.state_dict_prefix = args.get_state_dict_prefix("LinearAttention", layer_idx)

        logger.info(f"Initializing GatedDeltaNet (Layer {layer_idx})")
        logger.info(f"  hidden_size: {self.hidden_size}")
        logger.info(f"  num_k_heads: {self.num_k_heads}, num_v_heads: {self.num_v_heads}")
        logger.info(f"  key_dim: {self.key_dim}, value_dim: {self.value_dim}")
        logger.info(f"  conv_dim: {self.conv_dim} (for fused QKV)")

        # Weights (will be loaded from state dict)
        self.in_proj_qkv = None  # [hidden_size, conv_dim] - FUSED Q+K+V
        self.in_proj_z = None  # [hidden_size, value_dim] - output gating
        self.in_proj_b = None  # [hidden_size, num_v_heads] - beta gating for value interpolation
        self.in_proj_a = None  # [hidden_size, num_v_heads] - alpha gating for decay rate computation
        self.conv1d_weight = None  # [conv_dim, 1, conv_kernel_size]
        self.out_proj = None  # [value_dim, hidden_size]
        self.A_log = None  # [num_v_heads] - SSM state matrix parameter
        self.dt_bias = None  # [num_v_heads] - dt (delta-time) bias for temporal discretization
        self.norm_weight = None  # [head_v_dim] - RMSNorm weight

        # Precompute causal masks for chunk attention (constant, can be computed once)
        # Upper triangular mask with diagonal=1 means strictly above diagonal is True
        self.causal_mask_chunk = torch.triu(torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool), diagonal=1)
        # Lower triangular mask with diagonal=0 means diagonal and below is True
        self.causal_mask_intra = torch.triu(torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool), diagonal=1)

    def load_weights(self, state_dict):
        """
        Load weights from HuggingFace GatedDeltaNet state dict.

        Args:
            state_dict: HuggingFace model state dictionary

        Expected keys (prefix = "model.layers.{layer_idx}.linear_attn"):
            - in_proj_qkv.weight: [conv_dim, hidden_size]
            - in_proj_z.weight: [value_dim, hidden_size]
            - in_proj_b.weight: [num_v_heads, hidden_size]
            - in_proj_a.weight: [num_v_heads, hidden_size]
            - conv1d.weight: [conv_dim, 1, conv_kernel_size]
            - out_proj.weight: [hidden_size, value_dim]
            - A_log: [num_v_heads]
            - dt_bias: [num_v_heads]
            - norm.weight: [head_v_dim]

        Returns:
            None
        """
        prefix = self.state_dict_prefix

        # Load projection weights (transpose for ttnn)
        self.in_proj_qkv = ttnn.from_torch(
            state_dict[f"{prefix}.in_proj_qkv.weight"].transpose(0, 1),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.in_proj_z = ttnn.from_torch(
            state_dict[f"{prefix}.in_proj_z.weight"].transpose(0, 1),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.in_proj_b = ttnn.from_torch(
            state_dict[f"{prefix}.in_proj_b.weight"].transpose(0, 1),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.in_proj_a = ttnn.from_torch(
            state_dict[f"{prefix}.in_proj_a.weight"].transpose(0, 1),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # Conv1d weight (keep in ROW_MAJOR for conv operation)
        self.conv1d_weight = ttnn.from_torch(
            state_dict[f"{prefix}.conv1d.weight"], dtype=self.dtype, device=self.device
        )

        # Output projection
        self.out_proj = ttnn.from_torch(
            state_dict[f"{prefix}.out_proj.weight"].transpose(0, 1),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # SSM parameters
        self.A_log = ttnn.from_torch(state_dict[f"{prefix}.A_log"], dtype=self.dtype, device=self.device)
        self.dt_bias = ttnn.from_torch(state_dict[f"{prefix}.dt_bias"], dtype=self.dtype, device=self.device)

        # RMSNorm weight
        self.norm_weight = ttnn.from_torch(state_dict[f"{prefix}.norm.weight"], dtype=self.dtype, device=self.device)

        logger.info(f"Loaded GatedDeltaNet weights for layer {self.layer_idx}")

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

        # 4. Split QKV
        # mixed_qkv: [batch, seq_len, conv_dim] → split into [key_dim, key_dim, value_dim]
        query, key, value = self._split_qkv(mixed_qkv)

        # 5. Reshape to multi-head
        query = ttnn.reshape(query, (batch_size, seq_len, self.num_k_heads, self.head_k_dim))
        key = ttnn.reshape(key, (batch_size, seq_len, self.num_k_heads, self.head_k_dim))
        value = ttnn.reshape(value, (batch_size, seq_len, self.num_v_heads, self.head_v_dim))

        # 6. Compute beta and g
        beta = ttnn.sigmoid(b)  # [batch, seq_len, num_v_heads]
        g = self._compute_g(a)  # [batch, seq_len, num_v_heads]

        # 7. Repeat Q and K heads to match V heads if needed
        if self.num_v_heads // self.num_k_heads > 1:
            repeat_factor = self.num_v_heads // self.num_k_heads
            query = self._repeat_interleave(query, repeat_factor, dim=2)
            key = self._repeat_interleave(key, repeat_factor, dim=2)

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
        Apply causal conv1d on fused QKV tensor.

        Reference (from HF):
        - Prefill: F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        - Decode: torch_causal_conv1d_update with state management

        Args:
            x: [batch, seq_len, conv_dim] - Input tensor
            conv_state: [batch, conv_dim, kernel_size-1] or None - Previous conv state for decode mode
            use_cached: bool - True for decode mode (seq_len=1), False for prefill mode

        Returns:
            tuple:
                - x_conv: [batch, seq_len, conv_dim] - Convolved and activated output
                - new_conv_state: [batch, conv_dim, kernel_size-1] - Updated state for next iteration

        Shape notes:
            - conv_state stores last (kernel_size-1) positions to maintain causal context
            - In decode mode, concatenates state with new input before convolution
            - In prefill mode, left-pads input to ensure causality
        """
        batch_size, seq_len, _ = x.shape

        # Transpose to [batch, conv_dim, seq_len] for conv1d
        x_transposed = ttnn.permute(x, (0, 2, 1))

        if use_cached:
            # Decode mode: concatenate with conv_state for causal context
            # conv_state: [batch, conv_dim, kernel_size-1]
            # x_transposed: [batch, conv_dim, 1] (typically seq_len=1 in decode)
            if conv_state is not None:
                x_with_state = ttnn.concat([conv_state, x_transposed], dim=2)
            else:
                x_with_state = x_transposed

            # Apply depthwise conv1d (groups=conv_dim means each channel convolved independently)
            x_conv = ttnn.conv1d(
                x_with_state,
                self.conv1d_weight,
                bias_tensor=None,  # No bias in GatedDeltaNet conv1d
                groups=self.conv_dim,
                padding=0,
            )

            # Take only the last seq_len positions (output for current tokens)
            x_conv = ttnn.slice(x_conv, (0, 0, x_conv.shape[2] - seq_len), (batch_size, self.conv_dim, x_conv.shape[2]))

            # Keep last (kernel_size-1) positions as next iteration's state for causal convolution
            state_len = self.conv_kernel_size - 1
            new_conv_state = ttnn.slice(
                x_with_state,
                (0, 0, x_with_state.shape[2] - state_len),
                (batch_size, self.conv_dim, x_with_state.shape[2]),
            )

        else:
            # Prefill mode: apply conv1d with left padding for causality
            # Padding: kernel_size - 1 on the left ensures output only depends on past/current tokens
            padding = self.conv_kernel_size - 1
            x_padded = ttnn.pad(x_transposed, [(0, 0), (0, 0), (padding, 0)], 0.0)

            # Apply depthwise conv1d
            x_conv = ttnn.conv1d(
                x_padded,
                self.conv1d_weight,
                bias_tensor=None,  # No bias in GatedDeltaNet conv1d
                groups=self.conv_dim,
                padding=0,
            )

            # Take only first seq_len positions (causal)
            x_conv = ttnn.slice(x_conv, (0, 0, 0), (batch_size, self.conv_dim, seq_len))

            # Create new conv_state: last (kernel_size-1) positions of input
            state_len = self.conv_kernel_size - 1
            if seq_len >= state_len:
                new_conv_state = ttnn.slice(
                    x_transposed, (0, 0, seq_len - state_len), (batch_size, self.conv_dim, seq_len)
                )
            else:
                # Pad if sequence is shorter than state length
                pad_len = state_len - seq_len
                new_conv_state = ttnn.pad(x_transposed, [(0, 0), (0, 0), (pad_len, 0)], 0.0)

        # Apply SiLU activation
        x_conv = ttnn.silu(x_conv)

        # Transpose back to [batch, seq_len, conv_dim]
        x_conv = ttnn.permute(x_conv, (0, 2, 1))

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

    def _split_qkv(self, mixed_qkv):
        """
        Split fused QKV tensor.

        Args:
            mixed_qkv: [batch, seq_len, conv_dim=8192]

        Returns:
            query: [batch, seq_len, key_dim=2048]
            key: [batch, seq_len, key_dim=2048]
            value: [batch, seq_len, value_dim=4096]
        """
        # Split along last dimension: conv_dim = key_dim + key_dim + value_dim
        # = 2048 + 2048 + 4096 = 8192
        query = ttnn.slice(mixed_qkv, (0, 0, 0), (mixed_qkv.shape[0], mixed_qkv.shape[1], self.key_dim))
        key = ttnn.slice(mixed_qkv, (0, 0, self.key_dim), (mixed_qkv.shape[0], mixed_qkv.shape[1], self.key_dim * 2))
        value = ttnn.slice(mixed_qkv, (0, 0, self.key_dim * 2), (mixed_qkv.shape[0], mixed_qkv.shape[1], self.conv_dim))
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
            q, k: [batch, seq_len, num_v_heads, head_k_dim] - Query and key tensors
            v: [batch, seq_len, num_v_heads, head_v_dim] - Value tensor
            g, beta: [batch, seq_len, num_v_heads] - Decay rate and beta gating
            initial_state: [batch, num_v_heads, head_k_dim, head_v_dim] or None - Previous recurrent state
            output_final_state: bool - Whether to return final state

        Returns:
            tuple:
                - output: [batch, seq_len, num_v_heads, head_v_dim] - Attention output
                - final_state: [batch, num_v_heads, head_k_dim, head_v_dim] or None - Updated state
        """
        batch_size, seq_len, num_heads, k_head_dim = k.shape
        v_head_dim = v.shape[-1]

        # Apply L2 normalization to Q and K for numerical stability
        q = self._l2norm(q, dim=-1, eps=EPSILON)
        k = self._l2norm(k, dim=-1, eps=EPSILON)

        # Transpose to [batch, num_heads, seq_len, head_dim] for batch matrix operations
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))
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
        scale = 1.0 / (k_head_dim**ATTENTION_SCALE_DIVISOR)
        q = ttnn.mul(q, scale)

        # Compute v_beta and k_beta
        beta_expanded = ttnn.reshape(beta, (batch_size, num_heads, total_seq_len, 1))
        v_beta = ttnn.mul(v, beta_expanded)
        k_beta = ttnn.mul(k, beta_expanded)

        # Reshape to chunks: [batch, num_heads, num_chunks, chunk_size, head_dim]
        q = ttnn.reshape(q, (batch_size, num_heads, num_chunks, chunk_size, k_head_dim))
        k = ttnn.reshape(k, (batch_size, num_heads, num_chunks, chunk_size, k_head_dim))
        v = ttnn.reshape(v, (batch_size, num_heads, num_chunks, chunk_size, v_head_dim))
        k_beta = ttnn.reshape(k_beta, (batch_size, num_heads, num_chunks, chunk_size, k_head_dim))
        v_beta = ttnn.reshape(v_beta, (batch_size, num_heads, num_chunks, chunk_size, v_head_dim))
        g = ttnn.reshape(g, (batch_size, num_heads, num_chunks, chunk_size))

        # Compute cumulative g for decay
        g = ttnn.cumsum(g, dim=-1)

        # Initialize state
        if initial_state is None:
            last_recurrent_state = ttnn.zeros(
                (batch_size, num_heads, k_head_dim, v_head_dim),
                dtype=self.dtype,
                device=self.device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        else:
            last_recurrent_state = initial_state

        # Initialize output
        core_attn_out = ttnn.zeros(
            (batch_size, num_heads, num_chunks, chunk_size, v_head_dim),
            dtype=self.dtype,
            device=self.device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Use precomputed causal mask for intra-chunk attention
        # Upper triangular mask (diagonal=1 means strictly above diagonal)
        mask_tt = ttnn.from_torch(self.causal_mask_intra, device=self.device, dtype=ttnn.bool)

        # Process each chunk
        for chunk_idx in range(num_chunks):
            # Extract chunk: [batch, num_heads, chunk_size, head_dim]
            q_chunk = ttnn.slice(
                q, (0, 0, chunk_idx, 0, 0), (batch_size, num_heads, chunk_idx + 1, chunk_size, k_head_dim)
            )
            k_chunk = ttnn.slice(
                k, (0, 0, chunk_idx, 0, 0), (batch_size, num_heads, chunk_idx + 1, chunk_size, k_head_dim)
            )
            v_chunk = ttnn.slice(
                v, (0, 0, chunk_idx, 0, 0), (batch_size, num_heads, chunk_idx + 1, chunk_size, v_head_dim)
            )
            k_beta_chunk = ttnn.slice(
                k_beta, (0, 0, chunk_idx, 0, 0), (batch_size, num_heads, chunk_idx + 1, chunk_size, k_head_dim)
            )
            v_beta_chunk = ttnn.slice(
                v_beta, (0, 0, chunk_idx, 0, 0), (batch_size, num_heads, chunk_idx + 1, chunk_size, v_head_dim)
            )
            g_chunk = ttnn.slice(g, (0, 0, chunk_idx, 0), (batch_size, num_heads, chunk_idx + 1, chunk_size))

            # Squeeze chunk dimension
            q_chunk = ttnn.reshape(q_chunk, (batch_size, num_heads, chunk_size, k_head_dim))
            k_chunk = ttnn.reshape(k_chunk, (batch_size, num_heads, chunk_size, k_head_dim))
            v_chunk = ttnn.reshape(v_chunk, (batch_size, num_heads, chunk_size, v_head_dim))
            k_beta_chunk = ttnn.reshape(k_beta_chunk, (batch_size, num_heads, chunk_size, k_head_dim))
            v_beta_chunk = ttnn.reshape(v_beta_chunk, (batch_size, num_heads, chunk_size, v_head_dim))
            g_chunk = ttnn.reshape(g_chunk, (batch_size, num_heads, chunk_size))

            # Compute intra-chunk attention with decay
            # attn = (q @ k^T) * decay_mask, masked with causal mask
            attn = ttnn.matmul(
                q_chunk, ttnn.permute(k_chunk, (0, 1, 3, 2))
            )  # [batch, num_heads, chunk_size, chunk_size]

            # Apply decay mask: exp(g[i] - g[j]) for i >= j
            # This is a simplified version - full implementation would compute proper decay
            g_expanded_i = ttnn.reshape(g_chunk, (batch_size, num_heads, chunk_size, 1))
            g_expanded_j = ttnn.reshape(g_chunk, (batch_size, num_heads, 1, chunk_size))
            decay_diff = ttnn.sub(g_expanded_i, g_expanded_j)
            decay_mask = ttnn.exp(decay_diff)

            # Apply lower triangular mask (zero out upper triangle)
            decay_mask = ttnn.where(mask_tt, 0.0, decay_mask)
            attn = ttnn.mul(attn, decay_mask)

            # Apply causal mask
            attn = ttnn.where(mask_tt, 0.0, attn)

            # Compute intra-chunk output: attn @ v_beta
            v_intra = ttnn.matmul(attn, v_beta_chunk)

            # Compute inter-chunk contribution from recurrent state
            # v_prime = (q_chunk * exp(g_chunk)) @ last_recurrent_state
            g_exp = ttnn.exp(g_chunk)
            g_exp_expanded = ttnn.reshape(g_exp, (batch_size, num_heads, chunk_size, 1))
            q_decayed = ttnn.mul(q_chunk, g_exp_expanded)

            # Reshape for matmul: [batch, num_heads, chunk_size, k_head_dim] @ [batch, num_heads, k_head_dim, v_head_dim]
            v_inter = ttnn.matmul(q_decayed, last_recurrent_state)

            # Combine intra and inter chunk contributions
            v_total = ttnn.add(v_intra, v_inter)

            # Store output
            v_total_expanded = ttnn.reshape(v_total, (batch_size, num_heads, 1, chunk_size, v_head_dim))
            core_attn_out = ttnn.slice_write(core_attn_out, v_total_expanded, (0, 0, chunk_idx, 0, 0))

            # Update recurrent state for next chunk
            # last_recurrent_state = last_recurrent_state * exp(g[-1]) + k^T @ delta_v
            g_last = ttnn.slice(g_chunk, (0, 0, chunk_size - 1), (batch_size, num_heads, chunk_size))
            g_last_expanded = ttnn.reshape(ttnn.exp(g_last), (batch_size, num_heads, 1, 1))
            last_recurrent_state = ttnn.mul(last_recurrent_state, g_last_expanded)

            # Compute k^T @ v contribution
            k_chunk_t = ttnn.permute(k_chunk, (0, 1, 3, 2))  # [batch, num_heads, k_head_dim, chunk_size]
            kv_update = ttnn.matmul(k_chunk_t, v_beta_chunk)  # [batch, num_heads, k_head_dim, v_head_dim]
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

        Processes tokens sequentially, updating the recurrent state at each step.
        This mode is optimized for autoregressive generation where seq_len=1.

        Reference (from HF torch_recurrent_gated_delta_rule):
            for i in range(sequence_length):
                last_recurrent_state = last_recurrent_state * exp(g[i])
                kv_mem = (last_recurrent_state * k[i]).sum(dim=-2)
                delta = (v[i] - kv_mem) * beta[i]
                last_recurrent_state = last_recurrent_state + k[i] * delta
                output[i] = (last_recurrent_state * q[i]).sum(dim=-2)

        Args:
            q, k: [batch, seq_len, num_v_heads, head_k_dim] - Query and key tensors
            v: [batch, seq_len, num_v_heads, head_v_dim] - Value tensor
            g, beta: [batch, seq_len, num_v_heads] - Decay rate and beta gating
            initial_state: [batch, num_v_heads, head_k_dim, head_v_dim] or None - Previous recurrent state
            output_final_state: bool - Whether to return final state (False for memory efficiency when not caching)

        Returns:
            tuple:
                - output: [batch, seq_len, num_v_heads, head_v_dim] - Attention output
                - final_state: [batch, num_v_heads, head_k_dim, head_v_dim] or None - Updated state
        """
        batch_size, seq_len, num_heads, k_head_dim = k.shape
        v_head_dim = v.shape[-1]

        # Recurrent mode is designed for decode (seq_len=1), warn if used otherwise
        # Processing multiple tokens sequentially is inefficient; use chunk mode instead
        if seq_len > 1:
            logger.warning(
                f"Recurrent delta rule called with seq_len={seq_len}. "
                "This is inefficient for seq_len>1. Consider using chunk mode for prefill."
            )

        # Apply L2 normalization to Q and K for numerical stability
        q = self._l2norm(q, dim=-1, eps=EPSILON)
        k = self._l2norm(k, dim=-1, eps=EPSILON)

        # Transpose to [batch, num_heads, seq_len, head_dim] for easier processing
        q = ttnn.permute(q, (0, 2, 1, 3))  # [batch, num_heads, seq_len, k_head_dim]
        k = ttnn.permute(k, (0, 2, 1, 3))  # [batch, num_heads, seq_len, k_head_dim]
        v = ttnn.permute(v, (0, 2, 1, 3))  # [batch, num_heads, seq_len, v_head_dim]
        g = ttnn.permute(g, (0, 2, 1))  # [batch, num_heads, seq_len]
        beta = ttnn.permute(beta, (0, 2, 1))  # [batch, num_heads, seq_len]

        # Apply scaled dot-product attention normalization: scale = 1/sqrt(head_dim)
        scale = 1.0 / (k_head_dim**ATTENTION_SCALE_DIVISOR)
        q = ttnn.mul(q, scale)

        # Initialize output and state
        core_attn_out = ttnn.zeros(
            (batch_size, num_heads, seq_len, v_head_dim),
            dtype=self.dtype,
            device=self.device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        if initial_state is None:
            last_recurrent_state = ttnn.zeros(
                (batch_size, num_heads, k_head_dim, v_head_dim),
                dtype=self.dtype,
                device=self.device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        else:
            last_recurrent_state = initial_state

        # Process each timestep
        for i in range(seq_len):
            # Extract timestep tensors: [batch, num_heads, 1, head_dim]
            q_t = ttnn.slice(q, (0, 0, i, 0), (batch_size, num_heads, i + 1, k_head_dim))
            k_t = ttnn.slice(k, (0, 0, i, 0), (batch_size, num_heads, i + 1, k_head_dim))
            v_t = ttnn.slice(v, (0, 0, i, 0), (batch_size, num_heads, i + 1, v_head_dim))
            g_t = ttnn.slice(g, (0, 0, i), (batch_size, num_heads, i + 1))
            beta_t = ttnn.slice(beta, (0, 0, i), (batch_size, num_heads, i + 1))

            # Squeeze to [batch, num_heads, head_dim] and [batch, num_heads]
            q_t = ttnn.reshape(q_t, (batch_size, num_heads, k_head_dim))
            k_t = ttnn.reshape(k_t, (batch_size, num_heads, k_head_dim))
            v_t = ttnn.reshape(v_t, (batch_size, num_heads, v_head_dim))
            g_t = ttnn.reshape(g_t, (batch_size, num_heads))
            beta_t = ttnn.reshape(beta_t, (batch_size, num_heads))

            # 1. Decay state: last_recurrent_state = last_recurrent_state * exp(g_t)
            # g_t: [batch, num_heads] -> [batch, num_heads, 1, 1]
            g_t_exp = ttnn.exp(g_t)
            g_t_expanded = ttnn.reshape(g_t_exp, (batch_size, num_heads, 1, 1))
            last_recurrent_state = ttnn.mul(last_recurrent_state, g_t_expanded)

            # 2. Compute memory retrieval: kv_mem = (last_recurrent_state * k_t).sum(dim=-2)
            # k_t: [batch, num_heads, k_head_dim] -> [batch, num_heads, k_head_dim, 1]
            k_t_expanded = ttnn.reshape(k_t, (batch_size, num_heads, k_head_dim, 1))
            state_k = ttnn.mul(last_recurrent_state, k_t_expanded)
            kv_mem = ttnn.sum(state_k, dim=-2)  # [batch, num_heads, v_head_dim]

            # 3. Compute delta: delta = (v_t - kv_mem) * beta_t
            # beta_t: [batch, num_heads] -> [batch, num_heads, 1]
            beta_t_expanded = ttnn.reshape(beta_t, (batch_size, num_heads, 1))
            v_diff = ttnn.sub(v_t, kv_mem)
            delta = ttnn.mul(v_diff, beta_t_expanded)  # [batch, num_heads, v_head_dim]

            # 4. Update state: last_recurrent_state = last_recurrent_state + k_t * delta
            # k_t: [batch, num_heads, k_head_dim] -> [batch, num_heads, k_head_dim, 1]
            # delta: [batch, num_heads, v_head_dim] -> [batch, num_heads, 1, v_head_dim]
            delta_expanded = ttnn.reshape(delta, (batch_size, num_heads, 1, v_head_dim))
            k_delta = ttnn.mul(k_t_expanded, delta_expanded)  # [batch, num_heads, k_head_dim, v_head_dim]
            last_recurrent_state = ttnn.add(last_recurrent_state, k_delta)

            # 5. Compute output: output[i] = (last_recurrent_state * q_t).sum(dim=-2)
            # q_t: [batch, num_heads, k_head_dim] -> [batch, num_heads, k_head_dim, 1]
            q_t_expanded = ttnn.reshape(q_t, (batch_size, num_heads, k_head_dim, 1))
            state_q = ttnn.mul(last_recurrent_state, q_t_expanded)
            output_t = ttnn.sum(state_q, dim=-2)  # [batch, num_heads, v_head_dim]

            # Store output
            output_t_expanded = ttnn.reshape(output_t, (batch_size, num_heads, 1, v_head_dim))
            # Write back to output tensor at position i
            core_attn_out = ttnn.experimental.slice_write(core_attn_out, output_t_expanded, (0, 0, i, 0))

        # Transpose back to [batch, seq_len, num_heads, v_head_dim]
        core_attn_out = ttnn.permute(core_attn_out, (0, 2, 1, 3))

        # Set state to None for memory efficiency when caching is disabled
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
