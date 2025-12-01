# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class TtMultiheadAttention:
    def __init__(
        self,
        params,
        device,
        embed_dims=256,
        num_heads=8,
        init_cfg=None,
        batch_first=False,
    ):
        super().__init__()
        self.params = params
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.device = device
        self.batch_first = batch_first
        self.head_dim = embed_dims // num_heads

        self.attn_in_proj__weight = params.in_proj.weight
        self.attn_in_proj__bias = params.in_proj.bias

        # Pre-split Q, K, V weights in __init__ to avoid runtime slicing
        # Note: preprocess_linear_weight already transposes, so shape is [embed_dims, 3*embed_dims]
        # We slice along the second dimension (columns) to get Q, K, V
        self.q_weight = self.attn_in_proj__weight[:, :embed_dims]
        self.k_weight = self.attn_in_proj__weight[:, embed_dims : 2 * embed_dims]
        self.v_weight = self.attn_in_proj__weight[:, 2 * embed_dims :]

        self.attn_in_proj__bias_squeeze = ttnn.squeeze(self.attn_in_proj__bias, 0)
        self.attn_out_proj_weight = params.out_proj.weight
        self.attn_out_proj_bias = params.out_proj.bias

        # SDPA configuration for Wormhole
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 4),  # 8x4 cores for Wormhole n300
            q_chunk_size=32,  # Chunk size for query processing
            k_chunk_size=32,  # Chunk size for key processing
            exp_approx_mode=False,  # Use exact exp for softmax
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,  # Balance speed and accuracy
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        batch_first=False,
        **kwargs,
    ):
        if use_signpost:
            signpost(header="TtMultiheadAttention_call_start")
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos

        if query_pos is not None:
            if query.get_layout() != ttnn.TILE_LAYOUT:
                query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
            if query_pos.get_layout() != ttnn.TILE_LAYOUT:
                query_pos = ttnn.to_layout(query_pos, ttnn.TILE_LAYOUT)
            query = query + query_pos
        if key_pos is not None:
            if key.get_layout() != ttnn.TILE_LAYOUT:
                key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
            if key_pos.get_layout() != ttnn.TILE_LAYOUT:
                key_pos = ttnn.to_layout(key_pos, ttnn.TILE_LAYOUT)
            key = key + key_pos

        if batch_first:
            query = ttnn.permute(query, (1, 0))
            key = ttnn.permute(key, (1, 0))
            value = ttnn.permute(value, (1, 0))

        in_proj_bias = self.attn_in_proj__bias_squeeze

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        # Split biases for Q, K, V projections
        q_bias = in_proj_bias[: self.embed_dims]
        k_bias = in_proj_bias[self.embed_dims : 2 * self.embed_dims]
        v_bias = in_proj_bias[2 * self.embed_dims :]

        # Project Q, K, V using pre-split and pre-permuted weights (no runtime permute!)
        query = ttnn.linear(query, self.q_weight, bias=q_bias)
        key = ttnn.linear(key, self.k_weight, bias=k_bias)

        if value.get_layout() != ttnn.TILE_LAYOUT:
            value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, self.v_weight, bias=v_bias)

        # Reshape and transpose for SDPA format
        # SDPA expects: [batch, num_heads, seq_len, head_dim]
        query = ttnn.reshape(query, (tgt_len, bsz, self.num_heads, self.head_dim))
        query = ttnn.permute(query, (1, 2, 0, 3))  # [bsz, num_heads, tgt_len, head_dim]

        key = ttnn.reshape(key, (src_len, bsz, self.num_heads, self.head_dim))
        key = ttnn.permute(key, (1, 2, 0, 3))  # [bsz, num_heads, src_len, head_dim]

        value = ttnn.reshape(value, (src_len, bsz, self.num_heads, self.head_dim))
        value = ttnn.permute(value, (1, 2, 0, 3))  # [bsz, num_heads, src_len, head_dim]

        # Use optimized SDPA - fuses Q@K^T + softmax + softmax@V
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=False,
            attn_mask=attn_mask,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Reshape back: [bsz, num_heads, tgt_len, head_dim] -> [tgt_len, bsz, embed_dim]
        attn_output = ttnn.permute(attn_output, (2, 0, 1, 3))  # [tgt_len, bsz, num_heads, head_dim]
        attn_output = ttnn.reshape(attn_output, (tgt_len * bsz, embed_dim))

        # Output projection
        attn_output = ttnn.linear(attn_output, self.attn_out_proj_weight, bias=self.attn_out_proj_bias)
        attn_output = ttnn.reshape(attn_output, (tgt_len, bsz, attn_output.shape[1]))

        identity = ttnn.to_layout(identity, ttnn.TILE_LAYOUT)
        if use_signpost:
            signpost(header="TtMultiheadAttention_call_end")
        return attn_output + identity
