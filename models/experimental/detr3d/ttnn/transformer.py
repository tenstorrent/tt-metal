import ttnn
import math


class TTNNMultiheadAttention:
    def __init__(self, d_model, nhead, device):
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.device = device

        # These will be set from PyTorch weights
        self.qkv_weight = None
        self.qkv_bias = None
        self.out_weight = None
        self.out_bias = None

    def __call__(self, query, key, value):
        batch_size, seq_len, hidden_size = query.shape

        # Linear projection for Q, K, V using fused weights
        qkv = ttnn.linear(query, self.qkv_weight, bias=self.qkv_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Split and reshape for multi-head attention
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_heads=self.nhead,
            transpose_key=False,
        )

        # Use SDPA instead of manual attention computation
        context = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,  # Set to False if you don't want causal masking
            scale=1.0 / math.sqrt(self.head_dim),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Concatenate heads
        context = ttnn.transformer.concatenate_heads(context, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Output projection
        output = ttnn.linear(context, self.out_weight, bias=self.out_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return output
