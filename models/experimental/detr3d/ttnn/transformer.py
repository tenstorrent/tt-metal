import ttnn
import math
from models.common.lightweightmodule import LightweightModule


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


class TTTransformerDecoderLayer(LightweightModule):
    def __init__(
        self,
        device,
        d_model,
        nhead=4,
        dim_feedforward=256,
        dropout=0.1,
        normalize_before=True,
        model_config=None,
        parameters=None,
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.normalize_before = normalize_before
        self.model_config = model_config or {}

        # Load preprocessed parameters
        if parameters is not None:
            self.load_parameters(parameters)
        else:
            # Initialize weights as None (for backward compatibility)
            self.self_attn_weights = None
            self.cross_attn_weights = None
            self.ff_weights1 = None
            self.ff_weights2 = None
            self.norm1_weights = None
            self.norm2_weights = None
            self.norm3_weights = None

    def load_parameters(self, parameters):
        """Load preprocessed parameters from the preprocessor"""
        # Self-attention weights
        if "self_attn" in parameters:
            self.self_attn_q_weight = parameters["self_attn"].get("q_weight")
            self.self_attn_k_weight = parameters["self_attn"].get("k_weight")
            self.self_attn_v_weight = parameters["self_attn"].get("v_weight")
            self.self_attn_out_weight = parameters["self_attn"].get("out_weight")
            self.self_attn_out_bias = parameters["self_attn"].get("out_bias")

        # Cross-attention weights
        if "multihead_attn" in parameters:
            self.cross_attn_q_weight = parameters["multihead_attn"].get("q_weight")
            self.cross_attn_k_weight = parameters["multihead_attn"].get("k_weight")
            self.cross_attn_v_weight = parameters["multihead_attn"].get("v_weight")
            self.cross_attn_out_weight = parameters["multihead_attn"].get("out_weight")

        # Feedforward weights
        if "linear1" in parameters:
            self.ff_weights1 = parameters["linear1"]["weight"]
            self.ff1_bias = parameters["linear1"].get("bias")
        if "linear2" in parameters:
            self.ff_weights2 = parameters["linear2"]["weight"]
            self.ff2_bias = parameters["linear2"].get("bias")

        # Normalization weights
        for i, norm_name in enumerate(["norm1", "norm2", "norm3"], 1):
            if norm_name in parameters:
                setattr(self, f"norm{i}_weights", parameters[norm_name]["weight"])
                setattr(self, f"norm{i}_bias", parameters[norm_name].get("bias"))

    # def self_attention(self, x, query_pos=None, attn_mask=None):
    #     """Perform self-attention using preprocessed weights"""
    #     q_input = k_input = self.with_pos_embed(x, query_pos)

    #     # Apply QKV projections
    #     q = ttnn.linear(q_input, self.self_attn_q_weight)
    #     k = ttnn.linear(k_input, self.self_attn_k_weight)
    #     v = ttnn.linear(x, self.self_attn_v_weight)

    #     # Scaled dot-product attention
    #     attn_output = ttnn.transformer.scaled_dot_product_attention(
    #         q, k, v, attn_mask=attn_mask, is_causal=attn_mask is None
    #     )

    #     # Output projection
    #     if self.self_attn_out_weight is not None:
    #         attn_output = ttnn.linear(attn_output, self.self_attn_out_weight,
    #                                 bias=self.self_attn_out_bias)

    #     return attn_output

    def self_attention(self, x, query_pos=None, attn_mask=None):
        """Perform self-attention using preprocessed weights"""
        q_input = k_input = self.with_pos_embed(x, query_pos)

        # Apply QKV projections
        q = ttnn.linear(q_input, self.self_attn_q_weight)
        k = ttnn.linear(k_input, self.self_attn_k_weight)
        v = ttnn.linear(x, self.self_attn_v_weight)

        # Reshape for multi-head attention: [batch, seq_len, d_model] -> [batch, num_heads, seq_len, head_dim]
        batch_size, seq_len, _ = q.shape
        head_dim = self.d_model // self.nhead

        q = ttnn.reshape(q, [batch_size, seq_len, self.nhead, head_dim])
        q = ttnn.transpose(q, 1, 2)  # [batch, num_heads, seq_len, head_dim]

        k = ttnn.reshape(k, [batch_size, seq_len, self.nhead, head_dim])
        k = ttnn.transpose(k, 1, 2)

        v = ttnn.reshape(v, [batch_size, seq_len, self.nhead, head_dim])
        v = ttnn.transpose(v, 1, 2)

        # Scaled dot-product attention
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=attn_mask is None
        )

        # Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, d_model]
        attn_output = ttnn.transpose(attn_output, 1, 2)
        attn_output = ttnn.reshape(attn_output, [batch_size, seq_len, self.d_model])

        # Output projection
        if self.self_attn_out_weight is not None:
            attn_output = ttnn.linear(attn_output, self.self_attn_out_weight, bias=self.self_attn_out_bias)

        return attn_output

    def cross_attention(self, tgt, memory, query_pos=None, pos=None, attn_mask=None):
        """Perform cross-attention using preprocessed weights"""
        q_input = self.with_pos_embed(tgt, query_pos)
        k_input = v_input = self.with_pos_embed(memory, pos)

        # Apply QKV projections
        q = ttnn.linear(q_input, self.cross_attn_q_weight)
        k = ttnn.linear(k_input, self.cross_attn_k_weight)
        v = ttnn.linear(v_input, self.cross_attn_v_weight)

        # Reshape for multi-head attention: [batch, seq_len, d_model] -> [batch, num_heads, seq_len, head_dim]
        batch_size, seq_len, _ = q.shape
        head_dim = self.d_model // self.nhead

        q = ttnn.reshape(q, [batch_size, seq_len, self.nhead, head_dim])
        q = ttnn.transpose(q, 1, 2)  # [batch, num_heads, seq_len, head_dim]

        k = ttnn.reshape(k, [batch_size, seq_len, self.nhead, head_dim])
        k = ttnn.transpose(k, 1, 2)

        v = ttnn.reshape(v, [batch_size, seq_len, self.nhead, head_dim])
        v = ttnn.transpose(v, 1, 2)

        # Scaled dot-product attention
        attn_output = ttnn.transformer.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)

        # Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, d_model]
        attn_output = ttnn.transpose(attn_output, 1, 2)
        attn_output = ttnn.reshape(attn_output, [batch_size, seq_len, self.d_model])
        # Output projection
        if self.cross_attn_out_weight is not None:
            attn_output = ttnn.linear(attn_output, self.cross_attn_out_weight)

        return attn_output

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else ttnn.add(tensor, pos)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
        return_attn_weights=False,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
                return_attn_weights,
            )
        else:
            return self.forward_post(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
                return_attn_weights,
            )

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
        return_attn_weights=False,
    ):
        # Self-attention using proper QKV projections
        tgt2 = self.self_attention(tgt, query_pos, tgt_mask)
        tgt = ttnn.add(tgt, tgt2)
        tgt = ttnn.layer_norm(tgt, weight=self.norm1_weights, bias=getattr(self, "norm1_bias", None))

        # Cross-attention using proper QKV projections
        tgt2 = self.cross_attention(tgt, memory, query_pos, pos, memory_mask)
        tgt = ttnn.add(tgt, tgt2)
        tgt = ttnn.layer_norm(tgt, weight=self.norm2_weights, bias=getattr(self, "norm2_bias", None))

        # Feedforward
        tgt2 = ttnn.linear(tgt, self.ff_weights1, bias=getattr(self, "ff1_bias", None))
        tgt2 = ttnn.relu(tgt2)
        tgt2 = ttnn.linear(tgt2, self.ff_weights2, bias=getattr(self, "ff2_bias", None))
        tgt = ttnn.add(tgt, tgt2)
        tgt = ttnn.layer_norm(tgt, weight=self.norm3_weights, bias=getattr(self, "norm3_bias", None))

        if return_attn_weights:
            return tgt, None  # TTNN doesn't return attention weights
        return tgt, None

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
        return_attn_weights=False,
    ):
        # Pre-norm self-attention
        tgt2 = ttnn.layer_norm(tgt, weight=self.norm1_weights, bias=getattr(self, "norm1_bias", None))
        tgt2 = self.self_attention(tgt2, query_pos, tgt_mask)
        tgt = ttnn.add(tgt, tgt2)

        # Pre-norm cross-attention
        tgt2 = ttnn.layer_norm(tgt, weight=self.norm2_weights, bias=getattr(self, "norm2_bias", None))
        tgt2 = self.cross_attention(tgt2, memory, query_pos, pos, memory_mask)
        tgt = ttnn.add(tgt, tgt2)

        # Pre-norm feedforward
        tgt2 = ttnn.layer_norm(tgt, weight=self.norm3_weights, bias=getattr(self, "norm3_bias", None))
        tgt2 = ttnn.linear(tgt2, self.ff_weights1, bias=getattr(self, "ff1_bias", None))
        tgt2 = ttnn.relu(tgt2)
        tgt2 = ttnn.linear(tgt2, self.ff_weights2, bias=getattr(self, "ff2_bias", None))
        tgt = ttnn.add(tgt, tgt2)

        if return_attn_weights:
            return tgt, None
        return tgt, None
