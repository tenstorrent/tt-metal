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
        self.q_weight = None
        self.k_weight = None
        self.v_weight = None
        self.q_bias = None
        self.k_bias = None
        self.v_bias = None
        self.out_weight = None
        self.out_bias = None

    def __call__(self, query, key, value, attn_mask=None):
        batch_size, seq_len, hidden_size = query.shape

        # Linear projection for Q, K, V using fused weights
        q = ttnn.linear(query, self.q_weight, bias=self.q_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.linear(key, self.k_weight, bias=self.k_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.linear(value, self.v_weight, bias=self.v_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        qkv = ttnn.concat([q, k, v], dim=2)

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
            attn_mask=attn_mask,
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
        dropout=0.0,
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

        self.self_attn = TTNNMultiheadAttention(d_model, nhead, device)
        self.multihead_attn = TTNNMultiheadAttention(d_model, nhead, device)

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
            self.self_attn.q_weight = parameters["self_attn"].get("q_weight")
            self.self_attn.k_weight = parameters["self_attn"].get("k_weight")
            self.self_attn.v_weight = parameters["self_attn"].get("v_weight")
            self.self_attn.q_bias = parameters["self_attn"].get("q_bias")
            self.self_attn.k_bias = parameters["self_attn"].get("k_bias")
            self.self_attn.v_bias = parameters["self_attn"].get("v_bias")
            self.self_attn.out_weight = parameters["self_attn"].get("out_weight")
            self.self_attn.out_bias = parameters["self_attn"].get("out_bias")

        # Cross-attention weights
        if "multihead_attn" in parameters:
            self.multihead_attn.q_weight = parameters["multihead_attn"].get("q_weight")
            self.multihead_attn.k_weight = parameters["multihead_attn"].get("k_weight")
            self.multihead_attn.v_weight = parameters["multihead_attn"].get("v_weight")
            self.multihead_attn.q_bias = parameters["multihead_attn"].get("q_bias")
            self.multihead_attn.k_bias = parameters["multihead_attn"].get("k_bias")
            self.multihead_attn.v_bias = parameters["multihead_attn"].get("v_bias")
            self.multihead_attn.out_weight = parameters["multihead_attn"].get("out_weight")
            self.multihead_attn.out_bias = parameters["multihead_attn"].get("out_bias")

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
        q = k = self.with_pos_embed(tgt, query_pos)

        # Self-attention using proper QKV projections
        tgt2 = self.self_attn(q, k, tgt, tgt_mask)
        # tgt2 = self.self_attention(q, k, tgt, query_pos, tgt_mask)
        tgt = ttnn.add(tgt, tgt2)
        tgt = ttnn.layer_norm(tgt, weight=self.norm1_weights, bias=getattr(self, "norm1_bias", None))

        # Cross-attention using proper QKV projections
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
        )
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
        q = k = self.with_pos_embed(tgt2, query_pos)
        # q=k=v=tgt2=tgt
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask)
        # return tgt2, None
        tgt = ttnn.add(tgt, tgt2)

        # Pre-norm cross-attention
        tgt2 = ttnn.layer_norm(tgt, weight=self.norm2_weights, bias=getattr(self, "norm2_bias", None))
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
        )
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


class TTTransformerEncoderLayer(LightweightModule):
    def __init__(
        self,
        device,
        d_model,
        nhead=4,
        dim_feedforward=128,
        dropout=0.0,
        normalize_before=True,
        use_ffn=True,
        model_config=None,
        parameters=None,
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.normalize_before = normalize_before
        self.use_ffn = use_ffn
        self.model_config = model_config or {}

        self.self_attn = TTNNMultiheadAttention(d_model, nhead, device)

        # Load preprocessed parameters
        if parameters is not None:
            self.load_parameters(parameters)
        else:
            # Initialize weights as None (for backward compatibility)
            self.self_attn_weights = None
            self.ff_weights1 = None
            self.ff_weights2 = None
            self.norm1_weights = None
            self.norm2_weights = None

    def load_parameters(self, parameters):
        """Load preprocessed parameters from the preprocessor"""
        # Self-attention weights
        if "self_attn" in parameters:
            self.self_attn.q_weight = parameters["self_attn"].get("q_weight")
            self.self_attn.k_weight = parameters["self_attn"].get("k_weight")
            self.self_attn.v_weight = parameters["self_attn"].get("v_weight")
            self.self_attn.q_bias = parameters["self_attn"].get("q_bias")
            self.self_attn.k_bias = parameters["self_attn"].get("k_bias")
            self.self_attn.v_bias = parameters["self_attn"].get("v_bias")
            self.self_attn.out_weight = parameters["self_attn"].get("out_weight")
            self.self_attn.out_bias = parameters["self_attn"].get("out_bias")

        # Feedforward weights
        if "linear1" in parameters:
            self.ff_weights1 = parameters["linear1"]["weight"]
            self.ff1_bias = parameters["linear1"].get("bias")
        if "linear2" in parameters:
            self.ff_weights2 = parameters["linear2"]["weight"]
            self.ff2_bias = parameters["linear2"].get("bias")

        # Normalization weights
        for i, norm_name in enumerate(["norm1", "norm2"], 1):
            if norm_name in parameters:
                setattr(self, f"norm{i}_weights", parameters[norm_name]["weight"])
                setattr(self, f"norm{i}_bias", parameters[norm_name].get("bias"))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else ttnn.add(tensor, pos)

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        pos=None,
        return_attn_weights=False,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, return_attn_weights)
        else:
            return self.forward_post(src, src_mask, src_key_padding_mask, pos, return_attn_weights)

    def forward_post(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        pos=None,
        return_attn_weights=False,
    ):
        q = k = self.with_pos_embed(src, pos)
        value = src

        # Self-attention
        src2 = self.self_attn(q, k, value, src_mask)
        src = ttnn.add(src, src2)
        src = ttnn.layer_norm(src, weight=self.norm1_weights, bias=getattr(self, "norm1_bias", None))

        # Feedforward network
        if self.use_ffn:
            src2 = ttnn.linear(src, self.ff_weights1, bias=getattr(self, "ff1_bias", None))
            src2 = ttnn.relu(src2)
            src2 = ttnn.linear(src2, self.ff_weights2, bias=getattr(self, "ff2_bias", None))
            src = ttnn.add(src, src2)
            src = ttnn.layer_norm(src, weight=self.norm2_weights, bias=getattr(self, "norm2_bias", None))

        if return_attn_weights:
            return src, None  # TTNN doesn't return attention weights
        return src

    def forward_pre(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        pos=None,
        return_attn_weights=False,
    ):
        # Pre-norm self-attention
        src2 = ttnn.layer_norm(src, weight=self.norm1_weights, bias=getattr(self, "norm1_bias", None))
        value = src2
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value, src_mask)
        src = ttnn.add(src, src2)

        # Pre-norm feedforward
        if self.use_ffn:
            src2 = ttnn.layer_norm(src, weight=self.norm2_weights, bias=getattr(self, "norm2_bias", None))
            src2 = ttnn.linear(src2, self.ff_weights1, bias=getattr(self, "ff1_bias", None))
            src2 = ttnn.relu(src2)
            src2 = ttnn.linear(src2, self.ff_weights2, bias=getattr(self, "ff2_bias", None))
            src = ttnn.add(src, src2)

        if return_attn_weights:
            return src, None
        return src
