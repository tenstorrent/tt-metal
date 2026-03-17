# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.rope import TTNNRotaryPositionEmbedding
from models.experimental.tt_symbiote.modules.attention import TTNNSDPAAttention


class TTNNQwen3OmniMoeCode2WavAttention(TTNNModule):
    """
    Hybrid TT implementation of Qwen3 Code2Wav Attention

    Phase 1: Exact HF match using SDPA + sliding mask
    Phase 2: Optional optimized windowed attention
    """

    def __init__(self):
        super().__init__()

        self.sdpa = TTNNSDPAAttention()
        self.rope = TTNNRotaryPositionEmbedding()

        self.num_heads = None
        self.num_kv_heads = None
        self.num_kv_groups = None
        self.head_dim = None
        self.hidden_size = None

        self.scaling = None
        self.sliding_window = None

        self.use_windowed_attention = False  # 🔥 toggle for optimization

    # ------------------------------------------------------------
    # INIT FROM TORCH
    # ------------------------------------------------------------
    @classmethod
    def from_torch(cls, torch_attn):
        new_attn = cls()
        new_attn._fallback_torch_layer = torch_attn

        config = torch_attn.config

        new_attn.hidden_size = config.hidden_size
        new_attn.num_heads = config.num_attention_heads
        new_attn.num_kv_heads = config.num_key_value_heads
        new_attn.num_kv_groups = new_attn.num_heads // new_attn.num_kv_heads
        new_attn.head_dim = torch_attn.head_dim

        new_attn.scaling = torch_attn.scaling
        new_attn.sliding_window = config.sliding_window

        # projections
        new_attn.q_proj = TTNNLinear.from_torch(torch_attn.q_proj)
        new_attn.k_proj = TTNNLinear.from_torch(torch_attn.k_proj)
        new_attn.v_proj = TTNNLinear.from_torch(torch_attn.v_proj)
        new_attn.o_proj = TTNNLinear.from_torch(torch_attn.o_proj)

        return new_attn

    @staticmethod
    def _to_raw_ttnn(tensor):
        """Unwrap TorchTTNNTensor to raw ttnn.Tensor for ttnn ops that don't accept the wrapper."""
        if hasattr(tensor, "ttnn_tensor"):
            return tensor.ttnn_tensor
        return tensor

    # ------------------------------------------------------------
    # KV REPEAT (VERY IMPORTANT)
    # ------------------------------------------------------------
    def repeat_kv(self, x):
        if self.num_kv_groups == 1:
            return x

        x = self._to_raw_ttnn(x)
        B, H, S, D = x.shape

        x = ttnn.reshape(x, (B, H, 1, S, D))
        x = ttnn.repeat(x, (1, 1, self.num_kv_groups, 1, 1))
        x = ttnn.reshape(x, (B, H * self.num_kv_groups, S, D))

        return x

    # ------------------------------------------------------------
    # SLIDING WINDOW MASK (HF EXACT)
    # ------------------------------------------------------------
    def build_sliding_mask(self, seq_len, device, dtype):
        W = self.sliding_window

        mask = torch.full((seq_len, seq_len), float("-inf"))
        for i in range(seq_len):
            start = max(0, i - W)
            mask[i, start : i + 1] = 0

        mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,S,S]

        return ttnn.from_torch(
            mask,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------
    def forward(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        B, S, _ = hidden_states.shape

        # --------------------------------------------------------
        # QKV PROJECTION
        # --------------------------------------------------------
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # reshape → [B, H, S, D] (unwrap: TTNNLinear returns TorchTTNNTensor)
        query = ttnn.reshape(self._to_raw_ttnn(query), (B, S, self.num_heads, self.head_dim))
        key = ttnn.reshape(self._to_raw_ttnn(key), (B, S, self.num_kv_heads, self.head_dim))
        value = ttnn.reshape(self._to_raw_ttnn(value), (B, S, self.num_kv_heads, self.head_dim))

        query = ttnn.permute(query, (0, 2, 1, 3))
        key = ttnn.permute(key, (0, 2, 1, 3))
        value = ttnn.permute(value, (0, 2, 1, 3))

        # --------------------------------------------------------
        # ROPE
        # --------------------------------------------------------
        cos, sin = position_embeddings
        query, key = self.rope(query, key, cos, sin)

        # --------------------------------------------------------
        # KV CACHE
        # --------------------------------------------------------
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self._fallback_torch_layer.layer_idx)

        # --------------------------------------------------------
        # REPEAT KV
        # --------------------------------------------------------
        key = self.repeat_kv(key)
        value = self.repeat_kv(value)

        # --------------------------------------------------------
        # 🔥 HYBRID PATH
        # --------------------------------------------------------

        # ========================================================
        # 🚀 FAST PATH (OPTIONAL)
        # ========================================================
        if self.use_windowed_attention and S % self.sliding_window == 0:
            W = self.sliding_window
            num_windows = S // W

            # reshape into windows
            query_w = ttnn.view(query, (B, self.num_heads, num_windows, W, self.head_dim))
            key_w = ttnn.view(key, (B, self.num_heads, num_windows, W, self.head_dim))
            value_w = ttnn.view(value, (B, self.num_heads, num_windows, W, self.head_dim))

            # merge batch+window
            query_w = ttnn.reshape(query_w, (B * num_windows, self.num_heads, W, self.head_dim))
            key_w = ttnn.reshape(key_w, (B * num_windows, self.num_heads, W, self.head_dim))
            value_w = ttnn.reshape(value_w, (B * num_windows, self.num_heads, W, self.head_dim))

            attn_output = self.sdpa(
                self,
                query_w,
                key_w,
                value_w,
                None,
                dropout=0.0,
                scaling=self.scaling,
                is_causal=True,
                transpose_output=True,
            )

            # reshape back (unwrap: SDPA can return TorchTTNNTensor)
            attn_output = ttnn.reshape(
                self._to_raw_ttnn(attn_output),
                (B, num_windows, W, self.num_heads, self.head_dim),
            )
            attn_output = ttnn.permute(attn_output, (0, 3, 1, 2, 4))
            attn_output = ttnn.reshape(attn_output, (B, self.num_heads, S, self.head_dim))

        # ========================================================
        # ✅ EXACT HF PATH
        # ========================================================
        else:
            if attention_mask is None:
                attention_mask = self.build_sliding_mask(S, hidden_states.device(), hidden_states.dtype)

            attn_output = self.sdpa(
                self,
                query,
                key,
                value,
                attention_mask,
                dropout=0.0,
                scaling=self.scaling,
                is_causal=False,
                transpose_output=True,
            )

        # --------------------------------------------------------
        # OUTPUT PROJECTION (unwrap: SDPA can return TorchTTNNTensor)
        # --------------------------------------------------------
        attn_output = ttnn.reshape(self._to_raw_ttnn(attn_output), (B, S, self.hidden_size))
        output = self.o_proj(attn_output)

        return output, None
