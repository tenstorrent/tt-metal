import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.attention import TTNNSDPAAttention


class TTNNQwen3VLMoeVisionAttention(TTNNModule):
    """
    Optimized TTNN implementation using fused QKV + nlp_create_qkv_heads
    """

    def __init__(self):
        super().__init__()

        self.hidden_size = None
        self.num_heads = None
        self.head_dim = None
        self.scaling = None

        self.qkv = None
        self.proj = None

        self.sdpa = TTNNSDPAAttention()
        self.is_causal = False
        self.core_grid = ttnn.CoreGrid(y=8, x=8)

    @classmethod
    def from_torch(cls, torch_attn):
        new_attn = cls()
        new_attn._fallback_torch_layer = torch_attn

        config = torch_attn.config

        new_attn.hidden_size = config.hidden_size
        new_attn.num_heads = config.num_heads
        new_attn.head_dim = config.hidden_size // config.num_heads
        new_attn.scaling = new_attn.head_dim**-0.5

        new_attn.qkv = TTNNLinear.from_torch(torch_attn.qkv)
        new_attn.proj = TTNNLinear.from_torch(torch_attn.proj)

        return new_attn

    def move_weights_to_device_impl(self):
        super().move_weights_to_device_impl()

        if self.sdpa.program_config is None:
            self.sdpa.program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=256,
                k_chunk_size=256,
                exp_approx_mode=False,
            )

            self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

    def rotate_half(self, x):
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        return ttnn.concat([ttnn.neg(x2), x1], dim=-1)

    def apply_rotary_pos_emb_vision(self, q, k, cos, sin):
        # Expand cos/sin to exact shape of q/k (1, num_heads, seq_len, head_dim) so binary_ng
        # sees identical last-two-dims -> NONE broadcast (avoids "Invalid subtile broadcast type").
        if len(cos.shape) == 3:
            cos = ttnn.unsqueeze(cos, 1)
        if len(sin.shape) == 3:
            sin = ttnn.unsqueeze(sin, 1)
        # (1, 1, seq_len, head_dim) -> (1, num_heads, seq_len, head_dim)
        cos = ttnn.repeat_interleave(cos, self.num_heads, dim=1)
        sin = ttnn.repeat_interleave(sin, self.num_heads, dim=1)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _to_raw_ttnn(tensor):
        if hasattr(tensor, "ttnn_tensor"):
            return tensor.ttnn_tensor
        return tensor

    def forward(
        self,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=None,
        **kwargs,
    ):
        """
        hidden_states: (seq_len, hidden_size)
        """

        seq_len = hidden_states.shape[0]

        # Ensure layout
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(
                hidden_states,
                ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Add batch dim → (1, seq_len, hidden); then (1, 1, seq_len, hidden) for nlp_create_qkv_heads
        hidden_states = ttnn.unsqueeze(hidden_states, 0)
        if len(hidden_states.shape) == 3:
            hidden_states = ttnn.unsqueeze(hidden_states, 1)

        # QKV projection
        qkv = self.qkv(hidden_states).ttnn_tensor

        # Move to L1 for fast split
        qkv = ttnn.to_memory_config(qkv, ttnn.L1_MEMORY_CONFIG)

        # 🚀 Fused QKV split (IMPORTANT OPTIMIZATION)
        query, key, value = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
        )

        # Shapes now:
        # (1, num_heads, seq_len, head_dim)

        # Rotary embeddings
        cos, sin = position_embeddings
        cos = self._to_raw_ttnn(cos)
        sin = self._to_raw_ttnn(sin)

        # reshape cos/sin to (1, seq_len, head_dim)
        cos = ttnn.unsqueeze(cos, 0)
        sin = ttnn.unsqueeze(sin, 0)

        query, key = self.apply_rotary_pos_emb_vision(query, key, cos, sin)

        # Attention
        attn_output = self.sdpa(
            self,
            query,
            key,
            value,
            attention_mask=None,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=False,
            transpose_output=True,
        )

        # Merge heads → (1, seq_len, hidden)
        attn_output = ttnn.experimental.nlp_concat_heads(attn_output)

        # Remove batch dim → (seq_len, hidden)
        attn_output = ttnn.squeeze(attn_output, 0)

        # Output projection
        attn_output = self.proj(attn_output)

        return attn_output
