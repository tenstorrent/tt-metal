import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.attention import TTNNSDPAAttention


class TTNNQwen3VLMoeVisionAttention(TTNNModule):
    """
    TTNN implementation of Qwen3VLMoeVisionAttention.
    Mirrors the PyTorch implementation but executes attention using TTNN kernels.
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
        """
        Create TTNNQwen3VLMoeVisionAttention from PyTorch attention.
        """

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
        """
        Configure SDPA kernels once device is known.
        """

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
        cos = ttnn.unsqueeze(cos, -2)
        sin = ttnn.unsqueeze(sin, -2)

        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed

    @staticmethod
    def _to_raw_ttnn(tensor, max_unwrap=5):
        """Unwrap TorchTTNNTensor to raw ttnn.Tensor for ttnn ops that don't accept the wrapper."""
        for _ in range(max_unwrap):
            if type(tensor).__name__ != "TorchTTNNTensor":
                return tensor
            raw = getattr(tensor, "ttnn_tensor", None)
            if raw is None:
                raw = tensor.to_ttnn
            tensor = raw
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

        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(
                hidden_states,
                ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # QKV projection
        qkv = self.qkv(hidden_states)
        # reshape to heads (unwrap so ttnn.reshape gets raw ttnn.Tensor)
        qkv = ttnn.reshape(
            self._to_raw_ttnn(qkv),
            (seq_len, 3, self.num_heads, self.head_dim),
        )

        # split
        q = qkv[:, 0]
        k = qkv[:, 1]
        v = qkv[:, 2]

        cos, sin = position_embeddings

        q, k = self.apply_rotary_pos_emb_vision(q, k, cos, sin)

        # reshape to SDPA format
        q = ttnn.permute(q, (1, 0, 2))
        k = ttnn.permute(k, (1, 0, 2))
        v = ttnn.permute(v, (1, 0, 2))

        q = ttnn.unsqueeze(q, 0)
        k = ttnn.unsqueeze(k, 0)
        v = ttnn.unsqueeze(v, 0)

        attn_output = self.sdpa(
            self,
            q,
            k,
            v,
            attention_mask=None,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=False,
            transpose_output=True,
        )

        # Under symbiote, attn_output can be TorchTTNNTensor; reshape expects raw ttnn.Tensor.
        attn_output = ttnn.reshape(
            self._to_raw_ttnn(attn_output),
            (seq_len, self.hidden_size),
        )

        attn_output = self.proj(attn_output)

        return attn_output
