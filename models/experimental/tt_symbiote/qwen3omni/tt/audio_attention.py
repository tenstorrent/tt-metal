import torch
import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.attention import TTNNSDPAAttention


class TTNNQwenAudioAttentionOptimized(TTNNModule):
    def __init__(self):
        super().__init__()

        self.embed_dim = None
        self.num_heads = None
        self.head_dim = None
        self.scaling = None

        self.qkv_proj = None
        self.out_proj = None

        self.sdpa = TTNNSDPAAttention()
        self.core_grid = ttnn.CoreGrid(y=8, x=8)

        self.is_causal = False

    @classmethod
    def from_torch(cls, torch_attn):
        new_attn = cls()
        new_attn._fallback_torch_layer = torch_attn

        new_attn.embed_dim = torch_attn.embed_dim
        new_attn.num_heads = torch_attn.num_heads
        new_attn.head_dim = torch_attn.head_dim
        new_attn.scaling = torch_attn.scaling

        # ---- fuse QKV weights ----

        qkv_weight = torch.cat(
            [
                torch_attn.q_proj.weight,
                torch_attn.k_proj.weight,
                torch_attn.v_proj.weight,
            ],
            dim=0,
        )

        q_bias = torch_attn.q_proj.bias
        k_bias = torch_attn.k_proj.bias
        v_bias = torch_attn.v_proj.bias

        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

        fused_qkv = torch.nn.Linear(
            torch_attn.embed_dim,
            torch_attn.embed_dim * 3,
            bias=True,
        )

        fused_qkv.weight = torch.nn.Parameter(qkv_weight)
        fused_qkv.bias = torch.nn.Parameter(qkv_bias)

        new_attn.qkv_proj = TTNNLinear.from_torch(fused_qkv)

        new_attn.out_proj = TTNNLinear.from_torch(torch_attn.out_proj)

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

    @staticmethod
    def _leading_seq_len(hidden_states) -> int:
        """TTNN path uses [seq, dim]; symbiote may wrap as [1, seq, dim]."""
        shape = hidden_states.shape
        if len(shape) == 2:
            return int(shape[0])
        if len(shape) == 3 and int(shape[0]) == 1:
            return int(shape[1])
        return int(shape[0])

    @staticmethod
    def _cu_seqlens_allows_ttnn_path(cu_seqlens, seq_len: int) -> bool:
        """HF passes cumulative lengths; TTNN path is full-sequence SDPA (one segment [0, seq_len])."""
        if cu_seqlens is None:
            return True
        if isinstance(cu_seqlens, torch.Tensor):
            cu = cu_seqlens.detach().cpu().flatten().to(torch.int64)
        else:
            cu = ttnn.to_torch(cu_seqlens).flatten().to(torch.int64)
        if cu.numel() < 2:
            return True
        n_seg = cu.numel() - 1
        if n_seg > 1:
            return False
        return int(cu[0].item()) == 0 and int(cu[-1].item()) == seq_len

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
        cu_seqlens=None,
        **kwargs,
    ):
        seq_len = self._leading_seq_len(hidden_states)

        if not self._cu_seqlens_allows_ttnn_path(cu_seqlens, seq_len):
            assert self._fallback_torch_layer is not None, "Audio attention fallback requires torch layer"
            hs = ttnn.to_torch(hidden_states)
            if hs.dim() == 3 and hs.shape[0] == 1:
                hs = hs.squeeze(0)
            if cu_seqlens is None:
                cu_t = None
            elif isinstance(cu_seqlens, torch.Tensor):
                cu_t = cu_seqlens.to(device=hs.device, dtype=torch.int32)
            else:
                cu_t = ttnn.to_torch(cu_seqlens).to(device=hs.device, dtype=torch.int32)
            am = attention_mask
            if am is not None and not isinstance(am, torch.Tensor):
                am = ttnn.to_torch(am)
            out = self._fallback_torch_layer(hidden_states=hs, attention_mask=am, cu_seqlens=cu_t, **kwargs)
            return ttnn.from_torch(
                out,
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.ROW_MAJOR,
            )

        if len(hidden_states.shape) == 3 and int(hidden_states.shape[0]) == 1:
            hidden_states = ttnn.squeeze(hidden_states, 0)

        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(
                hidden_states,
                ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # ---- fused QKV projection ----
        # nlp_create_qkv_heads expects 4D input [batch, 1, seq_len, embed_dim*3]
        hidden_states = ttnn.unsqueeze(hidden_states, 0)  # (seq, D) -> (1, seq, D)
        if len(hidden_states.shape) == 3:
            hidden_states = ttnn.unsqueeze(hidden_states, 1)  # (1, seq, D) -> (1, 1, seq, D)

        qkv = self.qkv_proj(hidden_states).ttnn_tensor

        qkv = ttnn.to_memory_config(qkv, ttnn.L1_MEMORY_CONFIG)

        query, key, value = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
        )

        ttnn.deallocate(qkv)

        # ---- scaled dot product attention ----

        attn_output = self.sdpa(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=False,
            transpose_output=False,
        )

        # Under symbiote, attn_output can be TorchTTNNTensor; nlp_concat_heads expects ttnn.Tensor
        attn_output_tt = getattr(attn_output, "to_ttnn", attn_output)
        attn_output = ttnn.experimental.nlp_concat_heads(attn_output_tt)

        attn_output = ttnn.squeeze(attn_output, 0)

        attn_output = self.out_proj(attn_output)

        return attn_output


# Alias for tests and callers that expect the non-Optimized name
TTNNQwenAudioAttention = TTNNQwenAudioAttentionOptimized
