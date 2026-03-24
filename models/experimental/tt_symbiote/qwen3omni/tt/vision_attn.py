import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.linear import TTNNLinear, TTNNLinearIReplicatedWColSharded
from models.experimental.tt_symbiote.modules.attention import TTNNSDPAAttention

# Long vision sequences (e.g. ~24k tokens) + mesh matmul can allocate multi-GB DRAM for QKV output.
# Chunking caps peak activation memory (same logical result).
_VISION_ATTN_SEQ_CHUNK = 4096


class TTNNQwen3VLMoeVisionAttention(TTNNModule):
    """
    TTNN implementation of Qwen3VLMoeVisionAttention (image / video frames in the vision tower).

    Mesh compatibility matches ``TTNNQwen3OmniAttention`` / audio attention: all-gather
    sharded activations on the last dim before linear/rotary so head_dim is not split across
    devices, and col-sharded output projection like thinker ``o_proj``.
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
        new_attn.proj = TTNNLinearIReplicatedWColSharded.from_torch(torch_attn.proj)

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

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _to_ttnn(self, tensor):
        """Extract raw ttnn tensor, bypassing TorchTTNNTensor shard config."""
        return tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor

    def _maybe_all_gather(self, tensor):
        """All-gather sharded last dim so QKV / rotary see full features (thinker_attention pattern)."""
        t = self._to_ttnn(tensor)
        if not self._is_distributed:
            return t
        return ttnn.experimental.all_gather_async(
            t,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Ring,
        )

    @staticmethod
    def _last_hidden_dim(hidden_states) -> int:
        return int(hidden_states.shape[-1])

    def _leading_seq_len(self, hidden_states) -> int:
        """Infer sequence length when mesh layouts differ (see ``TTNNQwenAudioAttentionOptimized``)."""
        t = self._to_ttnn(hidden_states)
        shape = t.shape
        hs = self.hidden_size
        if len(shape) == 2:
            return int(shape[0])
        if len(shape) == 3:
            last = int(shape[-1])
            if hs is not None and last == hs:
                return int(shape[1])
            if int(shape[0]) == 1:
                return int(shape[1])
        return int(shape[0])

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

    def _slice_along_seq(self, tensor, s: int, e: int):
        """Slice ``tensor`` on sequence dimension ``[s, e)`` using the tensor's actual shape."""
        t = self._to_ttnn(tensor)
        rank = len(t.shape)
        last = int(t.shape[-1])
        if rank == 2:
            return ttnn.slice(t, (s, 0), (e, last))
        if rank == 3 and int(t.shape[0]) == 1:
            return ttnn.slice(t, (0, s, 0), (1, e, last))
        if rank == 4:
            d1, d2, d3 = int(t.shape[1]), int(t.shape[2]), int(t.shape[3])
            return ttnn.slice(t, (s, 0, 0, 0), (e, d1, d2, d3))
        return ttnn.slice(t, (s, 0), (e, last))

    def _slice_rotary(self, cos_or_sin, s: int, e: int):
        """Slice rotary embeddings on sequence axis, respecting per-device shard width."""
        t = self._to_ttnn(cos_or_sin)
        last = int(t.shape[-1])
        return ttnn.slice(t, (s, 0), (e, last))

    def _forward_chunk(
        self,
        hidden_states,
        position_embeddings,
        seq_len: int,
    ):
        """Run attention on one sequence chunk (``seq_len`` = tokens in this chunk)."""

        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(
                hidden_states,
                ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        qkv = self.qkv(hidden_states)
        qkv = ttnn.reshape(
            self._to_raw_ttnn(qkv),
            (seq_len, 3, self.num_heads, self.head_dim),
        )

        q = qkv[:, 0]
        k = qkv[:, 1]
        v = qkv[:, 2]

        cos, sin = position_embeddings

        q, k = self.apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = ttnn.permute(q, (1, 0, 2))
        k = ttnn.permute(k, (1, 0, 2))
        v = ttnn.permute(v, (1, 0, 2))

        q = ttnn.unsqueeze(q, 0)
        k = ttnn.unsqueeze(k, 0)
        v = ttnn.unsqueeze(v, 0)

        head_dim_padded = ((self.head_dim + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        if head_dim_padded != self.head_dim:
            pad_size = head_dim_padded - self.head_dim
            q = ttnn.pad(q, ((0, 0), (0, 0), (0, 0), (0, pad_size)), value=0.0)
            k = ttnn.pad(k, ((0, 0), (0, 0), (0, 0), (0, pad_size)), value=0.0)
            v = ttnn.pad(v, ((0, 0), (0, 0), (0, 0), (0, pad_size)), value=0.0)

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

        attn_output = self._to_raw_ttnn(attn_output)
        if head_dim_padded != self.head_dim:
            attn_output = attn_output[:, :, :, : self.head_dim]

        attn_output = self._to_ttnn(attn_output)
        if self._is_distributed and int(attn_output.shape[-1]) != self.head_dim:
            attn_output = self._maybe_all_gather(attn_output)

        attn_output = ttnn.reshape(
            self._to_raw_ttnn(attn_output),
            (seq_len, self.hidden_size),
        )

        return self.proj(attn_output)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
        """
        hidden_states: (seq_len, hidden_size) or mesh layouts with sharded last dim.
        """

        expected_hidden = self.hidden_size
        if self._last_hidden_dim(self._to_ttnn(hidden_states)) != expected_hidden and self._is_distributed:
            hidden_states = self._maybe_all_gather(hidden_states)

        hidden_states = self._to_ttnn(hidden_states)
        if len(hidden_states.shape) == 3 and int(hidden_states.shape[0]) == 1:
            hidden_states = ttnn.squeeze(hidden_states, 0)

        seq_len = self._leading_seq_len(hidden_states)

        cos, sin = position_embeddings

        if seq_len <= _VISION_ATTN_SEQ_CHUNK:
            cos_sin = (self._maybe_all_gather(cos), self._maybe_all_gather(sin))
            return self._forward_chunk(hidden_states, cos_sin, seq_len)

        out_chunks = []
        for s in range(0, seq_len, _VISION_ATTN_SEQ_CHUNK):
            e = min(s + _VISION_ATTN_SEQ_CHUNK, seq_len)
            chunk_len = e - s
            hs_chunk = self._slice_along_seq(hidden_states, s, e)
            cos_chunk = self._slice_rotary(cos, s, e)
            sin_chunk = self._slice_rotary(sin, s, e)
            cos_sin = (self._maybe_all_gather(cos_chunk), self._maybe_all_gather(sin_chunk))
            out_chunks.append(self._forward_chunk(hs_chunk, cos_sin, chunk_len))

        if len(out_chunks) == 1:
            return out_chunks[0]
        raw_chunks = [self._to_raw_ttnn(c) for c in out_chunks]
        return ttnn.concat(raw_chunks, dim=0)
