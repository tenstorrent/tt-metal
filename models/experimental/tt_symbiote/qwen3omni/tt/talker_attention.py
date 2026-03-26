import os

import ttnn

from models.experimental.tt_symbiote.core.utils import safe_permute
from models.experimental.tt_symbiote.core.module import TTNNModule

from models.experimental.tt_symbiote.modules.linear import TTNNLinear, TTNNLinearIReplicatedWColSharded
from models.experimental.tt_symbiote.modules.rope import TTNNRotaryPositionEmbedding
from models.experimental.tt_symbiote.modules.attention import TTNNSDPAAttention


class TTNNQwen3Attention(TTNNModule):
    """
    TTNN implementation of Qwen3 Attention with sliding-window support
    """

    def __init__(self):
        super().__init__()

        self.sdpa = TTNNSDPAAttention()
        self.rope = TTNNRotaryPositionEmbedding()

        self.core_grid = ttnn.CoreGrid(y=8, x=8)

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

        # Pre-place q/k norm weights for inline ttnn.rms_norm to avoid module-boundary reshards.
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self._is_distributed else None
        q_norm = self.torch_layer.q_norm
        self._q_norm_weight = ttnn.from_torch(
            q_norm.weight.unsqueeze(0).expand(32, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._q_norm_eps = q_norm.variance_epsilon

        k_norm = self.torch_layer.k_norm
        self._k_norm_weight = ttnn.from_torch(
            k_norm.weight.unsqueeze(0).expand(32, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._k_norm_eps = k_norm.variance_epsilon

    # -------------------------
    # Parameter initialization
    # -------------------------

    def init_parameters(self):
        self.q_proj = TTNNLinear.from_torch(self.torch_layer.q_proj)
        self.k_proj = TTNNLinear.from_torch(self.torch_layer.k_proj)
        self.v_proj = TTNNLinear.from_torch(self.torch_layer.v_proj)
        # Use mesh-safe output projection (same pattern as thinker attention).
        self.o_proj = TTNNLinearIReplicatedWColSharded.from_torch(self.torch_layer.o_proj)

    @classmethod
    def from_torch(cls, torch_layer):
        new_attn = cls()

        new_attn._fallback_torch_layer = torch_layer

        new_attn.num_key_value_groups = getattr(torch_layer, "num_key_value_groups", 1)

        new_attn.head_dim = torch_layer.head_dim
        new_attn.scaling = torch_layer.scaling
        new_attn.is_causal = torch_layer.is_causal

        new_attn.sliding_window = getattr(torch_layer, "sliding_window", None)

        new_attn.init_parameters()

        return new_attn

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _to_ttnn(self, tensor):
        return tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor

    def _maybe_all_gather(self, tensor):
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

    def _effective_batch_chunk(self, batch_size):
        v = os.environ.get("TT_SYMBIOTE_ATTN_BATCH_CHUNK")
        if v is not None:
            return max(0, int(v))
        if batch_size > 256:
            return 64
        return 0

    def _reshape_permute_bshd_to_bhsd(self, x, batch_size, seq_length, num_heads, head_dim):
        chunk_b = self._effective_batch_chunk(batch_size)
        if chunk_b <= 0 or batch_size <= chunk_b:
            x = ttnn.reshape(x, (batch_size, seq_length, num_heads, head_dim))
            return safe_permute(x, (0, 2, 1, 3))

        hidden = num_heads * head_dim
        parts = []
        for b0 in range(0, batch_size, chunk_b):
            b1 = min(b0 + chunk_b, batch_size)
            xc = ttnn.slice(x, (b0, 0, 0), (b1, seq_length, hidden))
            xc = ttnn.reshape(xc, (b1 - b0, seq_length, num_heads, head_dim))
            xc = safe_permute(xc, (0, 2, 1, 3))
            parts.append(xc)
        return parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # -------------------------
    # Forward
    # -------------------------

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        # All-gather sharded hidden_states so QKV projections see full hidden dim.
        expected_hidden = self.q_proj.in_features
        if hidden_states.shape[-1] != expected_hidden and self._is_distributed:
            hidden_states = self._maybe_all_gather(hidden_states)

        # -------------------------
        # Q K V projections
        # -------------------------

        query_states = self._to_ttnn(self.q_proj(hidden_states))
        key_states = self._to_ttnn(self.k_proj(hidden_states))
        value_states = self._to_ttnn(self.v_proj(hidden_states))

        batch_size = query_states.shape[0]
        seq_length = query_states.shape[1]
        num_q_heads = query_states.shape[-1] // self.head_dim
        num_kv_heads = key_states.shape[-1] // self.head_dim

        query_states = self._reshape_permute_bshd_to_bhsd(
            query_states, batch_size, seq_length, num_q_heads, self.head_dim
        )
        key_states = self._reshape_permute_bshd_to_bhsd(key_states, batch_size, seq_length, num_kv_heads, self.head_dim)
        value_states = self._reshape_permute_bshd_to_bhsd(
            value_states, batch_size, seq_length, num_kv_heads, self.head_dim
        )

        # -------------------------
        # Q / K normalization
        # -------------------------

        query_states = ttnn.rms_norm(query_states, weight=self._q_norm_weight, epsilon=self._q_norm_eps)
        key_states = ttnn.rms_norm(key_states, weight=self._k_norm_weight, epsilon=self._k_norm_eps)

        # -------------------------
        # Rotary positional embedding
        # -------------------------

        cos, sin = position_embeddings
        cos = self._maybe_all_gather(cos)
        sin = self._maybe_all_gather(sin)

        query_states, key_states = self.rope(
            query_states,
            key_states,
            cos,
            sin,
        )
        query_states = self._to_ttnn(query_states)
        key_states = self._to_ttnn(key_states)

        # -------------------------
        # KV cache
        # -------------------------

        if past_key_values is not None:
            mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0) if self._is_distributed else None
            k_torch = ttnn.to_torch(key_states, mesh_composer=mesh_composer)
            v_torch = ttnn.to_torch(value_states, mesh_composer=mesh_composer)
            if self._is_distributed:
                k_torch = k_torch[:1]
                v_torch = v_torch[:1]

            k_torch, v_torch = past_key_values.update(
                k_torch,
                v_torch,
                self.torch_layer.layer_idx,
            )

            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self._is_distributed else None
            key_states = ttnn.from_torch(
                k_torch.contiguous(),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            value_states = ttnn.from_torch(
                v_torch.contiguous(),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Keep full-sequence SDPA output shape [B, H, S, D].
        # The chunked sliding-window path can collapse sequence length in prefill
        # and break residual add shape (e.g., 1024 vs 8192).
        attn_output = self.sdpa(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=self.is_causal,
            transpose_output=False,
        )

        # -------------------------
        # Merge heads
        # -------------------------
        # Under symbiote, attn_output can be TorchTTNNTensor; nlp_concat_heads expects ttnn.Tensor
        attn_output_tt = getattr(attn_output, "to_ttnn", attn_output)
        attn_output = ttnn.experimental.nlp_concat_heads(attn_output_tt)

        attn_output = ttnn.squeeze(attn_output, 1)

        # -------------------------
        # Output projection
        # -------------------------

        attn_output = self.o_proj(attn_output)

        return attn_output, None
