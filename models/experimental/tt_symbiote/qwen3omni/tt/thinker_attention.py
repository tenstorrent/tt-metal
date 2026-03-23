import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.linear import TTNNLinear, TTNNLinearIReplicatedWColSharded
from models.experimental.tt_symbiote.modules.rope import TTNNRotaryPositionEmbedding
from models.experimental.tt_symbiote.modules.attention import TTNNSDPAAttention


class TTNNQwen3OmniAttention(TTNNModule):
    """
    TTNN implementation of Qwen3 Omni Attention (without sliding window).

    Follows the Glm4MoeLiteAttention pattern for mesh compatibility:
    all-gather sharded hidden_states, then keep ALL intermediate tensors as
    raw ttnn tensors (ttnn.reshape / ttnn.permute / inline ttnn.rms_norm)
    so the framework never converts torch→ttnn at a module boundary — which
    would re-shard head_dim across devices.
    """

    def __init__(self):
        super().__init__()
        self.sdpa = TTNNSDPAAttention()
        self.rope = TTNNRotaryPositionEmbedding()
        self.core_grid = ttnn.CoreGrid(y=8, x=8)

    def init_parameters(self):
        self.q_proj = TTNNLinear.from_torch(self.torch_layer.q_proj)
        self.k_proj = TTNNLinear.from_torch(self.torch_layer.k_proj)
        self.v_proj = TTNNLinear.from_torch(self.torch_layer.v_proj)
        self.o_proj = TTNNLinearIReplicatedWColSharded.from_torch(self.torch_layer.o_proj)

    @classmethod
    def from_torch(cls, torch_layer):
        new_attn = cls()
        new_attn._fallback_torch_layer = torch_layer
        new_attn.head_dim = torch_layer.head_dim
        new_attn.scaling = torch_layer.scaling
        new_attn.is_causal = torch_layer.is_causal
        new_attn.num_key_value_groups = getattr(torch_layer, "num_key_value_groups", 1)
        new_attn.init_parameters()
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

        # Pre-place q_norm / k_norm weights for inline ttnn.rms_norm
        # (same pattern as Glm4MoeLiteAttention._kv_a_ln_weight)
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
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

    # ------------------------------------------------------------------ helpers

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _to_ttnn(self, tensor):
        """Extract the raw ttnn tensor, bypassing TorchTTNNTensor shard config."""
        return tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor

    def _maybe_all_gather(self, tensor):
        """Extract raw ttnn and all-gather; no-op when single-device."""
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

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        # 1) All-gather sharded hidden_states → full hidden_dim on every device
        expected_hidden = self.q_proj.in_features
        if hidden_states.shape[-1] != expected_hidden and self._is_distributed:
            hidden_states = self._maybe_all_gather(hidden_states)

        # 2) QKV projections (replicated weights, full input)
        #    Immediately extract raw ttnn so subsequent ops stay in ttnn land.
        query_states = self._to_ttnn(self.q_proj(hidden_states))
        key_states = self._to_ttnn(self.k_proj(hidden_states))
        value_states = self._to_ttnn(self.v_proj(hidden_states))

        batch_size = query_states.shape[0]
        seq_length = query_states.shape[1]
        num_q_heads = query_states.shape[-1] // self.head_dim
        num_kv_heads = key_states.shape[-1] // self.head_dim

        # 3) Reshape [B,S,H*D] → [B,H,S,D] using raw ttnn ops
        #    (ttnn.reshape / ttnn.permute keep data as ttnn — no torch conversion,
        #     so no re-sharding at the next module boundary)
        query_states = ttnn.reshape(query_states, (batch_size, seq_length, num_q_heads, self.head_dim))
        query_states = ttnn.permute(query_states, (0, 2, 1, 3))

        key_states = ttnn.reshape(key_states, (batch_size, seq_length, num_kv_heads, self.head_dim))
        key_states = ttnn.permute(key_states, (0, 2, 1, 3))

        value_states = ttnn.reshape(value_states, (batch_size, seq_length, num_kv_heads, self.head_dim))
        value_states = ttnn.permute(value_states, (0, 2, 1, 3))

        # 4) Q/K normalisation — inline ttnn.rms_norm (no module boundary)
        #    Same pattern as Glm4MoeLiteAttention line 1205.
        query_states = ttnn.rms_norm(query_states, weight=self._q_norm_weight, epsilon=self._q_norm_eps)
        key_states = ttnn.rms_norm(key_states, weight=self._k_norm_weight, epsilon=self._k_norm_eps)

        # 5) RoPE — all-gather cos/sin, then use TTNNRotaryPositionEmbedding
        #    (head_dim=128 after gather → 128 % 64 == 0 → satisfies kernel)
        cos, sin = position_embeddings
        cos = self._maybe_all_gather(cos)
        sin = self._maybe_all_gather(sin)

        query_states, key_states = self.rope(query_states, key_states, cos, sin)
        query_states = self._to_ttnn(query_states)
        key_states = self._to_ttnn(key_states)

        # 6) KV-cache update (torch round-trip for DynamicCache compat)
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

        # 7) Scaled dot-product attention
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

        # 8) Merge heads + output projection
        attn_output = ttnn.experimental.nlp_concat_heads(self._to_ttnn(attn_output))
        attn_output = ttnn.squeeze(attn_output, 1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None
