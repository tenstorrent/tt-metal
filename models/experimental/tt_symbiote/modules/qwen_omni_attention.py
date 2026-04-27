# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.attention import TTNNSDPAAttention
from models.experimental.tt_symbiote.modules.linear import TTNNLinear, TTNNLinearIReplicatedWColSharded
from models.experimental.tt_symbiote.modules.rope import TTNNRotaryPositionEmbedding
import ttnn
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
import torch

_VISION_ATTN_SEQ_CHUNK = 4096


class TTNNQwen3VLMoeVisionAttention(TTNNModule):
    """TTNN implementation of Qwen3VLMoeVisionAttention (image / video frames in the vision tower)."""

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
        new_attn.proj = TTNNLinearIReplicatedWColSharded.from_torch(torch_attn.proj)

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

    @staticmethod
    def _last_hidden_dim(hidden_states) -> int:
        return int(hidden_states.shape[-1])

    def _leading_seq_len(self, hidden_states) -> int:
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
        for _ in range(max_unwrap):
            if type(tensor).__name__ != "TorchTTNNTensor":
                return tensor
            raw = getattr(tensor, "ttnn_tensor", None)
            if raw is None:
                raw = tensor.to_ttnn
            tensor = raw
        return tensor

    def _slice_along_seq(self, tensor, s: int, e: int):
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
        t = self._to_ttnn(cos_or_sin)
        last = int(t.shape[-1])
        return ttnn.slice(t, (s, 0), (e, last))

    def _forward_chunk(
        self,
        hidden_states,
        position_embeddings,
        seq_len: int,
    ):
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


class TTNNQwen3OmniAttention(TTNNModule):
    """Qwen3-Omni thinker attention (no sliding window) on TTNN."""

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

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _is_symbiote_replicated(self, tensor) -> bool:
        """True if TorchTTNNTensor uses ReplicateTensorToMesh (skip dim=-1 AG on cos/sin or RoPE breaks)."""
        if isinstance(tensor, TorchTTNNTensor):
            cfg = tensor.ttnn_distributed_tensor_config
            if cfg is not None and cfg.mesh_mapper is not None:
                return "Replicate" in type(cfg.mesh_mapper).__name__
        return False

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

    def _maybe_all_gather_if_col_sharded(self, tensor):
        """All-gather column-sharded activations; pass through replicated tensors unchanged."""
        t = self._to_ttnn(tensor)
        if not self._is_distributed:
            return t
        if self._is_symbiote_replicated(tensor):
            return t
        return ttnn.experimental.all_gather_async(
            t,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Ring,
        )

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        expected_hidden = self.q_proj.in_features
        if hidden_states.shape[-1] != expected_hidden and self._is_distributed:
            hidden_states = self._maybe_all_gather(hidden_states)

        query_states = self._to_ttnn(self.q_proj(hidden_states))
        key_states = self._to_ttnn(self.k_proj(hidden_states))
        value_states = self._to_ttnn(self.v_proj(hidden_states))

        batch_size = query_states.shape[0]
        seq_length = query_states.shape[1]
        num_q_heads = query_states.shape[-1] // self.head_dim
        num_kv_heads = key_states.shape[-1] // self.head_dim

        query_states = ttnn.reshape(query_states, (batch_size, seq_length, num_q_heads, self.head_dim))
        query_states = ttnn.permute(query_states, (0, 2, 1, 3))

        key_states = ttnn.reshape(key_states, (batch_size, seq_length, num_kv_heads, self.head_dim))
        key_states = ttnn.permute(key_states, (0, 2, 1, 3))

        value_states = ttnn.reshape(value_states, (batch_size, seq_length, num_kv_heads, self.head_dim))
        value_states = ttnn.permute(value_states, (0, 2, 1, 3))

        query_states = ttnn.rms_norm(query_states, weight=self._q_norm_weight, epsilon=self._q_norm_eps)
        key_states = ttnn.rms_norm(key_states, weight=self._k_norm_weight, epsilon=self._k_norm_eps)

        cos, sin = position_embeddings
        cos = self._maybe_all_gather_if_col_sharded(cos)
        sin = self._maybe_all_gather_if_col_sharded(sin)

        query_states, key_states = self.rope(query_states, key_states, cos, sin)
        query_states = self._to_ttnn(query_states)
        key_states = self._to_ttnn(key_states)

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

        attn_output = ttnn.experimental.nlp_concat_heads(self._to_ttnn(attn_output))
        attn_output = ttnn.squeeze(attn_output, 1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None


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

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        expected_hidden = self.q_proj.in_features
        if hidden_states.shape[-1] != expected_hidden and self._is_distributed:
            hidden_states = self._maybe_all_gather(hidden_states)

        query_states = self._to_ttnn(self.q_proj(hidden_states))
        key_states = self._to_ttnn(self.k_proj(hidden_states))
        value_states = self._to_ttnn(self.v_proj(hidden_states))

        batch_size = query_states.shape[0]
        seq_length = query_states.shape[1]
        num_q_heads = query_states.shape[-1] // self.head_dim
        num_kv_heads = key_states.shape[-1] // self.head_dim

        query_states = ttnn.reshape(query_states, (batch_size, seq_length, num_q_heads, self.head_dim))
        query_states = ttnn.permute(query_states, (0, 2, 1, 3))

        key_states = ttnn.reshape(key_states, (batch_size, seq_length, num_kv_heads, self.head_dim))
        key_states = ttnn.permute(key_states, (0, 2, 1, 3))

        value_states = ttnn.reshape(value_states, (batch_size, seq_length, num_kv_heads, self.head_dim))
        value_states = ttnn.permute(value_states, (0, 2, 1, 3))

        query_states = ttnn.rms_norm(query_states, weight=self._q_norm_weight, epsilon=self._q_norm_eps)
        key_states = ttnn.rms_norm(key_states, weight=self._k_norm_weight, epsilon=self._k_norm_eps)

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

        attn_output_tt = getattr(attn_output, "to_ttnn", attn_output)
        attn_output = ttnn.experimental.nlp_concat_heads(attn_output_tt)
        attn_output = ttnn.squeeze(attn_output, 1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class TTNNQwen3OmniMoeCode2WavAttention(TTNNModule):
    """TTNN implementation of Qwen3OmniMoeCode2WavAttention (code2wav in the code2wav tower)."""

    def __init__(self):
        super().__init__()

        self.sdpa = TTNNSDPAAttention()
        self.rope = TTNNRotaryPositionEmbedding()
        self.core_grid = ttnn.CoreGrid(y=8, x=8)

        self.num_heads = None
        self.num_kv_heads = None
        self.num_kv_groups = None
        self.head_dim = None
        self.hidden_size = None

        self.scaling = None
        self.sliding_window = None

        self.use_windowed_attention = False

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
        new_attn.o_proj = TTNNLinearIReplicatedWColSharded.from_torch(torch_attn.o_proj)

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

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _to_ttnn(self, tensor):
        """Extract raw ttnn tensor (bypass TorchTTNNTensor shard metadata)."""
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

    def _is_symbiote_replicated(self, tensor) -> bool:
        """Replicated cos/sin must not be all-gathered on dim=-1 (concat breaks RoPE last dim)."""
        if isinstance(tensor, TorchTTNNTensor):
            cfg = tensor.ttnn_distributed_tensor_config
            if cfg is not None and cfg.mesh_mapper is not None:
                return "Replicate" in type(cfg.mesh_mapper).__name__
        return False

    def _maybe_all_gather_if_col_sharded(self, tensor):
        t = self._to_ttnn(tensor)
        if not self._is_distributed:
            return t
        if self._is_symbiote_replicated(tensor):
            return t
        return ttnn.experimental.all_gather_async(
            t,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Ring,
        )

    def _prepare_code2wav_rotary_cos_sin(self, cos, sin, seq_len: int):
        """Full sequence on every chip for RoPE; skip dim=-1 gather on replicated cos (breaks head dim)."""
        if not self._is_distributed or self.device.get_num_devices() <= 1:
            return self._maybe_all_gather_if_col_sharded(cos), self._maybe_all_gather_if_col_sharded(sin)

        nd = int(self.device.get_num_devices())
        cos_t = self._to_ttnn(cos)
        sin_t = self._to_ttnn(sin)
        gather_dim = None
        for d in range(len(cos_t.shape)):
            sl = int(cos_t.shape[d])
            if sl != seq_len and sl * nd == seq_len:
                gather_dim = d
                break
        if gather_dim is not None:
            cos_t = ttnn.experimental.all_gather_async(
                cos_t,
                dim=gather_dim,
                multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
                barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
                num_links=1,
                topology=ttnn.Topology.Ring,
            )
            sin_t = ttnn.experimental.all_gather_async(
                sin_t,
                dim=gather_dim,
                multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
                barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
                num_links=1,
                topology=ttnn.Topology.Ring,
            )
            return cos_t, sin_t

        if self._is_symbiote_replicated(cos):
            return cos_t, sin_t
        return self._maybe_all_gather_if_col_sharded(cos), self._maybe_all_gather_if_col_sharded(sin)

    @staticmethod
    def _to_raw_ttnn(tensor):
        """Unwrap TorchTTNNTensor to raw ttnn.Tensor for ttnn ops that don't accept the wrapper."""
        if hasattr(tensor, "ttnn_tensor"):
            return tensor.ttnn_tensor
        return tensor

    def repeat_kv(self, x):
        if self.num_kv_groups == 1:
            return x

        x = self._to_raw_ttnn(x)
        B, H, S, D = x.shape

        x = ttnn.reshape(x, (B, H, 1, S, D))
        x = ttnn.repeat(x, (1, 1, self.num_kv_groups, 1, 1))
        x = ttnn.reshape(x, (B, H * self.num_kv_groups, S, D))

        return x

    def build_sliding_mask(self, seq_len):
        W = self.sliding_window

        mask = torch.full((seq_len, seq_len), float("-inf"))
        for i in range(seq_len):
            start = max(0, i - W)
            mask[i, start : i + 1] = 0

        mask = mask.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)  # [1,1,S,S]

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self._is_distributed else None
        return ttnn.from_torch(
            mask,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _mesh_replicate_full_attention_mask(self, attention_mask, seq_len: int):
        t = self._to_ttnn(attention_mask)
        if not self._is_distributed:
            if t.dtype != ttnn.bfloat16:
                return ttnn.typecast(t, ttnn.bfloat16)
            return t

        if int(t.shape[-1]) == int(t.shape[-2]) == seq_len:
            if t.dtype != ttnn.bfloat16:
                return ttnn.typecast(t, ttnn.bfloat16)
            return attention_mask

        mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=-1)
        torch_m = ttnn.to_torch(t, mesh_composer=mesh_composer)
        if torch_m.dtype in (torch.uint8, torch.bool):
            tf = torch_m.to(torch.float32)
            torch_m = torch.where(tf > 0, 0.0, float("-inf")).to(torch.bfloat16)
        else:
            torch_m = torch_m.to(torch.bfloat16)

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.from_torch(
            torch_m.contiguous(),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        expected_hidden = self.q_proj.in_features
        if hidden_states.shape[-1] != expected_hidden and self._is_distributed:
            hidden_states = self._maybe_all_gather(hidden_states)

        hs = self._to_ttnn(hidden_states)
        B, S, _ = hs.shape

        query = self._to_ttnn(self.q_proj(hidden_states))
        key = self._to_ttnn(self.k_proj(hidden_states))
        value = self._to_ttnn(self.v_proj(hidden_states))

        query = ttnn.reshape(query, (B, S, self.num_heads, self.head_dim))
        key = ttnn.reshape(key, (B, S, self.num_kv_heads, self.head_dim))
        value = ttnn.reshape(value, (B, S, self.num_kv_heads, self.head_dim))

        query = ttnn.permute(query, (0, 2, 1, 3))
        key = ttnn.permute(key, (0, 2, 1, 3))
        value = ttnn.permute(value, (0, 2, 1, 3))

        cos, sin = position_embeddings
        cos, sin = self._prepare_code2wav_rotary_cos_sin(cos, sin, S)
        query, key = self.rope(query, key, cos, sin)
        query = self._to_ttnn(query)
        key = self._to_ttnn(key)

        if past_key_values is not None:
            mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0) if self._is_distributed else None
            k_torch = ttnn.to_torch(key, mesh_composer=mesh_composer)
            v_torch = ttnn.to_torch(value, mesh_composer=mesh_composer)
            if self._is_distributed:
                k_torch = k_torch[:1]
                v_torch = v_torch[:1]

            k_torch, v_torch = past_key_values.update(
                k_torch,
                v_torch,
                self._fallback_torch_layer.layer_idx,
            )

            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self._is_distributed else None
            key = ttnn.from_torch(
                k_torch.contiguous(),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            value = ttnn.from_torch(
                v_torch.contiguous(),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        key = self.repeat_kv(key)
        value = self.repeat_kv(value)

        if self.use_windowed_attention and past_key_values is None and S % self.sliding_window == 0:
            W = self.sliding_window
            num_windows = S // W

            query_w = ttnn.view(query, (B, self.num_heads, num_windows, W, self.head_dim))
            key_w = ttnn.view(key, (B, self.num_heads, num_windows, W, self.head_dim))
            value_w = ttnn.view(value, (B, self.num_heads, num_windows, W, self.head_dim))

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

            attn_output = ttnn.reshape(
                self._to_raw_ttnn(attn_output),
                (B, num_windows, W, self.num_heads, self.head_dim),
            )
            attn_output = ttnn.permute(attn_output, (0, 3, 1, 2, 4))
            attn_output = ttnn.reshape(attn_output, (B, self.num_heads, S, self.head_dim))

        else:
            if attention_mask is None:
                attention_mask = self.build_sliding_mask(S)
            am = self._to_ttnn(attention_mask)
            if self._is_distributed and int(am.shape[-1]) != int(am.shape[-2]):
                attention_mask = self._mesh_replicate_full_attention_mask(attention_mask, S)

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

        attn_output = ttnn.reshape(self._to_raw_ttnn(attn_output), (B, S, self.hidden_size))
        output = self.o_proj(attn_output)

        return output, None


class TTNNQwenAudioAttention(TTNNModule):
    """TTNN implementation of Qwen3AudioAttentionOptimized (audio attention in the audio tower)."""

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
        new_attn.out_proj = TTNNLinearIReplicatedWColSharded.from_torch(torch_attn.out_proj)

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

    @staticmethod
    def _last_hidden_dim(hidden_states) -> int:
        return int(hidden_states.shape[-1])

    def _leading_seq_len(self, hidden_states) -> int:
        shape = hidden_states.shape
        ed = self.embed_dim
        if len(shape) == 2:
            return int(shape[0])
        if len(shape) == 3:
            last = int(shape[-1])
            if ed is not None and last == ed:
                return int(shape[1])
            if int(shape[0]) == 1:
                return int(shape[1])
        return int(shape[0])

    def _to_torch_mesh_concat(self, tensor):
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        if isinstance(tensor, torch.Tensor) and not isinstance(tensor, TorchTTNNTensor):
            return tensor
        if isinstance(tensor, TorchTTNNTensor):
            return tensor.to_torch
        mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0) if self._is_distributed else None
        return ttnn.to_torch(self._to_ttnn(tensor), mesh_composer=mesh_composer)

    def _cu_seqlens_to_torch_int64(self, cu_seqlens):
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        if cu_seqlens is None:
            return None
        if isinstance(cu_seqlens, TorchTTNNTensor):
            cu = self._to_torch_mesh_concat(cu_seqlens)
        elif isinstance(cu_seqlens, torch.Tensor):
            return cu_seqlens.detach().cpu().flatten().to(torch.int64)
        else:
            cu = self._to_torch_mesh_concat(cu_seqlens)
        if self._is_distributed:
            n = self.device.get_num_devices()
            if cu.dim() == 2 and int(cu.shape[0]) == n:
                cu = cu[0]
        return cu.flatten().to(torch.int64)

    @staticmethod
    def _cu_seqlens_allows_ttnn_from_flat(cu_flat: torch.Tensor | None, seq_len: int) -> bool:
        if cu_flat is None:
            return True
        if cu_flat.numel() < 2:
            return True
        n_seg = cu_flat.numel() - 1
        if n_seg > 1:
            return False
        return int(cu_flat[0].item()) == 0 and int(cu_flat[-1].item()) == seq_len

    @staticmethod
    def _varlen_additive_mask_torch(cu_flat: torch.Tensor, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype)
        cu = cu_flat.flatten().to(torch.int64)
        for i in range(1, cu.numel()):
            a = int(cu[i - 1].item())
            b = int(cu[i].item())
            if a < b and b <= seq_len:
                mask[..., a:b, a:b] = 0
        return mask

    def _additive_mask_torch_to_ttnn(self, mask_torch: torch.Tensor) -> ttnn.Tensor:
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self._is_distributed else None
        return ttnn.from_torch(
            mask_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
        cu_seqlens=None,
        **kwargs,
    ):
        cu_flat = self._cu_seqlens_to_torch_int64(cu_seqlens)
        if cu_flat is not None and cu_flat.numel() >= 2:
            seq_len = int(cu_flat[-1].item())
        else:
            seq_len = self._leading_seq_len(hidden_states)

        allows_single_segment = self._cu_seqlens_allows_ttnn_from_flat(cu_flat, seq_len)

        sdpa_attn_mask = attention_mask
        if cu_flat is not None and cu_flat.numel() >= 2 and not allows_single_segment:
            sdpa_attn_mask = self._additive_mask_torch_to_ttnn(
                self._varlen_additive_mask_torch(cu_flat, seq_len, torch.bfloat16)
            )

        expected_hidden = self.embed_dim
        if self._last_hidden_dim(hidden_states) != expected_hidden and self._is_distributed:
            hidden_states = self._maybe_all_gather(hidden_states)

        if len(hidden_states.shape) == 3 and int(hidden_states.shape[0]) == 1:
            hidden_states = ttnn.squeeze(hidden_states, 0)

        hidden_states = self._to_ttnn(hidden_states)

        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(
                hidden_states,
                ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        hidden_states = ttnn.unsqueeze(hidden_states, 0)
        if len(hidden_states.shape) == 3:
            hidden_states = ttnn.unsqueeze(hidden_states, 1)

        qkv_out = self.qkv_proj(hidden_states)
        if hasattr(qkv_out, "to_ttnn"):
            qkv = qkv_out.to_ttnn
        elif getattr(qkv_out, "ttnn_tensor", None) is not None:
            qkv = qkv_out.ttnn_tensor
        else:
            qkv = qkv_out

        qkv = ttnn.to_memory_config(qkv, ttnn.L1_MEMORY_CONFIG)

        query, key, value = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
        )

        ttnn.deallocate(qkv)

        attn_output = self.sdpa(
            self,
            query,
            key,
            value,
            sdpa_attn_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=False,
            transpose_output=False,
        )

        attn_output_tt = getattr(attn_output, "to_ttnn", attn_output)
        attn_output = ttnn.experimental.nlp_concat_heads(attn_output_tt)

        attn_output = ttnn.squeeze(attn_output, 0)

        attn_output = self.out_proj(attn_output)

        return attn_output
