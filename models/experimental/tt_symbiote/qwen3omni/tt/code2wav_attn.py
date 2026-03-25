# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.linear import TTNNLinear, TTNNLinearIReplicatedWColSharded
from models.experimental.tt_symbiote.modules.rope import TTNNRotaryPositionEmbedding
from models.experimental.tt_symbiote.modules.attention import TTNNSDPAAttention


class TTNNQwen3OmniMoeCode2WavAttention(TTNNModule):
    """
    TTNN Code2Wav pre-transformer self-attention (sliding-window SDPA).

    Mesh compatibility matches ``TTNNQwen3OmniAttention`` / ``TTNNQwenAudioAttention``:
    all-gather sharded ``hidden_states`` and RoPE ``cos``/``sin`` on the last dim so QKV
    sees full ``hidden_size``, then keep intermediates as raw ttnn tensors. KV-cache uses
    the same torch round-trip as the thinker path for ``DynamicCache`` compatibility.
    """

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

        self.use_windowed_attention = False  # optional fast path; keep False for HF parity

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
        """Width-sharded masks must become full ``[1,1,S,S]`` bf16 on each device.

        ``all_gather_async`` on UINT8/bool fails (tilize unsupported). Merge shards on
        host via ``ConcatMeshToTensor``, convert packed/bool masks to additive ``0``/``-inf``,
        then ``from_torch`` with ``ReplicateTensorToMesh``.
        """
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
        expected_hidden = self.q_proj.in_features
        if hidden_states.shape[-1] != expected_hidden and self._is_distributed:
            hidden_states = self._maybe_all_gather(hidden_states)

        hs = self._to_ttnn(hidden_states)
        B, S, _ = hs.shape

        # --------------------------------------------------------
        # QKV PROJECTION
        # --------------------------------------------------------
        query = self._to_ttnn(self.q_proj(hidden_states))
        key = self._to_ttnn(self.k_proj(hidden_states))
        value = self._to_ttnn(self.v_proj(hidden_states))

        # reshape → [B, H, S, D]
        query = ttnn.reshape(query, (B, S, self.num_heads, self.head_dim))
        key = ttnn.reshape(key, (B, S, self.num_kv_heads, self.head_dim))
        value = ttnn.reshape(value, (B, S, self.num_kv_heads, self.head_dim))

        query = ttnn.permute(query, (0, 2, 1, 3))
        key = ttnn.permute(key, (0, 2, 1, 3))
        value = ttnn.permute(value, (0, 2, 1, 3))

        # --------------------------------------------------------
        # ROPE
        # --------------------------------------------------------
        cos, sin = position_embeddings
        cos = self._maybe_all_gather(cos)
        sin = self._maybe_all_gather(sin)
        query, key = self.rope(query, key, cos, sin)
        query = self._to_ttnn(query)
        key = self._to_ttnn(key)

        # --------------------------------------------------------
        # KV CACHE (torch round-trip — same as TTNNQwen3OmniAttention)
        # --------------------------------------------------------
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
        if self.use_windowed_attention and past_key_values is None and S % self.sliding_window == 0:
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
                attention_mask = self.build_sliding_mask(S)

            # On mesh, sliding mask may be width-sharded; SDPA needs full [1,1,S,S] bf16.
            # Do not use all_gather on uint8 shards (tilize/CCL dtype limits).
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

        # --------------------------------------------------------
        # OUTPUT PROJECTION (unwrap: SDPA can return TorchTTNNTensor)
        # --------------------------------------------------------
        attn_output = ttnn.reshape(self._to_raw_ttnn(attn_output), (B, S, self.hidden_size))
        output = self.o_proj(attn_output)

        return output, None
