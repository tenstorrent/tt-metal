import torch
import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.linear import TTNNLinear, TTNNLinearIReplicatedWColSharded
from models.experimental.tt_symbiote.modules.attention import TTNNSDPAAttention


class TTNNQwenAudioAttentionOptimized(TTNNModule):
    """
    TTNN audio encoder attention (fused QKV + SDPA).

    Packed audio uses cumulative ``cu_seqlens`` with multiple segments. HF FlashAttention uses varlen
    directly; here we build the same **block-diagonal additive mask** as
    ``Qwen3OmniMoeAudioEncoder._prepare_attention_mask`` so **multi-segment** runs on TTNN SDPA
    without a PyTorch fallback.

    **Performance notes**
    - Reading ``cu_seqlens`` on mesh still uses ``ConcatMeshToTensor`` + ``to_torch`` once per forward.
    - SDPA chunk sizes are set in ``move_weights_to_device_impl``; larger chunks may help long ``seq``.
    """

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

        # Mesh-safe output projection (same pattern as thinker_attention o_proj).
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
        """All-gather sharded last dim so QKV sees full embed_dim (thinker_attention pattern)."""
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
        """Infer packed sequence length from activations (mesh layouts vary).

        Prefer ``[*, seq, embed_dim]`` when the last dim matches ``embed_dim`` so we do not
        mistake ``[num_devices, seq, embed_dim]`` (concat/replicate along mesh) for
        ``[seq, ?, ?]`` — that bug made ``seq_len`` equal to device count and broke
        ``cu_seqlens`` gating vs full-sequence TTNN attention (garbled multimodal/audio).
        """
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
        """Mesh-distributed ttnn → torch via ConcatMeshToTensor (required on multi-device)."""
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        # TorchTTNNTensor is a torch.Tensor subclass but may hold mesh-backed ttnn data; do not return .elem alone.
        if isinstance(tensor, torch.Tensor) and not isinstance(tensor, TorchTTNNTensor):
            return tensor
        if isinstance(tensor, TorchTTNNTensor):
            # Uses per-tensor mesh_composer (replicate vs concat), not a guessed ConcatMeshToTensor.
            return tensor.to_torch
        mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0) if self._is_distributed else None
        return ttnn.to_torch(self._to_ttnn(tensor), mesh_composer=mesh_composer)

    def _cu_seqlens_to_torch_int64(self, cu_seqlens):
        """cu_seqlens on mesh must use mesh_composer; take logical replica [0, L] (first device row)."""
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
            # Do not truncate longer cu_seqlens (multi-segment audio); an old ``numel == 2*n``
            # heuristic could drop segment boundaries and incorrectly allow the dense TTNN path.
        return cu.flatten().to(torch.int64)

    @staticmethod
    def _cu_seqlens_allows_ttnn_from_flat(cu_flat: torch.Tensor | None, seq_len: int) -> bool:
        """Whether full-sequence TTNN SDPA matches HF cumulative lengths (single segment [0, seq_len])."""
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
        """Block-diagonal allow mask (0 = attend, finfo.min = block), matching HF eager audio mask."""
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
        # HF packed audio: cumulative lengths end at the true packed seq length. On mesh,
        # hidden_states.shape can reflect shard/replica layout and mis-infer seq (e.g. vs device count);
        # trusting cu_flat[-1] matches Qwen3OmniMoeAudioAttention's seq_length from hidden_states.size().
        if cu_flat is not None and cu_flat.numel() >= 2:
            seq_len = int(cu_flat[-1].item())
        else:
            seq_len = self._leading_seq_len(hidden_states)

        allows_single_segment = self._cu_seqlens_allows_ttnn_from_flat(cu_flat, seq_len)

        # Multi-segment packed sequence: TTNN SDPA + HF-style block mask (no torch fallback).
        sdpa_attn_mask = attention_mask
        if cu_flat is not None and cu_flat.numel() >= 2 and not allows_single_segment:
            sdpa_attn_mask = self._additive_mask_torch_to_ttnn(
                self._varlen_additive_mask_torch(cu_flat, seq_len, torch.bfloat16)
            )

        # All-gather sharded activations so fused QKV sees full embed_dim on mesh.
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
            sdpa_attn_mask,
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
