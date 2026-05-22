# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""S2V audio conditioning: CausalAudioEncoder + AdaLayerNormZero + AudioInjector_WAN.
Mirrors ``wan/modules/s2v/audio_utils.py`` (the underlying CausalConv1d /
MotionEncoder_tc live in ``auxi_blocks.py``, matching the reference layout)."""

from __future__ import annotations

import torch

import ttnn

from .....layers.linear import ColParallelLinear, prepare_chunked_linear_output
from .....layers.module import Module, ModuleList, Parameter
from .....parallel.config import DiTParallelConfig
from .....parallel.manager import CCLManager
from .....utils.tensor import local_device_to_torch
from ..attention_wan import WanAttention
from .auxi_blocks import MotionEncoder_tc


class CausalAudioEncoder(Module):
    """Weighted-sum of wav2vec2 hidden states + MotionEncoder_tc."""

    def __init__(
        self,
        *,
        dim: int,
        num_layers: int,
        out_dim: int,
        num_token: int = 4,
        need_global: bool = False,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        tp_mesh_axis: int | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.num_token = num_token
        self.need_global = need_global
        self.mesh_device = mesh_device
        self.dtype = dtype

        self.encoder = MotionEncoder_tc(
            in_dim=dim,
            hidden_dim=out_dim,
            num_heads=num_token,
            need_global=need_global,
            mesh_device=mesh_device,
            dtype=dtype,
            tp_mesh_axis=tp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        # Per-layer weights, initialized to 0.01 in the reference. Shape
        # [1, num_layers, 1, 1] so it broadcasts across [B, num_layers, dim, T].
        self.weights = Parameter(
            total_shape=[1, num_layers, 1, 1],
            device=mesh_device,
            dtype=ttnn.float32,
        )
        # Host cache of the silu-normalized weights — the Parameter is a
        # static 25-element learned table, so the silu/sum/normalize is the
        # same every clip. Materialized on first forward (after state-dict
        # load) and reused.
        self._cached_w_normalized: torch.Tensor | None = None

    def forward(self, features_torch: torch.Tensor) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run the learned weighted aggregation + MotionEncoder on host then
        upload at the boundaries.

        Args:
            features_torch: ``[B, num_layers, dim, T_video]`` float tensor with
                the wav2vec2 hidden states stacked along dim 1 and time-aligned
                to the target video frame rate.

        Returns:
            Either a single ``[B, T_video // 4, num_token + 1, out_dim]`` tensor
            (``need_global=False``) or a ``(global, local)`` tuple of two such
            tensors (``need_global=True``).
        """
        if self._cached_w_normalized is None:
            with torch.no_grad():
                weights_torch = local_device_to_torch(self.weights.data).reshape(1, self.num_layers, 1, 1)
                w = torch.nn.functional.silu(weights_torch.float())
                w_sum = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
                self._cached_w_normalized = (w / w_sum).contiguous()
        with torch.no_grad():
            weighted = (features_torch.float() * self._cached_w_normalized).sum(dim=1)  # [B, dim, T]
            agg = weighted.permute(0, 2, 1).contiguous()  # [B, T, dim]

        return self.encoder(agg)


class AdaLayerNormZero(Module):
    """ColParallel-sharded ``Linear(adain_dim, 2*dim)`` for AdaIN modulation."""

    def __init__(
        self,
        *,
        dim: int,
        adain_dim: int,
        mesh_device: ttnn.MeshDevice,
        tp_mesh_axis: int,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.adain_dim = adain_dim
        self.mesh_device = mesh_device
        self.tp_mesh_axis = tp_mesh_axis
        self.tp_factor = mesh_device.shape[tp_mesh_axis]
        self.linear = ColParallelLinear(
            adain_dim,
            dim * 2,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=tp_mesh_axis,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Pre-interleave proj rows so each chip's local chunk(2, -1) yields
        # (local_shift, local_scale) instead of mixing them across chips.
        prepare_chunked_linear_output(
            state,
            prefix="linear",
            device_count=self.tp_factor,
            chunks=2,
        )

    def forward(self, *_args, **_kwargs):
        raise NotImplementedError("AdaLayerNormZero is a parameter holder; call .linear directly")


class AudioInjector_WAN(Module):
    """Per-DiT-layer audio cross-attention slots + optional AdaIN modulators."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        inject_layers: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 27),
        enable_adain: bool = False,
        adain_dim: int | None = None,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.inject_layers = tuple(inject_layers)
        self.enable_adain = enable_adain
        self.injected_block_id = {layer_idx: i for i, layer_idx in enumerate(self.inject_layers)}
        n_inject = len(self.inject_layers)

        self.injector = ModuleList(
            WanAttention(
                dim=dim,
                num_heads=num_heads,
                qk_norm=True,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                is_fsdp=is_fsdp,
                is_self=False,
            )
            for _ in range(n_inject)
        )

        # The reference's ``injector_pre_norm_feat`` / ``injector_pre_norm_vec``
        # are ``nn.LayerNorm(elementwise_affine=False)`` and contribute no
        # parameters to the state dict. The actual spatial pre-norm runs via
        # the surrounding block's ``norm1`` (a no-affine DistributedLayerNorm)
        # inside ``WanS2VTransformer3DModel.after_transformer_block``.
        self.injector_pre_norm_feat = ModuleList()
        self.injector_pre_norm_vec = ModuleList()

        if enable_adain:
            adain_dim = adain_dim or dim
            tp_mesh_axis = parallel_config.tensor_parallel.mesh_axis
            self.injector_adain_layers = ModuleList(
                AdaLayerNormZero(dim=dim, adain_dim=adain_dim, mesh_device=mesh_device, tp_mesh_axis=tp_mesh_axis)
                for _ in range(n_inject)
            )
        else:
            self.injector_adain_layers = ModuleList()

        # Per-injector slot for cached (k_BHNE, v_BHNE). Audio embedding is
        # constant across a clip's diffusion loop, so ``to_kv`` + ``norm_k`` +
        # head-split produce the same K/V on every step. The slot is a
        # caller-owned 2-element list filled lazily by ``WanAttention.forward``
        # on first call and reused after.
        self._audio_kv_cache: dict[int, list[ttnn.Tensor]] = {}

    def kv_cache_slot(self, audio_attn_id: int) -> list[ttnn.Tensor]:
        return self._audio_kv_cache.setdefault(audio_attn_id, [])

    def invalidate_audio_kv_cache(self) -> None:
        for slot in self._audio_kv_cache.values():
            for t in slot:
                ttnn.deallocate(t)
        self._audio_kv_cache = {}

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # HF pre-norm LayerNorms have no weights; drop any keys that slip through.
        for k in list(state):
            if k.startswith("injector_pre_norm_feat.") or k.startswith("injector_pre_norm_vec."):
                state.pop(k)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "AudioInjector_WAN has no top-level forward; index into .injector / .injector_adain_layers"
        )
