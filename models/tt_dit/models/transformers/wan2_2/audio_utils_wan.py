# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""S2V audio conditioning modules.

Mirrors `wan/modules/s2v/audio_utils.py` and `wan/modules/s2v/auxi_blocks.py`
from the Wan-Video/Wan2.2 reference implementation:

  * ``CausalConv1d`` — Conv1d with left-only ("causal") temporal padding,
    expressed as ``ttnn.experimental.conv3d`` with ``kernel_size=(k, 1, 1)``.
  * ``MotionEncoder_tc`` — 3-stage causal conv stack that maps the
    learned-weighted wav2vec2 features to per-frame multi-token audio
    embeddings. Reduces the temporal rate by 4 (two ``stride=2`` convs).
  * ``CausalAudioEncoder`` — learned-weighted aggregation across wav2vec2
    hidden states (one weight per layer, init 0.01), followed by SiLU and
    ``MotionEncoder_tc``.
  * ``AudioInjector_WAN`` — ``ModuleList`` of cross-attention modules + two
    pre-norm LayerNorms each, indexed by transformer-block id. No ``forward``
    here; the DiT calls ``injector[id]``, ``injector_pre_norm_feat[id]``, etc.
    at the chosen layers.
  * ``AdaLayerNormZero`` — drop-in for ``diffusers.AdaLayerNorm`` used by the
    production ``adain_mode="attn_norm"`` path. SP-aware per-token application
    lives in ``transformer_wan_s2v._apply_adain_pre_norm``.
"""

from __future__ import annotations

import torch

import ttnn

from ....layers.linear import Linear
from ....layers.module import Module, ModuleList, Parameter
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils.conv3d import get_conv3d_config, register_conv3d_configs
from ....utils.tensor import local_device_to_torch
from .attention_wan import WanAttention

# Custom conv3d blockings for the MotionEncoder's CausalConv1d shapes. All are
# kernel-3 stride-{1,2} temporal convs over short sequences (T ≤ ~100 frames at
# the bucketed audio rate); spatial dims are H=W=1. Channel pairs match the
# WAN 2.2 5120-hidden / 4-token reference config (5120/4 = 1280, 5120/2 = 2560).
register_conv3d_configs(
    {
        # CausalAudioEncoder default: in=audio_dim=5120, head-expanded out=1280*num_heads=5120 (num_heads=4)
        (5120, 5120, (3, 1, 1)): (320, 64, 1, 1, 1),
        (1280, 2560, (3, 1, 1)): (320, 64, 1, 1, 1),
        (2560, 5120, (3, 1, 1)): (320, 64, 1, 1, 1),
        # wav2vec2-base feeds 768 (post-weighted-sum) -> conv1_local maps to 1280*4=5120
        (768, 5120, (3, 1, 1)): (256, 64, 1, 1, 1),
    }
)


def _layernorm_no_affine(x: ttnn.Tensor, eps: float = 1e-6) -> ttnn.Tensor:
    """LayerNorm over the last axis, no learned scale/bias. Matches HF's
    ``nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)``.

    Done manually because tt_dit's ``LayerNorm`` always allocates an affine
    weight tile when ``norm_elementwise_affine=True``; here we want the
    cheaper no-affine variant since the MotionEncoder does its own per-stage
    LayerNorms with no learned scale.
    """
    mean = ttnn.mean(x, dim=-1, keepdim=True)
    centered = ttnn.subtract(x, mean)
    var = ttnn.mean(ttnn.multiply(centered, centered), dim=-1, keepdim=True)
    return ttnn.multiply(centered, ttnn.rsqrt(ttnn.add(var, eps)))


class CausalConv1d(Module):
    """Conv1d with causal (left-only) temporal padding.

    HF: ``nn.Conv1d(chan_in, chan_out, kernel, stride)`` preceded by
    ``F.pad(x, (kernel-1, 0), mode='replicate')``.

    Here we re-express the op as ``ttnn.experimental.conv3d`` with
    ``kernel_size=(kernel, 1, 1)`` and ``padding=(0, 0, 0)``. The replicate-pad
    is applied on the host before upload — the audio encoder runs once per
    clip, so a small CPU pad before the device upload is negligible.

    Forward shape: ``[B, T_pre_pad, 1, 1, chan_in]`` ROW_MAJOR →
    ``[B, T_out, 1, 1, chan_out]`` TILE_LAYOUT (where
    ``T_out = (T_pre_pad - kernel) / stride + 1``).
    """

    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.mesh_device = mesh_device
        self.dtype = dtype

        self.conv_config = get_conv3d_config(
            chan_in,
            chan_out,
            (kernel_size, 1, 1),
            dtype,
            grid_size=mesh_device.compute_with_storage_grid_size(),
            h_factor=1,
            w_factor=1,
        )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Prepared weight layout is [d, out] where d = kT * kH * kW * in_aligned.
        # Allow the conv3d kernel to do its own prep on first call by keeping the
        # weight in rank-5 form on host.
        self.weight = Parameter(
            total_shape=[chan_out, chan_in, kernel_size, 1, 1],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            pad_value=0,
            dtype=dtype,
            on_host=True,
        )
        self.bias = Parameter(
            total_shape=[1, chan_out],
            device=mesh_device,
            pad_value=0,
            dtype=dtype,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "conv.weight" in state:
            w = state.pop("conv.weight")  # [out, in, k]
            assert w.shape == (self.chan_out, self.chan_in, self.kernel_size), w.shape
            state["weight"] = w.unsqueeze(-1).unsqueeze(-1).contiguous()  # [out, in, k, 1, 1]
        if "conv.bias" in state:
            state["bias"] = state.pop("conv.bias").reshape(1, -1)

    def forward(self, x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        out = ttnn.experimental.conv3d(
            input_tensor=x_BTHWC,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data,
            device=self.mesh_device,
            config=self.conv_config,
            output_channels=self.chan_out,
            kernel_size=(self.kernel_size, 1, 1),
            stride=(self.stride, 1, 1),
            padding=(0, 0, 0),
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.compute_kernel_config,
        )
        return ttnn.to_layout(out, ttnn.TILE_LAYOUT)


class MotionEncoder_tc(Module):
    """3-stage causal-conv encoder. Maps ``[B, T, in_dim]`` features to
    ``[B, T//4, num_heads + 1, hidden_dim]`` audio tokens.

    Matches the HF ``MotionEncoder_tc`` in
    ``wan/modules/s2v/auxi_blocks.py``. With ``need_global=True``, runs a
    parallel branch through ``conv1_global`` + ``final_linear`` (used by
    the production AdaIN path) and returns a ``(global, local)`` tuple.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        *,
        num_heads: int = 4,
        need_global: bool = False,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        assert hidden_dim % 4 == 0, "MotionEncoder hidden_dim must be divisible by 4"
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.need_global = need_global
        self.mesh_device = mesh_device
        self.dtype = dtype

        self.conv1_local = CausalConv1d(
            in_dim, hidden_dim // 4 * num_heads, kernel_size=3, stride=1, mesh_device=mesh_device, dtype=dtype
        )
        if need_global:
            self.conv1_global = CausalConv1d(
                in_dim, hidden_dim // 4, kernel_size=3, stride=1, mesh_device=mesh_device, dtype=dtype
            )
            # Final dense projection used only on the global branch.
            self.final_linear = Linear(hidden_dim, hidden_dim, bias=True, mesh_device=mesh_device)
        self.conv2 = CausalConv1d(
            hidden_dim // 4, hidden_dim // 2, kernel_size=3, stride=2, mesh_device=mesh_device, dtype=dtype
        )
        self.conv3 = CausalConv1d(
            hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, mesh_device=mesh_device, dtype=dtype
        )

        self.padding_tokens = Parameter(
            total_shape=[1, 1, 1, hidden_dim],
            device=mesh_device,
            dtype=dtype,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # padding_tokens is stored as-is, [1, 1, 1, hidden_dim].
        pass

    def _causal_pad_host(self, x_torch: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Replicate-pad the left of the temporal dim by kernel-1 frames.

        Mirrors HF's ``F.pad(x, (kernel-1, 0), mode='replicate')`` on the
        ``[B, C, T]`` layout, here applied to ``[B, T, C]``.
        """
        pad = kernel_size - 1
        if pad == 0:
            return x_torch
        # x_torch: [B, T, C]; replicate-pad on T by repeating the first frame.
        first = x_torch[:, :1, :].expand(-1, pad, -1)
        return torch.cat([first, x_torch], dim=1)

    def _conv_stage_BTC(
        self,
        x_tile_BTC: ttnn.Tensor,
        conv: CausalConv1d,
        B_eff: int,
        in_chan: int,
        out_chan: int,
    ) -> ttnn.Tensor:
        """One conv-stage: TILE ``[B_eff, T, in_chan]`` → causal-pad → conv →
        ``[B_eff, T_out, out_chan]`` → LN(no-affine) → SiLU → TILE.

        T_out follows ``⌈T / stride⌉`` for the conv's stride (recovered from
        the actual device tensor's element count).
        """
        x_torch = local_device_to_torch(x_tile_BTC).reshape(B_eff, -1, in_chan)
        x_torch_p = self._causal_pad_host(x_torch, kernel_size=3)
        x_5d = x_torch_p.reshape(B_eff, x_torch_p.shape[1], 1, 1, in_chan).contiguous()
        x_dev = ttnn.from_torch(x_5d, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = conv(x_dev)
        x_torch_full = local_device_to_torch(x)
        T_out = x_torch_full.numel() // (B_eff * out_chan)
        x_dev = ttnn.from_torch(
            x_torch_full.reshape(B_eff, T_out, out_chan),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        x_dev = _layernorm_no_affine(x_dev)
        return ttnn.silu(x_dev)

    def forward(self, x_torch: torch.Tensor) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run the 3-stage encoder.

        Input ``x_torch``: CPU ``[B, T, in_dim]`` (post-aggregation from
        :class:`CausalAudioEncoder`). The causal pad is host-side; convs +
        norms + SiLU + ``final_linear`` are on-device. Returns the local
        branch alone, or ``(global, local)`` when ``need_global=True``.
        """
        B, T, _ = x_torch.shape

        # --- Local branch ---
        # Stage 1: conv1_local maps in_dim → (hidden//4 * num_heads), then
        # split heads on host to [(B*n), T, hidden//4].
        x_torch_p = self._causal_pad_host(x_torch, kernel_size=3)
        x_5d = x_torch_p.reshape(B, x_torch_p.shape[1], 1, 1, self.in_dim).contiguous()
        x_dev = ttnn.from_torch(x_5d, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = self.conv1_local(x_dev)  # [B, T, 1, 1, (hidden/4) * num_heads]
        head_split = (
            local_device_to_torch(x)
            .reshape(B, T, self.num_heads, self.hidden_dim // 4)
            .permute(0, 2, 1, 3)
            .reshape(B * self.num_heads, T, self.hidden_dim // 4)
        )
        x_dev = ttnn.from_torch(head_split, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        x_dev = ttnn.silu(_layernorm_no_affine(x_dev))
        # Stages 2 & 3.
        B_local = B * self.num_heads
        x_dev = self._conv_stage_BTC(x_dev, self.conv2, B_local, self.hidden_dim // 4, self.hidden_dim // 2)
        x_dev = self._conv_stage_BTC(x_dev, self.conv3, B_local, self.hidden_dim // 2, self.hidden_dim)

        # Reshape [(B*n), T/4, hidden] → [B, T/4, n, hidden] and append the
        # learned padding token as an extra "head" along the token axis.
        local_out = local_device_to_torch(x_dev).reshape(B, self.num_heads, -1, self.hidden_dim)
        T4 = local_out.shape[2]
        local_out = local_out.permute(0, 2, 1, 3).contiguous()  # [B, T/4, n, hidden]
        pad_torch = (
            local_device_to_torch(self.padding_tokens.data)
            .reshape(1, 1, 1, self.hidden_dim)
            .expand(B, T4, 1, self.hidden_dim)
        )
        local_out = torch.cat([local_out, pad_torch.to(local_out.dtype)], dim=2)
        local_dev = ttnn.from_torch(local_out, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        if not self.need_global:
            return local_dev

        # --- Global branch ---
        # Mirrors the local branch but with conv1_global (in_dim → hidden//4,
        # no head split), then final_linear projects up to hidden. Output
        # shape ``[B, T/4, 1, hidden]`` (the n=1 dim matches the reference's
        # ``rearrange('(b n) t c -> b t n c', b=b)`` with B==(b*n)).
        x_5d = x_torch_p.reshape(B, x_torch_p.shape[1], 1, 1, self.in_dim).contiguous()
        x_dev = ttnn.from_torch(x_5d, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = self.conv1_global(x_dev)  # [B, T, 1, 1, hidden/4]
        x_torch = local_device_to_torch(x).reshape(B, T, self.hidden_dim // 4)
        x_dev = ttnn.from_torch(x_torch, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        x_dev = ttnn.silu(_layernorm_no_affine(x_dev))
        x_dev = self._conv_stage_BTC(x_dev, self.conv2, B, self.hidden_dim // 4, self.hidden_dim // 2)
        x_dev = self._conv_stage_BTC(x_dev, self.conv3, B, self.hidden_dim // 2, self.hidden_dim)
        x_dev = self.final_linear(x_dev)  # [B, T/4, hidden]
        global_dev = ttnn.unsqueeze(x_dev, 2)  # [B, T/4, 1, hidden]
        return global_dev, local_dev


class CausalAudioEncoder(Module):
    """Learned per-layer weighted sum of wav2vec2 hidden states followed by
    the causal MotionEncoder.

    Output shape depends on ``need_global``:

      * ``need_global=False``: ``[B, T_video, num_token+1, out_dim]``.
      * ``need_global=True``: ``(global, local)`` tuple where ``global`` and
        ``local`` are both ``[B, T_video, num_token+1, out_dim]``. The
        ``global`` tensor is what the production S2V model's AdaIN branch
        consumes; ``local`` feeds the cross-attention K/V slot.

    Args:
        dim: wav2vec2 hidden size (``768`` for wav2vec2-base; ``1024`` for
            wav2vec2-large-xlsr-53, which is the production audio encoder).
        num_layers: Number of wav2vec2 hidden states
            (``num_hidden_layers + 1``).
        out_dim: DiT hidden size; what the AudioInjector cross-attention will
            consume.
        num_token: Number of audio tokens per frame (typically ``4``).
        need_global: When ``True``, additionally produces the global audio
            embedding used by AdaIN. Set from
            ``WanS2VTransformer3DModel.enable_adain``.
    """

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
        )

        # Per-layer weights, initialized to 0.01 in the reference. Shape
        # [1, num_layers, 1, 1] so it broadcasts across [B, num_layers, dim, T].
        self.weights = Parameter(
            total_shape=[1, num_layers, 1, 1],
            device=mesh_device,
            dtype=ttnn.float32,
        )

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
        # The aggregation is cheap and is run once per clip; do it on host
        # so the MotionEncoder sees a single contiguous [B, T, dim] input.
        with torch.no_grad():
            weights_torch = local_device_to_torch(self.weights.data).reshape(1, self.num_layers, 1, 1)
            w = torch.nn.functional.silu(weights_torch.float())
            w_sum = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
            weighted = (features_torch.float() * w / w_sum).sum(dim=1)  # [B, dim, T]
            agg = weighted.permute(0, 2, 1).contiguous()  # [B, T, dim]

        return self.encoder(agg)


class AdaLayerNormZero(Module):
    """Drop-in for ``diffusers.models.attention.AdaLayerNorm`` (chunk_dim=1).

    Maps a global audio temb (``[bsz_t, adain_dim]``) to per-channel
    ``(scale, shift)`` via SiLU + Linear, applies LayerNorm-no-affine to the
    spatial input, then broadcasts the scale/shift over the patch-token axis:

        temb = linear(silu(temb))            # [bsz_t, 2 * dim]
        shift, scale = temb.chunk(2, dim=1)  # each [bsz_t, dim]
        x = norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]

    Matches the reference S2V's ``injector_adain_layers[i]`` config
    (``output_dim=dim*2, embedding_dim=adain_dim, chunk_dim=1``).
    """

    def __init__(
        self,
        *,
        dim: int,
        adain_dim: int,
        mesh_device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.adain_dim = adain_dim
        self.mesh_device = mesh_device
        # HF AdaLayerNorm uses a plain Linear (replicated across mesh — small
        # weight, single call per layer per step).
        self.linear = Linear(adain_dim, dim * 2, bias=True, mesh_device=mesh_device)

    def forward(self, x: ttnn.Tensor, temb: ttnn.Tensor) -> ttnn.Tensor:
        """Apply AdaIN modulation.

        Args:
            x: ``[bsz_t, N, dim]`` spatial Q tokens for one frame.
            temb: ``[bsz_t, adain_dim]`` global audio embedding for the
                corresponding frame.

        Returns:
            ``[bsz_t, N, dim]`` modulated tokens.
        """
        temb = ttnn.silu(temb)
        temb = self.linear(temb)  # [bsz_t, 2*dim]
        shift, scale = ttnn.chunk(temb, 2, dim=-1)  # each [bsz_t, dim]
        # Broadcast over N: unsqueeze shift/scale on axis 1.
        shift = ttnn.unsqueeze(shift, 1)  # [bsz_t, 1, dim]
        scale = ttnn.unsqueeze(scale, 1)
        x_normed = _layernorm_no_affine(x, eps=1e-5)
        return ttnn.add(ttnn.multiply(x_normed, ttnn.add(scale, 1.0)), shift)


class AudioInjector_WAN(Module):
    """Per-DiT-layer audio cross-attention injectors.

    Constructs an injector slot for each transformer block index listed in
    ``inject_layer``. Each slot has:

      * ``injector_pre_norm_feat[i]`` — LayerNorm (no affine) over spatial Q.
      * ``injector_pre_norm_vec[i]`` — LayerNorm (no affine) over audio K/V.
        Unused in v1 (the audio embeddings are pre-normalized by
        ``MotionEncoder_tc``) but kept for parity with the HF state dict.
      * ``injector[i]`` — ``WanAttention(is_self=False, qk_norm=True)``: standard
        cross-attention with the same num_heads as the DiT.

    When ``enable_adain=True`` (production Wan2.2-S2V-14B config), an
    additional ``injector_adain_layers[i]`` slot holds an :class:`AdaLayerNormZero`
    that modulates the spatial pre-norm with the global audio embedding.

    No ``forward`` is provided; the DiT's ``after_transformer_block`` calls
    these submodules directly at the chosen layers.
    """

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

        # HF uses elementwise_affine=False here, so we keep these as bookkeeping
        # in the state dict only — the actual norm is done on the fly via
        # `_layernorm_no_affine`. We still declare them so HF state-dict keys
        # (which have no weights for these LayerNorms) load cleanly without
        # tripping the strict loader.
        self.injector_pre_norm_feat = ModuleList()  # empty: no parameters
        self.injector_pre_norm_vec = ModuleList()

        if enable_adain:
            adain_dim = adain_dim or dim
            self.injector_adain_layers = ModuleList(
                AdaLayerNormZero(dim=dim, adain_dim=adain_dim, mesh_device=mesh_device) for _ in range(n_inject)
            )
        else:
            self.injector_adain_layers = ModuleList()

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # The HF pre-norm LayerNorms have no weights, so any leftover keys here
        # would be unexpected — defensively drop them if encountered.
        for k in list(state):
            if k.startswith("injector_pre_norm_feat.") or k.startswith("injector_pre_norm_vec."):
                state.pop(k)

    def forward(self, *args, **kwargs):
        msg = "AudioInjector_WAN has no top-level forward; index into .injector / .injector_adain_layers"
        raise NotImplementedError(msg)


def apply_pre_norm_feat(spatial_BNC: ttnn.Tensor) -> ttnn.Tensor:
    """Pre-norm helper for the injector's spatial Q input.

    Mirrors HF: ``LayerNorm(dim, elementwise_affine=False, eps=1e-6)`` on the
    last (channel) axis.
    """
    return _layernorm_no_affine(spatial_BNC, eps=1e-6)
