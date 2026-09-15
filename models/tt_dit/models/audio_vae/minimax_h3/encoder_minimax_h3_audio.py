# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 audio VAE encode: a DAC waveform encoder plus a causal-attention projection.

``(B, 1, samples)`` at 32 kHz to a 40 Hz, 32-channel posterior -- a hop of
``prod(encoder_rates) = 800``. Mono; stereo is batch 2.

Structure mirrors the reference one-for-one so the converted checkpoint loads with no
fixups. Everything nests under ``block`` ModuleLists because the reference uses
``nn.Sequential``, which is why the checkpoint keys read
``encoder.block.1.block.0.block.1.weight``.

Reused rather than rewritten, all from ``layers/audio_ops.py``:

* ``Snake`` with ``alpha_logscale=False`` -- the DAC activation stores ``alpha`` linearly,
  unlike the decoder's ``SnakeBeta`` which is log-scale;
* ``DilatedConv1d`` for the residual units' k=7 convs, whose ``eff_k // 2`` same-padding
  already equals the reference's ``((7-1)*d)//2``;
* ``_AlignedOutConv1d`` everywhere else, per its own docstring: a non-32-multiple channel
  count reaching ``conv3d`` produces a buffer whose page size does not divide its length.

The five strided channel-doubling convs pair ``kernel = 2 * stride`` with
``padding = ceil(stride / 2)``, passed through ``Conv1dViaConv3d``'s optional ``padding``
argument: the derived default ``eff_k // 2`` gives ``L/stride + 1`` instead of ``L/stride``.
The default stands, so LTX call sites are unaffected.

The reference's residual-unit shortcut crop is a no-op here -- the dilated conv's padding is
exact-same, so the crop width is zero -- but the lengths are asserted rather than assumed.
"""

from __future__ import annotations

import math

import torch

import ttnn

from ....layers.audio_ops import DEFAULT_MAX_C_IN_BLOCK, Snake, _AlignedOutConv1d
from ....layers.module import Module, ModuleList
from ....layers.normalization import LayerNorm
from ....parallel.config import ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.tensor import local_device_to_torch
from ..vocoder_ltx import DilatedConv1d
from .blockings_minimax_h3_audio import register_h3_audio_blockings


def _snake_row_major(snake: Snake, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
    """Apply ``Snake`` and hand back ROW_MAJOR, which is what the next conv requires.

    LTX only ever uses ``Snake``/``SnakeBeta`` inside ``Activation1d``, whose resamplers own
    the layout. H3's DAC encoder applies a **bare** Snake between convolutions, and the
    elementwise chain inside it returns TILE, so ``Conv1dViaConv3d``'s ROW_MAJOR assertion
    fires without this.
    """
    return ttnn.to_layout(snake(x_BTC), ttnn.ROW_MAJOR_LAYOUT)


class MiniMaxH3AudioResidualUnit(Module):
    """``Snake -> dilated Conv1d(k=7) -> Snake -> Conv1d(k=1)``, plus a residual add."""

    def __init__(
        self,
        dim: int,
        dilation: int,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
        split_mode: str = "off",
    ) -> None:
        super().__init__()
        shared = dict(mesh_device=mesh_device, dtype=dtype, parallel_config=parallel_config, ccl_manager=ccl_manager)
        # Conv-only levers: Snake takes neither, so they ride a separate dict from `shared`.
        levers = dict(split_mode=split_mode)
        self.block = ModuleList(
            [
                Snake(dim, alpha_logscale=False, **{k: v for k, v in shared.items() if k != "ccl_manager"}),
                DilatedConv1d(dim, dim, kernel_size=7, dilation=dilation, **shared, **levers),
                Snake(dim, alpha_logscale=False, **{k: v for k, v in shared.items() if k != "ccl_manager"}),
                _AlignedOutConv1d(dim, dim, kernel_size=1, **shared, **levers),
            ]
        )

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        residual = x_BTC
        for layer in self.block:
            x_BTC = _snake_row_major(layer, x_BTC) if isinstance(layer, Snake) else layer(x_BTC)
        assert x_BTC.shape[1] == residual.shape[1], (
            f"residual unit changed T from {residual.shape[1]} to {x_BTC.shape[1]}; "
            "the reference's centre crop is only a no-op when the dilated conv is exact-same"
        )
        return ttnn.add(residual, x_BTC)


class MiniMaxH3AudioEncoderBlock(Module):
    """Three residual units at dilations 1/3/9, then a strided channel-doubling conv."""

    def __init__(
        self,
        dim: int,
        stride: int,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
        split_mode: str = "off",
    ) -> None:
        super().__init__()
        shared = dict(mesh_device=mesh_device, dtype=dtype, parallel_config=parallel_config, ccl_manager=ccl_manager)
        levers = dict(split_mode=split_mode)
        inner = dim // 2
        self.stride = stride
        self.block = ModuleList(
            [
                MiniMaxH3AudioResidualUnit(inner, 1, **shared, **levers),
                MiniMaxH3AudioResidualUnit(inner, 3, **shared, **levers),
                MiniMaxH3AudioResidualUnit(inner, 9, **shared, **levers),
                Snake(inner, alpha_logscale=False, **{k: v for k, v in shared.items() if k != "ccl_manager"}),
                # kernel = 2 * stride with padding = ceil(stride / 2) is what makes the output
                # exactly L / stride; the class's default eff_k // 2 would give one extra frame.
                _AlignedOutConv1d(
                    inner,
                    dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                    **shared,
                    **levers,
                ),
            ]
        )

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        expected_out = x_BTC.shape[1] // self.stride
        for layer in self.block:
            x_BTC = _snake_row_major(layer, x_BTC) if isinstance(layer, Snake) else layer(x_BTC)
        assert (
            x_BTC.shape[1] == expected_out
        ), f"strided conv produced T={x_BTC.shape[1]}, expected {expected_out} -- check the padding override"
        return x_BTC


class MiniMaxH3AudioDACEncoder(Module):
    """``conv(1 -> encoder_dim, k7) -> 5 encoder blocks -> Snake -> conv(k3)``."""

    def __init__(
        self,
        *,
        encoder_dim: int = 64,
        encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5),
        latent_dim: int = 2048,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
        split_mode: str = "off",
    ) -> None:
        super().__init__()
        shared = dict(mesh_device=mesh_device, dtype=dtype, parallel_config=parallel_config, ccl_manager=ccl_manager)
        no_ccl = {k: v for k, v in shared.items() if k != "ccl_manager"}
        levers = dict(split_mode=split_mode)

        layers: list[Module] = [_AlignedOutConv1d(1, encoder_dim, kernel_size=7, **shared, **levers)]
        dim = encoder_dim
        for stride in encoder_rates:
            dim *= 2
            layers.append(MiniMaxH3AudioEncoderBlock(dim, stride, **shared, **levers))
        layers.append(Snake(dim, alpha_logscale=False, **no_ccl))
        layers.append(_AlignedOutConv1d(dim, latent_dim, kernel_size=3, **shared, **levers))
        self.block = ModuleList(layers)
        self.hop_length = math.prod(encoder_rates)

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        for layer in self.block:
            x_BTC = _snake_row_major(layer, x_BTC) if isinstance(layer, Snake) else layer(x_BTC)
        return x_BTC


class MiniMaxH3AudioCausalAttention(Module):
    """Causal attention that narrows ``in_dim`` to ``out_dim`` by pooling, not concatenating.

    The heads are **mean-pooled away** rather than concatenated, and the surviving head
    dimension is average-pooled from ``in_dim // num_heads`` down to ``out_dim``. Since
    ``256 % 32 == 0``, that pool is exactly a mean over contiguous groups of 8, so it is a
    reshape and a mean rather than anything adaptive.

    The checkpoint stores one fused biasless ``qkv`` plus separate ``q_bias``/``v_bias`` and
    a frozen-zero ``zero_k_bias``; the converter folds all three into ``qkv.bias``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        from ....layers.linear import Linear

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        assert (
            self.head_dim % out_dim == 0
        ), f"head_dim {self.head_dim} must be a whole multiple of out_dim {out_dim} for the pool to be a plain mean"
        self.pool_group = self.head_dim // out_dim
        self.mesh_device = mesh_device
        self.dtype = dtype

        self.qkv = Linear(in_dim, 3 * in_dim, bias=True, mesh_device=mesh_device, dtype=dtype)
        self.proj = Linear(out_dim, out_dim, bias=True, mesh_device=mesh_device, dtype=dtype)

    def forward(self, x_BSC: ttnn.Tensor) -> ttnn.Tensor:
        batch, seq_len, _ = x_BSC.shape
        # The qkv projection is 2048 -> 6144 over the whole latent sequence, and at a 10 s
        # clip (405 frames, batch 2 for stereo) the default matmul blocking overshoots L1
        # (1979264 B against 1572864 B). A smaller block trades a little throughput for a
        # working shape; the real fix is a tuned config, which is a performance-pass job.
        qkv = self.qkv(x_BSC, default_block_size=(4, 4, 4))
        query, key, value = (
            ttnn.permute(ttnn.reshape(part, (batch, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
            for part in ttnn.chunk(qkv, 3, dim=-1)
        )
        # SDPA is bf16-only (`sdpa_device_operation.cpp:43`), so this one op is a bf16 island
        # inside the otherwise-fp32 audio path -- the same shape of compromise the visual
        # encoder makes for GroupNorm.
        query, key, value = (
            ttnn.typecast(t, ttnn.bfloat16) if t.get_dtype() != ttnn.bfloat16 else t for t in (query, key, value)
        )
        attended = ttnn.transformer.scaled_dot_product_attention(query, key, value, is_causal=True)
        if attended.get_dtype() != self.dtype:
            attended = ttnn.typecast(attended, self.dtype)

        # (B, heads, S, head_dim) -> mean over heads -> (B, S, head_dim)
        pooled = ttnn.mean(ttnn.permute(attended, (0, 2, 1, 3)), dim=2)
        pooled = ttnn.reshape(pooled, (batch, seq_len, self.out_dim, self.pool_group))
        pooled = ttnn.mean(pooled, dim=3)
        return self.proj(pooled)


class MiniMaxH3AudioGeGluMlp(Module):
    """Pre-norm GeGLU: ``w2(gelu_tanh(w0(norm(x))) * w1(norm(x)))``."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        from ....layers.linear import Linear

        self.norm = LayerNorm(in_features, norm_eps=1e-5, bias=True, mesh_device=mesh_device)
        self.w0 = Linear(in_features, hidden_features, bias=True, mesh_device=mesh_device, dtype=dtype)
        self.w1 = Linear(in_features, hidden_features, bias=True, mesh_device=mesh_device, dtype=dtype)
        self.w2 = Linear(hidden_features, in_features, bias=True, mesh_device=mesh_device, dtype=dtype)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        normed = self.norm(x)
        # nn.GELU(approximate="tanh")
        gated = ttnn.gelu(self.w0(normed), fast_and_approximate_mode=True)
        return self.w2(ttnn.mul(gated, self.w1(normed)))


class MiniMaxH3AudioAttnProjection(Module):
    """``pre_block``: ``proj(norm3(x)) + attn(norm1(x))``, then ``+ mlp(norm2(.))``.

    Note the residual is a **projected** bypass, not ``x`` itself -- the block changes width
    from ``latent_dim`` to ``latent_channels`` inside the residual, which is unusual and
    easy to get wrong.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        *,
        mlp_ratio: int = 2,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        from ....layers.linear import Linear

        self.norm1 = LayerNorm(in_dim, norm_eps=1e-5, bias=True, mesh_device=mesh_device)
        self.attn = MiniMaxH3AudioCausalAttention(in_dim, out_dim, num_heads, mesh_device=mesh_device, dtype=dtype)
        self.proj = Linear(in_dim, out_dim, bias=True, mesh_device=mesh_device, dtype=dtype)
        self.norm3 = LayerNorm(in_dim, norm_eps=1e-5, bias=True, mesh_device=mesh_device)
        self.norm2 = LayerNorm(out_dim, norm_eps=1e-5, bias=True, mesh_device=mesh_device)
        self.mlp = MiniMaxH3AudioGeGluMlp(out_dim, out_dim * mlp_ratio, mesh_device=mesh_device, dtype=dtype)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        hidden = ttnn.add(self.proj(self.norm3(x)), self.attn(self.norm1(x)))
        return ttnn.add(hidden, self.mlp(self.norm2(hidden)))


class MiniMaxH3AudioEncoder(Module):
    """Waveform to posterior: DAC trunk, ``pre_block``, then ``mean_proj`` / ``logs_proj``.

    Returns ``(mean, logs)`` as torch tensors. Sampling stays on host, matching how the
    visual encoder leaves its posterior draw to the caller.
    """

    def __init__(
        self,
        *,
        encoder_dim: int = 64,
        encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5),
        latent_dim: int = 2048,
        latent_channels: int = 32,
        num_attention_heads: int = 8,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
        split_mode: str = "full",
        max_c_in_block: int = DEFAULT_MAX_C_IN_BLOCK,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.latent_channels = latent_channels
        self.hop_length = math.prod(encoder_rates)

        # The precision levers default to accurate, same rationale as the decoder. H3-only: LTX
        # constructs the same conv classes with its own fast defaults. Kept as attributes so the
        # pipeline's device-weight cache key (`weights_variant`) reads the exact values this module
        # was built with.
        self.split_mode = split_mode
        self.max_c_in_block = max_c_in_block

        # Every H3 audio conv shape misses _FP32_BLOCKINGS; seed stubs before any conv is built.
        register_h3_audio_blockings(max_c_in_block=max_c_in_block)

        self.encoder = MiniMaxH3AudioDACEncoder(
            encoder_dim=encoder_dim,
            encoder_rates=encoder_rates,
            latent_dim=latent_dim,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            split_mode=split_mode,
        )
        self.pre_block = MiniMaxH3AudioAttnProjection(
            latent_dim, latent_channels, num_attention_heads, mesh_device=mesh_device, dtype=dtype
        )
        self.mean_proj = _AlignedOutConv1d(
            latent_channels,
            latent_channels,
            kernel_size=1,
            mesh_device=mesh_device,
            dtype=dtype,
            split_mode=split_mode,
        )
        self.logs_proj = _AlignedOutConv1d(
            latent_channels,
            latent_channels,
            kernel_size=1,
            mesh_device=mesh_device,
            dtype=dtype,
            split_mode=split_mode,
        )

    def forward(self, waveform_BCT: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, 1, samples)`` torch in, ``(mean, logs)`` each ``(B, 32, samples/800)`` torch."""
        _, channels, num_samples = waveform_BCT.shape
        assert channels == 1, f"the audio VAE is mono; stereo is batch 2. Got {channels} channels"
        assert (
            num_samples % self.hop_length == 0
        ), f"{num_samples} samples is not a whole number of {self.hop_length}-sample hops"

        x = waveform_BCT.transpose(1, 2).float().contiguous()  # (B, T, 1)
        x_device = ttnn.from_torch(x, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.dtype)

        trunk = self.encoder(x_device)
        # pre_block is a transformer block, so it wants TILE; the convs want ROW_MAJOR.
        projected = self.pre_block(ttnn.to_layout(trunk, ttnn.TILE_LAYOUT))
        projected = ttnn.to_layout(projected, ttnn.ROW_MAJOR_LAYOUT)

        # The upload replicates and nothing here is fractured, so every device holds the same
        # result: read back one. A bare ``ttnn.to_torch`` asserts ``buffers.size() == 1`` and so
        # only works on a single-device mesh. Same fix as ``MiniMaxH3AudioDecoder.__call__``.
        def read(tensor: ttnn.Tensor) -> torch.Tensor:
            # See `MiniMaxH3AudioDecoder.__call__`: a storage slice keeps the parent's distribution
            # metadata and the converter rejects it on a multi-host mesh. The helper reads a shard this
            # host owns instead, which for a replicated tensor is the whole answer.
            return local_device_to_torch(tensor).float()

        mean = read(self.mean_proj(projected))
        logs = read(self.logs_proj(projected))
        return mean.transpose(1, 2).contiguous(), logs.transpose(1, 2).contiguous()
