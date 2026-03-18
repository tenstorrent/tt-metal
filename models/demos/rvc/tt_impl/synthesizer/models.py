# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch

import ttnn
from models.demos.rvc.tt_impl.conv1d import Conv1d
from models.demos.rvc.tt_impl.convtranspose1d import ConvTranspose1d
from models.demos.rvc.tt_impl.linear import Linear
from models.demos.rvc.tt_impl.synthesizer.attentions import FFN, MultiHeadAttention
from models.demos.rvc.tt_impl.synthesizer.modules import (
    LRELU_SLOPE,
    LayerNorm,
    ResBlock1,
    ResBlock2,
    ResidualCouplingLayer,
)


def ttnn_randn_fallback(shape, dtype, device):
    # Fallback random generator using PyTorch, since TTNN's random generation is not available in the current version.
    return ttnn.from_torch(
        torch.randn(shape, dtype=torch.float32),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )


def ttnn_cumsum_fallback(x: ttnn.Tensor, dim: int) -> ttnn.Tensor:
    # Fallback implementation of cumsum using to_host, torch.cumsum, and from_torch.
    x_torch = ttnn.to_torch(x)
    cumsum_torch = torch.cumsum(x_torch, dim=dim)
    cumsum = ttnn.from_torch(cumsum_torch, dtype=x.dtype, layout=x.layout, device=x.device())
    return cumsum


def _interpolate_1d(
    x: ttnn.Tensor,
    scale_factor: int | float,
    mode: str = "nearest",
) -> ttnn.Tensor:
    # 1D upsample for [N, L, C] via 2D NHWC upsample with height fixed to 1.
    if mode not in ("nearest", "linear"):
        raise ValueError(f"Unsupported 1D interpolate mode: {mode}")
    upsample_mode = "nearest" if mode == "nearest" else "bilinear"
    x_nhwc = ttnn.reshape(x, (x.shape[0], 1, x.shape[1], 1))
    y_nhwc = ttnn.upsample(
        x_nhwc,
        [1, scale_factor],
        mode=upsample_mode,
    )
    y = ttnn.reshape(y_nhwc, (y_nhwc.shape[0], y_nhwc.shape[2], y_nhwc.shape[3]))
    return y


def _flip_last_dim_ttnn(x: ttnn.Tensor) -> ttnn.Tensor:
    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    reverse_index = ttnn.arange(
        start=x.shape[-1] - 1,
        end=-1,
        step=-1,
        dtype=ttnn.int32,
        device=x.device(),
        layout=ttnn.TILE_LAYOUT,
    )
    reverse_index = ttnn.reshape(reverse_index, shape=(1,) * (len(x.shape) - 1) + (x.shape[-1],))
    reverse_index = ttnn.expand(reverse_index, tuple(x.shape))
    reverse_index = ttnn.typecast(reverse_index, ttnn.uint32)
    return ttnn.gather(x, dim=-1, index=reverse_index)


class Embedding:
    def __init__(self, device: ttnn.MeshDevice, num_embeddings: int, embedding_dim: int) -> None:
        self.device = device
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight: ttnn.Tensor | None = None

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        weight_key = f"{module_prefix}weight"
        self.weight = ttnn.from_torch(
            state_dict[weight_key].detach(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

    def __call__(self, indices: ttnn.Tensor) -> ttnn.Tensor:
        if self.weight is None:
            raise ValueError("Embedding state_dict are not loaded.")
        return ttnn.embedding(indices, self.weight, layout=ttnn.TILE_LAYOUT)


class Encoder:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        hidden_channels: int,
        filter_channels: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int = 1,
        window_size: int = 10,
    ) -> None:
        self.device = device
        self.num_layers = int(num_layers)
        self.attn_layers = [
            MultiHeadAttention(
                device=device,
                in_features=hidden_channels,
                out_features=hidden_channels,
                num_heads=num_heads,
                window_size=window_size,
            )
            for _ in range(self.num_layers)
        ]
        self.norm_layers_1 = [LayerNorm(device, hidden_channels) for _ in range(self.num_layers)]
        self.ffn_layers = [
            FFN(
                device=device,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                filter_channels=filter_channels,
                kernel_size=kernel_size,
            )
            for _ in range(self.num_layers)
        ]
        self.norm_layers_2 = [LayerNorm(device, hidden_channels) for _ in range(self.num_layers)]

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        for i in range(self.num_layers):
            self.attn_layers[i].load_state_dict(state_dict, module_prefix=f"{module_prefix}attn_layers.{i}.")
            self.norm_layers_1[i].load_state_dict(state_dict, module_prefix=f"{module_prefix}norm_layers_1.{i}.")
            self.ffn_layers[i].load_state_dict(state_dict, module_prefix=f"{module_prefix}ffn_layers.{i}.")
            self.norm_layers_2[i].load_state_dict(state_dict, module_prefix=f"{module_prefix}norm_layers_2.{i}.")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for i in range(self.num_layers):
            y = self.attn_layers[i](x, x)
            x = self.norm_layers_1[i](ttnn.add(x, y, output_tensor=x))
            y = self.ffn_layers[i](x)
            x = self.norm_layers_2[i](ttnn.add(x, y, output_tensor=x))
        return x


class TextEncoder:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        embedding_dims: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int,
        f0: bool = True,
    ) -> None:
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.emb_phone = Linear(device, embedding_dims, hidden_channels)
        self.use_f0 = f0
        self.emb_pitch = Embedding(device, 256, hidden_channels) if f0 else None
        self.encoder = Encoder(
            device=device,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
        )
        self.proj_linear = Linear(
            device=device,
            in_features=hidden_channels,
            out_features=out_channels * 2,
        )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        self.emb_phone.load_state_dict(state_dict, key="emb_phone", module_prefix=module_prefix)
        if self.use_f0 and self.emb_pitch is not None:
            self.emb_pitch.load_state_dict(state_dict, module_prefix=f"{module_prefix}emb_pitch.")
        self.encoder.load_state_dict(state_dict, module_prefix=f"{module_prefix}encoder.")
        proj_key = (
            "proj_linear"
            if (f"{module_prefix}proj_linear.weight" if module_prefix else "proj_linear.weight") in state_dict
            else "proj"
        )
        self.proj_linear.load_state_dict(state_dict, key=proj_key, module_prefix=module_prefix)

    def __call__(self, phone: ttnn.Tensor, pitch: ttnn.Tensor | None) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        x = self.emb_phone(phone)
        if self.use_f0 and pitch is not None and self.emb_pitch is not None:
            x = ttnn.add(x, self.emb_pitch(pitch), output_tensor=x)
        x = ttnn.multiply(x, math.sqrt(self.hidden_channels), output_tensor=x)
        x = ttnn.leaky_relu(x, negative_slope=0.1, output_tensor=x)
        x_e = self.encoder(x)
        stats = self.proj_linear(x_e)
        m, logs = ttnn.chunk(stats, chunks=2, dim=-1)
        return m, logs


class ResidualCouplingBlock:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        num_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        self.flows = [
            ResidualCouplingLayer(
                device,
                channels,
                hidden_channels,
                kernel_size,
                dilation_rate,
                num_layers,
                gin_channels=gin_channels,
            )
            for _ in range(num_flows)
        ]
        self.device = device

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        for i, flow in enumerate(self.flows):
            flow.load_state_dict(state_dict, module_prefix=f"{module_prefix}flows.{i}.")

    def __call__(self, x: ttnn.Tensor, g: ttnn.Tensor | None = None) -> ttnn.Tensor:
        for flow in self.flows:
            x = flow(_flip_last_dim_ttnn(x), g=g)
        return x


class Generator:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gin_channels: int = 0,
    ) -> None:
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = Conv1d(
            device=device,
            in_channels=initial_channel,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding="same",
        )

        self.ups: list[ConvTranspose1d] = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            self.ups.append(
                ConvTranspose1d(
                    device=device,
                    in_channels=upsample_initial_channel // (2**i),
                    out_channels=upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )

        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(resblock_cls(device, ch, k, tuple(d)))

        self.conv_post = Conv1d(
            device=device,
            in_channels=ch,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding="same",
            activation="tanh",
        )
        self.cond_linear = None
        if gin_channels != 0:
            self.cond_linear = Linear(
                device=device,
                in_features=gin_channels,
                out_features=upsample_initial_channel,
            )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        self.conv_pre.load_state_dict(state_dict, key="conv_pre", module_prefix=module_prefix)
        if self.cond_linear is not None:
            self.cond_linear.load_state_dict(state_dict, key="cond_linear", module_prefix=module_prefix)
        for i, up in enumerate(self.ups):
            up.load_state_dict(state_dict, key=f"ups.{i}", module_prefix=module_prefix)
        for i, rb in enumerate(self.resblocks):
            rb.load_state_dict(state_dict, module_prefix=f"{module_prefix}resblocks.{i}.")
        self.conv_post.load_state_dict(state_dict, key="conv_post", module_prefix=module_prefix)

    def __call__(self, x: ttnn.Tensor, g: ttnn.Tensor | None = None) -> ttnn.Tensor:
        x = self.conv_pre(x)
        if g is not None and self.cond_linear is not None:
            x = ttnn.add(x, self.cond_linear(g), output_tensor=x)

        for i in range(self.num_upsamples):
            x = ttnn.leaky_relu(x, negative_slope=LRELU_SLOPE, output_tensor=x)
            x = self.ups[i](x)
            xs = self.resblocks[i * self.num_kernels](x)
            for j in range(1, self.num_kernels):
                xs = ttnn.add(xs, self.resblocks[i * self.num_kernels + j](x), output_tensor=xs)
            x = ttnn.multiply(xs, 1.0 / self.num_kernels, output_tensor=xs)

        x = ttnn.leaky_relu(x, negative_slope=LRELU_SLOPE, output_tensor=x)
        x = self.conv_post(x)
        return x


class SineGen:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ) -> None:
        self.device = device
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def __call__(self, f0: ttnn.Tensor, upp: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        # f0: [B, T]
        # Upsample f0 to full resolution first using TTNN wrapper.
        f0_up = _interpolate_1d(f0, scale_factor=upp, mode="nearest")

        # Voiced/unvoiced mask.
        f0_up = ttnn.to_layout(f0_up, ttnn.TILE_LAYOUT)
        uv = ttnn.gt_(f0_up, self.voiced_threshold)

        # Expand for harmonics: [B, T*upp, H].
        harmonics = ttnn.arange(start=1, end=self.harmonic_num + 2, dtype=f0_up.dtype, device=self.device)
        f0_harm = f0_up * (harmonics / self.sampling_rate)

        # Accumulate phase and add random initial offset per harmonic.
        # phase = ttnn.cumsum(f0_harm, dim=1, out=f0_harm)
        # TODO: fallback is faster than native cumsum
        phase = ttnn_cumsum_fallback(f0_harm, dim=1)
        rand_ini = ttnn.rand((f0_up.shape[0], self.harmonic_num + 1), dtype=ttnn.bfloat16, device=self.device)
        phase = ttnn.add(phase, rand_ini, output_tensor=phase)
        phase = ttnn.multiply(phase, 2 * math.pi, output_tensor=phase)
        sine_waves = ttnn.multiply(ttnn.sin(phase, output_tensor=phase), self.sine_amp, output_tensor=phase)

        # Mix with noise based on voiced/unvoiced.
        noise_amp = uv * self.noise_std + ttnn.rsub(uv, 1) * self.sine_amp / 3
        noise_amp = ttnn.multiply(
            noise_amp,
            ttnn_randn_fallback(tuple(sine_waves.shape), dtype=ttnn.bfloat16, device=self.device),
            output_tensor=noise_amp,
        )
        return ttnn.add(sine_waves, noise_amp, output_tensor=sine_waves)


class SourceModuleHnNSF:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        sampling_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshod: float = 0,
    ) -> None:
        self.device = device
        self.l_sin_gen = SineGen(device, sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = Linear(
            device=device, in_features=harmonic_num + 1, out_features=1, dtype=ttnn.bfloat16, activation="tanh"
        )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        self.l_linear.load_state_dict(state_dict=state_dict, key="l_linear", module_prefix=module_prefix)

    def __call__(self, x: ttnn.Tensor, upp: int = 1) -> ttnn.Tensor:
        sine_wavs = self.l_sin_gen(x, upp)
        tt_linear = self.l_linear(sine_wavs)
        # this is needed since last dim is 1, padding due to tile layout would cause huge unnecessary memory usage
        tt_linear = ttnn.to_layout(tt_linear, ttnn.ROW_MAJOR_LAYOUT)
        return tt_linear


class GeneratorNSF:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gin_channels: int,
        sr: int,
    ) -> None:
        self.device = device
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(device=device, sampling_rate=sr, harmonic_num=0)
        self.upp = math.prod(upsample_rates)
        self.lrelu_slope = LRELU_SLOPE

        self.conv_pre = Conv1d(
            device=device,
            in_channels=initial_channel,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding="same",
        )

        self.ups: list[ConvTranspose1d] = []
        self.noise_convs: list[Conv1d | Linear] = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                ConvTranspose1d(
                    device=device,
                    in_channels=upsample_initial_channel // (2**i),
                    out_channels=c_cur,
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    Conv1d(
                        device=device,
                        in_channels=1,
                        out_channels=c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(
                    Linear(
                        device=device,
                        in_features=1,
                        out_features=c_cur,
                    )
                )

        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(resblock_cls(device, ch, k, tuple(d)))

        self.conv_post = Conv1d(
            device=device,
            in_channels=ch,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding="same",
            activation="tanh",
        )
        self.cond_linear = Linear(device=device, in_features=gin_channels, out_features=upsample_initial_channel)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        self.conv_pre.load_state_dict(state_dict, key="conv_pre", module_prefix=module_prefix)
        self.cond_linear.load_state_dict(state_dict, key="cond_linear", module_prefix=module_prefix)
        self.m_source.load_state_dict(state_dict, module_prefix=f"{module_prefix}m_source.")
        for i, up in enumerate(self.ups):
            up.load_state_dict(state_dict, key=f"ups.{i}", module_prefix=module_prefix)
        for i, nc in enumerate(self.noise_convs):
            nc.load_state_dict(state_dict, key=f"noise_convs.{i}", module_prefix=module_prefix)
        for i, rb in enumerate(self.resblocks):
            rb.load_state_dict(state_dict, module_prefix=f"{module_prefix}resblocks.{i}.")
        self.conv_post.load_state_dict(state_dict, key="conv_post", module_prefix=module_prefix)

    def __call__(self, x: ttnn.Tensor, f0: ttnn.Tensor, g: ttnn.Tensor | None = None) -> ttnn.Tensor:
        har_source_tt = self.m_source(f0, self.upp)
        x = self.conv_pre(x)
        if g is not None:
            x = ttnn.add(x, self.cond_linear(g), output_tensor=x)
        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs, strict=True)):
            x = ttnn.leaky_relu(x, negative_slope=self.lrelu_slope, output_tensor=x)
            x = ups(x)
            # the layout conversion happens inside noise_convs because doign it here causes oom for some reason
            # TODO: investigate the reasoning behind this
            x_source = noise_convs(har_source_tt)
            x = ttnn.add(x, x_source, output_tensor=x)
            xs = self.resblocks[i * self.num_kernels](x)
            for j in range(i * self.num_kernels + 1, (i + 1) * self.num_kernels):
                xs = ttnn.add(xs, self.resblocks[j](x), output_tensor=xs)
            x = ttnn.multiply(xs, 1.0 / self.num_kernels, output_tensor=xs)

        x = ttnn.leaky_relu(x, negative_slope=self.lrelu_slope, output_tensor=x)
        x = self.conv_post(x)
        return x


sr2sr = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


class SynthesizerTrnMsNSF:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        embedding_dims: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        spk_embed_dim: int,
        gin_channels: int,
        sr: int | str,
    ) -> None:
        if isinstance(sr, str):
            sr = sr2sr[sr]
        self.device = device
        self.enc_p = TextEncoder(
            device,
            embedding_dims,
            inter_channels,
            hidden_channels,
            filter_channels,
            num_heads,
            num_layers,
            kernel_size,
        )
        self.dec = GeneratorNSF(
            device,
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
        )
        self.flow = ResidualCouplingBlock(device, inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.emb_g = Embedding(device, spk_embed_dim, gin_channels)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        self.enc_p.load_state_dict(state_dict, module_prefix=f"{module_prefix}enc_p.")
        self.dec.load_state_dict(state_dict, module_prefix=f"{module_prefix}dec.")
        self.flow.load_state_dict(state_dict, module_prefix=f"{module_prefix}flow.")
        self.emb_g.load_state_dict(state_dict, module_prefix=f"{module_prefix}emb_g.")

    def __call__(
        self, phone: ttnn.Tensor, pitch: ttnn.Tensor, nsff0: ttnn.Tensor, speaker_id: ttnn.Tensor
    ) -> ttnn.Tensor:
        g = self.emb_g(speaker_id)
        g = ttnn.reshape(g, (g.shape[0], 1, g.shape[-1]))
        m_p, logs_p = self.enc_p(phone, pitch)
        z_p = (
            m_p
            + ttnn.exp(logs_p)
            * ttnn_randn_fallback(tuple(m_p.shape), dtype=ttnn.bfloat16, device=self.device)
            * 0.66666
        )
        z = self.flow(z_p, g=g)
        o = self.dec(z, nsff0, g=g)
        return o


class SynthesizerTrnMsNSF_nono:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        embedding_dims: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        spk_embed_dim: int,
        gin_channels: int,
        sr: int | None = None,
    ) -> None:
        self.device = device
        self.enc_p = TextEncoder(
            device,
            embedding_dims,
            inter_channels,
            hidden_channels,
            filter_channels,
            num_heads,
            num_layers,
            kernel_size,
            f0=False,
        )
        self.dec = Generator(
            device,
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(device, inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.emb_g = Embedding(device, spk_embed_dim, gin_channels)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        self.enc_p.load_state_dict(state_dict, module_prefix=f"{module_prefix}enc_p.")
        self.dec.load_state_dict(state_dict, module_prefix=f"{module_prefix}dec.")
        self.flow.load_state_dict(state_dict, module_prefix=f"{module_prefix}flow.")
        self.emb_g.load_state_dict(state_dict, module_prefix=f"{module_prefix}emb_g.")

    def __call__(self, phone: ttnn.Tensor, speaker_id: ttnn.Tensor) -> ttnn.Tensor:
        g = self.emb_g(speaker_id)
        g = ttnn.reshape(g, (g.shape[0], 1, g.shape[-1]))
        m_p, logs_p = self.enc_p(phone, None)
        z_p = (
            m_p
            + ttnn.exp(logs_p, output_tensor=logs_p)
            * ttnn_randn_fallback(tuple(m_p.shape), dtype=ttnn.bfloat16, device=self.device)
            * 0.66666
        )
        z = self.flow(z_p, g=g)
        out = self.dec(z, g=g)
        return out
