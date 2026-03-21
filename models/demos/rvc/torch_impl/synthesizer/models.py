# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F

from models.demos.rvc.torch_impl.synthesizer import attentions, modules
from models.demos.rvc.torch_impl.utils import linear_channel_first


class TextEncoder(nn.Module):
    def __init__(
        self,
        embedding_dims,
        out_channels,
        hidden_channels,
        filter_channels,
        num_heads,
        num_layers,
        kernel_size,
        f0=True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.emb_phone = nn.Linear(embedding_dims, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)  # pitch 256
        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            num_heads,
            num_layers,
            kernel_size,
        )
        self.proj_linear = nn.Linear(hidden_channels, out_channels * 2)

    def forward(self, phone: torch.Tensor, pitch: torch.Tensor | None):
        if pitch is None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x = self.encoder(x)
        stats = linear_channel_first(x, self.proj_linear)

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()

        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    num_layers,
                    gin_channels=gin_channels,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor | None = None,
    ):
        for flow in self.flows:
            x = torch.flip(x, [1])
            x = flow.forward(x, g=g)
        return x


class Generator(nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, kernel_size=7, padding="same")
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            self.ups.append(
                ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, kernel_size=7, padding="same", bias=False)
        if gin_channels != 0:
            self.cond_linear = nn.Linear(gin_channels, upsample_initial_channel)

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + linear_channel_first(g, self.cond_linear)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = self.resblocks[i * self.num_kernels](x)
            for j in range(1, self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class SineGen(nn.Module):
    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def forward(self, f0: torch.Tensor, upp: int):
        # f0: (B, T)
        with torch.no_grad():
            # Upsample f0 to full resolution first
            f0_up = F.interpolate(f0[:, None], scale_factor=float(upp), mode="nearest").squeeze(1)  # (B, T*upp)

            # Voiced/unvoiced mask
            uv = (f0_up > self.voiced_threshold).float().unsqueeze(-1)  # (B, T*upp, 1)

            # Expand for harmonics: (B, T*upp, H)
            harmonics = torch.arange(1, self.harmonic_num + 2, device=f0.device, dtype=f0.dtype)
            f0_harm = f0_up.unsqueeze(-1) * harmonics  # (B, T*upp, H)

            # Accumulate phase, add random initial offset per harmonic
            phase = torch.cumsum(f0_harm / self.sampling_rate, dim=1)
            rand_ini = torch.rand(f0.shape[0], self.harmonic_num + 1, device=f0.device)
            rand_ini[:, 0] = 0  # keep fundamental phase at 0
            phase = phase + rand_ini.unsqueeze(1)

            sine_waves = torch.sin(2 * torch.pi * phase) * self.sine_amp

            # Mix with noise based on voiced/unvoiced
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            sine_waves = sine_waves * uv + noise_amp * torch.randn_like(sine_waves)

        return sine_waves


class SourceModuleHnNSF(nn.Module):
    """SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshod=0,
    ):
        super().__init__()

        self.sine_amp = sine_amp
        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, upp: int = 1):
        sine_wavs = self.l_sin_gen(x, upp)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge


class GeneratorNSF(nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels,
        sr,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.f0_upsamp = torch.nn.Upsample(scale_factor=math.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sampling_rate=sr, harmonic_num=0)
        self.noise_convs = nn.ModuleList()
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, kernel_size=7, padding="same")
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(nn.Linear(1, c_cur))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, kernel_size=7, padding="same", bias=False)

        if gin_channels != 0:
            self.cond_linear = nn.Linear(gin_channels, upsample_initial_channel)

        self.upp = math.prod(upsample_rates)

        self.lrelu_slope = modules.LRELU_SLOPE

        assert (
            len(self.resblocks) == self.num_kernels * self.num_upsamples
        ), "num of resblocks should be num_kernels * num_upsamples"

    def forward(self, x, f0, g: torch.Tensor | None = None):
        har_source = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)
        if g is not None:
            x = x + linear_channel_first(g, self.cond_linear)
        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs, strict=True)):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = ups(x)
            if isinstance(noise_convs, nn.Linear):
                x_source = linear_channel_first(har_source, noise_convs)
            else:
                x_source = noise_convs(har_source)
            x = x + x_source
            xs = self.resblocks[i * self.num_kernels](x)
            for j in range(i * self.num_kernels + 1, (i + 1) * self.num_kernels):
                resblock = self.resblocks[j]
                xs += resblock(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


sr2sr = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


class SynthesizerTrnMsNSF(nn.Module):
    def __init__(
        self,
        embedding_dims,
        inter_channels,
        hidden_channels,
        filter_channels,
        num_heads,
        num_layers,
        kernel_size,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels,
        sr,
    ):
        super().__init__()
        if isinstance(sr, str):
            sr = sr2sr[sr]
        self.enc_p = TextEncoder(
            embedding_dims,
            inter_channels,
            hidden_channels,
            filter_channels,
            num_heads,
            num_layers,
            kernel_size,
        )
        self.dec = GeneratorNSF(
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
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def forward(
        self,
        phone: torch.Tensor,
        pitch: torch.Tensor,
        nsff0: torch.Tensor,
        speaker_id: torch.Tensor,
    ):
        g = self.emb_g(speaker_id).unsqueeze(-1)
        m_p, logs_p = self.enc_p(phone, pitch)
        # permutation trick below is needed to match the result from TT implementation which has the opposite order of C, T
        # this enables so that they give the same result with the same random seed.
        # z_p = m_p + torch.exp(logs_p) * torch.randn_like(m_p.permute(0, 2, 1).contiguous()).permute(0, 2, 1) * 0.66666
        # instead of randn_like use randn with shape
        m_p_shape = (m_p.shape[0], m_p.shape[2], m_p.shape[1])  # swap C and T
        z_p = m_p + torch.exp(logs_p) * torch.randn(m_p_shape).permute(0, 2, 1) * 0.66666
        z = self.flow(z_p, g=g)
        o = self.dec(z, nsff0, g=g)
        return o


class SynthesizerTrnMsNSF_nono(nn.Module):
    def __init__(
        self,
        embedding_dims,
        inter_channels,
        hidden_channels,
        filter_channels,
        num_heads,
        num_layers,
        kernel_size,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels,
        sr=None,
    ):
        super().__init__()
        self.enc_p = TextEncoder(
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
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def forward(
        self,
        phone: torch.Tensor,
        speaker_id: torch.Tensor,
    ):
        g = self.emb_g(speaker_id).unsqueeze(-1)
        m_p, logs_p = self.enc_p(phone, None)
        z_p = m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666
        z = self.flow(z_p, g=g)
        o = self.dec(z, g=g)
        return o
