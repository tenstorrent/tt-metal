# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.demos.rvc.torch_impl.utils import linear_channel_first


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts


LRELU_SLOPE = 0.1


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        num_heads,
        num_layers,
        kernel_size=1,
        window_size=10,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = int(num_layers)
        self.kernel_size = kernel_size
        self.window_size = window_size

        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(self.num_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    num_heads,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x):
        zippep = zip(self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2, strict=True)
        for attn_layers, norm_layers_1, ffn_layers, norm_layers_2 in zippep:
            y = attn_layers(x, x)
            x = norm_layers_1(x + y)

            y = ffn_layers(x)
            x = norm_layers_2(x + y)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_heads,
        window_size=None,
    ):
        super().__init__()
        assert in_features % num_heads == 0

        self.num_heads = num_heads
        self.window_size = window_size

        self.features_per_head = in_features // num_heads
        self.linear_q = nn.Linear(in_features, in_features)
        self.linear_k = nn.Linear(in_features, in_features)
        self.linear_v = nn.Linear(in_features, in_features)
        self.linear_o = nn.Linear(in_features, out_features)

        if window_size is not None:
            num_heads_rel = 1
            rel_stddev = self.features_per_head**-0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(num_heads_rel, window_size * 2 + 1, self.features_per_head) * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(num_heads_rel, window_size * 2 + 1, self.features_per_head) * rel_stddev
            )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        q = linear_channel_first(x, self.linear_q)
        k = linear_channel_first(c, self.linear_k)
        v = linear_channel_first(c, self.linear_v)

        x = self.attention(q, k, v)

        x = linear_channel_first(x, self.linear_o)
        return x

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s = key.size()
        t_t = query.size(2)
        query = query.view(b, self.num_heads, self.features_per_head, t_t).transpose(2, 3)
        key = key.view(b, self.num_heads, self.features_per_head, t_s)
        value = value.view(b, self.num_heads, self.features_per_head, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.features_per_head), key)
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.features_per_head), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length: int):
        # Pad first before slice to avoid using cond ops.
        pad_length: int = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                [0, 0, pad_length, pad_length, 0, 0],
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        _, _, length, _ = x.size()
        device = x.device
        idx_row = torch.arange(length, device=device).view(length, 1)
        idx_col = torch.arange(length, device=device).view(1, length)
        rel_idx = idx_col - idx_row + (length - 1)  # [l, l], in [0, 2*l-2]
        rel_idx = rel_idx.view(1, 1, length, length).expand(x.size(0), x.size(1), length, length)
        x_final = x.gather(dim=3, index=rel_idx)
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        device = x.device
        idx_row = torch.arange(length, device=device).view(length, 1)
        idx_col = torch.arange(length, device=device).view(1, length)
        rel_idx = idx_col - idx_row + (length - 1)  # [l, l], in [0, 2*l-2]
        rel_idx = rel_idx.view(1, 1, length, length).expand(batch, heads, length, length)
        out = x.new_zeros(batch, heads, length, 2 * length - 1)
        out.scatter_(dim=3, index=rel_idx, src=x)
        return out


class FFN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding="same")
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding="same")

    def forward(self, x: torch.Tensor):
        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        return x


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
        self.encoder = Encoder(
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


class WN(nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        gin_channels=0,
    ):
        super().__init__()

        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        if gin_channels != 0:
            self.cond_layer = nn.Linear(gin_channels, 2 * hidden_channels * num_layers)

        for i in range(num_layers):
            dilation = dilation_rate**i
            in_layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding="same",
            )
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < num_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_linear = nn.Linear(hidden_channels, res_skip_channels)
            self.res_skip_layers.append(res_skip_linear)

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None):
        output = torch.zeros_like(x)
        # n_channels_tensor = torch.IntTensor([self.hidden_channels])
        if g is not None:
            g = linear_channel_first(g, self.cond_layer)

        for i, (in_layer, res_skip_linear) in enumerate(zip(self.in_layers, self.res_skip_layers, strict=True)):
            x_in = in_layer(x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)
            res_skip_acts = linear_channel_first(acts, res_skip_linear)
            if i < self.num_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = x + res_acts
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=d_value,
                    padding="same",
                )
                for d_value in dilation
            ]
        )

        self.convs2 = nn.ModuleList([nn.Conv1d(channels, channels, kernel_size, padding="same") for _ in dilation])
        self.lrelu_slope = LRELU_SLOPE

    def forward(self, x: torch.Tensor):
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.lrelu_slope)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=d_value,
                    padding="same",
                )
                for d_value in dilation
            ]
        )
        self.lrelu_slope = LRELU_SLOPE

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = c(xt)
            x = xt + x
        return x


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        gin_channels=0,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.hidden_channels = hidden_channels
        self.half_channels = channels // 2
        self.pre_linear = nn.Linear(self.half_channels, hidden_channels)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            num_layers,
            gin_channels=gin_channels,
        )
        self.post_linear = nn.Linear(hidden_channels, self.half_channels)

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor | None = None,
    ):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = linear_channel_first(x0, self.pre_linear)
        h = self.enc(h, g=g)
        stats = linear_channel_first(h, self.post_linear)
        x1 = x1 - stats
        x = torch.cat([x0, x1], 1)
        return x


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
                ResidualCouplingLayer(
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
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, kernel_size=7, padding="same")
        resblock = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            self.ups.append(
                nn.ConvTranspose1d(
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

        self.conv_post = nn.Conv1d(ch, 1, kernel_size=7, padding="same", bias=False)
        if gin_channels != 0:
            self.cond_linear = nn.Linear(gin_channels, upsample_initial_channel)

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + linear_channel_first(g, self.cond_linear)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
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
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, kernel_size=7, padding="same")
        resblock = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                nn.ConvTranspose1d(
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
                    nn.Conv1d(
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

        self.conv_post = nn.Conv1d(ch, 1, kernel_size=7, padding="same", bias=False)

        if gin_channels != 0:
            self.cond_linear = nn.Linear(gin_channels, upsample_initial_channel)

        self.upp = math.prod(upsample_rates)

        self.lrelu_slope = LRELU_SLOPE

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
