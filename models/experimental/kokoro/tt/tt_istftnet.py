"""
TTNN-ported istftnet.py — AdaIN1d, AdaINResBlock1, SineGen, SourceModuleHnNSF,
Generator, UpSample1d, AdainResBlk1d, Decoder.

TTNN ops used per class:
  TTAdaIN1d:
    - nn.Linear (fc)      → ttnn.linear  (tt_linear)
    - InstanceNorm1d       → torch fallback
      TODO: replace with ttnn.group_norm(x, num_groups=num_features)
            GroupNorm(G=C) is mathematically identical to InstanceNorm1d(C).
            Pending verification of ttnn.group_norm(num_groups=C) correctness
            for arbitrary C and sequence lengths.
    - elementwise ×, +    → torch (scalar broadcast — no TTNN overhead)

  TTAdaINResBlock1:
    - AdaIN (fc, norm)    → TTAdaIN1d (ttnn.linear + torch InstanceNorm fallback)
    - Conv1d (dilated)    → torch fallback
      TODO: replace with ttnn.conv2d for stride-1, dilation-1 cases.
            Dilated convs (dilation 3, 5) need ttnn.conv2d dilation support
            which exists but requires additional conv_config tuning.
    - Snake1D activation  → ttnn.sin (tt_sin) for sin²; add/mul via ttnn

  TTSineGen:
    - All ops: torch fallback (stochastic cumsum + interpolate loop)
      TODO: the core cumsum+sin is in principle implementable in TTNN,
            but the batch-loop for flag_for_pulse=True makes it hard.

  TTSourceModuleHnNSF:
    - SineGen             → TTSineGen (torch-based inner)
    - l_linear (Linear)  → ttnn.linear (tt_linear)
    - l_tanh             → ttnn.tanh   (tt_tanh)

  TTGenerator:
    - f0_upsamp (Upsample) → torch fallback (F.interpolate)
      TODO: ttnn.upsample when available
    - ConvTranspose1d (ups) → torch fallback
      TODO: raise issue for ttnn.conv_transpose1d support
    - noise_convs (Conv1d) → torch fallback (dilated / strided)
    - noise_res / resblocks → TTAdaINResBlock1
    - conv_post (Conv1d)   → torch fallback
    - stft.transform / inverse → TTCustomSTFT (ttnn.conv2d + torch iSTFT)
    - exp/sin activations  → ttnn.exp / ttnn.sin (tt_exp, tt_sin)

  TTUpSample1d:
    - F.interpolate       → torch fallback (no TTNN interpolate op)
      TODO: ttnn.upsample when available

  TTAdainResBlk1d:
    - conv1/conv2 (Conv1d)         → torch fallback (dynamic stride/dilation)
    - pool (ConvTranspose1d)       → torch fallback
    - norm1/norm2 (TTAdaIN1d)      → ttnn.linear + torch InstanceNorm fallback
    - actv (LeakyReLU)             → ttnn.leaky_relu (tt_leaky_relu)
    - conv1x1 (1×1 Conv1d)         → ttnn.linear (tt_linear, equiv. math)
    - rsqrt scale                  → ttnn.rsqrt (tt_rsqrt)

  TTDecoder:
    - F0_conv / N_conv (Conv1d)    → torch fallback
    - asr_res (1×1 Conv1d)         → ttnn.linear (tt_linear)
    - encode / decode (AdainResBlk1d) → TTAdainResBlk1d
    - generator                    → TTGenerator
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


from .tt_utils import (
    load_tt_linear,
    load_tt_weight,
    tt_leaky_relu,
    tt_linear,
    tt_sin,
    tt_exp,
    tt_tanh,
)


# ─────────────────────────────────────────────
# TTAdaIN1d
# ─────────────────────────────────────────────


class TTAdaIN1d(nn.Module):
    """
    Port of AdaIN1d.

    forward(x, s):
        h = fc(s)                     # ttnn.linear
        gamma, beta = chunk(h, 2)
        return (1 + gamma) * norm(x) + beta   # norm = InstanceNorm1d fallback
    """

    def __init__(self, ref_adain, device):
        super().__init__()
        self.device = device
        self.num_features = ref_adain.norm.num_features
        # Store the InstanceNorm1d running stats / affine params
        self.norm_weight = ref_adain.norm.weight.detach().clone() if ref_adain.norm.affine else None
        self.norm_bias = ref_adain.norm.bias.detach().clone() if ref_adain.norm.affine else None

        # fc: Linear(style_dim, num_features * 2) → ttnn.linear
        self.fc_w, self.fc_b, self.fc_out = load_tt_linear(ref_adain.fc, device)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L), s: (B, style_dim)
        h = tt_linear(s, self.fc_w, self.fc_b, self.fc_out, self.device)  # (B, 2C)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)

        # InstanceNorm1d — torch fallback
        # TODO: replace with ttnn.group_norm(x, num_groups=self.num_features)
        #       which is mathematically identical to InstanceNorm1d(C).
        x_norm = F.instance_norm(
            x.float(),
            weight=self.norm_weight.to(x.device) if self.norm_weight is not None else None,
            bias=self.norm_bias.to(x.device) if self.norm_bias is not None else None,
            eps=1e-5,
        )
        return (1.0 + gamma) * x_norm + beta


# ─────────────────────────────────────────────
# TTAdaINResBlock1
# ─────────────────────────────────────────────


class TTAdaINResBlock1(nn.Module):
    """
    Port of AdaINResBlock1 (used inside Generator).

    Snake1D activation: x + (1/a) * sin(a*x)^2
      = x  +  sin²(ax) / a
      implemented using ttnn.sin for the sin part.

    Conv1d (dilated): torch fallback.
    TODO: replace with ttnn.conv2d once dilation support is verified.
    """

    def __init__(self, ref_blk, device):
        super().__init__()
        self.device = device
        # Keep Conv1d as torch (dynamic L, various dilations)
        self.convs1 = ref_blk.convs1
        self.convs2 = ref_blk.convs2
        # AdaIN → TTNN-hybrid
        self.adain1 = nn.ModuleList([TTAdaIN1d(n, device) for n in ref_blk.adain1])
        self.adain2 = nn.ModuleList([TTAdaIN1d(n, device) for n in ref_blk.adain2])
        # Snake1D learnable parameters
        self.alpha1 = ref_blk.alpha1
        self.alpha2 = ref_blk.alpha2

    def _snake(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Snake1D: x + (1/a) * sin(a*x)^2  — sin via ttnn.sin."""
        ax = alpha * x  # (B, C, L)
        sin_ax = tt_sin(ax, self.device)  # ttnn.sin
        return x + (1.0 / alpha) * (sin_ax**2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1,
            self.convs2,
            self.adain1,
            self.adain2,
            self.alpha1,
            self.alpha2,
        ):
            xt = n1(x, s)
            xt = self._snake(xt, a1)
            xt = c1(xt)  # torch Conv1d
            xt = n2(xt, s)
            xt = self._snake(xt, a2)
            xt = c2(xt)  # torch Conv1d
            x = xt + x
        return x


# ─────────────────────────────────────────────
# TTSineGen
# ─────────────────────────────────────────────


class TTSineGen(nn.Module):
    """
    Port of SineGen.

    All operations use torch fallback due to stochastic state and the
    sequential cumsum+interpolate structure that is difficult to express
    in static TTNN graphs.
    TODO: The core cumsum → sin path can be ported to TTNN for the common
          flag_for_pulse=False case using ttnn.cumsum and ttnn.sin.
    """

    def __init__(self, ref_sg):
        super().__init__()
        self.sine_amp = ref_sg.sine_amp
        self.noise_std = ref_sg.noise_std
        self.harmonic_num = ref_sg.harmonic_num
        self.dim = ref_sg.dim
        self.sampling_rate = ref_sg.sampling_rate
        self.voiced_threshold = ref_sg.voiced_threshold
        self.flag_for_pulse = ref_sg.flag_for_pulse
        self.upsample_scale = ref_sg.upsample_scale

    def _f02uv(self, f0):
        return (f0 > self.voiced_threshold).float()

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        if not self.flag_for_pulse:
            rad_values = F.interpolate(
                rad_values.transpose(1, 2),
                scale_factor=1 / self.upsample_scale,
                mode="linear",
            ).transpose(1, 2)
            phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
            phase = F.interpolate(
                phase.transpose(1, 2) * self.upsample_scale,
                scale_factor=self.upsample_scale,
                mode="linear",
            ).transpose(1, 2)
            sines = torch.sin(phase)
        else:
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            sines = torch.cos(i_phase * 2 * torch.pi)
        return sines

    def forward(self, f0: torch.Tensor):
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


# ─────────────────────────────────────────────
# TTSourceModuleHnNSF
# ─────────────────────────────────────────────


class TTSourceModuleHnNSF(nn.Module):
    """
    Port of SourceModuleHnNSF.

    l_sin_gen: TTSineGen (torch-based)
    l_linear:  ttnn.linear
    l_tanh:    ttnn.tanh
    """

    def __init__(self, ref_src, device):
        super().__init__()
        self.device = device
        self.sine_amp = ref_src.sine_amp
        self.noise_std = ref_src.noise_std
        self.l_sin_gen = TTSineGen(ref_src.l_sin_gen)
        self.l_linear_w, self.l_linear_b, self.l_linear_out = load_tt_linear(ref_src.l_linear, device)

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        # sine_merge = tanh(linear(sine_wavs))
        # sine_wavs: (B, L, harmonic_num+1) — apply linear over last dim
        sm = tt_linear(sine_wavs, self.l_linear_w, self.l_linear_b, self.l_linear_out, self.device)
        sine_merge = tt_tanh(sm, self.device)
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


# ─────────────────────────────────────────────
# TTGenerator
# ─────────────────────────────────────────────


class TTGenerator(nn.Module):
    """
    Port of Generator from istftnet.py.

    TTNN-accelerated parts:
      - m_source linear + tanh    → TTSourceModuleHnNSF
      - resblocks AdaIN.fc        → TTAdaINResBlock1 (ttnn.linear per AdaIN)
      - conv_post activation exp/sin → ttnn.exp / ttnn.sin
      - stft.transform            → TTCustomSTFT (ttnn.conv2d)

    Torch fallbacks:
      - f0_upsamp (Upsample/interpolate)  → torch (no TTNN interpolate)
        TODO: ttnn.upsample when available
      - noise_convs (Conv1d strided)      → torch
        TODO: ttnn.conv2d for stride>1
      - ups (ConvTranspose1d)             → torch
        TODO: raise issue for ttnn.conv_transpose1d
      - conv_post (Conv1d k=7)            → torch
        TODO: ttnn.conv2d for k=7, s=1, p=3
      - stft.inverse (iSTFT)              → torch conv_transpose1d
        TODO: raise issue for ttnn.conv_transpose2d
      - reflection_pad                    → torch
        TODO: ttnn reflection pad when available
    """

    def __init__(self, ref_gen, device):
        super().__init__()
        self.device = device
        self.num_kernels = ref_gen.num_kernels
        self.num_upsamples = ref_gen.num_upsamples
        self.post_n_fft = ref_gen.post_n_fft

        self.m_source = TTSourceModuleHnNSF(ref_gen.m_source, device)

        # torch fallbacks
        self.f0_upsamp = ref_gen.f0_upsamp
        self.noise_convs = ref_gen.noise_convs
        self.ups = ref_gen.ups
        self.conv_post = ref_gen.conv_post
        self.reflection_pad = ref_gen.reflection_pad

        # TTNN-hybrid resblocks
        self.noise_res = nn.ModuleList([TTAdaINResBlock1(b, device) for b in ref_gen.noise_res])
        self.resblocks = nn.ModuleList([TTAdaINResBlock1(b, device) for b in ref_gen.resblocks])

        # STFT — use TTCustomSTFT which uses ttnn.conv2d for forward transform
        from .tt_custom_stft import TTCustomSTFT

        # Build TTCustomSTFT from the reference stft module
        # If the ref stft is already a CustomSTFT, wrap it; otherwise store ref
        self.stft = TTCustomSTFT(ref_gen.stft, device) if hasattr(ref_gen.stft, "weight_forward_real") else ref_gen.stft

    def forward(self, x: torch.Tensor, s: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # torch fallback for interpolation
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
            har_source, noi_source, uv = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)

        for i in range(self.num_upsamples):
            x = tt_leaky_relu(x.contiguous(), 0.1, self.device)  # ttnn.leaky_relu
            x_source = self.noise_convs[i](har)  # torch Conv1d fallback
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)  # torch ConvTranspose1d fallback
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)  # torch fallback
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, s)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels

        x = tt_leaky_relu(x.contiguous(), 0.01, self.device)  # ttnn.leaky_relu
        x = self.conv_post(x)  # torch Conv1d fallback

        # Post-activation: exp for magnitude, sin for phase — ttnn.exp / ttnn.sin
        spec = tt_exp(x[:, : self.post_n_fft // 2 + 1, :], self.device)
        phase = tt_sin(x[:, self.post_n_fft // 2 + 1 :, :], self.device)

        return self.stft.inverse(spec, phase)


# ─────────────────────────────────────────────
# TTUpSample1d
# ─────────────────────────────────────────────


class TTUpSample1d(nn.Module):
    """
    Port of UpSample1d.

    F.interpolate → torch fallback.
    TODO: ttnn.upsample when available.
    """

    def __init__(self, layer_type: str):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_type == "none":
            return x
        # torch fallback — TTNN has no 1-D interpolate op
        return F.interpolate(x, scale_factor=2, mode="nearest")


# ─────────────────────────────────────────────
# TTAdainResBlk1d
# ─────────────────────────────────────────────


class TTAdainResBlk1d(nn.Module):
    """
    Port of AdainResBlk1d (used in Decoder, ProsodyPredictor).

    conv1 / conv2 (Conv1d k=3 s=1 p=1)  → torch fallback
      TODO: ttnn.conv2d(kernel=(3,1), stride=(1,1), padding=(1,0))
    pool (ConvTranspose1d)               → torch fallback
      TODO: ttnn.conv_transpose1d
    conv1x1 (Conv1d k=1)                 → ttnn.linear (equivalent math for k=1 conv)
    norm1 / norm2 (TTAdaIN1d)            → ttnn.linear + torch InstanceNorm fallback
    actv (LeakyReLU 0.2)                 → ttnn.leaky_relu
    rsqrt scale (1/√2)                   → ttnn.rsqrt via scalar
    """

    def __init__(self, ref_blk, device):
        super().__init__()
        self.device = device
        self.upsample_type = ref_blk.upsample_type
        self.learned_sc = ref_blk.learned_sc

        self.upsample = TTUpSample1d(ref_blk.upsample_type)
        self.pool = ref_blk.pool  # Conv1d/ConvTranspose1d or Identity — torch
        self.actv = ref_blk.actv  # LeakyReLU instance kept for fallback
        self.dropout = ref_blk.dropout

        # Conv1d k=3 — torch fallback
        self.conv1 = ref_blk.conv1
        self.conv2 = ref_blk.conv2

        # Adaptive norms → TTNN-hybrid
        self.norm1 = TTAdaIN1d(ref_blk.norm1, device)
        self.norm2 = TTAdaIN1d(ref_blk.norm2, device)

        # Shortcut 1×1 conv → store as TTNN linear weight
        if self.learned_sc:
            # conv1x1: (out_c, in_c, 1) — treat as linear (in_c, out_c)
            w = ref_blk.conv1x1.weight.detach().float().squeeze(-1).T.contiguous()  # (in_c, out_c)
            self._conv1x1_w = load_tt_weight(w, device)
            self._conv1x1_out = ref_blk.conv1x1.out_channels
            self._conv1x1_b = None  # conv1x1 has no bias in reference

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if self.learned_sc:
            # x: (B, in_c, T) → apply 1×1 conv as linear over channel dim
            x_t = x.transpose(1, 2)  # (B, T, in_c)
            out = tt_linear(x_t, self._conv1x1_w, self._conv1x1_b, self._conv1x1_out, self.device)
            x = out.transpose(1, 2)  # (B, out_c, T)
        return x

    def _residual(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x, s)
        x = tt_leaky_relu(x.contiguous(), 0.2, self.device)  # ttnn.leaky_relu
        x = self.pool(x)  # torch fallback
        x = self.conv1(self.dropout(x))  # torch Conv1d
        x = self.norm2(x, s)
        x = tt_leaky_relu(x.contiguous(), 0.2, self.device)  # ttnn.leaky_relu
        x = self.conv2(self.dropout(x))  # torch Conv1d
        return x

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) * (2.0**-0.5)  # 1/√2 scalar
        return out


# ─────────────────────────────────────────────
# TTDecoder
# ─────────────────────────────────────────────


class TTDecoder(nn.Module):
    """
    Port of Decoder from istftnet.py.

    F0_conv / N_conv (Conv1d k=3, s=2, p=1) → torch fallback
      TODO: ttnn.conv2d(kernel=(3,1), stride=(2,1), padding=(1,0))
    asr_res (Conv1d k=1, 512→64)             → ttnn.linear (1×1 conv = linear)
    encode / decode                          → TTAdainResBlk1d
    generator                                → TTGenerator
    """

    def __init__(self, ref_dec, device):
        super().__init__()
        self.device = device

        # Conv1d stride-2 → torch fallback
        self.F0_conv = ref_dec.F0_conv
        self.N_conv = ref_dec.N_conv

        # asr_res: Sequential(Conv1d(512, 64, 1)) → linear
        asr_conv = ref_dec.asr_res[0]  # the Conv1d(512, 64, 1)
        w = asr_conv.weight.detach().float().squeeze(-1).T.contiguous()  # (512, 64)
        self._asr_res_w = load_tt_weight(w, device)
        self._asr_res_out = asr_conv.out_channels
        b = asr_conv.bias
        # Plain torch tensor — tt_linear adds bias in torch (avoids TTNN tile-pad issue)
        self._asr_res_b = b.detach().float() if b is not None else None

        self.encode = TTAdainResBlk1d(ref_dec.encode, device)
        self.decode = nn.ModuleList([TTAdainResBlk1d(b, device) for b in ref_dec.decode])
        self.generator = TTGenerator(ref_dec.generator, device)

    def _asr_res(self, asr: torch.Tensor) -> torch.Tensor:
        """asr_res Sequential(Conv1d(512,64,1)) via ttnn.linear."""
        # asr: (B, 512, T) → (B, 64, T)
        asr_t = asr.transpose(1, 2)  # (B, T, 512)
        out = tt_linear(asr_t, self._asr_res_w, self._asr_res_b, self._asr_res_out, self.device)
        return out.transpose(1, 2)  # (B, 64, T)

    def forward(self, asr: torch.Tensor, F0_curve: torch.Tensor, N: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # torch fallback for strided Conv1d
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))

        x = torch.cat([asr, F0, N], axis=1)
        x = self.encode(x, s)

        asr_res = self._asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False

        x = self.generator(x, s, F0_curve)
        return x
