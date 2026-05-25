"""
TTNN-ported CustomSTFT from reference/custom_stft.py.

Forward STFT:
  Two Conv1d operations (real part, imaginary part) on waveform.
  Implemented via ttnn.conv2d:
    input  (B, 1, T)  → reshape (B, T, 1, 1)   [NHWC for TTNN]
    weight (freq_bins, 1, n_fft) → reshape (freq_bins, 1, n_fft, 1) [OIHW]
    kernel_size=(n_fft, 1), stride=(hop, 1), padding=(0, 0)
    output (B, frames, 1, freq_bins) → reshape (B, freq_bins, frames)

  ttnn.conv2d requires batch_size, input_height, input_width at call time,
  extracted dynamically from the input tensor shape.

Inverse STFT:
  Two Conv_transpose1d operations.
  torch fallback — TTNN has no conv_transpose2d op.
  TODO: raise issue for ttnn.conv_transpose2d support.

Phase computation (atan2):
  TTNN has no atan2 op.
  torch fallback for: magnitude, phase = sqrt(r²+i²), atan2(i, r).
  TODO: raise issue for ttnn.atan2 support or implement via ttnn.atan + quadrant logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn

from .tt_utils import from_tt


class TTCustomSTFT(nn.Module):
    """
    TTNN-ported version of CustomSTFT.

    Forward (transform): ttnn.conv2d for real/imag projections.
    Inverse (inverse): torch conv_transpose1d fallback.
    """

    def __init__(self, ref_stft, device):
        super().__init__()
        self.device = device
        self.filter_length = ref_stft.filter_length
        self.hop_length = ref_stft.hop_length
        self.win_length = ref_stft.win_length
        self.n_fft = ref_stft.n_fft
        self.freq_bins = ref_stft.freq_bins
        self.center = ref_stft.center
        self.pad_mode = ref_stft.pad_mode

        # Keep original buffers for the torch-fallback iSTFT path
        self.register_buffer("window", ref_stft.window.clone())
        self.register_buffer("weight_forward_real", ref_stft.weight_forward_real.clone())
        self.register_buffer("weight_forward_imag", ref_stft.weight_forward_imag.clone())
        self.register_buffer("weight_backward_real", ref_stft.weight_backward_real.clone())
        self.register_buffer("weight_backward_imag", ref_stft.weight_backward_imag.clone())

        # ── TTNN conv weights for forward STFT ──────────────────────────────
        # weight_forward_real / imag: (freq_bins, 1, n_fft)
        # ttnn.conv2d expects weight in (out_channels, in_channels/groups, kH, kW)
        # We use kH=n_fft, kW=1 to emulate Conv1d
        w_real = ref_stft.weight_forward_real.detach().float()  # (freq_bins, 1, n_fft)
        w_imag = ref_stft.weight_forward_imag.detach().float()  # (freq_bins, 1, n_fft)
        # Reshape to 4-D: (out_c, in_c, kH, kW) = (freq_bins, 1, n_fft, 1)
        self._w_real_4d = w_real.unsqueeze(-1)  # (freq_bins, 1, n_fft, 1)
        self._w_imag_4d = w_imag.unsqueeze(-1)

    def _conv1d_via_ttnn(self, x: torch.Tensor, weight_4d: torch.Tensor, stride: int) -> torch.Tensor:
        """
        Emulate Conv1d(in_channels=1, out_channels=freq_bins, kernel_size=n_fft,
                       stride=hop, padding=0) via ttnn.conv2d.

        x: (B, 1, T_padded) — already padded for center if needed.
        Returns: (B, freq_bins, frames)
        """
        B, C_in, T = x.shape
        n_fft = weight_4d.shape[2]  # kernel H
        freq_bins = weight_4d.shape[0]  # output channels
        frames = (T - n_fft) // stride + 1

        # TTNN conv2d input must be channel-last: (B, H, W, C_in)
        # We treat H=T, W=1, so input → (B, T, 1, 1)
        x_nhwc = x.permute(0, 2, 1).unsqueeze(2).contiguous().float()  # (B, T, 1, 1)

        # Convert to ROW_MAJOR for conv2d
        x_tt = ttnn.from_torch(
            x_nhwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Weight in (out_c, in_c, kH, kW) = (freq_bins, 1, n_fft, 1)
        w_tt = ttnn.from_torch(
            weight_4d.contiguous().float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        output_tt = ttnn.conv2d(
            input_tensor=x_tt,
            weight_tensor=w_tt,
            device=self.device,
            in_channels=C_in,
            out_channels=freq_bins,
            batch_size=B,
            input_height=T,
            input_width=1,
            kernel_size=(n_fft, 1),
            stride=(stride, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
        )

        # ttnn.conv2d returns (output_tensor, out_height, out_width) in some versions
        if isinstance(output_tt, (tuple, list)):
            output_tt = output_tt[0]

        # TTNN conv2d output is (1, 1, B*H_out*W_out, C_out) in ROW_MAJOR
        # Robustly flatten all leading dims, keep last dim = freq_bins
        out = from_tt(output_tt)  # torch, some shape (..., freq_bins)
        out = out.reshape(-1, freq_bins)  # (B*frames, freq_bins)
        out = out.reshape(B, frames, freq_bins)
        out = out.permute(0, 2, 1).contiguous()  # (B, freq_bins, frames)
        return out

    def transform(self, waveform: torch.Tensor):
        """
        Forward STFT → (magnitude, phase).
        Real/imag projections via ttnn.conv2d; phase via torch atan2 fallback.
        TODO: implement ttnn.atan2 when available to eliminate torch fallback for phase.
        """
        if self.center:
            pad_len = self.n_fft // 2
            waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)

        # x: (B, T) → (B, 1, T) for conv
        x = waveform.unsqueeze(1)

        real_out = self._conv1d_via_ttnn(x, self._w_real_4d, self.hop_length)
        imag_out = self._conv1d_via_ttnn(x, self._w_imag_4d, self.hop_length)

        # Magnitude and phase — torch fallback for sqrt/atan2
        # ttnn.atan2 not available; raise flag if needed in future
        magnitude = torch.sqrt(real_out**2 + imag_out**2 + 1e-14)
        phase = torch.atan2(imag_out, real_out)
        correction_mask = (imag_out == 0) & (real_out < 0)
        phase[correction_mask] = torch.pi
        return magnitude, phase

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor, length=None) -> torch.Tensor:
        """
        Inverse STFT via torch conv_transpose1d.
        TODO: raise issue for ttnn.conv_transpose2d support.
        """
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)

        real_rec = F.conv_transpose1d(
            real_part,
            self.weight_backward_real,
            bias=None,
            stride=self.hop_length,
            padding=0,
        )
        imag_rec = F.conv_transpose1d(
            imag_part,
            self.weight_backward_imag,
            bias=None,
            stride=self.hop_length,
            padding=0,
        )
        waveform = real_rec - imag_rec

        if self.center:
            pad_len = self.n_fft // 2
            waveform = waveform[..., pad_len:-pad_len]
        if length is not None:
            waveform = waveform[..., :length]
        return waveform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag, phase = self.transform(x)
        return self.inverse(mag, phase, length=x.shape[-1])
