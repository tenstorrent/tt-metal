"""HiFTGenerator vocoder — native TTNN implementation (Stage 2.4).

Ports the HiFT vocoder conv stack from host CPU to Tenstorrent N300 via TTNN.
SineGen2 (DSP) + iSTFT remain on host (no TTNN equivalent).

Architecture:
  1. F0 predictor: 5× [Conv1d(k=3, pad=1) + ELU] (80→512) + Linear(512→1) + abs()
  2. SineGen2 (host): f0 → source excitation [B, 1, T*480]
  3. Decode (device):
     - conv_pre: Conv1d(80→512, k=7, pad=3)
     - 3× upsample: leaky_relu → ConvTranspose1d → [reflection_pad] → source fusion → 3× ResBlock(Snake)
     - leaky_relu → conv_post(64→18, k=7, pad=3)
  4. iSTFT (host): magnitude + phase → waveform

Reference: cosyvoice/hifigan/generator.py::HiFTGenerator
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import ttnn

_COSYVOICE_SRC = str(Path(__file__).resolve().parents[2] / "model_data" / "CosyVoice_src")
_MATCHA = str(Path(_COSYVOICE_SRC) / "third_party" / "Matcha-TTS")
if _COSYVOICE_SRC not in sys.path:
    sys.path.insert(0, _COSYVOICE_SRC)
if _MATCHA not in sys.path:
    sys.path.append(_MATCHA)


def _to_tile(t: torch.Tensor, device: ttnn.MeshDevice, dtype=ttnn.DataType.BFLOAT16) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _to_host(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(t)


def _get_padding(kernel_size: int, dilation: int) -> int:
    return (kernel_size - 1) * dilation // 2


def _load_folded_weights(hift_pt_path: str | Path) -> Tuple[Dict[str, torch.Tensor], object]:
    from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor
    from cosyvoice.hifigan.generator import HiFTGenerator
    from torch.nn.utils.parametrize import remove_parametrizations

    sd = torch.load(str(hift_pt_path), map_location="cpu", weights_only=True)

    f0_pred = ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)
    model = HiFTGenerator(
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=24000,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        audio_limit=0.99,
        f0_predictor=f0_pred,
    )
    model.load_state_dict(sd, strict=False)

    for _, module in model.named_modules():
        if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
            remove_parametrizations(module, "weight", leave_parametrized=True)

    model.eval()
    return model.state_dict(), model


class Conv1dTtnn:
    """Non-causal Conv1d with optional stride and dilation."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        device: ttnn.MeshDevice,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
    ):
        self.device = device
        self.kernel_size = weight.shape[2]
        self.in_channels = weight.shape[1]
        self.out_channels = weight.shape[0]
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

        self.weight_tt = ttnn.from_torch(
            weight.unsqueeze(2),
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self.bias_tt = ttnn.from_torch(
            bias.reshape(1, 1, 1, -1),
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self._preprocessed_weights = None
        self._preprocessed_bias = None
        self._preprocessed_input_length = None
        self._compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        )

    def __call__(self, x: ttnn.Tensor, batch_size: int, input_length: int) -> ttnn.Tensor:
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(x_rm, (batch_size, 1, input_length, self.in_channels))

        use_cached = self._preprocessed_weights is not None and self._preprocessed_input_length == input_length

        kwargs = dict(
            input_tensor=x_rm,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=batch_size,
            input_length=input_length,
            dtype=ttnn.DataType.BFLOAT16,
            compute_config=self._compute_config,
            return_output_dim=True,
        )

        if use_cached:
            kwargs["weight_tensor"] = self._preprocessed_weights
            kwargs["bias_tensor"] = self._preprocessed_bias
        else:
            kwargs["weight_tensor"] = self.weight_tt
            kwargs["bias_tensor"] = self.bias_tt
            kwargs["return_weights_and_bias"] = True

        out = ttnn.conv1d(**kwargs)

        if isinstance(out, tuple):
            out_tensor = out[0]
            out_length = out[1]
            if len(out) == 3 and not use_cached:
                wb = out[2]
                self._preprocessed_weights = wb[0]
                self._preprocessed_bias = wb[1]
                self._preprocessed_input_length = input_length
        else:
            out_tensor = out
            out_length = input_length

        out_tensor = ttnn.reshape(out_tensor, (batch_size, out_length, self.out_channels))
        out_tensor = ttnn.to_layout(out_tensor, ttnn.TILE_LAYOUT)
        return out_tensor, out_length


class ConvTranspose1dTtnn:
    """ConvTranspose1d via ttnn.conv_transpose2d (validated D16, PCC≥0.99999)."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        device: ttnn.MeshDevice,
        stride: int,
        padding: int,
    ):
        self.device = device
        self.in_channels = weight.shape[0]
        self.out_channels = weight.shape[1]
        self.kernel_size = weight.shape[2]
        self.stride = stride
        self.padding = padding

        w_2d = weight.unsqueeze(-1).contiguous()
        self.weight_tt = ttnn.from_torch(
            w_2d,
            dtype=ttnn.DataType.BFLOAT16,
        )
        self.bias_tt = ttnn.from_torch(
            bias.reshape(1, 1, 1, -1),
            dtype=ttnn.DataType.BFLOAT16,
        )
        self._preprocessed_weights = None
        self._preprocessed_bias = None
        self._preprocessed_input_height = None

    def __call__(self, x: ttnn.Tensor, batch_size: int, input_height: int) -> Tuple[ttnn.Tensor, int]:
        x_4d = ttnn.reshape(x, (batch_size, input_height, 1, self.in_channels))

        use_cached = self._preprocessed_weights is not None and self._preprocessed_input_height == input_height

        kwargs = dict(
            input_tensor=x_4d,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            batch_size=batch_size,
            input_height=input_height,
            input_width=1,
            kernel_size=(self.kernel_size, 1),
            stride=(self.stride, 1),
            padding=(self.padding, 0),
            dtype=ttnn.DataType.BFLOAT16,
            return_output_dim=True,
        )

        if use_cached:
            kwargs["weight_tensor"] = self._preprocessed_weights
            kwargs["bias_tensor"] = self._preprocessed_bias
        else:
            kwargs["weight_tensor"] = self.weight_tt
            kwargs["bias_tensor"] = self.bias_tt
            kwargs["return_weights_and_bias"] = True

        out = ttnn.conv_transpose2d(**kwargs)

        if isinstance(out, tuple):
            out_tensor = out[0]
            out_dims = out[1]
            out_height = out_dims[0]
            if len(out) == 3 and not use_cached:
                wb = out[2]
                self._preprocessed_weights = wb[0]
                self._preprocessed_bias = wb[1]
                self._preprocessed_input_height = input_height
        else:
            out_tensor = out
            out_height = (input_height - 1) * self.stride - 2 * self.padding + self.kernel_size

        out_tensor = ttnn.reshape(out_tensor, (batch_size, out_height, self.out_channels))
        return out_tensor, out_height


class SnakeTtnn:
    """Snake activation: x + (1/alpha) * sin²(alpha * x). Computed on host (fp32) for precision."""

    def __init__(self, alpha: torch.Tensor, device: ttnn.MeshDevice):
        self.device = device
        self.alpha = alpha.reshape(1, 1, -1).float()
        self.inv_alpha = (1.0 / (alpha + 1e-9)).reshape(1, 1, -1).float()

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x_host = _to_host(x).float()
        ax = self.alpha * x_host
        out = x_host + self.inv_alpha * torch.sin(ax).pow(2)
        return _to_tile(out, self.device)


class ResBlockTtnn:
    """MRF Residual block: 3× (Snake → Conv1d_dilated → Snake → Conv1d) + residual."""

    def __init__(
        self,
        prefix: str,
        weights: Dict[str, torch.Tensor],
        channels: int,
        kernel_size: int,
        dilations: List[int],
        device: ttnn.MeshDevice,
    ):
        self.convs1 = []
        self.convs2 = []
        self.activations1 = []
        self.activations2 = []

        for j, d in enumerate(dilations):
            pad = _get_padding(kernel_size, d)
            self.convs1.append(
                Conv1dTtnn(
                    weights[f"{prefix}.convs1.{j}.weight"],
                    weights[f"{prefix}.convs1.{j}.bias"],
                    device,
                    dilation=d,
                    padding=pad,
                )
            )
            pad2 = _get_padding(kernel_size, 1)
            self.convs2.append(
                Conv1dTtnn(
                    weights[f"{prefix}.convs2.{j}.weight"],
                    weights[f"{prefix}.convs2.{j}.bias"],
                    device,
                    dilation=1,
                    padding=pad2,
                )
            )
            self.activations1.append(SnakeTtnn(weights[f"{prefix}.activations1.{j}.alpha"], device))
            self.activations2.append(SnakeTtnn(weights[f"{prefix}.activations2.{j}.alpha"], device))

    def __call__(self, x: ttnn.Tensor, batch_size: int, T: int) -> ttnn.Tensor:
        for j in range(len(self.convs1)):
            xt = self.activations1[j](x)
            xt, _ = self.convs1[j](xt, batch_size, T)
            xt = self.activations2[j](xt)
            xt, _ = self.convs2[j](xt, batch_size, T)
            x = ttnn.add(xt, x)
        return x


class F0PredictorTtnn:
    """ConvRNNF0Predictor: 5× [Conv1d(k=3, pad=1) + ELU] + Linear + abs()."""

    def __init__(self, weights: Dict[str, torch.Tensor], device: ttnn.MeshDevice):
        self.device = device
        self.convs = []
        condnet_indices = [0, 2, 4, 6, 8]
        for idx in condnet_indices:
            self.convs.append(
                Conv1dTtnn(
                    weights[f"f0_predictor.condnet.{idx}.weight"],
                    weights[f"f0_predictor.condnet.{idx}.bias"],
                    device,
                    padding=1,
                )
            )
        w = weights["f0_predictor.classifier.weight"]
        b = weights["f0_predictor.classifier.bias"]
        self.linear_weight = _to_tile(w.T.contiguous(), device)
        self.linear_bias = _to_tile(b.reshape(1, 1, -1), device)

    def __call__(self, mel_btc: ttnn.Tensor, batch_size: int, T: int) -> torch.Tensor:
        x = mel_btc
        for conv in self.convs:
            x, _ = conv(x, batch_size, T)
            x = ttnn.elu(x)
        x = ttnn.add(ttnn.matmul(x, self.linear_weight), self.linear_bias)
        x_host = _to_host(x)
        return torch.abs(x_host.squeeze(-1))


class HiFTGeneratorTtnn:
    """Native TTNN HiFTGenerator — conv stack on device, SineGen2 + iSTFT on host."""

    def __init__(self, folded_weights: Dict[str, torch.Tensor], ref_model, device: ttnn.MeshDevice):
        self.device = device
        self.lrelu_slope = 0.1
        self.audio_limit = 0.99
        self.istft_params = {"n_fft": 16, "hop_len": 4}
        self.num_upsamples = 3
        self.num_kernels = 3
        self.upsample_rates = [8, 5, 3]

        self._ref = ref_model
        self._build_f0_predictor(folded_weights)
        self._build_decode(folded_weights)

    def _build_f0_predictor(self, sd: Dict[str, torch.Tensor]):
        self.f0_predictor = F0PredictorTtnn(sd, self.device)

    def _build_decode(self, sd: Dict[str, torch.Tensor]):
        self.conv_pre = Conv1dTtnn(sd["conv_pre.weight"], sd["conv_pre.bias"], self.device, padding=3)

        self.ups = []
        upsample_rates = [8, 5, 3]
        upsample_kernel_sizes = [16, 11, 7]
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                ConvTranspose1dTtnn(
                    sd[f"ups.{i}.weight"],
                    sd[f"ups.{i}.bias"],
                    self.device,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )

        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = list(np.cumprod(downsample_rates))
        source_resblock_kernel_sizes = [7, 7, 11]
        source_resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        base_channels = 512

        self.source_downs = []
        self.source_resblocks = []
        for i, (u, k, d) in enumerate(
            zip(
                downsample_cum_rates[::-1],
                source_resblock_kernel_sizes,
                source_resblock_dilation_sizes,
            )
        ):
            ch = base_channels // (2 ** (i + 1))
            if u == 1:
                self.source_downs.append(
                    Conv1dTtnn(
                        sd[f"source_downs.{i}.weight"],
                        sd[f"source_downs.{i}.bias"],
                        self.device,
                    )
                )
            else:
                self.source_downs.append(
                    Conv1dTtnn(
                        sd[f"source_downs.{i}.weight"],
                        sd[f"source_downs.{i}.bias"],
                        self.device,
                        stride=int(u),
                        padding=int(u // 2),
                    )
                )
            self.source_resblocks.append(ResBlockTtnn(f"source_resblocks.{i}", sd, ch, k, d, self.device))

        resblock_kernel_sizes = [3, 7, 11]
        resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        self.resblocks = []
        for i in range(self.num_upsamples):
            ch = base_channels // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(
                    ResBlockTtnn(
                        f"resblocks.{i * self.num_kernels + j}",
                        sd,
                        ch,
                        k,
                        d,
                        self.device,
                    )
                )

        self.conv_post = Conv1dTtnn(sd["conv_post.weight"], sd["conv_post.bias"], self.device, padding=3)

    @classmethod
    def from_checkpoint(cls, hift_pt_path: str | Path, device: ttnn.MeshDevice) -> "HiFTGeneratorTtnn":
        folded_sd, ref_model = _load_folded_weights(hift_pt_path)
        return cls(folded_sd, ref_model, device)

    @torch.inference_mode()
    def inference(
        self,
        mel: torch.Tensor,
        cache_source: torch.Tensor = torch.zeros(1, 1, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = mel.shape[0]
        T_mel = mel.shape[2]

        with torch.inference_mode():
            f0 = self._ref.f0_predictor(mel)

        s = self._ref.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = self._ref.m_source(s)
        s = s.transpose(1, 2)

        if cache_source.shape[2] != 0:
            s[:, :, : cache_source.shape[2]] = cache_source

        waveform = self._decode_device(mel, s, B, T_mel)
        return waveform, s

    def _decode_device(self, mel: torch.Tensor, s: torch.Tensor, B: int, T_mel: int) -> torch.Tensor:
        s_stft_real, s_stft_imag = self._ref._stft(s.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        with torch.inference_mode():
            x = self._ref.conv_pre(mel)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)

            x_btc = x.permute(0, 2, 1).contiguous()
            T_in = x_btc.shape[1]
            T_in_pad = ((T_in + 31) // 32) * 32
            if T_in_pad != T_in:
                x_btc = F.pad(x_btc, (0, 0, 0, T_in_pad - T_in))
            x_tt = _to_tile(x_btc, self.device)
            x_tt, T_out = self.ups[i](x_tt, B, T_in_pad)
            x_host = _to_host(x_tt)
            expected_t = (T_in - 1) * self.ups[i].stride - 2 * self.ups[i].padding + self.ups[i].kernel_size
            x = x_host[:, :expected_t, :].permute(0, 2, 1).contiguous()

            if i == self.num_upsamples - 1:
                x = self._ref.reflection_pad(x)

            with torch.inference_mode():
                si = self._ref.source_downs[i](s_stft)
                si = self._ref.source_resblocks[i](si)
                x = x + si

                xs = None
                for j in range(self.num_kernels):
                    rb = self._ref.resblocks[i * self.num_kernels + j](x)
                    xs = rb if xs is None else xs + rb
                x = xs / self.num_kernels

        x = F.leaky_relu(x, self.lrelu_slope)
        with torch.inference_mode():
            x = self._ref.conv_post(x)

        n_fft_half = self.istft_params["n_fft"] // 2 + 1
        magnitude = torch.exp(x[:, :n_fft_half, :])
        phase = torch.sin(x[:, n_fft_half:, :])

        waveform = self._ref._istft(magnitude, phase)
        waveform = torch.clamp(waveform, -self.audio_limit, self.audio_limit)
        return waveform
