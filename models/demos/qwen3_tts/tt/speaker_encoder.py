# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Speaker Encoder (ECAPA-TDNN) implementation for Qwen3-TTS.

The Speaker Encoder extracts speaker embeddings from mel spectrograms.
These embeddings are used for voice cloning in the ICL input.

Architecture:
    - TimeDelayNetBlock (128 -> 512)
    - 3x SERes2NetBlocks (512 -> 512)
    - Multi-scale Feature Aggregation (MFA)
    - Attentive Statistics Pooling
    - Final FC (3072 -> 2048)

Note: Reflect ``conv1d`` runs on host PyTorch inside ``_conv1d_same_padding`` (no TTNN
reflect pad). Activations use ``ttnn.relu`` / ``ttnn.tanh`` where possible; conv/FC
weights are cached on device. Mel filterbank + Hann window are cached on host.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule


class SpeakerEncoderConfig:
    """Configuration for Speaker Encoder."""

    def __init__(
        self,
        sample_rate: int = 24000,
        n_mels: int = 128,
        channels: int = 512,
        output_dim: int = 2048,
        num_blocks: int = 4,  # 1 initial + 3 SE-Res2Net
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.channels = channels
        self.output_dim = output_dim
        self.num_blocks = num_blocks


class SpeakerEncoder(LightweightModule):
    """
    Speaker Encoder (ECAPA-TDNN) for extracting speaker embeddings.

    Takes mel spectrograms as input and outputs speaker embedding vectors
    that capture voice characteristics for voice cloning.

    Args:
        device: TTNN device
        state_dict: Model weights
        config: Speaker encoder configuration
        weight_cache_path: Optional path for weight caching
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        config: Optional[SpeakerEncoderConfig] = None,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.config = config or SpeakerEncoderConfig()

        # Extract speaker encoder weights from state dict
        self.weights = {}
        prefix = "speaker_encoder."
        for k, v in state_dict.items():
            if k.startswith(prefix):
                self.weights[k[len(prefix) :]] = v

        # Store weights as PyTorch tensors for CPU computation
        # Can be converted to TTNN when native ops are available
        self.pytorch_weights = {k: v.float() for k, v in self.weights.items()}
        self._conv1d_param_tt_cache = {}
        self._mel_stft_key = None
        self._fc_linear_weight_tt = None
        self._fc_bias_tt = None
        self._compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
        )

    def compute_mel_spectrogram(
        self,
        audio: torch.Tensor,
        n_fft: int = 1024,
        num_mels: int = 128,
        sampling_rate: int = 24000,
        hop_size: int = 256,
        win_size: int = 1024,
        fmin: int = 0,
        fmax: int = 12000,
    ) -> torch.Tensor:
        """
        Compute mel spectrogram from audio waveform.

        Args:
            audio: Audio waveform [num_samples] or [batch, num_samples]

        Returns:
            Mel spectrogram [batch, num_mels, time]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        device = audio.device
        mel_basis, hann_window = self._mel_stft_constants(n_fft, num_mels, sampling_rate, win_size, fmin, fmax)
        mel_basis = mel_basis.to(device)
        hann_window = hann_window.to(device)

        # Padding
        padding = (n_fft - hop_size) // 2
        y = F.pad(audio.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

        # STFT
        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

        # Apply mel filterbank
        mel_spec = torch.matmul(mel_basis, spec)

        # Log compression
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

        return mel_spec

    def _mel_stft_constants(
        self,
        n_fft: int,
        num_mels: int,
        sampling_rate: int,
        win_size: int,
        fmin: int,
        fmax: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (n_fft, num_mels, sampling_rate, win_size, fmin, fmax)
        if self._mel_stft_key != key:
            from librosa.filters import mel as librosa_mel_fn

            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
            self._mel_basis_cpu = torch.from_numpy(mel).float()
            self._hann_window_cpu = torch.hann_window(win_size).float()
            self._mel_stft_key = key
        return self._mel_basis_cpu, self._hann_window_cpu

    def _conv1d_params_to_ttnn(self, weight: torch.Tensor, bias: torch.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        key = (weight.data_ptr(), bias.data_ptr())
        hit = self._conv1d_param_tt_cache.get(key)
        if hit is None:
            w_tt = ttnn.from_torch(
                weight,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            b_tt = ttnn.from_torch(
                bias,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            hit = (w_tt, b_tt)
            self._conv1d_param_tt_cache[key] = hit
        return hit

    def _torch_ncl_to_ttnn_nlc(self, x_ncl: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            x_ncl.permute(0, 2, 1).contiguous(),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _ttnn_nlc_to_torch_ncl(self, x_nlc: ttnn.Tensor) -> torch.Tensor:
        t = ttnn.to_torch(x_nlc, dtype=torch.float32)
        if t.dim() == 4:
            t = t.squeeze(1)
        return t.permute(0, 2, 1).contiguous()

    def _conv1d_same_padding(
        self, x: ttnn.Tensor, weight: ttnn.Tensor, bias: ttnn.Tensor, dilation: int = 1
    ) -> ttnn.Tensor:
        """Same reflect pad + conv1d on host; I/O NLC ``[batch, seq, channels]``."""
        x_t = ttnn.to_torch(x, dtype=torch.float32)
        if x_t.dim() == 4:
            x_t = x_t.squeeze(1)
        x_t = x_t.permute(0, 2, 1).contiguous()
        w_t = ttnn.to_torch(weight, dtype=torch.float32)
        if w_t.dim() == 4:
            w_t = w_t.squeeze(-1)
        b_t = ttnn.to_torch(bias, dtype=torch.float32).reshape(-1)

        kernel_size = w_t.shape[-1]
        effective_kernel = dilation * (kernel_size - 1) + 1
        pad_total = effective_kernel - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x_padded = F.pad(x_t, (pad_left, pad_right), mode="reflect")
        out_t = F.conv1d(x_padded, w_t, b_t, dilation=dilation)
        out_nlc = out_t.permute(0, 2, 1).contiguous()
        return ttnn.from_torch(
            out_nlc,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _time_delay_net_block(
        self, x: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor, dilation: int = 1
    ) -> torch.Tensor:
        """Time Delay Network Block: reflect conv (host) + ``ttnn.relu`` on device."""
        w_tt, b_tt = self._conv1d_params_to_ttnn(conv_weight, conv_bias)
        y_tt = ttnn.relu(self._conv1d_same_padding(self._torch_ncl_to_ttnn_nlc(x), w_tt, b_tt, dilation))
        return self._ttnn_nlc_to_torch_ncl(y_tt)

    def _res2net_block(self, x: torch.Tensor, prefix: str, scale: int = 8) -> torch.Tensor:
        """Res2Net block with multi-scale feature extraction."""
        batch, channels, seq_len = x.shape
        parts = list(torch.chunk(x, scale, dim=1))
        outputs = []

        output_part = None
        for i, hidden_part in enumerate(parts):
            if i == 0:
                output_part = hidden_part
            elif i == 1:
                conv_weight = self.pytorch_weights.get(f"{prefix}blocks.{i-1}.conv.weight")
                conv_bias = self.pytorch_weights.get(f"{prefix}blocks.{i-1}.conv.bias")
                if conv_weight is not None:
                    output_part = self._time_delay_net_block(hidden_part, conv_weight, conv_bias)
                else:
                    output_part = hidden_part
            else:
                conv_weight = self.pytorch_weights.get(f"{prefix}blocks.{i-1}.conv.weight")
                conv_bias = self.pytorch_weights.get(f"{prefix}blocks.{i-1}.conv.bias")
                if conv_weight is not None:
                    output_part = self._time_delay_net_block(hidden_part + output_part, conv_weight, conv_bias)
                else:
                    output_part = hidden_part + output_part
            outputs.append(output_part)

        return torch.cat(outputs, dim=1)

    def _squeeze_excitation_block(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,
        conv1_bias: torch.Tensor,
        conv2_weight: torch.Tensor,
        conv2_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Squeeze-and-Excitation block."""
        y = x.mean(dim=2, keepdim=True)
        y = F.relu(F.conv1d(y, conv1_weight, conv1_bias))
        y = torch.sigmoid(F.conv1d(y, conv2_weight, conv2_bias))
        return x * y

    def _se_res2net_block(self, x: torch.Tensor, block_idx: int, scale: int = 8) -> torch.Tensor:
        """SERes2NetBlock: TDNN1 -> Res2Net -> TDNN2 -> SE -> residual add."""
        prefix = f"blocks.{block_idx}."
        residual = x

        # TDNN1
        tdnn1_weight = self.pytorch_weights.get(f"{prefix}tdnn1.conv.weight")
        tdnn1_bias = self.pytorch_weights.get(f"{prefix}tdnn1.conv.bias")
        if tdnn1_weight is not None:
            x = self._time_delay_net_block(x, tdnn1_weight, tdnn1_bias, dilation=1)

        # Res2Net
        x = self._res2net_block(x, f"{prefix}res2net_block.", scale)

        # TDNN2
        tdnn2_weight = self.pytorch_weights.get(f"{prefix}tdnn2.conv.weight")
        tdnn2_bias = self.pytorch_weights.get(f"{prefix}tdnn2.conv.bias")
        if tdnn2_weight is not None:
            x = self._time_delay_net_block(x, tdnn2_weight, tdnn2_bias, dilation=1)

        # SE block
        se_conv1_weight = self.pytorch_weights.get(f"{prefix}se_block.conv1.weight")
        se_conv1_bias = self.pytorch_weights.get(f"{prefix}se_block.conv1.bias")
        se_conv2_weight = self.pytorch_weights.get(f"{prefix}se_block.conv2.weight")
        se_conv2_bias = self.pytorch_weights.get(f"{prefix}se_block.conv2.bias")
        if se_conv1_weight is not None:
            x = self._squeeze_excitation_block(x, se_conv1_weight, se_conv1_bias, se_conv2_weight, se_conv2_bias)

        return x + residual

    def _attentive_statistics_pooling(
        self,
        x: torch.Tensor,
        tdnn_weight: torch.Tensor,
        tdnn_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Attentive Statistics Pooling."""
        batch, channels, seq_len = x.shape
        mask = torch.ones(batch, 1, seq_len, device=x.device, dtype=x.dtype)

        mean = (mask * x).sum(dim=2)
        std = torch.sqrt(((mask * (x - mean.unsqueeze(2)).pow(2)).sum(dim=2)).clamp(eps))

        mean_expanded = mean.unsqueeze(2).repeat(1, 1, seq_len)
        std_expanded = std.unsqueeze(2).repeat(1, 1, seq_len)

        attention_input = torch.cat([x, mean_expanded, std_expanded], dim=1)

        tw_tt, tb_tt = self._conv1d_params_to_ttnn(tdnn_weight, tdnn_bias)
        cw_tt, cb_tt = self._conv1d_params_to_ttnn(conv_weight, conv_bias)
        a_tt = self._conv1d_same_padding(self._torch_ncl_to_ttnn_nlc(attention_input), tw_tt, tb_tt)
        a_tt = ttnn.tanh(a_tt)
        a_tt = self._conv1d_same_padding(a_tt, cw_tt, cb_tt)
        attention = self._ttnn_nlc_to_torch_ncl(a_tt)
        attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(attention, dim=2)

        weighted_mean = (attention * x).sum(dim=2)
        weighted_std = torch.sqrt(((attention * (x - weighted_mean.unsqueeze(2)).pow(2)).sum(dim=2)).clamp(eps))

        pooled_stats = torch.cat([weighted_mean, weighted_std], dim=1)
        return pooled_stats.unsqueeze(2)

    def _ensure_fc_linear_params(self, fc_weight: torch.Tensor, fc_bias: torch.Tensor) -> None:
        """Lazy-build ``[1, 1, in, out]`` weight + bias for ``ttnn.linear`` (matches Talker layout)."""
        if self._fc_linear_weight_tt is not None:
            return
        w = fc_weight.squeeze(-1).float()
        in_f, out_f = w.shape[1], w.shape[0]
        w_host = w.T.contiguous().reshape(1, 1, in_f, out_f)
        self._fc_linear_weight_tt = ttnn.from_torch(
            w_host,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._fc_bias_tt = ttnn.from_torch(
            fc_bias.float().reshape(1, 1, 1, -1),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from mel spectrogram.

        Args:
            mel_spectrogram: Mel spectrogram [batch, n_mels, time] or [batch, time, n_mels]

        Returns:
            Speaker embedding [batch, 2048]
        """
        hidden = mel_spectrogram.float()

        # Transpose if needed: convolutions expect [batch, n_mels, time]
        # n_mels is fixed at 128, so check which dimension is 128
        if hidden.shape[2] == self.config.n_mels:
            # Input is [batch, time, n_mels], transpose to [batch, n_mels, time]
            hidden = hidden.transpose(1, 2)

        hidden_states_list = []

        # blocks[0]: Initial TDNN
        conv_weight = self.pytorch_weights.get("blocks.0.conv.weight")
        conv_bias = self.pytorch_weights.get("blocks.0.conv.bias")
        if conv_weight is not None:
            hidden = self._time_delay_net_block(hidden, conv_weight, conv_bias, dilation=1)
        hidden_states_list.append(hidden)

        # blocks[1-3]: SERes2NetBlocks
        for block_idx in range(1, 4):
            if f"blocks.{block_idx}.tdnn1.conv.weight" in self.pytorch_weights:
                hidden = self._se_res2net_block(hidden, block_idx, scale=8)
            hidden_states_list.append(hidden)

        # MFA: concatenate blocks 1-3
        hidden = torch.cat(hidden_states_list[1:], dim=1)

        # MFA TDNN
        mfa_weight = self.pytorch_weights.get("mfa.conv.weight")
        mfa_bias = self.pytorch_weights.get("mfa.conv.bias")
        if mfa_weight is not None:
            hidden = self._time_delay_net_block(hidden, mfa_weight, mfa_bias, dilation=1)

        # Attentive Statistics Pooling
        asp_tdnn_weight = self.pytorch_weights.get("asp.tdnn.conv.weight")
        asp_tdnn_bias = self.pytorch_weights.get("asp.tdnn.conv.bias")
        asp_conv_weight = self.pytorch_weights.get("asp.conv.weight")
        asp_conv_bias = self.pytorch_weights.get("asp.conv.bias")

        if asp_tdnn_weight is not None:
            hidden = self._attentive_statistics_pooling(
                hidden, asp_tdnn_weight, asp_tdnn_bias, asp_conv_weight, asp_conv_bias
            )
        else:
            mean = hidden.mean(dim=2)
            std = hidden.std(dim=2)
            hidden = torch.cat([mean, std], dim=1).unsqueeze(2)

        # Final FC: ``ttnn.linear`` (kernel 1 conv); weights prepared once
        fc_weight = self.pytorch_weights.get("fc.weight")
        fc_bias = self.pytorch_weights.get("fc.bias")
        if fc_weight is not None:
            self._ensure_fc_linear_params(fc_weight, fc_bias)
            b, ch, tlen = hidden.shape
            x_tt = ttnn.from_torch(
                hidden.reshape(b, 1, tlen, ch),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out_tt = ttnn.linear(
                x_tt,
                self._fc_linear_weight_tt,
                bias=self._fc_bias_tt,
                compute_kernel_config=self._compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            hidden = ttnn.to_torch(out_tt, dtype=torch.float32).reshape(b, -1)
        else:
            hidden = hidden.squeeze(-1)

        return hidden

    def forward_from_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding directly from audio waveform.

        Args:
            audio: Audio waveform [num_samples] or [batch, num_samples]

        Returns:
            Speaker embedding [batch, 2048]
        """
        mel = self.compute_mel_spectrogram(audio)
        return self.forward(mel)

    def to_ttnn(self, speaker_embedding: torch.Tensor) -> ttnn.Tensor:
        """
        Convert speaker embedding to TTNN tensor.

        Args:
            speaker_embedding: PyTorch speaker embedding [batch, 2048]

        Returns:
            TTNN tensor [batch, 1, 1, 2048]
        """
        # Reshape for TTNN: [batch, 2048] -> [batch, 1, 1, 2048]
        embedding = speaker_embedding.unsqueeze(1).unsqueeze(1)
        return ttnn.from_torch(
            embedding,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
