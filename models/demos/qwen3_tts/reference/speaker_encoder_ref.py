# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference implementation of the Qwen3-TTS Speaker Encoder (ECAPA-TDNN).

Architecture:
    Input: mel spectrogram [B, T, 128] (time-major, transposed to [B, 128, T] internally)
    -> TDNN Conv1d(128, 512, k=5, d=1) + ReLU
    -> SE-Res2Net Block (512, k=3, d=2, scale=8)
    -> SE-Res2Net Block (512, k=3, d=3, scale=8)
    -> SE-Res2Net Block (512, k=3, d=4, scale=8)
    -> MFA: cat blocks[1:] → Conv1d(1536, 1536, k=1) + ReLU
    -> Attentive Statistics Pooling(1536, att_ch=128)
    -> FC: Conv1d(3072, 2048, k=1) → [B, 2048]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDelayNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, dilation=dilation,
            padding="same", padding_mode="reflect",
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.conv(x))


class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.blocks = nn.ModuleList([
            TimeDelayNetBlock(in_channel, hidden_channel, kernel_size=kernel_size, dilation=dilation)
            for _ in range(scale - 1)
        ])
        self.scale = scale

    def forward(self, x):
        outputs = []
        output_part = None
        for i, hidden_part in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                output_part = hidden_part
            elif i == 1:
                output_part = self.blocks[i - 1](hidden_part)
            else:
                output_part = self.blocks[i - 1](hidden_part + output_part)
            outputs.append(output_part)
        return torch.cat(outputs, dim=1)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, se_channels, kernel_size=1, padding="same", padding_mode="reflect")
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(se_channels, out_channels, kernel_size=1, padding="same", padding_mode="reflect")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(dim=2, keepdim=True)
        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))
        return x * s


class SqueezeExcitationRes2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res2net_scale=8, se_channels=128, kernel_size=1, dilation=1):
        super().__init__()
        self.tdnn1 = TimeDelayNetBlock(in_channels, out_channels, kernel_size=1, dilation=1)
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TimeDelayNetBlock(out_channels, out_channels, kernel_size=1, dilation=1)
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)

    def forward(self, x):
        residual = x
        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x)
        return x + residual


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels, attention_channels=128):
        super().__init__()
        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(attention_channels, channels, kernel_size=1, padding="same", padding_mode="reflect")

    def _compute_statistics(self, x, m, dim=2):
        mean = (m * x).sum(dim)
        std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(self.eps))
        return mean, std

    def forward(self, x):
        seq_length = x.shape[-1]
        lengths = torch.ones(x.shape[0], device=x.device)

        mask = (
            torch.arange(seq_length, device=x.device, dtype=lengths.dtype).expand(len(lengths), seq_length)
            < (lengths * seq_length).unsqueeze(1)
        )
        mask = mask.unsqueeze(1).to(x.dtype)
        total = mask.sum(dim=2, keepdim=True)

        mean, std = self._compute_statistics(x, mask / total)
        mean = mean.unsqueeze(2).repeat(1, 1, seq_length)
        std = std.unsqueeze(2).repeat(1, 1, seq_length)
        attention = torch.cat([x, mean, std], dim=1)

        attention = self.conv(self.tanh(self.tdnn(attention)))
        attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(attention, dim=2)

        mean, std = self._compute_statistics(x, attention)
        pooled = torch.cat((mean, std), dim=1)
        return pooled.unsqueeze(2)


class SpeakerEncoderReference(nn.Module):
    """
    ECAPA-TDNN Speaker Encoder reference implementation.

    Default config (from Qwen3-TTS-12Hz-1.7B-Base):
        mel_dim=128, enc_dim=2048 (previously 1024 in docs but actual model is 2048)
        enc_channels=[512, 512, 512, 512, 1536]
        enc_kernel_sizes=[5, 3, 3, 3, 1]
        enc_dilations=[1, 2, 3, 4, 1]
        enc_attention_channels=128, enc_res2net_scale=8, enc_se_channels=128
    """

    def __init__(
        self,
        mel_dim=128,
        enc_dim=2048,
        enc_channels=None,
        enc_kernel_sizes=None,
        enc_dilations=None,
        enc_attention_channels=128,
        enc_res2net_scale=8,
        enc_se_channels=128,
    ):
        super().__init__()
        if enc_channels is None:
            enc_channels = [512, 512, 512, 512, 1536]
        if enc_kernel_sizes is None:
            enc_kernel_sizes = [5, 3, 3, 3, 1]
        if enc_dilations is None:
            enc_dilations = [1, 2, 3, 4, 1]

        self.enc_dim = enc_dim
        self.channels = enc_channels

        self.blocks = nn.ModuleList()
        self.blocks.append(TimeDelayNetBlock(mel_dim, enc_channels[0], enc_kernel_sizes[0], enc_dilations[0]))
        for i in range(1, len(enc_channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    enc_channels[i - 1], enc_channels[i],
                    res2net_scale=enc_res2net_scale,
                    se_channels=enc_se_channels,
                    kernel_size=enc_kernel_sizes[i],
                    dilation=enc_dilations[i],
                )
            )

        self.mfa = TimeDelayNetBlock(enc_channels[-1], enc_channels[-1], enc_kernel_sizes[-1], enc_dilations[-1])
        self.asp = AttentiveStatisticsPooling(enc_channels[-1], attention_channels=enc_attention_channels)
        self.fc = nn.Conv1d(enc_channels[-1] * 2, enc_dim, kernel_size=1, padding="same", padding_mode="reflect")

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [B, T, mel_dim] mel spectrogram (time-major)

        Returns:
            speaker_embedding: [B, enc_dim]
        """
        hidden_states = hidden_states.transpose(1, 2)  # [B, mel_dim, T]

        hidden_states_list = []
        for layer in self.blocks:
            hidden_states = layer(hidden_states)
            hidden_states_list.append(hidden_states)

        # MFA: concat SE-Res2Net outputs (skip initial TDNN)
        hidden_states = torch.cat(hidden_states_list[1:], dim=1)
        hidden_states = self.mfa(hidden_states)
        hidden_states = self.asp(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states.squeeze(-1)

    @classmethod
    def from_pretrained(cls, model_name_or_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
        """Load from HuggingFace pretrained weights."""
        from safetensors import safe_open

        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(model_name_or_path, filename="model.safetensors")
        except Exception:
            from pathlib import Path

            path = Path(model_name_or_path) / "model.safetensors"

        state_dict = {}
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                if key.startswith("speaker_encoder."):
                    state_dict[key[len("speaker_encoder."):]] = f.get_tensor(key)

        model = cls()
        model.load_state_dict(state_dict, strict=True)
        return model

    @classmethod
    def from_hf_state_dict(cls, state_dict):
        """Load from a pre-filtered state dict (keys already stripped of 'speaker_encoder.' prefix)."""
        model = cls()
        model.load_state_dict(state_dict, strict=True)
        return model
