# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Repo-owned PyTorch implementation of Kokoro predictor + text encoder.

This implements the modules from upstream `kokoro/modules.py` (TextEncoder,
ProsodyPredictor, and dependencies) and loads weights directly from the official
Kokoro checkpoint (`kokoro-v1_0.pth`).

Model source:
- https://huggingface.co/hexgrad/Kokoro-82M
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.nn.utils import weight_norm

from .kokoro_config import KokoroConfig

# ---- Copied/adapted from upstream kokoro/modules.py & kokoro/istftnet.py ----


class LinearNorm(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, w_init_gain: str = "linear"):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(self, channels: int, kernel_size: int, depth: int, n_symbols: int, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(
                nn.Sequential(
                    weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                    LayerNorm(channels),
                    actv,
                    nn.Dropout(0.2),
                )
            )
        self.lstm = nn.LSTM(channels, channels // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)  # [B, T, chn]
        lengths = input_lengths if input_lengths.device == torch.device("cpu") else input_lengths.to("cpu")
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
        x_pad[:, :, : x.shape[-1]] = x
        x = x_pad
        x.masked_fill_(m, 0.0)
        return x


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim: int, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class DurationEncoder(nn.Module):
    def __init__(self, sty_dim: int, d_model: int, nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(
                nn.LSTM(
                    d_model + sty_dim,
                    d_model // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout,
                )
            )
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(
        self, x: torch.Tensor, style: torch.Tensor, text_lengths: torch.Tensor, m: torch.Tensor
    ) -> torch.Tensor:
        masks = m
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                lengths = text_lengths if text_lengths.device == torch.device("cpu") else text_lengths.to("cpu")
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
                x_pad[:, :, : x.shape[-1]] = x
                x = x_pad
        return x.transpose(-1, -2)


class AdaIN1d(nn.Module):
    def __init__(self, style_dim: int, num_features: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=True)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_type == "none":
            return x
        return F.interpolate(x, scale_factor=2, mode="nearest")


class AdainResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        style_dim: int = 64,
        actv=nn.LeakyReLU(0.2),
        upsample: str = "none",
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        if upsample == "none":
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1)
            )

    def _build_weights(self, dim_in: int, dim_out: int, style_dim: int):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) * torch.rsqrt(torch.tensor(2))
        return out


class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim: int, d_hid: int, nlayers: int, max_dur: int = 50, dropout: float = 0.1):
        super().__init__()
        self.text_encoder = DurationEncoder(sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout)
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList(
            [
                AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
                AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
                AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout),
            ]
        )
        self.N = nn.ModuleList(
            [
                AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
                AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
                AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout),
            ]
        )
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def F0Ntrain(self, x: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.shared(x.transpose(-1, -2))
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)
        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        return F0.squeeze(1), N.squeeze(1)


# ---- Public repo-owned wrapper (loads from HF checkpoint) ----


@dataclass(frozen=True)
class KokoroPredictorOutput:
    d: torch.Tensor
    duration: torch.Tensor
    pred_dur: torch.LongTensor
    pred_aln_trg: torch.Tensor
    en: torch.Tensor
    F0_pred: torch.Tensor
    N_pred: torch.Tensor
    t_en: torch.Tensor
    asr: torch.Tensor


class KokoroPredictor(nn.Module):
    def __init__(self, predictor: ProsodyPredictor, text_encoder: TextEncoder):
        super().__init__()
        self.predictor = predictor
        self.text_encoder = text_encoder

    @torch.no_grad()
    def forward(
        self,
        *,
        d_en: torch.Tensor,
        ref_s: torch.FloatTensor,
        input_ids: torch.LongTensor,
        input_lengths: torch.LongTensor,
        text_mask: torch.Tensor,
        speed: float = 1.0,
    ) -> KokoroPredictorOutput:
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=input_ids.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=input_ids.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0], device=input_ids.device)] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(input_ids.device)

        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg

        return KokoroPredictorOutput(
            d=d,
            duration=duration,
            pred_dur=pred_dur,
            pred_aln_trg=pred_aln_trg,
            en=en,
            F0_pred=F0_pred,
            N_pred=N_pred,
            t_en=t_en,
            asr=asr,
        )


def _strip_module_prefix(sd: dict) -> dict:
    if any(k.startswith("module.") for k in sd.keys()):
        return {k[len("module.") :]: v for k, v in sd.items()}
    return sd


def load_predictor_from_huggingface(
    repo_id: str = KokoroConfig.repo_id,  # type: ignore[attr-defined]
    device: Optional[str] = None,
) -> KokoroPredictor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    predictor = ProsodyPredictor(
        style_dim=cfg["style_dim"],
        d_hid=cfg["hidden_dim"],
        nlayers=cfg["n_layer"],
        max_dur=cfg["max_dur"],
        dropout=cfg["dropout"],
    )
    text_encoder = TextEncoder(
        channels=cfg["hidden_dim"],
        kernel_size=cfg["text_encoder_kernel_size"],
        depth=cfg["n_layer"],
        n_symbols=cfg["n_token"],
    )

    ckpt_name = "kokoro-v1_0.pth"
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "predictor" not in state or "text_encoder" not in state:
        raise RuntimeError(f"Unexpected checkpoint format; missing keys in {ckpt_name}: {list(state.keys())[:20]}")

    # Upstream Kokoro loads some submodules with `strict=False` (e.g., InstanceNorm affine
    # parameters that may be absent in certain checkpoints). Mirror that behavior.
    predictor.load_state_dict(_strip_module_prefix(state["predictor"]), strict=False)
    text_encoder.load_state_dict(_strip_module_prefix(state["text_encoder"]), strict=True)

    model = KokoroPredictor(predictor=predictor, text_encoder=text_encoder).to(device).eval()
    return model
