"""MiniMaxMusic3ConditionEncoder as a plain torch module. Keys match condition_encoder/*.safetensors."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.autoports.minimaxai_minimax_music3.config import CondConfig


class Music3ConditionEncoder(nn.Module):
    def __init__(self, cfg: CondConfig = CondConfig()):
        super().__init__()
        self.cfg = cfg
        self.layer_weight_logits = nn.Parameter(torch.zeros(cfg.num_condition_layers))
        self.layer_scale = nn.Parameter(torch.ones(1))
        self.proj = nn.Conv1d(cfg.condition_hidden_dim, cfg.out_dim, kernel_size=3, padding=1)

    def latent_length(self, num_frames: int) -> int:
        c = self.cfg
        return max(
            1,
            int(num_frames * c.output_sampling_rate / c.input_sampling_rate * c.input_hop_length / c.output_hop_length),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[B, frames, layers*D] -> [B, latent_len, out_dim]."""
        b, n, _ = hidden_states.shape
        c = self.cfg
        x = hidden_states.transpose(1, 2).reshape(b, c.num_condition_layers, c.condition_hidden_dim, n)
        w = torch.softmax(self.layer_weight_logits, dim=0).to(x.dtype)
        x = torch.einsum("blht,l->bht", x, w)
        x = self.layer_scale.to(x.dtype) * x
        x = self.proj(x)
        x = F.interpolate(x, size=self.latent_length(n), mode="nearest")
        return x.transpose(1, 2)

    @staticmethod
    def load(snapshot, dtype=torch.float32, device="cpu", cfg: CondConfig | None = None) -> "Music3ConditionEncoder":
        from safetensors.torch import load_file

        m = Music3ConditionEncoder(cfg or CondConfig())
        sd = load_file(str(Path(snapshot, "condition_encoder", "diffusion_pytorch_model.safetensors")), device="cpu")
        m.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True)
        return m.to(device).eval()
