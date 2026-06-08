# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CausalAudioEncoder parity."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from loguru import logger

import ttnn

from .....models.transformers.wan2_2.s2v.audio_utils import CausalAudioEncoder
from .....utils.check import assert_quality
from .....utils.tensor import local_device_to_torch
from .....utils.test import line_params, ring_params

# Production Wan-AI/Wan2.2-S2V-14B config.
AUDIO_DIM = 1024
NUM_LAYERS = 25
OUT_DIM = 5120
NUM_TOKEN = 4
MOTION_FRAMES = (17, 5)
T_AUDIO = 80


# ---------------------------------------------------------------------------
# Torch reference — inlined ports of upstream WAN audio modules.
# References: wan/modules/s2v/audio_utils.py + wan/modules/s2v/auxi_blocks.py.
# ---------------------------------------------------------------------------


class TorchCausalConv1d(nn.Module):
    """``nn.Conv1d`` with replicate left-pad by ``kernel_size - 1`` (causal)."""

    def __init__(self, chan_in: int, chan_out: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.time_causal_padding, mode="replicate")
        return self.conv(x)


class TorchMotionEncoderTC(nn.Module):
    """Three causal-conv stages with 4x temporal downsample (need_global=False)."""

    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.conv1_local = TorchCausalConv1d(in_dim, hidden_dim // 4 * num_heads, 3, stride=1)
        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6)
        self.conv2 = TorchCausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2)
        self.norm2 = nn.LayerNorm(hidden_dim // 2, elementwise_affine=False, eps=1e-6)
        self.conv3 = TorchCausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.act = nn.SiLU()
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        x = x.permute(0, 2, 1)
        x = self.conv1_local(x)
        x = x.reshape(b, self.num_heads, self.hidden_dim // 4, t).permute(0, 1, 3, 2)
        x = x.reshape(b * self.num_heads, t, self.hidden_dim // 4)
        x = self.act(self.norm1(x))
        x = x.permute(0, 2, 1)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.act(self.norm2(x))
        x = x.permute(0, 2, 1)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x = self.act(self.norm3(x))
        x = x.reshape(b, self.num_heads, -1, self.hidden_dim).permute(0, 2, 1, 3)
        pad = self.padding_tokens.expand(b, x.shape[1], 1, self.hidden_dim)
        return torch.cat([x, pad], dim=2)


class TorchCausalAudioEncoder(nn.Module):
    """Weighted-sum of wav2vec2 hidden states + MotionEncoder_tc (need_global=False)."""

    def __init__(self, *, dim: int, num_layers: int, out_dim: int, num_token: int) -> None:
        super().__init__()
        self.encoder = TorchMotionEncoderTC(in_dim=dim, hidden_dim=out_dim, num_heads=num_token)
        self.weights = nn.Parameter(torch.ones(1, num_layers, 1, 1) * 0.01)
        self.act = nn.SiLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, num_layers, dim, T]
        w = self.act(self.weights)
        weighted = (features * w / w.sum(dim=1, keepdim=True)).sum(dim=1)
        return self.encoder(weighted.permute(0, 2, 1))


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((2, 4), (2, 4), 1, 0, 2, line_params, ttnn.Topology.Linear, False, id="bh_2x4sp1tp0"),
        pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, False, id="bh_4x8sp1tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_causal_audio_encoder(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    """End-to-end audio path parity: weighted-sum + MotionEncoder_tc."""
    torch.manual_seed(0)
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    ref = (
        TorchCausalAudioEncoder(dim=AUDIO_DIM, num_layers=NUM_LAYERS, out_dim=OUT_DIM, num_token=NUM_TOKEN)
        .eval()
        .to(torch.float32)
    )
    tt = CausalAudioEncoder(
        dim=AUDIO_DIM,
        num_layers=NUM_LAYERS,
        out_dim=OUT_DIM,
        num_token=NUM_TOKEN,
        need_global=False,
        mesh_device=mesh_device,
    )
    tt.load_torch_state_dict(ref.state_dict(), strict=False)

    audio_input_torch = torch.randn(1, NUM_LAYERS, AUDIO_DIM, T_AUDIO, dtype=torch.float32)
    # Left-pad by motion_frames[0] (wan/modules/s2v/model_s2v.py:683).
    pre = audio_input_torch[..., :1].expand(-1, -1, -1, MOTION_FRAMES[0])
    audio_input_padded = torch.cat([pre, audio_input_torch], dim=-1)

    with torch.no_grad():
        ref_out = ref(audio_input_padded)
        ref_merged = ref_out[:, MOTION_FRAMES[1] :, :, :]

    tt_out_torch = local_device_to_torch(tt(audio_input_padded))
    tt_merged = tt_out_torch[:, MOTION_FRAMES[1] :, :, :]

    assert tt_merged.shape == ref_merged.shape, f"shape: tt={tuple(tt_merged.shape)} ref={tuple(ref_merged.shape)}"
    assert_quality(tt_merged.float(), ref_merged.float(), pcc=0.99)
    logger.info(f"audio encoder parity ok: {tuple(tt_merged.shape)}")
