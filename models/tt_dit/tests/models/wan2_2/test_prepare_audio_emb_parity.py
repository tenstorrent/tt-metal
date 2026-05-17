# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Parity test for the S2V audio path.

Compares our on-device :class:`CausalAudioEncoder` against an inlined
pytorch reference of the reference repo's ``CausalAudioEncoder`` (the
math is in :class:`_RefCausalAudioEncoder` below).

Test bar: PCC ≥ 0.99. Catches regressions in audio bucketing / per-frame
alignment that surfaced as the lip-sync bug (motion_frames=73 vs 17).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from loguru import logger

import ttnn

from ....models.transformers.wan2_2.audio_utils_wan import CausalAudioEncoder
from ....utils.check import assert_quality
from ....utils.tensor import local_device_to_torch
from ....utils.test import line_params

# Reduced config — same shape contract as production but smaller.
# Production:  audio_dim=1024, num_layers=25, out_dim=5120, num_token=4.
# We shrink out_dim (the DiT inner dim — only affects MotionEncoder's last
# conv channel count) for fast CPU reference.
AUDIO_DIM = 1024
NUM_LAYERS = 25
OUT_DIM = 256
NUM_TOKEN = 4
MOTION_FRAMES = (17, 5)  # production [encoded_frames, latent_frames]
T_AUDIO = 80  # ≥ motion_frames[0] + small video extent at video_rate


class _RefCausalConv1d(nn.Module):
    """Reference WAN CausalConv1d: ``nn.Conv1d`` with replicate left-padding.

    Inlined from ``wan/modules/s2v/auxi_blocks.py:CausalConv1d`` so this test
    runs without the reference repo. Left-pad by ``kernel_size - 1`` so the
    output sample at position ``t`` only depends on inputs at positions
    ``<= t`` (causal).
    """

    def __init__(self, chan_in: int, chan_out: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.time_causal_padding, mode="replicate")
        return self.conv(x)


class _RefMotionEncoderTC(nn.Module):
    """Reference WAN MotionEncoder_tc (local branch only — ``need_global=False``).

    Inlined from ``wan/modules/s2v/auxi_blocks.py:MotionEncoder_tc``. Three
    causal-conv stages with stride (1, 2, 2) → 4× temporal downsample; outputs
    ``[B, T/4, num_heads, hidden_dim]`` plus a learned padding token appended
    as an extra "head" along the token axis, giving the final
    ``[B, T/4, num_heads + 1, hidden_dim]`` audio tokens.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.conv1_local = _RefCausalConv1d(in_dim, hidden_dim // 4 * num_heads, 3, stride=1)
        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6)
        self.conv2 = _RefCausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2)
        self.norm2 = nn.LayerNorm(hidden_dim // 2, elementwise_affine=False, eps=1e-6)
        self.conv3 = _RefCausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.act = nn.SiLU()
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, in_dim]
        b, t, _ = x.shape
        x = x.permute(0, 2, 1)  # [B, in_dim, T]
        x = self.conv1_local(x)  # [B, (hidden/4)*num_heads, T]
        # Head split: [B, num_heads, hidden/4, T] → flatten batch+heads, swap to BTC.
        x = x.reshape(b, self.num_heads, self.hidden_dim // 4, t).permute(0, 1, 3, 2)
        x = x.reshape(b * self.num_heads, t, self.hidden_dim // 4)
        x = self.act(self.norm1(x))
        x = x.permute(0, 2, 1)  # [(B*n), hidden/4, T]
        x = self.conv2(x)
        x = x.permute(0, 2, 1)  # [(B*n), T/2, hidden/2]
        x = self.act(self.norm2(x))
        x = x.permute(0, 2, 1)  # [(B*n), hidden/2, T/2]
        x = self.conv3(x)
        x = x.permute(0, 2, 1)  # [(B*n), T/4, hidden]
        x = self.act(self.norm3(x))
        # Unmerge heads: [B, num_heads, T/4, hidden] → permute to [B, T/4, num_heads, hidden].
        x = x.reshape(b, self.num_heads, -1, self.hidden_dim).permute(0, 2, 1, 3)
        # Append the learned padding token as an extra "head" on the token axis.
        pad = self.padding_tokens.expand(b, x.shape[1], 1, self.hidden_dim)
        return torch.cat([x, pad], dim=2)  # [B, T/4, num_heads + 1, hidden]


class _RefCausalAudioEncoder(nn.Module):
    """Reference WAN CausalAudioEncoder (need_global=False branch).

    Inlined from ``wan/modules/s2v/audio_utils.py:CausalAudioEncoder``. Per-
    layer learned weighted sum of the wav2vec2 hidden-state stack, then the
    causal :class:`_RefMotionEncoderTC` distills to per-video-frame audio
    tokens.
    """

    def __init__(self, *, dim: int, num_layers: int, out_dim: int, num_token: int) -> None:
        super().__init__()
        self.encoder = _RefMotionEncoderTC(in_dim=dim, hidden_dim=out_dim, num_heads=num_token)
        # Reference initializes weights to 0.01 (silu later); the actual
        # value doesn't matter for parity since we copy state_dict to TT.
        self.weights = nn.Parameter(torch.ones(1, num_layers, 1, 1) * 0.01)
        self.act = nn.SiLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, num_layers, dim, T]
        w = self.act(self.weights)
        weighted = (features * w / w.sum(dim=1, keepdim=True)).sum(dim=1)  # [B, dim, T]
        return self.encoder(weighted.permute(0, 2, 1))  # [B, T/4, num_heads + 1, hidden]


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            (2, 4),
            (2, 4),
            1,
            0,
            2,
            line_params,
            ttnn.Topology.Linear,
            False,
            id="bh_2x4sp1tp0",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_prepare_audio_emb_parity(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    """End-to-end audio path: CausalAudioEncoder + slice off motion_frames[1].

    Compares the ``[B, T_video, num_token+1, out_dim]`` tensor that the
    reference stores in ``self.merged_audio_emb`` (and that we flatten to
    ``[1, B, T_video*(num_token+1), out_dim]`` for cross-attn K/V).
    """
    torch.manual_seed(0)

    # ---- Build reference (CPU, inlined pytorch — see top of file) ----
    ref = (
        _RefCausalAudioEncoder(dim=AUDIO_DIM, num_layers=NUM_LAYERS, out_dim=OUT_DIM, num_token=NUM_TOKEN)
        .eval()
        .to(torch.float32)
    )
    logger.info(f"Reference CausalAudioEncoder built: dim={AUDIO_DIM}, num_layers={NUM_LAYERS}, out_dim={OUT_DIM}")

    # ---- Build ours (on device) ----
    tt = CausalAudioEncoder(
        dim=AUDIO_DIM,
        num_layers=NUM_LAYERS,
        out_dim=OUT_DIM,
        num_token=NUM_TOKEN,
        need_global=False,
        mesh_device=mesh_device,
    )

    # Weight transfer ref -> ours. Names line up 1:1 modulo our CausalConv1d
    # _prepare_torch_state handling of ``conv.weight`` -> ``weight`` (unsqueeze
    # spatial 1×1). The aggregation weights and padding_tokens transfer raw.
    ref_sd = ref.state_dict()
    incompat = tt.load_torch_state_dict(ref_sd, strict=False)
    logger.info(
        f"TT CausalAudioEncoder load: missing={len(incompat.missing_keys)} "
        f"unexpected={len(incompat.unexpected_keys)}"
    )
    if incompat.missing_keys:
        logger.warning(f"missing keys: {incompat.missing_keys[:5]}")
    if incompat.unexpected_keys:
        logger.warning(f"unexpected keys: {incompat.unexpected_keys[:5]}")

    # ---- Same audio input on both sides ----
    audio_input_torch = torch.randn(1, NUM_LAYERS, AUDIO_DIM, T_AUDIO, dtype=torch.float32)

    # Replicate the reference's left-pad-by-motion_frames[0] step
    # (wan/modules/s2v/model_s2v.py:683):
    pre = audio_input_torch[..., :1].expand(-1, -1, -1, MOTION_FRAMES[0])
    audio_input_padded = torch.cat([pre, audio_input_torch], dim=-1)

    # ---- Reference forward ----
    with torch.no_grad():
        ref_out = ref(audio_input_padded)  # [B, T, num_token+1, out_dim]
        ref_merged = ref_out[:, MOTION_FRAMES[1] :, :]  # slice off motion-latent prefix
    logger.info(f"Reference merged_audio_emb: shape={tuple(ref_merged.shape)}")

    # ---- Our forward ----
    tt_out_dev = tt(audio_input_padded)
    tt_out_torch = local_device_to_torch(tt_out_dev)  # [B, T, num_token+1, out_dim]
    # Slice off motion-latent prefix (matches our prepare_audio_emb).
    tt_merged = tt_out_torch[:, MOTION_FRAMES[1] :, :, :]
    logger.info(f"TT merged_audio_emb: shape={tuple(tt_merged.shape)}")

    # ---- Compare ----
    assert (
        tt_merged.shape == ref_merged.shape
    ), f"shape mismatch: tt={tuple(tt_merged.shape)} ref={tuple(ref_merged.shape)}"
    assert_quality(tt_merged.float(), ref_merged.float(), pcc=0.99)
