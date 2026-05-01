# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from models.experimental.audiox.reference.conditioners import (
    AudioAutoencoderConditioner,
    CLIPConditioner,
    Conditioner,
    MultiConditioner,
)
from models.experimental.audiox.reference.temptransformer import SATransformer


class _StubConditioner(Conditioner):
    """Records calls and returns a fixed-shape (embed, mask) tuple. Lets us
    test MultiConditioner without pulling in HF models."""

    def __init__(self, output_dim: int, seq_len: int):
        super().__init__(output_dim, output_dim)
        self.seq_len = seq_len
        self.last_inputs = None

    def forward(self, inputs, device):
        self.last_inputs = inputs
        batch = len(inputs)
        return (
            torch.zeros(batch, self.seq_len, self.output_dim),
            torch.ones(batch, self.seq_len, dtype=torch.bool),
        )


def test_multi_conditioner_dispatches_each_key():
    text_cond = _StubConditioner(output_dim=8, seq_len=4)
    video_cond = _StubConditioner(output_dim=8, seq_len=2)

    multi = MultiConditioner({"text_prompt": text_cond, "video_prompt": video_cond})

    batch = [
        {"text_prompt": "hello", "video_prompt": torch.zeros(1, 1, 3, 4, 4)},
        {"text_prompt": "world", "video_prompt": torch.zeros(1, 1, 3, 4, 4)},
    ]
    out = multi(batch, "cpu")

    assert set(out.keys()) == {"text_prompt", "video_prompt"}
    assert out["text_prompt"][0].shape == (2, 4, 8)
    assert out["video_prompt"][0].shape == (2, 2, 8)
    assert text_cond.last_inputs == ["hello", "world"]


def test_multi_conditioner_falls_back_to_default_key():
    cond = _StubConditioner(output_dim=8, seq_len=4)
    multi = MultiConditioner({"text_prompt": cond}, default_keys={"text_prompt": "prompt"})
    out = multi([{"prompt": "hi"}], "cpu")
    assert out["text_prompt"][0].shape == (1, 4, 8)
    assert cond.last_inputs == ["hi"]


def test_multi_conditioner_unwraps_singleton_list():
    cond = _StubConditioner(output_dim=8, seq_len=4)
    multi = MultiConditioner({"text_prompt": cond})
    multi([{"text_prompt": ["only"]}], "cpu")
    assert cond.last_inputs == ["only"]


def test_multi_conditioner_raises_on_missing_key():
    cond = _StubConditioner(output_dim=8, seq_len=4)
    multi = MultiConditioner({"text_prompt": cond})
    with pytest.raises(ValueError, match="text_prompt"):
        multi([{"other": "x"}], "cpu")


def test_conditioner_proj_out_identity_when_dims_match():
    c = Conditioner(dim=8, output_dim=8)
    assert isinstance(c.proj_out, nn.Identity)


def test_conditioner_proj_out_linear_when_dims_differ():
    c = Conditioner(dim=4, output_dim=8)
    assert isinstance(c.proj_out, nn.Linear)
    assert c.proj_out.in_features == 4
    assert c.proj_out.out_features == 8


def test_sa_transformer_shape_and_residual():
    torch.manual_seed(0)
    t = SATransformer(dim=16, depth=2, heads=4, dim_head=4, mlp_dim=64).eval()
    x = torch.randn(2, 8, 16)
    with torch.no_grad():
        y = t(x)
    assert y.shape == x.shape
    # Final LayerNorm zero-mean check (layer norm output has ~0 mean per token).
    assert y.mean(dim=-1).abs().max().item() < 1e-5


def test_clip_conditioner_empty_path_skips_encoder():
    """All-zero video tensor -> empty_visual_feat shortcut, never touches HF CLIP."""
    cond = CLIPConditioner(output_dim=768).eval()
    # Initialize empty_visual_feat to a recognizable value so we can check it propagated.
    with torch.no_grad():
        cond.empty_visual_feat.fill_(0.5)

    # AudioX video tensor: [1, T=duration*fps=50, C=3, H=224, W=224]; all zeros.
    video = torch.zeros(1, 50, 3, 224, 224)
    with torch.no_grad():
        out, mask = cond([video], "cpu")

    assert out.shape == (1, 128, 768)
    assert torch.allclose(out, torch.full_like(out, 0.5))
    assert mask.shape == (1, 1)
    # visual_encoder_model must remain None — the empty path never built it.
    assert cond.visual_encoder_model is None


def test_clip_conditioner_param_names_match_upstream_convention():
    """Pretrained loader needs these names to remap upstream weights."""
    cond = CLIPConditioner(output_dim=768)
    names = set(dict(cond.named_parameters()).keys())
    assert "empty_visual_feat" in names
    assert "proj.weight" in names and "proj_sync.weight" in names
    assert "sync_weight" in names
    assert "Temp_pos_embedding" in names
    assert "Temp_transformer.blocks.0.attn.to_qkv.weight" in names
    assert "Temp_transformer.norm.weight" in names


class _StubPretransform(nn.Module):
    """Stand-in for the Oobleck VAE encoder. We only need the duck-typed
    interface (encoded_channels + encode) to test the conditioner shape flow."""

    def __init__(self, channels: int, downsample: int):
        super().__init__()
        self.encoded_channels = channels
        self.downsample = downsample
        self.weight = nn.Parameter(torch.eye(channels))  # so .to(device) has something

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        # [B, C_in, T] -> [B, channels, T // downsample]
        b, _, t = audio.shape
        return torch.zeros(b, self.encoded_channels, t // self.downsample, device=audio.device)


def test_audio_autoencoder_conditioner_empty_path():
    pre = _StubPretransform(channels=64, downsample=4)
    cond = AudioAutoencoderConditioner(pretransform=pre, output_dim=768).eval()
    with torch.no_grad():
        cond.empty_audio_feat.fill_(0.25)

    audio = torch.zeros(2, 2, 1024)
    with torch.no_grad():
        out, mask = cond([audio[0:1], audio[1:2]], "cpu")

    assert out.shape == (2, 215, 768)
    assert torch.allclose(out, torch.full_like(out, 0.25))
    # Upstream quirk: mask shape is [B, output_dim], not [B, S].
    assert mask.shape == (2, 768)


def test_audio_autoencoder_conditioner_full_path_runs_encoder():
    pre = _StubPretransform(channels=64, downsample=2)
    cond = AudioAutoencoderConditioner(pretransform=pre, output_dim=32).eval()

    audio = torch.randn(1, 2, 64)  # non-zero -> hits encode path
    with torch.no_grad():
        out, _ = cond(audio, "cpu")

    # encode produces [B, 64, 32]; permute -> [B, 32, 64]; proj_out 64 -> 32 -> [B, 32, 32].
    assert out.shape == (1, 32, 32)
