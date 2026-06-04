# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Wiring smoke-tests for the text-to-audio demo. We don't pull in the real
HuggingFace AudioX checkpoint here — the heavy bits (T5, CLIP, DiT layers)
have their own parity tests. These tests check the demo helpers stitch the
pieces together with the right shapes and that the empty-feat shortcuts fire
on the text-to-audio path."""

import torch

from models.experimental.audiox.demo import demo as demo_mod


def test_metadata_batch_zero_inputs_have_text_only_shapes():
    """Default metadata for text-to-audio: zero video tensor and zero audio
    tensor at the AudioX HF config sizes, plus the user prompt string."""
    batch = demo_mod._build_metadata_batch("hello")
    assert len(batch) == 1
    assert batch[0]["text_prompt"] == "hello"

    fps = 5
    duration = demo_mod._HF_CONFIG["duration_seconds"]
    assert batch[0]["video_prompt"].shape == (1, fps * duration, 3, 224, 224)
    assert torch.all(batch[0]["video_prompt"] == 0)

    samples = demo_mod._HF_CONFIG["sample_rate"] * duration
    assert batch[0]["audio_prompt"].shape == (1, 2, samples)
    assert torch.all(batch[0]["audio_prompt"] == 0)


def test_metadata_batch_preserves_provided_visual_prompt():
    video = torch.randn(1, 50, 3, 224, 224)
    batch = demo_mod._build_metadata_batch_with_inputs("hello", video_prompt=video)
    assert batch[0]["video_prompt"] is video
    assert batch[0]["text_prompt"] == "hello"


def test_metadata_batch_preserves_provided_audio_prompt():
    audio = torch.randn(1, 2, 32)
    batch = demo_mod._build_metadata_batch_with_inputs("hello", audio_prompt=audio)
    assert batch[0]["audio_prompt"] is audio
    assert batch[0]["text_prompt"] == "hello"


def test_metadata_batch_uses_requested_duration_for_empty_inputs():
    batch = demo_mod._build_metadata_batch_with_inputs("hello", duration_seconds=30)
    assert batch[0]["video_prompt"].shape == (1, 150, 3, 224, 224)
    assert batch[0]["audio_prompt"].shape == (1, 2, demo_mod._HF_CONFIG["sample_rate"] * 30)


def test_resolve_audio_prompt_prefers_direct_tensor():
    audio = torch.randn(1, 2, 32)
    resolved = demo_mod._resolve_audio_prompt(None, audio)
    assert resolved is audio


def test_make_cross_attn_cond_concats_in_audiox_order():
    """Order matters for cross-attn cond: video, text, audio (AudioX uses this)."""
    multi_out = {
        "video_prompt": (torch.full((1, 5, 768), 1.0), None),
        "text_prompt": (torch.full((1, 7, 768), 2.0), None),
        "audio_prompt": (torch.full((1, 3, 768), 3.0), None),
    }
    cat = demo_mod._make_cross_attn_cond(multi_out)
    assert cat.shape == (1, 5 + 7 + 3, 768)
    # First 5 rows are video (1.0), next 7 text (2.0), last 3 audio (3.0).
    assert torch.all(cat[:, :5] == 1.0)
    assert torch.all(cat[:, 5:12] == 2.0)
    assert torch.all(cat[:, 12:] == 3.0)


def test_hf_config_invariants():
    """Pin the HF config values the demo relies on. Building a full-shape DiT
    + decoder in PyTorch is too expensive for a unit test — the per-module
    PCC tests already cover construction at HF dims."""
    cfg = demo_mod._HF_CONFIG
    # downsample = prod(decoder strides), and total samples must divide cleanly
    # into latent frames at this rate.
    prod_strides = 1
    for s in cfg["decoder_strides"]:
        prod_strides *= s
    assert cfg["downsample"] == prod_strides
    assert cfg["embed_dim"] % cfg["num_heads"] == 0
    assert cfg["sample_rate"] * cfg["duration_seconds"] > 0
    assert cfg["output_sample_rate"] == 16000


def test_parse_args_requires_at_least_one_conditioner_input():
    try:
        demo_mod._parse_args(["--checkpoint", "/tmp/fake.safetensors"])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("expected argparse failure when no prompt/video/image is supplied")


def test_parse_args_accepts_audio_only():
    args = demo_mod._parse_args(["--checkpoint", "/tmp/fake.safetensors", "--audio", "/tmp/fake.wav"])
    assert str(args.audio) == "/tmp/fake.wav"


def test_parse_args_accepts_duration_override():
    args = demo_mod._parse_args(
        ["--checkpoint", "/tmp/fake.safetensors", "--prompt", "hello", "--duration-seconds", "30"]
    )
    assert args.duration_seconds == 30
