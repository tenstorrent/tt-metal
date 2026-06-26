# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Wiring smoke-tests for the text-to-audio demo. We don't pull in the real
HuggingFace AudioX checkpoint here — the heavy bits (T5, CLIP, DiT layers)
have their own parity tests. These tests check the demo helpers stitch the
pieces together with the right shapes and that the empty-feat shortcuts fire
on the text-to-audio path."""

import torch
from torch import nn
from torch.nn.utils import weight_norm

from models.experimental.audiox.demo import demo as demo_mod
from models.experimental.audiox.demo import tt_runtime as tt_runtime_mod


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


def test_tt_open_kwargs_read_optional_env(monkeypatch):
    monkeypatch.setenv("AUDIOX_TT_L1_SMALL_SIZE", "0x4000")
    monkeypatch.setenv("AUDIOX_TT_TRACE_REGION_SIZE", "0x800000")
    monkeypatch.setenv("AUDIOX_TT_NUM_COMMAND_QUEUES", "2")
    monkeypatch.setenv("AUDIOX_TT_WORKER_L1_SIZE", "983040")

    kwargs = tt_runtime_mod.tt_open_kwargs_from_env()

    assert kwargs == {
        "l1_small_size": 0x4000,
        "trace_region_size": 0x800000,
        "num_command_queues": 2,
        "worker_l1_size": 983040,
    }


def test_tt_runtime_apply_and_restore_env(monkeypatch):
    monkeypatch.setenv("AUDIOX_TT_TRACE_REGION_SIZE", "123")

    previous = tt_runtime_mod.apply_tt_env_overrides(
        open_mode="direct",
        local_mesh_width=2,
        num_command_queues=2,
        conv_transpose_input_chunk=65536,
    )
    try:
        assert tt_runtime_mod.os.environ["AUDIOX_TT_OPEN_MODE"] == "direct"
        assert tt_runtime_mod.os.environ["AUDIOX_TT_LOCAL_MESH_WIDTH"] == "2"
        assert tt_runtime_mod.os.environ["AUDIOX_TT_NUM_COMMAND_QUEUES"] == "2"
        assert tt_runtime_mod.os.environ["AUDIOX_TT_TRACE_REGION_SIZE"] == "123"
        assert tt_runtime_mod.os.environ["AUDIOX_TT_CONV_TRANSPOSE_INPUT_CHUNK"] == "65536"
    finally:
        tt_runtime_mod.restore_tt_env(previous)

    assert tt_runtime_mod.os.environ.get("AUDIOX_TT_TRACE_REGION_SIZE") == "123"
    assert "AUDIOX_TT_OPEN_MODE" not in tt_runtime_mod.os.environ
    assert "AUDIOX_TT_LOCAL_MESH_WIDTH" not in tt_runtime_mod.os.environ
    assert "AUDIOX_TT_NUM_COMMAND_QUEUES" not in tt_runtime_mod.os.environ
    assert "AUDIOX_TT_CONV_TRANSPOSE_INPUT_CHUNK" not in tt_runtime_mod.os.environ


def test_tt_demo_main_uses_requested_device_id(monkeypatch, tmp_path):
    from models.experimental.audiox.demo import tt_demo as tt_demo_mod

    opened = []
    closed = []

    monkeypatch.setattr(tt_demo_mod, "apply_tt_env_overrides", lambda **_kwargs: {})
    monkeypatch.setattr(tt_demo_mod, "restore_tt_env", lambda _previous: None)
    monkeypatch.setattr(tt_demo_mod, "open_tt_device", lambda *, device_id: opened.append(device_id) or "device")
    monkeypatch.setattr(tt_demo_mod, "close_tt_device", lambda device: closed.append(device))
    monkeypatch.setattr(tt_demo_mod, "run_tt_demo", lambda **_kwargs: tmp_path / "out.wav")

    rc = tt_demo_mod.main(
        [
            "--checkpoint",
            "/tmp/fake.safetensors",
            "--prompt",
            "hello",
            "--output",
            str(tmp_path / "out.wav"),
            "--tt-device-id",
            "3",
        ]
    )

    assert rc == 0
    assert opened == [3]
    assert closed == ["device"]


def test_tt_demo_cpu_decode_env_default_and_override(monkeypatch):
    from models.experimental.audiox.demo import tt_demo as tt_demo_mod

    monkeypatch.delenv("AUDIOX_TT_CPU_DECODE", raising=False)
    assert tt_demo_mod._should_use_cpu_decode() is True

    monkeypatch.setenv("AUDIOX_TT_CPU_DECODE", "0")
    assert tt_demo_mod._should_use_cpu_decode() is False


def test_tt_demo_cpu_decoder_fuses_weight_norm(monkeypatch):
    from models.experimental.audiox.demo import tt_demo as tt_demo_mod

    decoder = nn.Sequential(weight_norm(nn.Conv1d(2, 2, 3, padding=1)))
    monkeypatch.setattr(tt_demo_mod, "_build_decoder", lambda: decoder)
    monkeypatch.setattr(tt_demo_mod, "load_into", lambda module, state_dict, label: None)

    fused = tt_demo_mod._build_cpu_decoder_fused({})

    assert fused is decoder
    assert not hasattr(fused[0], "weight_g")
    assert not hasattr(fused[0], "weight_v")
