# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint access for Qwen3-TTS: the config, the weight layout, and the reader.

These tests never open a device. They guard the seam that silently rots when an upstream
checkpoint moves: the speaker encoder's tensor names and shapes are derived from
config.json alone, so a weights bump that stops matching its own config fails here rather
than surfacing as a PCC miss later.

The checkpoint (3.6 GB at 1.7B, 1.8 GB at 0.6B) is fetched once and cached by
huggingface_hub. Nothing is skipped when it is absent: a skip would turn an unreachable
checkpoint into a green run.

Run:
    pytest -svv models/demos/audio/qwen3_tts/tests/test_checkpoint_loading.py
"""

import pytest
import torch

from models.demos.audio.qwen3_tts import weights

# What the pinned revision declares. A checkpoint bump that changes any of these changes
# the TTNN graph, so the test names them rather than reading them back from the file.
EXPECTED_CONFIG = {
    "mel_dim": 128,
    "enc_dim": 2048,
    "enc_channels": [512, 512, 512, 512, 1536],
    "enc_kernel_sizes": [5, 3, 3, 3, 1],
    "enc_dilations": [1, 2, 3, 4, 1],
    "enc_attention_channels": 128,
    "enc_res2net_scale": 8,
    "enc_se_channels": 128,
    "sample_rate": 24000,
}

# The one field the two sizes disagree on: the embedding is as wide as the talker.
EXPECTED_ENC_DIM = {"1b7": 2048, "0b6": 1024}

# Which moves only the output projection, `fc`: 3072 x enc_dim weights plus enc_dim biases.
EXPECTED_SPEAKER_PARAMETERS = {"1b7": 12_001_088, "0b6": 8_854_336}


@pytest.fixture(scope="module")
def speaker_config():
    return weights.speaker_encoder_config()


def test_speaker_config_matches_the_pinned_revision(speaker_config):
    assert speaker_config == dict(EXPECTED_CONFIG, enc_dim=EXPECTED_ENC_DIM[weights.model_size()])


def test_speaker_embedding_width_matches_the_talker(speaker_config):
    """The encoder's output drops straight into the talker's sequence, so the widths must
    agree; a projection would have to be built if they ever stop matching."""
    talker = weights.model_config()["talker_config"]
    assert speaker_config["enc_dim"] == talker["hidden_size"]


def test_mel_front_end_feeds_the_declared_bin_count(speaker_config):
    assert weights.SPEAKER_MEL["num_mels"] == speaker_config["mel_dim"]
    assert weights.SPEAKER_MEL["sampling_rate"] == speaker_config["sample_rate"]


def test_checkpoint_holds_the_two_expected_prefixes():
    assert weights.prefixes_in_file() == {"speaker_encoder", "talker"}


def test_speaker_tensor_shapes_match_the_config(speaker_config):
    expected = weights.expected_speaker_shapes(speaker_config)
    found = weights.speaker_shapes_in_file()

    assert sorted(found) == sorted(expected), "checkpoint and config disagree on which tensors exist"
    mismatched = {name: (found[name], expected[name]) for name in expected if found[name] != expected[name]}
    assert not mismatched, f"shape mismatch (file, config): {mismatched}"


def test_the_derived_layout_covers_every_block(speaker_config):
    """Guards the derivation itself: one TDNN, three SE-Res2Net blocks, aggregation,
    pooling and the projection, with seven grouped convolutions inside each block."""
    names = set(weights.expected_speaker_shapes(speaker_config))
    scale = speaker_config["enc_res2net_scale"]
    for block in (1, 2, 3):
        grouped = {n for n in names if n.startswith(f"blocks.{block}.res2net_block.")}
        assert len(grouped) == 2 * (scale - 1), f"block {block} has {len(grouped) // 2} grouped convolutions"
    assert "blocks.4.tdnn1.conv.weight" not in names, "the last channel entry sizes aggregation, not a block"
    assert {"mfa.conv.weight", "asp.tdnn.conv.weight", "asp.conv.weight", "fc.weight"} <= names


def test_loaded_weights_are_finite_fp32(speaker_config):
    state = weights.load_speaker_state()
    expected = weights.expected_speaker_shapes(speaker_config)

    assert sorted(state) == sorted(expected)
    assert all(t.dtype is torch.float32 for t in state.values())
    assert all(torch.isfinite(t).all() for t in state.values())

    params = sum(t.numel() for t in state.values())
    assert params == EXPECTED_SPEAKER_PARAMETERS[weights.model_size()], f"speaker encoder has {params} parameters"


def test_loading_the_speaker_leaves_the_talker_alone():
    """The reader names its keys; it must not drag in the talker."""
    state = weights.load_speaker_state()
    assert not any(name.startswith(weights.TALKER_PREFIX) for name in state)
    assert weights.TALKER_PREFIX.rstrip(".") in weights.prefixes_in_file()


def test_bf16_is_available_without_conversion(speaker_config):
    state = weights.load_speaker_state(dtype=None)
    assert all(t.dtype is torch.bfloat16 for t in state.values()), "the checkpoint is stored in bf16"


# ── the derivation refuses a config it cannot build (no checkpoint needed) ──


def test_mismatched_config_lists_are_refused(expect_error):
    cfg = dict(EXPECTED_CONFIG, enc_dilations=[1, 2, 3])
    with expect_error(ValueError, "same length"):
        weights.expected_speaker_shapes(cfg)


def test_channels_that_do_not_divide_by_the_res2net_scale_are_refused(expect_error):
    cfg = dict(EXPECTED_CONFIG, enc_channels=[512, 500, 512, 512, 1536])
    with expect_error(ValueError, "divisible"):
        weights.expected_speaker_shapes(cfg)
