# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the MiniMax-H3 adapter settings resolver: argument precedence, environment
fallbacks and the checkpoint defaults. These are the knobs a serving deployment reaches the pipeline
through, and every one of them is silent when wrong -- a mis-scaled or mis-shifted adapter produces a
video, not an error."""

from pathlib import Path

import pytest

from models.tt_dit.pipelines.minimax_h3 import weights_minimax_h3 as weights

DEFAULTS = {"default_video_shift": 12.0, "default_audio_shift": 3.0}


@pytest.fixture
def clean_env(monkeypatch):
    for name in (weights.LORA_PATH_ENV, weights.LORA_STRENGTH_ENV, weights.VIDEO_SHIFT_ENV, weights.AUDIO_SHIFT_ENV):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_nothing_set_runs_the_base_model_at_checkpoint_shifts(clean_env):
    settings = weights.resolve_adapter_settings(**DEFAULTS)
    assert settings == weights.AdapterSettings(lora_path=None, lora_strength=1.0, video_shift=12.0, audio_shift=3.0)


def test_environment_fills_every_unset_field(clean_env):
    clean_env.setenv(weights.LORA_PATH_ENV, "/adapters/turbo.safetensors")
    clean_env.setenv(weights.LORA_STRENGTH_ENV, "0.5")
    clean_env.setenv(weights.VIDEO_SHIFT_ENV, "6")
    clean_env.setenv(weights.AUDIO_SHIFT_ENV, "3.0")
    settings = weights.resolve_adapter_settings(**DEFAULTS)
    assert settings.lora_path == Path("/adapters/turbo.safetensors")
    assert settings.lora_strength == 0.5
    assert (settings.video_shift, settings.audio_shift) == (6.0, 3.0)


def test_explicit_arguments_win_over_the_environment(clean_env):
    clean_env.setenv(weights.LORA_PATH_ENV, "/adapters/env.safetensors")
    clean_env.setenv(weights.LORA_STRENGTH_ENV, "0.5")
    clean_env.setenv(weights.VIDEO_SHIFT_ENV, "6")
    settings = weights.resolve_adapter_settings(
        lora_path="/adapters/arg.safetensors", lora_strength=2.0, video_shift=12.0, **DEFAULTS
    )
    assert settings.lora_path == Path("/adapters/arg.safetensors")
    assert settings.lora_strength == 2.0
    assert settings.video_shift == 12.0


def test_shift_overrides_apply_without_an_adapter(clean_env):
    clean_env.setenv(weights.VIDEO_SHIFT_ENV, "6")
    settings = weights.resolve_adapter_settings(**DEFAULTS)
    assert settings.lora_path is None
    assert settings.video_shift == 6.0


def test_empty_path_variable_means_no_adapter(clean_env):
    clean_env.setenv(weights.LORA_PATH_ENV, "")
    assert weights.resolve_adapter_settings(**DEFAULTS).lora_path is None


@pytest.mark.parametrize("field", ["lora_strength", "video_shift", "audio_shift"])
def test_non_positive_values_are_rejected(clean_env, field, expect_error):
    with expect_error(ValueError, field):
        weights.resolve_adapter_settings(**{field: 0.0}, **DEFAULTS)
