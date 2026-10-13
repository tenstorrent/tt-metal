# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Switching a test module onto a sibling release, at whichever size the suite runs.

The suite runs on one ambient checkpoint, a Base release at 1.7B or 0.6B. Some modules need
CustomVoice or VoiceDesign instead, and they must get the one at the ambient size: a 0.6B
run that quietly tested the 1.7B CustomVoice would pass while proving nothing about 0.6B.
Where no sibling exists at that size (there is no 0.6B VoiceDesign) the module skips.
"""

import os

import pytest

from models.demos.audio.qwen3_tts import frontend, weights


def clear_caches():
    """Drop every cached view of the checkpoint, so switching does not leak across tests.

    `frontend` caches too, and forgetting it let a checkpoint switch change what a later
    tokenizer test saw: CustomVoice's `codec_language_id` carries 12 entries, the ten
    languages plus `beijing_dialect` and `sichuan_dialect` for its dialect speakers.
    """
    for cached in (
        weights.checkpoint_dir,
        weights._model_config_json,
        weights.codec_dir,
        weights._codec_config_json,
        frontend.tokenizer,
        frontend.special_tokens,
        frontend.language_ids,
    ):
        cached.cache_clear()


def use_release(kind):
    """Point `$QWEN3_TTS_CKPT` at the `kind` release of the ambient size, for one module.

    A generator for a module-scoped fixture: yields the checkpoint directory and puts the
    environment back afterwards.
    """
    release = weights.sibling_repo(kind)
    if release is None:
        pytest.skip(f"no {kind} release at {weights.model_size()}")
    path = weights.fetch_release(release[0])

    previous = os.environ.get("QWEN3_TTS_CKPT")
    os.environ["QWEN3_TTS_CKPT"] = path
    clear_caches()
    try:
        yield path
    finally:
        if previous is None:
            os.environ.pop("QWEN3_TTS_CKPT", None)
        else:
            os.environ["QWEN3_TTS_CKPT"] = previous
        clear_caches()


def hidden_width():
    """The talker's width, which every prompt position and speaker vector shares."""
    return weights.talker_config()["hidden_size"]
