# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Do utterances stop, over enough seeds to mean something?

The failure no per-position measurement catches: in non-streaming mode the talker gets no
text signal after the prefill, so a run that loses its place keeps emitting.

**One seed proves nothing.** Frames per word for a four-word sentence, eight seeds, three
builds of this model: 32.8 38.0 4.0 6.0 4.8 4.8 9.2 3.5 before tuning, 17.0 8.5 3.8 5.0
4.0 4.5 8.0 4.0 after, 10.0 8.5 5.2 5.2 4.5 4.2 6.5 3.8 with block-float MLP weights.
Most seeds stop promptly and which ones wander moves with the last bits, so this counts
wandering seeds rather than judging one.

Needs the CustomVoice checkpoint, like `test_pipeline.py`.
"""

import os

import pytest

from models.demos.audio.qwen3_tts import frontend, weights
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import Qwen3TTSPipeline

CUSTOM_VOICE_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEVICE_PARAMS = [{"l1_small_size": 65536, "trace_region_size": 90_000_000}]

SPEAKER = "ryan"
LANGUAGE = "English"
MAX_FRAMES = 400
SEEDS = tuple(range(8))
CASES = (
    "The kettle is on.",
    "The kettle is on, and the rain has not let up since yesterday morning.",
)

# Frames per word past which a run has stopped reading the text.
MAX_FRAMES_PER_WORD = 12.0

# Seeds per sentence allowed over that ceiling. Reaching the frame cap is never allowed.
MAX_WANDERING_SEEDS = 2


def _clear_caches():
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


@pytest.fixture(scope="module", autouse=True)
def custom_voice_checkpoint():
    from huggingface_hub import snapshot_download

    path = snapshot_download(CUSTOM_VOICE_REPO)
    previous = os.environ.get("QWEN3_TTS_CKPT")
    os.environ["QWEN3_TTS_CKPT"] = path
    _clear_caches()
    yield path
    if previous is None:
        os.environ.pop("QWEN3_TTS_CKPT", None)
    else:
        os.environ["QWEN3_TTS_CKPT"] = previous
    _clear_caches()


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_most_seeds_stop_promptly_and_none_runs_to_the_cap(device):
    """Eight seeds, two lengths, codes only: the codec would compile a program per length."""
    pipeline = Qwen3TTSPipeline(device, max_frames=MAX_FRAMES)
    failures = []
    for text in CASES:
        words = len(text.split())
        per_word, capped, wandering = [], [], 0
        for seed in SEEDS:
            pipeline.reseed(seed)
            frames = pipeline.codes(text, speaker=SPEAKER, language=LANGUAGE).shape[0]
            per_word.append(round(frames / words, 1))
            if frames >= MAX_FRAMES:
                capped.append(seed)
            elif frames / words > MAX_FRAMES_PER_WORD:
                wandering += 1
        print(f"  {words:2d} words: frames per word {per_word}, worst {max(per_word)}, {wandering} wandering")

        if capped:
            failures.append(f"{words} words: seeds {capped} hit the {MAX_FRAMES}-frame cap")
        if wandering > MAX_WANDERING_SEEDS:
            failures.append(f"{words} words: {wandering} of {len(SEEDS)} seeds over {MAX_FRAMES_PER_WORD} per word")

    assert not failures, "; ".join(failures)
