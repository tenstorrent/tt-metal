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

Needs the CustomVoice checkpoint at the ambient size, like `test_pipeline.py`.
"""

import pytest

from models.demos.audio.qwen3_tts.tests.checkpoints import use_release
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import Qwen3TTSPipeline

DEVICE_PARAMS = [{"l1_small_size": 65536, "trace_region_size": 90_000_000}]

SPEAKER = "ryan"
LANGUAGE = "English"
MAX_FRAMES = 400
SEEDS = tuple(range(8))
# The four-word sentence alone: the docstring's wandering seeds are all on it, and the
# fourteen-word one was cut for CI time.
CASES = ("The kettle is on.",)

# Frames per word past which a run has stopped reading the text.
MAX_FRAMES_PER_WORD = 12.0

# Seeds per sentence allowed over that ceiling. Reaching the frame cap is never allowed.
MAX_WANDERING_SEEDS = 2


@pytest.fixture(scope="module", autouse=True)
def custom_voice_checkpoint():
    yield from use_release("custom_voice")


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_most_seeds_stop_promptly_and_none_runs_to_the_cap(device):
    """Eight seeds, codes only: the codec would compile a program per length."""
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
