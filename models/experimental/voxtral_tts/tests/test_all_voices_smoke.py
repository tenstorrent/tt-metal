# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every voice preset runs, on a sentence in its own language.

The checkpoint ships **20** voices and `prompt_fixture.json` exercises **13**, so seven were never
run by any test: de_female, es_male, fr_male, hi_male, it_female, nl_female, pt_female. That is not
cosmetic -- the placeholder count is voice-specific (ar_male 67 rows, casual_female 214), so an
untested voice is an untested prompt geometry, and a wrong placeholder count shifts every
downstream position.

This is the analogue of the sibling xtts_v2 suite's `test_all_languages_smoke.py`: breadth, not
accuracy. It asserts the request completes and emits structurally valid codes through the TRACED
path -- accuracy per block is the PCC files' job.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_all_voices_smoke.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

pytestmark = pytest.mark.slow

from models.experimental.voxtral_tts.reference.voxtral_common_ref import END_AUDIO_ID  # noqa: E402
from models.experimental.voxtral_tts.tests.reference_helpers import (  # noqa: E402
    all_voices,
    corpus_embeds,
)
from models.experimental.voxtral_tts.tests.sentence_corpus import (  # noqa: E402
    SENTENCES,
    first_sentence_for,
    lang_of,
)
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import (  # noqa: E402
    TtVoxtralPipeline,
    open_device,
)

N_FRAMES = 4          # breadth, not depth: 20 voices x a few frames each
N_CODES = 37          # 1 semantic + 36 acoustic


@pytest.fixture(scope="module")
def pipe():
    d = open_device()
    p = TtVoxtralPipeline(d)
    yield p
    p.close()
    ttnn.close_device(d)


def test_corpus_covers_every_voices_language():
    """A voice whose language has no sentences would silently fall back to English prose, which
    would then be scored as that language later. Catch it here instead."""
    missing = sorted({
        v.split("_")[0] for v in all_voices()
        if "_" in v and len(v.split("_")[0]) == 2 and v.split("_")[0] not in SENTENCES
    })
    assert not missing, f"voices exist for languages with no corpus sentences: {missing}"


def test_every_voice_has_a_sentence():
    for v in all_voices():
        assert first_sentence_for(v).strip(), f"{v} ({lang_of(v)}) has no sentence"


@pytest.mark.parametrize("voice", all_voices())
def test_voice_smoke(pipe, voice):
    """Prompt assembly through Block 1 through Block 2, on the traced path, for one voice."""
    embeds = corpus_embeds(first_sentence_for(voice), voice, pipe.wb)
    pipe.backbone.reset()
    frames, _, _ = pipe.generate(embeds, max_frames=N_FRAMES, seed=0, verbose=False)

    assert frames.ndim == 2 and frames.shape[1] == N_CODES, \
        f"{voice}: frames shaped {tuple(frames.shape)}, expected [T, {N_CODES}]"
    assert frames.shape[0] >= 1, f"{voice}: no frames emitted"
    assert not bool((frames[:, 0] == END_AUDIO_ID).any()), \
        f"{voice}: [END_AUDIO] leaked into the returned frames"
    assert bool((frames >= 0).all()), f"{voice}: negative code id"
    print(f"  {voice:>16} ({lang_of(voice)}) P={embeds.shape[1]:>4} -> {frames.shape[0]} frames")
