# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every voice preset runs, on a sentence in its own language.

The placeholder count is voice-specific, so each preset is a different prompt geometry. Prefill is
compared against fp32 in two parts: the placeholder region, which the geometry controls, and the
text region, which varies with the token sequence.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_all_voices_smoke.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

pytestmark = pytest.mark.slow

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref  # noqa: E402
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (  # noqa: E402
    END_AUDIO_ID,
    N_LAYERS,
)
from models.experimental.voxtral_tts.reference.voxtral_tokenizer_ref import (  # noqa: E402
    TekkenTokenizer,
)
from models.experimental.voxtral_tts.tests.gates import compare_hidden  # noqa: E402
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
# The placeholder region is the voice's own geometry and is stable across texts; the text region
# varies with the token sequence, so it gets the looser pooled gate.
VOICE_PCC_PLACEHOLDERS = 0.999
VOICE_PCC_POOLED = 0.998


@pytest.fixture(scope="module")
def pipe():
    d = open_device()
    p = TtVoxtralPipeline(d)
    yield p
    p.close()
    ttnn.close_device(d)


def test_corpus_covers_every_voices_language():
    """A voice whose language has no corpus sentence would silently fall back to English."""
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
    """Prompt assembly through Blocks 1 and 2 for one voice, prefill checked against fp32."""
    embeds = corpus_embeds(first_sentence_for(voice), voice, pipe.wb)

    exp = bref.reference_forward(embeds, pipe.wb, n_layers=N_LAYERS)
    pipe.backbone.reset()
    got = pipe.backbone.prefill(embeds, last_only=False)
    m = compare_hidden(got, exp)
    # The preset rows sit after BOS + begin_audio, one per frame of this voice's reference clip.
    ph_end = 2 + TekkenTokenizer().n_audio_tokens(voice)
    m_ph = compare_hidden(got[:, 2:ph_end], exp[:, 2:ph_end])
    m_tx = compare_hidden(got[:, ph_end:], exp[:, ph_end:])
    assert m_ph["pcc"] > VOICE_PCC_PLACEHOLDERS, (
        f"{voice}: PLACEHOLDER region PCC {m_ph['pcc']:.6f} over {ph_end - 2} preset rows -- this "
        f"voice's prompt geometry is wrong, which shifts every downstream position")
    assert m["pcc"] > VOICE_PCC_POOLED, (
        f"{voice}: pooled prefill PCC {m['pcc']:.6f} over {embeds.shape[1]} positions")

    pipe.backbone.reset()
    frames, _, _ = pipe.generate(embeds, max_frames=N_FRAMES, seed=0, verbose=False)

    assert frames.ndim == 2 and frames.shape[1] == N_CODES, \
        f"{voice}: frames shaped {tuple(frames.shape)}, expected [T, {N_CODES}]"
    assert frames.shape[0] >= 1, f"{voice}: no frames emitted"
    assert not bool((frames[:, 0] == END_AUDIO_ID).any()), \
        f"{voice}: [END_AUDIO] leaked into the returned frames"
    assert bool((frames >= 0).all()), f"{voice}: negative code id"
    print(f"  {voice:>16} ({lang_of(voice)}) P={embeds.shape[1]:>4} -> {frames.shape[0]} frames  "
          f"placeholders {m_ph['pcc']:.6f}  text {m_tx['pcc']:.6f}  pooled {m['pcc']:.6f}")
