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
# TWO gates, because the two regions of a prompt behave differently and only one of them is what
# this test is about.
#
# The PLACEHOLDER region is the voice's own preset rows -- the geometry this test exists to check --
# and it is remarkably stable: measured 0.999705..0.999809 across voices, and IDENTICAL to six
# decimals for a given voice no matter what text follows it (nl_male reads 0.999769 with Dutch,
# English and Arabic sentences).
#
# The TEXT region varies with the token sequence, 0.994736..0.999789 across (voice, text) pairs --
# the same content-dependent bf16 sensitivity seen everywhere else in this port. nl_male + its Dutch
# sentence is the low tail (0.994736) while nl_male + an English sentence is 0.999789 and the same
# Dutch sentence on nl_female is 0.999205. Gating that tightly would be gating the text, not the
# voice.
# Measured over ALL 20 voices, 2026-08-27 (I set this gate twice from partial samples first and it
# failed twice -- 5 voices, then 19):
#
#   placeholders  0.999487 (es_male, 208 preset rows) .. 0.999816 (hi_female)
#   text          0.994736 (nl_male)                  .. 0.999821
#   pooled        0.998901 (nl_male)                  .. 0.999805
#
# es_male is the low tail on placeholders and has the most preset rows of any voice, which is a
# benign reason to sit slightly lower. A wrong placeholder COUNT does not shave 2e-4 off -- it
# shifts every downstream position and collapses the number -- so the floor is set to catch that.
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
    """Prompt assembly through Block 1 through Block 2, on the traced path, for one voice.

    Compared against the fp32 reference, not just checked for plausible shape: each voice has its
    own placeholder count (ar_male 67 rows, casual_female 214), so each is a different prompt
    geometry, and a geometry that assembles but shifts positions would emit valid-looking codes.
    """
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
