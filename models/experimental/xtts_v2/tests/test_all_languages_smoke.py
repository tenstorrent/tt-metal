# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Every supported language through the full XttsV2 request path, on device.

Nothing after the tokenizer is language-aware — the device sees token ids — so this is not checking
the vocoder or decode again (test_request_path_repeatability and test_vocoder_request_path own
that). What it checks is that each language's real prompt survives the whole path: prefill at that
language's token count, decode, the bucket the utterance lands in, and a bit-identical repeat.

Per language, three requests of growing prompt AND growing audio, the latter capped rather than left
to the model: the reference clip is synthetic noise, whose length picks the conv shapes rather than
its content, and with no real speaker to condition on the model can run to the audio code cap
however short the text is.

A plausibility check, not a correctness one — it never compares against a reference. Cleaning
correctness is covered by test_tokenizer_multilingual against coqui's own vectors.
"""
import pytest
import torch

from models.experimental.xtts_v2.frontend import SUPPORTED_LANGUAGES
from models.experimental.xtts_v2.tests.language_corpus import SENTENCES
from models.experimental.xtts_v2.tt.ttnn_xtts_model import HOP, OUTPUT_SR, VOC_BUCKETS, XttsV2, _voc_bucket

MS_PER_CODE = 1000 * 1024 / 22050  # a code is 1024 samples @22.05kHz, resampled without changing time


@pytest.fixture(scope="module")
def tts():
    """One model for the whole module: a warmup per language would dominate the runtime."""
    model = XttsV2()
    model.warmup()
    yield model
    model.close()


@pytest.fixture(scope="module")
def voice(tts):
    gen = torch.Generator().manual_seed(0)
    return tts.compute_voice(torch.randn(int(6.0 * 22050), generator=gen) * 0.1, 22050)


def test_corpus_covers_every_language():
    assert sorted(SENTENCES) == sorted(SUPPORTED_LANGUAGES)
    assert all(len(v) == 3 for v in SENTENCES.values()), "three sentences give the three lengths"


CAPS = (60, 150, 300)  # max_new_tokens per request: three audio lengths, hence three buckets


@pytest.mark.slow
@pytest.mark.parametrize("lang", SUPPORTED_LANGUAGES)
def test_language_smoke(lang, tts, voice):
    parts = SENTENCES[lang]
    last = None
    for n, cap in enumerate(CAPS):
        text = " ".join(parts[: n + 1])
        wav = tts.generate(text, voice, language=lang, seed=0, max_new_tokens=cap)
        codes = tts.last_timings["codes"]

        assert wav.shape[-1] > 0, f"{lang}: no audio for {text[:40]!r}"
        assert torch.isfinite(wav).all(), f"{lang}: non-finite samples"
        assert wav.abs().max() <= 1.0, f"{lang}: samples outside [-1,1] ({wav.abs().max():.3f})"
        assert codes <= cap, f"{lang}: {codes} codes exceeds max_new_tokens {cap}"
        # a caller cannot tell a finished sentence from a cut-off one without this
        assert tts.last_timings["truncated"] == (codes == cap), f"{lang}: truncation misreported"
        assert _voc_bucket(wav.shape[-1] // HOP) in VOC_BUCKETS
        # a trimmed run-on shortens the audio, so add back what generate reports removing
        audio_ms = 1000 * (wav.shape[-1] / OUTPUT_SR + tts.last_timings["run_on_s"])
        assert abs(audio_ms - codes * MS_PER_CODE) < MS_PER_CODE, f"{lang}: {audio_ms:.0f}ms, {codes} codes"
        last = (text, cap, wav)

    # The longest again: a repeat must be bit-identical, which is what catches trace corruption.
    text, cap, wav = last
    again = tts.generate(text, voice, language=lang, seed=0, max_new_tokens=cap)
    assert again.shape == wav.shape and torch.equal(again, wav), f"{lang}: repeat differs"
