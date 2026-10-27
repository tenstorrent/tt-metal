# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Naturalness, per language, as a pytest gate against fixed floors.

Generates the per-language clip set (`write_language_set`) on the device and scores it with
DistillMOS via `tests/mos_score.py` in `/tmp/mosvenv` (build it once with `tests/mos_setup.sh`).
A missing venv FAILS rather than skips, since a skipped gate reads as a pass.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_mos.py      # device + CPU
"""

import json
import math
import os
import subprocess
import wave

import pytest

from models.experimental.voxtral_tts.reference.voxtral_common_ref import DEFAULT_CKPT
from models.experimental.voxtral_tts.tests.sentence_corpus import WER_SENTENCES

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(MODEL)))
MOSVENV = "/tmp/mosvenv/bin/python"
SCORE = os.path.join(HERE, "mos_score.py")
SEED = 0
MIN_WORDS = 19  # the shortest medium-band sentence; shorter clips give a noisy MOS

# Floor on each language's MEAN MOS, just below its lowest seed mean so a healthy build passes, and on
# any single clip, since one clip turning to noise barely moves a mean. Per language because the
# predictor is biased by language, so each language is compared only with itself.
MOS_FLOOR = {"ar": 4.61, "de": 4.66, "en": 4.64, "es": 4.67, "fr": 4.63, "hi": 4.49, "it": 4.62, "nl": 4.70, "pt": 4.62}
CLIP_FLOOR = 3.97

# The predictor's own calibration bounds: real speech scores high, silence and noise low, far apart.
SPEECH_MIN, NON_SPEECH_MAX, SEPARATION_MIN = 4.0, 2.5, 2.0

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_CKPT), reason=f"no checkpoint at {DEFAULT_CKPT}")


def test_every_language_has_a_floor():
    """Host-side, so a language added to the corpus without a floor fails here and costs nothing."""
    missing, extra = sorted(set(WER_SENTENCES) - set(MOS_FLOOR)), sorted(set(MOS_FLOOR) - set(WER_SENTENCES))
    assert not missing and not extra, f"languages without a MOS floor {missing}, unexpected {extra}"


def _env():
    env = dict(os.environ)
    env.setdefault("TT_METAL_HOME", REPO)
    env["PYTHONPATH"] = f"{REPO}/ttnn:{REPO}/tools:{REPO}"
    return env


def _need_mosvenv():
    if not os.path.exists(MOSVENV):
        pytest.fail(
            f"{MOSVENV} is missing, so MOS cannot be scored. Run "
            f"models/experimental/voxtral_tts/tests/mos_setup.sh once. This fails "
            f"rather than skipping on purpose: a skipped naturalness gate reads as a pass."
        )


def _score(clip_dir):
    """-> {"means": {lang: mean}, "clips": [manifest row + "mos"]}, scored in the MOS venv."""
    r = subprocess.run([MOSVENV, SCORE, clip_dir], cwd=REPO, env=_env(), capture_output=True, text=True, timeout=3600)
    for line in r.stdout.splitlines():
        if line.startswith("MOS_JSON: "):
            return json.loads(line[len("MOS_JSON: ") :])
    raise AssertionError(
        f"the MOS scorer printed no MOS_JSON line (exit {r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    )


def _frame_budget(text):
    """Frame cap, same rule as the WER gate."""
    return max(320, int(math.ceil(len(text) / 18.0 * 12.5 * 2.2)))


def _save_wav(wav, path, sr=24000):
    import torch

    x = (wav.reshape(-1).clamp(-1, 1) * 32767).to(torch.int16).numpy()
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(x.tobytes())


def write_language_set(pipe, out, seed=SEED, langs=None, verbose=False):
    """One clip per (language, voice, sentence of >= MIN_WORDS words) into `out`, plus manifest.json
    carrying the language label for the scorer, which cannot import ttnn."""
    from models.experimental.voxtral_tts.tests.reference_helpers import all_voices, corpus_embeds
    from models.experimental.voxtral_tts.tests.sentence_corpus import lang_of, wer_band

    os.makedirs(out, exist_ok=True)
    rows = []
    for lang in sorted(langs or WER_SENTENCES):
        texts = [t for b in ("medium", "long") for t in wer_band(lang, b) if len(t.split()) >= MIN_WORDS]
        for voice in [v for v in all_voices() if lang_of(v) == lang]:
            for i, text in enumerate(texts):
                pipe.backbone.reset()
                frames, _, _ = pipe.generate(
                    corpus_embeds(text, voice, pipe.wb), max_frames=_frame_budget(text), seed=seed, verbose=False
                )
                name = f"{lang}_{voice}_s{i}.wav"
                _save_wav(pipe.decode(frames), os.path.join(out, name))
                rows.append(
                    {
                        "file": name,
                        "lang": lang,
                        "voice": voice,
                        "sentence": i,
                        "words": len(text.split()),
                        "frames": int(frames.shape[0]),
                        "seconds": round(frames.shape[0] / 12.5, 2),
                    }
                )
                if verbose:
                    print(f"  {lang}/{voice} s{i}: {frames.shape[0]} frames -> {name}", flush=True)
    json.dump(rows, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    return rows


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    """Generate the per-language set on the device, then score it. One pass for the whole module."""
    ttnn = pytest.importorskip("ttnn")
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline, open_device

    _need_mosvenv()
    out = str(tmp_path_factory.mktemp("mos_lang"))
    dev = open_device()
    try:
        pipe = TtVoxtralPipeline(dev)
        pipe.warmup(verbose=False)
        write_language_set(pipe, out)
        pipe.close()
    finally:
        ttnn.close_device(dev)
    return _score(out)


@pytest.mark.slow
@pytest.mark.timeout(7200)
def test_every_language_was_scored(scored):
    """Every corpus language produced clips and a mean."""
    got = set(scored["means"])
    assert got == set(WER_SENTENCES), f"scored {sorted(got)}, expected {sorted(WER_SENTENCES)}"
    per = {l: sum(1 for c in scored["clips"] if c["lang"] == l) for l in got}
    print("\n  clips per language: " + ", ".join(f"{l} {n}" for l, n in sorted(per.items())))
    assert min(per.values()) >= 2, f"a language was scored on fewer than two clips: {per}"


@pytest.mark.slow
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("lang", sorted(WER_SENTENCES))
def test_mos_per_language(scored, lang):
    """This language's own floor, so a failure names the language."""
    m = scored["means"][lang]
    vals = [c["mos"] for c in scored["clips"] if c["lang"] == lang]
    print(f"\n  {lang}: MOS {m:.4f} over {len(vals)} clips (min {min(vals):.3f}), floor {MOS_FLOOR[lang]}")
    assert m >= MOS_FLOOR[lang], f"{lang}: mean MOS {m:.4f} below its floor {MOS_FLOOR[lang]}"


@pytest.mark.slow
@pytest.mark.timeout(7200)
def test_no_clip_collapses(scored):
    worst = sorted(scored["clips"], key=lambda c: c["mos"])[:3]
    print("\n  worst clips: " + ", ".join(f"{c['file']} {c['mos']:.3f}" for c in worst))
    assert (
        worst[0]["mos"] >= CLIP_FLOOR
    ), f"{worst[0]['file']} scored MOS {worst[0]['mos']:.3f}, below the per-clip floor {CLIP_FLOOR}"


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_predictor_separates_speech_from_noise(tmp_path):
    """fp32-reference speech must score far above silence and noise. No device: the speech is the
    ASR calibration fixture decoded by the fp32 codec."""
    import wave

    import torch

    from models.experimental.voxtral_tts.reference import voxtral_codec_ref as cref

    _need_mosvenv()
    clips = torch.load(os.path.join(HERE, "asr_calibration_fixture.pt"))["clips"]
    w = cref.load_codec_state()
    rows = []

    def save(name, wav):
        x = (wav.reshape(-1).clamp(-1, 1) * 32767).to(torch.int16).numpy()
        with wave.open(str(tmp_path / name), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(x.tobytes())
        rows.append(
            {
                "file": name,
                "lang": name.split(".")[0],
                "voice": "-",
                "sentence": 0,
                "words": 0,
                "seconds": round(wav.numel() / 24000, 2),
            }
        )

    speech = cref.reference_decode(cref.strip_offset_and_trim(clips["en_medium"]["frames"].long()), w)
    save("speech.wav", speech)
    save("silence.wav", torch.zeros(8 * 24000))
    g = torch.Generator().manual_seed(0)
    save("noise.wav", torch.randn(8 * 24000, generator=g) * float(speech.pow(2).mean().sqrt()))
    json.dump(rows, open(tmp_path / "manifest.json", "w"))
    m = _score(str(tmp_path))["means"]
    print(f"\n  speech {m['speech']:.3f}  silence {m['silence']:.3f}  noise {m['noise']:.3f}")
    assert m["speech"] >= SPEECH_MIN, f"fp32-reference speech scored {m['speech']:.3f} < {SPEECH_MIN}"
    assert max(m["noise"], m["silence"]) <= NON_SPEECH_MAX, f"noise/silence scored like speech: {m}"
    assert m["speech"] - max(m["noise"], m["silence"]) >= SEPARATION_MIN, f"speech not clearly separated: {m}"
