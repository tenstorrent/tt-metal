# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Live reference builders for the on-device PCC tests — no stored golden tensors.

The fp32 PyTorch `reference/` package is the oracle; these helpers just feed it the same inputs
the pipeline would. Everything here is cached at module scope because the backbone state is ~13 GB
and every device test needs the same copy.

REAL PROMPTS ARE NOT OPTIONAL for a Block 1 accuracy number (BUG-9 / STATUS 5.12): random
embeddings are off-manifold and read PCC 0.892 where these give 0.9994 on the same weights. This
is why there is no `synthetic_speech()`-style generator for Block 1 -- the sibling xtts_v2 suite
can synthesise its inputs, and this one deliberately cannot.
"""

import functools
import json
import os

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(HERE, "tests", "prompt_fixture.json")
FRAMES = os.path.join(HERE, "tests", "real_frames_fixture.pt")


@functools.lru_cache(maxsize=1)
def fixture_cases():
    """-> the fixture's case list (real tokenized prompts + voice names)."""
    with open(FIXTURE) as fh:
        return json.load(fh)["cases"]


def case_ids():
    """-> [0, 1, ... n-1], for parametrize. All of them, deliberately: a 2-prompt default let a
    regression through once, because per-case mean worst-sample ranges ~0.45 pp, so an aggregate
    over a different prompt set is not comparable to a recorded one (STATUS 6.8/6.15)."""
    return list(range(len(fixture_cases())))


@functools.lru_cache(maxsize=1)
def backbone_state():
    """-> the fp32 backbone weights, loaded once per process (~13 GB)."""
    from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref

    return bref.load_backbone_state()


def fixture_embeds(case_idx, w=None):
    """Fixture case -> (real prompt embeds [1,P,3072], case dict), exactly as the pipeline builds
    them. Moved here from `gates.py` (formerly `tt_gates.py`) so the CLI, the tests and the quality report
    cannot drift to two different notions of "the real prompt"."""
    from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref

    w = backbone_state() if w is None else w
    case = fixture_cases()[case_idx]
    ids = torch.tensor(case["ids"], dtype=torch.long)
    return pref.build_inputs_embeds(ids, pref.load_voice(case["voice"]), w), case


@functools.lru_cache(maxsize=1)
def real_frames():
    """-> genuine Block 1+2 output frames [T,37], for teacher-forced decode.

    Teacher forcing matters: both sides must advance on the SAME embedding every step, so each
    step is an independent measurement. Feeding each side its own codes compares two diverging
    trajectories and measures nothing."""
    return torch.load(FRAMES).long()


def worst_sample_pct(got, exp):
    """Max absolute deviation as a percentage of the reference's scale.

    Always report this next to a PCC. PCC is a correlation: it sits at 0.9998 while individual
    samples are badly wrong, and for audio the outliers are what you hear (STATUS 5.9)."""
    return (got - exp).abs().max().item() / exp.abs().max().item() * 100


def corpus_embeds(text, voice, w=None):
    """(text, voice) -> prompt embeds [1,P,3072], tokenized by the IN-REPO tokenizer.

    `fixture_embeds` is limited to the 15 cases whose mistral_common ids are stored. This one takes
    any text, so breadth costs nothing -- `TekkenTokenizer.build_prompt` is pure torch and
    `test_tokenizer_ref.py` pins it against those stored ids, so it is trustworthy for new text.
    """
    from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
    from models.experimental.voxtral_tts.reference.voxtral_tokenizer_ref import TekkenTokenizer

    import torch as _t

    w = backbone_state() if w is None else w
    ids = _t.tensor(TekkenTokenizer().build_prompt(text, voice), dtype=_t.long)
    return pref.build_inputs_embeds(ids, pref.load_voice(voice), w)


@functools.lru_cache(maxsize=1)
def all_voices():
    """-> every voice preset the checkpoint ships, sorted. 20 of them; the fixture uses 13."""
    from models.experimental.voxtral_tts.reference.voxtral_tokenizer_ref import TekkenTokenizer

    return tuple(sorted(TekkenTokenizer().voices))


# The device caches a rotated head INTERLEAVED (pairs adjacent); the fp32 reference lays it out
# HALF-SPLIT (first half, then second). RoPE applies the same permutation to Q, so Q.K is unchanged
# and attention is identical -- but raw cached K differs, reading PCC ~0.02 in all 26 layers on a
# perfectly healthy model (measured -0.0035 as-is vs 0.999975 permuted at layer 0). V is never
# rotated and needs no permutation, and that asymmetry is the tell. Shared because two test files
# compare caches and a second copy of this is exactly the drift this suite keeps removing.
_HALF_TO_INTERLEAVED = None


def as_device_k_layout(k_ref):
    """Reference K (half-split head dim) -> the device's interleaved order."""
    global _HALF_TO_INTERLEAVED
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import HEAD_DIM

    if _HALF_TO_INTERLEAVED is None:
        idx = torch.empty(HEAD_DIM, dtype=torch.long)
        idx[: HEAD_DIM // 2] = torch.arange(0, HEAD_DIM, 2)
        idx[HEAD_DIM // 2 :] = torch.arange(1, HEAD_DIM, 2)
        _HALF_TO_INTERLEAVED = idx
    return k_ref[..., _HALF_TO_INTERLEAVED]


@functools.lru_cache(maxsize=8)
def _fixture_text(reps):
    """The fixture's own 15 texts joined, repeated `reps` times."""
    return " ".join([c["text"] for c in fixture_cases()] * reps)


def long_prompt_embeds(S, w=None, voice="ar_male"):
    """-> (embeds [1,S,3072], repeated: bool) built from the FIXTURE's own texts.

    Provenance matters more than length here. The fixture texts are the same corpus every accuracy
    gate in this suite uses, joined into ONE well-formed prompt with a single header -- so up to the
    length they naturally reach, a long-shape test is running real text.

    Measured: the 15 texts joined are 1920 chars -> 540 tokens with `ar_male` (fewest placeholder
    rows, 67, leaving the most room for text), which covers Sp 128/256/384/512. Beyond that the text
    has to be repeated, and `repeated=True` says so, so the caller can gate accordingly instead of
    quoting a tight number on text that no request looks like.
    """
    from models.experimental.voxtral_tts.reference.voxtral_tokenizer_ref import TekkenTokenizer

    w = backbone_state() if w is None else w
    tok = TekkenTokenizer()
    reps = 1
    while len(tok.build_prompt(_fixture_text(reps), voice)) < S:
        reps += 1
        if reps > 64:
            raise AssertionError(f"cannot reach {S} tokens from the fixture texts")
    return corpus_embeds(_fixture_text(reps), voice, w)[:, :S], reps > 1
