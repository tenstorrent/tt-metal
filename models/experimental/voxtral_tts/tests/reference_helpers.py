# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Live reference builders and shared helpers for the on-device tests.

The fp32 `reference/` package is the oracle; these helpers feed it the same inputs the pipeline
would, cached at module scope so every device test shares one backbone state.

There is deliberately no synthetic-input builder: random embeddings are off-manifold and understate
accuracy.
"""

import functools
import json
import os

import pytest
import torch

from models.experimental.voxtral_tts.reference.voxtral_common_ref import DEFAULT_CKPT

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(HERE, "tests", "prompt_fixture.json")
FRAMES = os.path.join(HERE, "tests", "real_frames_fixture.pt")
FRAMES_LONG = os.path.join(HERE, "tests", "real_frames_long_fixture.pt")
CONDITIONING = os.path.join(HERE, "tests", "conditioning_fixture.json")

needs_checkpoint = pytest.mark.skipif(not os.path.exists(DEFAULT_CKPT), reason=f"no checkpoint at {DEFAULT_CKPT}")


@functools.lru_cache(maxsize=1)
def fixture_cases():
    """-> the fixture's case list (real tokenized prompts + voice names)."""
    with open(FIXTURE) as fh:
        return json.load(fh)["cases"]


@functools.lru_cache(maxsize=1)
def conditioning_fixture():
    """-> the ill-conditioned positions/frames fixture."""
    with open(CONDITIONING) as fh:
        return json.load(fh)


def ill_conditioned_positions(case_idx):
    """-> prefill positions where the fp32 reference itself is not pinned down at device precision,
    so a device result there is a rounding draw."""
    return frozenset(conditioning_fixture()["prefill"].get(str(case_idx), ()))


def ill_conditioned_frames(case_idx):
    """-> the same for decode frames of the prompt's own trajectory."""
    return frozenset(conditioning_fixture()["decode"].get(str(case_idx), ()))


def case_ids():
    """-> [0, 1, ... n-1], for parametrize. All of them: prompts differ more than most effects."""
    return list(range(len(fixture_cases())))


@functools.lru_cache(maxsize=1)
def backbone_state():
    """-> the fp32 backbone weights, loaded once per process."""
    from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref

    return bref.load_backbone_state()


def fixture_embeds(case_idx, w=None):
    """Fixture case -> (prompt embeds [1,P,3072], case dict), as the pipeline builds them."""
    from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref

    w = backbone_state() if w is None else w
    case = fixture_cases()[case_idx]
    ids = torch.tensor(case["ids"], dtype=torch.long)
    return pref.build_inputs_embeds(ids, pref.load_voice(case["voice"]), w), case


@functools.lru_cache(maxsize=1)
def real_frames():
    """-> real frames [T,37] from the backbone and flow model, for teacher-forced decode."""
    return torch.load(FRAMES).long()


def worst_sample_pct(got, exp):
    """Max absolute deviation as a percentage of the reference's scale; report it next to a PCC,
    which can sit high while individual samples are badly wrong."""
    return (got - exp).abs().max().item() / exp.abs().max().item() * 100


def corpus_embeds(text, voice, w=None):
    """(text, voice) -> prompt embeds [1,P,3072], tokenized by the in-repo tokenizer."""
    from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
    from models.experimental.voxtral_tts.reference.voxtral_tokenizer_ref import TekkenTokenizer

    import torch as _t

    w = backbone_state() if w is None else w
    ids = _t.tensor(TekkenTokenizer().build_prompt(text, voice), dtype=_t.long)
    return pref.build_inputs_embeds(ids, pref.load_voice(voice), w)


@functools.lru_cache(maxsize=1)
def all_voices():
    """-> every voice preset the checkpoint ships, sorted."""
    from models.experimental.voxtral_tts.reference.voxtral_tokenizer_ref import TekkenTokenizer

    return tuple(sorted(TekkenTokenizer().voices))


# The reference caches a rotated head interleaved (pairs adjacent); the device caches it half-split.
# RoPE applies the same permutation to Q, so attention is identical.
_HALF_TO_INTERLEAVED = None


def as_device_k_layout(k_ref):
    """Reference K (interleaved head dim) -> the device's half-split order."""
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
    """-> (embeds [1,S,3072], repeated) from the fixture's texts joined into one prompt. `repeated`
    means the texts had to repeat to reach S, so the caller should gate loosely."""
    from models.experimental.voxtral_tts.reference.voxtral_tokenizer_ref import TekkenTokenizer

    w = backbone_state() if w is None else w
    tok = TekkenTokenizer()
    reps = 1
    while len(tok.build_prompt(_fixture_text(reps), voice)) < S:
        reps += 1
        if reps > 64:
            raise AssertionError(f"cannot reach {S} tokens from the fixture texts")
    return corpus_embeds(_fixture_text(reps), voice, w)[:, :S], reps > 1


@functools.lru_cache(maxsize=1)
def _long_frames_by_case():
    return torch.load(FRAMES_LONG)


def real_frames_long(case_idx):
    """-> that prompt's OWN full utterance of real frames [T,37]; another utterance's frames
    would be a mismatched pair.
    """
    return _long_frames_by_case()[case_idx].long()


def long_frame_cases():
    """-> the prompts that have a full-utterance frame capture, sorted."""
    return tuple(sorted(_long_frames_by_case()))
