# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Live reference builders and shared helpers for the on-device tests.

The fp32 `reference/` package is the oracle; these helpers feed it the same inputs the pipeline
would. Everything is cached at module scope because the backbone state is ~13 GB and every device
test needs the same copy.

Block 1 accuracy must be judged on real tokenized prompts: random embeddings are off-manifold and
read far worse than real text on the same weights, so there is no synthetic-input builder here.
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
    """-> [0, 1, ... n-1], for parametrize. All of them: per-case accuracy varies enough that an
    aggregate over a different subset is not comparable to a recorded one."""
    return list(range(len(fixture_cases())))


@functools.lru_cache(maxsize=1)
def backbone_state():
    """-> the fp32 backbone weights, loaded once per process (~13 GB)."""
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
    """-> real Block 1+2 output frames [T,37], for teacher-forced decode.

    Both sides must advance on the same embedding each step, or later frames compare diverging
    trajectories rather than measuring error."""
    return torch.load(FRAMES).long()


def worst_sample_pct(got, exp):
    """Max absolute deviation as a percentage of the reference's scale. Always report it next to
    a PCC: a correlation can sit high while individual samples are badly wrong."""
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


# The device caches a rotated head interleaved (pairs adjacent); the reference lays it out
# half-split. RoPE applies the same permutation to Q, so attention is identical either way.
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
    """-> (embeds [1,S,3072], repeated: bool) from the fixture's own texts joined into one prompt.

    `repeated` is True once the texts had to be repeated to reach S, so the caller can gate loosely:
    unrelated texts run together are not one natural prompt, whatever their provenance.
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
