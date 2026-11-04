# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic inputs and live references for the Qwen3-TTS tests.

No golden files. Each helper builds its input from a seed and computes the reference
in-process from the checkpoint, so the suite needs only the checkpoint and (for the device
tests) a card. Results are cached per process: loading the encoder is the slow part.
"""

import functools
import math

import torch
import torch.nn.functional as F

from models.demos.audio.qwen3_tts import frontend, weights
from models.demos.audio.qwen3_tts.reference.qwen3_speaker_ref import SpeakerReference, speaker_mel

# A sentence long enough to give the talker something to attend over.
TALKER_TEXT = "Hello from Tenstorrent. This is a real sentence for the talker to read."

# A reference clip is 3 s in the model card's usage; the mel hop of 256 at 24 kHz puts that
# at 281 frames, which is what the device tests compile for.
CLIP_SECONDS = 3.0


# Two voices far enough apart that the encoder separates them: a low, dark, slow one and a
# high, bright, fast one. Keyed by name so a test can ask for a contrast without restating
# the parameters.
VOICES = {
    "low": {"f0": 110.0, "drift": 35.0, "formant": 4.0, "syllables": 3.1, "harmonics": 20},
    "high": {"f0": 215.0, "drift": 60.0, "formant": 11.0, "syllables": 5.4, "harmonics": 12},
}


def synthetic_voiced_clip(seconds=CLIP_SECONDS, seed=0, voice="low"):
    """A deterministic voiced-speech stand-in: harmonic stack, drifting f0, breath noise.

    Real speech would mean shipping an audio fixture. This keeps the input in the same
    territory the mel front-end expects (harmonic structure, a moving pitch, an amplitude
    envelope, peak below 1.0) without one. `voice` selects a timbre from VOICES.
    """
    settings = VOICES[voice]
    sample_rate = weights.SPEAKER_MEL["sampling_rate"]
    count = int(seconds * sample_rate)
    t = torch.arange(count, dtype=torch.float64) / sample_rate

    # f0 drifts around its centre the way a spoken phrase does.
    f0 = settings["f0"] + settings["drift"] * torch.sin(2 * math.pi * 0.45 * t)
    phase = 2 * math.pi * torch.cumsum(f0, dim=0) / sample_rate

    signal = torch.zeros_like(t)
    for harmonic in range(1, settings["harmonics"] + 1):
        # 1/k rolloff, with a broad formant-like bump that sets the timbre.
        weight = (1.0 / harmonic) * (1.0 + 0.6 * math.exp(-((harmonic - settings["formant"]) ** 2) / 8.0))
        signal += weight * torch.sin(harmonic * phase)

    # Syllable-rate envelope, never fully closing, plus a low noise floor.
    envelope = 0.55 + 0.45 * torch.sin(2 * math.pi * settings["syllables"] * t).abs()
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(count, generator=generator, dtype=torch.float64) * 0.005

    clip = signal * envelope + noise
    clip = clip / clip.abs().max() * 0.9
    return clip.to(torch.float32)


@functools.lru_cache(maxsize=None)
def speaker_reference(seconds=CLIP_SECONDS, seed=0):
    """Input mel, reference embedding and per-block intermediates, computed live.

    Returns a dict with `mel` [1, T, 128], `embedding` [1, enc_dim] and `intermediates`,
    the latter keyed by the names in `qwen3_speaker_ref.INTERMEDIATES` and laid out
    channel-first the way the reference works.
    """
    clip = synthetic_voiced_clip(seconds, seed)
    mel = speaker_mel(clip)
    embedding, intermediates = SpeakerReference()(mel, return_intermediates=True)
    return {"mel": mel, "embedding": embedding, "intermediates": intermediates}


@functools.lru_cache(maxsize=None)
def talker_prompt(text=TALKER_TEXT):
    """A realistic talker input: real token ids through the model's own embedding path.

    Random embeddings are a bad proxy here and measurably so. Pushed through the 28-layer
    stack they land far outside the activation distribution the weights were trained on,
    and the device result drifts to PCC 0.936 against the fp32 reference with only 71% of
    top-1 codec tokens agreeing. The same graph on this prompt holds 0.9957 and agrees on
    every token. Test with what the model will actually see.

    Mirrors the text half of `generate_icl_prompt`: token ids -> text_embedding ->
    text_projection, which is `linear_fc2(silu(linear_fc1(x)))`. The codec track is absent,
    so this is the text stream alone rather than a full dual-track prompt.

    Returns (embeddings [1, T, hidden], position_ids [3, 1, T]).
    """
    table = weights.load_prefixed("talker.model.text_embedding.")["weight"]
    projection = weights.load_prefixed("talker.text_projection.")

    ids = torch.tensor(frontend.text_ids(text))
    hidden = F.linear(table[ids], projection["linear_fc1.weight"], projection["linear_fc1.bias"])
    embeddings = F.linear(F.silu(hidden), projection["linear_fc2.weight"], projection["linear_fc2.bias"])
    embeddings = embeddings.unsqueeze(0)

    length = embeddings.shape[1]
    positions = torch.arange(length, dtype=torch.long).reshape(1, 1, length)
    return embeddings, positions.expand(3, 1, length).contiguous()


@functools.lru_cache(maxsize=None)
def codec_head():
    """The projection from hidden states to codec logits, for token-agreement checks."""
    return weights.load_prefixed("talker.codec_head.")["weight"]


@functools.lru_cache(maxsize=None)
def code_predictor_prompt():
    """A realistic code-predictor input, produced by the model itself.

    Random hidden states and random codes are out of distribution, and the talker showed
    what that costs: a correct port measured far worse on noise than on real activations.
    So this chains the real thing. Run the talker on a real prompt, take its last hidden
    state, read codebook 0 off `codec_head`, then let the reference decode codebooks 1 to
    15 greedily. The result is a self-consistent frame the model would actually produce.

    Returns (talker_hidden [1, 1, hidden], first_code, codes, embeddings [1, 16, hidden]),
    where `codes` is codebooks 1 to 15 and `embeddings` teacher-forces all 16 positions.
    """
    from models.demos.audio.qwen3_tts.reference.qwen3_code_predictor_ref import (
        CodePredictorReference,
        build_input_embeddings,
    )
    from models.demos.audio.qwen3_tts.reference.qwen3_talker_ref import TalkerReference

    embeddings, positions = talker_prompt()
    hidden = TalkerReference(dtype=torch.float32)(embeddings, position_ids=positions)
    talker_hidden = hidden[:, -1:, :]
    first_code = int((talker_hidden[0, 0] @ codec_head().T).argmax())

    codes = CodePredictorReference().generate_greedy(talker_hidden, first_code)
    teacher_forced = build_input_embeddings(talker_hidden, [first_code] + codes[:-1])
    return talker_hidden, first_code, tuple(codes), teacher_forced


@functools.lru_cache(maxsize=1)
def codebook_size():
    """How many ids the codec can actually render."""
    return weights.codec_decoder_config()["codebook_size"]


@functools.lru_cache(maxsize=None)
def codec_frames(count=4):
    """Real codec frames, produced by the CPU references rather than drawn at random.

    Random codes are out of distribution for the decoder the same way random embeddings
    were for the talker. This runs the reference talker and code predictor greedily for a
    few frames off a real prompt, which is what the decoder will actually be handed.

    Returns codes [1, 16, count].
    """
    from models.demos.audio.qwen3_tts.reference.qwen3_code_predictor_ref import (
        CodePredictorReference,
        build_input_embeddings,
    )
    from models.demos.audio.qwen3_tts.reference.qwen3_talker_ref import TalkerReference, default_position_ids

    talker = TalkerReference(dtype=torch.float32)
    predictor = CodePredictorReference()
    head = codec_head()
    embeddings, _ = talker_prompt()

    frames = []
    for _ in range(count):
        hidden = talker(embeddings, position_ids=default_position_ids(embeddings.shape[1]))[:, -1:, :]
        # Restrict codebook 0 to the codec vocabulary. Ids 2048 and above are the talker's
        # own control tokens (pad, bos, eos, the think markers); upstream's loop stops on
        # them rather than emitting them, so the decoder never sees one.
        first = int((hidden[0, 0] @ head.T)[: codebook_size()].argmax())
        rest = predictor.generate_greedy(hidden, first)
        frames.append([first] + list(rest))
        # The next prompt position is the 16 codebook embeddings summed; close enough to
        # the real loop for the decoder's purposes, and it keeps the codes in distribution.
        summed = build_input_embeddings(hidden, [first] + rest[:-1])[:, 1:, :].sum(dim=1, keepdim=True)
        embeddings = torch.cat([embeddings, summed], dim=1)

    codes = torch.tensor(frames, dtype=torch.long)  # [count, 16]
    return codes.t().unsqueeze(0).contiguous()
