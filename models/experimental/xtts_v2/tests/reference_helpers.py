# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Live reference builders for the XTTS-v2 PCC tests — no stored golden files needed.

Each helper returns the exact tensors the tests previously loaded from the (gitignored,
never-committed) golden/**/*.pt fixtures, computed in-process from the checkpoint:

  * deterministic input: either seeded noise shaped to the block's real input statistics,
    or a synthetic voiced waveform pushed through the coqui-free front-end (frontend.py),
  * reference output: the op-by-op PyTorch mirrors in reference/ (each validated at
    PCC 1.0 against coqui activations during bringup), loaded from the same checkpoint
    the TTNN modules use.

The results are lru_cached so a pytest session builds each reference once (the checkpoint
state dict itself is cached inside reference.xtts_gpt_ref.load_full_state).

Cross-check mode: set XTTS_GOLDEN_DIR to a golden/ root holding fixtures captured from a
real coqui run, and every helper returns those coqui-validated fixtures instead, with
identical keys/shapes. The default path never touches golden files.
"""

import functools
import math
import os

import torch
import torch.nn.functional as F

# Import the HF GPT2 reference classes BEFORE any XTTS checkpoint load: unpickling the
# coqui checkpoint (reference.xtts_gpt_ref.load_full_state, weights_only=False) leaves
# torch.distributed._functional_collectives in a state where a *subsequent*
# `from transformers import GPT2Model` dies re-registering its autograd kernels.
# Importing transformers first is safe in every order tested (incl. before/after ttnn).
from transformers import GPT2Config, GPT2Model  # noqa: F401  (eager import, see above)

from models.experimental.xtts_v2.frontend import PromptTables, conditioning_mels, speaker_logmel
from models.experimental.xtts_v2.reference.xtts_cond_ref import CondReference
from models.experimental.xtts_v2.reference.xtts_gpt_ref import (
    build_reference,
    load_full_state,
    make_golden_input,
    reference_forward,
    reference_generate,
)
from models.experimental.xtts_v2.reference.xtts_hifigan_ref import HifiganReference
from models.experimental.xtts_v2.reference.xtts_speaker_ref import SpeakerReference


def golden_dir():
    """Non-empty -> use stored coqui-captured fixtures instead of the live reference.

    Read at every public helper call and passed INTO the lru_cached builders as the cache
    key — reading it inside the cached body would freeze whichever mode ran first for the
    rest of the process (a test toggling XTTS_GOLDEN_DIR would silently get stale results)."""
    return os.environ.get("XTTS_GOLDEN_DIR", "")


def _load_golden(golden, *rel):
    return torch.load(os.path.join(golden, *rel))


# ---------------------------------------------------------------------------------------
# Deterministic synthetic reference clip (input choice for Blocks 1 & 2)
# ---------------------------------------------------------------------------------------


def synthetic_speech(seconds=4.0, sr=22050, seed=17):
    """Deterministic speech-like waveform [1, seconds*sr]: harmonic series on a wandering
    ~130 Hz f0 with vibrato, formant-shaped harmonic amplitudes (~700/2200 Hz bumps),
    syllable-rate amplitude modulation, and a low seeded-noise floor. Not intelligible
    audio — but it exercises the mel front-ends with a realistic dynamic range (voiced
    harmonics, spectral tilt, near-silent troughs) so the PCC tests see activations of
    real-data scale. Built in float64 and cast, so it is bit-stable across platforms
    up to torch.randn's generator determinism."""
    n = int(seconds * sr)
    t = torch.arange(n, dtype=torch.float64) / sr
    g = torch.Generator().manual_seed(seed)
    f0 = 130.0 * (1.0 + 0.08 * torch.sin(2 * math.pi * 0.6 * t) + 0.03 * torch.sin(2 * math.pi * 5.3 * t))
    phase = 2 * math.pi * torch.cumsum(f0, 0) / sr
    wav = torch.zeros(n, dtype=torch.float64)
    k = 1
    while k * 130.0 * 1.11 < 8000.0:  # keep all harmonics (incl. vibrato excursion) < 8 kHz
        fk = k * 130.0
        formant = 1.0 + 2.0 * math.exp(-(((fk - 700) / 400) ** 2)) + 1.2 * math.exp(-(((fk - 2200) / 600) ** 2))
        wav = wav + (formant / k) * torch.sin(k * phase)
        k += 1
    am = 0.55 + 0.45 * torch.sin(2 * math.pi * 2.7 * t + 0.7)  # syllable-rate envelope
    wav = wav * am
    wav = wav / wav.abs().max() * 0.6
    wav = wav + 0.003 * torch.randn(n, generator=g, dtype=torch.float64)  # noise floor
    return wav.to(torch.float32).reshape(1, -1)


# ---------------------------------------------------------------------------------------
# Block 1: conditioning encoder + Perceiver
# ---------------------------------------------------------------------------------------


def cond_reference():
    """-> dict: mel_in [1,80,T], enc_out [1,1024,T], perc_out [1,32,1024],
    gpt_cond_latent [1,32,1024] (== perc_out for a single <=6s chunk)."""
    return _cond_reference(golden_dir())


@functools.lru_cache(maxsize=2)  # live ("") and one golden dir can coexist in a session
def _cond_reference(golden):
    if golden:
        return {
            k: _load_golden(golden, "cond", f"{k}.pt") for k in ("mel_in", "enc_out", "perc_out", "gpt_cond_latent")
        }
    mel = conditioning_mels(synthetic_speech(), 22050, PromptTables().mel_stats)[0]  # single chunk
    enc, perc = CondReference().get_style_emb(mel)
    return {"mel_in": mel, "enc_out": enc, "perc_out": perc, "gpt_cond_latent": perc}


# ---------------------------------------------------------------------------------------
# Block 2: ResNet speaker encoder
# ---------------------------------------------------------------------------------------


def speaker_reference():
    """-> dict: logmel [1,64,T], speaker_embedding [1,512,1] (d-vector, L2-normalized)."""
    return _speaker_reference(golden_dir())


@functools.lru_cache(maxsize=2)
def _speaker_reference(golden):
    if golden:
        return {k: _load_golden(golden, "speaker", f"{k}.pt") for k in ("logmel", "speaker_embedding")}
    logmel = speaker_logmel(synthetic_speech(), 22050)
    emb = SpeakerReference().core(logmel, l2_norm=True).unsqueeze(-1)
    return {"logmel": logmel, "speaker_embedding": emb}


# ---------------------------------------------------------------------------------------
# Block 4: HiFi-GAN generator
# ---------------------------------------------------------------------------------------

HIFIGAN_L = 256  # synthetic latent length (frames @ ~93.75 Hz -> L*256 samples @ 24 kHz)


def hifigan_reference():
    """-> dict: z [1,1024,L], g [1,512,1], wav [1,1,L*256], dbg {conv_pre, ups0..3}.

    z is seeded noise pushed through the checkpoint's gpt.final_norm — the generator's
    real input IS a final_norm output (GPT latents), so this matches the per-channel
    scale/offset of real data exactly. g is a seeded unit-norm vector, matching the
    L2-normalized d-vector's distribution."""
    return _hifigan_reference(golden_dir())


@functools.lru_cache(maxsize=2)
def _hifigan_reference(golden):
    if golden:
        z = _load_golden(golden, "hifigan", "z.pt")
        if z.dim() == 2:
            z = z.unsqueeze(0)
        return {
            "z": z,
            "g": _load_golden(golden, "hifigan", "g.pt"),
            "wav": _load_golden(golden, "hifigan", "wav.pt"),
            "dbg": {
                key: _load_golden(golden, "hifigan", f"dbg_{key}.pt")
                for key in ("conv_pre", "ups0", "ups1", "ups2", "ups3")
            },
        }
    full = load_full_state()
    w = full["gpt.final_norm.weight"].float()
    b = full["gpt.final_norm.bias"].float()
    gen = torch.Generator().manual_seed(23)
    x = torch.randn(1, HIFIGAN_L, 1024, generator=gen)
    z = F.layer_norm(x, (1024,), w, b).permute(0, 2, 1).contiguous()  # [1,1024,L]
    g = F.normalize(torch.randn(1, 512, generator=gen), dim=1).reshape(1, 512, 1)
    wav, dbg = HifiganReference()(z, g, return_intermediates=True)
    return {"z": z, "g": g, "wav": wav, "dbg": dbg}


# ---------------------------------------------------------------------------------------
# Block 3: GPT transformer core (+ generation)
# ---------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _built_gpt_reference():
    return build_reference()


def gpt_reference():
    """-> dict: inputs_embeds [1,64,1024], latents [1,64,1024].

    The input is the seeded lookup of real checkpoint embedding tables that generated the
    original goldens (reference.xtts_gpt_ref.make_golden_input), so the live path is
    bit-identical to the golden files."""
    return _gpt_reference(golden_dir())


@functools.lru_cache(maxsize=2)
def _gpt_reference(golden):
    if golden:
        return {
            "inputs_embeds": _load_golden(golden, "gpt", "inputs_embeds.pt"),
            "latents": _load_golden(golden, "gpt", "latents.pt"),
        }
    inputs_embeds = make_golden_input()
    gpt, final_norm = _built_gpt_reference()
    _, latents = reference_forward(gpt, final_norm, inputs_embeds)
    return {"inputs_embeds": inputs_embeds, "latents": latents}


def gpt_generate_reference():
    """-> dict with prompt_embeds, step_inputs, ref_codes, ref_logits, ref_latents,
    start_token, stop_token (greedy decode is deterministic — bit-identical to the
    original generation goldens)."""
    return _gpt_generate_reference(golden_dir())


@functools.lru_cache(maxsize=2)
def _gpt_generate_reference(golden):
    if golden:
        out = {
            k: _load_golden(golden, "gpt", "generate", f"{k}.pt")
            for k in ("prompt_embeds", "step_inputs", "ref_codes", "ref_logits", "ref_latents")
        }
        meta = _load_golden(golden, "gpt", "generate", "meta.pt")
        out["start_token"] = meta["start_token"]
        out["stop_token"] = meta["stop_token"]
        return out
    return reference_generate(model=_built_gpt_reference())
